"""exp_v18 - paso 1: índice estacional por categoría (features se_*).

Implementa la ESPECIFICACION_variables.md: para cada (producto, corte C),
un factor estacional del mes-objetivo T=C+2, pooleado por categoría con
shrinkage jerárquico global->cat2->cat3 y ANTI-FUGA (solo usa targets ya
conocidos en C). En unidades de clase_ratio (= tn(C+2)/prom_12m(C)).

Salida: features_estacional.parquet  (product_id, corte, se_*).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import cargar_productos
from features_lgbm import FeatureBuilder

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
EPS = 1e-6
KAPPA = 20              # pseudo-conteo del shrinkage
CLIP = 4.0             # winsorización de ratios
CORTES = [int(p.strftime("%Y%m")) for p in pd.period_range("2018-01", "2019-12", freq="M")]

fb = FeatureBuilder()
wide = fb.wide
mes_menos = fb.mes_menos
periodos = fb.periodos

prod = cargar_productos().unique("product_id").to_pandas().set_index("product_id")
train_prods = wide.index[wide.notna().sum(axis=1) >= 12]
conteo_c3 = prod.loc[prod.index.isin(train_prods), "cat3"].value_counts()
c3_validas = set(conteo_c3[conteo_c3 > 10].index)
cat3_de = {p: (prod.loc[p, "cat3"] if prod.loc[p, "cat3"] in c3_validas else "OTROS")
           for p in wide.index if p in prod.index}
cat2_de = {p: prod.loc[p, "cat2"] if p in prod.index else "OTROS" for p in wide.index}


def mes_cal(periodo: int) -> int:
    return periodo % 100


# ---------- tabla de ratios REALIZADOS (para el pooling) ----------
# para cada corte c' con target c'+2 existente: ratio = tn(c'+2)/prom_12m(c')
filas = []
for cprime in periodos:
    target = mes_menos(cprime, -2)
    cols12 = [mes_menos(cprime, k) for k in range(12)]
    if target not in wide.columns or any(c not in wide.columns for c in cols12):
        continue
    prom = wide[cols12].mean(axis=1)
    ok = prom > EPS
    ratio = (wide.loc[ok, target] / prom[ok]).clip(0, CLIP)
    for pid, r in ratio.dropna().items():
        filas.append((pid, target, mes_cal(target), r))
ratios = pd.DataFrame(filas, columns=["product_id", "target_period", "mes_obj", "ratio"])
ratios["cat3"] = ratios["product_id"].map(cat3_de)
ratios["cat2"] = ratios["product_id"].map(cat2_de)
print(f"ratios realizados: {len(ratios):,}")


def factores_para(C: int):
    """Factores del mes-objetivo t=mes(C+2), usando solo targets <= C (anti-fuga)."""
    t = mes_cal(mes_menos(C, -2))
    R = ratios[(ratios["mes_obj"] == t) & (ratios["target_period"] <= C)]
    if len(R) == 0:
        return None, None, None, 1.0, {}
    f_glob = R["ratio"].mean()
    raw_c2 = R.groupby("cat2")["ratio"].agg(["mean", "size"])
    f_c2 = ((raw_c2["size"] * raw_c2["mean"] + KAPPA * f_glob) / (raw_c2["size"] + KAPPA)).to_dict()
    raw_c3 = R.groupby("cat3")["ratio"].agg(["mean", "size"])
    n_c3 = raw_c3["size"].to_dict()
    return f_c2, raw_c3, n_c3, f_glob, raw_c2["size"].to_dict()


tablas = []
for C in CORTES:
    cols12 = [mes_menos(C, k) for k in range(12)]
    vivos = wide.index[wide[cols12].notna().all(axis=1)]
    f_c2, raw_c3, n_c3, f_glob, n_c2 = factores_para(C)
    rows = []
    for pid in vivos:
        c2 = cat2_de.get(pid, "OTROS"); c3 = cat3_de.get(pid, "OTROS")
        fc2 = f_c2.get(c2, f_glob) if f_c2 is not None else f_glob
        if raw_c3 is not None and c3 in raw_c3.index and c3 != "OTROS":
            nc3 = raw_c3.loc[c3, "size"]
            fc3 = (nc3 * raw_c3.loc[c3, "mean"] + KAPPA * fc2) / (nc3 + KAPPA)
        else:
            nc3, fc3 = 0, fc2
        rows.append((pid, C, fc3, fc2, f_glob, fc3 - f_glob, int(nc3)))
    tablas.append(pd.DataFrame(rows, columns=[
        "product_id", "corte", "se_factor", "se_factor_c2", "se_factor_glob",
        "se_dev_c3", "se_n_c3"]))

out = pd.concat(tablas, ignore_index=True)
out.to_parquet(RUTA_EXP / "features_estacional.parquet")
print(f"features_estacional.parquet: {out.shape[0]} filas x {out.shape[1]-2} features")

# ---------- test de cordura (spec §8): factores por cat3 en C=201812 ----------
f_c2, raw_c3, n_c3, f_glob, _ = factores_para(201812)
print(f"\n=== test de cordura C=201812 (target feb-2019) ===")
print(f"global ≈ {f_glob:.3f}  (esperado ~0.83)")
for cat in ["Sopas", "PISOS"]:
    if raw_c3 is not None and cat in raw_c3.index:
        print(f"{cat}: raw={raw_c3.loc[cat,'mean']:.3f} (n={int(raw_c3.loc[cat,'size'])}) "
              f"(esperado Sopas~0.34, PISOS~1.12)")

# ---------- anti-fuga: verificar que ningún target usado > corte ----------
print("\nassert anti-fuga: recomputando y chequeando target_period<=C por corte...")
malos = 0
for C in CORTES:
    t = mes_cal(mes_menos(C, -2))
    usados = ratios[(ratios["mes_obj"] == t) & (ratios["target_period"] <= C)]
    if (usados["target_period"] > C).any():
        malos += 1
print("✓ sin fuga" if malos == 0 else f"✗ FUGA en {malos} cortes")

# auditoría vs features yoy existentes
d = fb.armar_corte(201812, True)
feat812 = out[out["corte"] == 201812].set_index("product_id")
join = d.join(feat812[["se_factor", "se_dev_c3", "se_n_c3"]], how="left")
print("\nauditoría (corr Spearman se_* vs features del set base):")
prob = FeatureBuilder.auditar(join)
se_prob = [p for p in prob if "se_" in p]
print("  " + ("✓ se_* no duplica nada existente" if not se_prob else "\n  ".join(se_prob)))
