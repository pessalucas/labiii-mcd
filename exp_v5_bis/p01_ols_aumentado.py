"""exp_v5_bis: OLS mágico (z403, ganador 0.231) AUMENTADO con las top
features de v11 y contexto de categoría.

Las top-5 de v11 (s2, roll_med_3, ncli_idx, dprom_24, s3) están
normalizadas por el promedio; en un OLS que predice TONELADAS absolutas
una feature adimensional (~1) actúa como intercepto y no aporta. Por eso
se incorporan en VERSIÓN TONELADAS (medias/medianas móviles absolutas,
conteo de clientes), que sí escalan con el producto.

Variantes (control + incrementales), medidas en backtest honesto
(entreno 201712->201802, evalúo 201812->201902) antes de submitear:
  V0 base    : réplica z403 (12 lags, 182 mágicos)           [debe ~0.231]
  V1 +top v11: + ma2, ma3, med3 (tn) + ncli_0 + dprom24_abs
  V2 +categ  : V1 + venta reciente de la categoría (tn) + su tendencia
"""

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_productos
from features_lgbm import FeatureBuilder

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
COMPETENCIA = "labo-iii-2026-ba"
EPS = 1e-6

fb = FeatureBuilder()
wide = fb.wide
mes_menos = fb.mes_menos

# panel de nº de clientes por producto-mes (para ncli_0)
sellin_ncli = fb.ventas  # tiene ncli
w_ncli = (sellin_ncli.to_pandas().pivot(index="product_id", columns="periodo",
          values="ncli").reindex(columns=fb.periodos).fillna(0.0))

# panel de categoría (suma cat3) para el contexto
prodcat = cargar_productos().unique("product_id").to_pandas().set_index("product_id")
cat_de = prodcat["cat3"].fillna("SIN_CAT").to_dict()
w_catall = wide.copy()
w_catall["__c"] = [cat_de.get(p, "SIN_CAT") for p in wide.index]
w_cat = w_catall.groupby("__c").sum(min_count=1)


def cargar_magicos() -> list[int]:
    nb = json.loads((RUTA_PROYECTO / "src/Estadistica/z403_RegresionLineal_local.ipynb").read_text())
    celda = next("".join(c["source"]) for c in nb["cells"]
                 if c["cell_type"] == "code" and "productos_magicos" in "".join(c["source"]))
    return [int(x) for x in re.findall(r"productos_magicos = \[(.*?)\]",
            celda, flags=re.S)[-1].replace("\n", " ").split(",")]


magicos = cargar_magicos()


def matriz(cut: int, variante: str) -> pd.DataFrame:
    """Features en el corte `cut` para todos los productos con 12m de historia."""
    cols = [mes_menos(cut, k) for k in range(12)]
    d = wide[cols].dropna()
    A = d.values
    X = pd.DataFrame(index=d.index)
    for k in range(12):
        X[f"tn_{k}"] = A[:, k]            # V0: los 12 lags crudos (z403)
    if variante in ("V1", "V2"):
        X["ma2"] = A[:, :2].mean(1)       # s2 en tn
        X["ma3"] = A[:, :3].mean(1)       # s3 en tn
        X["med3"] = np.median(A[:, :3], 1)  # roll_med_3 en tn
        X["dprom24_abs"] = A[:, 0] - A.mean(1)  # dprom en tn (12m aquí)
        X["ncli_0"] = w_ncli.loc[d.index, cols[0]].values  # ncli_idx -> conteo
    if variante == "V2":
        cats = [cat_de.get(p, "SIN_CAT") for p in d.index]
        Ccat = w_cat.loc[cats, cols].values
        X["cat_ma2"] = Ccat[:, :2].mean(1)                 # categoría reciente (tn)
        cat_prom = np.nanmean(Ccat, 1)
        X["cat_tend"] = Ccat[:, :2].mean(1) / (cat_prom + EPS)  # categoría alza/baja
    return X


def entrenar_predecir(variante: str, cut_train: int, cut_pred: int) -> pd.Series:
    obj = mes_menos(cut_train, -2)
    Xtr = matriz(cut_train, variante)
    Xtr = Xtr[Xtr.index.isin(magicos)]
    ytr = wide.loc[Xtr.index, obj]
    ok = ytr.notna()
    Xtr, ytr = Xtr[ok.values], ytr[ok]
    Xm = np.column_stack([np.ones(len(Xtr)), Xtr.values])
    coef, *_ = np.linalg.lstsq(Xm, ytr.values, rcond=None)
    Xpr = matriz(cut_pred, variante)
    pred = np.column_stack([np.ones(len(Xpr)), Xpr.values]) @ coef
    return pd.Series(pred, index=Xpr.index).clip(lower=0)


# ---------- backtest honesto: train 201712->201802, eval 201812->201902 ----------
real = wide[201902].dropna()
prom12 = wide.loc[real.index, [mes_menos(201812, k) for k in range(12)]].mean(axis=1)


def wape(p): return float(np.abs(real - p.reindex(real.index).fillna(prom12)).sum() / real.sum())


print("=== backtest honesto (train 201712, eval 201812->feb19) ===")
res_bt = {}
for v in ("V0", "V1", "V2"):
    p = entrenar_predecir(v, 201712, 201812)
    res_bt[v] = wape(p)
    print(f"  {v}: WAPE {res_bt[v]:.4f}")
print(f"  (ref: z403 real da 0.231 en Kaggle; naif prom {wape(prom12):.4f})")

# ---------- final: train 201812->201902, predecir 201912 ----------
tb_prom = (fb.ventas.filter(pl.col("periodo").is_between(201901, 201912))
           .group_by("product_id").agg(pl.col("tn").mean()))
apredecir = cargar_apredecir()


def submit(v: str) -> None:
    pred = entrenar_predecir(v, 201812, 201912)
    tb = pl.DataFrame({"product_id": pred.index.to_list(), "pred": pred.to_list()})
    out = (apredecir.join(tb_prom, on="product_id", how="left")
           .join(tb, on="product_id", how="left")
           .with_columns(pl.coalesce([pl.col("pred"), pl.col("tn")]).alias("tn"))
           .select("product_id", "tn").sort("product_id"))
    assert out.height == 780 and out["tn"].null_count() == 0
    f = RUTA_EXP / f"exp_v5bis_{v}.csv"
    out.write_csv(f)
    subprocess.run([str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
                    "-c", COMPETENCIA, "-f", str(f), "-m", f"exp_v5bis OLS aumentado {v}"],
                   check=True)
    print(f"submit OK -> {f.name}")


# submitea las variantes que no empeoran claramente el backtest vs V0 (máx 2)
candidatas = [v for v in ("V1", "V2") if res_bt[v] <= res_bt["V0"] + 0.003]
if not candidatas:
    print("\nninguna variante mejora el backtest; submiteo V0 (control) + la mejor")
    candidatas = [min(("V1", "V2"), key=lambda v: res_bt[v])]
for v in candidatas[:2]:
    submit(v)

# guardar coeficientes de la mejor variante final
mejor = min(res_bt, key=res_bt.get)
Xtr = matriz(201812, mejor); Xtr = Xtr[Xtr.index.isin(magicos)]
ytr = wide.loc[Xtr.index, 201902]; ok = ytr.notna()
coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), Xtr[ok.values].values]),
                           ytr[ok].values, rcond=None)
nombres = ["const"] + list(Xtr.columns)
(RUTA_EXP / "coeficientes.txt").write_text(
    f"exp_v5_bis - coeficientes OLS variante {mejor}\n\n" +
    "\n".join(f"{n}: {c:+.4f}" for n, c in zip(nombres, coef)) +
    f"\n\nbacktest: " + " ".join(f"{k}={v:.4f}" for k, v in res_bt.items()))
print(f"\ncoeficientes de {mejor} guardados")
