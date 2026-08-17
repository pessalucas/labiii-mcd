"""exp_v12: modelo de OCURRENCIA (hurdle) a nivel cliente x producto.

Predice P(el cliente c compra el producto p en t+2) con un LGBM binario,
lo multiplica por el monto histórico del par cuando compra, y agrega a
nivel producto dos features para el modelo de producto (v11):
  occ_demanda_esp = sum_c P(compra_cp) x monto_par_cp   (tn esperadas)
  occ_n_esperados = sum_c P(compra_cp)                  (compradores esp.)
más versiones normalizadas por el promedio 12m del producto.

Sin fuga: 2-fold TEMPORAL (modelo entrenado en un tramo predice el otro)
para los cortes de train; modelo_full para el futuro (dic-2019 -> feb-2020,
que no puede tener fuga) y como artefacto reutilizable.

GUARDA todo lo reutilizable:
  modelo_ocurrencia_full.txt   booster LGBM (portable)
  modelo_ocurrencia_full.pkl   wrapper sklearn (joblib)
  features_ocurrencia.parquet  producto x corte, listo para joinear a v11
  feature_importance.txt       FI del clasificador
  hiperparametros.json         params usados
  metricas.txt                 AUC temporal, tasa base, correlaciones
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import cargar_sellin

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
SEMILLA = 102191
EPS = 1e-6
PERIODOS = [int(p.strftime("%Y%m")) for p in pd.period_range("2017-01", "2019-12", freq="M")]
# cortes con 12 meses de historia (target = corte+2)
CORTES = [int(p.strftime("%Y%m")) for p in pd.period_range("2017-12", "2019-10", freq="M")]
CORTE_FUT = 201912

PARAMS = dict(objective="binary", random_state=SEMILLA, verbosity=-1,
              n_estimators=300, learning_rate=0.05, num_leaves=48,
              min_child_samples=100, colsample_bytree=0.8,
              subsample=0.8, subsample_freq=1)


def mes_menos(p: int, k: int) -> int:
    return int((pd.Period(str(p), freq="M") - k).strftime("%Y%m"))


# ---------- paneles ----------
sellin = cargar_sellin()
pares = (sellin.group_by("customer_id", "product_id", "periodo")
         .agg(pl.col("tn").sum()))
w_par = (pares.to_pandas().pivot_table(index=["customer_id", "product_id"],
         columns="periodo", values="tn", aggfunc="sum").reindex(columns=PERIODOS))
# ceros desde la primera compra del par
first = w_par.apply(lambda r: r.first_valid_index(), axis=1)
Mp = w_par.values
colsarr = np.array(PERIODOS)
for i in range(len(w_par)):
    Mp[i, colsarr >= first.iloc[i]] = np.nan_to_num(Mp[i, colsarr >= first.iloc[i]])
w_par = pd.DataFrame(Mp, index=w_par.index, columns=PERIODOS)
print(f"panel de pares: {w_par.shape[0]:,} pares x {w_par.shape[1]} meses")

cli = sellin.group_by("customer_id", "periodo").agg(pl.col("tn").sum())
w_cli = (cli.to_pandas().pivot(index="customer_id", columns="periodo", values="tn")
         .reindex(columns=PERIODOS))
prod = sellin.group_by("product_id", "periodo").agg(pl.col("tn").sum())
w_prod = (prod.to_pandas().pivot(index="product_id", columns="periodo", values="tn")
          .reindex(columns=PERIODOS))
pm = w_prod.apply(lambda r: r.first_valid_index(), axis=1)
for pid in w_prod.index:
    w_prod.loc[pid, w_prod.columns >= pm[pid]] = w_prod.loc[pid, w_prod.columns >= pm[pid]].fillna(0.0)


def armar_corte(cut: int, con_target: bool) -> pd.DataFrame:
    cols12 = [mes_menos(cut, k) for k in range(12)]
    L = w_par[cols12]
    activos = L.notna().all(axis=1) & (L.fillna(0).sum(axis=1) > 0)
    L = L[activos]
    A = L.values
    compra = A > 0
    d = pd.DataFrame(index=L.index)
    # recencia / frecuencia / gaps (drivers de ocurrencia)
    d["recencia"] = np.where(compra.any(1), compra.argmax(1), 12)
    d["freq3"] = compra[:, :3].sum(1)
    d["freq6"] = compra[:, :6].sum(1)
    d["freq12"] = compra[:, :12].sum(1)
    d["gap_prom"] = 12.0 / (compra.sum(1) + EPS)
    ceros = ~compra
    racha = np.zeros(len(A)); act = np.zeros(len(A))
    for k in range(12):
        act = np.where(ceros[:, k], act + 1, 0); racha = np.maximum(racha, act)
    d["racha_ceros"] = racha
    d["compro_t0"] = compra[:, 0].astype(int)
    d["compro_t1"] = compra[:, 1].astype(int)
    # monto reciente del par (nivel)
    d["monto_mean"] = A.sum(1) / (compra.sum(1) + EPS)
    d["monto_s3"] = A[:, :3].mean(1)
    d["mes_obj"] = mes_menos(cut, -2) % 100
    # contexto cliente
    cids = d.index.get_level_values(0)
    C = w_cli.loc[cids, cols12].values
    d["c_log_tn"] = np.log1p(np.nansum(C, 1))
    d["c_idx"] = np.nanmean(C[:, :2], 1) / (np.nanmean(C, 1) + EPS)
    # contexto producto
    pids = d.index.get_level_values(1)
    P = w_prod.loc[pids, cols12].values
    d["q_log_prom"] = np.log1p(np.nanmean(P, 1))
    if con_target:
        obj = mes_menos(cut, -2)
        d["y"] = (w_par.loc[L.index, obj].fillna(0).values > 0).astype(int) if obj in w_par.columns else np.nan
        d = d.dropna(subset=["y"])
    return d


FEATURES = ["recencia", "freq3", "freq6", "freq12", "gap_prom", "racha_ceros",
            "compro_t0", "compro_t1", "monto_mean", "monto_s3", "mes_obj",
            "c_log_tn", "c_idx", "q_log_prom"]


def fit(cortes: list[int]) -> lgb.LGBMClassifier:
    d = pd.concat([armar_corte(c, True) for c in cortes])
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(d[FEATURES], d["y"])
    return m


# ---------- 2-fold temporal (sin fuga para el train del modelo de producto) ----------
early = [c for c in CORTES if mes_menos(c, -2) <= 201812]   # target <= dic18
late = [c for c in CORTES if mes_menos(c, -2) >= 201901]    # target >= ene19
print(f"fold early: {len(early)} cortes | fold late: {len(late)} cortes")
m_early = fit(early)   # predice los cortes 'late'
m_late = fit(late)     # predice los cortes 'early'
m_full = fit(CORTES)   # para el futuro + artefacto reutilizable

# holdout temporal para métrica honesta: entreno early, evalúo en late
dev = pd.concat([armar_corte(c, True) for c in late])
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(dev["y"], m_early.predict_proba(dev[FEATURES])[:, 1])
tasa_base = dev["y"].mean()
print(f"AUC ocurrencia (train early -> eval late): {auc:.4f} | tasa base compra: {tasa_base:.3f}")


# ---------- generar features agregadas a producto x corte ----------
def features_producto(cut: int, modelo) -> pd.DataFrame:
    d = armar_corte(cut, con_target=False)
    p = modelo.predict_proba(d[FEATURES])[:, 1]
    monto = d["monto_mean"].values
    pid = d.index.get_level_values(1)
    tmp = pd.DataFrame({"product_id": pid, "p": p, "dem": p * monto})
    agg = tmp.groupby("product_id").agg(occ_n_esperados=("p", "sum"),
                                        occ_demanda_esp=("dem", "sum"))
    prom12 = w_prod.loc[agg.index, [mes_menos(cut, k) for k in range(12)]].mean(axis=1)
    agg["occ_demanda_norm"] = agg["occ_demanda_esp"] / (prom12 + EPS)
    agg["corte"] = cut
    return agg.reset_index()


tablas = []
for c in CORTES:
    modelo = m_late if c in early else m_early   # el que NO vio ese corte
    tablas.append(features_producto(c, modelo))
tablas.append(features_producto(CORTE_FUT, m_full))   # futuro con modelo full
feat = pd.concat(tablas, ignore_index=True)
feat.to_parquet(RUTA_EXP / "features_ocurrencia.parquet")
print(f"features_ocurrencia.parquet: {feat.shape[0]} filas producto x corte")

# sanity: correlación de la demanda esperada con la venta real del producto
chk = features_producto(201812, m_early).set_index("product_id")
real = w_prod[201902].reindex(chk.index)
corr = np.corrcoef(chk["occ_demanda_esp"], real.fillna(0))[0, 1]
print(f"corr(occ_demanda_esp @dic18, venta real feb19): {corr:.3f}")

# ---------- GUARDAR artefactos ----------
m_full.booster_.save_model(str(RUTA_EXP / "modelo_ocurrencia_full.txt"))
joblib.dump(m_full, RUTA_EXP / "modelo_ocurrencia_full.pkl")
json.dump(PARAMS, open(RUTA_EXP / "hiperparametros.json", "w"), indent=2)

imp = pd.Series(m_full.booster_.feature_importance(importance_type="gain"),
                index=m_full.booster_.feature_name()).sort_values(ascending=False)
imp_pct = (100 * imp / imp.sum()).round(1)
(RUTA_EXP / "feature_importance.txt").write_text(
    "exp_v12 - FI del clasificador de ocurrencia (gain %)\n\n" + imp_pct.to_string())
print("\nFI ocurrencia:\n" + imp_pct.head(10).to_string())

(RUTA_EXP / "metricas.txt").write_text(
    f"exp_v12 - modelo de ocurrencia (hurdle)\n\n"
    f"panel: {w_par.shape[0]} pares\n"
    f"AUC (train early -> eval late): {auc:.4f}\n"
    f"tasa base de compra (t+2): {tasa_base:.4f}\n"
    f"corr(occ_demanda_esp, venta real) @dic18->feb19: {corr:.3f}\n"
    f"hiperparametros: {PARAMS}\n"
    f"features del clasificador: {FEATURES}\n"
    f"salida (features_ocurrencia.parquet): occ_n_esperados, "
    f"occ_demanda_esp, occ_demanda_norm x (product_id, corte)\n")
print("\nartefactos guardados en exp_v12/ (pkl, txt, parquet, json)")
