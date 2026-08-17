"""REPRODUCE EL MODELO GANADOR — ensamble OLS + LGBM calibrado (Kaggle 0.225).

Pipeline completo y DETERMINISTA (semilla 102191, sin Optuna: usa los
hiperparámetros ya seleccionados). Reproduce, desde los datos crudos y
los artefactos incluidos, la submission ganadora y verifica que coincide
bit a bit con submission_ganador_0227.csv.

Estructura del modelo ganador:
  predicción_final = factor(tendencia_p) * (0.70*OLS_magica + 0.30*LGBM_producto)
  factor(p) = clip(1.06 - 0.06*z_tendencia(p), 0.90, 1.20)  [exp_v23]

  - OLS_magica: regresión lineal (z403) sobre los 182 productos_magicos,
    12 lags, corte 201812->201902, fallback promedio 2019.
  - LGBM_producto: LightGBM (objective=l1) sobre target ratio tn(t+2)/prom,
    con el set canónico (features_lgbm.FeatureBuilder, 76 feats auditadas)
    + stacking de la OLS + categóricas cat1/2/3 + features de categoría
    cat3 (features_categoria.parquet) + features de ocurrencia/hurdle
    (features_ocurrencia_v16.parquet). Hiperparámetros FIJOS.

Reproducción RÁPIDA (este script, ~2 min): usa los dos parquets de
features incluidos en artefactos/.
Reproducción COMPLETA (regenerar esos parquets desde cero, ~1.5 h): ver
README.md, sección "Reproducción completa".

Uso:  ../../.venv/bin/python reproduce_ganador.py
"""

import json

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

RUTA = Path(__file__).resolve().parent
sys.path.insert(0, str(RUTA / "codigo"))
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_productos  # noqa: E402
from features_lgbm import FeatureBuilder  # noqa: E402

warnings.filterwarnings("ignore")
SEMILLA = 102191
EPS = 1e-6
ART = RUTA / "artefactos"

# hiperparámetros FIJOS del LGBM de producto (los que dieron el mejor LB;
# el fijo superó a Optuna en todos los experimentos del proyecto)
PARAMS_LGBM = dict(objective="l1", random_state=SEMILLA, verbosity=-1,
                   n_estimators=400, learning_rate=0.03, num_leaves=31,
                   min_child_samples=30, colsample_bytree=0.8,
                   subsample=0.8, subsample_freq=1)
PESO_OLS = 0.70        # ensamble: 70% OLS + 30% LGBM
# calibración post por TENDENCIA (exp_v23): factor(p)=clip(a+b*z, 0.90, 1.20)
# donde z = tendencia lineal 12m estandarizada. Corrige el sesgo del LGBM
# modulado por reversión a la media (b<0). Supera al 1.03 global (0.226->0.225).
CALIB_A, CALIB_B = 1.06, -0.06
CALIB_MU, CALIB_SD = None, None   # se fijan del backtest más abajo

fb = FeatureBuilder()
wide = fb.wide
mes_menos = fb.mes_menos


# productos_magicos: lista curada del notebook z403 (guardada como artefacto
# para que este paquete sea autocontenido, no dependa del notebook)
magicos = set(json.loads((ART / "productos_magicos.json").read_text())["productos_magicos"])
prod = cargar_productos().unique("product_id").to_pandas().set_index("product_id")
train_prods = wide.index[wide.notna().sum(axis=1) >= 12]
conteo_c3 = prod.loc[prod.index.isin(train_prods), "cat3"].value_counts()
c3_validas = set(conteo_c3[conteo_c3 > 10].index)
CATS1 = sorted(set(prod["cat1"].dropna().unique()) | {"OTROS"})
CATS2 = sorted(set(prod["cat2"].dropna().unique()) | {"OTROS"})
CATS3 = sorted(c3_validas | {"OTROS"})

# artefactos de features (incluidos; ver README para regenerarlos)
fcat = pd.read_parquet(ART / "features_categoria.parquet")
fcat_c = {c: g.set_index("product_id").drop(columns="corte") for c, g in fcat.groupby("corte")}
focc = pd.read_parquet(ART / "features_ocurrencia_v16.parquet")
focc_c = {c: g.set_index("product_id").drop(columns="corte") for c, g in focc.groupby("corte")}


def ols_coef(cut_train: int):
    cols = [mes_menos(cut_train, k) for k in range(12)]
    obj = mes_menos(cut_train, -2)
    if cols[-1] not in wide.columns or obj not in wide.columns:
        return None
    d = wide.loc[wide.index.isin(magicos), cols + [obj]].dropna()
    if len(d) < 30:
        return None
    X = np.column_stack([np.ones(len(d)), d[cols].values])
    return np.linalg.lstsq(X, d[obj].values, rcond=None)[0]


def armar(cut: int, con_clase: bool) -> pd.DataFrame:
    d = fb.armar_corte(cut, con_clase)
    for nombre, lag in [("pred_ols_yoy", 12), ("pred_ols_rec", 2)]:
        coef = ols_coef(mes_menos(cut, lag))
        if coef is not None:
            cols = [mes_menos(cut, k) for k in range(12)]
            da = wide[cols].dropna()
            p = pd.Series(np.column_stack([np.ones(len(da)), da.values]) @ coef,
                          index=da.index).clip(lower=0)
            d[nombre] = (p.reindex(d.index) / d["_prom"]).values
        else:
            d[nombre] = np.nan
    p = prod.reindex(d.index)
    d["cat1"] = pd.Categorical(p["cat1"].fillna("OTROS"), categories=CATS1)
    d["cat2"] = pd.Categorical(p["cat2"].fillna("OTROS"), categories=CATS2)
    c3 = p["cat3"].where(p["cat3"].isin(c3_validas), "OTROS").fillna("OTROS")
    d["cat3"] = pd.Categorical(c3, categories=CATS3)
    d = d.join(fcat_c[cut], how="left").join(focc_c[cut], how="left")
    return d


# ---------- 1) OLS mágica: pata del ensamble = salida del notebook z403 ----------
# La pata OLS del ensamble es la predicción del notebook z403 del profesor
# (regresión mágica, su implementación con shift por fila). Se toma de
# artefactos/pred_ols_magica.csv, que el profe reproduce corriendo z403.
# (NOTA: el stacking OLS *interno* del LGBM sí se recalcula acá con
#  lags de calendario vía ols_coef — es una feature del LGBM, no la pata.)
print("[1/4] OLS mágica (pata del ensamble = salida de z403)...")
pred_ols = (pl.read_csv(ART / "pred_ols_magica.csv")
            .to_pandas().set_index("product_id")["tn"].clip(lower=0))
tb_prom = (fb.ventas.filter(pl.col("periodo").is_between(201901, 201912))
           .group_by("product_id").agg(pl.col("tn").mean()))

# ---------- 2) LGBM de producto (params fijos) ----------
print("[2/4] LGBM de producto (22 cortes, params fijos)...")
d_final = pd.concat([armar(mes_menos(201910, k), True) for k in range(22)])
d_fut = armar(201912, False)
m = lgb.LGBMRegressor(**PARAMS_LGBM)
m.fit(d_final.drop(columns=["clase_ratio", "_prom"]), d_final["clase_ratio"])
ratio = pd.Series(m.predict(d_fut.drop(columns="_prom")), index=d_fut.index).clip(lower=0)
pred_lgbm = ratio * d_fut["_prom"]

# ---------- 3) ensamble + calibración por tendencia ----------
print("[3/4] ensamble 0.70*OLS + 0.30*LGBM, calibrado por tendencia...")


def tendencia(cut):
    cols = [mes_menos(cut, k) for k in range(12)]
    d = wide[cols].dropna(); prom = d.mean(axis=1)
    d = d[prom > EPS]; prom = prom[prom > EPS]
    Ln = d.div(prom, axis=0).values[:, ::-1]
    t = np.arange(12) - 5.5
    y = Ln - Ln.mean(axis=1, keepdims=True)
    return pd.Series((y * t).sum(axis=1) / (t ** 2).sum(), index=d.index)


tb = tendencia(201812)                    # backtest: fija mu/sd
CALIB_MU, CALIB_SD = float(tb.mean()), float(tb.std())
z_fin = ((tendencia(201912) - CALIB_MU) / (CALIB_SD + EPS)).clip(-3, 3)

apredecir = cargar_apredecir()
tb_ols = pl.DataFrame({"product_id": pred_ols.index.to_list(), "ols": pred_ols.to_list()})
tb_lgb = pl.DataFrame({"product_id": pred_lgbm.index.to_list(), "lgbm": pred_lgbm.to_list()})
tb_f = pl.DataFrame({"product_id": z_fin.index.to_list(),
                     "factor": np.clip(CALIB_A + CALIB_B * z_fin.values, 0.90, 1.20).tolist()})
base = (apredecir.join(tb_prom, on="product_id", how="left")
        .join(tb_ols, on="product_id", how="left")
        .join(tb_lgb, on="product_id", how="left")
        .join(tb_f, on="product_id", how="left")
        .with_columns(pl.coalesce([pl.col("ols"), pl.col("tn")]).alias("ols"),
                      pl.coalesce([pl.col("lgbm"), pl.col("tn")]).alias("lgbm"),
                      pl.col("factor").fill_null(CALIB_A)))  # sin tendencia -> factor base
out = (base.with_columns(
    (pl.col("factor") * (PESO_OLS * pl.col("ols") + (1 - PESO_OLS) * pl.col("lgbm")))
    .clip(lower_bound=0).alias("tn"))
    .select("product_id", "tn").sort("product_id"))
assert out.height == 780 and out["tn"].null_count() == 0

archivo = RUTA / "submission_reproducida.csv"
out.write_csv(archivo)
print(f"    escrito: {archivo.name} | suma tn = {out['tn'].sum():.1f}")

# ---------- 4) verificación contra la submission ganadora ----------
print("[4/4] verificación...")
orig = pl.read_csv(ART / "submission_ganador_0225.csv").sort("product_id")
comp = out.join(orig.rename({"tn": "tn_orig"}), on="product_id")
maxdif = (comp["tn"] - comp["tn_orig"]).abs().max()
print(f"    max diferencia vs submission_ganador_0225.csv: {maxdif:.2e}")
print("    ✓ REPRODUCE EXACTO" if maxdif < 1e-6 else "    ✗ hay diferencias (revisar entorno/versiones)")
