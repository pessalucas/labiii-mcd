"""exp_v19: LGBM SIN stacking OLS -> ensamble (maximizar diversidad).

Hipótesis (de v18 + observación del usuario): el stacking OLS dentro del
LGBM (pred_ols_yoy/rec, ~0.6% del gain) lo acerca a la OLS y reduce la
diversidad del ensamble. Quitándolo, el LGBM se vuelve más ORTOGONAL a la
OLS -> el ensamble podría bajar del 0.227, aunque el LGBM solo empeore un
poco.

Pipeline = ganador v16 (v11 + cat3 + ocurrencia_v16) SIN pred_ols_*.
Params fijos. Compara LGBM solo y ensamble a varios pesos.
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_productos
from features_lgbm import FeatureBuilder

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
PRU = RUTA_EXP.parent
COMPETENCIA = "labo-iii-2026-ba"
SEMILLA = 102191
EPS = 1e-6
PARAMS = dict(objective="l1", random_state=SEMILLA, verbosity=-1,
              n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, colsample_bytree=0.8, subsample=0.8, subsample_freq=1)

fb = FeatureBuilder()
wide = fb.wide
mes_menos = fb.mes_menos
fcat = pd.read_parquet(PRU / "exp_v11" / "features_categoria.parquet")
fcat_c = {c: g.set_index("product_id").drop(columns="corte") for c, g in fcat.groupby("corte")}
focc = pd.read_parquet(PRU / "exp_v16" / "features_ocurrencia_v16.parquet")
focc_c = {c: g.set_index("product_id").drop(columns="corte") for c, g in focc.groupby("corte")}

prod = cargar_productos().unique("product_id").to_pandas().set_index("product_id")
train_prods = wide.index[wide.notna().sum(axis=1) >= 12]
conteo_c3 = prod.loc[prod.index.isin(train_prods), "cat3"].value_counts()
c3_validas = set(conteo_c3[conteo_c3 > 10].index)
CATS1 = sorted(set(prod["cat1"].dropna().unique()) | {"OTROS"})
CATS2 = sorted(set(prod["cat2"].dropna().unique()) | {"OTROS"})
CATS3 = sorted(c3_validas | {"OTROS"})


def armar(cut: int, con_clase: bool) -> pd.DataFrame:
    d = fb.armar_corte(cut, con_clase)
    # >>> SIN stacking OLS (esa es la diferencia con el ganador) <<<
    p = prod.reindex(d.index)
    d["cat1"] = pd.Categorical(p["cat1"].fillna("OTROS"), categories=CATS1)
    d["cat2"] = pd.Categorical(p["cat2"].fillna("OTROS"), categories=CATS2)
    c3 = p["cat3"].where(p["cat3"].isin(c3_validas), "OTROS").fillna("OTROS")
    d["cat3"] = pd.Categorical(c3, categories=CATS3)
    d = d.join(fcat_c[cut], how="left").join(focc_c[cut], how="left")
    return d


def fit(dtr):
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(dtr.drop(columns=["clase_ratio", "_prom"]), dtr["clase_ratio"])
    return m


def wape_pred(m, dev, real):
    ratio = pd.Series(m.predict(dev.drop(columns="_prom")), index=dev.index).clip(lower=0)
    pred = ratio * dev["_prom"]
    comun = pred.index.intersection(real.index)
    return float(np.abs(real[comun] - pred[comun]).sum() / real[comun].sum())


# backtest multi-corte
FOLDS = [(201808, 1.0), (201810, 1.0), (201812, 2.0)]
tot, pes = 0.0, 0.0
for cut_eval, peso in FOLDS:
    n = (pd.Period(str(cut_eval), freq="M") - pd.Period("201801", freq="M")).n - 1
    dtr = pd.concat([armar(mes_menos(cut_eval, k), True) for k in range(2, 2 + n)])
    dev = armar(cut_eval, False)
    real = wide[mes_menos(cut_eval, -2)].dropna()
    tot += peso * wape_pred(fit(dtr), dev, real); pes += peso
print(f"backtest multi-corte SIN stacking: {tot/pes:.4f} (v16 con stacking 0.2363)")

# modelo final
d_final = pd.concat([armar(mes_menos(201910, k), True) for k in range(22)])
d_fut = armar(201912, False)
m = fit(d_final)
ratio = pd.Series(m.predict(d_fut.drop(columns="_prom")), index=d_fut.index).clip(lower=0)
pred_lgbm = ratio * d_fut["_prom"]

# correlación con la OLS (medir la diversidad ganada)
ols = pl.read_csv(RUTA_PROYECTO / "exp/LR01/linreg.csv").to_pandas().set_index("product_id")["tn"]
comun = pred_lgbm.index.intersection(ols.index)
corr = np.corrcoef(pred_lgbm[comun], ols[comun])[0, 1]
print(f"corr(LGBM sin stacking, OLS): {corr:.4f}")

tb_prom = (fb.ventas.filter(pl.col("periodo").is_between(201901, 201912))
           .group_by("product_id").agg(pl.col("tn").mean()))
apredecir = cargar_apredecir()
tb_lgb = pl.DataFrame({"product_id": pred_lgbm.index.to_list(), "lgbm": pred_lgbm.to_list()})
tb_ols = pl.DataFrame({"product_id": ols.index.to_list(), "ols": ols.to_list()})
base = (apredecir.join(tb_prom, on="product_id", how="left")
        .join(tb_ols, on="product_id", how="left").join(tb_lgb, on="product_id", how="left")
        .with_columns(pl.coalesce([pl.col("ols"), pl.col("tn")]).alias("ols"),
                      pl.coalesce([pl.col("lgbm"), pl.col("tn")]).alias("lgbm")))


def submit(nombre, serie):
    out = base.with_columns(serie.alias("tn")).select("product_id", pl.col("tn").clip(lower_bound=0)).sort("product_id")
    assert out.height == 780 and out["tn"].null_count() == 0
    f = RUTA_EXP / f"exp_v19_{nombre}.csv"; out.write_csv(f)
    subprocess.run([str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
                    "-c", COMPETENCIA, "-f", str(f), "-m", f"exp_v19 sin stacking {nombre}"], check=True)
    print(f"submit OK -> {f.name}")


submit("lgbm_solo", pl.col("lgbm"))
submit("ens_w70", 0.70 * pl.col("ols") + 0.30 * pl.col("lgbm"))
submit("ens_w65", 0.65 * pl.col("ols") + 0.35 * pl.col("lgbm"))
