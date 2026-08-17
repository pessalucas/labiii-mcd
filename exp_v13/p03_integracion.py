"""exp_v13 - paso 3: integración final al modelo de producto.

Pipeline v11 (76 limpias + stacking OLS + categóricas + categoría cat3)
+ features de CAT4 (c4_*/sh4_*, del paso 1)
+ features de OCURRENCIA con sustitución cat4 (occ_*, del paso 2).
Reusa la hiperparametría óptima de v11 (no re-tuneo). 2 submits (fijo +
v11opt) + FI reportando el gain de las familias nuevas.
"""

import json
import re
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
RUTA_V11 = RUTA_EXP.parent / "exp_v11"
COMPETENCIA = "labo-iii-2026-ba"
SEMILLA = 102191
EPS = 1e-6
PARAMS_FIJOS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
                    min_child_samples=30, colsample_bytree=0.8, subsample=0.8)
PARAMS_V11OPT = dict(n_estimators=602, learning_rate=0.029502796889054175,
                     num_leaves=49, min_child_samples=62,
                     lambda_l1=1.4254613967740818e-08, lambda_l2=0.5358744434094257,
                     colsample_bytree=0.7292874334451034, subsample=0.6062155688158164)

fb = FeatureBuilder()
wide = fb.wide
mes_menos = fb.mes_menos
fcat = pd.read_parquet(RUTA_V11 / "features_categoria.parquet")
fcat_c = {c: g.set_index("product_id").drop(columns="corte") for c, g in fcat.groupby("corte")}
fc4 = pd.read_parquet(RUTA_EXP / "features_cat4.parquet")
fc4_c = {c: g.set_index("product_id").drop(columns="corte") for c, g in fc4.groupby("corte")}
focc = pd.read_parquet(RUTA_EXP / "features_ocurrencia_v13.parquet")
focc_c = {c: g.set_index("product_id").drop(columns="corte") for c, g in focc.groupby("corte")}
print(f"cat3 {next(iter(fcat_c.values())).shape[1]} | cat4 {next(iter(fc4_c.values())).shape[1]} "
      f"| occ {next(iter(focc_c.values())).shape[1]}")


def cargar_magicos():
    nb = json.loads((RUTA_PROYECTO / "src/Estadistica/z403_RegresionLineal_local.ipynb").read_text())
    celda = next("".join(c["source"]) for c in nb["cells"]
                 if c["cell_type"] == "code" and "productos_magicos" in "".join(c["source"]))
    return set(int(x) for x in re.findall(r"productos_magicos = \[(.*?)\]",
               celda, flags=re.S)[-1].replace("\n", " ").split(","))


magicos = cargar_magicos()
prod = cargar_productos().unique("product_id").to_pandas().set_index("product_id")
train_prods = wide.index[wide.notna().sum(axis=1) >= 12]
conteo_c3 = prod.loc[prod.index.isin(train_prods), "cat3"].value_counts()
c3_validas = set(conteo_c3[conteo_c3 > 10].index)
CATS1 = sorted(set(prod["cat1"].dropna().unique()) | {"OTROS"})
CATS2 = sorted(set(prod["cat2"].dropna().unique()) | {"OTROS"})
CATS3 = sorted(c3_validas | {"OTROS"})


def ols_coef(cut_train):
    cols = [mes_menos(cut_train, k) for k in range(12)]; obj = mes_menos(cut_train, -2)
    if cols[-1] not in wide.columns or obj not in wide.columns:
        return None
    d = wide.loc[wide.index.isin(magicos), cols + [obj]].dropna()
    if len(d) < 30:
        return None
    return np.linalg.lstsq(np.column_stack([np.ones(len(d)), d[cols].values]), d[obj].values, rcond=None)[0]


def armar(cut, con_clase):
    d = fb.armar_corte(cut, con_clase)
    for nombre, lag in [("pred_ols_yoy", 12), ("pred_ols_rec", 2)]:
        coef = ols_coef(mes_menos(cut, lag))
        if coef is not None:
            cols = [mes_menos(cut, k) for k in range(12)]; da = wide[cols].dropna()
            p = pd.Series(np.column_stack([np.ones(len(da)), da.values]) @ coef, index=da.index).clip(lower=0)
            d[nombre] = (p.reindex(d.index) / d["_prom"]).values
        else:
            d[nombre] = np.nan
    p = prod.reindex(d.index)
    d["cat1"] = pd.Categorical(p["cat1"].fillna("OTROS"), categories=CATS1)
    d["cat2"] = pd.Categorical(p["cat2"].fillna("OTROS"), categories=CATS2)
    c3 = p["cat3"].where(p["cat3"].isin(c3_validas), "OTROS").fillna("OTROS")
    d["cat3"] = pd.Categorical(c3, categories=CATS3)
    d = d.join(fcat_c[cut], how="left").join(fc4_c[cut], how="left").join(focc_c[cut], how="left")
    return d


def fit(dtr, params):
    m = lgb.LGBMRegressor(objective="l1", random_state=SEMILLA, verbosity=-1, subsample_freq=1, **params)
    m.fit(dtr.drop(columns=["clase_ratio", "_prom"]), dtr["clase_ratio"]); return m


def wape_pred(m, dev, real):
    ratio = pd.Series(m.predict(dev.drop(columns="_prom")), index=dev.index).clip(lower=0)
    pred = ratio * dev["_prom"]; comun = pred.index.intersection(real.index)
    return float(np.abs(real[comun] - pred[comun]).sum() / real[comun].sum())


FOLDS = [(201808, 1.0), (201810, 1.0), (201812, 2.0)]
folds = []
for cut_eval, peso in FOLDS:
    n = (pd.Period(str(cut_eval), freq="M") - pd.Period("201801", freq="M")).n - 1
    dtr = pd.concat([armar(mes_menos(cut_eval, k), True) for k in range(2, 2 + n)])
    dev = armar(cut_eval, False); real = wide[mes_menos(cut_eval, -2)].dropna()
    folds.append((dtr, dev, real, peso))
d_final = pd.concat([armar(mes_menos(201910, k), True) for k in range(22)])
d_fut = armar(201912, False)
print(f"train final: {len(d_final)} filas, {d_final.shape[1]-2} features")


def evaluar(params):
    t, w = 0.0, 0.0
    for dtr, dev, real, peso in folds:
        t += peso * wape_pred(fit(dtr, params), dev, real); w += peso
    return t / w


print(f"backtest fijo:   {evaluar(PARAMS_FIJOS):.4f} (v11 0.2395 | v12 0.2363)")
print(f"backtest v11opt: {evaluar(PARAMS_V11OPT):.4f} (v11 0.2321 | v12 0.2361)")

tb_prom = (fb.ventas.filter(pl.col("periodo").is_between(201901, 201912))
           .group_by("product_id").agg(pl.col("tn").mean()))
apredecir = cargar_apredecir()
fi_txt = []


def submit(nombre, params):
    m = fit(d_final, params); b = m.booster_
    imp = pd.DataFrame({"gain": b.feature_importance("gain"), "splits": b.feature_importance("split")},
                       index=b.feature_name())
    imp["gain_pct"] = (100 * imp["gain"] / imp["gain"].sum()).round(1)
    imp = imp.sort_values("gain", ascending=False)
    g4 = imp[imp.index.str.startswith(("c4_", "sh4_"))]["gain_pct"].sum()
    gocc = imp[imp.index.str.startswith("occ_")]["gain_pct"].sum()
    print(f"\n=== FI {nombre} (top 12) ===\n{imp.head(12).round({'gain':0}).to_string()}")
    print(f"gain cat4 (c4_+sh4_): {g4:.1f}% | gain occ_: {gocc:.1f}%")
    fi_txt.append(f"\n{'='*60}\n{nombre} | cat4={g4:.1f}% occ={gocc:.1f}%\n{'='*60}\n" + imp.round(1).to_string())
    ratio = pd.Series(m.predict(d_fut.drop(columns="_prom")), index=d_fut.index).clip(lower=0)
    pred = ratio * d_fut["_prom"]
    tb = pl.DataFrame({"product_id": pred.index.to_list(), "pred": pred.to_list()})
    out = (apredecir.join(tb_prom, on="product_id", how="left").join(tb, on="product_id", how="left")
           .with_columns(pl.coalesce([pl.col("pred"), pl.col("tn")]).alias("tn"))
           .select("product_id", "tn").sort("product_id"))
    assert out.height == 780 and out["tn"].null_count() == 0
    f = RUTA_EXP / f"exp_v13_{nombre}.csv"; out.write_csv(f)
    subprocess.run([str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
                    "-c", COMPETENCIA, "-f", str(f), "-m", f"exp_v13 v12+cat4+sustitucion {nombre}"], check=True)
    print(f"submit OK -> {f.name}")


submit("fijo", PARAMS_FIJOS)
submit("v11opt", PARAMS_V11OPT)
(RUTA_EXP / "feature_importance.txt").write_text("exp_v13 FI\n" + "\n".join(fi_txt))
print("\nFI guardado")
