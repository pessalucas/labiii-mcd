"""exp_v11: ensamble OLS mágica + LGBM v11 (pata mejorada 0.240).

Elige el peso con BACKTEST local (dic18->feb19) para no barrer contra el
leaderboard, luego lo aplica a las predicciones finales y submitea el
mejor local + el 80/20 como seguro.
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
COMPETENCIA = "labo-iii-2026-ba"
SEMILLA = 102191
EPS = 1e-6
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, colsample_bytree=0.8, subsample=0.8)

fb = FeatureBuilder()
wide = fb.wide
mes_menos = fb.mes_menos
fcat = pd.read_parquet(RUTA_EXP / "features_categoria.parquet")
fcat_por_corte = {c: g.set_index("product_id").drop(columns="corte")
                  for c, g in fcat.groupby("corte")}


def cargar_magicos() -> set[int]:
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


def ols_coef(cut_train: int):
    cols = [mes_menos(cut_train, k) for k in range(12)]
    obj = mes_menos(cut_train, -2)
    d = wide.loc[wide.index.isin(magicos), cols + [obj]].dropna()
    X = np.column_stack([np.ones(len(d)), d[cols].values])
    return np.linalg.lstsq(X, d[obj].values, rcond=None)[0]


def ols_pred(coef, cut: int) -> pd.Series:
    cols = [mes_menos(cut, k) for k in range(12)]
    d = wide[cols].dropna()
    return pd.Series(np.column_stack([np.ones(len(d)), d.values]) @ coef,
                     index=d.index).clip(lower=0)


def armar(cut: int, con_clase: bool) -> pd.DataFrame:
    d = fb.armar_corte(cut, con_clase)
    for nombre, lag in [("pred_ols_yoy", 12), ("pred_ols_rec", 2)]:
        try:
            p = ols_pred(ols_coef(mes_menos(cut, lag)), cut)
            d[nombre] = (p.reindex(d.index) / d["_prom"]).values
        except Exception:
            d[nombre] = np.nan
    p = prod.reindex(d.index)
    d["cat1"] = pd.Categorical(p["cat1"].fillna("OTROS"), categories=CATS1)
    d["cat2"] = pd.Categorical(p["cat2"].fillna("OTROS"), categories=CATS2)
    c3 = p["cat3"].where(p["cat3"].isin(c3_validas), "OTROS").fillna("OTROS")
    d["cat3"] = pd.Categorical(c3, categories=CATS3)
    d = d.join(fcat_por_corte[cut], how="left")
    return d


def lgbm_pred(cortes_train: list[int], cut_pred: int) -> pd.Series:
    dtr = pd.concat([armar(c, True) for c in cortes_train])
    m = lgb.LGBMRegressor(objective="l1", random_state=SEMILLA, verbosity=-1,
                          subsample_freq=1, **PARAMS)
    m.fit(dtr.drop(columns=["clase_ratio", "_prom"]), dtr["clase_ratio"])
    dev = armar(cut_pred, False)
    ratio = pd.Series(m.predict(dev.drop(columns="_prom")), index=dev.index).clip(lower=0)
    return ratio * dev["_prom"]


# ---------- BACKTEST: elegir peso en dic18 -> feb19 ----------
real = wide[201902].dropna()
prom12_bt = wide.loc[real.index, [mes_menos(201812, k) for k in range(12)]].mean(axis=1)

ols_bt = ols_pred(ols_coef(201712), 201812).reindex(real.index).fillna(prom12_bt)
lgbm_bt = lgbm_pred([mes_menos(201810, k) for k in range(10)], 201812)
lgbm_bt = lgbm_bt.reindex(real.index).fillna(prom12_bt)


def wape(p): return float(np.abs(real - p).sum() / real.sum())


print(f"backtest OLS solo:  {wape(ols_bt):.4f}")
print(f"backtest LGBM solo: {wape(lgbm_bt):.4f}")
mejor_w, mejor_wape = 1.0, 1.0
for w in np.arange(0, 1.01, 0.05):
    e = wape((w * ols_bt + (1 - w) * lgbm_bt))
    if e < mejor_wape:
        mejor_wape, mejor_w = e, w
print(f"mejor peso local: {mejor_w:.2f} OLS / {1-mejor_w:.2f} LGBM -> WAPE {mejor_wape:.4f}")

# ---------- FINAL: aplicar peso a las predicciones ya submiteadas ----------
ols_fin = pl.read_csv(RUTA_PROYECTO / "exp/LR01/linreg.csv").rename({"tn": "ols"})
lgbm_fin = pl.read_csv(RUTA_EXP / "exp_v11_fijo.csv").rename({"tn": "lgbm"})
base = ols_fin.join(lgbm_fin, on="product_id")


def submit(nombre: str, w: float) -> None:
    out = base.with_columns(
        (w * pl.col("ols") + (1 - w) * pl.col("lgbm")).clip(lower_bound=0).alias("tn")
    ).select("product_id", "tn").sort("product_id")
    assert out.height == 780 and out["tn"].null_count() == 0
    f = RUTA_EXP / f"exp_v11_{nombre}.csv"
    out.write_csv(f)
    subprocess.run([str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
                    "-c", COMPETENCIA, "-f", str(f),
                    "-m", f"exp_v11 ensamble OLS{int(w*100)}/LGBMv11{int((1-w)*100)}"],
                   check=True)
    print(f"submit OK -> {f.name} (w={w:.2f})")


submit(f"ens_best_w{int(mejor_w*100)}", mejor_w)
if abs(mejor_w - 0.8) > 0.01:
    submit("ens_w80", 0.80)
