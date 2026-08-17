"""exp_v21: integración = pipeline v11 + features de OCURRENCIA (hurdle).

Toma el mejor pipeline (v11: 76 limpias + stacking OLS + categóricas +
categoría/penetración) y le suma las 3 features de ocurrencia agregadas
(occ_n_esperados, occ_demanda_esp, occ_demanda_norm) del parquet de
exp_v21/p01. Optuna 30 + fijo. 2 submits. Reporta gain de occ_*.
"""

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_productos
from features_lgbm import FeatureBuilder

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
RUTA_EXP = Path(__file__).resolve().parent
RUTA_V11 = RUTA_EXP.parent / "exp_v11"
COMPETENCIA = "labo-iii-2026-ba"
SEMILLA = 102191
EPS = 1e-6
N_TRIALS = 30
PARAMS_FIJOS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
                    min_child_samples=30, colsample_bytree=0.8, subsample=0.8)

fb = FeatureBuilder()
wide = fb.wide
mes_menos = fb.mes_menos

# features de categoría (v11) + ocurrencia (v12), indexadas por corte
fcat = pd.read_parquet(RUTA_V11 / "features_categoria.parquet")
fcat_c = {c: g.set_index("product_id").drop(columns="corte") for c, g in fcat.groupby("corte")}
focc = pd.read_parquet(RUTA_EXP / "features_ocurrencia_v21.parquet")
focc_c = {c: g.set_index("product_id").drop(columns="corte") for c, g in focc.groupby("corte")}
fext = pd.read_parquet(RUTA_EXP / "features_extra_regresor.parquet")
fext_c = {c: g.set_index("product_id").drop(columns="corte") for c, g in fext.groupby("corte")}
print(f"categoría: {next(iter(fcat_c.values())).shape[1]} feats | "
      f"ocurrencia: {next(iter(focc_c.values())).shape[1]} feats")


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
    d = d.join(fcat_c[cut], how="left")          # categoría/penetración (v11)
    d = d.join(focc_c[cut], how="left")          # ocurrencia v21 (con pair_req_*)
    d = d.join(fext_c[cut], how="left")          # stock + qty + mix (v21)
    return d


def fit(dtr: pd.DataFrame, params: dict) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(objective="l1", random_state=SEMILLA, verbosity=-1,
                          subsample_freq=1, **params)
    m.fit(dtr.drop(columns=["clase_ratio", "_prom"]), dtr["clase_ratio"])
    return m


def wape_pred(m, dev, real) -> float:
    ratio = pd.Series(m.predict(dev.drop(columns="_prom")), index=dev.index).clip(lower=0)
    pred = ratio * dev["_prom"]
    comun = pred.index.intersection(real.index)
    return float(np.abs(real[comun] - pred[comun]).sum() / real[comun].sum())


FOLDS = [(201808, 1.0), (201810, 1.0), (201812, 2.0)]
folds = []
for cut_eval, peso in FOLDS:
    n_train = (pd.Period(str(cut_eval), freq="M") - pd.Period("201801", freq="M")).n - 1
    cortes = [mes_menos(cut_eval, k) for k in range(2, 2 + n_train)]
    dtr = pd.concat([armar(c, True) for c in cortes])
    dev = armar(cut_eval, False)
    real = wide[mes_menos(cut_eval, -2)].dropna()
    folds.append((dtr, dev, real, peso))

d_final = pd.concat([armar(mes_menos(201910, k), True) for k in range(22)])
d_fut = armar(201912, False)
print(f"train final: {len(d_final)} filas, {d_final.shape[1]-2} features")


def evaluar(params: dict) -> float:
    t, w = 0.0, 0.0
    for dtr, dev, real, peso in folds:
        t += peso * wape_pred(fit(dtr, params), dev, real); w += peso
    return t / w


print(f"backtest multi-corte fijo: {evaluar(PARAMS_FIJOS):.4f} (v16 sin stk/qty 0.2363)")

# ---------- OPTUNA COMPLETO n=50 (pedido del usuario) ----------
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objetivo(trial):
    return evaluar(dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 900),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 16, 80),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 120),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0)))


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEMILLA))
study.optimize(objetivo, n_trials=50, show_progress_bar=False)
print(f"optuna n=50 best backtest: {study.best_value:.4f} | {study.best_params}")

# pata del ensamble = params FIJOS (ganador histórico; optuna no transfiere al LB)
tb_prom = (fb.ventas.filter(pl.col("periodo").is_between(201901, 201912))
           .group_by("product_id").agg(pl.col("tn").mean()))
apredecir = cargar_apredecir()

m = fit(d_final, PARAMS_FIJOS)
b = m.booster_
imp = pd.DataFrame({"gain": b.feature_importance("gain"), "splits": b.feature_importance("split")},
                   index=b.feature_name())
imp["gain_pct"] = (100 * imp["gain"] / imp["gain"].sum()).round(1)
imp = imp.sort_values("gain", ascending=False)
for pref, nom in [("stk_", "STOCK"), ("q_", "QTY"), (("mix_", "qtn"), "MIX"), ("occ_", "OCC")]:
    g = imp[imp.index.str.startswith(pref)]["gain_pct"].sum()
    print(f"gain {nom}: {g:.1f}%")
print(f"\n=== FI (top 15) ===\n{imp.head(15).round({'gain':0}).to_string()}")
(RUTA_EXP / "feature_importance_integracion.txt").write_text(imp.round(1).to_string())

ratio = pd.Series(m.predict(d_fut.drop(columns="_prom")), index=d_fut.index).clip(lower=0)
pred_lgbm = ratio * d_fut["_prom"]

# decorrelación: corr de la pata LGBM vs OLS (hipótesis: baja = más diversidad)
ols = pl.read_csv(RUTA_PROYECTO / "exp/LR01/linreg.csv").to_pandas().set_index("product_id")["tn"]
comun = pred_lgbm.index.intersection(ols.index)
print(f"\ncorr(LGBM v21, OLS): {np.corrcoef(pred_lgbm[comun], ols[comun])[0,1]:.4f} "
      f"(v16 sin stk/qty era 0.9931; menor = más diversidad)")

# ---------- ensamble + submits SIN y CON calibración 1.03 ----------
tb_ols = pl.DataFrame({"product_id": ols.index.to_list(), "ols": ols.to_list()})
tb_lgb = pl.DataFrame({"product_id": pred_lgbm.index.to_list(), "lgbm": pred_lgbm.to_list()})
base = (apredecir.join(tb_prom, on="product_id", how="left")
        .join(tb_ols, on="product_id", how="left").join(tb_lgb, on="product_id", how="left")
        .with_columns(pl.coalesce([pl.col("ols"), pl.col("tn")]).alias("ols"),
                      pl.coalesce([pl.col("lgbm"), pl.col("tn")]).alias("lgbm")))


def submit(nombre, factor):
    out = (base.with_columns(
        (factor * (0.70 * pl.col("ols") + 0.30 * pl.col("lgbm"))).clip(lower_bound=0).alias("tn"))
        .select("product_id", "tn").sort("product_id"))
    assert out.height == 780 and out["tn"].null_count() == 0
    f = RUTA_EXP / f"exp_v21_{nombre}.csv"; out.write_csv(f)
    subprocess.run([str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
                    "-c", COMPETENCIA, "-f", str(f), "-m", f"exp_v21 stock+qty+mix {nombre}"], check=True)
    print(f"submit OK -> {f.name}")


submit("ens_sin_calib", 1.00)     # sin 1.03 (efecto features limpio)
submit("ens_con_calib", 1.03)     # con 1.03 (comparable al ganador 0.226)
print("\nlisto")
