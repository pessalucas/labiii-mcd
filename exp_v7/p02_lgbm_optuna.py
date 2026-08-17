"""exp_v7 (bis): LGBM de ratios optimizado con Optuna (30 trials).

Mismo pipeline que p01 pero con búsqueda de hiperparámetros y del nivel
de features (1=base, 2=+pico estacional, 3=+request).

Objetivo de la optimización: WAPE multi-corte para robustecer la brújula
local (lección de v4-v7: un solo corte de validación rankea distinto que
el leaderboard). Tres folds sin fuga (train cuts <= eval-2):
  eval @201808 -> 201810   (peso 1)
  eval @201810 -> 201812   (peso 1)
  eval @201812 -> 201902   (peso 2, es el examen dic->feb)
El mejor trial se reentrena con todos los cortes y se submitea.
"""

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
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_sellin

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
RUTA_EXP = Path(__file__).resolve().parent
COMPETENCIA = "labo-iii-2026-ba"
SEMILLA = 102191
EPS = 1e-6
N_TRIALS = 30
COLS_LAGS = [f"tn_{k}" for k in range(13)]

sellin = cargar_sellin()
ventas = sellin.group_by("product_id", "periodo").agg(
    pl.col("tn").sum(),
    pl.col("cust_request_tn").sum().alias("req"),
)
periodos = [int(p.strftime("%Y%m")) for p in pd.period_range("2017-01", "2019-12", freq="M")]


def panel(col: str) -> pd.DataFrame:
    w = (ventas.to_pandas().pivot(index="product_id", columns="periodo", values=col)
         .reindex(columns=periodos))
    pm = w.apply(lambda r: r.first_valid_index(), axis=1)
    for pid in w.index:
        w.loc[pid, w.columns >= pm[pid]] = w.loc[pid, w.columns >= pm[pid]].fillna(0.0)
    return w


wide = panel("tn")
wide_req = panel("req")


def mes_menos(p: int, k: int) -> int:
    return int((pd.Period(str(p), freq="M") - k).strftime("%Y%m"))


def armar_corte(cut: int, con_clase: bool, nivel: int) -> pd.DataFrame:
    cols = [mes_menos(cut, k) for k in range(13)]
    d = wide[cols].copy()
    d.columns = COLS_LAGS
    d = d.dropna()
    prom = d[[f"tn_{k}" for k in range(12)]].mean(axis=1)
    d = d[prom > EPS]
    prom = prom[prom > EPS]
    if con_clase:
        obj = mes_menos(cut, -2)
        if obj not in wide.columns:
            return pd.DataFrame()
        d["clase_ratio"] = wide.loc[d.index, obj] / prom
        d = d.dropna(subset=["clase_ratio"])
        prom = prom.loc[d.index]
    if nivel >= 2:
        d["rebote_hist"] = ((d["tn_10"] + EPS) / (d["tn_12"] + EPS)).clip(upper=10)
        d["momentum"] = ((d["tn_0"] + EPS) / (d["tn_1"] + EPS)).clip(upper=10)
        d["idx_objetivo"] = d["tn_10"] / (prom + EPS)
        d["idx_actual"] = d["tn_0"] / (prom + EPS)
    if nivel >= 3:
        req_cols = [mes_menos(cut, k) for k in range(3)]
        gap = (wide_req.loc[d.index, req_cols].values - wide.loc[d.index, req_cols].values)
        d["req_gap_0"] = gap[:, 0] / (prom + EPS)
        d["req_gap_3m"] = gap.mean(axis=1) / (prom + EPS)
    d[COLS_LAGS] = d[COLS_LAGS].div(prom, axis=0)
    d["cv"] = d[[f"tn_{k}" for k in range(12)]].std(axis=1)
    d["anio_corte"] = cut // 100
    d["mes_corte"] = cut % 100
    d["_prom"] = prom
    return d


def fit_lgbm(cortes, nivel, params) -> lgb.LGBMRegressor:
    d = pd.concat([armar_corte(c, True, nivel) for c in cortes])
    m = lgb.LGBMRegressor(objective="l1", random_state=SEMILLA, verbosity=-1, **params)
    m.fit(d.drop(columns=["clase_ratio", "_prom"]), d["clase_ratio"])
    return m


def wape_en(m, cut_eval: int, nivel: int) -> float:
    d = armar_corte(cut_eval, False, nivel)
    ratio = pd.Series(m.predict(d.drop(columns="_prom")), index=d.index).clip(lower=0)
    pred = ratio * d["_prom"]
    real = wide[mes_menos(cut_eval, -2)].dropna()
    comun = pred.index.intersection(real.index)
    return float(np.abs(real[comun] - pred[comun]).sum() / real[comun].sum())


FOLDS = [  # (cut_eval, peso); train cuts: 201801 .. cut_eval-2
    (201808, 1.0),
    (201810, 1.0),
    (201812, 2.0),  # el examen dic->feb pesa doble
]


def objetivo(trial: optuna.Trial) -> float:
    nivel = trial.suggest_categorical("nivel", [1, 2, 3])
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 150, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 64),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
    )
    total, pesos = 0.0, 0.0
    for cut_eval, peso in FOLDS:
        n_train = (pd.Period(str(cut_eval), freq="M") - pd.Period("201801", freq="M")).n - 1
        cortes = [mes_menos(cut_eval, k) for k in range(2, 2 + n_train)]
        m = fit_lgbm(cortes, nivel, params)
        total += peso * wape_en(m, cut_eval, nivel)
        pesos += peso
    return total / pesos


study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEMILLA),
)
study.optimize(objetivo, n_trials=N_TRIALS, show_progress_bar=False)

print(f"mejor WAPE multi-corte: {study.best_value:.4f}")
print("mejores parámetros:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# referencia: los params fijos de p01 en la misma vara multi-corte
params_p01 = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
                  min_child_samples=30, colsample_bytree=0.8, subsample=0.8,
                  subsample_freq=1)
total, pesos = 0.0, 0.0
for cut_eval, peso in FOLDS:
    n_train = (pd.Period(str(cut_eval), freq="M") - pd.Period("201801", freq="M")).n - 1
    cortes = [mes_menos(cut_eval, k) for k in range(2, 2 + n_train)]
    m = fit_lgbm(cortes, 2, params_p01)
    total += peso * wape_en(m, cut_eval, 2)
    pesos += peso
print(f"\nreferencia p01 (params fijos, nivel 2) en la misma vara: {total/pesos:.4f}")

# ---------- final: mejor trial con todos los cortes ----------
best = dict(study.best_params)
nivel_best = best.pop("nivel")
best["subsample_freq"] = 1
cortes_fin = [mes_menos(201910, k) for k in range(22)]
m_fin = fit_lgbm(cortes_fin, nivel_best, best)

# feature importance del modelo final: gain (cuánto error reduce cada
# feature en total) y split (cuántas veces se usa para partir)
booster = m_fin.booster_
imp = pd.DataFrame({
    "gain": booster.feature_importance(importance_type="gain"),
    "splits": booster.feature_importance(importance_type="split"),
}, index=booster.feature_name())
imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
print("\n=== feature importance (modelo final, ordenado por gain) ===")
print(imp.sort_values("gain", ascending=False)
      .round({"gain": 0, "gain_pct": 1}).to_string())

d_fut = armar_corte(201912, False, nivel_best)
ratio = pd.Series(m_fin.predict(d_fut.drop(columns="_prom")), index=d_fut.index).clip(lower=0)
pred = ratio * d_fut["_prom"]

tb_prom = (
    ventas.filter(pl.col("periodo").is_between(201901, 201912))
    .group_by("product_id").agg(pl.col("tn").mean())
)
tb_reg = pl.DataFrame({"product_id": pred.index.to_list(), "tn_pred": pred.to_list()})
tb_final = (
    cargar_apredecir()
    .join(tb_prom, on="product_id", how="left")
    .join(tb_reg, on="product_id", how="left")
    .with_columns(pl.coalesce([pl.col("tn_pred"), pl.col("tn")]).alias("tn"))
    .select("product_id", "tn").sort("product_id")
)
assert tb_final.height == 780 and tb_final["tn"].null_count() == 0
archivo = RUTA_EXP / "exp_v7_optuna30.csv"
tb_final.write_csv(archivo)
subprocess.run(
    [str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
     "-c", COMPETENCIA, "-f", str(archivo),
     "-m", f"exp_v7 lgbm optuna30 nivel={nivel_best} wape_local={study.best_value:.4f}"],
    check=True,
)
print(f"\nsubmit OK -> {archivo.name} (nivel features: {nivel_best})")
