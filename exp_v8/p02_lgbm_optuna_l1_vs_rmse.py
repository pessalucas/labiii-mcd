"""exp_v8 (bis): Optuna sobre el feature set completo + feature edad_serie,
comparando objetivo L1 (alineado a WAPE) vs RMSE (L2).

- edad_serie: cantidad de meses con datos de la serie al momento del corte.
- Dos estudios Optuna de 30 trials c/u con la vara multi-corte (3 folds
  sin fuga, dic pesa doble):
    L1:   objective='l1',  selección por WAPE
    RMSE: objective='l2',  selección por RMSE
- FI de ambos modelos finales -> feature_importance_optuna.txt
- Submit de ambos para comparar en el leaderboard.
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
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_sellin

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
RUTA_EXP = Path(__file__).resolve().parent
COMPETENCIA = "labo-iii-2026-ba"
SEMILLA = 102191
EPS = 1e-6
N_LAGS = 24
N_TRIALS = 30
LAGS12 = [f"tn_{k}" for k in range(12)]


def cargar_magicos() -> set[int]:
    nb = json.loads(
        (RUTA_PROYECTO / "src/Estadistica/z403_RegresionLineal_local.ipynb").read_text()
    )
    celda = next(
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "productos_magicos" in "".join(c["source"])
    )
    bloque = re.findall(r"productos_magicos = \[(.*?)\]", celda, flags=re.S)[-1]
    return set(int(x) for x in bloque.replace("\n", " ").split(","))


magicos = cargar_magicos()
sellin = cargar_sellin()
ventas = sellin.group_by("product_id", "periodo").agg(
    pl.col("tn").sum(),
    pl.col("customer_id").n_unique().alias("ncli"),
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
wide_ncli = panel("ncli")
primer_mes = wide.apply(lambda r: r.first_valid_index(), axis=1)


def mes_menos(p: int, k: int) -> int:
    return int((pd.Period(str(p), freq="M") - k).strftime("%Y%m"))


def slope(M: np.ndarray) -> np.ndarray:
    k = M.shape[1]
    t = np.arange(k) - (k - 1) / 2
    y = M - np.nanmean(M, axis=1, keepdims=True)
    s = (y * t).sum(axis=1) / (t ** 2).sum()
    s[np.isnan(M).any(axis=1)] = np.nan
    return s


def ols_magica(cut_train: int):
    cols = [mes_menos(cut_train, k) for k in range(12)]
    obj = mes_menos(cut_train, -2)
    if cols[-1] not in wide.columns or obj not in wide.columns:
        return None
    d = wide.loc[wide.index.isin(magicos), cols + [obj]].dropna()
    if len(d) < 30:
        return None
    X = np.column_stack([np.ones(len(d)), d[cols].values])
    coef, *_ = np.linalg.lstsq(X, d[obj].values, rcond=None)
    return coef


def ols_aplicar(coef, cut_apply: int) -> pd.Series:
    cols = [mes_menos(cut_apply, k) for k in range(12)]
    d = wide[cols].dropna()
    X = np.column_stack([np.ones(len(d)), d.values])
    return pd.Series(X @ coef, index=d.index).clip(lower=0)


def armar_corte(cut: int, con_clase: bool) -> pd.DataFrame:
    cols24 = [mes_menos(cut, k) for k in range(N_LAGS)]
    cols_exist = [c for c in cols24 if c in wide.columns]
    d = wide[cols_exist].copy()
    d.columns = [f"tn_{k}" for k in range(len(cols_exist))]
    for k in range(len(cols_exist), N_LAGS):
        d[f"tn_{k}"] = np.nan
    d = d.dropna(subset=LAGS12)
    prom = d[LAGS12].mean(axis=1)
    d = d[prom > EPS]
    prom = prom[prom > EPS]
    if con_clase:
        obj = mes_menos(cut, -2)
        if obj not in wide.columns:
            return pd.DataFrame()
        d["clase_ratio"] = wide.loc[d.index, obj] / prom
        d = d.dropna(subset=["clase_ratio"])
        prom = prom.loc[d.index]

    L = d[[f"tn_{k}" for k in range(N_LAGS)]].values

    d["r_0_1"] = ((L[:, 0] + EPS) / (L[:, 1] + EPS)).clip(0, 10)
    d["r_1_2"] = ((L[:, 1] + EPS) / (L[:, 2] + EPS)).clip(0, 10)
    d["r_02_34"] = ((L[:, 0] + L[:, 2] + EPS) / (L[:, 3] + L[:, 4] + EPS)).clip(0, 10)
    d["r_tri"] = ((L[:, :3].sum(1) + EPS) / (L[:, 3:6].sum(1) + EPS)).clip(0, 10)
    d["r_sem"] = ((L[:, :6].sum(1) + EPS) / (L[:, 6:12].sum(1) + EPS)).clip(0, 10)
    with np.errstate(invalid="ignore"):
        d["r_yoy_tri"] = ((L[:, :3].sum(1) + EPS) / (L[:, 12:15].sum(1) + EPS)).clip(0, 10)

    for k in (2, 3, 6, 12):
        d[f"s{k}"] = L[:, :k].sum(1) / (k * prom + EPS)

    Ln = L[:, :12] / (prom.values[:, None] + EPS)
    for k in (3, 6, 12):
        d[f"pend_{k}"] = slope(Ln[:, :k][:, ::-1])

    ccols = [mes_menos(cut, k) for k in range(12)]
    C = wide_ncli.loc[d.index, ccols].values
    cmean = np.nanmean(C, axis=1)
    d["ncli_0"] = C[:, 0]
    d["ncli_idx"] = C[:, 0] / (cmean + EPS)
    d["ncli_prom"] = cmean
    d["basket_idx"] = ((L[:, 0] + EPS) / (C[:, 0] + EPS)) / (
        (prom.values + EPS) / (cmean + EPS))
    Cn = C / (cmean[:, None] + EPS)
    for k in (3, 6, 12):
        d[f"ncli_pend_{k}"] = slope(Cn[:, :k][:, ::-1])

    # NUEVA: edad de la serie en meses al corte
    d["edad_serie"] = [
        (pd.Period(str(cut), freq="M") - pd.Period(str(int(primer_mes[pid])), freq="M")).n + 1
        for pid in d.index
    ]

    for nombre, lag_train in [("pred_ols_yoy", 12), ("pred_ols_rec", 2)]:
        coef = ols_magica(mes_menos(cut, lag_train))
        if coef is not None:
            p = ols_aplicar(coef, cut)
            d[nombre] = (p.reindex(d.index) / prom).values
        else:
            d[nombre] = np.nan

    d[[f"tn_{k}" for k in range(N_LAGS)]] = L / (prom.values[:, None] + EPS)
    d["cv"] = d[LAGS12].std(axis=1)
    d["anio_corte"] = cut // 100
    d["mes_corte"] = cut % 100
    d["_prom"] = prom
    return d


# ---------- folds precomputados (features no dependen de los params) ----------
FOLDS = [(201808, 1.0), (201810, 1.0), (201812, 2.0)]
folds_data = []
for cut_eval, peso in FOLDS:
    n_train = (pd.Period(str(cut_eval), freq="M") - pd.Period("201801", freq="M")).n - 1
    cortes = [mes_menos(cut_eval, k) for k in range(2, 2 + n_train)]
    dtr = pd.concat([armar_corte(c, True) for c in cortes])
    dev = armar_corte(cut_eval, False)
    real = wide[mes_menos(cut_eval, -2)].dropna()
    folds_data.append((dtr, dev, real, peso))
d_final = pd.concat([armar_corte(mes_menos(201910, k), True) for k in range(22)])
d_fut = armar_corte(201912, False)
print(f"folds listos | train final: {len(d_final)} filas, {d_final.shape[1]-2} features")


def evaluar(params, objective, metrica) -> float:
    total, pesos = 0.0, 0.0
    for dtr, dev, real, peso in folds_data:
        m = lgb.LGBMRegressor(objective=objective, random_state=SEMILLA,
                              verbosity=-1, subsample_freq=1, **params)
        m.fit(dtr.drop(columns=["clase_ratio", "_prom"]), dtr["clase_ratio"])
        ratio = pd.Series(m.predict(dev.drop(columns="_prom")), index=dev.index).clip(lower=0)
        pred = ratio * dev["_prom"]
        comun = pred.index.intersection(real.index)
        err = real[comun] - pred[comun]
        v = (float(np.sqrt((err ** 2).mean())) if metrica == "rmse"
             else float(err.abs().sum() / real[comun].sum()))
        total += peso * v
        pesos += peso
    return total / pesos


def espacio(trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 150, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 64),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
    )


resultados_fi = []
for etiqueta, objective, metrica in [("L1", "l1", "wape"), ("RMSE", "l2", "rmse")]:
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEMILLA))
    study.optimize(lambda t: evaluar(espacio(t), objective, metrica),
                   n_trials=N_TRIALS, show_progress_bar=False)
    print(f"\n[{etiqueta}] mejor {metrica} multi-corte: {study.best_value:.4f}")
    print(f"[{etiqueta}] params: {study.best_params}")

    m_fin = lgb.LGBMRegressor(objective=objective, random_state=SEMILLA,
                              verbosity=-1, subsample_freq=1, **study.best_params)
    m_fin.fit(d_final.drop(columns=["clase_ratio", "_prom"]), d_final["clase_ratio"])

    booster = m_fin.booster_
    imp = pd.DataFrame({
        "gain": booster.feature_importance(importance_type="gain"),
        "splits": booster.feature_importance(importance_type="split"),
    }, index=booster.feature_name())
    imp["gain_pct"] = (100 * imp["gain"] / imp["gain"].sum()).round(1)
    imp = imp.sort_values("gain", ascending=False)
    print(f"\n=== FI {etiqueta} (top 15) ===")
    print(imp.head(15).round({"gain": 0}).to_string())
    resultados_fi.append(f"\n{'='*60}\nMODELO {etiqueta} (objective={objective}, "
                         f"seleccion por {metrica})\nparams: {study.best_params}\n"
                         f"{'='*60}\n" + imp.round(1).to_string())

    ratio = pd.Series(m_fin.predict(d_fut.drop(columns="_prom")),
                      index=d_fut.index).clip(lower=0)
    pred = ratio * d_fut["_prom"]
    tb_prom = (ventas.filter(pl.col("periodo").is_between(201901, 201912))
               .group_by("product_id").agg(pl.col("tn").mean()))
    tb_reg = pl.DataFrame({"product_id": pred.index.to_list(), "tn_pred": pred.to_list()})
    tb_final = (
        cargar_apredecir()
        .join(tb_prom, on="product_id", how="left")
        .join(tb_reg, on="product_id", how="left")
        .with_columns(pl.coalesce([pl.col("tn_pred"), pl.col("tn")]).alias("tn"))
        .select("product_id", "tn").sort("product_id")
    )
    assert tb_final.height == 780 and tb_final["tn"].null_count() == 0
    archivo = RUTA_EXP / f"exp_v8_optuna_{etiqueta}.csv"
    tb_final.write_csv(archivo)
    subprocess.run(
        [str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
         "-c", COMPETENCIA, "-f", str(archivo),
         "-m", f"exp_v8 optuna30 {etiqueta} +edad_serie"],
        check=True,
    )
    print(f"[{etiqueta}] submit OK -> {archivo.name}")

(RUTA_EXP / "feature_importance_optuna.txt").write_text(
    "exp_v8 - FI de los modelos Optuna L1 vs RMSE (+edad_serie)\n"
    + "\n".join(resultados_fi))
print("\nFI guardado en feature_importance_optuna.txt")
