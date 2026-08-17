"""exp_v9: ensamble LGBM (sin stacking OLS, con DIFFs) + OLS mágica.

Spec del usuario:
  - LGBM con el feature set de v8 SIN pred_ols_* (tecnologías independientes)
  - + DIFFs de los meses recientes: tn_k - tn_{k+1} para k=0..5, más
    diff_0_2 y la interanual diff_0_12 (sobre lags normalizados)
  - en paralelo, la OLS mágica de z403 entrenada tal cual (réplica exacta:
    lags por fila, 182 mágicos @201812, fallback promedio 2019)
  - predicción final = promedio simple de ambas tecnologías
  - SIN Optuna: params fijos de v8-p01 (config superadora, 0.250)
Submits: LGBM solo y ensamble (atribución).
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
import polars.selectors as cs
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_sellin, cargar_ventas

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
COMPETENCIA = "labo-iii-2026-ba"
SEMILLA = 102191
EPS = 1e-6
N_LAGS = 24
LAGS12 = [f"tn_{k}" for k in range(12)]
PARAMS = dict(objective="l1", random_state=SEMILLA, verbosity=-1,
              n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, colsample_bytree=0.8,
              subsample=0.8, subsample_freq=1)


def cargar_magicos() -> list[int]:
    nb = json.loads(
        (RUTA_PROYECTO / "src/Estadistica/z403_RegresionLineal_local.ipynb").read_text()
    )
    celda = next(
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "productos_magicos" in "".join(c["source"])
    )
    bloque = re.findall(r"productos_magicos = \[(.*?)\]", celda, flags=re.S)[-1]
    return [int(x) for x in bloque.replace("\n", " ").split(",")]


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
    Ln = L / (prom.values[:, None] + EPS)  # normalizados (NaN se preservan)

    # ratios recientes
    d["r_0_1"] = ((L[:, 0] + EPS) / (L[:, 1] + EPS)).clip(0, 10)
    d["r_1_2"] = ((L[:, 1] + EPS) / (L[:, 2] + EPS)).clip(0, 10)
    d["r_02_34"] = ((L[:, 0] + L[:, 2] + EPS) / (L[:, 3] + L[:, 4] + EPS)).clip(0, 10)
    d["r_tri"] = ((L[:, :3].sum(1) + EPS) / (L[:, 3:6].sum(1) + EPS)).clip(0, 10)
    d["r_sem"] = ((L[:, :6].sum(1) + EPS) / (L[:, 6:12].sum(1) + EPS)).clip(0, 10)
    with np.errstate(invalid="ignore"):
        d["r_yoy_tri"] = ((L[:, :3].sum(1) + EPS) / (L[:, 12:15].sum(1) + EPS)).clip(0, 10)

    # DIFFs (spec v9): consecutivos recientes + salto 0-2 + interanual
    for k in range(6):
        d[f"diff_{k}_{k+1}"] = Ln[:, k] - Ln[:, k + 1]
    d["diff_0_2"] = Ln[:, 0] - Ln[:, 2]
    d["diff_0_12"] = Ln[:, 0] - Ln[:, 12]  # NaN si no hay 2do año

    # sumas (= medias móviles normalizadas)
    for k in (2, 3, 6, 12):
        d[f"s{k}"] = L[:, :k].sum(1) / (k * prom + EPS)

    # rolling stats sobre lags normalizados: desvío, min, max, mediana
    for k in (3, 6, 12):
        V = Ln[:, :k]
        d[f"roll_std_{k}"] = np.nanstd(V, axis=1)
        d[f"roll_min_{k}"] = np.nanmin(V, axis=1)
        d[f"roll_max_{k}"] = np.nanmax(V, axis=1)
        d[f"roll_med_{k}"] = np.nanmedian(V, axis=1)

    # MEDIAS MÓVILES desplazadas en el tiempo y sus cruces
    # (s3/s6/s12 son las MA ancladas al corte; estas son las de antes)
    ma3_act = L[:, 0:3].mean(1) / (prom + EPS)
    for off in (3, 6, 9):  # MA de 3 meses centrada off meses atrás
        d[f"ma3_lag{off}"] = L[:, off:off + 3].mean(1) / (prom + EPS)
    d["ma6_lag6"] = L[:, 6:12].mean(1) / (prom + EPS)
    ma12 = L[:, :12].mean(1) / (prom + EPS)
    d["cruce_ma3_ma12"] = (ma3_act + EPS) / (ma12 + EPS)   # corta vs larga
    d["acel_ma3"] = ma3_act - d["ma3_lag3"]                # aceleración de la MA
    d["cruce_ma6"] = (L[:, 0:6].mean(1) / (prom + EPS) + EPS) / (d["ma6_lag6"] + EPS)

    # DIFFs vs min y max de 6/12/24 meses (posicion actual vs extremos)
    for k in (6, 12, 24):
        V = Ln[:, :k]
        d[f"dmin_{k}"] = Ln[:, 0] - np.nanmin(V, axis=1)
        d[f"dmax_{k}"] = Ln[:, 0] - np.nanmax(V, axis=1)
        d[f"dprom_{k}"] = Ln[:, 0] - np.nanmean(V, axis=1)

    # tendencias
    for k in (3, 6, 12):
        d[f"pend_{k}"] = slope(Ln[:, :12][:, :k][:, ::-1])

    # clientela
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

    d["edad_serie"] = [
        (pd.Period(str(cut), freq="M") - pd.Period(str(int(primer_mes[pid])), freq="M")).n + 1
        for pid in d.index
    ]

    d[[f"tn_{k}" for k in range(N_LAGS)]] = Ln
    d["cv"] = d[LAGS12].std(axis=1)
    d["anio_corte"] = cut // 100
    d["mes_corte"] = cut % 100
    d["_prom"] = prom
    return d


def entrenar(cortes: list[int]):
    d = pd.concat([armar_corte(c, True) for c in cortes])
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(d.drop(columns=["clase_ratio", "_prom"]), d["clase_ratio"])
    return m, len(d)


def predecir(m, cut: int) -> pd.Series:
    d = armar_corte(cut, False)
    ratio = pd.Series(m.predict(d.drop(columns="_prom")), index=d.index).clip(lower=0)
    return ratio * d["_prom"]


# ---------- OLS mágica: réplica EXACTA de z403 (lags por fila) ----------
tb_ventas = cargar_ventas()
lags_z = [-2, *range(0, 12)]
tb_lags = (
    tb_ventas.sort(["product_id", "periodo"])
    .with_columns(
        [pl.col("tn").shift(l).over("product_id").alias(f"tn_{l}") for l in lags_z]
    )
    .rename({"tn_-2": "clase"})
)
dtrain_ols = tb_lags.filter(
    (pl.col("periodo") == 201812) & pl.col("product_id").is_in(magicos)
).drop_nulls(["clase"] + LAGS12)
X_ols = sm.add_constant(dtrain_ols.select(cs.starts_with("tn_")).to_pandas())
modelo_ols = sm.OLS(dtrain_ols["clase"].to_pandas(), X_ols).fit()

dfut_ols = tb_lags.filter((pl.col("periodo") == 201912) & pl.col("tn_11").is_not_null())
Xf_ols = sm.add_constant(dfut_ols.select(cs.starts_with("tn_")).to_pandas())
pred_ols = pl.DataFrame({
    "product_id": dfut_ols["product_id"],
    "tn_ols": modelo_ols.predict(Xf_ols).to_numpy(),
})

# ---------- backtest honesto del LGBM y del ensamble ----------
cortes_bt = [mes_menos(201810, k) for k in range(10)]
m_bt, _ = entrenar(cortes_bt)
pred_bt = predecir(m_bt, 201812)
real = wide[201902].dropna()

# OLS del backtest: entrenada @201712 (réplica del examen anterior)
dtr_bt = tb_lags.filter(
    (pl.col("periodo") == 201712) & pl.col("product_id").is_in(magicos)
).drop_nulls(["clase"] + LAGS12)
m_ols_bt = sm.OLS(
    dtr_bt["clase"].to_pandas(),
    sm.add_constant(dtr_bt.select(cs.starts_with("tn_")).to_pandas()),
).fit()
dfut_bt = tb_lags.filter((pl.col("periodo") == 201812) & pl.col("tn_11").is_not_null())
p_ols_bt = pd.Series(
    m_ols_bt.predict(sm.add_constant(dfut_bt.select(cs.starts_with("tn_")).to_pandas())).to_numpy(),
    index=dfut_bt["product_id"].to_list(),
).clip(lower=0)

for nombre, p in [
    ("lgbm", pred_bt),
    ("ols", p_ols_bt),
    ("ensamble", ((pred_bt + p_ols_bt.reindex(pred_bt.index)) / 2).fillna(pred_bt)),
]:
    comun = p.index.intersection(real.index)
    w = float(np.abs(real[comun] - p[comun]).sum() / real[comun].sum())
    print(f"backtest @201812->201902 {nombre:>8}: WAPE {w:.4f}")

# ---------- final: LGBM, OLS y ensamble ----------
cortes_fin = [mes_menos(201910, k) for k in range(22)]
m_fin, n_fin = entrenar(cortes_fin)
pred_lgbm = predecir(m_fin, 201912)
print(f"\ntrain final LGBM: {n_fin} filas, "
      f"{len(m_fin.booster_.feature_name())} features")

booster = m_fin.booster_
imp = pd.DataFrame({
    "gain": booster.feature_importance(importance_type="gain"),
    "splits": booster.feature_importance(importance_type="split"),
}, index=booster.feature_name())
imp["gain_pct"] = (100 * imp["gain"] / imp["gain"].sum()).round(1)
imp = imp.sort_values("gain", ascending=False)
print("\n=== FI LGBM v9 (top 15) ===")
print(imp.head(15).round({"gain": 0}).to_string())
(RUTA_EXP / "feature_importance.txt").write_text(
    "exp_v9 LGBM (sin stacking, con diffs) - feature importance\n\n"
    + imp.round(1).to_string())

tb_prom = (
    ventas.filter(pl.col("periodo").is_between(201901, 201912))
    .group_by("product_id").agg(pl.col("tn").mean())
)
apredecir = cargar_apredecir()
tb_lgbm = pl.DataFrame({
    "product_id": pred_lgbm.index.to_list(),
    "tn_lgbm": pred_lgbm.to_list(),
})

base = (
    apredecir.join(tb_prom, on="product_id", how="left")
    .join(pred_ols, on="product_id", how="left")
    .join(tb_lgbm, on="product_id", how="left")
)


def submitear(nombre: str, tabla: pl.DataFrame) -> None:
    assert tabla.height == 780 and tabla["tn"].null_count() == 0
    archivo = RUTA_EXP / f"exp_v9_{nombre}.csv"
    tabla.write_csv(archivo)
    subprocess.run(
        [str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
         "-c", COMPETENCIA, "-f", str(archivo), "-m", f"exp_v9 {nombre}"],
        check=True,
    )
    print(f"submit OK -> {archivo.name}")


# LGBM solo (fallback promedio)
submitear("lgbm_final", base.with_columns(
    pl.coalesce([pl.col("tn_lgbm"), pl.col("tn")]).alias("tn")
).select("product_id", "tn").sort("product_id"))

# ensamble: promedio de las tecnologías donde ambas existen,
# la que exista si falta una, promedio 2019 si no hay ninguna
submitear("ensamble_final_5050", base.with_columns(
    pl.coalesce([
        (pl.col("tn_ols") + pl.col("tn_lgbm")) / 2,
        pl.col("tn_ols"),
        pl.col("tn_lgbm"),
        pl.col("tn"),
    ]).alias("tn")
).select("product_id", "tn").sort("product_id"))

# ensamble ponderado 80/20 OLS/LGBM (el 50/50 diluía a la pata fuerte)
submitear("ensamble_final_w80", base.with_columns(
    pl.coalesce([
        0.8 * pl.col("tn_ols") + 0.2 * pl.col("tn_lgbm"),
        pl.col("tn_ols"),
        pl.col("tn_lgbm"),
        pl.col("tn"),
    ]).clip(lower_bound=0).alias("tn")
).select("product_id", "tn").sort("product_id"))
