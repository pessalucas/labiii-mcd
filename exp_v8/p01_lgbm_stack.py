"""exp_v8: LGBM con ventana 24m, ratios recientes, clientela, tendencias
y stacking de la OLS mágica.

Spec del usuario:
  - 24 lags (NaN si el producto no existía: LGBM los maneja nativo)
  - ratios enfocados en tn_0..tn_5 (donde está el gain según exp_v7)
  - sumas acumuladas hasta 12 meses
  - métricas de clientes (cuántos compran, compra promedio, tendencias)
    -> la pista de z601: la composición de la venta anticipa el futuro
  - tendencias lineales de 12/6/3 puntos, si existen
  - la OLS como input (stacking):
      pred_ols_yoy: OLS mágica entrenada en el corte C-12 (para el corte
        final 201912 es EXACTAMENTE el modelo 0.231) -> sin fuga
      pred_ols_rec: OLS mágica entrenada en C-2 (último examen cerrado)
Target ratio (tn(t+2)/prom), todos los productos, cortes 201801..201910.
Params LGBM fijos de exp_v7-p01 (el tuning local demostró no transferir).
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
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_sellin

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
COMPETENCIA = "labo-iii-2026-ba"
SEMILLA = 102191
EPS = 1e-6
N_LAGS = 24
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


def mes_menos(p: int, k: int) -> int:
    return int((pd.Period(str(p), freq="M") - k).strftime("%Y%m"))


def slope(M: np.ndarray) -> np.ndarray:
    """Pendiente lineal por fila. Columnas = tiempo ascendente. NaN si falta algo."""
    k = M.shape[1]
    t = np.arange(k) - (k - 1) / 2
    y = M - np.nanmean(M, axis=1, keepdims=True)
    s = (y * t).sum(axis=1) / (t ** 2).sum()
    s[np.isnan(M).any(axis=1)] = np.nan
    return s


def ols_magica(cut_train: int) -> np.ndarray | None:
    """Coeficientes de la OLS z403 entrenada en cut_train (target +2)."""
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


def ols_aplicar(coef: np.ndarray, cut_apply: int) -> pd.Series:
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
        d[f"tn_{k}"] = np.nan  # meses fuera del rango de datos
    d = d.dropna(subset=LAGS12)  # exijo el año movil completo
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

    L = d[[f"tn_{k}" for k in range(N_LAGS)]].values  # crudos, con NaN

    # ---- ratios recientes (tn_0..tn_5, spec usuario), clip [0,10] ----
    d["r_0_1"] = ((L[:, 0] + EPS) / (L[:, 1] + EPS)).clip(0, 10)
    d["r_1_2"] = ((L[:, 1] + EPS) / (L[:, 2] + EPS)).clip(0, 10)
    d["r_02_34"] = ((L[:, 0] + L[:, 2] + EPS) / (L[:, 3] + L[:, 4] + EPS)).clip(0, 10)
    d["r_tri"] = ((L[:, :3].sum(1) + EPS) / (L[:, 3:6].sum(1) + EPS)).clip(0, 10)
    d["r_sem"] = ((L[:, :6].sum(1) + EPS) / (L[:, 6:12].sum(1) + EPS)).clip(0, 10)
    with np.errstate(invalid="ignore"):
        d["r_yoy_tri"] = ((L[:, :3].sum(1) + EPS) / (L[:, 12:15].sum(1) + EPS)).clip(0, 10)

    # ---- sumas acumuladas (normalizadas por prom) ----
    for k in (2, 3, 6, 12):
        d[f"s{k}"] = L[:, :k].sum(1) / (k * prom + EPS)

    # ---- tendencias 12/6/3 (sobre lags normalizados, tiempo ascendente) ----
    Ln = L[:, :12] / (prom.values[:, None] + EPS)
    for k in (3, 6, 12):
        d[f"pend_{k}"] = slope(Ln[:, :k][:, ::-1])

    # ---- clientes (pista z601) ----
    ccols = [mes_menos(cut, k) for k in range(12)]
    C = wide_ncli.loc[d.index, ccols].values
    cmean = np.nanmean(C, axis=1)
    d["ncli_0"] = C[:, 0]
    d["ncli_idx"] = C[:, 0] / (cmean + EPS)
    d["ncli_prom"] = cmean
    # compra promedio del mes del corte vs la historica
    d["basket_idx"] = ((L[:, 0] + EPS) / (C[:, 0] + EPS)) / (
        (prom.values + EPS) / (cmean + EPS))
    Cn = C / (cmean[:, None] + EPS)
    for k in (3, 6, 12):
        d[f"ncli_pend_{k}"] = slope(Cn[:, :k][:, ::-1])

    # ---- stacking OLS (sin fuga: entrenadas solo con targets <= cut) ----
    for nombre, lag_train in [("pred_ols_yoy", 12), ("pred_ols_rec", 2)]:
        coef = ols_magica(mes_menos(cut, lag_train))
        if coef is not None:
            p = ols_aplicar(coef, cut)
            d[nombre] = (p.reindex(d.index) / prom).values
        else:
            d[nombre] = np.nan

    # ---- lags normalizados + flags ----
    d[[f"tn_{k}" for k in range(N_LAGS)]] = L / (prom.values[:, None] + EPS)
    d["cv"] = d[LAGS12].std(axis=1)
    d["anio_corte"] = cut // 100
    d["mes_corte"] = cut % 100
    d["_prom"] = prom
    return d


PARAMS = dict(objective="l1", random_state=SEMILLA, verbosity=-1,
              n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, colsample_bytree=0.8,
              subsample=0.8, subsample_freq=1)


def entrenar(cortes: list[int]) -> tuple[lgb.LGBMRegressor, int]:
    d = pd.concat([armar_corte(c, True) for c in cortes])
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(d.drop(columns=["clase_ratio", "_prom"]), d["clase_ratio"])
    return m, len(d)


def predecir(m, cut: int) -> pd.Series:
    d = armar_corte(cut, False)
    ratio = pd.Series(m.predict(d.drop(columns="_prom")), index=d.index).clip(lower=0)
    return ratio * d["_prom"]


# ---------- backtest honesto @201812 -> 201902 ----------
cortes_bt = [mes_menos(201810, k) for k in range(10)]
m_bt, n_bt = entrenar(cortes_bt)
pred_bt = predecir(m_bt, 201812)
real = wide[201902].dropna()
comun = pred_bt.index.intersection(real.index)
w = float(np.abs(real[comun] - pred_bt[comun]).sum() / real[comun].sum())
print(f"backtest @201812->201902: WAPE {w:.4f} (train {n_bt} filas)")
print("(refs misma vara: D1 0.249, OLS 0.198)")

# ---------- final ----------
cortes_fin = [mes_menos(201910, k) for k in range(22)]
m_fin, n_fin = entrenar(cortes_fin)
pred = predecir(m_fin, 201912)

booster = m_fin.booster_
imp = pd.DataFrame({
    "gain": booster.feature_importance(importance_type="gain"),
    "splits": booster.feature_importance(importance_type="split"),
}, index=booster.feature_name())
imp["gain_pct"] = (100 * imp["gain"] / imp["gain"].sum()).round(1)
imp = imp.sort_values("gain", ascending=False)
print("\n=== feature importance (top 20 por gain) ===")
print(imp.head(20).round({"gain": 0}).to_string())
(RUTA_EXP / "feature_importance.txt").write_text(
    "exp_v8 LGBM stack - feature importance (gain/splits)\n\n" + imp.round(1).to_string()
)

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
archivo = RUTA_EXP / "exp_v8_lgbm_stack.csv"
tb_final.write_csv(archivo)
subprocess.run(
    [str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
     "-c", COMPETENCIA, "-f", str(archivo),
     "-m", f"exp_v8 lgbm 24lags+ratios+clientes+tendencias+stackOLS (train {n_fin})"],
    check=True,
)
print(f"\nsubmit OK -> {archivo.name} (train final {n_fin} filas)")
