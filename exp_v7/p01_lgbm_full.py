"""exp_v7: LGBM ratio con TODOS los productos + features de pico estacional
+ demanda insatisfecha (cust_request_tn).

Diagnóstico previo (exp_v6): el LGBM de ratios subestima el rebote
dic->feb de los productos grandes, y solo comió 182 productos.
Tres dosis con ablación:
  D1: base C2 (lags normalizados+cv+flags) pero train con TODOS los
      productos (~1200) x cortes 201801..201910 -> ~15k filas (amplitud)
  D2: D1 + features anti-defecto:
      rebote_hist = tn_10/tn_12 (rebote mes_objetivo/mes_corte del año
      pasado), idx_objetivo, idx_actual, momentum = tn_0/tn_1
  D3: D2 + request: req_gap_0 y req_gap_3m = (pedido-entregado)/promedio
      (demanda insatisfecha reciente, señal de quiebre de stock)
Backtest honesto @201812->201902 informado antes de cada submit.
"""

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
COLS_LAGS = [f"tn_{k}" for k in range(13)]  # 13 lags: tn_12 = mes del corte, año previo

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
    """nivel 1=base, 2=+pico estacional, 3=+request."""
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
        # features de pico estacional (antes de normalizar los lags)
        d["rebote_hist"] = ((d["tn_10"] + EPS) / (d["tn_12"] + EPS)).clip(upper=10)
        d["momentum"] = ((d["tn_0"] + EPS) / (d["tn_1"] + EPS)).clip(upper=10)
        d["idx_objetivo"] = d["tn_10"] / (prom + EPS)
        d["idx_actual"] = d["tn_0"] / (prom + EPS)
    if nivel >= 3:
        req_cols = [mes_menos(cut, k) for k in range(3)]
        gap = (wide_req.loc[d.index, req_cols].values
               - wide.loc[d.index, req_cols].values)
        d["req_gap_0"] = gap[:, 0] / (prom + EPS)
        d["req_gap_3m"] = gap.mean(axis=1) / (prom + EPS)

    d[COLS_LAGS] = d[COLS_LAGS].div(prom, axis=0)
    d["cv"] = d[[f"tn_{k}" for k in range(12)]].std(axis=1)
    d["anio_corte"] = cut // 100
    d["mes_corte"] = cut % 100
    d["_prom"] = prom
    return d


def entrenar(cortes: list[int], nivel: int) -> lgb.LGBMRegressor:
    d = pd.concat([armar_corte(c, True, nivel) for c in cortes])
    m = lgb.LGBMRegressor(
        objective="l1", random_state=SEMILLA, verbosity=-1,
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        min_child_samples=30, colsample_bytree=0.8,
        subsample=0.8, subsample_freq=1,
    )
    m.fit(d.drop(columns=["clase_ratio", "_prom"]), d["clase_ratio"])
    return m, len(d)


def predecir(m, cut: int, nivel: int) -> pd.Series:
    d = armar_corte(cut, False, nivel)
    ratio = pd.Series(m.predict(d.drop(columns="_prom")), index=d.index).clip(lower=0)
    return ratio * d["_prom"]


# ---------- backtest honesto y submits ----------
cortes_bt = [mes_menos(201810, k) for k in range(10)]    # 201801..201810
cortes_fin = [mes_menos(201910, k) for k in range(22)]   # 201801..201910
real = wide[201902].dropna()

tb_prom = (
    ventas.filter(pl.col("periodo").is_between(201901, 201912))
    .group_by("product_id").agg(pl.col("tn").mean())
)
apredecir = cargar_apredecir()

for nivel, nombre in [(1, "D1_todos"), (2, "D2_pico"), (3, "D3_request")]:
    m_bt, n_bt = entrenar(cortes_bt, nivel)
    pred_bt = predecir(m_bt, 201812, nivel)
    comun = pred_bt.index.intersection(real.index)
    w = float(np.abs(real[comun] - pred_bt[comun]).sum() / real[comun].sum())
    print(f"{nombre}: backtest WAPE {w:.4f} (train {n_bt} filas) "
          f"[refs: C2-magicos 0.194, OLS 0.198]")

    m_fin, n_fin = entrenar(cortes_fin, nivel)
    pred = predecir(m_fin, 201912, nivel)
    tb_reg = pl.DataFrame({"product_id": pred.index.to_list(), "tn_pred": pred.to_list()})
    tb_final = (
        apredecir.join(tb_prom, on="product_id", how="left")
        .join(tb_reg, on="product_id", how="left")
        .with_columns(pl.coalesce([pl.col("tn_pred"), pl.col("tn")]).alias("tn"))
        .select("product_id", "tn").sort("product_id")
    )
    assert tb_final.height == 780 and tb_final["tn"].null_count() == 0
    archivo = RUTA_EXP / f"exp_v7_{nombre}.csv"
    tb_final.write_csv(archivo)
    subprocess.run(
        [str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
         "-c", COMPETENCIA, "-f", str(archivo), "-m", f"exp_v7 lgbm {nombre}"],
        check=True,
    )
    print(f"  submit OK -> {archivo.name} (train final {n_fin} filas)")
    if nivel == 3:
        imp = pd.Series(m_fin.feature_importances_,
                        index=pd.concat([armar_corte(201812, True, 3)])
                        .drop(columns=["clase_ratio", "_prom"]).columns)
        print("\nimportancia de features D3 (top 10):")
        print(imp.sort_values(ascending=False).head(10).to_string())
