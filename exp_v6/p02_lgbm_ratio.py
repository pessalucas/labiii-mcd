"""exp_v6 variante C: LGBM prediciendo RATIOS en vez de toneladas.

Fix al defecto estructural detectado en A/B: los árboles no extrapolan
escala (la predicción queda capada al rango de toneladas visto en train).
  - objetivo: clase / promedio_12m del producto (escala propia)
  - features: lags también normalizados por el promedio + cv + flags
  - sample_weight = promedio (la pérdida L1 en ratios, ponderada por
    toneladas, aproxima el WAPE real)
  - predicción final = ratio_predicho * promedio_12m
Mismo train (mágicos, cortes 201712..201910) y validación honesta @201812.
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
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_ventas

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
COMPETENCIA = "labo-iii-2026-ba"
SEMILLA = 102191
EPS = 1e-6
COLS_LAGS = [f"tn_{k}" for k in range(12)]


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


magicos = set(cargar_magicos())
ventas = cargar_ventas()

periodos = [int(p.strftime("%Y%m")) for p in pd.period_range("2017-01", "2019-12", freq="M")]
wide = (
    ventas.to_pandas()
    .pivot(index="product_id", columns="periodo", values="tn")
    .reindex(columns=periodos)
)
primer_mes = wide.apply(lambda r: r.first_valid_index(), axis=1)
for pid in wide.index:
    wide.loc[pid, wide.columns >= primer_mes[pid]] = (
        wide.loc[pid, wide.columns >= primer_mes[pid]].fillna(0.0)
    )


def mes_menos(periodo: int, k: int) -> int:
    return int((pd.Period(str(periodo), freq="M") - k).strftime("%Y%m"))


def armar_corte(cut: int, con_clase: bool) -> pd.DataFrame:
    cols = [mes_menos(cut, k) for k in range(12)]
    d = wide[cols].copy()
    d.columns = COLS_LAGS
    d = d.dropna()
    prom = d[COLS_LAGS].mean(axis=1)
    d = d[prom > EPS]  # sin promedio no hay ratio
    prom = prom[prom > EPS]
    if con_clase:
        objetivo = mes_menos(cut, -2)
        if objetivo not in wide.columns:
            return pd.DataFrame()
        d["clase_ratio"] = wide.loc[d.index, objetivo] / prom
        d = d.dropna(subset=["clase_ratio"])
        prom = prom.loc[d.index]
    # lags normalizados: el arbol ve FORMA, no escala
    d[COLS_LAGS] = d[COLS_LAGS].div(prom, axis=0)
    d["cv"] = d[COLS_LAGS].std(axis=1)
    d["anio_corte"] = cut // 100
    d["mes_corte"] = cut % 100
    d["_prom"] = prom
    return d


def entrenar(cortes: list[int]) -> lgb.LGBMRegressor:
    d = pd.concat([armar_corte(c, True) for c in cortes])
    d = d[d.index.isin(magicos)]
    m = lgb.LGBMRegressor(
        objective="l1", random_state=SEMILLA, verbosity=-1,
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        min_child_samples=30, colsample_bytree=0.8,
        subsample=0.8, subsample_freq=1,
    )
    # NOTA (post-mortem): sample_weight=_prom con pesos de 4 órdenes de
    # magnitud hace DIVERGER el objetivo l1 de LightGBM (ratios predichos
    # de miles, WAPE 263 / Kaggle 0.760). Sin pesos: backtest 0.1936.
    m.fit(d.drop(columns=["clase_ratio", "_prom"]), d["clase_ratio"])
    return m


def predecir(m, cut: int) -> pd.Series:
    d = armar_corte(cut, con_clase=False)
    ratio = pd.Series(m.predict(d.drop(columns="_prom")), index=d.index).clip(lower=0)
    return ratio * d["_prom"]


# backtest honesto
cortes_bt = [mes_menos(201810, k) for k in range(11)]
m_bt = entrenar(cortes_bt)
pred_bt = predecir(m_bt, 201812)
real = wide[201902].dropna()
comun = pred_bt.index.intersection(real.index)
wape = float(np.abs(real[comun] - pred_bt[comun]).sum() / real[comun].sum())
print(f"backtest honesto @201812->201902: WAPE {wape:.4f}")
print("(referencias: OLS mágica 0.198, naif 0.252, lgbm-B tons 0.324)")

# final
cortes_fin = [mes_menos(201910, k) for k in range(23)]
m_fin = entrenar(cortes_fin)
pred = predecir(m_fin, 201912)

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
archivo = RUTA_EXP / "exp_v6_C2_ratio_fix.csv"
tb_final.write_csv(archivo)
subprocess.run(
    [str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
     "-c", COMPETENCIA, "-f", str(archivo), "-m", "exp_v6 lgbm C2 ratio normalizado FIX sin pesos"],
    check=True,
)
print("submit OK -> exp_v6_C2_ratio_fix.csv")
