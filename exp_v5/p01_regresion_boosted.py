"""exp_v5: regresión mágica "boosted" — más historia + features derivadas.

Mejoras sobre z403 (0.231), sin cambiar la tecnología (OLS) ni el train
set (182 mágicos):
  1. DOBLE CORTE: se suma el salto dic-2017->feb-2018 como filas extra
     de entrenamiento (mismo examen, un año antes).
  2. FEATURES DERIVADAS no lineales (las lineales son redundantes con
     los lags): feb_idx = feb_prev/promedio (dependencia de febrero),
     dic_idx = dic/promedio (pico navideño), cv = desvío/promedio
     (volatilidad relativa).
Lags alineados por CALENDARIO sobre panel con ceros (corrige el shift
por fila de z403, que desalinea si falta un mes).

Validación honesta (lección de exp_v4): entrenar en corte 2017 y
evaluar en corte 2018 (año distinto). Submits con ablación:
  A) lags, doble corte           (aísla el efecto historia)
  B) lags+features, corte 2018   (aísla el efecto features)
  C) lags+features, doble corte  (el boost completo)
"""

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_ventas

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
COMPETENCIA = "labo-iii-2026-ba"
EPS = 1e-6


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

# panel ancho calendario: fila=producto, columna=periodo; 0 desde el primer
# mes de vida del producto, NaN antes (no fabricar historia pre-lanzamiento)
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
    p = pd.Period(str(periodo), freq="M") - k
    return int(p.strftime("%Y%m"))


def armar_dataset(cut: int, con_clase: bool) -> pd.DataFrame:
    """Filas con lags calendario tn_0..tn_11 en `cut` (+ clase en cut+2)."""
    cols_lags = [mes_menos(cut, k) for k in range(12)]
    d = wide[cols_lags].copy()
    d.columns = [f"tn_{k}" for k in range(12)]
    d = d.dropna()  # producto vivo durante toda la ventana
    if con_clase:
        d["clase"] = wide.loc[d.index, mes_menos(cut, -2)]
        d = d.dropna(subset=["clase"])
    return d


def agregar_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    prom = d[[f"tn_{k}" for k in range(12)]].mean(axis=1)
    sd = d[[f"tn_{k}" for k in range(12)]].std(axis=1)
    d["feb_idx"] = d["tn_10"] / (prom + EPS)   # feb del año previo / promedio
    d["dic_idx"] = d["tn_0"] / (prom + EPS)    # diciembre / promedio
    d["cv"] = sd / (prom + EPS)                # volatilidad relativa
    return d


def entrenar(dfs: list[pd.DataFrame], features: bool):
    d = pd.concat([agregar_features(x) if features else x for x in dfs])
    X = sm.add_constant(d.drop(columns="clase"), has_constant="add")
    return sm.OLS(d["clase"], X).fit()


def predecir(modelo, d: pd.DataFrame, features: bool) -> pd.Series:
    d2 = agregar_features(d) if features else d
    X = sm.add_constant(d2, has_constant="add")
    return modelo.predict(X).clip(lower=0)


# ---------- backtest honesto: train corte 2017 -> eval corte 2018 ----------
dt17 = armar_dataset(201712, con_clase=True)
dt18 = armar_dataset(201812, con_clase=True)
dt17_mag = dt17[dt17.index.isin(magicos)]
dt18_mag = dt18[dt18.index.isin(magicos)]

print(f"filas train: corte2017 mágicos {len(dt17_mag)}, corte2018 mágicos {len(dt18_mag)}")
print("\n=== backtest honesto: train mágicos@2017 -> eval TODOS@2018->feb2019 ===")
eval_X = dt18.drop(columns="clase")
eval_y = dt18["clase"]
for nombre, feats in [("solo lags", False), ("lags+features", True)]:
    m = entrenar([dt17_mag], feats)
    pred = predecir(m, eval_X, feats)
    wape = float(np.abs(eval_y - pred).sum() / eval_y.sum())
    print(f"  {nombre:>14}: WAPE {wape:.4f}")

# referencia naif en la misma vara
wape_naif = float(np.abs(eval_y - eval_X[[f"tn_{k}" for k in range(12)]].mean(axis=1)).sum() / eval_y.sum())
print(f"  {'naif promedio':>14}: WAPE {wape_naif:.4f}")

# coeficientes de las features nuevas (train doble corte)
m_full = entrenar([dt17_mag, dt18_mag], True)
print("\ncoef de las features nuevas (doble corte):")
for f in ["feb_idx", "dic_idx", "cv"]:
    print(f"  {f}: {m_full.params[f]:+.3f} (t={m_full.tvalues[f]:+.1f})")

# ---------- prediccion final y submits ----------
dfut = armar_dataset(201912, con_clase=False)
tb_prom = (
    ventas.filter(pl.col("periodo").is_between(201901, 201912))
    .group_by("product_id").agg(pl.col("tn").mean())
)
apredecir = cargar_apredecir()

variantes = {
    "A_lags_doblecorte": ([dt17_mag, dt18_mag], False),
    "B_feats_corte18": ([dt18_mag], True),
    "C_feats_doblecorte": ([dt17_mag, dt18_mag], True),
}
for nombre, (dfs, feats) in variantes.items():
    modelo = entrenar(dfs, feats)
    pred = predecir(modelo, dfut, feats)
    tb_reg = pl.DataFrame({"product_id": dfut.index.to_list(), "tn_pred": pred.to_list()})
    tb_final = (
        apredecir.join(tb_prom, on="product_id", how="left")
        .join(tb_reg, on="product_id", how="left")
        .with_columns(pl.coalesce([pl.col("tn_pred"), pl.col("tn")]).alias("tn"))
        .select("product_id", "tn").sort("product_id")
    )
    assert tb_final.height == 780 and tb_final["tn"].null_count() == 0
    archivo = RUTA_EXP / f"exp_v5_{nombre}.csv"
    tb_final.write_csv(archivo)
    subprocess.run(
        [str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
         "-c", COMPETENCIA, "-f", str(archivo), "-m", f"exp_v5 {nombre}"],
        check=True,
    )
    print(f"submit OK -> {archivo.name} (aplica modelo a {len(dfut)} productos)")
