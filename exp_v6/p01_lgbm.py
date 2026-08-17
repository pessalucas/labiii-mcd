"""exp_v6: LightGBM sobre el diseño de la regresión mágica.

Dos variantes:
  A) Réplica exacta del diseño z403: 182 mágicos, corte único dic-2018,
     12 lags calendario. Solo cambia OLS -> LGBM (objetivo L1, que
     optimiza el numerador del WAPE).
  B) "Toda la historia": cortes deslizantes t -> t+2 (t = 201712..201910)
     de los mágicos + features derivadas (idx_objetivo = mismo mes del
     target un año antes / promedio, idx_actual, cv) + FLAGS DE REGIMEN:
     anio_corte y mes_corte, para que el árbol pueda separar la dinámica
     de cada año (efecto macro 2018/2019 detectado en exp_v5) y
     especializarse en los cortes de diciembre.

Validación honesta (estándar exp_v4/v5): evaluar en el corte 201812 ->
feb-2019 con modelos entrenados SOLO con información anterior.
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


def armar_corte(cut: int, con_clase: bool, features: bool) -> pd.DataFrame:
    cols = [mes_menos(cut, k) for k in range(12)]
    d = wide[cols].copy()
    d.columns = COLS_LAGS
    d = d.dropna()
    if con_clase:
        objetivo = mes_menos(cut, -2)
        if objetivo not in wide.columns:
            return pd.DataFrame()
        d["clase"] = wide.loc[d.index, objetivo]
        d = d.dropna(subset=["clase"])
    if features:
        prom = d[COLS_LAGS].mean(axis=1)
        sd = d[COLS_LAGS].std(axis=1)
        # tn_10 = mismo mes calendario que el objetivo, un año antes
        d["idx_objetivo"] = d["tn_10"] / (prom + EPS)
        d["idx_actual"] = d["tn_0"] / (prom + EPS)
        d["cv"] = sd / (prom + EPS)
        d["anio_corte"] = cut // 100
        d["mes_corte"] = cut % 100
    return d


def entrenar_lgbm(d: pd.DataFrame, chico: bool) -> lgb.LGBMRegressor:
    params = dict(
        objective="l1",
        random_state=SEMILLA,
        verbosity=-1,
        n_estimators=400,
        learning_rate=0.03,
    )
    if chico:  # 182 filas: arbol minusculo y muy regularizado
        params.update(num_leaves=4, min_child_samples=20, colsample_bytree=0.7,
                      subsample=0.8, subsample_freq=1)
    else:
        params.update(num_leaves=31, min_child_samples=30, colsample_bytree=0.8,
                      subsample=0.8, subsample_freq=1)
    m = lgb.LGBMRegressor(**params)
    m.fit(d.drop(columns="clase"), d["clase"])
    return m


def wape(real: pd.Series, pred: np.ndarray) -> float:
    return float(np.abs(real - pred).sum() / real.sum())


# ---------- backtest honesto: eval corte 201812 -> feb-2019 ----------
print("=== backtest honesto (eval: TODOS los productos @201812 -> 201902) ===")
ev = armar_corte(201812, con_clase=True, features=False)
ev_f = armar_corte(201812, con_clase=True, features=True)

# A: replica del diseño con corte 201712 (unico corte previo disponible)
dA = armar_corte(201712, True, False)
dA = dA[dA.index.isin(magicos)]
mA = entrenar_lgbm(dA, chico=True)
print(f"A lgbm corte unico   (train {len(dA)}): "
      f"WAPE {wape(ev['clase'], mA.predict(ev.drop(columns='clase')).clip(0)):.4f}")

# B: toda la historia previa (cortes 201712..201810) + features + flags
cortesB_bt = [mes_menos(201810, k) for k in range(11)]  # 201712..201810
dB = pd.concat([armar_corte(c, True, True) for c in cortesB_bt])
dB = dB[dB.index.isin(magicos)]
mB = entrenar_lgbm(dB, chico=False)
print(f"B lgbm historia+flags (train {len(dB)}): "
      f"WAPE {wape(ev_f['clase'], mB.predict(ev_f.drop(columns='clase')).clip(0)):.4f}")
print("(referencias en esta vara: OLS mágica 0.198, naif promedio 0.252)")

# ---------- modelos finales y submits ----------
tb_prom = (
    ventas.filter(pl.col("periodo").is_between(201901, 201912))
    .group_by("product_id").agg(pl.col("tn").mean())
)
apredecir = cargar_apredecir()


def submitear(nombre: str, modelo, features: bool) -> None:
    dfut = armar_corte(201912, con_clase=False, features=features)
    pred = pd.Series(modelo.predict(dfut), index=dfut.index).clip(lower=0)
    tb_reg = pl.DataFrame({"product_id": dfut.index.to_list(), "tn_pred": pred.to_list()})
    tb_final = (
        apredecir.join(tb_prom, on="product_id", how="left")
        .join(tb_reg, on="product_id", how="left")
        .with_columns(pl.coalesce([pl.col("tn_pred"), pl.col("tn")]).alias("tn"))
        .select("product_id", "tn").sort("product_id")
    )
    assert tb_final.height == 780 and tb_final["tn"].null_count() == 0
    archivo = RUTA_EXP / f"exp_v6_{nombre}.csv"
    tb_final.write_csv(archivo)
    subprocess.run(
        [str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
         "-c", COMPETENCIA, "-f", str(archivo), "-m", f"exp_v6 lgbm {nombre}"],
        check=True,
    )
    print(f"submit OK -> {archivo.name}")


# A final: corte 201812, magicos, sin features (replica z403 con lgbm)
dAf = armar_corte(201812, True, False)
dAf = dAf[dAf.index.isin(magicos)]
submitear("A_replica", entrenar_lgbm(dAf, chico=True), features=False)

# B final: cortes 201712..201910, magicos, features + flags
cortesB = [mes_menos(201910, k) for k in range(23)]  # 201712..201910
dBf = pd.concat([armar_corte(c, True, True) for c in cortesB])
dBf = dBf[dBf.index.isin(magicos)]
print(f"\nB final: {len(dBf)} filas de entrenamiento "
      f"({len(cortesB)} cortes x ~182 mágicos)")
modeloB = entrenar_lgbm(dBf, chico=False)
imp = pd.Series(modeloB.feature_importances_,
                index=dBf.drop(columns="clase").columns).sort_values(ascending=False)
print("importancia de features (top 8):")
print(imp.head(8).to_string())
submitear("B_historia_flags", modeloB, features=True)
