"""exp_v3: regresión de z403 con la lista mágica AMPLIADA.

Al train set de 182 productos_magicos se le suman los productos con:
  - historia larga: 36 meses de venta (todo 201701-201912)
  - estacionalidad marcada: fuerza estacional STL > umbral
  - elegibilidad z403: 2018 completo + venta en 201902
Dos variantes de umbral (0.6 estricto, 0.5 laxo). El resto del pipeline
replica z403: lags [t-11..t] en 201812, clase = t+2 (201902), OLS con
intercepto, aplicación en 201912 a los 656 con 2019 completo, fallback
promedio 2019. Submit automático de ambas variantes.
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
import polars.selectors as cs
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_ventas

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
COMPETENCIA = "labo-iii-2026-ba"


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
ventas = cargar_ventas()

# --- atributos por producto: elegibilidad, historia, estacionalidad ---
attrs = ventas.group_by("product_id").agg(
    pl.col("periodo").filter(pl.col("periodo").is_between(201801, 201812))
    .n_unique().alias("m18"),
    (pl.col("periodo") == 201902).any().alias("f19"),
    pl.col("periodo").n_unique().alias("meses"),
)
elegibles = attrs.filter((pl.col("m18") == 12) & pl.col("f19"))

periodos = pd.period_range("2017-01", "2019-12", freq="M")
wide = (
    ventas.filter(pl.col("product_id").is_in(elegibles["product_id"].implode()))
    .to_pandas()
    .pivot(index="product_id", columns="periodo", values="tn")
    .reindex(columns=[int(p.strftime("%Y%m")) for p in periodos])
    .fillna(0.0)
)
fuerzas = {}
for pid, serie in wide.iterrows():
    y = pd.Series(serie.values, index=periodos.to_timestamp())
    stl = STL(y, period=12, robust=True).fit()
    fuerzas[pid] = max(0.0, 1.0 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))

elegibles = elegibles.with_columns(
    pl.col("product_id").replace_strict(fuerzas, default=None).alias("fs")
)

# --- dataset aplanado con lags, identico a z403 ---
lags = [-2, *range(0, 12)]
tb_lags = (
    ventas.sort(["product_id", "periodo"])
    .with_columns(
        [pl.col("tn").shift(lag).over("product_id").alias(f"tn_{lag}") for lag in lags]
    )
    .rename({"tn_-2": "clase"})
)

# promedio 2019 (fallback, identico a z403)
tb_prom = (
    ventas.filter(pl.col("periodo").is_between(201901, 201912))
    .group_by("product_id")
    .agg(pl.col("tn").mean())
)

dfuture = tb_lags.filter(
    (pl.col("periodo") == 201912) & (pl.col("tn_11").is_not_null())
)


def correr_variante(nombre: str, train_ids: list[int]) -> None:
    dtrain = tb_lags.filter(
        (pl.col("periodo") == 201812) & (pl.col("product_id").is_in(train_ids))
    ).drop_nulls(["clase"] + [f"tn_{k}" for k in range(12)])
    print(f"\n[{nombre}] train: {dtrain.height} productos")

    X = sm.add_constant(dtrain.select(cs.starts_with("tn_")).to_pandas())
    y = dtrain["clase"].to_pandas()
    modelo = sm.OLS(y, X).fit()
    print(f"[{nombre}] R2 ajustado: {modelo.rsquared_adj:.4f}")

    Xf = sm.add_constant(dfuture.select(cs.starts_with("tn_")).to_pandas())
    tb_reg = dfuture.select("product_id").with_columns(
        pl.Series("tn_pred", modelo.predict(Xf)).clip(lower_bound=0)
    )

    tb_final = (
        cargar_apredecir()
        .join(tb_prom, on="product_id", how="left")
        .join(tb_reg, on="product_id", how="left")
        .with_columns(pl.coalesce([pl.col("tn_pred"), pl.col("tn")]).alias("tn"))
        .select("product_id", "tn")
        .sort("product_id")
    )
    assert tb_final.height == 780 and tb_final["tn"].null_count() == 0

    archivo = RUTA_EXP / f"exp_v3_{nombre}.csv"
    tb_final.write_csv(archivo)
    subprocess.run(
        [str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
         "-c", COMPETENCIA, "-f", str(archivo),
         "-m", f"exp_v3 regresion ampliada {nombre} (train n={dtrain.height})"],
        check=True,
    )
    print(f"[{nombre}] submit OK -> {archivo.name}")


for umbral, nombre in [(0.6, "fs06"), (0.5, "fs05")]:
    extra = elegibles.filter(
        (pl.col("meses") == 36) & (pl.col("fs") > umbral)
    )["product_id"].to_list()
    train_ids = sorted(set(magicos) | set(extra))
    print(f"\n=== variante {nombre}: {len(magicos)} mágicos + {len(extra)} "
          f"candidatos (36m, fs>{umbral}) = {len(train_ids)} únicos "
          f"(+{len(train_ids) - len(magicos)} nuevos)")
    correr_variante(nombre, train_ids)
