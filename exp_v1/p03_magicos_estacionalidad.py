"""p03: ¿Los productos_magicos son los que tienen estacionalidad?

Para cada uno de los 562 elegibles (2018 completo + venta en 201902) calcula:
- fuerza estacional (Hyndman): Fs = max(0, 1 - Var(resid)/Var(seasonal+resid))
  sobre una descomposición STL con período 12
- autocorrelación a lag 12 de la serie sin tendencia
y compara la distribución entre mágicos y no mágicos.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.seasonal import STL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import RUTA_PROYECTO, cargar_ventas

warnings.filterwarnings("ignore")


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

elegibles = (
    ventas.group_by("product_id")
    .agg(
        pl.col("periodo").filter(pl.col("periodo").is_between(201801, 201812))
        .n_unique().alias("m18"),
        (pl.col("periodo") == 201902).any().alias("f19"),
    )
    .filter((pl.col("m18") == 12) & pl.col("f19"))
    .get_column("product_id")
    .to_list()
)

# serie mensual completa 201701-201912 con huecos en 0
periodos = pd.period_range("2017-01", "2019-12", freq="M")
wide = (
    ventas.filter(pl.col("product_id").is_in(elegibles))
    .to_pandas()
    .pivot(index="product_id", columns="periodo", values="tn")
    .reindex(columns=[int(p.strftime("%Y%m")) for p in periodos])
    .fillna(0.0)
)

filas = []
for pid, serie in wide.iterrows():
    y = pd.Series(serie.values, index=periodos.to_timestamp())
    try:
        stl = STL(y, period=12, robust=True).fit()
        var_r = np.var(stl.resid)
        fs = max(0.0, 1.0 - var_r / np.var(stl.seasonal + stl.resid))
        # ACF lag 12 de la serie sin tendencia
        z = (y - stl.trend).to_numpy()
        z = z[~np.isnan(z)]
        z = z - z.mean()
        acf12 = float(np.dot(z[12:], z[:-12]) / np.dot(z, z)) if np.dot(z, z) > 0 else np.nan
    except Exception:
        fs, acf12 = np.nan, np.nan
    filas.append((pid, fs, acf12, pid in magicos))

res = pl.DataFrame(
    filas, schema=["product_id", "fuerza_estacional", "acf12", "magico"], orient="row"
)

print(f"series analizadas: {res.height} (mágicos: {res['magico'].sum()})\n")
print("=== distribución de la fuerza estacional STL (0=nada, 1=pura estación) ===")
print(
    res.group_by("magico").agg(
        pl.len().alias("n"),
        pl.col("fuerza_estacional").quantile(0.25).alias("p25"),
        pl.col("fuerza_estacional").median().alias("mediana"),
        pl.col("fuerza_estacional").quantile(0.75).alias("p75"),
        pl.col("acf12").median().alias("acf12_mediana"),
    ).sort("magico")
)

print("\n=== % de productos 'estacionales' según umbral de fuerza ===")
for u in [0.4, 0.5, 0.6, 0.7]:
    t = res.group_by("magico").agg(
        (pl.col("fuerza_estacional") > u).mean().alias("pct")
    ).sort("magico")
    no_m, si_m = t["pct"].to_list()
    print(f"  fuerza > {u}:  no-mágicos {no_m:.1%}  |  mágicos {si_m:.1%}")

print("\n=== tasa de mágicos por decil de fuerza estacional (global 32%) ===")
print(
    res.with_columns(
        pl.col("fuerza_estacional").qcut(10, labels=[f"D{i}" for i in range(1, 11)]).alias("decil")
    )
    .group_by("decil")
    .agg(pl.len().alias("n"), pl.col("magico").mean().alias("pct_magico"))
    .sort("decil")
)
