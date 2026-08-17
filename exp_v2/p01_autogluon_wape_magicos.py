"""exp_v2: AutoGluon con métrica WAPE + la receta de la regresión lineal.

Réplica de las condiciones de z403 (score 0.231) pero con AutoGluon:
  1. entrena SOLO con los 182 productos_magicos
  2. predice con el modelo los 656 productos con 2019 completo
  3. los 124 restantes van con el promedio de 2019
  4. eval_metric = WAPE (la métrica de la competencia), no RMSE como z316

Correr con: ../../.venv/bin/python p01_autogluon_wape_magicos.py
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_ventas

RUTA_EXP = Path(__file__).resolve().parent
SEMILLA = 102191
TIME_LIMIT = 3600
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


# ---------- datos ----------
magicos = cargar_magicos()
ventas = cargar_ventas().with_columns(
    pl.col("periodo").cast(pl.String).str.to_datetime("%Y%m").alias("timestamp")
)

# panel mensual regular por producto (desde su primer mes hasta 201912),
# meses sin ventas = 0 (AutoGluon necesita indice temporal sin huecos)
meses = ventas.select(pl.col("timestamp").unique()).sort("timestamp")
spans = ventas.group_by("product_id").agg(pl.col("timestamp").min().alias("t0"))
panel = (
    spans.join(meses, how="cross")
    .filter(pl.col("timestamp") >= pl.col("t0"))
    .join(ventas.select("product_id", "timestamp", "tn"),
          on=["product_id", "timestamp"], how="left")
    .with_columns(pl.col("tn").fill_null(0.0))
    .sort(["product_id", "timestamp"])
    .select("product_id", "timestamp", "tn")
)

# mismo filtro que z403: modelo para los que tienen los 12 meses de 2019 con venta
completos_2019 = (
    ventas.filter(pl.col("periodo").is_between(201901, 201912))
    .group_by("product_id")
    .agg(pl.col("periodo").n_unique().alias("m"))
    .filter(pl.col("m") == 12)
    .get_column("product_id")
    .to_list()
)
print(f"mágicos (train): {len(magicos)} | con 2019 completo (modelo): "
      f"{len(completos_2019)} | resto (promedio): {780 - len(completos_2019)}")

# ---------- entrenamiento ----------
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

ts_train = TimeSeriesDataFrame.from_data_frame(
    panel.filter(pl.col("product_id").is_in(magicos)).to_pandas(),
    id_column="product_id",
    timestamp_column="timestamp",
)

predictor = TimeSeriesPredictor(
    prediction_length=2,
    target="tn",
    freq="MS",
    eval_metric="WAPE",
    path=str(RUTA_EXP / "ag_model"),
)
predictor.fit(
    ts_train,
    num_val_windows=2,
    time_limit=TIME_LIMIT,
    presets="best_quality",
    random_seed=SEMILLA,
)
print(predictor.leaderboard())

# ---------- prediccion ----------
ts_future = TimeSeriesDataFrame.from_data_frame(
    panel.filter(pl.col("product_id").is_in(completos_2019)).to_pandas(),
    id_column="product_id",
    timestamp_column="timestamp",
)
forecast = predictor.predict(ts_future, random_seed=SEMILLA)

tb_modelo = (
    pl.from_pandas(forecast.reset_index())
    .filter(pl.col("timestamp") == datetime(2020, 2, 1))
    .select(
        pl.col("item_id").alias("product_id"),
        pl.col("mean").clip(lower_bound=0).alias("tn_modelo"),
    )
)

# fallback: promedio 2019
tb_prom = (
    ventas.filter(pl.col("periodo").is_between(201901, 201912))
    .group_by("product_id")
    .agg(pl.col("tn").mean())
)
tb_final = (
    cargar_apredecir()
    .join(tb_prom, on="product_id", how="left")
    .join(tb_modelo, on="product_id", how="left")
    .with_columns(pl.coalesce([pl.col("tn_modelo"), pl.col("tn")]).alias("tn"))
    .select("product_id", "tn")
    .sort("product_id")
)
assert tb_final.height == 780 and tb_final["tn"].null_count() == 0

archivo = RUTA_EXP / "exp_v2_autogluon_wape_magicos.csv"
tb_final.write_csv(archivo)
print(f"submission: {archivo} | suma tn = {tb_final['tn'].sum():.1f}")

# ---------- submit ----------
kaggle = RUTA_PROYECTO / ".venv/bin/kaggle"
subprocess.run(
    [str(kaggle), "competitions", "submit", "-c", COMPETENCIA,
     "-f", str(archivo), "-m", "exp_v2 AutoGluon WAPE, train=182 magicos, fallback promedio"],
    check=True,
)
print("submit enviado OK")
