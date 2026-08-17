"""exp_v4: regresiones por grupos de estacionalidad compartida.

Hipótesis (de exp_v3): la OLS funciona cuando el train comparte una misma
dinámica dic->feb. En vez de una lista curada, agrupamos objetivamente:
  A) clusters k-means sobre el perfil estacional STL normalizado (forma
     del año de cada producto, sin escala)
  B) grupos por categoría cat2 (la agrupación "de negocio")
y ajustamos UNA regresión z403 por grupo.

Validación local antes de submitear: replicar todo en el salto
dic-2017 -> feb-2018 y medir WAPE contra lo realmente vendido en 201802.
Baseline a batir en esa misma vara: la regresión con lista mágica.

Fallbacks de predicción: grupo chico o producto sin cluster -> modelo
mágico global; sin 2019 completo -> promedio 2019.
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
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import STL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_productos, cargar_ventas

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
COMPETENCIA = "labo-iii-2026-ba"
SEMILLA = 102191
MIN_TRAIN = 30  # minimo de filas para ajustar una OLS de 13 parametros


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
apredecir = cargar_apredecir()

# ---------- lags estilo z403 + control de alineacion calendario ----------
lags = [-2, *range(0, 12)]
tb_lags = (
    ventas.sort(["product_id", "periodo"])
    .with_columns(
        [pl.col("tn").shift(lag).over("product_id").alias(f"tn_{lag}") for lag in lags]
        + [pl.col("periodo").shift(-2).over("product_id").alias("periodo_clase")]
    )
    .rename({"tn_-2": "clase"})
)
COLS_X = [f"tn_{k}" for k in range(12)]


def entrenar(cut: int, clase_esperada: int, ids):
    dtrain = tb_lags.filter(
        (pl.col("periodo") == cut)
        & (pl.col("periodo_clase") == clase_esperada)
        & (pl.col("product_id").is_in(ids))
    ).drop_nulls(["clase"] + COLS_X)
    if dtrain.height < MIN_TRAIN:
        return None
    X = sm.add_constant(dtrain.select(cs.starts_with("tn_")).to_pandas(), has_constant="add")
    return sm.OLS(dtrain["clase"].to_pandas(), X).fit()


def predecir(modelo, dfut: pl.DataFrame) -> pl.DataFrame:
    X = sm.add_constant(dfut.select(cs.starts_with("tn_")).to_pandas(), has_constant="add")
    return dfut.select("product_id").with_columns(
        pl.Series("tn_pred", modelo.predict(X)).clip(lower_bound=0)
    )


# ---------- perfil estacional y clusters ----------
periodos = pd.period_range("2017-01", "2019-12", freq="M")
prod36 = (
    ventas.group_by("product_id").agg(pl.col("periodo").n_unique().alias("m"))
    .filter(pl.col("m") == 36)["product_id"].to_list()
)
wide = (
    ventas.filter(pl.col("product_id").is_in(prod36))
    .to_pandas()
    .pivot(index="product_id", columns="periodo", values="tn")
    .reindex(columns=[int(p.strftime("%Y%m")) for p in periodos])
    .fillna(0.0)
)
perfiles = {}
for pid, serie in wide.iterrows():
    y = pd.Series(serie.values, index=periodos.to_timestamp())
    stl = STL(y, period=12, robust=True).fit()
    prof = pd.Series(stl.seasonal.values, index=periodos.month).groupby(level=0).mean()
    sd = prof.std()
    perfiles[pid] = (prof / sd).values if sd > 0 else np.zeros(12)

ids_perfil = list(perfiles.keys())
M = np.vstack([perfiles[p] for p in ids_perfil])

# ---------- grupos por categoria ----------
cats = cargar_productos().unique("product_id").select("product_id", "cat2")


def asignaciones_cluster(k: int) -> dict[int, int]:
    km = KMeans(n_clusters=k, random_state=SEMILLA, n_init=10).fit(M)
    return dict(zip(ids_perfil, km.labels_))


def grupos_de(variante) -> dict[int, list[int]]:
    """Devuelve {grupo: [product_ids]} para 'catN' o k entero."""
    if variante == "cat2":
        g = {}
        for pid, c in cats.iter_rows():
            g.setdefault(c, []).append(pid)
        return g
    asig = asignaciones_cluster(variante)
    g = {}
    for pid, lab in asig.items():
        g.setdefault(lab, []).append(pid)
    return g


# ---------- evaluacion en el backtest dic-2017 -> feb-2018 ----------
reales_1802 = ventas.filter(pl.col("periodo") == 201802).select("product_id", "tn")
dfut_bt = tb_lags.filter(
    (pl.col("periodo") == 201712) & (pl.col("tn_11").is_not_null())
)


def wape_backtest(preds: pl.DataFrame) -> float:
    j = reales_1802.join(preds, on="product_id", how="inner")
    return float((j["tn"] - j["tn_pred"]).abs().sum() / j["tn"].sum())


def evaluar_variante(variante) -> tuple[float, float]:
    """WAPE backtest de la variante y cobertura (share de tn evaluada por grupo)."""
    grupos = grupos_de(variante)
    modelo_global = entrenar(201712, 201802, magicos)
    piezas, tn_grupo = [], 0.0
    for gid, ids in grupos.items():
        dfut_g = dfut_bt.filter(pl.col("product_id").is_in(ids))
        if dfut_g.height == 0:
            continue
        m = entrenar(201712, 201802, ids)
        if m is not None:
            piezas.append(predecir(m, dfut_g))
            tn_grupo += dfut_g["tn_0"].sum()
        else:
            piezas.append(predecir(modelo_global, dfut_g))
    # productos sin grupo (sin 36m / sin cat) -> modelo global
    con_grupo = {p for ids in grupos.values() for p in ids}
    resto = dfut_bt.filter(~pl.col("product_id").is_in(list(con_grupo)))
    if resto.height:
        piezas.append(predecir(modelo_global, resto))
    todas = pl.concat(piezas).unique("product_id")
    return wape_backtest(todas), tn_grupo / dfut_bt["tn_0"].sum()


print("=== backtest dic-2017 -> feb-2018 (WAPE local; menor es mejor) ===")
base = entrenar(201712, 201802, magicos)
print(f"baseline magicos: {wape_backtest(predecir(base, dfut_bt)):.4f}")
resultados = {}
for v in [4, 6, 8, "cat2"]:
    w, cob = evaluar_variante(v)
    resultados[v] = w
    print(f"variante {v!s:>5}: {w:.4f}  (cobertura por grupos: {cob:.0%})")


# ---------- prediccion final 201912 -> 202002 y submit ----------
dfut_fin = tb_lags.filter(
    (pl.col("periodo") == 201912) & (pl.col("tn_11").is_not_null())
)
tb_prom = (
    ventas.filter(pl.col("periodo").is_between(201901, 201912))
    .group_by("product_id").agg(pl.col("tn").mean())
)


def submit_variante(variante, nombre: str) -> None:
    grupos = grupos_de(variante)
    modelo_global = entrenar(201812, 201902, magicos)
    piezas = []
    for gid, ids in grupos.items():
        dfut_g = dfut_fin.filter(pl.col("product_id").is_in(ids))
        if dfut_g.height == 0:
            continue
        m = entrenar(201812, 201902, ids)
        piezas.append(predecir(m if m is not None else modelo_global, dfut_g))
    con_grupo = {p for ids in grupos.values() for p in ids}
    resto = dfut_fin.filter(~pl.col("product_id").is_in(list(con_grupo)))
    if resto.height:
        piezas.append(predecir(modelo_global, resto))
    tb_reg = pl.concat(piezas).unique("product_id")

    tb_final = (
        apredecir.join(tb_prom, on="product_id", how="left")
        .join(tb_reg, on="product_id", how="left")
        .with_columns(pl.coalesce([pl.col("tn_pred"), pl.col("tn")]).alias("tn"))
        .select("product_id", "tn").sort("product_id")
    )
    assert tb_final.height == 780 and tb_final["tn"].null_count() == 0
    archivo = RUTA_EXP / f"exp_v4_{nombre}.csv"
    tb_final.write_csv(archivo)
    subprocess.run(
        [str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
         "-c", COMPETENCIA, "-f", str(archivo),
         "-m", f"exp_v4 regresion por grupos {nombre}"],
        check=True,
    )
    print(f"submit OK -> {archivo.name}")


# submitear solo lo que en el backtest le gane o empate al baseline
wape_base = wape_backtest(predecir(base, dfut_bt))
for v in [4, 6, 8, "cat2"]:
    if resultados[v] <= wape_base * 1.02:  # tolerancia 2%
        submit_variante(v, f"k{v}" if isinstance(v, int) else v)
    else:
        print(f"variante {v}: descartada por backtest ({resultados[v]:.4f} vs base {wape_base:.4f})")
