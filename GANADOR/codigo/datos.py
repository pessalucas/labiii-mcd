"""Carga de datos compartida para las pruebas.

Uso:
    from datos import cargar_ventas, cargar_apredecir, RUTA_DATASETS
"""

from pathlib import Path

import polars as pl

def _buscar_datasets() -> Path:
    """Sube directorios desde este archivo hasta encontrar datasets/sell-in.txt.gz.
    Robusto ante mover la carpeta GANADOR/. Alternativa: env LABO3_DATASETS."""
    import os
    if os.environ.get("LABO3_DATASETS"):
        return Path(os.environ["LABO3_DATASETS"]).resolve().parent
    aqui = Path(__file__).resolve()
    for base in [aqui, *aqui.parents]:
        cand = base / "datasets" / "sell-in.txt.gz"
        if cand.exists():
            return base
    raise FileNotFoundError(
        "No encontré datasets/sell-in.txt.gz subiendo desde " + str(aqui)
        + ". Poné los 4 archivos en una carpeta 'datasets/' o definí LABO3_DATASETS.")


RUTA_PROYECTO = _buscar_datasets()
RUTA_DATASETS = RUTA_PROYECTO / "datasets"
RUTA_EXP = RUTA_PROYECTO / "exp"


def cargar_sellin() -> pl.DataFrame:
    """Sell-in crudo: (periodo, customer_id, product_id, ...) 2.9M filas."""
    return pl.read_csv(RUTA_DATASETS / "sell-in.txt.gz", separator="\t")


def cargar_apredecir() -> pl.DataFrame:
    """Los 780 product_id a predecir para 202002."""
    return pl.read_csv(RUTA_DATASETS / "product_id_apredecir201912.txt", separator="\t")


def cargar_productos() -> pl.DataFrame:
    """Maestro de productos: cat1 > cat2 > cat3, brand, sku_size."""
    return pl.read_csv(RUTA_DATASETS / "tb_productos.txt", separator="\t")


def cargar_stocks() -> pl.DataFrame:
    """Stock a fin de mes por producto (solo 201810-201912)."""
    return pl.read_csv(RUTA_DATASETS / "tb_stocks.txt", separator="\t")


def cargar_ventas(solo_apredecir: bool = True) -> pl.DataFrame:
    """Serie mensual por producto: (product_id, periodo, tn), ordenada.

    Es la tabla base de todos los modelos (tn agregada sobre clientes).
    """
    ventas = (
        cargar_sellin()
        .group_by("product_id", "periodo")
        .agg(pl.col("tn").sum())
    )
    if solo_apredecir:
        ventas = ventas.join(cargar_apredecir(), on="product_id", how="inner")
    return ventas.sort(["product_id", "periodo"])
