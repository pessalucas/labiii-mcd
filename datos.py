"""Carga de datos compartida para las pruebas.

Uso:
    from datos import cargar_ventas, cargar_apredecir, RUTA_DATASETS
"""

from pathlib import Path

import polars as pl

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
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
