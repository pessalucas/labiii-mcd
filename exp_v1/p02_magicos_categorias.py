"""p02: ¿Los productos_magicos responden a alguna categoría o rango de toneladas?

Cruza la lista mágica contra cat1/cat2/cat3, brand, sku_size y deciles de tn.
Comparación dentro de los 562 "elegibles" (2018 completo + venta en 201902)
para no confundir el criterio de elegibilidad con el de selección.
"""

import json
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import RUTA_PROYECTO, cargar_productos, cargar_ventas


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

base = (
    ventas.group_by("product_id")
    .agg(
        pl.col("periodo").filter(pl.col("periodo").is_between(201801, 201812))
        .n_unique().alias("meses_2018"),
        (pl.col("periodo") == 201902).any().alias("tiene_201902"),
        pl.col("tn").filter(pl.col("periodo") >= 201901).sum().alias("tn_2019"),
    )
    .filter((pl.col("meses_2018") == 12) & pl.col("tiene_201902"))
    .with_columns(pl.col("product_id").is_in(magicos).alias("magico"))
    .join(cargar_productos().unique("product_id"), on="product_id", how="left")
)

tasa_global = base["magico"].mean()
print(f"elegibles: {base.height} | mágicos: {base['magico'].sum()} "
      f"| tasa global: {tasa_global:.1%}\n")


def tasa_por(col, min_n=10):
    t = (
        base.group_by(col)
        .agg(pl.len().alias("n"), pl.col("magico").mean().alias("pct_magico"))
        .filter(pl.col("n") >= min_n)
        .sort("pct_magico", descending=True)
    )
    print(f"=== % de mágicos por {col} (grupos con n>={min_n}; global {tasa_global:.0%}) ===")
    print(t)
    print()


tasa_por("cat1", min_n=1)
tasa_por("cat2", min_n=5)
tasa_por("cat3", min_n=10)
tasa_por("brand", min_n=10)

# tamaño de envase
base = base.with_columns(
    pl.col("sku_size").qcut(4, labels=["chico", "medio", "grande", "XL"]).alias("sku_q")
)
tasa_por("sku_q", min_n=1)

# deciles de toneladas 2019
base = base.with_columns(
    pl.col("tn_2019").qcut(10, labels=[f"D{i}" for i in range(1, 11)]).alias("decil_tn")
)
print("=== % de mágicos por decil de tn 2019 (D1=menor volumen, D10=mayor) ===")
print(
    base.group_by("decil_tn")
    .agg(pl.len().alias("n"), pl.col("magico").mean().alias("pct_magico"))
    .sort("decil_tn")
)
