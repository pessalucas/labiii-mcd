"""p01: ¿Con qué criterio fueron elegidos los productos_magicos de z403?

Compara los productos de la lista contra el resto de los 780 a predecir:
tamaño (tn 2019), completitud de historia, estabilidad (coef. de variación).
"""

import json
import re

import polars as pl

from datos import RUTA_PROYECTO, cargar_ventas

# extraigo la lista de productos_magicos directamente del notebook
nb = json.loads(
    (RUTA_PROYECTO / "src/Estadistica/z403_RegresionLineal_local.ipynb").read_text()
)
celda = next(
    "".join(c["source"])
    for c in nb["cells"]
    if c["cell_type"] == "code" and "productos_magicos" in "".join(c["source"])
)
# la celda define la lista dos veces; la segunda (la vigente) es la ultima
bloques = re.findall(r"productos_magicos = \[(.*?)\]", celda, flags=re.S)
magicos = [int(x) for x in bloques[-1].replace("\n", " ").split(",")]
print(f"productos_magicos: {len(magicos)}")

ventas = cargar_ventas()

resumen = (
    ventas.group_by("product_id")
    .agg(
        pl.col("periodo").n_unique().alias("meses_con_venta"),
        pl.col("periodo").min().alias("primer_mes"),
        pl.col("tn").filter(pl.col("periodo") >= 201901).sum().alias("tn_2019"),
        (pl.col("tn").std() / pl.col("tn").mean()).alias("coef_variacion"),
    )
    .with_columns(pl.col("product_id").is_in(magicos).alias("magico"))
)

print("\n=== Comparación mágicos vs resto (medianas por grupo) ===")
print(
    resumen.group_by("magico").agg(
        pl.len().alias("n"),
        pl.col("meses_con_venta").median(),
        pl.col("primer_mes").max().alias("primer_mes_mas_tardio"),
        pl.col("tn_2019").median().alias("tn_2019_mediana"),
        pl.col("coef_variacion").median().alias("cv_mediana"),
    )
)

total_2019 = resumen["tn_2019"].sum()
magicos_2019 = resumen.filter(pl.col("magico"))["tn_2019"].sum()
print(f"\nlos mágicos son {len(magicos)}/780 productos "
      f"pero concentran {100 * magicos_2019 / total_2019:.1f}% de las tn de 2019")

print("\n=== ¿Todos los mágicos tienen historia completa (36 meses)? ===")
print(resumen.filter(pl.col("magico")).group_by("meses_con_venta").len().sort("meses_con_venta"))

print("\n=== ranking por tamaño: ¿los mágicos son los más grandes? ===")
top = resumen.sort("tn_2019", descending=True).with_row_index("ranking", offset=1)
print("mágicos en el top 50 por tn 2019:", top.head(50)["magico"].sum(), "/ 50")
print("mágicos en el top 200 por tn 2019:", top.head(200)["magico"].sum(), "/ 200")
print("peor ranking de un mágico:", top.filter(pl.col("magico"))["ranking"].max())
