"""exp_v11: generación de FEATURES DE CATEGORÍA (cat3) + PENETRACIÓN.

Spec del usuario:
  - Para cada producto, tomar la SUMA mensual de todos los productos de
    su cat3 (ej: una mayonesa hereda la serie "Salsas") y calcular sobre
    esa serie toda la maquinaria de features (lags, diffs, ratios, sumas,
    rolling, dmin/dmax/dprom, tendencias) -> prefijo c3_
  - Ratios de representatividad del producto en su categoría y su
    evolución (¿está ganando penetración?) -> prefijo sh_
  - Solo GENERACIÓN (sin entrenar): output en features_categoria.parquet
    con clave (product_id, corte), cortes 201801..201912.

Auditorías integradas:
  1. la serie de categoría cuadra con la suma de sus miembros
  2. shares en [0,1] y share promedio de la categoría suma ~1
  3. categorías mono-producto (categoría==producto: redundancia esperada)
  4. detector de constantes/duplicados (Spearman) sobre un corte
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import cargar_productos, cargar_sellin
from features_lgbm import FeatureBuilder

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
EPS = 1e-6
PERIODOS = [int(p.strftime("%Y%m")) for p in pd.period_range("2017-01", "2019-12", freq="M")]
CORTES = [int(p.strftime("%Y%m")) for p in pd.period_range("2018-01", "2019-12", freq="M")]


def mes_menos(p: int, k: int) -> int:
    return int((pd.Period(str(p), freq="M") - k).strftime("%Y%m"))


# ---------- series base ----------
sellin = cargar_sellin()
ventas = sellin.group_by("product_id", "periodo").agg(pl.col("tn").sum())
prod_cat = (cargar_productos().unique("product_id")
            .select("product_id", pl.col("cat3").fill_null("SIN_CAT")))

# panel producto (ceros desde el primer mes de vida)
w_prod = (ventas.to_pandas().pivot(index="product_id", columns="periodo", values="tn")
          .reindex(columns=PERIODOS))
pm = w_prod.apply(lambda r: r.first_valid_index(), axis=1)
for pid in w_prod.index:
    w_prod.loc[pid, w_prod.columns >= pm[pid]] = (
        w_prod.loc[pid, w_prod.columns >= pm[pid]].fillna(0.0))

# panel categoría: suma de los miembros (0 y NaN de miembros -> el total
# de la categoría existe desde que nace su primer miembro)
cat_de = dict(zip(prod_cat["product_id"].to_list(), prod_cat["cat3"].to_list()))
w_cat = w_prod.copy()
w_cat["__cat"] = [cat_de.get(p, "SIN_CAT") for p in w_prod.index]
w_cat = w_cat.groupby("__cat").sum(min_count=1)
print(f"categorías cat3: {len(w_cat)} | productos: {len(w_prod)}")


# ---------- maquinaria sobre una matriz de lags (espejo FeatureBuilder) ----------
def maquinaria(L: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """L: (n, 24) lags crudos con NaN donde no existen. Devuelve features."""
    prom = np.nanmean(L[:, :12], axis=1)
    Ln = L / (prom[:, None] + EPS)
    f = {}
    for k in range(24):
        f[f"{prefix}tn_{k}"] = Ln[:, k]
    f[f"{prefix}log_prom"] = np.log1p(prom)
    f[f"{prefix}r_0_1"] = np.clip((L[:, 0] + EPS) / (L[:, 1] + EPS), 0, 10)
    f[f"{prefix}r_tri"] = np.clip(
        (L[:, :3].sum(1) + EPS) / (L[:, 3:6].sum(1) + EPS), 0, 10)
    with np.errstate(invalid="ignore"):
        f[f"{prefix}r_yoy_tri"] = np.clip(
            (L[:, :3].sum(1) + EPS) / (L[:, 12:15].sum(1) + EPS), 0, 10)
    for k in range(3):
        f[f"{prefix}diff_{k}_{k+1}"] = Ln[:, k] - Ln[:, k + 1]
    f[f"{prefix}diff_0_12"] = Ln[:, 0] - Ln[:, 12]
    for k in (2, 3, 6):
        f[f"{prefix}s{k}"] = L[:, :k].sum(1) / (k * prom + EPS)
    for k in (3, 6, 12):
        f[f"{prefix}roll_med_{k}"] = np.nanmedian(Ln[:, :k], axis=1)
    f[f"{prefix}roll_max_12"] = np.nanmax(Ln[:, :12], axis=1)
    f[f"{prefix}dmax_12"] = Ln[:, 0] - np.nanmax(Ln[:, :12], axis=1)
    f[f"{prefix}dprom_24"] = Ln[:, 0] - np.nanmean(Ln[:, :24], axis=1)
    f[f"{prefix}ma3_lag3"] = L[:, 3:6].mean(1) / (prom + EPS)
    f[f"{prefix}acel_ma3"] = (L[:, :3].mean(1) - L[:, 3:6].mean(1)) / (prom + EPS)

    def pend(M):
        k = M.shape[1]
        t = np.arange(k) - (k - 1) / 2
        y = M - np.nanmean(M, axis=1, keepdims=True)
        s = (y * t).sum(axis=1) / (t ** 2).sum()
        s[np.isnan(M).any(axis=1)] = np.nan
        return s

    for k in (6, 12):
        f[f"{prefix}pend_{k}"] = pend(Ln[:, :k][:, ::-1])
    f[f"{prefix}cv"] = np.nanstd(Ln[:, :12], axis=1)
    return f


def lags24(panel: pd.DataFrame, idx, cut: int) -> np.ndarray:
    cols = [mes_menos(cut, k) for k in range(24)]
    out = np.full((len(idx), 24), np.nan)
    for j, c in enumerate(cols):
        if c in panel.columns:
            out[:, j] = panel.loc[idx, c].values
    return out


# ---------- generación por corte ----------
tablas = []
for cut in CORTES:
    # productos vivos (12 meses de historia al corte)
    cols12 = [mes_menos(cut, k) for k in range(12)]
    vivos = w_prod.index[w_prod[cols12].notna().all(axis=1)]
    cats = pd.Index([cat_de.get(p, "SIN_CAT") for p in vivos])

    # features de la CATEGORÍA (se calculan 1 vez por categoría y se asignan)
    Lc_cat = lags24(w_cat, w_cat.index, cut)
    fc = maquinaria(Lc_cat, "c3_")
    pos = {c: i for i, c in enumerate(w_cat.index)}
    rows = np.array([pos[c] for c in cats])
    d = pd.DataFrame({k: v[rows] for k, v in fc.items()}, index=vivos)

    # PENETRACIÓN: share del producto en su categoría, mes a mes
    Lp = lags24(w_prod, vivos, cut)          # (n, 24) producto
    Lc = Lc_cat[rows]                        # (n, 24) su categoría
    with np.errstate(invalid="ignore", divide="ignore"):
        SH = Lp / (Lc + EPS)                 # share por mes (0..1)
    d["sh_0"] = SH[:, 0]
    d["sh_prom12"] = np.nanmean(SH[:, :12], axis=1)
    d["sh_ma3"] = np.nanmean(SH[:, :3], axis=1)
    d["sh_ma3_lag3"] = np.nanmean(SH[:, 3:6], axis=1)
    d["sh_acel"] = d["sh_ma3"] - d["sh_ma3_lag3"]
    d["sh_yoy"] = SH[:, 0] - SH[:, 12]       # NaN si no hay año previo

    def pend_sh(M):
        k = M.shape[1]
        t = np.arange(k) - (k - 1) / 2
        y = M - np.nanmean(M, axis=1, keepdims=True)
        s = (y * t).sum(axis=1) / (t ** 2).sum()
        s[np.isnan(M).any(axis=1)] = np.nan
        return s

    for k in (3, 6, 12):
        d[f"sh_pend_{k}"] = pend_sh(SH[:, :k][:, ::-1])

    d.insert(0, "corte", cut)
    d.insert(0, "product_id", vivos)
    tablas.append(d.reset_index(drop=True))

out = pd.concat(tablas, ignore_index=True)
archivo = RUTA_EXP / "features_categoria.parquet"
out.to_parquet(archivo)
print(f"generado: {archivo.name} | {out.shape[0]} filas x {out.shape[1]-2} features "
      f"| cortes {CORTES[0]}..{CORTES[-1]}")

# ================= AUDITORÍAS =================
print("\n=== AUDITORÍA 1: la categoría cuadra con la suma de sus miembros ===")
cat_test = "Mayonesa" if "Mayonesa" in w_cat.index else w_cat.index[0]
miembros = [p for p, c in cat_de.items() if c == cat_test and p in w_prod.index]
suma_manual = w_prod.loc[miembros, 201912].sum()
print(f"  {cat_test}: {len(miembros)} miembros | suma manual 201912 = "
      f"{suma_manual:.2f} vs panel = {w_cat.loc[cat_test, 201912]:.2f} "
      f"{'✓' if abs(suma_manual - w_cat.loc[cat_test, 201912]) < 1e-6 else '✗ ERROR'}")

print("\n=== AUDITORÍA 2: shares en [0,1] ===")
sh_cols = [c for c in out.columns if c in ("sh_0", "sh_prom12", "sh_ma3")]
mal = out[(out[sh_cols] < -1e-9).any(axis=1) | (out[sh_cols] > 1 + 1e-9).any(axis=1)]
print(f"  filas con share fuera de [0,1]: {len(mal)} "
      f"{'✓' if len(mal) == 0 else '✗ REVISAR'}")
print(f"  sh_0: min={out['sh_0'].min():.4f} max={out['sh_0'].max():.4f} "
      f"mediana={out['sh_0'].median():.4f}")

print("\n=== AUDITORÍA 3: categorías mono-producto (share==1, redundancia) ===")
mono = (out.groupby("product_id")["sh_prom12"].max() > 0.999)
print(f"  productos que SON su categoría (sh~1): {mono.sum()} de {mono.shape[0]}")

print("\n=== AUDITORÍA 4: constantes/duplicados (corte 201812) ===")
d812 = out[out["corte"] == 201812].drop(columns=["product_id", "corte"])
problemas = FeatureBuilder.auditar(d812)
print("  " + ("✓ limpia" if not problemas else "\n  ".join(problemas)))

print("\n=== AUDITORÍA 5 (sentido de negocio): ejemplo de penetración ===")
d19 = out[out["corte"] == 201912].set_index("product_id")
top_pen = d19.nlargest(3, "sh_pend_12")[["sh_0", "sh_prom12", "sh_pend_12", "sh_yoy"]]
print("  3 productos ganando MAS penetración (sh_pend_12):")
print(top_pen.round(4).to_string())
caida = d19.nsmallest(2, "c3_pend_12")[["c3_s2", "c3_pend_12"]]
print("  2 productos cuya CATEGORÍA más cae (c3_pend_12):")
print(caida.round(4).to_string())
