"""exp_v13 - paso 1: features de CAT4 (cat2,brand) + penetración en cat4.

Espejo de exp_v11/p01_features_categoria.py pero con la agrupación cat4
(sustitución por tamaño Y sabor de la misma marca). Genera, por
(product_id, corte):
  c4_*  : maquinaria completa sobre la serie agregada del cat4
  sh4_* : share del producto dentro de su cat4 y su evolución
Salida: features_cat4.parquet
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import cargar_sellin

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
RUTA_PRUEBAS = RUTA_EXP.parent
EPS = 1e-6
PERIODOS = [int(p.strftime("%Y%m")) for p in pd.period_range("2017-01", "2019-12", freq="M")]
CORTES = [int(p.strftime("%Y%m")) for p in pd.period_range("2018-01", "2019-12", freq="M")]


def mes_menos(p: int, k: int) -> int:
    return int((pd.Period(str(p), freq="M") - k).strftime("%Y%m"))


sellin = cargar_sellin()
ventas = sellin.group_by("product_id", "periodo").agg(pl.col("tn").sum())
cat4map = pd.read_parquet(RUTA_PRUEBAS / "cat4_mapping.parquet")
cat4_de = dict(zip(cat4map["product_id"], cat4map["cat4"]))

w_prod = (ventas.to_pandas().pivot(index="product_id", columns="periodo", values="tn")
          .reindex(columns=PERIODOS))
pm = w_prod.apply(lambda r: r.first_valid_index(), axis=1)
for pid in w_prod.index:
    w_prod.loc[pid, w_prod.columns >= pm[pid]] = w_prod.loc[pid, w_prod.columns >= pm[pid]].fillna(0.0)

# panel de cat4: suma de miembros
w4 = w_prod.copy()
w4["__c"] = [cat4_de.get(p, "SIN") for p in w_prod.index]
w4 = w4.groupby("__c").sum(min_count=1)
print(f"grupos cat4: {len(w4)} | productos: {len(w_prod)}")


def maquinaria(L: np.ndarray, pfx: str) -> dict:
    prom = np.nanmean(L[:, :12], axis=1)
    Ln = L / (prom[:, None] + EPS)
    f = {}
    for k in range(24):
        f[f"{pfx}tn_{k}"] = Ln[:, k]
    f[f"{pfx}log_prom"] = np.log1p(prom)
    f[f"{pfx}r_0_1"] = np.clip((L[:, 0] + EPS) / (L[:, 1] + EPS), 0, 10)
    f[f"{pfx}r_tri"] = np.clip((L[:, :3].sum(1) + EPS) / (L[:, 3:6].sum(1) + EPS), 0, 10)
    with np.errstate(invalid="ignore"):
        f[f"{pfx}r_yoy_tri"] = np.clip((L[:, :3].sum(1) + EPS) / (L[:, 12:15].sum(1) + EPS), 0, 10)
    for k in range(3):
        f[f"{pfx}diff_{k}_{k+1}"] = Ln[:, k] - Ln[:, k + 1]
    f[f"{pfx}diff_0_12"] = Ln[:, 0] - Ln[:, 12]
    for k in (2, 3, 6):
        f[f"{pfx}s{k}"] = L[:, :k].sum(1) / (k * prom + EPS)
    for k in (3, 6, 12):
        f[f"{pfx}roll_med_{k}"] = np.nanmedian(Ln[:, :k], axis=1)
    f[f"{pfx}roll_max_12"] = np.nanmax(Ln[:, :12], axis=1)
    f[f"{pfx}dmax_12"] = Ln[:, 0] - np.nanmax(Ln[:, :12], axis=1)
    f[f"{pfx}dprom_24"] = Ln[:, 0] - np.nanmean(Ln[:, :24], axis=1)
    f[f"{pfx}ma3_lag3"] = L[:, 3:6].mean(1) / (prom + EPS)
    f[f"{pfx}acel_ma3"] = (L[:, :3].mean(1) - L[:, 3:6].mean(1)) / (prom + EPS)

    def pend(M):
        k = M.shape[1]; t = np.arange(k) - (k - 1) / 2
        y = M - np.nanmean(M, axis=1, keepdims=True)
        s = (y * t).sum(axis=1) / (t ** 2).sum()
        s[np.isnan(M).any(axis=1)] = np.nan
        return s
    for k in (6, 12):
        f[f"{pfx}pend_{k}"] = pend(Ln[:, :k][:, ::-1])
    f[f"{pfx}cv"] = np.nanstd(Ln[:, :12], axis=1)
    return f


def lags24(panel, idx, cut):
    cols = [mes_menos(cut, k) for k in range(24)]
    out = np.full((len(idx), 24), np.nan)
    for j, c in enumerate(cols):
        if c in panel.columns:
            out[:, j] = panel.loc[idx, c].values
    return out


tablas = []
for cut in CORTES:
    cols12 = [mes_menos(cut, k) for k in range(12)]
    vivos = w_prod.index[w_prod[cols12].notna().all(axis=1)]
    c4s = pd.Index([cat4_de.get(p, "SIN") for p in vivos])
    Lc = lags24(w4, w4.index, cut)
    fc = maquinaria(Lc, "c4_")
    pos = {c: i for i, c in enumerate(w4.index)}
    rows = np.array([pos[c] for c in c4s])
    d = pd.DataFrame({k: v[rows] for k, v in fc.items()}, index=vivos)

    Lp = lags24(w_prod, vivos, cut)
    Lcat = Lc[rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        SH = Lp / (Lcat + EPS)
    d["sh4_0"] = SH[:, 0]
    d["sh4_prom12"] = np.nanmean(SH[:, :12], axis=1)
    d["sh4_ma3"] = np.nanmean(SH[:, :3], axis=1)
    d["sh4_acel"] = np.nanmean(SH[:, :3], axis=1) - np.nanmean(SH[:, 3:6], axis=1)
    d["sh4_yoy"] = SH[:, 0] - SH[:, 12]

    def pend_sh(M):
        k = M.shape[1]; t = np.arange(k) - (k - 1) / 2
        y = M - np.nanmean(M, axis=1, keepdims=True)
        s = (y * t).sum(axis=1) / (t ** 2).sum()
        s[np.isnan(M).any(axis=1)] = np.nan
        return s
    for k in (3, 6, 12):
        d[f"sh4_pend_{k}"] = pend_sh(SH[:, :k][:, ::-1])

    d.insert(0, "corte", cut)
    d.insert(0, "product_id", vivos)
    tablas.append(d.reset_index(drop=True))

out = pd.concat(tablas, ignore_index=True)
out.to_parquet(RUTA_EXP / "features_cat4.parquet")
print(f"features_cat4.parquet: {out.shape[0]} filas x {out.shape[1]-2} features")

from features_lgbm import FeatureBuilder
d812 = out[out["corte"] == 201812].drop(columns=["product_id", "corte"])
prob = FeatureBuilder.auditar(d812)
print("auditoría:", "✓ limpia" if not prob else prob)
