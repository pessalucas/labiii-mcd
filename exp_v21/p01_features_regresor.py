"""exp_v21 - paso 1: features de STOCK (stk_*) + QTY/MIX (q_*, mix_*) para el regresor.

Señales que la OLS estructuralmente no tiene (solo ve lags de tn) -> suben
la decorrelación de la pata LGBM. Todas scale-free (normalizadas por la
media del propio producto). Ver ESPECIFICACION_variables.md partes 1-2.
Salida: features_extra_regresor.parquet (product_id, corte, stk_*, q_*, mix_*).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import cargar_sellin, cargar_stocks
from features_lgbm import FeatureBuilder

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
EPS = 1e-6
CORTES = [int(p.strftime("%Y%m")) for p in pd.period_range("2018-01", "2019-12", freq="M")]

fb = FeatureBuilder()
wide = fb.wide                       # tn por producto x periodo (con ceros)
mes_menos = fb.mes_menos
periodos = fb.periodos


def panel(df: pl.DataFrame, val: str) -> pd.DataFrame:
    w = (df.to_pandas().pivot_table(index="product_id", columns="periodo", values=val, aggfunc="sum")
         .reindex(columns=periodos))
    return w


# panel de stock (NaN antes de 201810; NO rellenar con 0: ausencia != cero stock)
wide_stk = panel(cargar_stocks().rename({"stock_final": "v"}), "v")
# panel de qty pedida (historia completa; ceros desde primer mes, como tn)
sellin = cargar_sellin()
wq = panel(sellin.group_by("product_id", "periodo").agg(pl.col("cust_request_qty").sum().alias("v")), "v")
pm = wq.apply(lambda r: r.first_valid_index(), axis=1)
for pid in wq.index:
    wq.loc[pid, wq.columns >= pm[pid]] = wq.loc[pid, wq.columns >= pm[pid]].fillna(0.0)
wide_qty = wq
print(f"paneles: stock {wide_stk.notna().any(axis=1).sum()} prods | qty {len(wide_qty)} prods")


def pend(M):  # pendiente OLS por fila (tiempo ascendente); NaN si falta algo
    k = M.shape[1]; t = np.arange(k) - (k - 1) / 2
    y = M - np.nanmean(M, axis=1, keepdims=True)
    s = (y * t).sum(axis=1) / (t ** 2).sum()
    s[np.isnan(M).any(axis=1)] = np.nan
    return s


def cols(panel, idx, cut, n):
    cs = [mes_menos(cut, k) for k in range(n)]
    out = np.full((len(idx), n), np.nan)
    for j, c in enumerate(cs):
        if c in panel.columns:
            out[:, j] = panel.reindex(idx)[c].values
    return out


tablas = []
for C in CORTES:
    cols12 = [mes_menos(C, k) for k in range(12)]
    vivos = wide.index[wide[cols12].notna().all(axis=1)]
    d = pd.DataFrame(index=vivos)
    TN = cols(wide, vivos, C, 13)
    prom_tn = np.nanmean(TN[:, :12], axis=1)
    venta_prom6 = np.nanmean(TN[:, :6], axis=1)

    # ---- STOCK ----
    S = cols(wide_stk, vivos, C, 6)
    media_stk = np.nanmean(S, axis=1)
    d["stk_cob"] = S[:, 0] / (venta_prom6 + EPS)
    S2 = cols(wide_stk, vivos, mes_menos(C, 2), 6)
    cob_lag2 = S2[:, 0] / (np.nanmean(cols(wide, vivos, mes_menos(C, 2), 6)[:, :6], axis=1) + EPS)
    d["stk_cob_delta"] = d["stk_cob"] - cob_lag2
    d["stk_nivel"] = S[:, 0] / (media_stk + EPS)
    d["stk_delta1"] = (S[:, 0] - S[:, 1]) / (media_stk + EPS)
    d["stk_pend3"] = pend(S[:, :4][:, ::-1] / (media_stk[:, None] + EPS))
    d["stk_vs_tn"] = S[:, 0] / (TN[:, 0] + EPS)
    d["stk_n"] = np.sum(~np.isnan(cols(wide_stk, vivos, C, 15)), axis=1)

    # ---- QTY (tendencia de pedidos) ----
    Q = cols(wide_qty, vivos, C, 13)
    prom_q = np.nanmean(Q[:, :12], axis=1)
    Qn = Q / (prom_q[:, None] + EPS)
    d["q_s3"] = Q[:, :3].sum(1) / (3 * prom_q + EPS)
    d["q_s6"] = Q[:, :6].sum(1) / (6 * prom_q + EPS)
    d["q_diff_0_1"] = Qn[:, 0] - Qn[:, 1]
    d["q_diff_0_2"] = Qn[:, 0] - Qn[:, 2]
    d["q_pend6"] = pend(Qn[:, :6][:, ::-1])
    d["q_pend12"] = pend(Qn[:, :12][:, ::-1])
    d["q_yoy"] = Q[:, 0] / (Q[:, 12] + EPS)
    d["q_roll_std6"] = np.nanstd(Qn[:, :6], axis=1)
    d["q_dmax6"] = Qn[:, 0] - np.nanmax(Qn[:, :6], axis=1)

    # ---- MIX (tn/qty = peso por unidad) ----
    mix = TN / (Q + EPS)                        # mix por mes
    prom_mix = np.nanmean(mix[:, :12], axis=1)
    d["mix_ratio"] = mix[:, 0] / (prom_mix + EPS)
    d["mix_pend6"] = pend((mix[:, :6] / (prom_mix[:, None] + EPS))[:, ::-1])
    d["mix_yoy"] = mix[:, 0] / (mix[:, 12] + EPS)
    # ¿pedidos crecen más rápido que entregas? (leading)
    d["qtn_align"] = (Q[:, 0] / (prom_q + EPS)) / ((TN[:, 0] / (prom_tn + EPS)) + EPS)

    d.insert(0, "corte", C); d.insert(0, "product_id", vivos)
    tablas.append(d.reset_index(drop=True))

out = pd.concat(tablas, ignore_index=True)
out = out.replace([np.inf, -np.inf], np.nan)
out.to_parquet(RUTA_EXP / "features_extra_regresor.parquet")
print(f"features_extra_regresor.parquet: {out.shape[0]} filas x {out.shape[1]-2} features")
print(f"  stk_*: {sum(c.startswith('stk_') for c in out.columns)} | "
      f"q_*: {sum(c.startswith('q_') for c in out.columns)} | "
      f"mix_*/qtn: {sum(c.startswith(('mix_','qtn')) for c in out.columns)}")

# test de cordura + auditoría
d812 = out[out["corte"] == 201812]
print(f"\nstk_cob mediana: {d812['stk_cob'].median():.3f} (spec ~0.68)")
print(f"mix_ratio mediana: {d812['mix_ratio'].median():.3f} (esperado ~1)")
print(f"qtn_align mediana: {d812['qtn_align'].median():.3f}")
dd = fb.armar_corte(201812, True).join(
    d812.set_index("product_id").drop(columns="corte"), how="left")
prob = [p for p in FeatureBuilder.auditar(dd) if any(k in p for k in ("stk_", "q_", "mix_", "qtn"))]
print("auditoría:", "✓ sin duplicados nuevos" if not prob else "\n  ".join(prob))
