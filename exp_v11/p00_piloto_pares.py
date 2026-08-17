"""exp_v11 PILOTO: modelo a nivel cliente x producto con 10% de productos.

Objetivo: veredicto rápido (~15 min, CERO submits) sobre si el enfoque
de pares tiene señal. Backtest honesto: train cortes 201801..201810,
predecir feb-2019 por par, agregar a producto, WAPE sobre el sample
comparado contra la OLS mágica y el naif promedio EN LOS MISMOS productos.

Sample: 10% de los 780 (cada 10mo por ranking de tn 2019: estratificado
por tamaño). Features en 3 capas: par (lags, frecuencia, recencia),
cliente (volumen, tendencia, surtido), producto (s2, cv, clientela).
"""

import json
import re
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_sellin

warnings.filterwarnings("ignore")
SEMILLA = 102191
EPS = 1e-6
PERIODOS = [int(p.strftime("%Y%m")) for p in pd.period_range("2017-01", "2019-12", freq="M")]


def mes_menos(p: int, k: int) -> int:
    return int((pd.Period(str(p), freq="M") - k).strftime("%Y%m"))


sellin = cargar_sellin()
apredecir = cargar_apredecir()

# ---------- sample estratificado: cada 10mo por ranking de tn 2019 ----------
tn2019 = (sellin.filter(pl.col("periodo").is_between(201901, 201912))
          .group_by("product_id").agg(pl.col("tn").sum())
          .join(apredecir, on="product_id", how="inner")
          .sort("tn", descending=True))
sample = tn2019.with_row_index("rk")["rk", "product_id"].filter(
    pl.col("rk") % 10 == 0)["product_id"].to_list()
print(f"sample: {len(sample)} productos (del top al fondo, cada 10mo)")

# ---------- paneles ----------
def panel_desde(df: pl.DataFrame, idx_cols: list[str]) -> pd.DataFrame:
    """Wide por mes con ceros desde la primera compra de la entidad."""
    p = df.to_pandas().pivot_table(index=idx_cols, columns="periodo",
                                   values="tn", aggfunc="sum")
    p = p.reindex(columns=PERIODOS)
    first = p.apply(lambda r: r.first_valid_index(), axis=1)
    M = p.values
    cols = np.array(PERIODOS)
    for i in range(len(p)):
        M[i, cols >= first.iloc[i]] = np.nan_to_num(M[i, cols >= first.iloc[i]])
    return pd.DataFrame(M, index=p.index, columns=PERIODOS)


pares = (sellin.filter(pl.col("product_id").is_in(sample))
         .group_by("customer_id", "product_id", "periodo").agg(pl.col("tn").sum()))
w_par = panel_desde(pares, ["customer_id", "product_id"])
print(f"pares con historia: {len(w_par)}")

cli = sellin.group_by("customer_id", "periodo").agg(pl.col("tn").sum())
w_cli = panel_desde(cli, ["customer_id"])

prod_all = sellin.group_by("product_id", "periodo").agg(pl.col("tn").sum())
w_prod = panel_desde(prod_all, ["product_id"])

# n productos distintos por cliente-mes (surtido)
nprod = (sellin.group_by("customer_id", "periodo")
         .agg(pl.col("product_id").n_unique().alias("tn")))  # 'tn' para reusar panel
w_nprod = panel_desde(nprod, ["customer_id"])


def armar_corte_par(cut: int, con_clase: bool) -> pd.DataFrame:
    cols12 = [mes_menos(cut, k) for k in range(12)]
    L = w_par[cols12]
    activos = L.notna().all(axis=1) & (L.fillna(0).sum(axis=1) > 0)
    L = L[activos]
    d = pd.DataFrame(index=L.index)
    A = L.values

    # --- capa PAR ---
    for k in range(12):
        d[f"p_tn_{k}"] = A[:, k]
    compra = A > 0
    d["p_freq12"] = compra.sum(axis=1)
    d["p_freq3"] = compra[:, :3].sum(axis=1)
    # recencia: primer k con compra (0 = compró este mes)
    d["p_recencia"] = np.where(compra.any(axis=1), compra.argmax(axis=1), 12)
    d["p_mean_compra"] = A.sum(axis=1) / (compra.sum(axis=1) + EPS)
    d["p_feb_prev"] = A[:, 10]
    # la maquinaria v8/v9 aplicada a la serie del par:
    # sumas / medias moviles ancladas
    for k in (2, 3, 6):
        d[f"p_s{k}"] = A[:, :k].mean(axis=1)
    # diffs consecutivos recientes
    d["p_diff_0_1"] = A[:, 0] - A[:, 1]
    d["p_diff_1_2"] = A[:, 1] - A[:, 2]
    # rolling min/max/mediana (3/6/12)
    for k in (3, 6, 12):
        d[f"p_roll_max_{k}"] = A[:, :k].max(axis=1)
        d[f"p_roll_med_{k}"] = np.median(A[:, :k], axis=1)
    d["p_roll_min_12"] = A.min(axis=1)
    # posicion actual vs extremos y promedio (dmin/dmax/dprom, ventana 12)
    d["p_dmax_12"] = A[:, 0] - A.max(axis=1)
    d["p_dmin_12"] = A[:, 0] - A.min(axis=1)
    d["p_dprom_12"] = A[:, 0] - A.mean(axis=1)
    # ratio "mes actual vs mejor mes" (cuanto del potencial esta comprando)
    d["p_vs_max"] = A[:, 0] / (A.max(axis=1) + EPS)
    # media movil desplazada y aceleracion
    d["p_ma3_lag3"] = A[:, 3:6].mean(axis=1)
    d["p_acel_ma3"] = A[:, :3].mean(axis=1) - A[:, 3:6].mean(axis=1)
    # familia CEROS/intermitencia (freq_k ya es el complemento del conteo de ceros)
    d["p_freq6"] = compra[:, :6].sum(axis=1)
    # racha maxima de ceros consecutivos en la ventana de 12
    ceros = ~compra
    racha = np.zeros(len(A)); actual = np.zeros(len(A))
    for k in range(12):
        actual = np.where(ceros[:, k], actual + 1, 0)
        racha = np.maximum(racha, actual)
    d["p_racha_max_ceros"] = racha
    # gap promedio entre compras (12 / cantidad de compras; 12 si no compro)
    d["p_gap_promedio"] = 12.0 / (compra.sum(axis=1) + EPS)
    # edad del par: meses desde su primera compra historica hasta el corte
    primeras = w_par.apply(lambda r: r.first_valid_index(), axis=1).loc[L.index]
    d["p_edad_par"] = [
        (pd.Period(str(cut), freq="M") - pd.Period(str(int(x)), freq="M")).n + 1
        for x in primeras
    ]

    # --- capa CLIENTE ---
    cids = d.index.get_level_values(0)
    C = w_cli.loc[cids, cols12].values
    d["c_log_tn"] = np.log1p(np.nansum(C, axis=1))
    d["c_idx"] = np.nanmean(C[:, :2], axis=1) / (np.nanmean(C, axis=1) + EPS)
    NP = w_nprod.loc[cids, cols12].values
    d["c_nprod_idx"] = NP[:, 0] / (np.nanmean(NP, axis=1) + EPS)

    # --- capa PRODUCTO ---
    pids = d.index.get_level_values(1)
    P = w_prod.loc[pids, cols12].values
    pprom = np.nanmean(P, axis=1)
    d["q_s2"] = np.nanmean(P[:, :2], axis=1) / (pprom + EPS)
    d["q_cv"] = np.nanstd(P, axis=1) / (pprom + EPS)
    d["q_log_prom"] = np.log1p(pprom)

    d["mes_corte"] = cut % 100
    if con_clase:
        obj = mes_menos(cut, -2)
        if obj not in w_par.columns:
            return pd.DataFrame()
        d["clase"] = w_par.loc[L.index, obj].values
        d = d.dropna(subset=["clase"])
    return d


# ---------- train (cortes 201801..201810) y eval @201812 -> feb2019 ----------
cortes = [mes_menos(201810, k) for k in range(10)]
dtr = pd.concat([armar_corte_par(c, True) for c in cortes])
print(f"filas de entrenamiento (pares x cortes): {len(dtr)}")

dev = armar_corte_par(201812, False)
resultados = {}
modelos = {}
for objetivo in ("l1", "l2", "tweedie"):
    m = lgb.LGBMRegressor(objective=objetivo, random_state=SEMILLA, verbosity=-1,
                          n_estimators=400, learning_rate=0.03, num_leaves=31,
                          min_child_samples=30, colsample_bytree=0.8,
                          subsample=0.8, subsample_freq=1)
    m.fit(dtr.drop(columns="clase"), dtr["clase"])
    modelos[objetivo] = m
    pred_par = pd.Series(m.predict(dev), index=dev.index).clip(lower=0)
    resultados[objetivo] = pred_par.groupby(level=1).sum()
print(f"pares evaluados: {len(dev)}")
pred_prod = resultados["l1"]
m = modelos["tweedie"]

# reales feb-2019 del sample
real = w_prod.loc[w_prod.index.isin(sample), 201902].dropna()

# fallback promedio para sampleados sin pred de pares
prom12 = w_prod.loc[real.index, [mes_menos(201812, k) for k in range(12)]].mean(axis=1)
pred_full = pred_prod.reindex(real.index).fillna(prom12)


def wape(p): return float(np.abs(real - p.reindex(real.index)).sum() / real.sum())


# ---------- baselines EN EL MISMO subset ----------
# OLS mágica calendario entrenada @201712, fallback promedio
nb = json.loads((RUTA_PROYECTO / "src/Estadistica/z403_RegresionLineal_local.ipynb").read_text())
celda = next("".join(c["source"]) for c in nb["cells"]
             if c["cell_type"] == "code" and "productos_magicos" in "".join(c["source"]))
magicos = set(int(x) for x in re.findall(r"productos_magicos = \[(.*?)\]",
                                          celda, flags=re.S)[-1].replace("\n", " ").split(","))
cols_tr = [mes_menos(201712, k) for k in range(12)]
dm = w_prod.loc[w_prod.index.isin(magicos), cols_tr + [201802]].dropna()
X = np.column_stack([np.ones(len(dm)), dm[cols_tr].values])
coef, *_ = np.linalg.lstsq(X, dm[201802].values, rcond=None)
cols_ap = [mes_menos(201812, k) for k in range(12)]
da = w_prod.loc[real.index, cols_ap].dropna()
ols_pred = pd.Series(
    np.column_stack([np.ones(len(da)), da.values]) @ coef, index=da.index
).clip(lower=0).reindex(real.index).fillna(prom12)

print("\n=== VEREDICTO (WAPE feb-2019, mismos productos del sample) ===")
total_real = real.sum()
for objetivo, pp in resultados.items():
    full = pp.reindex(real.index).fillna(prom12)
    print(f"  PARES objetivo={objetivo:>7}: WAPE {wape(full):.4f} | "
          f"total pred {full.sum():.0f} tn vs real {total_real:.0f} tn "
          f"(sesgo {100*(full.sum()/total_real-1):+.0f}%)")
print(f"  OLS mágica (baseline a batir):          {wape(ols_pred):.4f}")
print(f"  naif promedio 12m:                      {wape(prom12):.4f}")

imp = pd.Series(m.booster_.feature_importance(importance_type="gain"),
                index=m.booster_.feature_name()).sort_values(ascending=False)
print("\nFI top 12:")
print((100 * imp / imp.sum()).round(1).head(12).to_string())
