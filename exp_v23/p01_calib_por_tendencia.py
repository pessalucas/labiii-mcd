"""exp_v23: calibración del factor por TENDENCIA del producto (en vez de 1.03 global).

Idea: el LGBM comprime más a los productos que se mueven fuerte (v17), así el
factor de corrección óptimo depende de la pendiente de cada producto.
  factor(p) = clip(a + b * tendencia_std(p), lo, hi)
a, b se calibran en el backtest (dic18->feb19) minimizando WAPE. Se aplica a
la submission ganadora (ensamble 70/30). Se compara contra el 1.03 global.
"""

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import RUTA_PROYECTO
from features_lgbm import FeatureBuilder

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
COMPETENCIA = "labo-iii-2026-ba"
EPS = 1e-6

fb = FeatureBuilder()
wide = fb.wide
mes_menos = fb.mes_menos


def tendencia(cut: int) -> pd.Series:
    """Pendiente lineal (OLS) de tn normalizado por prom12, últimos 12 meses."""
    cols = [mes_menos(cut, k) for k in range(12)]
    d = wide[cols].dropna()
    prom = d.mean(axis=1)
    d = d[prom > EPS]; prom = prom[prom > EPS]
    Ln = (d.div(prom, axis=0)).values[:, ::-1]   # tiempo ascendente
    t = np.arange(12) - 5.5
    y = Ln - Ln.mean(axis=1, keepdims=True)
    pend = (y * t).sum(axis=1) / (t ** 2).sum()
    return pd.Series(pend, index=d.index)


# ---------- backtest: ensamble 70/30 (de v17) + tendencia en 201812 ----------
e = pl.read_csv("../exp_v17/errores_con_ols.csv" if (RUTA_EXP.parent / "exp_v17/errores_con_ols.csv").exists()
                else RUTA_EXP.parent / "exp_v17/errores_con_ols.csv").to_pandas().set_index("product_id")
real = e["real_feb19"]
pred_bt = 0.70 * e["pred_ols"] + 0.30 * e["pred_lgbm"]
tend_bt = tendencia(201812).reindex(pred_bt.index)
# estandarizar la tendencia con stats del backtest (reusar en el futuro)
mu, sd = tend_bt.mean(), tend_bt.std()
z_bt = ((tend_bt - mu) / (sd + EPS)).fillna(0).clip(-3, 3)


def wape(p):
    m = real.notna() & p.notna()
    return float((real[m] - p[m]).abs().sum() / real[m].sum())


# baseline: factor global 1.03
print(f"WAPE backtest sin calib:        {wape(pred_bt):.4f}")
print(f"WAPE backtest factor global 1.03:{wape(1.03 * pred_bt):.4f}")

# calibrar factor(p) = clip(a + b*z, 0.90, 1.20)
mejor = (1.03, 0.0, wape(1.03 * pred_bt))
for a in np.arange(0.99, 1.10, 0.01):
    for b in np.arange(-0.06, 0.09, 0.01):
        f = np.clip(a + b * z_bt, 0.90, 1.20)
        w = wape(f * pred_bt)
        if w < mejor[2]:
            mejor = (a, b, w)
a, b, w = mejor
print(f"\nmejor factor(tendencia): a={a:.2f}, b={b:+.2f} -> WAPE {w:.4f}")
print(f"  (b>0: sube el factor a los productos que CRECEN; b<0 al revés)")

# ---------- aplicar a la submission ganadora final ----------
tend_fin = tendencia(201912)
z_fin = ((tend_fin - mu) / (sd + EPS)).clip(-3, 3)
sub = pl.read_csv("../exp_v16/exp_v16_ens_w70.csv").to_pandas().set_index("product_id")["tn"]
z_al = z_fin.reindex(sub.index).fillna(0)
factor = np.clip(a + b * z_al, 0.90, 1.20)
out_var = (sub * factor).clip(lower=0)

outdf = pl.DataFrame({"product_id": out_var.index.to_list(), "tn": out_var.to_list()}).sort("product_id")
f1 = RUTA_EXP / "exp_v23_calib_tendencia.csv"; outdf.write_csv(f1)
print(f"\nfactor por producto: min {factor.min():.3f}, mediana {np.median(factor):.3f}, max {factor.max():.3f}")
subprocess.run([str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
                "-c", COMPETENCIA, "-f", str(f1),
                "-m", f"exp_v23 calib por tendencia a={a:.2f} b={b:+.2f}"], check=True)
print("submit OK -> exp_v23_calib_tendencia.csv")
