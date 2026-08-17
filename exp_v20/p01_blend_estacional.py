"""exp_v20 p01: BLEND ESTACIONAL SUAVE (candidato de submit).

Idea (validada en exp_v17, backtest feb-2019: 0.1981 -> 0.1903):
  pred = (1 - w_naive)·ensamble + w_naive·naive_estacional
  w_naive(prod) = min(0.6, k · amplitud_estacional(cat3 del prod))
El naive estacional = ventas del MISMO mes del año pasado (para feb-2020 =
feb-2019, wide[201902]). La amplitud se mide EX-ANTE (febreros previos), así
el peso se desvanece solo en categorías planas y sube en Sopas/Caldo/Opaco.

Base del ensamble = OLS80/LGBM-v11-20 (exp_v11_ens_w80.csv, el 0.228 validado).
NO submitea solo; imprime el diff y deja el CSV. El submit lo hace el usuario
o el paso siguiente.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from features_lgbm import FeatureBuilder
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_productos
warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
EPS = 1e-6
K = 1.0          # pendiente peso-naive vs amplitud (mejor en backtest)
W_MAX = 0.6      # tope del peso naive

fb = FeatureBuilder(); wide = fb.wide; mm = fb.mes_menos
cat3 = cargar_productos().unique("product_id").to_pandas().set_index("product_id")["cat3"]
apre = cargar_apredecir().to_pandas()["product_id"]

# ---------- base ensamble (0.228) y naive estacional ----------
ens = pd.read_csv(RUTA_EXP.parent / "exp_v11/exp_v11_ens_w80.csv").set_index("product_id")["tn"]
naive = wide.reindex(apre)[201902]            # feb-2019 = mismo mes año pasado para feb-2020

# ---------- amplitud estacional de cada cat3, EX-ANTE (febreros previos) ----------
def factores_feb(cut_dic):
    cols = [mm(cut_dic, k) for k in range(12)]; tgt = mm(cut_dic, -2)
    prom = wide[cols].mean(axis=1); prom = prom[prom > EPS]
    return (wide.loc[prom.index, tgt] / prom).clip(0, 4).dropna()
f = pd.concat([factores_feb(201712), factores_feb(201812)])   # feb-2018 + feb-2019
glob = f.median()
fac_c3 = pd.DataFrame({"f": f.values, "c3": cat3.reindex(f.index).values}).dropna() \
           .groupby("c3")["f"].median()
ampl = cat3.reindex(apre).map(fac_c3).sub(glob).abs()          # |factor_cat3 - global|

# ---------- peso y blend ----------
w = (K * ampl).clip(0, W_MAX).fillna(0.0)
w[naive.isna().values] = 0.0                                   # sin feb-2019 -> ensamble puro
pred = (1 - w.values) * ens.reindex(apre).values + w.values * naive.fillna(0).values
pred = np.clip(pred, 0, None)
out = pd.DataFrame({"product_id": apre.values, "tn": pred}).sort_values("product_id")
assert len(out) == 780 and out["tn"].isna().sum() == 0 and (out["tn"] < 0).sum() == 0
out.to_csv(RUTA_EXP / "candidato_blend_estacional.csv", index=False)

# ---------- reporte ----------
diff = pd.Series(pred, index=apre) - ens.reindex(apre)
cambian = (diff.abs() > 0.01) & (w.values > 0)
print(f"peso naive: medio={w.mean():.3f}, >0 en {int((w>0).sum())} productos "
      f"(máx {w.max():.2f}) | k={K}, glob={glob:.2f}")
print(f"productos que cambian: {int(cambian.sum())} | delta total: {diff.sum():+.1f} tn "
      f"sobre {ens.reindex(apre).sum():.0f}")
top = pd.DataFrame({"cat3": cat3.reindex(apre), "w_naive": w.values,
                    "ens": ens.reindex(apre).values, "naive": naive.values,
                    "blend": pred, "delta": diff.values}, index=apre)
top = top[cambian.values].reindex(top[cambian.values]["delta"].abs().sort_values(ascending=False).index)
print("\nmayores cambios:")
print(top.head(12).round(1).to_string())
print("\ncandidato_blend_estacional.csv escrito.")
