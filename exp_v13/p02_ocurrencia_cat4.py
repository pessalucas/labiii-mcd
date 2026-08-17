"""exp_v13 - paso 2: clasificador de OCURRENCIA enriquecido con SUSTITUCIÓN cat4.

Al clasificador de v12 (P(compra) del par cliente-producto) le suma
features del comportamiento del cliente a nivel cat4, para capturar si
el cliente 'sigue en la familia' aunque cambió de producto:
  cli4_freq12/recencia : actividad del cliente en el cat4 (cualquier producto)
  cli4_activo_t0       : compró algo del cat4 este mes
  share_pc4            : fidelidad del cliente a ESTE producto dentro del cat4
  sustituto_3m         : compró el cat4 en 3m pero NO este producto (migró)
Regenera features_ocurrencia_v13.parquet (occ_* a nivel producto).
2-fold temporal sin fuga + modelo full. Guarda artefactos.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import cargar_sellin

warnings.filterwarnings("ignore")
RUTA_EXP = Path(__file__).resolve().parent
RUTA_PRUEBAS = RUTA_EXP.parent
SEMILLA = 102191
EPS = 1e-6
PERIODOS = [int(p.strftime("%Y%m")) for p in pd.period_range("2017-01", "2019-12", freq="M")]
CORTES = [int(p.strftime("%Y%m")) for p in pd.period_range("2017-12", "2019-10", freq="M")]
CORTE_FUT = 201912
PARAMS = dict(objective="binary", random_state=SEMILLA, verbosity=-1,
              n_estimators=300, learning_rate=0.05, num_leaves=48,
              min_child_samples=100, colsample_bytree=0.8, subsample=0.8, subsample_freq=1)


def mes_menos(p: int, k: int) -> int:
    return int((pd.Period(str(p), freq="M") - k).strftime("%Y%m"))


sellin = cargar_sellin()
cat4map = pd.read_parquet(RUTA_PRUEBAS / "cat4_mapping.parquet")
cat4_de = dict(zip(cat4map["product_id"], cat4map["cat4"]))
sellin = sellin.with_columns(
    pl.col("product_id").replace_strict(cat4_de, default="SIN").alias("cat4"))


def panel(df, idx, val="tn"):
    p = (df.to_pandas().pivot_table(index=idx, columns="periodo", values=val, aggfunc="sum")
         .reindex(columns=PERIODOS))
    first = p.apply(lambda r: r.first_valid_index(), axis=1)
    M = p.values; ca = np.array(PERIODOS)
    for i in range(len(p)):
        M[i, ca >= first.iloc[i]] = np.nan_to_num(M[i, ca >= first.iloc[i]])
    return pd.DataFrame(M, index=p.index, columns=PERIODOS)


w_par = panel(sellin.group_by("customer_id", "product_id", "periodo").agg(pl.col("tn").sum()),
              ["customer_id", "product_id"])
w_pc4 = panel(sellin.group_by("customer_id", "cat4", "periodo").agg(pl.col("tn").sum()),
              ["customer_id", "cat4"])       # cliente x cat4 (sustitución)
w_cli = panel(sellin.group_by("customer_id", "periodo").agg(pl.col("tn").sum()), ["customer_id"])
w_prod = panel(sellin.group_by("product_id", "periodo").agg(pl.col("tn").sum()), ["product_id"])
print(f"paneles: par {w_par.shape[0]:,} | cliente×cat4 {w_pc4.shape[0]:,}")


def armar_corte(cut, con_target):
    cols12 = [mes_menos(cut, k) for k in range(12)]
    L = w_par[cols12]
    act = L.notna().all(1) & (L.fillna(0).sum(1) > 0)
    L = L[act]; A = L.values; compra = A > 0
    d = pd.DataFrame(index=L.index)
    d["recencia"] = np.where(compra.any(1), compra.argmax(1), 12)
    d["freq3"] = compra[:, :3].sum(1); d["freq6"] = compra[:, :6].sum(1)
    d["freq12"] = compra[:, :12].sum(1)
    d["gap_prom"] = 12.0 / (compra.sum(1) + EPS)
    ceros = ~compra; racha = np.zeros(len(A)); a = np.zeros(len(A))
    for k in range(12):
        a = np.where(ceros[:, k], a + 1, 0); racha = np.maximum(racha, a)
    d["racha_ceros"] = racha
    d["compro_t0"] = compra[:, 0].astype(int); d["compro_t1"] = compra[:, 1].astype(int)
    d["monto_mean"] = A.sum(1) / (compra.sum(1) + EPS)
    d["monto_s3"] = A[:, :3].mean(1)
    d["mes_obj"] = mes_menos(cut, -2) % 100
    cids = d.index.get_level_values(0); pids = d.index.get_level_values(1)
    C = w_cli.loc[cids, cols12].values
    d["c_log_tn"] = np.log1p(np.nansum(C, 1))
    d["c_idx"] = np.nanmean(C[:, :2], 1) / (np.nanmean(C, 1) + EPS)
    d["q_log_prom"] = np.log1p(np.nanmean(w_prod.loc[pids, cols12].values, 1))

    # ---- SUSTITUCIÓN cat4: comportamiento del cliente en la familia ----
    c4s = [cat4_de.get(p, "SIN") for p in pids]
    idx_pc4 = list(zip(cids, c4s))
    PC4 = w_pc4.reindex(idx_pc4)[cols12].values      # cliente x su cat4
    comp4 = PC4 > 0
    d["cli4_freq12"] = comp4[:, :12].sum(1)
    d["cli4_recencia"] = np.where(comp4.any(1), comp4.argmax(1), 12)
    d["cli4_activo_t0"] = comp4[:, 0].astype(int)
    # fidelidad: cuánto del cat4 que compra el cliente es ESTE producto
    d["share_pc4"] = A[:, :12].sum(1) / (np.nansum(PC4[:, :12], 1) + EPS)
    # sustituto: compró el cat4 en 3m pero NO este producto (migró a otro)
    d["sustituto_3m"] = ((comp4[:, :3].any(1)) & (~compra[:, :3].any(1))).astype(int)
    # ratio de tendencia del cliente en el cat4
    d["cli4_idx"] = np.nanmean(PC4[:, :2], 1) / (np.nanmean(PC4, 1) + EPS)

    if con_target:
        obj = mes_menos(cut, -2)
        d["y"] = (w_par.loc[L.index, obj].fillna(0).values > 0).astype(int) if obj in w_par.columns else np.nan
        d = d.dropna(subset=["y"])
    return d


FEATURES = ["recencia", "freq3", "freq6", "freq12", "gap_prom", "racha_ceros",
            "compro_t0", "compro_t1", "monto_mean", "monto_s3", "mes_obj",
            "c_log_tn", "c_idx", "q_log_prom",
            "cli4_freq12", "cli4_recencia", "cli4_activo_t0", "share_pc4",
            "sustituto_3m", "cli4_idx"]


def fit(cortes):
    d = pd.concat([armar_corte(c, True) for c in cortes])
    m = lgb.LGBMClassifier(**PARAMS); m.fit(d[FEATURES], d["y"]); return m


early = [c for c in CORTES if mes_menos(c, -2) <= 201812]
late = [c for c in CORTES if mes_menos(c, -2) >= 201901]
m_early, m_late, m_full = fit(early), fit(late), fit(CORTES)
dev = pd.concat([armar_corte(c, True) for c in late])
auc = roc_auc_score(dev["y"], m_early.predict_proba(dev[FEATURES])[:, 1])
print(f"AUC (train early->eval late): {auc:.4f} (v12 sin cat4 fue 0.787)")


def feat_producto(cut, modelo):
    d = armar_corte(cut, False)
    p = modelo.predict_proba(d[FEATURES])[:, 1]
    tmp = pd.DataFrame({"product_id": d.index.get_level_values(1),
                        "p": p, "dem": p * d["monto_mean"].values})
    agg = tmp.groupby("product_id").agg(occ_n_esperados=("p", "sum"),
                                        occ_demanda_esp=("dem", "sum"))
    prom12 = w_prod.loc[agg.index, [mes_menos(cut, k) for k in range(12)]].mean(axis=1)
    agg["occ_demanda_norm"] = agg["occ_demanda_esp"] / (prom12 + EPS)
    agg["corte"] = cut
    return agg.reset_index()


tablas = [feat_producto(c, m_late if c in early else m_early) for c in CORTES]
tablas.append(feat_producto(CORTE_FUT, m_full))
feat = pd.concat(tablas, ignore_index=True)
feat.to_parquet(RUTA_EXP / "features_ocurrencia_v13.parquet")

chk = feat_producto(201812, m_early).set_index("product_id")
real = w_prod[201902].reindex(chk.index)
corr = np.corrcoef(chk["occ_demanda_esp"], real.fillna(0))[0, 1]
print(f"corr(occ_demanda_esp, real): {corr:.3f}")

m_full.booster_.save_model(str(RUTA_EXP / "modelo_ocurrencia_v13.txt"))
joblib.dump(m_full, RUTA_EXP / "modelo_ocurrencia_v13.pkl")
imp = pd.Series(m_full.booster_.feature_importance("gain"),
                index=m_full.booster_.feature_name()).sort_values(ascending=False)
imp_pct = (100 * imp / imp.sum()).round(1)
(RUTA_EXP / "fi_ocurrencia_v13.txt").write_text(imp_pct.to_string())
print("\nFI ocurrencia v13 (top 12):\n" + imp_pct.head(12).to_string())
cat4_gain = imp_pct[imp_pct.index.str.startswith(("cli4", "share_pc4", "sustituto"))].sum()
print(f"\ngain de las features de sustitución cat4: {cat4_gain:.1f}%")
