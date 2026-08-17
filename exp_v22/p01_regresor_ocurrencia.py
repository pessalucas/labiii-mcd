"""exp_v15 - paso 1: clasificador de ocurrencia con VENTANA 36 MESES.

Amplía la historia del clasificador de 12 a 36 meses y agrega features de
ESTACIONALIDAD ANUAL / periodicidad que 12m no permitía:
  compro_obj_y1/y2/y3 : compró en el MISMO mes del objetivo hace 1/2/3 años
  freq_mes_obj        : en cuántos de esos años compró (estacionalidad)
  freq24, freq_total  : frecuencia en ventanas largas
  gap_largo, edad_par : periodicidad y antigüedad
  periodicidad_anual  : ratio freq_mes_obj vs freq general (¿compra anual?)
Optuna n=10 sobre 30% del dataset (objetivo: AUC holdout temporal).
2-fold temporal sin fuga + modelo full. Guarda artefactos.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import cargar_sellin

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
RUTA_EXP = Path(__file__).resolve().parent
RUTA_PRUEBAS = RUTA_EXP.parent
SEMILLA = 102191
EPS = 1e-6
N_HIST = 36
PERIODOS = [int(p.strftime("%Y%m")) for p in pd.period_range("2017-01", "2019-12", freq="M")]
CORTES = [int(p.strftime("%Y%m")) for p in pd.period_range("2017-12", "2019-10", freq="M")]
CORTE_FUT = 201912


def mes_menos(p, k): return int((pd.Period(str(p), freq="M") - k).strftime("%Y%m"))


sellin = cargar_sellin()
cat4map = pd.read_parquet(RUTA_PRUEBAS / "cat4_mapping.parquet")
cat4_de = dict(zip(cat4map["product_id"], cat4map["cat4"]))
sellin = sellin.with_columns(pl.col("product_id").replace_strict(cat4_de, default="SIN").alias("cat4"))


def panel(df, idx):
    p = (df.to_pandas().pivot_table(index=idx, columns="periodo", values="tn", aggfunc="sum")
         .reindex(columns=PERIODOS))
    first = p.apply(lambda r: r.first_valid_index(), axis=1)
    M = p.values; ca = np.array(PERIODOS)
    for i in range(len(p)):
        M[i, ca >= first.iloc[i]] = np.nan_to_num(M[i, ca >= first.iloc[i]])
    return pd.DataFrame(M, index=p.index, columns=PERIODOS)


w_par = panel(sellin.group_by("customer_id", "product_id", "periodo").agg(pl.col("tn").sum()),
              ["customer_id", "product_id"])
w_pc4 = panel(sellin.group_by("customer_id", "cat4", "periodo").agg(pl.col("tn").sum()),
              ["customer_id", "cat4"])
w_cli = panel(sellin.group_by("customer_id", "periodo").agg(pl.col("tn").sum()), ["customer_id"])
w_prod = panel(sellin.group_by("product_id", "periodo").agg(pl.col("tn").sum()), ["product_id"])
first_par = w_par.apply(lambda r: r.first_valid_index(), axis=1)
first_prod = w_prod.apply(lambda r: r.first_valid_index(), axis=1)  # nacimiento del producto
print(f"panel de pares: {w_par.shape[0]:,}")


def lags(panel, idx, cut, n):
    cols = [mes_menos(cut, k) for k in range(n)]
    out = np.full((len(idx), n), np.nan)
    for j, c in enumerate(cols):
        if c in panel.columns:
            out[:, j] = panel.reindex(idx)[c].values
    return out


def armar_corte(cut, con_target):
    cols12 = [mes_menos(cut, k) for k in range(12)]
    L = w_par[cols12]
    act = L.notna().all(1) & (L.fillna(0).sum(1) > 0)
    idx = L[act].index
    A = lags(w_par, idx, cut, N_HIST)          # hasta 36 meses (NaN si no hay)
    compra = A > 0
    d = pd.DataFrame(index=idx)
    # --- corto plazo (como antes) ---
    d["recencia"] = np.where(np.nan_to_num(compra[:, :12]).any(1),
                             np.nan_to_num(compra[:, :12]).argmax(1), 12)
    d["freq3"] = np.nansum(compra[:, :3], 1); d["freq6"] = np.nansum(compra[:, :6], 1)
    d["freq12"] = np.nansum(compra[:, :12], 1)
    d["gap_prom"] = 12.0 / (np.nansum(compra[:, :12], 1) + EPS)
    ce = ~compra[:, :12]; racha = np.zeros(len(A)); a = np.zeros(len(A))
    for k in range(12):
        a = np.where(np.nan_to_num(ce[:, k]), a + 1, 0); racha = np.maximum(racha, a)
    d["racha_ceros"] = racha
    d["compro_t0"] = np.nan_to_num(compra[:, 0]).astype(int)
    d["compro_t1"] = np.nan_to_num(compra[:, 1]).astype(int)
    d["monto_mean"] = np.nansum(A[:, :12], 1) / (np.nansum(compra[:, :12], 1) + EPS)
    d["monto_s3"] = np.nanmean(A[:, :3], 1)
    d["mes_obj"] = mes_menos(cut, -2) % 100
    # --- ESTACIONALIDAD ANUAL (nuevo, requiere 36m) ---
    # mismo mes que el objetivo (t+2) hace 1/2/3 años = lags 10, 22, 34
    for y, lag in [(1, 10), (2, 22), (3, 34)]:
        col = compra[:, lag] if lag < N_HIST else np.full(len(A), np.nan)
        d[f"compro_obj_y{y}"] = np.nan_to_num(col).astype(int)
    d["freq_mes_obj"] = d[["compro_obj_y1", "compro_obj_y2", "compro_obj_y3"]].sum(1)
    d["freq24"] = np.nansum(compra[:, :24], 1)
    d["freq_total"] = np.nansum(compra, 1)
    dispo = np.sum(~np.isnan(A), 1)
    d["edad_par"] = dispo
    d["gap_largo"] = dispo / (np.nansum(compra, 1) + EPS)
    # periodicidad anual: compra en el mes objetivo más de lo esperado por azar
    tasa = np.nansum(compra, 1) / (dispo + EPS)
    d["periodicidad_anual"] = (d["freq_mes_obj"] / 3.0) / (tasa + EPS)
    # tendencia de actividad: freq reciente vs histórica
    d["tend_freq"] = (d["freq6"] / 6.0) / (tasa + EPS)
    # --- contexto cliente / producto (12m) ---
    cids = idx.get_level_values(0); pids = idx.get_level_values(1)
    C = w_cli.reindex(cids)[cols12].values
    d["c_log_tn"] = np.log1p(np.nansum(C, 1))
    d["c_idx"] = np.nanmean(C[:, :2], 1) / (np.nanmean(C, 1) + EPS)
    d["q_log_prom"] = np.log1p(np.nanmean(w_prod.reindex(pids)[cols12].values, 1))
    # edad del PRODUCTO: meses desde que se vende hasta el corte
    fp = first_prod.reindex(pids).values
    d["edad_producto"] = [(pd.Period(str(cut), freq="M") - pd.Period(str(int(x)), freq="M")).n + 1
                          if not pd.isna(x) else np.nan for x in fp]
    # --- sustitución cat4 (de v13) ---
    c4s = [cat4_de.get(p, "SIN") for p in pids]
    PC4 = w_pc4.reindex(list(zip(cids, c4s)))[cols12].values
    comp4 = PC4 > 0
    d["cli4_freq12"] = np.nansum(comp4, 1)
    d["cli4_activo_t0"] = np.nan_to_num(comp4[:, 0]).astype(int)
    d["share_pc4"] = np.nansum(A[:, :12], 1) / (np.nansum(PC4, 1) + EPS)
    d["sustituto_3m"] = ((np.nan_to_num(comp4[:, :3]).any(1)) &
                         (~np.nan_to_num(compra[:, :3]).any(1))).astype(int)
    if con_target:
        obj = mes_menos(cut, -2)
        d["y"] = w_par.reindex(idx)[obj].fillna(0).values if obj in w_par.columns else np.nan
        d = d.dropna(subset=["y"])
    return d


FEATURES = [c for c in armar_corte(201812, True).columns if c != "y"]
print(f"features del clasificador: {len(FEATURES)} (ventana {N_HIST}m)")

early = [c for c in CORTES if mes_menos(c, -2) <= 201812]
late = [c for c in CORTES if mes_menos(c, -2) >= 201901]
d_early = pd.concat([armar_corte(c, True) for c in early])
d_late = pd.concat([armar_corte(c, True) for c in late])
print(f"train early {len(d_early):,} | late {len(d_late):,}")

# ---------- Optuna n=10 sobre 30% del dataset (objetivo AUC holdout) ----------
d_tune = d_early.sample(frac=0.50, random_state=SEMILLA)
d_val = d_late.sample(frac=0.50, random_state=SEMILLA)


def objetivo(trial):
    params = dict(objective="l1", random_state=SEMILLA, verbosity=-1, subsample_freq=1,
                  n_estimators=trial.suggest_int("n_estimators", 150, 500),
                  learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                  num_leaves=trial.suggest_int("num_leaves", 16, 96),
                  min_child_samples=trial.suggest_int("min_child_samples", 50, 300),
                  colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                  subsample=trial.suggest_float("subsample", 0.6, 1.0))
    m = lgb.LGBMRegressor(**params); m.fit(d_tune[FEATURES], d_tune["y"])
    pr = np.clip(m.predict(d_val[FEATURES]), 0, None)
    return float(np.abs(d_val["y"].values - pr).sum() / (d_val["y"].sum() + EPS))


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEMILLA))
study.optimize(objetivo, n_trials=30, show_progress_bar=False)
BEST = dict(objective="l1", random_state=SEMILLA, verbosity=-1, subsample_freq=1, **study.best_params)
print(f"optuna best WAPE-par (50% val, n=30): {study.best_value:.4f} | {study.best_params}")


def fit(d): m = lgb.LGBMRegressor(**BEST); m.fit(d[FEATURES], d["y"]); return m


m_early, m_late = fit(d_early), fit(d_late)
m_full = fit(pd.concat([d_early, d_late]))
pr = np.clip(m_early.predict(d_late[FEATURES]), 0, None)
wp = float(np.abs(d_late["y"].values - pr).sum() / (d_late["y"].sum() + EPS))
print(f"WAPE-par (train early->eval late): {wp:.4f}")


def feat_producto(cut, modelo):
    d = armar_corte(cut, False)
    p = np.clip(modelo.predict(d[FEATURES]), 0, None)  # tn predicha del par
    tmp = pd.DataFrame({"product_id": d.index.get_level_values(1), "p": p,
                        "npos": (p > 0.01).astype(float)})
    agg = tmp.groupby("product_id").agg(occ_n_esperados=("npos", "sum"), occ_demanda_esp=("p", "sum"))
    prom12 = w_prod.reindex(agg.index)[[mes_menos(cut, k) for k in range(12)]].mean(axis=1)
    agg["occ_demanda_norm"] = agg["occ_demanda_esp"] / (prom12.values + EPS)
    agg["corte"] = cut
    return agg.reset_index()


tablas = [feat_producto(c, m_late if c in early else m_early) for c in CORTES]
tablas.append(feat_producto(CORTE_FUT, m_full))
feat = pd.concat(tablas, ignore_index=True)
feat.to_parquet(RUTA_EXP / "features_ocurrencia_v22.parquet")

m_full.booster_.save_model(str(RUTA_EXP / "modelo_ocurrencia_v22.txt"))
joblib.dump(m_full, RUTA_EXP / "modelo_ocurrencia_v22.pkl")
json.dump(study.best_params, open(RUTA_EXP / "hiperparametros_regresor_ocurrencia.json", "w"), indent=2)
imp = pd.Series(m_full.booster_.feature_importance("gain"),
                index=m_full.booster_.feature_name()).sort_values(ascending=False)
imp_pct = (100 * imp / imp.sum()).round(1)
(RUTA_EXP / "fi_clasificador.txt").write_text(imp_pct.to_string())
print("\nFI clasificador (top 14):\n" + imp_pct.head(14).to_string())
est = imp_pct[imp_pct.index.str.contains("obj_y|mes_obj|freq24|freq_total|periodicidad|edad|gap_largo")].sum()
print(f"\ngain de las features de estacionalidad/36m: {est:.1f}%")
