"""exp_v10: set limpio v9 + stacking OLS reincorporado + categorías.

Spec del usuario:
  - base: FeatureBuilder canónico (76 features, post-auditoría v9)
  - + pred_ols_yoy / pred_ols_rec (el stacking que v9 quitó)
  - + categorías como features categóricas nativas de LGBM:
      cat1, cat2 completas; cat3 solo niveles con >10 productos en el
      train, el resto colapsa a "OTROS"
Optuna 30 trials (regla del usuario) + params fijos v8. 2 submits:
fijo y optuna-best. Backtest honesto multi-corte informado.
"""

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datos import RUTA_PROYECTO, cargar_apredecir, cargar_productos
from features_lgbm import FeatureBuilder

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
RUTA_EXP = Path(__file__).resolve().parent
COMPETENCIA = "labo-iii-2026-ba"
SEMILLA = 102191
EPS = 1e-6
N_TRIALS = 30
PARAMS_FIJOS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
                    min_child_samples=30, colsample_bytree=0.8,
                    subsample=0.8)

fb = FeatureBuilder()
wide = fb.wide
mes_menos = fb.mes_menos


def cargar_magicos() -> set[int]:
    nb = json.loads(
        (RUTA_PROYECTO / "src/Estadistica/z403_RegresionLineal_local.ipynb").read_text()
    )
    celda = next(
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "productos_magicos" in "".join(c["source"])
    )
    bloque = re.findall(r"productos_magicos = \[(.*?)\]", celda, flags=re.S)[-1]
    return set(int(x) for x in bloque.replace("\n", " ").split(","))


magicos = cargar_magicos()

# ---------- categorías: cat1, cat2, cat3 (>10 productos, sino OTROS) ----------
prod = cargar_productos().unique("product_id").to_pandas().set_index("product_id")
train_prods = wide.index[wide.notna().sum(axis=1) >= 12]
conteo_c3 = prod.loc[prod.index.isin(train_prods), "cat3"].value_counts()
c3_validas = set(conteo_c3[conteo_c3 > 10].index)
print(f"cat3: {len(conteo_c3)} niveles -> {len(c3_validas)} con >10 productos, resto OTROS")

CATS1 = sorted(set(prod["cat1"].dropna().unique()) | {"OTROS"})
CATS2 = sorted(set(prod["cat2"].dropna().unique()) | {"OTROS"})
CATS3 = sorted(c3_validas | {"OTROS"})


def agregar_categorias(d: pd.DataFrame) -> pd.DataFrame:
    p = prod.reindex(d.index)
    d["cat1"] = pd.Categorical(p["cat1"].fillna("OTROS"), categories=CATS1)
    d["cat2"] = pd.Categorical(p["cat2"].fillna("OTROS"), categories=CATS2)
    c3 = p["cat3"].where(p["cat3"].isin(c3_validas), "OTROS").fillna("OTROS")
    d["cat3"] = pd.Categorical(c3, categories=CATS3)
    return d


# ---------- stacking OLS (idéntico a v8: sin fuga) ----------
def ols_magica(cut_train: int):
    cols = [mes_menos(cut_train, k) for k in range(12)]
    obj = mes_menos(cut_train, -2)
    if cols[-1] not in wide.columns or obj not in wide.columns:
        return None
    d = wide.loc[wide.index.isin(magicos), cols + [obj]].dropna()
    if len(d) < 30:
        return None
    X = np.column_stack([np.ones(len(d)), d[cols].values])
    coef, *_ = np.linalg.lstsq(X, d[obj].values, rcond=None)
    return coef


def ols_aplicar(coef, cut_apply: int) -> pd.Series:
    cols = [mes_menos(cut_apply, k) for k in range(12)]
    d = wide[cols].dropna()
    X = np.column_stack([np.ones(len(d)), d.values])
    return pd.Series(X @ coef, index=d.index).clip(lower=0)


def agregar_stacking(d: pd.DataFrame, cut: int) -> pd.DataFrame:
    for nombre, lag_train in [("pred_ols_yoy", 12), ("pred_ols_rec", 2)]:
        coef = ols_magica(mes_menos(cut, lag_train))
        if coef is not None:
            p = ols_aplicar(coef, cut)
            d[nombre] = (p.reindex(d.index) / d["_prom"]).values
        else:
            d[nombre] = np.nan
    return d


def armar(cut: int, con_clase: bool) -> pd.DataFrame:
    d = fb.armar_corte(cut, con_clase)
    d = agregar_stacking(d, cut)
    d = agregar_categorias(d)
    return d


def fit(dtrain: pd.DataFrame, params: dict) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(objective="l1", random_state=SEMILLA, verbosity=-1,
                          subsample_freq=1, **params)
    m.fit(dtrain.drop(columns=["clase_ratio", "_prom"]), dtrain["clase_ratio"])
    return m


def wape_pred(m, dev: pd.DataFrame, real: pd.Series) -> float:
    ratio = pd.Series(m.predict(dev.drop(columns="_prom")), index=dev.index).clip(lower=0)
    pred = ratio * dev["_prom"]
    comun = pred.index.intersection(real.index)
    return float(np.abs(real[comun] - pred[comun]).sum() / real[comun].sum())


# ---------- folds precomputados ----------
FOLDS = [(201808, 1.0), (201810, 1.0), (201812, 2.0)]
folds = []
for cut_eval, peso in FOLDS:
    n_train = (pd.Period(str(cut_eval), freq="M") - pd.Period("201801", freq="M")).n - 1
    cortes = [mes_menos(cut_eval, k) for k in range(2, 2 + n_train)]
    dtr = pd.concat([armar(c, True) for c in cortes])
    dev = armar(cut_eval, False)
    real = wide[mes_menos(cut_eval, -2)].dropna()
    folds.append((dtr, dev, real, peso))

d_final = pd.concat([armar(mes_menos(201910, k), True) for k in range(22)])
d_fut = armar(201912, False)
print(f"train final: {len(d_final)} filas, {d_final.shape[1]-2} features")

# auditoría de las features nuevas
problemas = [p for p in FeatureBuilder.auditar(
    d_final.drop(columns=["cat1", "cat2", "cat3"]).sample(2000, random_state=1))
    if "anio_corte" not in p and "mes_corte" not in p]
print("auditoría:", "✓ limpia" if not problemas else problemas)


def evaluar(params: dict) -> float:
    total, pesos = 0.0, 0.0
    for dtr, dev, real, peso in folds:
        total += peso * wape_pred(fit(dtr, params), dev, real)
        pesos += peso
    return total / pesos


print(f"\nbacktest multi-corte params fijos: {evaluar(PARAMS_FIJOS):.4f}")

# ---------- optuna ----------
def objetivo(trial: optuna.Trial) -> float:
    return evaluar(dict(
        n_estimators=trial.suggest_int("n_estimators", 150, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 64),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
    ))


study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=SEMILLA))
study.optimize(objetivo, n_trials=N_TRIALS, show_progress_bar=False)
print(f"optuna best (multi-corte): {study.best_value:.4f}")
print(f"params: {study.best_params}")

# ---------- finales + FI + submits ----------
tb_prom = (
    fb.ventas.filter(pl.col("periodo").is_between(201901, 201912))
    .group_by("product_id").agg(pl.col("tn").mean())
)
apredecir = cargar_apredecir()
fi_textos = []


def submitear(nombre: str, params: dict) -> None:
    m = fit(d_final, params)
    booster = m.booster_
    imp = pd.DataFrame({
        "gain": booster.feature_importance(importance_type="gain"),
        "splits": booster.feature_importance(importance_type="split"),
    }, index=booster.feature_name())
    imp["gain_pct"] = (100 * imp["gain"] / imp["gain"].sum()).round(1)
    imp = imp.sort_values("gain", ascending=False)
    print(f"\n=== FI {nombre} (top 12) ===")
    print(imp.head(12).round({"gain": 0}).to_string())
    fi_textos.append(f"\n{'='*60}\n{nombre} | params: {params}\n{'='*60}\n"
                     + imp.round(1).to_string())

    ratio = pd.Series(m.predict(d_fut.drop(columns="_prom")), index=d_fut.index).clip(lower=0)
    pred = ratio * d_fut["_prom"]
    tb_reg = pl.DataFrame({"product_id": pred.index.to_list(), "tn_pred": pred.to_list()})
    tb_final = (
        apredecir.join(tb_prom, on="product_id", how="left")
        .join(tb_reg, on="product_id", how="left")
        .with_columns(pl.coalesce([pl.col("tn_pred"), pl.col("tn")]).alias("tn"))
        .select("product_id", "tn").sort("product_id")
    )
    assert tb_final.height == 780 and tb_final["tn"].null_count() == 0
    archivo = RUTA_EXP / f"exp_v10_{nombre}.csv"
    tb_final.write_csv(archivo)
    subprocess.run(
        [str(RUTA_PROYECTO / ".venv/bin/kaggle"), "competitions", "submit",
         "-c", COMPETENCIA, "-f", str(archivo),
         "-m", f"exp_v10 lgbm limpio+stackOLS+cats {nombre}"],
        check=True,
    )
    print(f"submit OK -> {archivo.name}")


submitear("fijo", PARAMS_FIJOS)
submitear("optuna", study.best_params)
(RUTA_EXP / "feature_importance.txt").write_text(
    "exp_v10 - FI (set limpio + stacking OLS + categorias)\n" + "\n".join(fi_textos))
print("\nFI guardado en feature_importance.txt")
