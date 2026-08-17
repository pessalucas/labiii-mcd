"""Builder de features CANÓNICO para los LGBM del proyecto (post-auditoría v9).

Correcciones sobre el set de exp_v9 (7 degeneraciones algebraicas
detectadas por auditoría automática, ver exp_v9/summary.txt):
  ELIMINADAS (duplicados/constantes exactos):
    s12              == 1 por construcción; solo filtraba el tamaño del
                        producto vía epsilon (fuga de escala accidental)
    dprom_12         == tn_0 - 1 (duplicado exacto de tn_0)
    pend_3           == diff_0_2 / 2 (proporcional exacto)
    cruce_ma3_ma12   == s3 (ma12 es 1 por construcción)
    r_sem            == transformación monótona de s6 (mismo ranking:
                        ambas son funciones de semestre1/semestre2)
    ma6_lag6         == 2 - s6 (las dos mitades promedian el año)
    cruce_ma6        == s6/(2-s6), transformación monótona de s6
    roll_std_12      == cv (misma fórmula)
  AGREGADA:
    log_prom         la escala del producto, EXPLÍCITA (antes entraba de
                     contrabando por el epsilon de s12)

Uso:
    from features_lgbm import FeatureBuilder
    fb = FeatureBuilder()                  # carga sell-in y arma paneles
    d  = fb.armar_corte(201812, con_clase=True)   # features + clase_ratio + _prom
    fb.auditar(d)                          # chequeo de degeneraciones
"""

import warnings

import numpy as np
import pandas as pd
import polars as pl

from datos import cargar_sellin

warnings.filterwarnings("ignore")
EPS = 1e-6
N_LAGS = 24
LAGS12 = [f"tn_{k}" for k in range(12)]


class FeatureBuilder:
    def __init__(self) -> None:
        sellin = cargar_sellin()
        self.ventas = sellin.group_by("product_id", "periodo").agg(
            pl.col("tn").sum(),
            pl.col("customer_id").n_unique().alias("ncli"),
        )
        self.periodos = [
            int(p.strftime("%Y%m")) for p in pd.period_range("2017-01", "2019-12", freq="M")
        ]
        self.wide = self._panel("tn")
        self.wide_ncli = self._panel("ncli")
        self.primer_mes = self.wide.apply(lambda r: r.first_valid_index(), axis=1)

    def _panel(self, col: str) -> pd.DataFrame:
        w = (self.ventas.to_pandas()
             .pivot(index="product_id", columns="periodo", values=col)
             .reindex(columns=self.periodos))
        pm = w.apply(lambda r: r.first_valid_index(), axis=1)
        for pid in w.index:
            w.loc[pid, w.columns >= pm[pid]] = w.loc[pid, w.columns >= pm[pid]].fillna(0.0)
        return w

    @staticmethod
    def mes_menos(p: int, k: int) -> int:
        return int((pd.Period(str(p), freq="M") - k).strftime("%Y%m"))

    @staticmethod
    def _slope(M: np.ndarray) -> np.ndarray:
        k = M.shape[1]
        t = np.arange(k) - (k - 1) / 2
        y = M - np.nanmean(M, axis=1, keepdims=True)
        s = (y * t).sum(axis=1) / (t ** 2).sum()
        s[np.isnan(M).any(axis=1)] = np.nan
        return s

    def armar_corte(self, cut: int, con_clase: bool) -> pd.DataFrame:
        wide, mes_menos, slope = self.wide, self.mes_menos, self._slope
        cols24 = [mes_menos(cut, k) for k in range(N_LAGS)]
        cols_exist = [c for c in cols24 if c in wide.columns]
        d = wide[cols_exist].copy()
        d.columns = [f"tn_{k}" for k in range(len(cols_exist))]
        for k in range(len(cols_exist), N_LAGS):
            d[f"tn_{k}"] = np.nan
        d = d.dropna(subset=LAGS12)
        prom = d[LAGS12].mean(axis=1)
        d = d[prom > EPS]
        prom = prom[prom > EPS]
        if con_clase:
            obj = mes_menos(cut, -2)
            if obj not in wide.columns:
                return pd.DataFrame()
            d["clase_ratio"] = wide.loc[d.index, obj] / prom
            d = d.dropna(subset=["clase_ratio"])
            prom = prom.loc[d.index]

        L = d[[f"tn_{k}" for k in range(N_LAGS)]].values
        Ln = L / (prom.values[:, None] + EPS)

        # escala EXPLÍCITA (reemplaza la fuga accidental de s12)
        d["log_prom"] = np.log1p(prom)

        # ratios recientes
        d["r_0_1"] = ((L[:, 0] + EPS) / (L[:, 1] + EPS)).clip(0, 10)
        d["r_1_2"] = ((L[:, 1] + EPS) / (L[:, 2] + EPS)).clip(0, 10)
        d["r_02_34"] = ((L[:, 0] + L[:, 2] + EPS) / (L[:, 3] + L[:, 4] + EPS)).clip(0, 10)
        d["r_tri"] = ((L[:, :3].sum(1) + EPS) / (L[:, 3:6].sum(1) + EPS)).clip(0, 10)
        with np.errstate(invalid="ignore"):
            d["r_yoy_tri"] = ((L[:, :3].sum(1) + EPS) / (L[:, 12:15].sum(1) + EPS)).clip(0, 10)

        # diffs consecutivos + interanual (sin diff_0_2: es 2*pend_3)
        for k in range(6):
            d[f"diff_{k}_{k+1}"] = Ln[:, k] - Ln[:, k + 1]
        d["diff_0_12"] = Ln[:, 0] - Ln[:, 12]

        # medias móviles ancladas (sin s12: es constante 1)
        for k in (2, 3, 6):
            d[f"s{k}"] = L[:, :k].sum(1) / (k * prom + EPS)

        # rolling stats (sin roll_std_12: es cv)
        for k in (3, 6, 12):
            V = Ln[:, :k]
            if k < 12:
                d[f"roll_std_{k}"] = np.nanstd(V, axis=1)
            d[f"roll_min_{k}"] = np.nanmin(V, axis=1)
            d[f"roll_max_{k}"] = np.nanmax(V, axis=1)
            d[f"roll_med_{k}"] = np.nanmedian(V, axis=1)

        # medias móviles desplazadas (sin ma6_lag6 ni cruces degenerados)
        ma3_act = L[:, 0:3].mean(1) / (prom + EPS)
        for off in (3, 6, 9):
            d[f"ma3_lag{off}"] = L[:, off:off + 3].mean(1) / (prom + EPS)
        d["acel_ma3"] = ma3_act - d["ma3_lag3"]

        # diffs vs extremos y promedio (sin dprom_12: es tn_0 - 1)
        for k in (6, 12, 24):
            V = Ln[:, :k]
            d[f"dmin_{k}"] = Ln[:, 0] - np.nanmin(V, axis=1)
            d[f"dmax_{k}"] = Ln[:, 0] - np.nanmax(V, axis=1)
            if k != 12:
                d[f"dprom_{k}"] = Ln[:, 0] - np.nanmean(V, axis=1)

        # tendencias (sin pend_3: es diff_0_2/2; diff_0_2 tampoco hace falta)
        for k in (6, 12):
            d[f"pend_{k}"] = slope(Ln[:, :k][:, ::-1])

        # clientela (z601)
        ccols = [mes_menos(cut, k) for k in range(12)]
        C = self.wide_ncli.loc[d.index, ccols].values
        cmean = np.nanmean(C, axis=1)
        d["ncli_0"] = C[:, 0]
        d["ncli_idx"] = C[:, 0] / (cmean + EPS)
        d["ncli_prom"] = cmean
        d["basket_idx"] = ((L[:, 0] + EPS) / (C[:, 0] + EPS)) / (
            (prom.values + EPS) / (cmean + EPS))
        Cn = C / (cmean[:, None] + EPS)
        for k in (3, 6, 12):
            d[f"ncli_pend_{k}"] = slope(Cn[:, :k][:, ::-1])

        # edad, forma, flags
        d["edad_serie"] = [
            (pd.Period(str(cut), freq="M")
             - pd.Period(str(int(self.primer_mes[pid])), freq="M")).n + 1
            for pid in d.index
        ]
        d[[f"tn_{k}" for k in range(N_LAGS)]] = Ln
        d["cv"] = d[LAGS12].std(axis=1)
        d["anio_corte"] = cut // 100
        d["mes_corte"] = cut % 100
        d["_prom"] = prom
        return d

    @staticmethod
    def auditar(d: pd.DataFrame, umbral_corr: float = 0.999) -> list[str]:
        """Detecta features casi constantes o duplicadas (corr Spearman).

        Spearman (rangos) evita los falsos positivos de Pearson con colas
        pesadas. Devuelve la lista de problemas encontrados.
        """
        X = d.drop(columns=[c for c in ("clase_ratio", "_prom") if c in d.columns])
        problemas = []
        for c in X.columns:
            v = X[c].dropna()
            if len(v) and v.std() < 1e-4 * (abs(v.mean()) + 1e-12):
                problemas.append(f"CONSTANTE: {c}")
        corr = X.sample(min(len(X), 600), random_state=1).corr(method="spearman")
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                if abs(corr.loc[a, b]) > umbral_corr:
                    problemas.append(f"DUPLICADO: {a} ~ {b} ({corr.loc[a, b]:+.4f})")
        return problemas
