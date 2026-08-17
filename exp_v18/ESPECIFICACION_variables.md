# exp_v18 — Índice estacional por categoría (features `se_*`)

**Spec para implementar.** Origen del hallazgo: `exp_v17` (los productos que más
mueven el WAPE tienen un componente estacional de febrero que ni la OLS ni el
LGBM capturan; el naive "mismo mes del año pasado" gana en 121/861 productos).
La idea es darle al modelo un **ancla estacional pooleada por categoría** —
estable entre años, ortogonal a lo que ya tiene.

---

## 0. Motivación en una línea

El **nivel** lo pone cada producto (`prom_12m`); la **forma estacional** la pone
la categoría (cat3), estimada juntando todos sus productos para que sea estable
con pocos años de datos. Ver factores reales medidos: rango **0.34 (Sopas) →
1.12 (PISOS)**, global ≈ **0.83**. No es plano → hay señal.

---

## 1. Geometría del problema (respetarla o hay fuga)

- Salto **t+2**: parado en el corte `C` (un diciembre en el caso real),
  se predice el mes `T = C+2` (**febrero**), **salteando** `C+1` (enero).
- Base del modelo = `prom_12m(p, C)` = media de `tn` en los 12 meses que
  **terminan en `C`** (= el `_prom` que ya calcula `FeatureBuilder`).
  Enero-del-año-a-predecir **no entra** (ni en la base ni en el target).
- Target del LGBM (ya existente): `clase_ratio = tn(p, C+2) / prom_12m(p, C)`.

> **Clave**: el índice estacional se define en las **mismas unidades de ratio**
> que `clase_ratio`. Por eso `se_factor` es, literalmente, una **predicción del
> target** (la del naive estacional) y además sirve como feature en la escala
> correcta.

El índice **NO es solo de febrero**: se calcula por **mes-objetivo**. Como el
modelo se entrena con ~22 cortes cuyos targets caen en muchos meses, cada
ejemplo recibe el factor del mes que ese ejemplo predice; en la inferencia del
submit, al corte de diciembre le toca el factor de febrero.

---

## 2. Definición formal del factor

Para un corte `C`, sea `T = C+2` y `t = mes_calendario(T)` (1..12).

Para un nivel `L ∈ {cat3, cat2, global}` definí el **conjunto de referencia
sin fuga**:

```
R_L(C) = { (p, c') :  p pertenece al grupo L,
                      mes_calendario(c'+2) == t,          # mismo mes objetivo
                      (c'+2) <= C,                         # ya conocido en C (ANTI-FUGA)
                      prom_12m(p, c') calculable (>EPS) }
```

y el **factor crudo** como media winsorizada de los ratios realizados:

```
raw_L(C)  = mean_{(p,c') in R_L(C)}  clip( tn(p, c'+2) / prom_12m(p, c'),  0, 4 )
n_L(C)    = |R_L(C)|            # nº de observaciones producto×año que lo respaldan
```

- `clip(...,0,4)` winsoriza colas (productos que se reactivan y disparan el
  ratio). Alternativa robusta: **mediana** en vez de media (elegir una y dejarla
  documentada; sugerido: media winsorizada, es lo que asume el shrinkage de §3).
- El pooling es **cross-producto** (decenas/cientos de obs por cat3) aunque haya
  1–2 años; por eso es estable pese a los pocos febreros.

---

## 3. Shrinkage jerárquico (global → cat2 → cat3)

Con pseudo-conteo `κ` (sugerido **κ = 20**; se puede tunear):

```
f_glob(C) = raw_global(C)               # si R vacío -> 1.0
f_c2(C)   = (n_c2·raw_c2 + κ·f_glob) / (n_c2 + κ)
f_c3(C)   = (n_c3·raw_c3 + κ·f_c2)   / (n_c3 + κ)
se_factor = f_c3(C)
```

- Categorías con muchas obs (Sopas: decenas) confían en lo suyo; categorías
  finas tiran hacia cat2 y luego al global. Evita que una cat3 con 3 obs
  invente estacionalidad.
- Si `n_c3 = 0` (cortes tempranos sin historia del mismo mes), el shrinkage
  devuelve el padre automáticamente → nunca rompe.

---

## 4. Artefacto de salida

Un parquet **con la misma estructura que `exp_v11/features_categoria.parquet`**
para que `armar()` lo joinee por `(product_id, corte)`:

`exp_v18/features_estacional.parquet` — una fila por `(product_id, corte)`,
cubriendo **todos los cortes 201801..201912** (como el de categoría).

### Columnas a producir

| columna | descripción |
|---|---|
| `product_id` | id |
| `corte` | corte `C` (mes del "diciembre" del ejemplo) |
| `se_factor` | **principal**. Factor cat3 con shrinkage para el mes-objetivo `C+2`. Está en unidades de `clase_ratio` (= predicción estacional del ratio). |
| `se_factor_c2` | factor a nivel cat2 (con shrinkage al global) |
| `se_factor_glob` | factor global del mes-objetivo (efecto-mes marginal) |
| `se_dev_c3` | `se_factor − se_factor_glob`. La desviación **pura** de la categoría respecto del mes promedio → señal categórica limpia, ortogonal al nivel global. **Candidata a ser la más útil.** |
| `se_n_c3` | `n_c3(C)` = soporte del factor cat3 (confianza; el GBM puede modular con esto) |

Opcionales (probar si suman, con auditoría):
| `se_own_factor` | historia del **propio producto** en el mismo mes, con shrinkage fuerte hacia `se_factor` (κ_own alto, p.ej. 8). Para productos con estacionalidad idiosincrática. |
| `se_own_n` | soporte propio |

### Mapeo de categorías
Reusar el criterio de `p02_entrenamiento`: cat3 válidas = las con `>10`
productos en train; el resto → `OTROS` (cae al nivel cat2/global vía shrinkage).
cat3/cat2 desde `cargar_productos().unique('product_id')`.

---

## 5. Integración (dos usos, en este orden)

1. **Feature del LGBM** (bajo riesgo, primero):
   en `armar()`, `d = d.join(feat_estacional_por_corte[cut], how='left')`
   igual que con `features_categoria`. El GBM decide cuánto confiar.
   `se_factor` ya está en la escala del target.

2. **3ª pata del ensamble** (si la feature ayuda): predicción estacional pura
   `pred_est(p) = prom_12m(p, C) · se_factor(p, C)`; mezclar con OLS+LGBM.
   Peso elegido por **backtest**, no contra el LB.

---

## 6. Validación (obligatoria antes de submit)

- **`fb.auditar(d)`** sobre el frame enriquecido. **Ojo puntual**: chequear que
  `se_factor`/`se_dev_c3` no queden casi-duplicados de las features yoy que ya
  existen (`tn_12`, `diff_0_12`, `r_yoy_tri`). Si Spearman > 0.999 con alguna,
  sobra. La apuesta es que `se_*` aporta señal **cross-seccional** (de la
  categoría) que las yoy per-producto no tienen.
- **Backtest multi-corte** (folds `201808`, `201810`, `201812` con feb×2, como
  `p02`): baseline v11 vs v11 + `se_*`. Reportar por fold.
- **Optuna**: correrlo (regla del usuario), pero **reusar `v11opt`** como punto
  de partida — cambian pocas features. Ver hiperparámetros en `contexto.txt`.
- **Submits FRUGALES**: máx 1–2 (candidato + 1 ablación). Nada de barridos
  contra el LB.

---

## 7. Casos borde

- Corte temprano sin historia del mismo mes (`n_c3=0`) → shrinkage devuelve
  cat2/global/1.0. No romper.
- Producto sin cat3 o cat3 rara → `OTROS` → cae a cat2/global.
- `prom_12m` no calculable → producto excluido (igual que `FeatureBuilder`).
- Ratios extremos por reactivación → ya winsorizados en §2.

---

## 8. Checklist para el otro agente

- [ ] `p01_features_estacional.py`: genera `features_estacional.parquet`
      (§2–§4), con anti-fuga verificado (assert: ningún `c'+2 > C` usado).
- [ ] Test de cordura: imprimir factores por cat3 para `C=201812` (deben
      parecerse a Sopas≈0.34, PISOS≈1.12, global≈0.83).
- [ ] `p02_entrenamiento.py`: v11 + join `se_*`; `fb.auditar`; backtest
      multi-corte; Optuna (reusar v11opt); FI para ver el gain de `se_*`.
- [ ] `summary.txt`: pregunta / qué se hizo / resultados (backtest+Kaggle) /
      conclusiones. Guardar el parquet y la FI.
- [ ] Submit frugal solo si el backtest no muestra desastre.

---

## 9. Qué mido yo (analista) cuando esté el parquet

Traeme `features_estacional.parquet` + la FI y el backtest por fold, y evalúo:
cuánto gain se lleva `se_*`, si desplaza a las yoy, en qué segmentos ayuda
(cruzándolo con `exp_v17/errores_con_ols.csv`), y si el naive estacional puro
ya explica parte del error de los top-movers estacionales (20004, 20009,
20075, 20080, 20094).
