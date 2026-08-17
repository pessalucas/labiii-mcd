# exp_v21 — Señales nuevas: STOCK (regresor) + REQUEST/QTY (regresor y clasificador)

**Spec para implementar.** Dos fuentes de datos que hoy el pipeline ignora:
`cust_request_qty`/`cust_request_tn` (demanda pedida, nivel comprador) y
`stock_final` (inventario, agregado por producto). Verificado en los datasets:
qty tiene historia completa 2017-2019; stock cubre **779/780** en 201912 pero
solo existe desde **201810** (15 meses).

## 0. Arquitectura (3 modelos — respetar el ruteo)

| modelo | qué es | recibe |
|---|---|---|
| **OLS mágica** | regresión 12 lags tn, 182 prods | **NADA. No se toca nunca** (lección #3: 182 filas → varianza) |
| **LGBM regresor** | pata del ensamble, target `clase_ratio`; ya lleva `occ_*` adentro | **STOCK** (`stk_*`) + **QTY/MIX agregado a producto** (`q_*`, `mix_*`) |
| **LGBM clasificador** (ocurrencia/hurdle) | nivel cliente×producto, target binario "compró en t+2" → produce `occ_*` | **REQUEST a nivel PAR** (`pair_req_*`) |

- El **stock NO va al clasificador**: es agregado por producto, no informa
  "¿el cliente c compra el producto p?".
- El **request va a los DOS**: agregado a producto (regresor) y a nivel par
  (clasificador). Y como el clasificador alimenta `occ_*` del regresor, el
  request llega al regresor por dos caminos → más señal, más diversidad.
- Por qué esto es mejor que `se_*` (exp_v18): son señales que la **OLS
  estructuralmente no tiene** (solo ve lags de tn) → suben la **decorrelación**
  de la pata LGBM vs la OLS → el ensamble 80/20 se beneficia MÁS, no menos.

## Principio de construcción (lo que pidió el usuario)

Todas las features deben capturar **TENDENCIA** (pendientes/diffs/aceleración) y
**RATIOS VS MIS VENTAS** (mix, cobertura, alineación qty↔tn). Y ser
**scale-free** (normalizadas por la propia media del producto), como las de tn
(el target es un ratio; no filtrar la escala del producto — lección del s12).

---

## PARTE 1 — REGRESOR: features de STOCK (`stk_*`)

**Fuente**: `cargar_stocks()` → `(periodo, product_id, stock_final)`. Armar panel
`wide_stock` (producto × periodo) análogo a `fb.wide`.

Para cada corte `C` (usa datos ≤ C, predice C+2). `venta_prom6 = mean(tn, C-5..C)`:

| feature | fórmula | qué capta |
|---|---|---|
| `stk_cob` | `stock(C) / (venta_prom6 + EPS)` | **meses de cobertura** (ratio clave vs ventas) |
| `stk_cob_delta` | `stk_cob(C) − stk_cob(C-2)` | tendencia de la cobertura |
| `stk_nivel` | `stock(C) / (media_stock_disponible + EPS)` | nivel vs su propia historia |
| `stk_delta1` | `(stock(C) − stock(C-1)) / (media_stock + EPS)` | cambio reciente (¿acumulando o vaciando?) |
| `stk_pend3` | pendiente OLS del stock en C..C-3 (normaliz.) | tendencia corta (solo 15 meses) |
| `stk_vs_tn` | `stock(C) / (tn(C) + EPS)` | stock relativo a la última venta |
| `stk_n` | nº de meses de stock disponibles del producto | confianza (para modular) |

**Signo a APRENDER, no hardcodear**: el dataset no dice de quién es el stock
(empresa vs canal). Si es del canal, sobre-cobertura → frena reposición → baja
sell-in; si es propio, ambiguo. Dejá que el LGBM aprenda el signo. **No** hacer
regla "sobre-stock → bajar".

**⚠️ Soporte**: `stk_*` son NaN antes de 201810. El LGBM come NaN nativo, pero
solo aprende a usarlas con los cortes ~201812-201910. Es la limitación que
hundió a `se_*`; asumirla. Para el submit real (corte 201912) sí están.

---

## PARTE 2 — REGRESOR: features de QTY / MIX (`q_*`, `mix_*`)

**Fuente**: `cargar_sellin()` → `group_by(product_id, periodo).agg(sum(cust_request_qty))`
= qty agregada. Armar `wide_qty`. Historia completa (sin problema de soporte).
Normalizar la serie qty por su media 12m (scale-free como tn).

**Tendencia de pedidos** (`q_*`):

| feature | fórmula |
|---|---|
| `q_s3`, `q_s6` | `sum(qty, C..C-k) / (k · prom12_qty)` |
| `q_diff_0_1`, `q_diff_0_2` | diffs normalizados de qty (aceleración) |
| `q_pend6`, `q_pend12` | pendiente OLS de la serie qty normalizada |
| `q_yoy` | `qty(C) / (qty(C-12) + EPS)` |
| `q_roll_std6`, `q_dmax6` | volatilidad / distancia al máximo |

**Ratios vs ventas / mix** (`mix_*` = tn/qty = peso por unidad):

| feature | fórmula | qué capta |
|---|---|---|
| `mix_ratio` | `mix(C) / mix_prom12` con `mix(t)=tn(t)/(qty(t)+EPS)` | **cambio de mix reciente** (migración de envase) |
| `mix_pend6` | pendiente del mix | tendencia de migración |
| `mix_yoy` | `mix(C) / (mix(C-12)+EPS)` | mix interanual |
| `qtn_align` | `(qty(C)/prom_qty) / (tn(C)/prom_tn)` | **¿pedidos crecen más rápido que entregas?** (leading) |

> `mix_ratio`, `mix_pend6`, `qtn_align` son el corazón: señal de mix que hoy
> NO existe en ningún modelo. El **nivel** de mix (`mix_0` crudo) es escala del
> producto → si se incluye, normalizar o dejar que quede en `mix_ratio`.

**Auditoría obligatoria**: `q_*` de nivel correlacionará con `tn_*` y con
`ncli_*` (más pedidos ≈ más ventas ≈ más clientes). Correr `fb.auditar` y
quedarse con **diffs/pendientes/ratios** (aportan), descartar niveles
redundantes. `mix_*` y `qtn_align` deberían pasar limpias.

---

## PARTE 3 — CLASIFICADOR: features de REQUEST a nivel PAR (`pair_req_*`)

El clasificador de ocurrencia opera a nivel **cliente×producto×corte** (target
binario "compró en t+2", ver exp_v16/p01). Ahí el request tiene su granularidad
natural: la fila cruda de sell-in ya trae `cust_request_qty`/`cust_request_tn`
por (cliente, producto, periodo).

Agregar al set de features del par (además de recencia/frecuencia ya existentes):

| feature | definición | qué capta |
|---|---|---|
| `pair_req_reciente` | flags: ¿el par pidió en los últimos 1/2/3 meses? | **intención directa** de compra |
| `pair_req_qty_0` | qty pedida por el par el último mes / su media | volumen de intención |
| `pair_req_pend3` | pendiente de la qty pedida por el par | tendencia de la intención |
| `pair_req_fillrate` | `cust_request_tn / (tn + EPS)` del par (últ. 3m) | ¿pide más de lo que recibe? (demanda insatisfecha del par) |

**Consideración clave (usuario)**: NO re-derivar conteos de clientes — eso ya lo
cubren `ncli_*` y las features de par existentes, y el piloto de pares agregado
no aportó (exp_v11). El valor nuevo es la **cantidad pedida por par** y su
**recencia/tendencia**.

> Nota sobre fill-rate: a nivel global `cust_request_tn>tn` pasa en solo 0.6% de
> las filas (casi no hay quiebres). A nivel par puede concentrarse en algunos;
> incluir `pair_req_fillrate` pero con expectativa baja — si en la FI del
> clasificador pesa <0.2%, sacarla.

Estas mejoran el clasificador → mejores `occ_n_esperados`/`occ_demanda_esp` →
que ya entran al regresor. Regenerar `features_ocurrencia_vXX.parquet`.

---

## Artefactos de salida

- `exp_v21/features_extra_regresor.parquet` — `(product_id, corte, stk_*, q_*, mix_*)`,
  se joinea en `armar()` igual que `features_categoria`/`_estacional`.
- `exp_v21/features_ocurrencia_v21.parquet` — regenerado con `pair_req_*`
  (reemplaza a v16 en el join del regresor).

## Integración y validación

1. `p01_features_regresor.py`: genera `features_extra_regresor.parquet` (Partes 1-2).
   Test de cordura: `stk_cob` mediana ~0.68 (ya medido); imprimir distribución.
2. `p02_ocurrencia_v21.py`: reentrena el clasificador con `pair_req_*` (Parte 3),
   regenera el parquet de ocurrencia y su FI (`fi_clasificador.txt`).
3. `p03_integracion.py`: pipeline del regresor v18 + join de los dos parquets.
   - `fb.auditar` sobre el frame (Spearman >0.999 → sacar la feature).
   - Backtest multi-corte (folds 201808/201810/201812, feb×2) baseline vs +nuevas.
     Reportar por fold y el **gain de `stk_*`, `q_*`, `mix_*`** en la FI.
   - **Optuna: reusar `v11opt`** (cambian pocas features; Optuna no transfiere).
   - **Chequear decorrelación** de la pata LGBM vs OLS antes/después (correlación
     de residuos): la hipótesis es que SUBE. Si sube, re-optimizar el peso del
     ensamble por backtest.
4. `summary.txt`: pregunta / qué se hizo / resultados (backtest + Kaggle) /
   conclusiones. Guardar parquets, pickles del clasificador, FIs.
5. **Submits FRUGALES**: máx 1-2 (candidato + 1 ablación). Nada de barridos.

## Qué mido yo (analista) cuando estén los artefactos

Traeme los dos parquets + las FIs (regresor y clasificador) + backtest por fold
y la correlación de residuos pata-LGBM vs OLS. Evalúo: gain de cada bloque
(`stk_`/`q_`/`mix_`/`occ_` mejorado), si el mix aporta señal ortogonal real,
cuánto sube la decorrelación, y en qué segmentos ayuda (cruzando con
`exp_v17/errores_con_ols.csv`). Con eso decidimos peso de ensamble y submit.
