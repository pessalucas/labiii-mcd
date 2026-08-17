# Arquitectura del modelo ganador (Kaggle 0.225)

Documento técnico de la solución. Explica cómo está construido el modelo
end-to-end, las decisiones de diseño y por qué funciona.

---

## 1. Vista general

El modelo final es un **ensamble de dos predictores de familias distintas**
cuyos errores se decorrelacionan. Predice las toneladas de febrero-2020
(t+2) para 780 productos, con datos hasta diciembre-2019 (t0).

```
                          DATOS CRUDOS (sell-in, productos, stocks)
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              │                                                     │
       ┌──────▼───────┐                                    ┌────────▼────────┐
       │  OLS MÁGICA  │                                    │  LGBM PRODUCTO  │
       │   (z403)     │                                    │   (ratio, l1)   │
       │  0.231 solo  │                                    │   0.235 solo    │
       └──────┬───────┘                                    └────────┬────────┘
              │  pred_ols (780)                        pred_lgbm (780) │
              └─────────────────────┬───────────────────────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │   ENSAMBLE PONDERADO    │
                        │  0.70·OLS + 0.30·LGBM   │
                        └───────────┬────────────┘
                                    │
                          submission (780)  →  Kaggle 0.225
```

Por qué dos modelos y no uno: la OLS es un modelo **simple y robusto**
(especialista en el salto dic→feb, poca varianza); el LGBM es **rico**
(momentum + contexto de categoría + clientela). Ponderan meses y señales
distintas → sus errores no están correlacionados → el promedio pondera
menos error que cualquiera solo. El peso 70/30 (más OLS) refleja que la
OLS generaliza mejor al mes objetivo real; el LGBM aporta corrección.

---

## 2. Formulación del problema

- **Grano de evaluación:** producto × mes. Predecir `tn(producto, 202002)`.
- **Horizonte:** t+2 (de diciembre se predice febrero, salteando enero).
- **Métrica:** WAPE = `Σ|real−pred| / Σreal`. Ponderada por volumen ⇒ los
  productos grandes dominan ⇒ el objetivo natural es **error absoluto (L1)**.
- **Datos:** 36 meses (201701–201912), 780 productos a predecir, ~600
  clientes. El sell-in es cliente×producto×mes; se agrega a producto×mes.

---

## 3. Pata A — OLS mágica (notebook z403)

Regresión lineal clásica, el modelo del profesor:

```
tn(t+2)  ≈  β0 + β1·tn(t) + β2·tn(t-1) + ... + β12·tn(t-11)
```

- **Entrenamiento:** 1 sola fila por producto, corte fijo 201812 → target
  201902 (el "mismo examen", un año antes). Solo 182 `productos_magicos`
  (lista curada del profesor).
- **Aplicación:** a los 656 productos con 12 meses de 2019 completos;
  fallback al promedio 2019 para los 124 restantes.
- **Clave:** su fuerza es la simplicidad (13 parámetros, 182 filas). No
  admite enriquecimiento — agregarle features lo empeora (probado). Aprende
  que febrero se explica sobre todo por noviembre, el febrero previo, marzo
  y junio; diciembre casi no pesa.

---

## 4. Pata B — LGBM de producto

LightGBM con `objective='l1'` (alineado a WAPE). No predice toneladas
directas sino un **ratio normalizado**, lo que le da escala propia a cada
producto:

```
target   =  tn(t+2) / promedio_12m(producto)
predicción =  ratio_predicho · promedio_12m(producto)
```

### 4.1 Esquema temporal (cortes deslizantes)

En vez de 1 corte (como la OLS), usa **22 cortes** 201801→201910, cada uno
con su ventana de features y su target en t+2. Esto multiplica los datos
(~18.700 filas = productos × cortes) y le da al árbol variedad para
aprender el patrón general.

```
corte C:   [ ventana de 24 meses de historia ]  ─────►  target tn(C+2)
           C-23 ................ C-1  C0                 C+2
```

### 4.2 Capas de features (≈140 en total)

```
┌─ SERIE DEL PRODUCTO (features_lgbm.FeatureBuilder, 76 auditadas) ────────┐
│  · 24 lags normalizados          · sumas / medias móviles (s2,s3,s6)     │
│  · ratios (r_0_1, r_tri, yoy)    · rolling std/min/max/mediana           │
│  · diffs consecutivos            · dmin/dmax/dprom vs ventana            │
│  · tendencias (pendientes 6,12)  · medias móviles desplazadas            │
│  · CLIENTELA: ncli_0, ncli_idx, ncli_pend_* (nº compradores y su tend.)  │
│  · cv, edad_serie, flags anio/mes                                        │
├─ STACKING DE LA OLS ─────────────────────────────────────────────────────┤
│  · pred_ols_yoy, pred_ols_rec (predicción de la OLS como feature, sin fuga)│
├─ CATEGÓRICAS NATIVAS ────────────────────────────────────────────────────┤
│  · cat1, cat2, cat3 (LightGBM las maneja nativo)                         │
├─ CONTEXTO DE CATEGORÍA cat3 (features_categoria.parquet, 55) ────────────┤
│  · c3_*: toda la maquinaria sobre la SERIE AGREGADA de la categoría      │
│  · sh_*: penetración del producto en su categoría y su evolución         │
├─ OCURRENCIA / HURDLE (features_ocurrencia_v16.parquet, 3) ───────────────┤
│  · occ_demanda_esp, occ_n_esperados, occ_demanda_norm                    │
└──────────────────────────────────────────────────────────────────────────┘
```

Drivers reales (por importancia): `s2` (nivel reciente suavizado, ~40%),
`tn_0`, `roll_med_3`, la **clientela** `ncli_*` (~12%) y el **contexto de
categoría** `c3_*` (~13%). El resto aporta poco; el contexto de categoría
fue la mejora que llevó el LGBM de 0.250 a 0.240→0.235.

### 4.3 Submódulo — clasificador de ocurrencia (hurdle)

Genera las features `occ_*`. Es un modelo aparte, a nivel **par
cliente-producto** (~4,3M filas de entrenamiento):

```
  P(compra | cliente, producto, t+2)   [LGBM binario, AUC 0.79]
        × monto_medio_del_par
  ───────────────────────────────────
  = demanda esperada del par
        Σ sobre los ~600 clientes
  ───────────────────────────────────
  = occ_demanda_esp(producto)   →  feature del LGBM de producto
```

Ventana 36 meses, sin fuga (2-fold temporal: el modelo que predice un
corte no vio su target). NOTA: esta señal resultó **redundante** con la
clientela agregada (aporta <1% del gain); se incluye porque no daña y el
modelo v16 con ella fue la mejor pata, pero no es el motor de la mejora.

---

## 5. El ensamble

```
final = 0.70 · pred_OLS  +  0.30 · pred_LGBM      (clip a ≥ 0)
```

Peso elegido probando el leaderboard: con la pata LGBM de 0.235, el óptimo
es 70/30 (versiones con pata más débil preferían 80/20). Fallback al
promedio 2019 si a un producto le falta alguna pata.

---

## 6. Decisiones de diseño y su fundamento

| Decisión | Por qué |
|---|---|
| Target = ratio, no toneladas | da escala propia a cada producto; los árboles no extrapolan escala |
| `objective='l1'` | la métrica es error absoluto (WAPE); L1 > L2 (probado) |
| Cortes deslizantes | multiplica datos y enseña el patrón general del salto t+2 |
| Contexto de categoría (cat3) | señal nueva real: la familia sube/baja afecta al producto (+; llevó 0.250→0.240) |
| Hiperparámetros FIJOS | Optuna sobre validación local NO transfirió al leaderboard (5+ veces) |
| Ensamble OLS+LGBM | errores decorrelacionados; el promedio < cualquiera solo |
| Auditoría de features | evita features degeneradas/duplicadas que inflan varianza |

## 7. Lo que se probó y NO entró (para no repetirlo)

- Enriquecer la OLS con features → la empeora (satura con 182 filas).
- cat4 (sustitución tamaño/sabor), sumada o reemplazando cat3 → peor (bloat).
- Predicción directa a nivel cliente-producto → mucho peor (0.29): la señal
  fina no sobrevive a la agregación a producto.
- Optuna en cualquiera de sus formas → no transfiere al leaderboard.

## 8. Rendimiento

| Modelo | WAPE público |
|---|---|
| Naifs / AutoARIMA | 0.27 – 0.34 |
| AutoGluon | 0.249 |
| LGBM de producto (solo) | 0.235 |
| OLS mágica (solo, z403) | 0.231 |
| **ENSAMBLE (final)** | **0.225** |

Contexto: el "pelotón" del curso (quienes corrieron z403) quedó en 0.231;
el tope del leaderboard es 0.217. El ensamble alcanzó el puesto 3.
```
