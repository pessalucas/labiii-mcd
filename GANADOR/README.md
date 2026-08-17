# Modelo GANADOR — ensamble OLS + LGBM calibrado (Kaggle público 0.225)

Competencia **labo-iii-2026-ba**: predecir toneladas de 780 productos en
202002 (febrero 2020) con datos hasta 201912. Métrica: WAPE =
`sum(|real - pred|) / sum(real)`.

**Resultado:** 0.225 en el leaderboard público (puesto 3 al momento de
subirlo; top 0.217, 2do 0.224). Supera al notebook base del profesor (0.231).

---

## El modelo en una línea

```
predicción_final(producto) = factor(tendencia) * ( 0.70 * OLS_mágica  +  0.30 * LGBM_producto )
```

El **factor por tendencia** (exp_v23) es la calibración final: factor(p)=
clip(1.06-0.06*z, 0.90, 1.20) con z=tendencia lineal 12m estandarizada.
Corrige el sesgo del LGBM modulado por reversión a la media (más factor a
los productos que caían, menos a los que subían). Baja el WAPE a 0.225.

Dos modelos de familias distintas cuyos errores se decorrelacionan:

- **OLS_mágica** = el notebook **z403** del profesor tal cual: regresión
  lineal sobre 12 lags, entrenada en los 182 `productos_magicos`, corte
  201812→201902, con fallback al promedio de 2019. (Score solo: 0.231.)
- **LGBM_producto** = LightGBM `objective='l1'` que predice el ratio
  `tn(t+2)/promedio_12m` (la predicción final es `ratio * promedio`),
  entrenado con 22 cortes deslizantes (201801→201910). Features:
  - set canónico auditado (`features_lgbm.FeatureBuilder`, 76 features:
    lags 24, ratios, diffs, sumas/medias móviles, rolling, tendencias,
    clientela, edad, flags);
  - stacking de la OLS (predicción de la OLS mágica como feature);
  - categóricas nativas cat1/cat2/cat3;
  - features de **categoría cat3** (serie agregada de la familia +
    penetración): `features_categoria.parquet`;
  - features de **ocurrencia/hurdle** (P(compra) del par cliente-producto
    × monto, agregado a producto): `features_ocurrencia_v16.parquet`.
  (Score solo: 0.235.) Hiperparámetros FIJOS (ver abajo); en este proyecto
  Optuna sobre validación local no transfirió al leaderboard.

El peso 0.70/0.30 es el óptimo hallado (con una pata LGBM más fuerte, el
ensamble tolera más dosis de LGBM que el 0.80/0.20 de versiones previas).

---

## Reproducción RÁPIDA (~2 min) — verifica bit a bit

Regenera el LGBM y el ensamble desde los datos + los parquets de features
incluidos, y verifica que da EXACTO la submission ganadora.

```
# 1. tener los 4 datasets en una carpeta datasets/ (ver abajo)
# 2. instalar dependencias:  pip install -r requirements.txt
python reproduce_ganador.py
```

Salida esperada: `✓ REPRODUCE EXACTO` y el archivo
`submission_reproducida.csv` (idéntico a `artefactos/submission_ganador_0226.csv`).

**Datasets:** el script busca `datasets/sell-in.txt.gz` subiendo desde su
ubicación; poné los 4 archivos (`sell-in.txt.gz`, `tb_productos.txt`,
`tb_stocks.txt`, `product_id_apredecir201912.txt`) en un `datasets/` en
cualquier carpeta ancestro, o definí la variable de entorno
`LABO3_DATASETS=/ruta/a/sell-in.txt.gz`.

---

## Reproducción COMPLETA (~1.5 h) — regenera también los parquets

Los dos parquets de `artefactos/` son features intermedias. Para
regenerarlos desde cero:

1. **Features de categoría cat3** (`features_categoria.parquet`):
   `pruebas/exp_v11/p01_features_categoria.py` (~5 min).
2. **Features de ocurrencia** (`features_ocurrencia_v16.parquet`):
   `pruebas/exp_v16/p01_ocurrencia_edadprod.py` (~1.5 h; entrena el
   clasificador hurdle sobre 4,3M pares cliente-producto, ventana 36m).
   Los hiperparámetros del clasificador están en
   `artefactos/hiperparametros_clasificador.json` (salida de su Optuna
   n=30); el modelo entrenado está en `modelo_ocurrencia_v16.pkl`.
3. Copiar ambos parquets a `artefactos/` y correr `reproduce_ganador.py`.

La **pata OLS** (`pred_ols_magica.csv`) es la salida del notebook z403 del
profesor — se reproduce corriendo ese notebook (sección de Regresión Lineal).

---

## Contenido

```
GANADOR/
├── README.md                      este archivo
├── requirements.txt               versiones exactas de las librerías
├── reproduce_ganador.py           script maestro (reproduce y verifica)
├── codigo/
│   ├── datos.py                   carga de datasets (autolocaliza datasets/)
│   └── features_lgbm.py           FeatureBuilder canónico (76 feats + auditoría)
└── artefactos/
    ├── submission_ganador_0226.csv   LA submission (780 filas)
    ├── pred_ols_magica.csv           pata OLS (salida de z403)
    ├── pred_lgbm_v16.csv             pata LGBM sola (0.235)
    ├── features_categoria.parquet    features cat3 (insumo del LGBM)
    ├── features_ocurrencia_v16.parquet features hurdle (insumo del LGBM)
    ├── modelo_ocurrencia_v16.pkl     clasificador hurdle entrenado
    ├── hiperparametros_clasificador.json
    └── productos_magicos.json        los 182 ids (de z403)
```

## Determinismo
Semilla 102191 en todo. Sin Optuna en la reproducción (hiperparámetros
fijos ya seleccionados). Verificado: reproduce la submission con
diferencia máxima < 1e-15.
