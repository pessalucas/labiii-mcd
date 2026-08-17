Partiendo de la OLS (0.231), fui probando enfoques y midiendo cómo
movían el error:

- LightGBM con feature engineering: armé un modelo con lags, ratios, medias
  móviles, diffs, tendencias y demás. Lo que más ayudó fue sumarle las features
  de categoría (cat3): lo bajó de ~0.250 a ~0.240.

- Ensamble OLS + LightGBM: combinando 70% la mágica del profe con 30% el
  LightGBM, más una calibración final por tendencia de cada producto (si viene
  subiendo o cayendo), llegué a 0.225.

- Modelos de ocurrencia / hurdle (predecir primero si el cliente compra y
  después cuánto): ganancia mínima (<1% de importancia en el ensamble de modelos).

- Nivel cliente-producto: terminó siendo redundante, predecía casi lo mismo que
  el modelo de producto.

- Redes neuronales (AutoGluon: DeepAR, TFT, PatchTST): ganaban en algunos
  segmentos (series que crecen) pero quedaban muy correlacionadas con la OLS,
  así que no aportaban al ensamble.

- Routing y clustering (elegir el mejor modelo por segmento o por cluster de
  series): mejoraban la validación local pero no transferían al leaderboard.

- Escalado y normalización (global, por producto, log, z-score con stats del
  año previo): neutro o peor.

El que quedó fue el ensamble OLS + LightGBM + calibración (0.225, 3er puesto),
empaquetado y reproducible. 
