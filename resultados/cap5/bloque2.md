# Bloque pendiente del Cap. 5: semillas, Mamba, regla, ablaciones, Holm, camara y figuras

Generado por `python evaluacion_cap5.py`. Prueba evaluada una sola vez; todo se decidio en validacion. Media +- desviacion entre semillas; IC 95 % por bootstrap pareado de videos (1000 remuestreos, semilla 42, los mismos de la fase F).

Semillas por modelo: CfC 5, GRU 5, LSTM 5, Mamba 5, gradient boosting 1, regresión logística 1, regla sin aprendizaje 1, clase mayoritaria 1.

## F1 macro final en validacion (umbral 0.5 + histeresis 2 s, referencia de la IA)

| modelo | semillas | F1 macro final en validacion |
|---|---|---|
| CfC | 5 | 0.539 +- 0.019 |
| GRU | 5 | 0.614 +- 0.035 |
| LSTM | 5 | 0.545 +- 0.038 |
| Mamba | 5 | 0.563 +- 0.038 |
| gradient boosting | 1 | 0.653 |
| regresión logística | 1 | 0.479 |
| regla sin aprendizaje | 1 | 0.573 |

## Nivel por segundo en la prueba (5 semillas)

| modelo | referencia | F1 macro [IC 95 %] | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa [IC 95 %] | mAP tIoU 0.3 | mAP 0.5 | mAP 0.7 |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | IA | 0.458 +- 0.013 [0.396, 0.519] | 0.523 | 0.223 | 0.630 | 0.486 | 0.331 [0.202, 0.448] | 0.301 | 0.229 | 0.174 |
| CfC | humana | 0.459 +- 0.035 [0.400, 0.516] | 0.602 | 0.279 | 0.496 | 0.468 | 0.346 [0.252, 0.437] | 0.185 | 0.108 | 0.041 |
| GRU | IA | 0.526 +- 0.015 [0.442, 0.606] | 0.560 | 0.393 | 0.625 | 0.531 | 0.384 [0.233, 0.530] | 0.396 | 0.308 | 0.237 |
| GRU | humana | 0.503 +- 0.017 [0.419, 0.585] | 0.576 | 0.419 | 0.513 | 0.496 | 0.364 [0.252, 0.479] | 0.290 | 0.142 | 0.072 |
| LSTM | IA | 0.462 +- 0.016 [0.393, 0.528] | 0.537 | 0.224 | 0.624 | 0.492 | 0.334 [0.205, 0.454] | 0.327 | 0.261 | 0.197 |
| LSTM | humana | 0.466 +- 0.010 [0.398, 0.537] | 0.614 | 0.284 | 0.499 | 0.477 | 0.353 [0.252, 0.455] | 0.219 | 0.121 | 0.047 |
| Mamba | IA | 0.526 +- 0.020 [0.451, 0.598] | 0.540 | 0.400 | 0.637 | 0.523 | 0.399 [0.282, 0.518] | 0.357 | 0.267 | 0.173 |
| Mamba | humana | 0.540 +- 0.020 [0.463, 0.609] | 0.588 | 0.498 | 0.533 | 0.535 | 0.399 [0.287, 0.504] | 0.258 | 0.137 | 0.075 |
| gradient boosting | IA | 0.442 [0.368, 0.525] | 0.510 | 0.212 | 0.605 | 0.476 | 0.292 [0.178, 0.413] | 0.246 | 0.204 | 0.130 |
| gradient boosting | humana | 0.451 [0.362, 0.531] | 0.597 | 0.293 | 0.463 | 0.461 | 0.320 [0.218, 0.426] | 0.193 | 0.125 | 0.028 |
| regresión logística | IA | 0.491 [0.407, 0.570] | 0.475 | 0.387 | 0.610 | 0.490 | 0.311 [0.158, 0.449] | 0.277 | 0.204 | 0.147 |
| regresión logística | humana | 0.527 [0.433, 0.604] | 0.567 | 0.475 | 0.540 | 0.523 | 0.365 [0.220, 0.485] | 0.172 | 0.069 | 0.021 |
| regla sin aprendizaje | IA | 0.434 [0.331, 0.514] | 0.395 | 0.247 | 0.659 | 0.497 | 0.291 [0.163, 0.408] | 0.195 | 0.165 | 0.082 |
| regla sin aprendizaje | humana | 0.312 [0.236, 0.384] | 0.372 | 0.156 | 0.409 | 0.318 | 0.175 [0.091, 0.266] | 0.063 | 0.028 | 0.002 |
| clase mayoritaria | IA | 0.180 [0.135, 0.219] | 0.000 | 0.000 | 0.541 | 0.371 | 0.000 [0.000, 0.000] | 0.116 | 0.091 | 0.043 |
| clase mayoritaria | humana | 0.100 [0.064, 0.138] | 0.001 | 0.000 | 0.299 | 0.176 | 0.000 [0.000, 0.001] | 0.035 | 0.006 | 0.000 |

## Alerta (alfa 0.10), tiempo al peligro y confiabilidad en la prueba (5 semillas)

| modelo | falsas alarmas sin fuego | controlados IA | controlados humana | EWT mediana IA (s) | antes del cambio IA | EWT mediana humana (s) | antes del cambio humana | MAE tau IA (s) | MAE tau humana (s) | mediana tau IA (s) | mediana tau humana (s) | ECE IA | ECE humana | Brier IA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.000 | 0.213 | 0.193 | 0.2 | 9.0 de 21 | 25.5 | 12.4 de 16 | 18.5 | 62.8 | 15.2 | 45.1 | 0.164 | 0.242 | 0.640 |
| GRU | 0.000 | 0.180 | 0.152 | -8.3 | 7.2 de 21 | 17.1 | 10.4 de 16 | 25.0 | 57.8 | 19.1 | 39.1 | 0.275 | 0.319 | 0.695 |
| LSTM | 0.000 | 0.227 | 0.221 | 1.9 | 8.4 de 21 | 26.7 | 10.6 de 16 | 20.7 | 59.2 | 17.6 | 40.6 | 0.225 | 0.301 | 0.665 |
| Mamba | 0.000 | 0.067 | 0.076 | -3.0 | 7.2 de 21 | 26.3 | 10.4 de 16 | 85.1 | 76.8 | 21.0 | 40.7 | 0.192 | 0.210 | 0.620 |
| gradient boosting | 0.031 | 0.267 | 0.207 | 3.8 | 12.0 de 21 | 26.6 | 13.0 de 16 | - | - | - | - | 0.263 | 0.297 | 0.726 |
| regresión logística | 0.000 | 0.100 | 0.069 | -10.5 | 5.0 de 21 | 13.2 | 10.0 de 16 | - | - | - | - | 0.149 | 0.148 | 0.616 |
| regla sin aprendizaje | 0.031 | 0.100 | 0.069 | -22.6 | 1.0 de 21 | 11.6 | 8.0 de 16 | - | - | - | - | - | - | - |

## Alerta con alfa 0.05 (5 semillas)

| modelo | referencia | sin fuego | controlados | EWT mediana (s) | antes del cambio | sin alerta |
|---|---|---|---|---|---|---|
| CfC | IA | 0.000 | 0.060 | -22.9 | 3.4 de 21 | 12.2 |
| CfC | humana | 0.000 | 0.048 | 13.5 | 6.2 de 16 | 8.4 |
| GRU | IA | 0.000 | 0.047 | -27.8 | 3.8 de 21 | 9.4 |
| GRU | humana | 0.000 | 0.034 | 4.3 | 5.8 de 16 | 5.4 |
| LSTM | IA | 0.000 | 0.027 | -14.9 | 3.4 de 21 | 12.6 |
| LSTM | humana | 0.000 | 0.028 | 13.4 | 5.2 de 16 | 8.6 |
| Mamba | IA | 0.000 | 0.033 | -11.4 | 2.8 de 21 | 10.4 |
| Mamba | humana | 0.000 | 0.034 | 9.8 | 5.6 de 16 | 7.6 |
| gradient boosting | IA | 0.000 | 0.033 | -5.7 | 4.0 de 21 | 10.0 |
| gradient boosting | humana | 0.000 | 0.034 | 20.2 | 6.0 de 16 | 6.0 |
| regresión logística | IA | 0.000 | 0.000 | -58.6 | 2.0 de 21 | 12.0 |
| regresión logística | humana | 0.000 | 0.000 | -8.3 | 2.0 de 16 | 8.0 |
| regla sin aprendizaje | IA | 0.031 | 0.067 | -33.5 | 1.0 de 21 | 10.0 |
| regla sin aprendizaje | humana | 0.031 | 0.034 | 11.6 | 8.0 de 16 | 6.0 |

## Cambios de nivel (5 semillas)

| modelo | referencia | a incipiente: detectados | mediana del error (s) | a incontrolado: detectados | mediana del error (s) | cambios de mas por video | cambios por minuto | cambios por minuto de la referencia |
|---|---|---|---|---|---|---|---|---|
| CfC | IA | 8.6 de 11 | 10.7 | 15.6 de 21 | 15.1 | 2.37 | 2.23 | 0.41 |
| CfC | humana | 4.0 de 4 | 9.3 | 12.8 de 16 | 14.2 | 4.44 | 2.63 | 0.40 |
| GRU | IA | 6.6 de 11 | 5.0 | 16.0 de 21 | 26.7 | 1.57 | 1.64 | 0.41 |
| GRU | humana | 3.0 de 4 | 11.3 | 13.8 de 16 | 17.5 | 2.47 | 1.72 | 0.40 |
| LSTM | IA | 8.6 de 11 | 14.6 | 16.0 de 21 | 18.2 | 2.41 | 2.42 | 0.41 |
| LSTM | humana | 4.0 de 4 | 16.4 | 13.6 de 16 | 16.6 | 4.14 | 2.66 | 0.40 |
| Mamba | IA | 8.2 de 11 | 8.8 | 16.2 de 21 | 18.1 | 1.76 | 1.86 | 0.41 |
| Mamba | humana | 3.6 de 4 | 12.8 | 13.8 de 16 | 13.8 | 3.15 | 2.15 | 0.40 |
| gradient boosting | IA | 10.0 de 11 | 9.3 | 19.0 de 21 | 8.3 | 2.71 | 2.78 | 0.41 |
| gradient boosting | humana | 4.0 de 4 | 7.2 | 14.0 de 16 | 4.8 | 5.10 | 3.33 | 0.40 |
| regresión logística | IA | 9.0 de 11 | 16.5 | 15.0 de 21 | 12.9 | 2.53 | 2.55 | 0.41 |
| regresión logística | humana | 4.0 de 4 | 5.6 | 13.0 de 16 | 7.4 | 5.07 | 3.29 | 0.40 |
| regla sin aprendizaje | IA | 8.0 de 11 | 7.8 | 15.0 de 21 | 4.8 | 1.03 | 1.16 | 0.41 |
| regla sin aprendizaje | humana | 4.0 de 4 | 14.0 | 11.0 de 16 | 31.3 | 1.90 | 1.38 | 0.40 |
| clase mayoritaria | IA | 0.0 de 11 | - | 0.0 de 21 | - | 0.00 | 0.00 | 0.41 |
| clase mayoritaria | humana | 0.0 de 4 | - | 0.0 de 16 | - | 0.00 | 0.00 | 0.40 |

## B4. Confiabilidad (5 semillas)

| modelo | referencia | ECE | Brier |
|---|---|---|---|
| CfC | IA | 0.164 | 0.640 |
| CfC | humana | 0.242 | 0.762 |
| GRU | IA | 0.275 | 0.695 |
| GRU | humana | 0.319 | 0.795 |
| LSTM | IA | 0.225 | 0.665 |
| LSTM | humana | 0.301 | 0.796 |
| Mamba | IA | 0.192 | 0.620 |
| Mamba | humana | 0.210 | 0.688 |
| gradient boosting | IA | 0.263 | 0.726 |
| gradient boosting | humana | 0.297 | 0.816 |
| regresión logística | IA | 0.149 | 0.616 |
| regresión logística | humana | 0.148 | 0.640 |

## B5. Cobertura de la alerta conformal (alfa 0.10, por video, 5 semillas)

| conjunto | modelo | controlados | sin fuego | juntos | corridas con el 10 % dentro del IC (juntos) |
|---|---|---|---|---|---|
| calibracion | CfC | 20.0 % (5.0 de 25) | 0.0 % (0.0 de 27) | 9.6 % (5.0 de 52) | 5 de 5 |
| calibracion | GRU | 20.0 % (5.0 de 25) | 0.0 % (0.0 de 27) | 9.6 % (5.0 de 52) | 5 de 5 |
| calibracion | LSTM | 20.0 % (5.0 de 25) | 0.0 % (0.0 de 27) | 9.6 % (5.0 de 52) | 5 de 5 |
| calibracion | Mamba | 20.0 % (5.0 de 25) | 0.0 % (0.0 de 27) | 9.6 % (5.0 de 52) | 5 de 5 |
| calibracion | gradient boosting | 20.0 % (5.0 de 25) | 0.0 % (0.0 de 27) | 9.6 % (5.0 de 52) | 1 de 1 |
| calibracion | regresión logística | 20.0 % (5.0 de 25) | 0.0 % (0.0 de 27) | 9.6 % (5.0 de 52) | 1 de 1 |
| calibracion | regla sin aprendizaje | 20.0 % (5.0 de 25) | 0.0 % (0.0 de 27) | 9.6 % (5.0 de 52) | 1 de 1 |
| validacion | CfC | 16.8 % (4.2 de 25) | 0.0 % (0.0 de 28) | 7.9 % (4.2 de 53) | 5 de 5 |
| validacion | GRU | 10.4 % (2.6 de 25) | 0.0 % (0.0 de 28) | 4.9 % (2.6 de 53) | 5 de 5 |
| validacion | LSTM | 13.6 % (3.4 de 25) | 0.0 % (0.0 de 28) | 6.4 % (3.4 de 53) | 5 de 5 |
| validacion | Mamba | 9.6 % (2.4 de 25) | 0.0 % (0.0 de 28) | 4.5 % (2.4 de 53) | 5 de 5 |
| validacion | gradient boosting | 36.0 % (9.0 de 25) | 7.1 % (2.0 de 28) | 20.8 % (11.0 de 53) | 0 de 1 |
| validacion | regresión logística | 8.0 % (2.0 de 25) | 0.0 % (0.0 de 28) | 3.8 % (2.0 de 53) | 1 de 1 |
| validacion | regla sin aprendizaje | 4.0 % (1.0 de 25) | 3.6 % (1.0 de 28) | 3.8 % (2.0 de 53) | 1 de 1 |
| prueba (IA) | CfC | 21.3 % (6.4 de 30) | 0.0 % (0.0 de 32) | 10.3 % (6.4 de 62) | 5 de 5 |
| prueba (IA) | GRU | 18.0 % (5.4 de 30) | 0.0 % (0.0 de 32) | 8.7 % (5.4 de 62) | 5 de 5 |
| prueba (IA) | LSTM | 22.7 % (6.8 de 30) | 0.0 % (0.0 de 32) | 11.0 % (6.8 de 62) | 5 de 5 |
| prueba (IA) | Mamba | 6.7 % (2.0 de 30) | 0.0 % (0.0 de 32) | 3.2 % (2.0 de 62) | 5 de 5 |
| prueba (IA) | gradient boosting | 26.7 % (8.0 de 30) | 3.1 % (1.0 de 32) | 14.5 % (9.0 de 62) | 1 de 1 |
| prueba (IA) | regresión logística | 10.0 % (3.0 de 30) | 0.0 % (0.0 de 32) | 4.8 % (3.0 de 62) | 1 de 1 |
| prueba (IA) | regla sin aprendizaje | 10.0 % (3.0 de 30) | 3.1 % (1.0 de 32) | 6.5 % (4.0 de 62) | 1 de 1 |
| prueba (humana) | CfC | 19.3 % (5.6 de 29) | 0.0 % (0.0 de 32) | 9.2 % (5.6 de 61) | 5 de 5 |
| prueba (humana) | GRU | 15.2 % (4.4 de 29) | 0.0 % (0.0 de 32) | 7.2 % (4.4 de 61) | 4 de 5 |
| prueba (humana) | LSTM | 22.1 % (6.4 de 29) | 0.0 % (0.0 de 32) | 10.5 % (6.4 de 61) | 5 de 5 |
| prueba (humana) | Mamba | 7.6 % (2.2 de 29) | 0.0 % (0.0 de 32) | 3.6 % (2.2 de 61) | 5 de 5 |
| prueba (humana) | gradient boosting | 20.7 % (6.0 de 29) | 3.1 % (1.0 de 32) | 11.5 % (7.0 de 61) | 1 de 1 |
| prueba (humana) | regresión logística | 6.9 % (2.0 de 29) | 0.0 % (0.0 de 32) | 3.3 % (2.0 de 61) | 1 de 1 |
| prueba (humana) | regla sin aprendizaje | 6.9 % (2.0 de 29) | 3.1 % (1.0 de 32) | 4.9 % (3.0 de 61) | 1 de 1 |

## B8. Tiempo al peligro (5 semillas)

| modelo | referencia | videos | error absoluto mediano (s) | MAE (s) |
|---|---|---|---|---|
| CfC | IA | 21 | 15.2 +- 1.3 | 18.5 +- 1.7 |
| CfC | humana | 16 | 45.1 +- 3.1 | 62.8 +- 2.9 |
| GRU | IA | 21 | 19.1 +- 2.0 | 25.0 +- 2.1 |
| GRU | humana | 16 | 39.1 +- 2.9 | 57.8 +- 1.2 |
| LSTM | IA | 21 | 17.6 +- 2.7 | 20.7 +- 1.9 |
| LSTM | humana | 16 | 40.6 +- 2.6 | 59.2 +- 2.3 |
| Mamba | IA | 21 | 21.0 +- 2.7 | 85.1 +- 50.2 |
| Mamba | humana | 16 | 40.7 +- 1.8 | 76.8 +- 18.4 |

## Costo por paso (lote 1)

| modelo | parametros | CPU mediana (ms) | CPU p95 (ms) | GPU mediana (ms) | GPU p95 (ms) |
|---|---|---|---|---|---|
| CfC | 46133 | 0.208 | 0.331 | 0.665 | 0.759 |
| GRU | 19829 | 0.172 | 0.194 | 0.385 | 0.437 |
| LSTM | 26293 | 0.234 | 0.303 | 0.394 | 0.534 |
| Mamba | 35445 | 0.535 | 0.725 | 1.017 | 1.386 |

## Comparaciones pareadas de la prueba

| bloque | comparacion | diferencia | IC 95 % | p | p de Holm (global) | significativa tras Holm |
|---|---|---|---|---|---|---|
| fase F (3 semillas) | GRU - CfC, f1, IA (3 semillas) | +0.059 | -0.006 a +0.128 | 0.0822 | 1.0000 | no |
| fase F (3 semillas) | GRU - gradient boosting, f1, IA (3 semillas) | +0.077 | -0.008 a +0.151 | 0.0643 | 1.0000 | no |
| fase F (5 semillas y Mamba) | GRU - CfC, f1, IA (5 semillas) | +0.068 | -0.004 a +0.135 | 0.0527 | 1.0000 | no |
| fase F (5 semillas y Mamba) | GRU - gradient boosting, f1, IA (5 semillas) | +0.084 | -0.002 a +0.158 | 0.0467 | 1.0000 | no |
| fase F (5 semillas y Mamba) | Mamba - GRU, f1, IA (5 semillas) | -0.000 | -0.058 a +0.050 | 0.9892 | 1.0000 | no |
| fase F (5 semillas y Mamba) | Mamba - CfC, f1, IA (5 semillas) | +0.067 | -0.010 a +0.140 | 0.0813 | 1.0000 | no |
| fase F (3 semillas) | GRU - CfC, kappa, IA (3 semillas) | +0.052 | -0.038 a +0.153 | 0.2698 | 1.0000 | no |
| fase F (3 semillas) | GRU - gradient boosting, kappa, IA (3 semillas) | +0.079 | -0.054 a +0.207 | 0.2461 | 1.0000 | no |
| fase F (5 semillas y Mamba) | GRU - CfC, kappa, IA (5 semillas) | +0.053 | -0.045 a +0.147 | 0.2838 | 1.0000 | no |
| fase F (5 semillas y Mamba) | GRU - gradient boosting, kappa, IA (5 semillas) | +0.092 | -0.044 a +0.223 | 0.1771 | 1.0000 | no |
| fase F (5 semillas y Mamba) | Mamba - GRU, kappa, IA (5 semillas) | +0.015 | -0.073 a +0.101 | 0.7388 | 1.0000 | no |
| fase F (5 semillas y Mamba) | Mamba - CfC, kappa, IA (5 semillas) | +0.068 | -0.040 a +0.173 | 0.2159 | 1.0000 | no |
| fase F (3 semillas) | GRU - CfC, f1, humana (3 semillas) | +0.023 | -0.037 a +0.088 | 0.4818 | 1.0000 | no |
| fase F (3 semillas) | GRU - gradient boosting, f1, humana (3 semillas) | +0.046 | -0.039 a +0.137 | 0.3036 | 1.0000 | no |
| fase F (5 semillas y Mamba) | GRU - CfC, f1, humana (5 semillas) | +0.044 | -0.024 a +0.114 | 0.2101 | 1.0000 | no |
| fase F (5 semillas y Mamba) | GRU - gradient boosting, f1, humana (5 semillas) | +0.052 | -0.038 a +0.145 | 0.2743 | 1.0000 | no |
| fase F (5 semillas y Mamba) | Mamba - GRU, f1, humana (5 semillas) | +0.037 | -0.020 a +0.091 | 0.1948 | 1.0000 | no |
| fase F (5 semillas y Mamba) | Mamba - CfC, f1, humana (5 semillas) | +0.081 | +0.019 a +0.138 | 0.0105 | 1.0000 | no |
| fase F (3 semillas) | GRU - CfC, kappa, humana (3 semillas) | +0.007 | -0.078 a +0.094 | 0.8783 | 1.0000 | no |
| fase F (3 semillas) | GRU - gradient boosting, kappa, humana (3 semillas) | +0.032 | -0.084 a +0.158 | 0.5993 | 1.0000 | no |
| fase F (5 semillas y Mamba) | GRU - CfC, kappa, humana (5 semillas) | +0.018 | -0.078 a +0.114 | 0.7083 | 1.0000 | no |
| fase F (5 semillas y Mamba) | GRU - gradient boosting, kappa, humana (5 semillas) | +0.044 | -0.077 a +0.180 | 0.4860 | 1.0000 | no |
| fase F (5 semillas y Mamba) | Mamba - GRU, kappa, humana (5 semillas) | +0.035 | -0.050 a +0.113 | 0.3905 | 1.0000 | no |
| fase F (5 semillas y Mamba) | Mamba - CfC, kappa, humana (5 semillas) | +0.053 | -0.036 a +0.134 | 0.2398 | 1.0000 | no |

## Ablaciones (GRU y CfC, semillas 42 a 44)

Diferencia de F1 macro frente al sistema completo (las mismas semillas congeladas). * = el IC sigue sin el cero despues de Holm (familia global).

| variante | GRU IA | GRU humana | CfC IA | CfC humana |
|---|---|---|---|---|
| sistema completo (F1) | 0.519 | 0.497 | 0.460 | 0.475 |
| a. sin contexto de cámara en la compuerta | +0.035 [-0.009, +0.071] | +0.019 [-0.026, +0.063] | +0.063 [+0.019, +0.112] | -0.002 [-0.045, +0.047] |
| b. dt constante de 0,1 s | -0.022 [-0.058, +0.021] | -0.029 [-0.080, +0.019] | +0.015 [-0.016, +0.049] | +0.002 [-0.034, +0.032] |
| c. sin compuerta | -0.034 [-0.075, +0.017] | -0.012 [-0.071, +0.048] | -0.015 [-0.050, +0.021] | +0.009 [-0.026, +0.042] |
| d. umbral de alerta fijo 0.5 | +0.000 (no cambia el nivel) | +0.000 (no cambia el nivel) | +0.000 (no cambia el nivel) | +0.000 (no cambia el nivel) |
| e. sin detección (d1) | +0.029 [-0.001, +0.066] | +0.037 [+0.003, +0.076] | +0.009 [-0.032, +0.057] | -0.017 [-0.060, +0.024] |
| e. sin tamaño y crecimiento (d2 a d4) | +0.015 [-0.025, +0.054] | +0.023 [-0.037, +0.092] | -0.010 [-0.044, +0.026] | +0.001 [-0.035, +0.040] |
| e. sin posición y movimiento (d5 a d7) | +0.008 [-0.040, +0.057] | +0.016 [-0.047, +0.066] | +0.020 [-0.016, +0.059] | -0.003 [-0.045, +0.040] |
| e. sin comportamiento en el tiempo (d8 y d9) | +0.020 [-0.013, +0.050] | -0.001 [-0.032, +0.032] | -0.001 [-0.038, +0.036] | +0.009 [-0.031, +0.053] |
| e. sin humo (d10 a d12) | +0.049 [-0.004, +0.099] | +0.008 [-0.040, +0.067] | -0.003 [-0.056, +0.064] | -0.058 [-0.110, -0.011] |
| e. sin escena (d13) | +0.024 [-0.010, +0.053] | +0.002 [-0.036, +0.042] | +0.004 [-0.025, +0.040] | +0.017 [-0.018, +0.057] |
| e. sin estadísticas de detección del fotograma | +0.001 [-0.025, +0.030] | +0.010 [-0.024, +0.045] | -0.003 [-0.033, +0.034] | -0.005 [-0.032, +0.025] |
| f. sin compensación de cámara | +0.026 [-0.008, +0.061] | -0.015 [-0.047, +0.022] | -0.018 [-0.051, +0.013] | -0.033 [-0.068, -0.002] |

Variante d (umbral fijo 0.5, sin reentrenar), semillas 42 a 44:

| modelo | referencia | falsas alarmas en controlados: conformal | umbral fijo | EWT mediana: conformal (s) | umbral fijo (s) | antes del cambio: conformal | umbral fijo |
|---|---|---|---|---|---|---|---|
| GRU | IA | 0.222 | 0.378 | -2.7 | 7.9 | 8.3 | 13.3 |
| GRU | humana | 0.207 | 0.322 | 20.5 | 38.4 | 11.3 | 13.7 |
| CfC | IA | 0.233 | 0.422 | 0.7 | 8.3 | 8.3 | 12.0 |
| CfC | humana | 0.184 | 0.368 | 23.6 | 37.0 | 12.3 | 13.3 |

## Holm

Familia global: 212 comparaciones pareadas. Valor p por la aproximacion normal del bootstrap.

- Con el IC sin corregir no incluyen el cero 25 de 212; despues de Holm siguen significativas 0.
- ablaciones: 44 comparaciones; sin corregir 4; Holm global 0; Holm dentro del bloque 0.
- fase F (3 semillas): 8 comparaciones; sin corregir 0; Holm global 0; Holm dentro del bloque 0.
- fase F (5 semillas y Mamba): 16 comparaciones; sin corregir 1; Holm global 0; Holm dentro del bloque 0.
- robustez: 144 comparaciones; sin corregir 20; Holm global 0; Holm dentro del bloque 0.

| bloque | comparacion | diferencia | IC 95 % | p | p de Holm |
|---|---|---|---|---|---|
| robustez | parte1, caida de F1 de gradient boosting con quitar_25, IA | +0.020 | +0.009 a +0.032 | 1.05e-03 | 2.22e-01 |
| robustez | parte1, caida relativa de kappa GRU - CfC con quitar_75, IA | +0.435 | +0.149 a +0.745 | 3.87e-03 | 8.17e-01 |
| robustez | reentrenado - congelado, kappa de GRU con quitar_75, IA | +0.148 | +0.042 a +0.247 | 4.38e-03 | 9.19e-01 |
| robustez | reentrenado - congelado, kappa de CfC con quitar_50, humana | +0.039 | +0.011 a +0.067 | 5.21e-03 | 1.00e+00 |
| robustez | reentrenado - congelado, kappa de GRU con quitar_75, humana | +0.143 | +0.036 a +0.243 | 7.64e-03 | 1.00e+00 |
| ablaciones | a. sin contexto de cámara en la compuerta, CfC, F1, IA | +0.063 | +0.019 a +0.112 | 8.37e-03 | 1.00e+00 |
| fase F (5 semillas y Mamba) | Mamba - CfC, f1, humana (5 semillas) | +0.081 | +0.019 a +0.138 | 1.05e-02 | 1.00e+00 |
| robustez | reentrenado - congelado, f1 de GRU con quitar_75, humana | +0.097 | +0.020 a +0.172 | 1.47e-02 | 1.00e+00 |
| robustez | reentrenado - congelado, f1 de LSTM con quitar_75, IA | +0.052 | +0.012 a +0.097 | 2.06e-02 | 1.00e+00 |
| robustez | parte1, caida relativa de f1 GRU - CfC con quitar_75, IA | +0.185 | +0.017 a +0.323 | 2.08e-02 | 1.00e+00 |
| robustez | reentrenado - congelado, f1 de GRU con quitar_75, IA | +0.088 | +0.011 a +0.165 | 2.28e-02 | 1.00e+00 |
| robustez | parte1, caida relativa de kappa GRU - CfC con quitar_25, IA | +0.206 | +0.010 a +0.368 | 2.43e-02 | 1.00e+00 |
| ablaciones | e. sin humo (d10 a d12), CfC, F1, humana | -0.058 | -0.110 a -0.011 | 2.49e-02 | 1.00e+00 |
| robustez | parte1, caida de F1 de gradient boosting con quitar_25, humana | +0.015 | +0.002 a +0.030 | 2.64e-02 | 1.00e+00 |
| robustez | parte1, caida relativa de kappa GRU - CfC con quitar_75, humana | +0.355 | +0.020 a +0.643 | 2.69e-02 | 1.00e+00 |
| robustez | parte1, caida de F1 de GRU con quitar_75, IA | -0.110 | -0.207 a -0.012 | 2.97e-02 | 1.00e+00 |
| robustez | parte2, caida de F1 de LSTM con quitar_50, humana | +0.026 | +0.000 a +0.050 | 3.76e-02 | 1.00e+00 |
| robustez | parte1, caida relativa de f1 GRU - CfC con quitar_75, humana | +0.176 | +0.006 a +0.336 | 3.94e-02 | 1.00e+00 |
| robustez | parte1, caida de F1 de LSTM con quitar_75, IA | -0.037 | -0.073 a -0.004 | 3.99e-02 | 1.00e+00 |
| robustez | parte1, caida relativa de f1 GRU - CfC con quitar_25, IA | +0.095 | +0.001 a +0.191 | 4.48e-02 | 1.00e+00 |
| ablaciones | e. sin detección (d1), GRU, F1, humana | +0.037 | +0.003 a +0.076 | 4.80e-02 | 1.00e+00 |
| robustez | parte2, caida de F1 de LSTM con quitar_50, IA | +0.022 | +0.001 a +0.047 | 5.40e-02 | 1.00e+00 |
| robustez | reentrenado - congelado, f1 de GRU con quitar_25, IA | +0.037 | +0.000 a +0.075 | 5.44e-02 | 1.00e+00 |
| robustez | parte1, caida de F1 de regresión logística con quitar_50, IA | +0.027 | +0.002 a +0.060 | 5.47e-02 | 1.00e+00 |
| ablaciones | f. sin compensación de cámara, CfC, F1, humana | -0.033 | -0.068 a -0.002 | 6.00e-02 | 1.00e+00 |

## Camara movil contra fija (criterio del 2026-11-13; 5 semillas)

| modelo | referencia | F1 movil [IC] | F1 fija [IC] | kappa movil [IC] | kappa fija [IC] |
|---|---|---|---|---|---|
| CfC | IA | 0.455 [0.378, 0.527] | 0.451 [0.331, 0.543] | 0.319 [0.179, 0.461] | 0.311 [0.069, 0.504] |
| CfC | humana | 0.401 [0.314, 0.477] | 0.501 [0.384, 0.588] | 0.257 [0.133, 0.368] | 0.398 [0.223, 0.543] |
| GRU | IA | 0.477 [0.376, 0.571] | 0.569 [0.418, 0.683] | 0.320 [0.151, 0.484] | 0.412 [0.135, 0.639] |
| GRU | humana | 0.466 [0.376, 0.561] | 0.554 [0.399, 0.668] | 0.284 [0.139, 0.421] | 0.402 [0.144, 0.613] |
| LSTM | IA | 0.470 [0.389, 0.548] | 0.451 [0.305, 0.561] | 0.353 [0.203, 0.505] | 0.286 [-0.004, 0.488] |
| LSTM | humana | 0.427 [0.331, 0.512] | 0.494 [0.361, 0.597] | 0.302 [0.180, 0.423] | 0.372 [0.158, 0.536] |
| Mamba | IA | 0.502 [0.396, 0.597] | 0.544 [0.426, 0.645] | 0.368 [0.199, 0.529] | 0.407 [0.215, 0.582] |
| Mamba | humana | 0.496 [0.403, 0.588] | 0.603 [0.458, 0.686] | 0.326 [0.193, 0.471] | 0.451 [0.250, 0.609] |
| gradient boosting | IA | 0.472 [0.362, 0.586] | 0.402 [0.302, 0.507] | 0.355 [0.198, 0.515] | 0.190 [0.006, 0.369] |
| gradient boosting | humana | 0.404 [0.282, 0.512] | 0.499 [0.339, 0.613] | 0.283 [0.140, 0.418] | 0.316 [0.105, 0.491] |
| regresión logística | IA | 0.556 [0.437, 0.668] | 0.435 [0.305, 0.539] | 0.376 [0.182, 0.564] | 0.233 [-0.047, 0.426] |
| regresión logística | humana | 0.512 [0.401, 0.622] | 0.551 [0.378, 0.648] | 0.353 [0.195, 0.518] | 0.370 [0.103, 0.553] |
| regla sin aprendizaje | IA | 0.390 [0.271, 0.500] | 0.473 [0.304, 0.566] | 0.243 [0.092, 0.413] | 0.316 [0.118, 0.475] |
| regla sin aprendizaje | humana | 0.304 [0.188, 0.408] | 0.322 [0.204, 0.426] | 0.157 [0.042, 0.281] | 0.171 [0.027, 0.318] |

Videos por grupo: IA movil 39; IA fija 31; humana movil 39; humana fija 30.

## E8 con Mamba (validacion, 5 semillas)

| variante | F1 macro | falsas alarmas en controlados | sin fuego | cambios por minuto | EWT mediana (s) | antes del cambio |
|---|---|---|---|---|---|---|
| conformal, histeresis 2 s (elegida) | 0.563 | 0.096 | 0.000 | 1.75 | -10.2 | 1.0 de 6 |
| umbral fijo 0.5 | 0.563 | 0.568 | 0.000 | 1.75 | 15.4 | 3.6 de 6 |
| histeresis 0 s | 0.594 | 0.096 | 0.000 | 5.58 | -10.2 | 1.0 de 6 |
| histeresis 1 s | 0.584 | 0.096 | 0.000 | 2.56 | -10.2 | 1.0 de 6 |
| histeresis 3 s | 0.553 | 0.096 | 0.000 | 1.39 | -10.2 | 1.0 de 6 |

## Eleccion de Mamba en validacion

- Peso A (factor 1): F1 suavizado con la semilla 42 0.001: 0.5464, 0.003: 0.5420; tasa elegida 0.001; F1 final en validacion (semillas 42 a 44) 0.564, 0.561, 0.528 (media 0.551).
- Peso B (factor 2): F1 suavizado con la semilla 42 0.001: 0.5382, 0.003: 0.5593; tasa elegida 0.003; F1 final en validacion (semillas 42 a 44) 0.517, 0.630, 0.543 (media 0.563).
- Elegido: peso B.

## Regla sin aprendizaje

Umbrales elegidos en validacion: fuego 0.0221 y 0.0851; humo 0.0911 y 0.0911 (fraccion del fotograma). F1 macro final en validacion 0.573. Umbral de alerta (riesgo sin tope) 10.439.

