# Evaluacion en la prueba con la configuracion congelada

Generado por `python evaluar_prueba.py`. Configuracion: `resultados/configuracion_final.json` (2026-11-13). Modelos temporales: media +- desviacion de 3 semillas.

- Videos: 70 de fuego con referencia de la IA, 69 con referencia humana (29 con tramo util y cambios marcados reproduciendo el video; el resto solo tiene un nivel para todo el video) y 32 sin fuego.
- Nivel por segundo: nivel final (umbral 0.5 + histeresis de 2 s), mayoria de los fotogramas del segundo, dentro del tramo util de cada referencia.
- Las dos referencias pesan igual. Localizacion, cambios, tiempo al peligro y adelanto contra la referencia humana usan solo los 29 videos con cambios marcados.
- Clase mayoritaria: incontrolado (la mas frecuente por segundo en entrenamiento). No tiene alerta.

## Nivel por segundo, referencia de la IA (70 videos)

| modelo | F1 macro | IC 95 % | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | IC 95 % | cambios de nivel por video | F1 macro antes de la histeresis |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.460 +- 0.002 | 0.393 a 0.522 | 0.507 +- 0.039 | 0.252 +- 0.053 | 0.621 +- 0.023 | 0.478 +- 0.014 | 0.319 +- 0.036 | 0.183 a 0.442 | 2.35 +- 0.10 | 0.470 +- 0.008 |
| GRU | 0.519 +- 0.009 | 0.433 a 0.603 | 0.561 +- 0.022 | 0.386 +- 0.032 | 0.609 +- 0.025 | 0.524 +- 0.011 | 0.371 +- 0.029 | 0.225 a 0.516 | 1.79 +- 0.11 | 0.529 +- 0.009 |
| LSTM | 0.460 +- 0.021 | 0.396 a 0.526 | 0.536 +- 0.006 | 0.232 +- 0.031 | 0.611 +- 0.039 | 0.491 +- 0.019 | 0.323 +- 0.022 | 0.204 a 0.438 | 2.83 +- 0.24 | 0.476 +- 0.020 |
| gradient boosting | 0.442 | 0.368 a 0.525 | 0.510 | 0.212 | 0.605 | 0.476 | 0.292 | 0.178 a 0.413 | 3.11 | 0.445 |
| regresion logistica | 0.491 | 0.407 a 0.570 | 0.475 | 0.387 | 0.610 | 0.490 | 0.311 | 0.158 a 0.449 | 2.86 | 0.484 |
| clase mayoritaria | 0.180 | 0.135 a 0.219 | 0.000 | 0.000 | 0.541 | 0.371 | 0.000 | 0.000 a 0.000 | 0.00 | 0.180 |

Diferencias pareadas (bootstrap de videos, mismos remuestreos):

| diferencia | F1 macro | IC 95 % | remuestreos con diferencia positiva | kappa | IC 95 % | remuestreos con diferencia positiva |
|---|---|---|---|---|---|---|
| GRU - CfC | +0.059 | -0.006 a +0.128 | 95.6 % | +0.052 | -0.038 a +0.153 | 87.7 % |
| GRU - gradient boosting | +0.077 | -0.008 a +0.151 | 95.7 % | +0.079 | -0.054 a +0.207 | 87.5 % |

Matrices de confusion en segundos (filas: referencia; columnas: modelo; media de las semillas):

| modelo | referencia | controlado | incipiente | incontrolado |
|---|---|---|---|---|
| CfC | controlado | 935 | 363 | 166 |
|  | incipiente | 894 | 331 | 271 |
|  | incontrolado | 374 | 386 | 983 |
| GRU | controlado | 866 | 436 | 162 |
|  | incipiente | 487 | 573 | 436 |
|  | incontrolado | 262 | 455 | 1025 |
| LSTM | controlado | 1083 | 229 | 151 |
|  | incipiente | 963 | 261 | 272 |
|  | incontrolado | 527 | 253 | 963 |
| gradient boosting | controlado | 905 | 202 | 357 |
|  | incipiente | 825 | 236 | 435 |
|  | incontrolado | 353 | 291 | 1099 |
| regresion logistica | controlado | 780 | 560 | 124 |
|  | incipiente | 647 | 594 | 255 |
|  | incontrolado | 395 | 417 | 931 |
| clase mayoritaria | controlado | 0 | 0 | 1464 |
|  | incipiente | 0 | 0 | 1496 |
|  | incontrolado | 0 | 0 | 1743 |

## Nivel por segundo, referencia humana (69 videos)

| modelo | F1 macro | IC 95 % | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | IC 95 % | cambios de nivel por video | F1 macro antes de la histeresis |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.475 +- 0.022 | 0.410 a 0.540 | 0.587 +- 0.052 | 0.328 +- 0.022 | 0.510 +- 0.035 | 0.477 +- 0.028 | 0.345 +- 0.057 | 0.241 a 0.449 | 2.31 +- 0.10 | 0.472 +- 0.024 |
| GRU | 0.497 +- 0.019 | 0.416 a 0.579 | 0.570 +- 0.029 | 0.419 +- 0.026 | 0.504 +- 0.035 | 0.492 +- 0.018 | 0.352 +- 0.040 | 0.238 a 0.472 | 1.84 +- 0.09 | 0.502 +- 0.017 |
| LSTM | 0.460 +- 0.007 | 0.391 a 0.528 | 0.607 +- 0.012 | 0.282 +- 0.015 | 0.493 +- 0.029 | 0.475 +- 0.004 | 0.340 +- 0.013 | 0.244 a 0.440 | 2.86 +- 0.17 | 0.464 +- 0.005 |
| gradient boosting | 0.451 | 0.362 a 0.531 | 0.597 | 0.293 | 0.463 | 0.461 | 0.320 | 0.218 a 0.426 | 3.12 | 0.434 |
| regresion logistica | 0.527 | 0.433 a 0.604 | 0.567 | 0.475 | 0.540 | 0.523 | 0.365 | 0.220 a 0.485 | 2.93 | 0.504 |
| clase mayoritaria | 0.100 | 0.064 a 0.138 | 0.001 | 0.000 | 0.299 | 0.176 | 0.000 | 0.000 a 0.001 | 0.01 | 0.100 |

Solo los 29 videos con cambios marcados:

| modelo | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal |
|---|---|---|---|---|---|---|
| CfC | 0.428 +- 0.018 | 0.425 +- 0.042 | 0.343 +- 0.028 | 0.515 +- 0.037 | 0.415 +- 0.015 | 0.293 +- 0.047 |
| GRU | 0.446 +- 0.026 | 0.357 +- 0.048 | 0.485 +- 0.017 | 0.495 +- 0.039 | 0.463 +- 0.019 | 0.269 +- 0.048 |
| LSTM | 0.405 +- 0.026 | 0.439 +- 0.027 | 0.278 +- 0.015 | 0.497 +- 0.063 | 0.392 +- 0.023 | 0.286 +- 0.044 |
| gradient boosting | 0.429 | 0.475 | 0.320 | 0.491 | 0.418 | 0.306 |
| regresion logistica | 0.525 | 0.455 | 0.567 | 0.553 | 0.535 | 0.353 |
| clase mayoritaria | 0.108 | 0.000 | 0.000 | 0.324 | 0.193 | 0.000 |

Diferencias pareadas (bootstrap de videos, mismos remuestreos):

| diferencia | F1 macro | IC 95 % | remuestreos con diferencia positiva | kappa | IC 95 % | remuestreos con diferencia positiva |
|---|---|---|---|---|---|---|
| GRU - CfC | +0.023 | -0.037 a +0.088 | 74.5 % | +0.007 | -0.078 a +0.094 | 53.5 % |
| GRU - gradient boosting | +0.046 | -0.039 a +0.137 | 84.0 % | +0.032 | -0.084 a +0.158 | 64.8 % |

Matrices de confusion en segundos (filas: referencia; columnas: modelo; media de las semillas):

| modelo | referencia | controlado | incipiente | incontrolado |
|---|---|---|---|---|
| CfC | controlado | 1100 | 339 | 117 |
|  | incipiente | 990 | 550 | 751 |
|  | incontrolado | 83 | 160 | 579 |
| GRU | controlado | 893 | 518 | 145 |
|  | incipiente | 625 | 783 | 883 |
|  | incontrolado | 58 | 141 | 623 |
| LSTM | controlado | 1238 | 189 | 129 |
|  | incipiente | 1134 | 427 | 731 |
|  | incontrolado | 149 | 119 | 554 |
| gradient boosting | controlado | 1083 | 184 | 289 |
|  | incipiente | 873 | 437 | 981 |
|  | incontrolado | 118 | 73 | 631 |
| regresion logistica | controlado | 947 | 535 | 74 |
|  | incipiente | 701 | 912 | 678 |
|  | incontrolado | 135 | 105 | 582 |
| clase mayoritaria | controlado | 1 | 0 | 1555 |
|  | incipiente | 0 | 0 | 2291 |
|  | incontrolado | 0 | 0 | 822 |

## Localizacion temporal, referencia de la IA (70 videos)

mAP de segmentos por nivel. Puntuacion de un segmento: probabilidad media de su nivel. AP con interpolacion de todos los puntos.

| modelo | mAP tIoU 0.3 | controlado | incipiente | incontrolado | mAP tIoU 0.5 | controlado | incipiente | incontrolado | mAP tIoU 0.7 | controlado | incipiente | incontrolado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.289 +- 0.033 | 0.358 +- 0.086 | 0.067 +- 0.025 | 0.442 +- 0.034 | 0.216 +- 0.039 | 0.322 +- 0.089 | 0.012 +- 0.009 | 0.315 +- 0.042 | 0.159 +- 0.036 | 0.224 +- 0.098 | 0.002 +- 0.002 | 0.249 +- 0.019 |
| GRU | 0.403 +- 0.023 | 0.444 +- 0.042 | 0.207 +- 0.055 | 0.557 +- 0.041 | 0.299 +- 0.023 | 0.392 +- 0.050 | 0.060 +- 0.014 | 0.446 +- 0.033 | 0.235 +- 0.006 | 0.339 +- 0.013 | 0.015 +- 0.012 | 0.350 +- 0.013 |
| LSTM | 0.329 +- 0.018 | 0.458 +- 0.095 | 0.046 +- 0.007 | 0.483 +- 0.067 | 0.258 +- 0.022 | 0.419 +- 0.106 | 0.007 +- 0.002 | 0.348 +- 0.051 | 0.194 +- 0.022 | 0.300 +- 0.095 | 0.002 +- 0.001 | 0.278 +- 0.033 |
| gradient boosting | 0.246 | 0.362 | 0.048 | 0.329 | 0.204 | 0.301 | 0.019 | 0.292 | 0.130 | 0.185 | 0.000 | 0.206 |
| regresion logistica | 0.277 | 0.302 | 0.155 | 0.374 | 0.204 | 0.270 | 0.096 | 0.245 | 0.147 | 0.221 | 0.040 | 0.180 |
| clase mayoritaria | 0.116 | 0.000 | 0.000 | 0.347 | 0.091 | 0.000 | 0.000 | 0.274 | 0.043 | 0.000 | 0.000 | 0.128 |

## Localizacion temporal, referencia humana (29 videos)

mAP de segmentos por nivel. Puntuacion de un segmento: probabilidad media de su nivel. AP con interpolacion de todos los puntos.

| modelo | mAP tIoU 0.3 | controlado | incipiente | incontrolado | mAP tIoU 0.5 | controlado | incipiente | incontrolado | mAP tIoU 0.7 | controlado | incipiente | incontrolado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.187 +- 0.040 | 0.212 +- 0.048 | 0.064 +- 0.011 | 0.283 +- 0.083 | 0.103 +- 0.048 | 0.141 +- 0.072 | 0.013 +- 0.004 | 0.154 +- 0.073 | 0.044 +- 0.033 | 0.040 +- 0.037 | 0.003 +- 0.002 | 0.088 +- 0.065 |
| GRU | 0.296 +- 0.038 | 0.146 +- 0.044 | 0.329 +- 0.047 | 0.414 +- 0.041 | 0.136 +- 0.015 | 0.129 +- 0.060 | 0.086 +- 0.015 | 0.193 +- 0.085 | 0.066 +- 0.037 | 0.075 +- 0.094 | 0.026 +- 0.007 | 0.096 +- 0.021 |
| LSTM | 0.218 +- 0.026 | 0.299 +- 0.022 | 0.028 +- 0.011 | 0.328 +- 0.082 | 0.120 +- 0.032 | 0.163 +- 0.093 | 0.006 +- 0.003 | 0.191 +- 0.036 | 0.042 +- 0.009 | 0.046 +- 0.043 | 0.001 +- 0.002 | 0.078 +- 0.037 |
| gradient boosting | 0.193 | 0.200 | 0.027 | 0.353 | 0.125 | 0.200 | 0.000 | 0.173 | 0.028 | 0.016 | 0.000 | 0.067 |
| regresion logistica | 0.172 | 0.074 | 0.171 | 0.272 | 0.069 | 0.017 | 0.018 | 0.172 | 0.021 | 0.000 | 0.002 | 0.061 |
| clase mayoritaria | 0.035 | 0.000 | 0.000 | 0.106 | 0.006 | 0.000 | 0.000 | 0.019 | 0.000 | 0.000 | 0.000 | 0.000 |

## Cambios de nivel, referencia de la IA (70 videos)

Un cambio a un nivel se da por detectado si el nivel final del modelo cruza hacia ese nivel (o mas) dentro del tramo util; el error es el instante del cruce mas cercano menos el de la referencia (positivo = el modelo llega tarde). Desde el inicio = el modelo ya estaba en ese nivel o por encima; no llega = nunca lo alcanza.

| modelo | cambio | cambios | detectados | mediana del error absoluto (s) | mediana con signo (s) | dentro de 5 s | dentro de 10 s | desde el inicio | no llega | cambios de mas por video |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | a incipiente | 11 | 8.0 +- 0.8 | 11.6 +- 2.6 | 10.2 +- 3.9 | 2.3 +- 1.2 | 4.0 +- 0.8 | 2.0 +- 1.4 | 1.0 +- 0.8 | 2.17 +- 0.12 |
| CfC | a incontrolado | 21 | 14.3 +- 0.9 | 16.9 +- 1.2 | 16.5 +- 1.3 | 2.0 +- 0.8 | 4.0 +- 1.4 | 1.7 +- 0.5 | 5.0 +- 0.8 | 2.17 +- 0.12 |
| GRU | a incipiente | 11 | 6.7 +- 0.5 | 4.2 +- 1.2 | 0.4 +- 0.8 | 3.7 +- 0.9 | 4.7 +- 0.9 | 4.3 +- 0.5 | 0.0 +- 0.0 | 1.51 +- 0.13 |
| GRU | a incontrolado | 21 | 16.0 +- 1.4 | 27.8 +- 0.6 | 11.2 +- 5.1 | 2.0 +- 0.8 | 5.0 +- 0.8 | 0.0 +- 0.0 | 5.0 +- 1.4 | 1.51 +- 0.13 |
| LSTM | a incipiente | 11 | 9.0 +- 1.4 | 14.8 +- 8.3 | 13.5 +- 9.5 | 2.3 +- 1.7 | 4.3 +- 1.2 | 2.0 +- 1.4 | 0.0 +- 0.0 | 2.53 +- 0.19 |
| LSTM | a incontrolado | 21 | 15.7 +- 0.5 | 17.4 +- 5.8 | 14.3 +- 8.4 | 3.3 +- 1.2 | 5.3 +- 1.2 | 0.0 +- 0.0 | 5.3 +- 0.5 | 2.53 +- 0.19 |
| gradient boosting | a incipiente | 11 | 10.0 | 9.3 | 6.9 | 3.0 | 5.0 | 1.0 | 0.0 | 2.71 |
| gradient boosting | a incontrolado | 21 | 19.0 | 8.3 | 4.2 | 7.0 | 10.0 | 2.0 | 0.0 | 2.71 |
| regresion logistica | a incipiente | 11 | 9.0 | 16.5 | 2.2 | 3.0 | 3.0 | 1.0 | 1.0 | 2.53 |
| regresion logistica | a incontrolado | 21 | 15.0 | 12.9 | 12.9 | 3.0 | 5.0 | 1.0 | 5.0 | 2.53 |
| clase mayoritaria | a incipiente | 11 | 0.0 | - | - | 0.0 | 0.0 | 11.0 | 0.0 | 0.00 |
| clase mayoritaria | a incontrolado | 21 | 0.0 | - | - | 0.0 | 0.0 | 21.0 | 0.0 | 0.00 |

## Cambios de nivel, referencia humana (29 videos)

Un cambio a un nivel se da por detectado si el nivel final del modelo cruza hacia ese nivel (o mas) dentro del tramo util; el error es el instante del cruce mas cercano menos el de la referencia (positivo = el modelo llega tarde). Desde el inicio = el modelo ya estaba en ese nivel o por encima; no llega = nunca lo alcanza.

| modelo | cambio | cambios | detectados | mediana del error absoluto (s) | mediana con signo (s) | dentro de 5 s | dentro de 10 s | desde el inicio | no llega | cambios de mas por video |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | a incipiente | 4 | 4.0 +- 0.0 | 8.8 +- 2.1 | -2.0 +- 6.6 | 1.0 +- 0.8 | 2.3 +- 0.5 | 0.0 +- 0.0 | 0.0 +- 0.0 | 3.98 +- 0.28 |
| CfC | a incontrolado | 16 | 12.7 +- 0.5 | 13.6 +- 5.8 | -5.1 +- 3.1 | 4.0 +- 1.6 | 6.3 +- 1.2 | 1.0 +- 0.0 | 2.3 +- 0.5 | 3.98 +- 0.28 |
| GRU | a incipiente | 4 | 3.0 +- 0.0 | 9.4 +- 5.9 | -2.9 +- 5.5 | 1.0 +- 0.8 | 1.0 +- 0.8 | 1.0 +- 0.0 | 0.0 +- 0.0 | 2.32 +- 0.11 |
| GRU | a incontrolado | 16 | 13.7 +- 0.9 | 17.5 +- 4.6 | -0.5 +- 2.3 | 3.3 +- 1.7 | 4.0 +- 1.6 | 0.3 +- 0.5 | 2.0 +- 0.8 | 2.32 +- 0.11 |
| LSTM | a incipiente | 4 | 4.0 +- 0.0 | 15.9 +- 7.2 | 14.0 +- 8.3 | 0.7 +- 0.5 | 1.3 +- 0.5 | 0.0 +- 0.0 | 0.0 +- 0.0 | 4.41 +- 0.20 |
| LSTM | a incontrolado | 16 | 13.3 +- 0.9 | 17.6 +- 2.3 | -6.8 +- 1.2 | 3.7 +- 1.2 | 4.7 +- 1.2 | 0.0 +- 0.0 | 2.7 +- 0.9 | 4.41 +- 0.20 |
| gradient boosting | a incipiente | 4 | 4.0 | 7.2 | 7.2 | 1.0 | 4.0 | 0.0 | 0.0 | 5.10 |
| gradient boosting | a incontrolado | 16 | 14.0 | 4.8 | -3.1 | 7.0 | 9.0 | 1.0 | 1.0 | 5.10 |
| regresion logistica | a incipiente | 4 | 4.0 | 5.6 | -3.9 | 2.0 | 3.0 | 0.0 | 0.0 | 5.07 |
| regresion logistica | a incontrolado | 16 | 13.0 | 7.4 | -6.7 | 2.0 | 8.0 | 1.0 | 2.0 | 5.07 |
| clase mayoritaria | a incipiente | 4 | 0.0 | - | - | 0.0 | 0.0 | 4.0 | 0.0 | 0.00 |
| clase mayoritaria | a incontrolado | 16 | 0.0 | - | - | 0.0 | 0.0 | 16.0 | 0.0 | 0.00 |

## Los 4 videos con paso de controlado a incipiente, uno por uno

Instante (s) del primer paso del nivel final del modelo a incipiente o mas, por semilla (42 / 43 / 44).

| video | IA: inicial, a incipiente, a incontrolado | humano: inicial, a incipiente, a incontrolado | CfC | GRU | LSTM | gradient boosting | regresion logistica |
|---|---|---|---|---|---|---|---|
| 1. 700_F_1633820181_qwiNmqTW57MMB3OTqVnQuTK | controlado, 25.63, - | controlado, -, - | desde el inicio / desde el inicio / desde el inicio | desde el inicio / desde el inicio / desde el inicio | desde el inicio / desde el inicio / desde el inicio | 14.5 | desde el inicio |
| 2. Har du en fungerande brandvarnare_.mp4 | controlado, 16.33, - | incipiente, -, 20.53 | 18.7 / nunca / desde el inicio | desde el inicio / desde el inicio / desde el inicio | desde el inicio / desde el inicio / desde el inicio | desde el inicio | desde el inicio |
| 3. Wooden Model Ship Battleship Jean Bart B | controlado, 23.07, - | controlado, 263.77, - | 17.0 / 251.9 / 108.3 | desde el inicio / desde el inicio / desde el inicio | desde el inicio / 130.3 / desde el inicio | 4.5 | 4.5 |
| 4. videoplayback (3).mp4 | controlado, 34.0, - | incipiente, -, - | 47.0 / 35.7 / desde el inicio | desde el inicio / desde el inicio / desde el inicio | desde el inicio / desde el inicio / desde el inicio | 8.5 | desde el inicio |

## Tiempo al peligro

MAE de tau en segundos, en los fotogramas anteriores al cambio a incontrolado. La fila de la clase mayoritaria es una constante: la mediana de tau en entrenamiento (52.7 s).

| modelo | MAE, referencia de la IA (s) | videos | fotogramas | MAE, referencia humana (s) | videos | fotogramas |
|---|---|---|---|---|---|---|
| CfC | 18.9 +- 1.2 | 21 | 5802 | 63.4 +- 2.8 | 16 | 10293 |
| GRU | 25.7 +- 2.3 | 21 | 5802 | 58.4 +- 0.7 | 16 | 10293 |
| LSTM | 20.1 +- 1.9 | 21 | 5802 | 60.1 +- 2.3 | 16 | 10293 |
| constante (mediana de entrenamiento) | 30.4 | 21 | 5802 | 48.1 | 16 | 10293 |

## Alerta con alfa 0.1 (configuracion final)

EWT = instante del cambio a incontrolado menos instante de la primera alerta; positivo = la alerta llega antes.

Referencia de la IA:

| modelo | umbral | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | videos con cambio | EWT mediana (s) | EWT cuartil 1 (s) | EWT cuartil 3 (s) | con alerta antes del cambio | sin alerta |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.723 +- 0.064 | 0.000 +- 0.000 (0.0 +- 0.0 de 32) | 0.233 +- 0.072 (7.0 +- 2.2 de 30) | 0.700 +- 0.020 (de 40) | 21 | 0.7 +- 3.8 | -34.1 +- 10.1 | 14.2 +- 1.0 | 8.3 +- 0.9 | 4.0 +- 0.8 |
| GRU | 0.768 +- 0.113 | 0.000 +- 0.000 (0.0 +- 0.0 de 32) | 0.222 +- 0.031 (6.7 +- 0.9 de 30) | 0.750 +- 0.020 (de 40) | 21 | -2.7 +- 10.6 | -29.5 +- 13.1 | 14.4 +- 9.2 | 8.3 +- 2.1 | 3.3 +- 0.9 |
| LSTM | 0.599 +- 0.025 | 0.000 +- 0.000 (0.0 +- 0.0 de 32) | 0.233 +- 0.072 (7.0 +- 2.2 de 30) | 0.708 +- 0.047 (de 40) | 21 | 4.4 +- 4.7 | -39.7 +- 12.0 | 14.0 +- 3.5 | 9.0 +- 1.4 | 4.7 +- 0.5 |
| gradient boosting | 0.869 | 0.031 (1.0 de 32) | 0.267 (8.0 de 30) | 0.775 (de 40) | 21 | 3.8 | -15.8 | 16.3 | 12.0 | 1.0 |
| regresion logistica | 0.832 | 0.000 (0.0 de 32) | 0.100 (3.0 de 30) | 0.675 (de 40) | 21 | -10.5 | -51.3 | 1.7 | 5.0 | 5.0 |

Referencia humana:

| modelo | umbral | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | videos con cambio | EWT mediana (s) | EWT cuartil 1 (s) | EWT cuartil 3 (s) | con alerta antes del cambio | sin alerta |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.723 +- 0.064 | 0.000 +- 0.000 (0.0 +- 0.0 de 32) | 0.184 +- 0.065 (5.3 +- 1.9 de 29) | 0.733 +- 0.012 (de 40) | 16 | 23.6 +- 8.7 | 4.0 +- 0.8 | 69.5 +- 8.2 | 12.3 +- 0.5 | 2.0 +- 0.0 |
| GRU | 0.768 +- 0.113 | 0.000 +- 0.000 (0.0 +- 0.0 de 32) | 0.207 +- 0.028 (6.0 +- 0.8 de 29) | 0.758 +- 0.047 (de 40) | 16 | 20.5 +- 11.7 | -0.1 +- 2.5 | 67.1 +- 17.7 | 11.3 +- 2.1 | 1.3 +- 0.5 |
| LSTM | 0.599 +- 0.025 | 0.000 +- 0.000 (0.0 +- 0.0 de 32) | 0.230 +- 0.043 (6.7 +- 1.2 de 29) | 0.717 +- 0.031 (de 40) | 16 | 29.7 +- 4.0 | 1.6 +- 1.8 | 69.6 +- 10.0 | 10.7 +- 1.2 | 2.3 +- 0.5 |
| gradient boosting | 0.869 | 0.031 (1.0 de 32) | 0.207 (6.0 de 29) | 0.800 (de 40) | 16 | 26.6 | 3.6 | 94.5 | 13.0 | 1.0 |
| regresion logistica | 0.832 | 0.000 (0.0 de 32) | 0.069 (2.0 de 29) | 0.700 (de 40) | 16 | 13.2 | -3.3 | 48.4 | 10.0 | 2.0 |

## Alerta con alfa 0.05 (aparte)

EWT = instante del cambio a incontrolado menos instante de la primera alerta; positivo = la alerta llega antes.

Referencia de la IA:

| modelo | umbral | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | videos con cambio | EWT mediana (s) | EWT cuartil 1 (s) | EWT cuartil 3 (s) | con alerta antes del cambio | sin alerta |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.912 +- 0.102 | 0.000 +- 0.000 (0.0 +- 0.0 de 32) | 0.089 +- 0.103 (2.7 +- 3.1 de 30) | 0.433 +- 0.194 (de 40) | 21 | -9.6 +- 7.0 | -37.4 +- 8.2 | 9.7 +- 5.5 | 4.0 +- 2.9 | 12.0 +- 5.1 |
| GRU | 0.998 +- 0.001 | 0.000 +- 0.000 (0.0 +- 0.0 de 32) | 0.044 +- 0.016 (1.3 +- 0.5 de 30) | 0.608 +- 0.031 (de 40) | 21 | -32.1 +- 12.1 | -66.7 +- 11.8 | 3.2 +- 0.9 | 4.0 +- 0.0 | 8.7 +- 1.2 |
| LSTM | 0.978 +- 0.008 | 0.000 +- 0.000 (0.0 +- 0.0 de 32) | 0.022 +- 0.031 (0.7 +- 0.9 de 30) | 0.458 +- 0.024 (de 40) | 21 | -14.8 +- 4.3 | -55.0 +- 17.7 | 8.8 +- 3.8 | 3.0 +- 0.0 | 13.0 +- 0.8 |
| gradient boosting | 0.989 | 0.000 (0.0 de 32) | 0.033 (1.0 de 30) | 0.550 (de 40) | 21 | -5.7 | -78.7 | 19.8 | 4.0 | 10.0 |
| regresion logistica | 0.971 | 0.000 (0.0 de 32) | 0.000 (0.0 de 30) | 0.400 (de 40) | 21 | -58.6 | -87.5 | -4.4 | 2.0 | 12.0 |

Referencia humana:

| modelo | umbral | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | videos con cambio | EWT mediana (s) | EWT cuartil 1 (s) | EWT cuartil 3 (s) | con alerta antes del cambio | sin alerta |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.912 +- 0.102 | 0.000 +- 0.000 (0.0 +- 0.0 de 32) | 0.069 +- 0.074 (2.0 +- 2.2 de 29) | 0.450 +- 0.215 (de 40) | 16 | 20.3 +- 3.7 | 4.5 +- 2.1 | 39.3 +- 12.5 | 7.0 +- 4.3 | 8.0 +- 4.3 |
| GRU | 0.998 +- 0.001 | 0.000 +- 0.000 (0.0 +- 0.0 de 32) | 0.034 +- 0.000 (1.0 +- 0.0 de 29) | 0.617 +- 0.042 (de 40) | 16 | -3.4 +- 8.7 | -16.8 +- 9.8 | 40.7 +- 6.3 | 6.0 +- 0.8 | 4.7 +- 1.2 |
| LSTM | 0.978 +- 0.008 | 0.000 +- 0.000 (0.0 +- 0.0 de 32) | 0.023 +- 0.033 (0.7 +- 0.9 de 29) | 0.458 +- 0.024 (de 40) | 16 | 12.6 +- 2.6 | 3.9 +- 5.9 | 29.2 +- 2.1 | 5.3 +- 0.5 | 9.0 +- 0.8 |
| gradient boosting | 0.989 | 0.000 (0.0 de 32) | 0.034 (1.0 de 29) | 0.550 (de 40) | 16 | 20.2 | -8.1 | 46.2 | 6.0 | 6.0 |
| regresion logistica | 0.971 | 0.000 (0.0 de 32) | 0.000 (0.0 de 29) | 0.400 (de 40) | 16 | -8.3 | -19.0 | -1.7 | 2.0 | 8.0 |
