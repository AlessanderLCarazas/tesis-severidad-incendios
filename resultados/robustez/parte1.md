# Robustez, parte 1: fotogramas quitados en la prueba

Generado por `python evaluar_robustez.py --modelos congelados`. Modelos congelados (`resultados/configuracion_final.json`). Sin reentrenar ni recalibrar en la evaluacion: umbrales de CORAL en 0.5, histeresis de 2 s y umbral conformal de cada corrida. Vector de escena armado con `escena_v2.vectores_carpeta` y la normalizacion congelada del entrenamiento. Modelos temporales: media +- desviacion de 3 semillas.

- Caida = valor con fotogramas quitados menos valor en la prueba completa (con el mismo modelo).
- Nivel por segundo como en la fase F: un segundo sin fotogramas cuenta como controlado. Columna aparte: el segundo sin fotogramas toma el ultimo nivel decidido (lo que mostraria el sistema en vivo).
- Control: la corrida completa reproduce la fase F (diferencia maxima 0.0e+00).

| fraccion | fotogramas | usados en tramos de 10 o mas | cortes de escena | videos con corte | tramos cortos descartados | dt mediano (s) | dt percentil 95 (s) |
|---|---|---|---|---|---|---|---|
| completa | 51105 | 51076 | 45 | 13 | 7 | 0.10 | 0.12 |
| 25 % quitado | 38281 | 38247 | 48 | 16 | 8 | 0.10 | 0.30 |
| 50 % quitado | 25547 | 25488 | 45 | 15 | 14 | 0.17 | 0.50 |
| 75 % quitado | 12845 | 12750 | 48 | 18 | 22 | 0.30 | 1.10 |

## Nivel por segundo, referencia de la IA (70 videos)

| modelo | fraccion | F1 macro | caida | F1 incipiente | kappa lineal | caida | segundos sin fotogramas | F1 macro manteniendo el ultimo nivel | kappa manteniendo el ultimo nivel |
|---|---|---|---|---|---|---|---|---|---|
| CfC | completa | 0.460 +- 0.002 | +0.000 | 0.252 +- 0.053 | 0.319 +- 0.036 | +0.000 | 0 de 4703 | 0.460 +- 0.002 | 0.319 +- 0.036 |
| CfC | 25 % quitado | 0.460 +- 0.009 | -0.000 | 0.247 +- 0.040 | 0.323 +- 0.041 | +0.003 | 0 de 4703 | 0.460 +- 0.009 | 0.323 +- 0.041 |
| CfC | 50 % quitado | 0.462 +- 0.012 | +0.002 | 0.252 +- 0.063 | 0.330 +- 0.021 | +0.010 | 6 de 4703 | 0.463 +- 0.012 | 0.331 +- 0.021 |
| CfC | 75 % quitado | 0.448 +- 0.005 | -0.013 | 0.276 +- 0.052 | 0.288 +- 0.030 | -0.031 | 295 de 4703 | 0.458 +- 0.006 | 0.309 +- 0.031 |
| GRU | completa | 0.519 +- 0.009 | +0.000 | 0.386 +- 0.032 | 0.371 +- 0.029 | +0.000 | 0 de 4703 | 0.519 +- 0.009 | 0.371 +- 0.029 |
| GRU | 25 % quitado | 0.469 +- 0.021 | -0.050 | 0.336 +- 0.068 | 0.298 +- 0.039 | -0.073 | 0 de 4703 | 0.469 +- 0.021 | 0.298 +- 0.039 |
| GRU | 50 % quitado | 0.476 +- 0.012 | -0.042 | 0.361 +- 0.044 | 0.287 +- 0.029 | -0.084 | 6 de 4703 | 0.477 +- 0.012 | 0.288 +- 0.029 |
| GRU | 75 % quitado | 0.409 +- 0.022 | -0.110 | 0.345 +- 0.036 | 0.173 +- 0.052 | -0.198 | 295 de 4703 | 0.417 +- 0.023 | 0.187 +- 0.053 |
| LSTM | completa | 0.460 +- 0.021 | +0.000 | 0.232 +- 0.031 | 0.323 +- 0.022 | +0.000 | 0 de 4703 | 0.460 +- 0.021 | 0.323 +- 0.022 |
| LSTM | 25 % quitado | 0.465 +- 0.008 | +0.005 | 0.226 +- 0.012 | 0.339 +- 0.008 | +0.016 | 0 de 4703 | 0.465 +- 0.008 | 0.339 +- 0.008 |
| LSTM | 50 % quitado | 0.460 +- 0.010 | -0.000 | 0.213 +- 0.029 | 0.338 +- 0.015 | +0.015 | 6 de 4703 | 0.460 +- 0.010 | 0.339 +- 0.015 |
| LSTM | 75 % quitado | 0.423 +- 0.013 | -0.037 | 0.225 +- 0.056 | 0.267 +- 0.029 | -0.055 | 295 de 4703 | 0.434 +- 0.014 | 0.286 +- 0.030 |
| gradient boosting | completa | 0.442 | +0.000 | 0.212 | 0.292 | +0.000 | 0 de 4703 | 0.442 | 0.292 |
| gradient boosting | 25 % quitado | 0.462 | +0.020 | 0.242 | 0.316 | +0.024 | 0 de 4703 | 0.462 | 0.316 |
| gradient boosting | 50 % quitado | 0.450 | +0.008 | 0.213 | 0.308 | +0.016 | 6 de 4703 | 0.451 | 0.310 |
| gradient boosting | 75 % quitado | 0.431 | -0.012 | 0.211 | 0.262 | -0.030 | 295 de 4703 | 0.444 | 0.286 |
| regresion logistica | completa | 0.491 | +0.000 | 0.387 | 0.311 | +0.000 | 0 de 4703 | 0.491 | 0.311 |
| regresion logistica | 25 % quitado | 0.501 | +0.010 | 0.409 | 0.323 | +0.012 | 0 de 4703 | 0.501 | 0.323 |
| regresion logistica | 50 % quitado | 0.518 | +0.027 | 0.420 | 0.354 | +0.043 | 6 de 4703 | 0.519 | 0.356 |
| regresion logistica | 75 % quitado | 0.500 | +0.010 | 0.423 | 0.309 | -0.002 | 295 de 4703 | 0.512 | 0.335 |

## Nivel por segundo, referencia humana (69 videos)

| modelo | fraccion | F1 macro | caida | F1 incipiente | kappa lineal | caida | segundos sin fotogramas | F1 macro manteniendo el ultimo nivel | kappa manteniendo el ultimo nivel |
|---|---|---|---|---|---|---|---|---|---|
| CfC | completa | 0.475 +- 0.022 | +0.000 | 0.328 +- 0.022 | 0.345 +- 0.057 | +0.000 | 1 de 4669 | 0.475 +- 0.022 | 0.345 +- 0.057 |
| CfC | 25 % quitado | 0.477 +- 0.022 | +0.002 | 0.333 +- 0.014 | 0.349 +- 0.056 | +0.004 | 1 de 4669 | 0.477 +- 0.022 | 0.349 +- 0.056 |
| CfC | 50 % quitado | 0.470 +- 0.011 | -0.005 | 0.336 +- 0.022 | 0.326 +- 0.041 | -0.019 | 8 de 4669 | 0.470 +- 0.011 | 0.326 +- 0.041 |
| CfC | 75 % quitado | 0.484 +- 0.016 | +0.009 | 0.390 +- 0.007 | 0.315 +- 0.037 | -0.030 | 286 de 4669 | 0.494 +- 0.017 | 0.333 +- 0.037 |
| GRU | completa | 0.497 +- 0.019 | +0.000 | 0.419 +- 0.026 | 0.352 +- 0.040 | +0.000 | 1 de 4669 | 0.497 +- 0.019 | 0.352 +- 0.040 |
| GRU | 25 % quitado | 0.461 +- 0.013 | -0.036 | 0.375 +- 0.024 | 0.297 +- 0.038 | -0.055 | 1 de 4669 | 0.461 +- 0.013 | 0.297 +- 0.039 |
| GRU | 50 % quitado | 0.456 +- 0.004 | -0.041 | 0.354 +- 0.054 | 0.287 +- 0.030 | -0.065 | 8 de 4669 | 0.456 +- 0.004 | 0.287 +- 0.031 |
| GRU | 75 % quitado | 0.419 +- 0.025 | -0.078 | 0.396 +- 0.069 | 0.196 +- 0.068 | -0.156 | 286 de 4669 | 0.428 +- 0.028 | 0.210 +- 0.071 |
| LSTM | completa | 0.460 +- 0.007 | +0.000 | 0.282 +- 0.015 | 0.340 +- 0.013 | +0.000 | 1 de 4669 | 0.460 +- 0.007 | 0.340 +- 0.013 |
| LSTM | 25 % quitado | 0.465 +- 0.021 | +0.005 | 0.281 +- 0.040 | 0.349 +- 0.023 | +0.009 | 1 de 4669 | 0.465 +- 0.021 | 0.348 +- 0.023 |
| LSTM | 50 % quitado | 0.457 +- 0.031 | -0.004 | 0.273 +- 0.033 | 0.336 +- 0.044 | -0.004 | 8 de 4669 | 0.456 +- 0.031 | 0.336 +- 0.044 |
| LSTM | 75 % quitado | 0.437 +- 0.030 | -0.024 | 0.314 +- 0.020 | 0.278 +- 0.054 | -0.062 | 286 de 4669 | 0.447 +- 0.032 | 0.294 +- 0.056 |
| gradient boosting | completa | 0.451 | +0.000 | 0.293 | 0.320 | +0.000 | 1 de 4669 | 0.451 | 0.320 |
| gradient boosting | 25 % quitado | 0.466 | +0.015 | 0.310 | 0.343 | +0.023 | 1 de 4669 | 0.466 | 0.343 |
| gradient boosting | 50 % quitado | 0.451 | +0.000 | 0.279 | 0.329 | +0.009 | 8 de 4669 | 0.451 | 0.328 |
| gradient boosting | 75 % quitado | 0.422 | -0.029 | 0.243 | 0.288 | -0.032 | 286 de 4669 | 0.432 | 0.305 |
| regresion logistica | completa | 0.527 | +0.000 | 0.475 | 0.365 | +0.000 | 1 de 4669 | 0.527 | 0.364 |
| regresion logistica | 25 % quitado | 0.528 | +0.000 | 0.482 | 0.364 | -0.001 | 1 de 4669 | 0.527 | 0.364 |
| regresion logistica | 50 % quitado | 0.542 | +0.015 | 0.486 | 0.392 | +0.027 | 8 de 4669 | 0.543 | 0.392 |
| regresion logistica | 75 % quitado | 0.510 | -0.017 | 0.467 | 0.337 | -0.028 | 286 de 4669 | 0.520 | 0.358 |

## Alerta con alfa 0.1 (configuracion final)

Referencia de la IA:

| modelo | fraccion | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | EWT mediana (s) | caida | con alerta antes del cambio | sin alerta |
|---|---|---|---|---|---|---|---|---|
| CfC | completa | 0.000 +- 0.000 (de 32) | 0.233 +- 0.072 (de 30) | 0.700 +- 0.020 (de 40) | 0.7 +- 3.8 | +0.000 | 8.3 +- 0.9 de 21 | 4.0 +- 0.8 |
| CfC | 25 % quitado | 0.000 +- 0.000 (de 32) | 0.222 +- 0.057 (de 30) | 0.708 +- 0.012 (de 40) | -5.5 +- 1.2 | -6.227 | 7.7 +- 0.5 de 21 | 3.7 +- 0.5 |
| CfC | 50 % quitado | 0.000 +- 0.000 (de 32) | 0.189 +- 0.031 (de 30) | 0.692 +- 0.012 (de 40) | -13.1 +- 4.6 | -13.742 | 6.7 +- 0.5 de 21 | 4.3 +- 0.5 |
| CfC | 75 % quitado | 0.000 +- 0.000 (de 32) | 0.122 +- 0.016 (de 30) | 0.600 +- 0.020 (de 40) | -5.2 +- 1.8 | -5.863 | 5.7 +- 0.5 de 21 | 8.3 +- 0.5 |
| GRU | completa | 0.000 +- 0.000 (de 32) | 0.222 +- 0.031 (de 30) | 0.750 +- 0.020 (de 40) | -2.7 +- 10.6 | +0.000 | 8.3 +- 2.1 de 21 | 3.3 +- 0.9 |
| GRU | 25 % quitado | 0.000 +- 0.000 (de 32) | 0.144 +- 0.068 (de 30) | 0.667 +- 0.066 (de 40) | -14.5 +- 13.4 | -11.802 | 6.0 +- 1.6 de 21 | 5.7 +- 2.5 |
| GRU | 50 % quitado | 0.000 +- 0.000 (de 32) | 0.144 +- 0.113 (de 30) | 0.650 +- 0.114 (de 40) | -6.3 +- 6.4 | -3.568 | 6.3 +- 0.9 de 21 | 6.3 +- 3.7 |
| GRU | 75 % quitado | 0.000 +- 0.000 (de 32) | 0.144 +- 0.113 (de 30) | 0.542 +- 0.112 (de 40) | -3.2 +- 3.5 | -0.526 | 5.3 +- 0.5 de 21 | 10.3 +- 2.9 |
| LSTM | completa | 0.000 +- 0.000 (de 32) | 0.233 +- 0.072 (de 30) | 0.708 +- 0.047 (de 40) | 4.4 +- 4.7 | +0.000 | 9.0 +- 1.4 de 21 | 4.7 +- 0.5 |
| LSTM | 25 % quitado | 0.000 +- 0.000 (de 32) | 0.167 +- 0.047 (de 30) | 0.700 +- 0.035 (de 40) | 0.8 +- 2.5 | -3.582 | 8.7 +- 1.2 de 21 | 4.3 +- 0.9 |
| LSTM | 50 % quitado | 0.000 +- 0.000 (de 32) | 0.156 +- 0.083 (de 30) | 0.692 +- 0.062 (de 40) | -0.6 +- 1.7 | -4.970 | 7.7 +- 0.5 de 21 | 5.0 +- 1.6 |
| LSTM | 75 % quitado | 0.000 +- 0.000 (de 32) | 0.089 +- 0.083 (de 30) | 0.575 +- 0.020 (de 40) | -3.8 +- 3.1 | -8.164 | 6.0 +- 0.8 de 21 | 8.3 +- 0.5 |
| gradient boosting | completa | 0.031 (de 32) | 0.267 (de 30) | 0.775 (de 40) | 3.8 | +0.000 | 12.0 de 21 | 1.0 |
| gradient boosting | 25 % quitado | 0.000 (de 32) | 0.300 (de 30) | 0.775 (de 40) | 8.6 | +4.845 | 13.0 de 21 | 1.0 |
| gradient boosting | 50 % quitado | 0.000 (de 32) | 0.333 (de 30) | 0.825 (de 40) | 2.4 | -1.370 | 12.0 de 21 | 0.0 |
| gradient boosting | 75 % quitado | 0.031 (de 32) | 0.367 (de 30) | 0.875 (de 40) | -1.2 | -5.000 | 9.0 de 21 | 0.0 |
| regresion logistica | completa | 0.000 (de 32) | 0.100 (de 30) | 0.675 (de 40) | -10.5 | +0.000 | 5.0 de 21 | 5.0 |
| regresion logistica | 25 % quitado | 0.000 (de 32) | 0.100 (de 30) | 0.700 (de 40) | -13.2 | -2.635 | 5.0 de 21 | 4.0 |
| regresion logistica | 50 % quitado | 0.000 (de 32) | 0.133 (de 30) | 0.725 (de 40) | -12.4 | -1.850 | 5.0 de 21 | 3.0 |
| regresion logistica | 75 % quitado | 0.000 (de 32) | 0.133 (de 30) | 0.675 (de 40) | -17.2 | -6.635 | 6.0 de 21 | 4.0 |

Referencia humana:

| modelo | fraccion | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | EWT mediana (s) | caida | con alerta antes del cambio | sin alerta |
|---|---|---|---|---|---|---|---|---|
| CfC | completa | 0.000 +- 0.000 (de 32) | 0.184 +- 0.065 (de 29) | 0.733 +- 0.012 (de 40) | 23.6 +- 8.7 | +0.000 | 12.3 +- 0.5 de 16 | 2.0 +- 0.0 |
| CfC | 25 % quitado | 0.000 +- 0.000 (de 32) | 0.172 +- 0.049 (de 29) | 0.725 +- 0.020 (de 40) | 24.4 +- 5.0 | +0.810 | 11.0 +- 0.8 de 16 | 2.3 +- 0.5 |
| CfC | 50 % quitado | 0.000 +- 0.000 (de 32) | 0.172 +- 0.049 (de 29) | 0.683 +- 0.012 (de 40) | 15.2 +- 6.1 | -8.322 | 10.0 +- 0.8 de 16 | 2.7 +- 0.5 |
| CfC | 75 % quitado | 0.000 +- 0.000 (de 32) | 0.138 +- 0.000 (de 29) | 0.592 +- 0.012 (de 40) | 21.1 +- 1.6 | -2.507 | 8.3 +- 0.5 de 16 | 5.3 +- 0.5 |
| GRU | completa | 0.000 +- 0.000 (de 32) | 0.207 +- 0.028 (de 29) | 0.758 +- 0.047 (de 40) | 20.5 +- 11.7 | +0.000 | 11.3 +- 2.1 de 16 | 1.3 +- 0.5 |
| GRU | 25 % quitado | 0.000 +- 0.000 (de 32) | 0.126 +- 0.043 (de 29) | 0.667 +- 0.072 (de 40) | 11.6 +- 11.5 | -8.917 | 7.7 +- 2.5 de 16 | 3.7 +- 1.2 |
| GRU | 50 % quitado | 0.000 +- 0.000 (de 32) | 0.126 +- 0.086 (de 29) | 0.650 +- 0.122 (de 40) | 8.1 +- 7.1 | -12.430 | 7.3 +- 1.2 de 16 | 4.3 +- 2.1 |
| GRU | 75 % quitado | 0.000 +- 0.000 (de 32) | 0.149 +- 0.117 (de 29) | 0.542 +- 0.112 (de 40) | 15.3 +- 6.5 | -5.200 | 7.0 +- 1.6 de 16 | 6.7 +- 2.5 |
| LSTM | completa | 0.000 +- 0.000 (de 32) | 0.230 +- 0.043 (de 29) | 0.717 +- 0.031 (de 40) | 29.7 +- 4.0 | +0.000 | 10.7 +- 1.2 de 16 | 2.3 +- 0.5 |
| LSTM | 25 % quitado | 0.000 +- 0.000 (de 32) | 0.149 +- 0.059 (de 29) | 0.708 +- 0.031 (de 40) | 26.2 +- 0.6 | -3.455 | 11.0 +- 0.8 de 16 | 2.7 +- 1.2 |
| LSTM | 50 % quitado | 0.000 +- 0.000 (de 32) | 0.172 +- 0.102 (de 29) | 0.675 +- 0.061 (de 40) | 24.3 +- 2.1 | -5.350 | 10.0 +- 0.8 de 16 | 2.7 +- 1.7 |
| LSTM | 75 % quitado | 0.000 +- 0.000 (de 32) | 0.103 +- 0.102 (de 29) | 0.567 +- 0.012 (de 40) | 20.2 +- 2.6 | -9.442 | 9.0 +- 0.8 de 16 | 5.3 +- 0.5 |
| gradient boosting | completa | 0.031 (de 32) | 0.207 (de 29) | 0.800 (de 40) | 26.6 | +0.000 | 13.0 de 16 | 1.0 |
| gradient boosting | 25 % quitado | 0.000 (de 32) | 0.241 (de 29) | 0.800 (de 40) | 38.9 | +12.370 | 13.0 de 16 | 1.0 |
| gradient boosting | 50 % quitado | 0.000 (de 32) | 0.276 (de 29) | 0.850 (de 40) | 38.9 | +12.370 | 13.0 de 16 | 1.0 |
| gradient boosting | 75 % quitado | 0.031 (de 32) | 0.345 (de 29) | 0.875 (de 40) | 26.6 | +0.000 | 12.0 de 16 | 1.0 |
| regresion logistica | completa | 0.000 (de 32) | 0.069 (de 29) | 0.700 (de 40) | 13.2 | +0.000 | 10.0 de 16 | 2.0 |
| regresion logistica | 25 % quitado | 0.000 (de 32) | 0.069 (de 29) | 0.700 (de 40) | 15.2 | +2.000 | 10.0 de 16 | 2.0 |
| regresion logistica | 50 % quitado | 0.000 (de 32) | 0.103 (de 29) | 0.725 (de 40) | 22.1 | +8.980 | 12.0 de 16 | 2.0 |
| regresion logistica | 75 % quitado | 0.000 (de 32) | 0.103 (de 29) | 0.700 (de 40) | 17.5 | +4.315 | 12.0 de 16 | 2.0 |

## Alerta con alfa 0.05 (aparte)

Referencia de la IA:

| modelo | fraccion | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | EWT mediana (s) | caida | con alerta antes del cambio | sin alerta |
|---|---|---|---|---|---|---|---|---|
| CfC | completa | 0.000 +- 0.000 (de 32) | 0.089 +- 0.103 (de 30) | 0.433 +- 0.194 (de 40) | -9.6 +- 7.0 | +0.000 | 4.0 +- 2.9 de 21 | 12.0 +- 5.1 |
| CfC | 25 % quitado | 0.000 +- 0.000 (de 32) | 0.067 +- 0.094 (de 30) | 0.458 +- 0.225 (de 40) | -12.7 +- 5.6 | -3.157 | 3.7 +- 2.5 de 21 | 11.0 +- 6.2 |
| CfC | 50 % quitado | 0.000 +- 0.000 (de 32) | 0.056 +- 0.079 (de 30) | 0.400 +- 0.225 (de 40) | -1.9 +- 20.9 | +7.713 | 3.0 +- 1.6 de 21 | 13.0 +- 6.2 |
| CfC | 75 % quitado | 0.000 +- 0.000 (de 32) | 0.044 +- 0.063 (de 30) | 0.325 +- 0.215 (de 40) | -26.8 +- 22.9 | -17.229 | 2.7 +- 2.5 de 21 | 14.7 +- 5.3 |
| GRU | completa | 0.000 +- 0.000 (de 32) | 0.044 +- 0.016 (de 30) | 0.608 +- 0.031 (de 40) | -32.1 +- 12.1 | +0.000 | 4.0 +- 0.0 de 21 | 8.7 +- 1.2 |
| GRU | 25 % quitado | 0.000 +- 0.000 (de 32) | 0.011 +- 0.016 (de 30) | 0.450 +- 0.035 (de 40) | -30.6 +- 39.2 | +1.493 | 4.0 +- 0.0 de 21 | 13.0 +- 0.8 |
| GRU | 50 % quitado | 0.000 +- 0.000 (de 32) | 0.000 +- 0.000 (de 30) | 0.400 +- 0.041 (de 40) | -25.5 +- 7.2 | +6.563 | 1.7 +- 0.9 de 21 | 14.0 +- 0.8 |
| GRU | 75 % quitado | 0.000 +- 0.000 (de 32) | 0.000 +- 0.000 (de 30) | 0.217 +- 0.047 (de 40) | -31.8 +- 31.8 | +0.237 | 0.7 +- 0.9 de 21 | 18.3 +- 1.2 |
| LSTM | completa | 0.000 +- 0.000 (de 32) | 0.022 +- 0.031 (de 30) | 0.458 +- 0.024 (de 40) | -14.8 +- 4.3 | +0.000 | 3.0 +- 0.0 de 21 | 13.0 +- 0.8 |
| LSTM | 25 % quitado | 0.000 +- 0.000 (de 32) | 0.000 +- 0.000 (de 30) | 0.467 +- 0.012 (de 40) | -28.5 +- 15.3 | -13.628 | 3.0 +- 0.0 de 21 | 12.3 +- 0.5 |
| LSTM | 50 % quitado | 0.000 +- 0.000 (de 32) | 0.000 +- 0.000 (de 30) | 0.400 +- 0.074 (de 40) | -30.3 +- 8.4 | -15.442 | 2.0 +- 0.0 de 21 | 14.3 +- 2.1 |
| LSTM | 75 % quitado | 0.000 +- 0.000 (de 32) | 0.000 +- 0.000 (de 30) | 0.250 +- 0.020 (de 40) | -50.7 +- 9.2 | -35.810 | 0.3 +- 0.5 de 21 | 18.0 +- 0.8 |
| gradient boosting | completa | 0.000 (de 32) | 0.033 (de 30) | 0.550 (de 40) | -5.7 | +0.000 | 4.0 de 21 | 10.0 |
| gradient boosting | 25 % quitado | 0.000 (de 32) | 0.033 (de 30) | 0.575 (de 40) | -0.9 | +4.820 | 6.0 de 21 | 9.0 |
| gradient boosting | 50 % quitado | 0.000 (de 32) | 0.067 (de 30) | 0.550 (de 40) | -3.9 | +1.800 | 5.0 de 21 | 10.0 |
| gradient boosting | 75 % quitado | 0.000 (de 32) | 0.100 (de 30) | 0.575 (de 40) | -24.2 | -18.530 | 5.0 de 21 | 8.0 |
| regresion logistica | completa | 0.000 (de 32) | 0.000 (de 30) | 0.400 (de 40) | -58.6 | +0.000 | 2.0 de 21 | 12.0 |
| regresion logistica | 25 % quitado | 0.000 (de 32) | 0.000 (de 30) | 0.475 (de 40) | -63.3 | -4.700 | 2.0 de 21 | 9.0 |
| regresion logistica | 50 % quitado | 0.000 (de 32) | 0.033 (de 30) | 0.475 (de 40) | -39.5 | +19.065 | 2.0 de 21 | 9.0 |
| regresion logistica | 75 % quitado | 0.000 (de 32) | 0.000 (de 30) | 0.500 (de 40) | -56.6 | +1.950 | 1.0 de 21 | 9.0 |

Referencia humana:

| modelo | fraccion | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | EWT mediana (s) | caida | con alerta antes del cambio | sin alerta |
|---|---|---|---|---|---|---|---|---|
| CfC | completa | 0.000 +- 0.000 (de 32) | 0.069 +- 0.074 (de 29) | 0.450 +- 0.215 (de 40) | 20.3 +- 3.7 | +0.000 | 7.0 +- 4.3 de 16 | 8.0 +- 4.3 |
| CfC | 25 % quitado | 0.000 +- 0.000 (de 32) | 0.057 +- 0.081 (de 29) | 0.458 +- 0.225 (de 40) | 18.5 +- 5.2 | -1.789 | 5.7 +- 3.9 de 16 | 7.7 +- 4.5 |
| CfC | 50 % quitado | 0.000 +- 0.000 (de 32) | 0.057 +- 0.081 (de 29) | 0.392 +- 0.214 (de 40) | 16.5 +- 7.1 | -3.809 | 5.0 +- 3.7 de 16 | 9.3 +- 4.9 |
| CfC | 75 % quitado | 0.000 +- 0.000 (de 32) | 0.046 +- 0.065 (de 29) | 0.325 +- 0.215 (de 40) | 3.3 +- 16.6 | -16.972 | 3.3 +- 4.0 de 16 | 10.7 +- 4.5 |
| GRU | completa | 0.000 +- 0.000 (de 32) | 0.034 +- 0.000 (de 29) | 0.617 +- 0.042 (de 40) | -3.4 +- 8.7 | +0.000 | 6.0 +- 0.8 de 16 | 4.7 +- 1.2 |
| GRU | 25 % quitado | 0.000 +- 0.000 (de 32) | 0.011 +- 0.016 (de 29) | 0.450 +- 0.035 (de 40) | 2.1 +- 3.4 | +5.508 | 3.7 +- 0.5 de 16 | 9.0 +- 0.8 |
| GRU | 50 % quitado | 0.000 +- 0.000 (de 32) | 0.000 +- 0.000 (de 29) | 0.400 +- 0.041 (de 40) | 10.4 +- 14.2 | +13.768 | 3.3 +- 0.5 de 16 | 9.7 +- 0.9 |
| GRU | 75 % quitado | 0.000 +- 0.000 (de 32) | 0.000 +- 0.000 (de 29) | 0.217 +- 0.047 (de 40) | 15.4 +- 34.5 | +18.797 | 1.3 +- 0.5 de 16 | 13.7 +- 1.2 |
| LSTM | completa | 0.000 +- 0.000 (de 32) | 0.023 +- 0.033 (de 29) | 0.458 +- 0.024 (de 40) | 12.6 +- 2.6 | +0.000 | 5.3 +- 0.5 de 16 | 9.0 +- 0.8 |
| LSTM | 25 % quitado | 0.000 +- 0.000 (de 32) | 0.000 +- 0.000 (de 29) | 0.467 +- 0.012 (de 40) | 5.7 +- 3.6 | -6.981 | 4.3 +- 0.5 de 16 | 8.3 +- 0.5 |
| LSTM | 50 % quitado | 0.000 +- 0.000 (de 32) | 0.000 +- 0.000 (de 29) | 0.400 +- 0.074 (de 40) | 9.6 +- 11.1 | -3.088 | 3.3 +- 0.9 de 16 | 10.3 +- 2.1 |
| LSTM | 75 % quitado | 0.000 +- 0.000 (de 32) | 0.000 +- 0.000 (de 29) | 0.250 +- 0.020 (de 40) | -21.5 +- 11.8 | -34.104 | 1.0 +- 0.8 de 16 | 13.7 +- 1.2 |
| gradient boosting | completa | 0.000 (de 32) | 0.034 (de 29) | 0.550 (de 40) | 20.2 | +0.000 | 6.0 de 16 | 6.0 |
| gradient boosting | 25 % quitado | 0.000 (de 32) | 0.034 (de 29) | 0.575 (de 40) | 19.9 | -0.315 | 7.0 de 16 | 5.0 |
| gradient boosting | 50 % quitado | 0.000 (de 32) | 0.069 (de 29) | 0.550 (de 40) | 21.7 | +1.500 | 6.0 de 16 | 6.0 |
| gradient boosting | 75 % quitado | 0.000 (de 32) | 0.103 (de 29) | 0.575 (de 40) | 19.9 | -0.315 | 7.0 de 16 | 5.0 |
| regresion logistica | completa | 0.000 (de 32) | 0.000 (de 29) | 0.400 (de 40) | -8.3 | +0.000 | 2.0 de 16 | 8.0 |
| regresion logistica | 25 % quitado | 0.000 (de 32) | 0.000 (de 29) | 0.475 (de 40) | -12.6 | -4.265 | 2.0 de 16 | 6.0 |
| regresion logistica | 50 % quitado | 0.000 (de 32) | 0.034 (de 29) | 0.475 (de 40) | -6.0 | +2.350 | 4.0 de 16 | 5.0 |
| regresion logistica | 75 % quitado | 0.000 (de 32) | 0.000 (de 29) | 0.500 (de 40) | -9.4 | -1.120 | 2.0 de 16 | 5.0 |
