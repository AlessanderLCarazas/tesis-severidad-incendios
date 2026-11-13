# Camara movil contra camara fija en la prueba (parte 1.3 de la robustez)

Generado por `python analisis_prueba.py --partes camara` con las predicciones de la fase F (configuracion congelada).

- Movimiento por fotograma = norma de la traslacion (fraccion del fotograma) + 0.5 (|s - 1| + |rotacion|), de la similitud estimada por `camara.py`; puntaje del video = media por fotograma multiplicada por 10 (fraccion del fotograma por segundo).
- Camara movil: puntaje >= 0.01; camara fija: menor. El umbral se fijo mirando solo la distribucion del movimiento, antes de ver los resultados por grupo.
- Videos de prueba: 59 con camara movil y 43 con camara fija (de fuego: 39 y 31; sin fuego: 20 y 12).

## Nivel por segundo, referencia de la IA

| subconjunto | videos | segundos (controlado / incipiente / incontrolado) | modelo | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal |
|---|---|---|---|---|---|---|---|---|---|
| camara fija | 31 | 888 / 718 / 775 | CfC | 0.470 +- 0.023 | 0.559 +- 0.029 | 0.249 +- 0.117 | 0.602 +- 0.023 | 0.494 +- 0.001 | 0.317 +- 0.026 |
| camara fija | 31 | 888 / 718 / 775 | GRU | 0.556 +- 0.026 | 0.597 +- 0.017 | 0.428 +- 0.075 | 0.643 +- 0.012 | 0.559 +- 0.024 | 0.404 +- 0.016 |
| camara fija | 31 | 888 / 718 / 775 | LSTM | 0.444 +- 0.050 | 0.539 +- 0.004 | 0.194 +- 0.053 | 0.600 +- 0.096 | 0.483 +- 0.037 | 0.274 +- 0.065 |
| camara fija | 31 | 888 / 718 / 775 | gradient boosting | 0.402 | 0.494 | 0.212 | 0.499 | 0.422 | 0.190 |
| camara fija | 31 | 888 / 718 / 775 | regresion logistica | 0.435 | 0.410 | 0.292 | 0.601 | 0.421 | 0.233 |
| camara movil | 39 | 576 / 778 / 968 | CfC | 0.438 +- 0.027 | 0.428 +- 0.059 | 0.250 +- 0.019 | 0.635 +- 0.026 | 0.461 +- 0.029 | 0.294 +- 0.054 |
| camara movil | 39 | 576 / 778 / 968 | GRU | 0.476 +- 0.049 | 0.496 +- 0.087 | 0.345 +- 0.039 | 0.587 +- 0.034 | 0.488 +- 0.045 | 0.298 +- 0.080 |
| camara movil | 39 | 576 / 778 / 968 | LSTM | 0.472 +- 0.010 | 0.532 +- 0.015 | 0.266 +- 0.025 | 0.618 +- 0.004 | 0.498 +- 0.006 | 0.342 +- 0.021 |
| camara movil | 39 | 576 / 778 / 968 | gradient boosting | 0.472 | 0.536 | 0.212 | 0.668 | 0.532 | 0.355 |
| camara movil | 39 | 576 / 778 / 968 | regresion logistica | 0.556 | 0.557 | 0.497 | 0.616 | 0.561 | 0.376 |

## Nivel por segundo, referencia humana

| subconjunto | videos | segundos (controlado / incipiente / incontrolado) | modelo | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal |
|---|---|---|---|---|---|---|---|---|---|
| camara fija | 30 | 1028 / 953 / 348 | CfC | 0.534 +- 0.028 | 0.685 +- 0.048 | 0.353 +- 0.048 | 0.566 +- 0.068 | 0.561 +- 0.039 | 0.416 +- 0.068 |
| camara fija | 30 | 1028 / 953 / 348 | GRU | 0.548 +- 0.023 | 0.596 +- 0.019 | 0.383 +- 0.063 | 0.667 +- 0.013 | 0.534 +- 0.027 | 0.394 +- 0.037 |
| camara fija | 30 | 1028 / 953 / 348 | LSTM | 0.473 +- 0.029 | 0.677 +- 0.005 | 0.255 +- 0.014 | 0.487 +- 0.101 | 0.529 +- 0.006 | 0.348 +- 0.054 |
| camara fija | 30 | 1028 / 953 / 348 | gradient boosting | 0.499 | 0.625 | 0.371 | 0.501 | 0.522 | 0.316 |
| camara fija | 30 | 1028 / 953 / 348 | regresion logistica | 0.551 | 0.567 | 0.463 | 0.624 | 0.536 | 0.370 |
| camara movil | 39 | 528 / 1338 / 474 | CfC | 0.403 +- 0.022 | 0.425 +- 0.059 | 0.308 +- 0.037 | 0.476 +- 0.018 | 0.394 +- 0.022 | 0.244 +- 0.046 |
| camara movil | 39 | 528 / 1338 / 474 | GRU | 0.458 +- 0.038 | 0.517 +- 0.066 | 0.447 +- 0.005 | 0.410 +- 0.049 | 0.451 +- 0.032 | 0.262 +- 0.074 |
| camara movil | 39 | 528 / 1338 / 474 | LSTM | 0.429 +- 0.014 | 0.494 +- 0.025 | 0.301 +- 0.019 | 0.493 +- 0.017 | 0.421 +- 0.013 | 0.294 +- 0.026 |
| camara movil | 39 | 528 / 1338 / 474 | gradient boosting | 0.404 | 0.549 | 0.220 | 0.444 | 0.400 | 0.283 |
| camara movil | 39 | 528 / 1338 / 474 | regresion logistica | 0.512 | 0.568 | 0.485 | 0.482 | 0.509 | 0.353 |

## Alerta con alfa 0.1

| camara | modelo | falsas alarmas sin fuego | falsas alarmas en controlados (IA) | propagacion con alerta (IA) | EWT mediana, IA (s) | antes del cambio, IA | EWT mediana, humana (s) | antes del cambio, humana |
|---|---|---|---|---|---|---|---|---|
| fija | CfC | 0.000 +- 0.000 (de 12) | 0.167 +- 0.120 (de 18) | 0.564 +- 0.036 (de 13) | -42.5 +- 3.8 | 0.7 +- 0.5 de 7 | 18.1 +- 8.9 | 3.7 +- 0.5 de 6 |
| fija | GRU | 0.000 +- 0.000 (de 12) | 0.148 +- 0.026 (de 18) | 0.718 +- 0.036 (de 13) | -23.0 +- 11.1 | 0.7 +- 0.9 de 7 | 3.7 +- 3.3 | 3.7 +- 0.9 de 6 |
| fija | LSTM | 0.000 +- 0.000 (de 12) | 0.167 +- 0.045 (de 18) | 0.590 +- 0.073 (de 13) | -35.4 +- 16.4 | 1.0 +- 0.8 de 7 | 29.8 +- 7.6 | 3.7 +- 0.5 de 6 |
| fija | gradient boosting | 0.000 (de 12) | 0.167 (de 18) | 0.692 (de 13) | -18.8 | 2.0 de 7 | 13.1 | 4.0 de 6 |
| fija | regresion logistica | 0.000 (de 12) | 0.056 (de 18) | 0.538 (de 13) | -28.6 | 0.0 de 7 | 19.9 | 3.0 de 6 |
| movil | CfC | 0.000 +- 0.000 (de 20) | 0.333 +- 0.068 (de 12) | 0.765 +- 0.017 (de 27) | 10.3 +- 0.6 | 7.7 +- 0.5 de 14 | 36.5 +- 13.4 | 8.7 +- 0.5 de 10 |
| movil | GRU | 0.000 +- 0.000 (de 20) | 0.333 +- 0.068 (de 12) | 0.765 +- 0.046 (de 27) | 9.2 +- 4.8 | 7.7 +- 1.2 de 14 | 38.6 +- 13.2 | 7.7 +- 1.2 de 10 |
| movil | LSTM | 0.000 +- 0.000 (de 20) | 0.333 +- 0.118 (de 12) | 0.765 +- 0.035 (de 27) | 9.7 +- 1.8 | 8.0 +- 0.8 de 14 | 29.5 +- 0.4 | 7.0 +- 0.8 de 10 |
| movil | gradient boosting | 0.050 (de 20) | 0.417 (de 12) | 0.815 (de 27) | 11.2 | 10.0 de 14 | 60.0 | 9.0 de 10 |
| movil | regresion logistica | 0.000 (de 20) | 0.167 (de 12) | 0.741 (de 27) | -0.4 | 5.0 de 14 | 6.4 | 7.0 de 10 |
