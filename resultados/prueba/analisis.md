# Analisis descriptivo de la prueba

Generado por `python analisis_prueba.py`. No cambia nada: repite la inferencia con la configuracion congelada (diferencia maxima de F1 con `metricas.json`: 0.0e+00) y desglosa los mismos resultados.

## a. Metricas por subconjunto, referencia de la IA

| subconjunto | videos | segundos (controlado / incipiente / incontrolado) | modelo | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | mAP tIoU 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| 40 al azar (`prueba_humana`) | 40 | 1112 / 219 / 315 | GRU | 0.541 +- 0.013 | 0.703 +- 0.039 | 0.153 +- 0.066 | 0.766 +- 0.044 | 0.629 +- 0.028 | 0.455 +- 0.046 | 0.502 +- 0.021 |
| 40 al azar (`prueba_humana`) | 40 | 1112 / 219 / 315 | CfC | 0.538 +- 0.028 | 0.678 +- 0.066 | 0.184 +- 0.049 | 0.753 +- 0.046 | 0.612 +- 0.057 | 0.418 +- 0.071 | 0.444 +- 0.050 |
| 40 al azar (`prueba_humana`) | 40 | 1112 / 219 / 315 | gradient boosting | 0.489 | 0.663 | 0.184 | 0.618 | 0.591 | 0.329 | 0.407 |
| 30 con cambios (`prueba_cambios`) | 30 | 352 / 1277 / 1428 | GRU | 0.435 +- 0.005 | 0.306 +- 0.010 | 0.437 +- 0.025 | 0.563 +- 0.025 | 0.467 +- 0.010 | 0.221 +- 0.018 | 0.159 +- 0.010 |
| 30 con cambios (`prueba_cambios`) | 30 | 352 / 1277 / 1428 | CfC | 0.387 +- 0.013 | 0.308 +- 0.016 | 0.271 +- 0.060 | 0.582 +- 0.020 | 0.406 +- 0.009 | 0.241 +- 0.010 | 0.110 +- 0.029 |
| 30 con cambios (`prueba_cambios`) | 30 | 352 / 1277 / 1428 | gradient boosting | 0.381 | 0.323 | 0.218 | 0.600 | 0.415 | 0.233 | 0.112 |
| los 70 | 70 | 1464 / 1496 / 1743 | GRU | 0.519 +- 0.009 | 0.561 +- 0.022 | 0.386 +- 0.032 | 0.609 +- 0.025 | 0.524 +- 0.011 | 0.371 +- 0.029 | 0.299 +- 0.023 |
| los 70 | 70 | 1464 / 1496 / 1743 | CfC | 0.460 +- 0.002 | 0.507 +- 0.039 | 0.252 +- 0.053 | 0.621 +- 0.023 | 0.478 +- 0.014 | 0.319 +- 0.036 | 0.216 +- 0.039 |
| los 70 | 70 | 1464 / 1496 / 1743 | gradient boosting | 0.442 | 0.510 | 0.212 | 0.605 | 0.476 | 0.292 | 0.204 |

Tipos de video segun la IA: azar: controlado 27, controlado_a_incipiente 1, incipiente 4, incontrolado 8; cambios: controlado 3, controlado_a_incipiente 4, controlado_a_incontrolado 6, incipiente 1, incipiente_a_incontrolado 15, incontrolado 1.

## b. Proporcion de niveles por segundo

| conjunto | referencia | videos de fuego | segundos | controlado | incipiente | incontrolado |
|---|---|---|---|---|---|---|
| entrenamiento | IA | 215 | 17421 | 23.7 % (4133) | 13.0 % (2260) | 63.3 % (11028) |
| validacion | IA | 47 | 2757 | 37.5 % (1033) | 11.0 % (303) | 51.5 % (1421) |
| calibracion | IA | 45 | 4515 | 20.7 % (935) | 23.4 % (1055) | 55.9 % (2525) |
| prueba | IA | 70 | 4703 | 31.1 % (1464) | 31.8 % (1496) | 37.1 % (1743) |
| prueba, solo `prueba_humana` | IA | 40 | 1646 | 67.6 % (1112) | 13.3 % (219) | 19.1 % (315) |
| prueba, solo `prueba_cambios` | IA | 30 | 3057 | 11.5 % (352) | 41.8 % (1277) | 46.7 % (1428) |
| prueba | humana | 69 | 4669 | 33.3 % (1556) | 49.1 % (2291) | 17.6 % (822) |
| prueba, solo `prueba_humana` | humana | 40 | 1674 | 61.9 % (1036) | 23.5 % (394) | 14.6 % (244) |
| prueba, solo `prueba_cambios` | humana | 29 | 2995 | 17.4 % (520) | 63.3 % (1897) | 19.3 % (578) |

## c. Intervalo binomial del 95 % de las falsas alarmas (alfa 0.1)

Intervalo exacto de Clopper-Pearson, por corrida (cada semilla es un modelo distinto sobre los mismos videos).

| corrida | sin fuego: alarmas | tasa | IC 95 % | controlados (IA): alarmas | tasa | IC 95 % | controlados (humana): alarmas | tasa | IC 95 % |
|---|---|---|---|---|---|---|---|---|---|
| cfc_s42 | 0 de 32 | 0.0 % | 0.0 a 10.9 % | 5 de 30 | 16.7 % | 5.6 a 34.7 % | 4 de 29 | 13.8 % | 3.9 a 31.7 % |
| cfc_s43 | 0 de 32 | 0.0 % | 0.0 a 10.9 % | 6 de 30 | 20.0 % | 7.7 a 38.6 % | 4 de 29 | 13.8 % | 3.9 a 31.7 % |
| cfc_s44 | 0 de 32 | 0.0 % | 0.0 a 10.9 % | 10 de 30 | 33.3 % | 17.3 a 52.8 % | 8 de 29 | 27.6 % | 12.7 a 47.2 % |
| gru_s42 | 0 de 32 | 0.0 % | 0.0 a 10.9 % | 8 de 30 | 26.7 % | 12.3 a 45.9 % | 6 de 29 | 20.7 % | 8.0 a 39.7 % |
| gru_s43 | 0 de 32 | 0.0 % | 0.0 a 10.9 % | 6 de 30 | 20.0 % | 7.7 a 38.6 % | 5 de 29 | 17.2 % | 5.8 a 35.8 % |
| gru_s44 | 0 de 32 | 0.0 % | 0.0 a 10.9 % | 6 de 30 | 20.0 % | 7.7 a 38.6 % | 7 de 29 | 24.1 % | 10.3 a 43.5 % |
| lstm_s42 | 0 de 32 | 0.0 % | 0.0 a 10.9 % | 5 de 30 | 16.7 % | 5.6 a 34.7 % | 5 de 29 | 17.2 % | 5.8 a 35.8 % |
| lstm_s43 | 0 de 32 | 0.0 % | 0.0 a 10.9 % | 10 de 30 | 33.3 % | 17.3 a 52.8 % | 8 de 29 | 27.6 % | 12.7 a 47.2 % |
| lstm_s44 | 0 de 32 | 0.0 % | 0.0 a 10.9 % | 6 de 30 | 20.0 % | 7.7 a 38.6 % | 7 de 29 | 24.1 % | 10.3 a 43.5 % |
| gradient boosting | 1 de 32 | 3.1 % | 0.1 a 16.2 % | 8 de 30 | 26.7 % | 12.3 a 45.9 % | 6 de 29 | 20.7 % | 8.0 a 39.7 % |
| regresion logistica | 0 de 32 | 0.0 % | 0.0 a 10.9 % | 3 de 30 | 10.0 % | 2.1 a 26.5 % | 2 de 29 | 6.9 % | 0.8 a 22.8 % |

## d. Intervalo por bootstrap de videos del mAP promedio con tIoU 0.5

1000 remuestreos de videos, semilla 42; en los modelos con 3 semillas se promedia el mAP de las semillas dentro de cada remuestreo; la diferencia usa los mismos remuestreos.

| referencia | videos | modelo | mAP tIoU 0.5 | IC 95 % |
|---|---|---|---|---|
| IA | 70 | GRU | 0.299 +- 0.023 | 0.219 a 0.400 |
| IA | 70 | CfC | 0.216 +- 0.039 | 0.151 a 0.303 |
| IA | 70 | gradient boosting | 0.204 | 0.140 a 0.305 |
| humana | 29 | GRU | 0.136 +- 0.015 | 0.071 a 0.290 |
| humana | 29 | CfC | 0.103 +- 0.048 | 0.066 a 0.199 |
| humana | 29 | gradient boosting | 0.125 | 0.066 a 0.257 |

| referencia | diferencia | mAP tIoU 0.5 | IC 95 % | remuestreos con diferencia positiva |
|---|---|---|---|---|
| IA | GRU - CfC | +0.083 | +0.002 a +0.162 | 97.9 % |
| humana | GRU - CfC | +0.033 | -0.076 a +0.186 | 68.9 % |
