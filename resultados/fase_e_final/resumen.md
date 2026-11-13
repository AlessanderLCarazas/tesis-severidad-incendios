# Fase E final en validacion: histeresis, alerta conformal y segmentos

Generado por `python fase_e_final.py --pesos runs_fuego/modelos_final/final`. La prueba no se usa. Media +- desviacion entre semillas.

- Nivel: umbrales de CORAL en 0.5 e histeresis causal de 2 s.
- Alerta conformal por video: umbral = cuantil conformal del riesgo maximo de los 52 videos de calibracion sin fuego o controlados; se apaga cuando el riesgo baja de 0.8 veces el umbral durante 2 s.
- Referencias sin tiempo: una prediccion por segundo; riesgo = 0.5 P(incipiente) + P(incontrolado).
- Validacion: 47 videos de fuego para el nivel, 28 sin fuego, 25 controlados, 22 con propagacion. Videos con cambio a incontrolado: 6 en validacion y 5 en calibracion (estos no entran en el umbral conformal).
- Adelanto = instante del cambio a incontrolado menos instante de la primera alerta; negativo = la alerta llega despues.

| modelo | corridas | etapa | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | cambios de nivel por video |
|---|---|---|---|---|---|---|---|---|---|
| CFC | 3 | antes (umbral 0.5) | 0.497 +- 0.024 | 0.467 +- 0.052 | 0.299 +- 0.009 | 0.726 +- 0.012 | 0.526 +- 0.027 | 0.436 +- 0.026 | 5.11 +- 1.19 |
| CFC | 3 | umbral 0.5 + histeresis (final) | 0.485 +- 0.034 | 0.432 +- 0.069 | 0.296 +- 0.018 | 0.725 +- 0.016 | 0.513 +- 0.035 | 0.426 +- 0.037 | 2.27 +- 0.38 |
| GRU | 3 | antes (umbral 0.5) | 0.660 +- 0.008 | 0.760 +- 0.005 | 0.416 +- 0.017 | 0.805 +- 0.027 | 0.714 +- 0.008 | 0.641 +- 0.022 | 2.52 +- 0.35 |
| GRU | 3 | umbral 0.5 + histeresis (final) | 0.641 +- 0.009 | 0.738 +- 0.003 | 0.397 +- 0.023 | 0.789 +- 0.025 | 0.690 +- 0.008 | 0.615 +- 0.022 | 1.89 +- 0.16 |
| LSTM | 3 | antes (umbral 0.5) | 0.552 +- 0.011 | 0.551 +- 0.057 | 0.313 +- 0.018 | 0.791 +- 0.057 | 0.607 +- 0.014 | 0.498 +- 0.040 | 3.87 +- 0.10 |
| LSTM | 3 | umbral 0.5 + histeresis (final) | 0.536 +- 0.014 | 0.528 +- 0.074 | 0.302 +- 0.024 | 0.777 +- 0.057 | 0.587 +- 0.011 | 0.478 +- 0.030 | 2.23 +- 0.13 |
| gradient boosting | 1 | antes (umbral 0.5) | 0.622 | 0.623 | 0.444 | 0.799 | 0.679 | 0.519 | 12.81 |
| gradient boosting | 1 | umbral 0.5 + histeresis (final) | 0.653 | 0.664 | 0.449 | 0.847 | 0.720 | 0.598 | 2.30 |
| regresion logistica | 1 | antes (umbral 0.5) | 0.482 | 0.468 | 0.285 | 0.691 | 0.514 | 0.357 | 12.28 |
| regresion logistica | 1 | umbral 0.5 + histeresis (final) | 0.479 | 0.488 | 0.268 | 0.681 | 0.509 | 0.371 | 3.13 |

| modelo | alfa | umbral de riesgo | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | adelanto mediano en validacion (s) | antes del cambio en validacion | adelanto mediano en validacion + calibracion (s) | antes del cambio en validacion + calibracion | sin alerta (validacion + calibracion) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFC | 0.1 (final) | 0.777 +- 0.042 | 0.000 +- 0.000 | 0.133 +- 0.019 | 0.818 +- 0.037 | -6.7 +- 4.1 | 2.0 +- 0.0 de 6 | -3.2 +- 5.2 | 4.3 +- 0.5 de 11 | 0.7 +- 0.5 |
| CFC | 0.05 | 0.973 +- 0.015 | 0.000 +- 0.000 | 0.080 +- 0.000 | 0.621 +- 0.021 | -14.6 +- 3.3 | 0.3 +- 0.5 de 6 | -13.4 +- 2.2 | 1.3 +- 0.5 de 11 | 4.0 +- 1.6 |
| GRU | 0.1 (final) | 0.768 +- 0.113 | 0.000 +- 0.000 | 0.147 +- 0.050 | 0.833 +- 0.057 | -2.7 +- 2.1 | 1.7 +- 0.5 de 6 | -2.2 +- 2.5 | 4.0 +- 0.8 de 11 | 1.7 +- 0.9 |
| GRU | 0.05 | 0.998 +- 0.001 | 0.000 +- 0.000 | 0.027 +- 0.019 | 0.606 +- 0.077 | -32.0 +- 11.4 | 0.0 +- 0.0 de 6 | -32.0 +- 11.4 | 1.0 +- 0.0 de 11 | 6.0 +- 0.8 |
| LSTM | 0.1 (final) | 0.907 +- 0.054 | 0.000 +- 0.000 | 0.093 +- 0.019 | 0.758 +- 0.057 | -11.7 +- 6.0 | 0.3 +- 0.5 de 6 | -9.0 +- 6.2 | 2.3 +- 0.9 de 11 | 1.0 +- 0.8 |
| LSTM | 0.05 | 0.993 +- 0.003 | 0.000 +- 0.000 | 0.053 +- 0.019 | 0.606 +- 0.043 | -21.3 +- 6.4 | 0.0 +- 0.0 de 6 | -18.8 +- 4.2 | 1.7 +- 0.5 de 11 | 3.3 +- 0.9 |
| gradient boosting | 0.1 (final) | 0.869 | 0.071 | 0.360 | 0.864 | 0.4 | 3.0 de 6 | 1.3 | 6.0 de 11 | 0.0 |
| gradient boosting | 0.05 | 0.989 | 0.000 | 0.000 | 0.727 | -18.4 | 1.0 de 6 | -18.4 | 2.0 de 11 | 4.0 |
| regresion logistica | 0.1 (final) | 0.832 | 0.000 | 0.080 | 0.682 | -9.7 | 1.0 de 6 | -10.4 | 2.0 de 11 | 3.0 |
| regresion logistica | 0.05 | 0.971 | 0.000 | 0.000 | 0.409 | 1.6 | 1.0 de 6 | -98.1 | 1.0 de 11 | 8.0 |
