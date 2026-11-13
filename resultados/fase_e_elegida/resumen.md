# Fase E en validacion con la configuracion elegida por modelo

Generado por `python fase_e_elegida.py`. La prueba no se usa. Media +- desviacion entre semillas.

- CfC: configuracion A: peso simple del incipiente y tasa 1e-3.
- GRU: configuracion B: doble peso del incipiente y tasa 3e-3.
- LSTM: configuracion A: peso simple del incipiente y tasa 1e-3.
- Nivel: umbrales de CORAL en 0.5 e histeresis causal de 2 s.
- Alerta conformal por video: umbral = cuantil conformal del riesgo maximo de los 52 videos de calibracion sin fuego o controlados; se apaga cuando el riesgo baja de 0.8 veces el umbral durante 2 s.
- Referencias sin tiempo: una prediccion por segundo; riesgo = 0.5 P(incipiente) + P(incontrolado).
- Validacion: 47 videos de fuego para el nivel, 28 sin fuego, 25 controlados, 22 con propagacion. Videos con cambio a incontrolado: 6 en validacion y 5 en calibracion (estos no entran en el umbral conformal).
- Adelanto = instante del cambio a incontrolado menos instante de la primera alerta; negativo = la alerta llega despues.

| modelo | corridas | etapa | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | cambios de nivel por video |
|---|---|---|---|---|---|---|---|---|---|
| CfC | 3 | antes (umbral 0.5) | 0.544 +- 0.015 | 0.711 +- 0.051 | 0.170 +- 0.034 | 0.751 +- 0.009 | 0.644 +- 0.014 | 0.543 +- 0.040 | 5.26 +- 0.67 |
| CfC | 3 | umbral 0.5 + histeresis (final) | 0.536 +- 0.019 | 0.706 +- 0.050 | 0.169 +- 0.030 | 0.734 +- 0.006 | 0.629 +- 0.018 | 0.529 +- 0.041 | 2.16 +- 0.13 |
| GRU | 3 | antes (umbral 0.5) | 0.660 +- 0.008 | 0.760 +- 0.005 | 0.416 +- 0.017 | 0.805 +- 0.027 | 0.714 +- 0.008 | 0.641 +- 0.022 | 2.52 +- 0.35 |
| GRU | 3 | umbral 0.5 + histeresis (final) | 0.641 +- 0.009 | 0.738 +- 0.003 | 0.397 +- 0.023 | 0.789 +- 0.025 | 0.690 +- 0.008 | 0.615 +- 0.022 | 1.89 +- 0.16 |
| LSTM | 3 | antes (umbral 0.5) | 0.552 +- 0.048 | 0.738 +- 0.035 | 0.137 +- 0.061 | 0.782 +- 0.063 | 0.685 +- 0.042 | 0.583 +- 0.074 | 4.76 +- 0.08 |
| LSTM | 3 | umbral 0.5 + histeresis (final) | 0.548 +- 0.048 | 0.726 +- 0.036 | 0.156 +- 0.054 | 0.763 +- 0.067 | 0.663 +- 0.044 | 0.563 +- 0.078 | 2.71 +- 0.06 |
| gradient boosting | 1 | antes (umbral 0.5) | 0.622 | 0.623 | 0.444 | 0.799 | 0.679 | 0.519 | 12.81 |
| gradient boosting | 1 | umbral 0.5 + histeresis (final) | 0.653 | 0.664 | 0.449 | 0.847 | 0.720 | 0.598 | 2.30 |
| regresion logistica | 1 | antes (umbral 0.5) | 0.482 | 0.468 | 0.285 | 0.691 | 0.514 | 0.357 | 12.28 |
| regresion logistica | 1 | umbral 0.5 + histeresis (final) | 0.479 | 0.488 | 0.268 | 0.681 | 0.509 | 0.371 | 3.13 |

| modelo | alfa | umbral de riesgo | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | adelanto mediano en validacion (s) | antes del cambio en validacion | adelanto mediano en validacion + calibracion (s) | antes del cambio en validacion + calibracion | sin alerta (validacion + calibracion) |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.1 (final) | 0.723 +- 0.064 | 0.000 +- 0.000 | 0.187 +- 0.038 | 0.803 +- 0.077 | -6.8 +- 1.8 | 1.3 +- 0.5 de 6 | -4.3 +- 3.6 | 3.3 +- 0.9 de 11 | 0.3 +- 0.5 |
| GRU | 0.1 (final) | 0.768 +- 0.113 | 0.000 +- 0.000 | 0.147 +- 0.050 | 0.833 +- 0.057 | -2.7 +- 2.1 | 1.7 +- 0.5 de 6 | -2.2 +- 2.5 | 4.0 +- 0.8 de 11 | 1.7 +- 0.9 |
| LSTM | 0.1 (final) | 0.599 +- 0.025 | 0.000 +- 0.000 | 0.093 +- 0.019 | 0.833 +- 0.043 | -8.7 +- 8.4 | 1.7 +- 0.5 de 6 | -1.7 +- 0.9 | 4.7 +- 0.5 de 11 | 0.0 +- 0.0 |
| gradient boosting | 0.1 (final) | 0.869 | 0.071 | 0.360 | 0.864 | 0.4 | 3.0 de 6 | 1.3 | 6.0 de 11 | 0.0 |
| regresion logistica | 0.1 (final) | 0.832 | 0.000 | 0.080 | 0.682 | -9.7 | 1.0 de 6 | -10.4 | 2.0 de 11 | 3.0 |
| CfC | 0.05 | 0.912 +- 0.102 | 0.000 +- 0.000 | 0.040 +- 0.057 | 0.591 +- 0.111 | -10.7 +- 8.8 | 0.3 +- 0.5 de 6 | -28.7 +- 22.0 | 1.0 +- 0.8 de 11 | 5.7 +- 2.9 |
| GRU | 0.05 | 0.998 +- 0.001 | 0.000 +- 0.000 | 0.027 +- 0.019 | 0.606 +- 0.077 | -32.0 +- 11.4 | 0.0 +- 0.0 de 6 | -32.0 +- 11.4 | 1.0 +- 0.0 de 11 | 6.0 +- 0.8 |
| LSTM | 0.05 | 0.978 +- 0.008 | 0.000 +- 0.000 | 0.027 +- 0.019 | 0.530 +- 0.043 | -24.1 +- 3.2 | 0.0 +- 0.0 de 6 | -27.8 +- 5.0 | 0.3 +- 0.5 de 11 | 6.7 +- 1.2 |
| gradient boosting | 0.05 | 0.989 | 0.000 | 0.000 | 0.727 | -18.4 | 1.0 de 6 | -18.4 | 2.0 de 11 | 4.0 |
| regresion logistica | 0.05 | 0.971 | 0.000 | 0.000 | 0.409 | 1.6 | 1.0 de 6 | -98.1 | 1.0 de 11 | 8.0 |
