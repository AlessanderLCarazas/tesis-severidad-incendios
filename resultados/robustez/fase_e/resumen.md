# Fase E en validacion con la configuracion elegida por modelo

Generado por `python fase_e_elegida.py`. La prueba no se usa. Media +- desviacion entre semillas.

- Nivel: umbrales de CORAL en 0.5 e histeresis causal de 2 s.
- Alerta conformal por video: umbral = cuantil conformal del riesgo maximo de los 52 videos de calibracion sin fuego o controlados; se apaga cuando el riesgo baja de 0.8 veces el umbral durante 2 s.
- Referencias sin tiempo: una prediccion por segundo; riesgo = 0.5 P(incipiente) + P(incontrolado).
- Validacion: 47 videos de fuego para el nivel, 28 sin fuego, 25 controlados, 22 con propagacion. Videos con cambio a incontrolado: 6 en validacion y 5 en calibracion (estos no entran en el umbral conformal).
- Adelanto = instante del cambio a incontrolado menos instante de la primera alerta; negativo = la alerta llega despues.

| modelo | corridas | etapa | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | cambios de nivel por video |
|---|---|---|---|---|---|---|---|---|---|
| CfC | 3 | antes (umbral 0.5) | 0.548 +- 0.039 | 0.714 +- 0.041 | 0.144 +- 0.006 | 0.787 +- 0.085 | 0.678 +- 0.074 | 0.570 +- 0.098 | 6.33 +- 1.53 |
| CfC | 3 | umbral 0.5 + histeresis (final) | 0.537 +- 0.033 | 0.700 +- 0.039 | 0.129 +- 0.026 | 0.781 +- 0.087 | 0.663 +- 0.068 | 0.556 +- 0.094 | 2.64 +- 0.47 |
| GRU | 3 | antes (umbral 0.5) | 0.599 +- 0.037 | 0.702 +- 0.078 | 0.301 +- 0.019 | 0.793 +- 0.017 | 0.660 +- 0.048 | 0.595 +- 0.061 | 3.23 +- 0.27 |
| GRU | 3 | umbral 0.5 + histeresis (final) | 0.588 +- 0.046 | 0.682 +- 0.083 | 0.303 +- 0.045 | 0.780 +- 0.018 | 0.640 +- 0.051 | 0.575 +- 0.065 | 2.13 +- 0.10 |
| LSTM | 3 | antes (umbral 0.5) | 0.569 +- 0.019 | 0.740 +- 0.032 | 0.168 +- 0.004 | 0.800 +- 0.028 | 0.691 +- 0.031 | 0.601 +- 0.048 | 5.40 +- 0.13 |
| LSTM | 3 | umbral 0.5 + histeresis (final) | 0.561 +- 0.014 | 0.726 +- 0.036 | 0.171 +- 0.017 | 0.787 +- 0.024 | 0.671 +- 0.028 | 0.584 +- 0.047 | 2.91 +- 0.05 |

| modelo | alfa | umbral de riesgo | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | adelanto mediano en validacion (s) | antes del cambio en validacion | adelanto mediano en validacion + calibracion (s) | antes del cambio en validacion + calibracion | sin alerta (validacion + calibracion) |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.1 (final) | 0.711 +- 0.075 | 0.000 +- 0.000 | 0.133 +- 0.019 | 0.864 +- 0.037 | -3.8 +- 3.6 | 2.3 +- 0.5 de 6 | -2.6 +- 5.2 | 4.7 +- 1.2 de 11 | 0.0 +- 0.0 |
| GRU | 0.1 (final) | 0.622 +- 0.071 | 0.000 +- 0.000 | 0.173 +- 0.068 | 0.909 +- 0.037 | -1.3 +- 0.4 | 1.7 +- 0.5 de 6 | -1.6 +- 1.3 | 4.0 +- 1.4 de 11 | 0.0 +- 0.0 |
| LSTM | 0.1 (final) | 0.702 +- 0.005 | 0.000 +- 0.000 | 0.080 +- 0.033 | 0.773 +- 0.000 | -5.6 +- 1.2 | 1.7 +- 0.5 de 6 | -5.0 +- 1.1 | 3.3 +- 0.5 de 11 | 1.3 +- 0.5 |
| CfC | 0.05 | 0.949 +- 0.046 | 0.000 +- 0.000 | 0.053 +- 0.019 | 0.591 +- 0.074 | -20.2 +- 6.1 | 0.3 +- 0.5 de 6 | -12.6 +- 6.4 | 2.0 +- 1.6 de 11 | 4.7 +- 2.9 |
| GRU | 0.05 | 0.899 +- 0.118 | 0.000 +- 0.000 | 0.067 +- 0.019 | 0.773 +- 0.074 | -8.1 +- 5.5 | 0.7 +- 0.9 de 6 | -7.3 +- 5.7 | 2.3 +- 1.9 de 11 | 2.7 +- 1.9 |
| LSTM | 0.05 | 0.981 +- 0.004 | 0.000 +- 0.000 | 0.027 +- 0.019 | 0.591 +- 0.037 | -23.7 +- 6.3 | 0.0 +- 0.0 de 6 | -21.6 +- 6.1 | 1.0 +- 0.0 de 11 | 5.3 +- 0.5 |
