# Fase E en validacion: histeresis, alerta conformal y segmentos

Generado por `python fase_e.py`. La prueba no se usa. Media +- desviacion entre semillas.

- **Configuracion final: umbrales de CORAL en 0.5 con histeresis, y alerta con alfa 0.10.**
- Histeresis de nivel: el nivel sube cuando el nivel crudo se mantiene por encima durante 2 s y baja cuando se mantiene por debajo durante 2 s (causal).
- Umbrales ajustados (solo como comparacion): rejilla de 0.05 sobre las dos probabilidades de CORAL, eligiendo el mayor F1 macro de escena por segundo en los videos de fuego de calibracion.
- Alerta conformal por video: la puntuacion de cada video de calibracion sin fuego o controlado (52) es su riesgo maximo; el umbral es el cuantil conformal. La alerta se activa cuando el riesgo llega al umbral y se apaga cuando baja de 0.8 veces el umbral durante 2 s.
- Referencias sin tiempo: una prediccion por segundo; riesgo = 0.5 P(incipiente) + P(incontrolado); misma histeresis y misma alerta.
- Validacion: 47 videos de fuego para el nivel, 28 sin fuego, 25 controlados, 22 con propagacion y 6 con cambio a incontrolado.

| modelo | corridas | etapa | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | cambios de nivel por video |
|---|---|---|---|---|---|---|---|---|---|
| CFC | 3 | antes (umbral 0.5) | 0.544 +- 0.015 | 0.711 +- 0.051 | 0.170 +- 0.034 | 0.751 +- 0.009 | 0.644 +- 0.014 | 0.543 +- 0.040 | 5.26 +- 0.67 |
| CFC | 3 | umbral 0.5 + histeresis (final) | 0.536 +- 0.019 | 0.706 +- 0.050 | 0.169 +- 0.030 | 0.734 +- 0.006 | 0.629 +- 0.018 | 0.529 +- 0.041 | 2.16 +- 0.13 |
| CFC | 3 | solo umbrales ajustados | 0.477 +- 0.019 | 0.690 +- 0.046 | 0.115 +- 0.031 | 0.627 +- 0.041 | 0.547 +- 0.036 | 0.465 +- 0.037 | 6.65 +- 0.40 |
| CFC | 3 | umbrales ajustados + histeresis | 0.464 +- 0.020 | 0.678 +- 0.050 | 0.113 +- 0.034 | 0.602 +- 0.044 | 0.529 +- 0.036 | 0.444 +- 0.033 | 2.20 +- 0.14 |
| GRU | 3 | antes (umbral 0.5) | 0.621 +- 0.008 | 0.788 +- 0.015 | 0.208 +- 0.052 | 0.866 +- 0.018 | 0.753 +- 0.012 | 0.698 +- 0.022 | 3.41 +- 0.26 |
| GRU | 3 | umbral 0.5 + histeresis (final) | 0.612 +- 0.002 | 0.771 +- 0.015 | 0.209 +- 0.040 | 0.855 +- 0.022 | 0.732 +- 0.017 | 0.679 +- 0.026 | 2.21 +- 0.06 |
| GRU | 3 | solo umbrales ajustados | 0.579 +- 0.041 | 0.802 +- 0.006 | 0.175 +- 0.135 | 0.761 +- 0.033 | 0.690 +- 0.020 | 0.633 +- 0.014 | 2.70 +- 0.37 |
| GRU | 3 | umbrales ajustados + histeresis | 0.561 +- 0.043 | 0.787 +- 0.009 | 0.155 +- 0.137 | 0.740 +- 0.033 | 0.666 +- 0.018 | 0.607 +- 0.013 | 1.67 +- 0.11 |
| LSTM | 3 | antes (umbral 0.5) | 0.552 +- 0.048 | 0.738 +- 0.035 | 0.137 +- 0.061 | 0.782 +- 0.063 | 0.685 +- 0.042 | 0.583 +- 0.074 | 4.76 +- 0.08 |
| LSTM | 3 | umbral 0.5 + histeresis (final) | 0.548 +- 0.048 | 0.726 +- 0.036 | 0.156 +- 0.054 | 0.763 +- 0.067 | 0.663 +- 0.044 | 0.563 +- 0.078 | 2.71 +- 0.06 |
| LSTM | 3 | solo umbrales ajustados | 0.528 +- 0.050 | 0.735 +- 0.018 | 0.139 +- 0.072 | 0.708 +- 0.063 | 0.633 +- 0.030 | 0.536 +- 0.066 | 4.80 +- 0.62 |
| LSTM | 3 | umbrales ajustados + histeresis | 0.516 +- 0.047 | 0.720 +- 0.021 | 0.138 +- 0.062 | 0.691 +- 0.062 | 0.611 +- 0.030 | 0.514 +- 0.069 | 2.75 +- 0.18 |
| gradient boosting | 1 | antes (umbral 0.5) | 0.622 | 0.623 | 0.444 | 0.799 | 0.679 | 0.519 | 12.81 |
| gradient boosting | 1 | umbral 0.5 + histeresis (final) | 0.653 | 0.664 | 0.449 | 0.847 | 0.720 | 0.598 | 2.30 |
| regresion logistica | 1 | antes (umbral 0.5) | 0.482 | 0.468 | 0.285 | 0.691 | 0.514 | 0.357 | 12.28 |
| regresion logistica | 1 | umbral 0.5 + histeresis (final) | 0.479 | 0.488 | 0.268 | 0.681 | 0.509 | 0.371 | 3.13 |

| modelo | alfa | umbral de riesgo | falsas alarmas en videos sin fuego | falsas alarmas en videos controlados | videos con propagacion con alerta | adelanto mediano (s) | videos con alerta antes del cambio | sin alerta |
|---|---|---|---|---|---|---|---|---|
| CFC | 0.1 (final) | 0.723 +- 0.064 | 0.000 +- 0.000 | 0.187 +- 0.038 | 0.803 +- 0.077 | -6.8 +- 1.8 | 1.3 +- 0.5 de 6 | 0.3 +- 0.5 |
| CFC | 0.05 | 0.912 +- 0.102 | 0.000 +- 0.000 | 0.040 +- 0.057 | 0.591 +- 0.111 | -10.7 +- 8.8 | 0.3 +- 0.5 de 6 | 3.0 +- 1.6 |
| GRU | 0.1 (final) | 0.614 +- 0.037 | 0.000 +- 0.000 | 0.107 +- 0.038 | 0.848 +- 0.043 | -4.0 +- 3.6 | 2.0 +- 0.8 de 6 | 0.3 +- 0.5 |
| GRU | 0.05 | 0.968 +- 0.034 | 0.000 +- 0.000 | 0.013 +- 0.019 | 0.606 +- 0.021 | -19.8 +- 6.3 | 0.0 +- 0.0 de 6 | 2.7 +- 0.5 |
| LSTM | 0.1 (final) | 0.599 +- 0.025 | 0.000 +- 0.000 | 0.093 +- 0.019 | 0.833 +- 0.043 | -8.7 +- 8.4 | 1.7 +- 0.5 de 6 | 0.0 +- 0.0 |
| LSTM | 0.05 | 0.978 +- 0.008 | 0.000 +- 0.000 | 0.027 +- 0.019 | 0.530 +- 0.043 | -24.1 +- 3.2 | 0.0 +- 0.0 de 6 | 3.3 +- 0.5 |
| gradient boosting | 0.1 (final) | 0.869 | 0.071 | 0.360 | 0.864 | 0.4 | 3.0 de 6 | 0.0 |
| gradient boosting | 0.05 | 0.989 | 0.000 | 0.000 | 0.727 | -18.4 | 1.0 de 6 | 1.0 |
| regresion logistica | 0.1 (final) | 0.832 | 0.000 | 0.080 | 0.682 | -9.7 | 1.0 de 6 | 1.0 |
| regresion logistica | 0.05 | 0.971 | 0.000 | 0.000 | 0.409 | 1.6 | 1.0 de 6 | 5.0 |

| corrida | umbral > controlado | umbral > incipiente | F1 en calibracion | segmentos por video (final) |
|---|---|---|---|---|
| cfc_s42 | 0.60 | 0.85 | 0.551 | 2.5 |
| cfc_s43 | 0.50 | 0.80 | 0.583 | 2.4 |
| cfc_s44 | 0.45 | 0.90 | 0.535 | 2.6 |
| gru_s42 | 0.40 | 0.85 | 0.560 | 2.9 |
| gru_s43 | 0.65 | 0.95 | 0.513 | 2.8 |
| gru_s44 | 0.60 | 0.65 | 0.571 | 2.5 |
| lstm_s42 | 0.50 | 0.75 | 0.534 | 3.2 |
| lstm_s43 | 0.55 | 0.90 | 0.614 | 2.9 |
| lstm_s44 | 0.40 | 0.70 | 0.579 | 3.1 |
| gradient boosting | - | - | - | 2.5 |
| regresion logistica | - | - | - | 3.0 |
