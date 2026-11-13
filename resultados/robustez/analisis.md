# Robustez: caidas, pruebas pareadas y figura

Generado por `python analisis_robustez.py`. Repite la inferencia de las partes 1 y 2 (diferencia maxima de F1 y kappa con `parte1.json` y `parte2.json`: 0.0e+00) y hace un bootstrap pareado de videos (1000 remuestreos, semilla 42; los mismos remuestreos para todos los modelos, fracciones y partes de cada referencia). En los modelos con 3 semillas se promedia la metrica de las semillas dentro de cada remuestreo, como en la fase F. Medida principal (la de la fase F).

- Caida relativa = (F1 con todos los fotogramas - F1 con fotogramas quitados) / F1 con todos; positiva = el modelo empeora.
- Diferencia de caidas relativas = caida de la GRU (o la LSTM) menos caida de la CfC; positiva = la CfC cae menos.
- Significativa = el intervalo del 95 % no incluye el cero.

## Parte 1: modelos congelados

### Caida de F1 macro respecto a la prueba completa

| referencia | modelo | quitado | F1 macro | caida absoluta | IC 95 % | caida relativa | IC 95 % |
|---|---|---|---|---|---|---|---|
| IA | CfC | 25 % | 0.460 | -0.000 | -0.016 a +0.018 | +0.1 % | -3.9 a +3.5 % |
| IA | CfC | 50 % | 0.462 | +0.002 | -0.021 a +0.027 | -0.5 % | -6.2 a +4.4 % |
| IA | CfC | 75 % | 0.448 | -0.013 | -0.058 a +0.026 | +2.7 % | -5.9 a +12.2 % |
| IA | GRU | 25 % | 0.469 | -0.050 | -0.106 a +0.001 | +9.6 % | -0.2 a +18.9 % |
| IA | GRU | 50 % | 0.476 | -0.042 | -0.115 a +0.026 | +8.2 % | -5.5 a +20.2 % |
| IA | GRU | 75 % | 0.409 | -0.110 | -0.207 a -0.012 | +21.3 % | +2.3 a +36.5 % |
| IA | LSTM | 25 % | 0.465 | +0.005 | -0.016 a +0.029 | -1.0 % | -6.5 a +3.5 % |
| IA | LSTM | 50 % | 0.460 | -0.000 | -0.026 a +0.027 | +0.1 % | -6.2 a +5.5 % |
| IA | LSTM | 75 % | 0.423 | -0.037 | -0.073 a -0.004 | +8.0 % | +0.8 a +15.2 % |
| IA | gradient boosting | 25 % | 0.462 | +0.020 | +0.009 a +0.032 | -4.5 % | -7.6 a -1.8 % |
| IA | gradient boosting | 50 % | 0.450 | +0.008 | -0.017 a +0.029 | -1.7 % | -7.2 a +3.5 % |
| IA | gradient boosting | 75 % | 0.431 | -0.012 | -0.052 a +0.023 | +2.6 % | -5.8 a +11.0 % |
| IA | regresion logistica | 25 % | 0.501 | +0.010 | -0.007 a +0.028 | -2.1 % | -6.3 a +1.3 % |
| IA | regresion logistica | 50 % | 0.518 | +0.027 | +0.002 a +0.060 | -5.6 % | -13.3 a -0.4 % |
| IA | regresion logistica | 75 % | 0.500 | +0.010 | -0.036 a +0.052 | -2.0 % | -11.9 a +6.7 % |
| humana | CfC | 25 % | 0.477 | +0.002 | -0.014 a +0.020 | -0.5 % | -4.6 a +2.9 % |
| humana | CfC | 50 % | 0.470 | -0.005 | -0.025 a +0.015 | +0.9 % | -3.2 a +5.5 % |
| humana | CfC | 75 % | 0.484 | +0.009 | -0.026 a +0.041 | -1.9 % | -9.1 a +5.6 % |
| humana | GRU | 25 % | 0.461 | -0.036 | -0.089 a +0.013 | +7.3 % | -2.7 a +16.9 % |
| humana | GRU | 50 % | 0.456 | -0.041 | -0.104 a +0.025 | +8.3 % | -5.5 a +20.2 % |
| humana | GRU | 75 % | 0.419 | -0.078 | -0.171 a +0.007 | +15.7 % | -1.6 a +31.0 % |
| humana | LSTM | 25 % | 0.465 | +0.005 | -0.018 a +0.033 | -1.0 % | -7.3 a +4.0 % |
| humana | LSTM | 50 % | 0.457 | -0.004 | -0.034 a +0.027 | +0.9 % | -6.0 a +7.2 % |
| humana | LSTM | 75 % | 0.437 | -0.024 | -0.057 a +0.010 | +5.1 % | -2.3 a +12.1 % |
| humana | gradient boosting | 25 % | 0.466 | +0.015 | +0.002 a +0.030 | -3.4 % | -7.0 a -0.6 % |
| humana | gradient boosting | 50 % | 0.451 | +0.000 | -0.019 a +0.023 | -0.1 % | -5.4 a +4.3 % |
| humana | gradient boosting | 75 % | 0.422 | -0.029 | -0.059 a +0.006 | +6.4 % | -1.4 a +12.1 % |
| humana | regresion logistica | 25 % | 0.528 | +0.000 | -0.017 a +0.019 | -0.1 % | -4.1 a +3.1 % |
| humana | regresion logistica | 50 % | 0.542 | +0.015 | -0.010 a +0.048 | -2.9 % | -10.3 a +1.8 % |
| humana | regresion logistica | 75 % | 0.510 | -0.017 | -0.052 a +0.027 | +3.3 % | -5.7 a +9.5 % |

### Diferencia de caidas relativas contra la CfC (F1 macro y kappa)

| referencia | comparacion | quitado | F1: diferencia | IC 95 % | remuestreos con la CfC cayendo menos | kappa: diferencia | IC 95 % |
|---|---|---|---|---|---|---|---|
| IA | GRU - CfC | 25 % | +9.5 pp | +0.1 a +19.1 pp | 97.6 % | +20.6 pp | +1.0 a +36.8 pp |
| IA | GRU - CfC | 50 % | +8.7 pp | -4.7 a +19.6 pp | 90.2 % | +25.9 pp | -2.3 a +54.3 pp |
| IA | GRU - CfC | 75 % | +18.5 pp | +1.7 a +32.3 pp | 98.4 % | +43.5 pp | +14.9 a +74.5 pp |
| IA | LSTM - CfC | 25 % | -1.1 pp | -5.8 a +3.5 pp | 33.9 % | -4.1 pp | -21.3 a +6.7 pp |
| IA | LSTM - CfC | 50 % | +0.6 pp | -6.2 a +6.8 pp | 53.0 % | -1.5 pp | -20.5 a +17.9 pp |
| IA | LSTM - CfC | 75 % | +5.2 pp | -2.9 a +13.8 pp | 89.6 % | +7.4 pp | -11.5 a +29.0 pp |
| humana | GRU - CfC | 25 % | +7.8 pp | -1.6 a +17.0 pp | 94.4 % | +16.9 pp | -3.2 a +36.2 pp |
| humana | GRU - CfC | 50 % | +7.4 pp | -6.9 a +19.0 pp | 87.1 % | +12.9 pp | -16.6 a +34.5 pp |
| humana | GRU - CfC | 75 % | +17.6 pp | +0.6 a +33.6 pp | 97.7 % | +35.5 pp | +2.0 a +64.3 pp |
| humana | LSTM - CfC | 25 % | -0.6 pp | -6.0 a +4.0 pp | 37.1 % | -1.4 pp | -13.9 a +9.1 pp |
| humana | LSTM - CfC | 50 % | -0.1 pp | -8.2 a +6.7 pp | 47.2 % | -4.5 pp | -22.7 a +11.2 pp |
| humana | LSTM - CfC | 75 % | +7.0 pp | -1.9 a +15.6 pp | 93.0 % | +9.5 pp | -9.9 a +27.4 pp |

## Parte 2: modelos reentrenados con fotogramas quitados

### Caida de F1 macro respecto a la prueba completa

| referencia | modelo | quitado | F1 macro | caida absoluta | IC 95 % | caida relativa | IC 95 % |
|---|---|---|---|---|---|---|---|
| IA | CfC | 25 % | 0.457 | +0.008 | -0.005 a +0.025 | -1.8 % | -5.9 a +1.2 % |
| IA | CfC | 50 % | 0.461 | +0.012 | -0.002 a +0.027 | -2.7 % | -6.7 a +0.4 % |
| IA | CfC | 75 % | 0.466 | +0.017 | -0.023 a +0.053 | -3.8 % | -12.7 a +4.8 % |
| IA | GRU | 25 % | 0.506 | -0.002 | -0.039 a +0.028 | +0.5 % | -6.0 a +7.1 % |
| IA | GRU | 50 % | 0.508 | -0.001 | -0.018 a +0.019 | +0.1 % | -3.7 a +3.5 % |
| IA | GRU | 75 % | 0.497 | -0.011 | -0.050 a +0.026 | +2.2 % | -5.7 a +9.6 % |
| IA | LSTM | 25 % | 0.464 | +0.010 | -0.007 a +0.029 | -2.3 % | -6.8 a +1.4 % |
| IA | LSTM | 50 % | 0.476 | +0.022 | +0.001 a +0.047 | -4.9 % | -10.4 a -0.2 % |
| IA | LSTM | 75 % | 0.475 | +0.021 | -0.008 a +0.046 | -4.7 % | -10.2 a +1.7 % |
| humana | CfC | 25 % | 0.469 | -0.002 | -0.022 a +0.017 | +0.5 % | -3.9 a +4.6 % |
| humana | CfC | 50 % | 0.483 | +0.011 | -0.005 a +0.030 | -2.4 % | -6.8 a +1.0 % |
| humana | CfC | 75 % | 0.478 | +0.007 | -0.024 a +0.038 | -1.5 % | -8.5 a +4.9 % |
| humana | GRU | 25 % | 0.474 | -0.030 | -0.066 a +0.003 | +6.0 % | -0.6 a +12.6 % |
| humana | GRU | 50 % | 0.503 | -0.001 | -0.022 a +0.022 | +0.2 % | -4.6 a +4.2 % |
| humana | GRU | 75 % | 0.516 | +0.012 | -0.030 a +0.048 | -2.4 % | -10.9 a +5.5 % |
| humana | LSTM | 25 % | 0.464 | +0.010 | -0.011 a +0.030 | -2.2 % | -7.0 a +2.5 % |
| humana | LSTM | 50 % | 0.480 | +0.026 | +0.000 a +0.050 | -5.7 % | -11.4 a -0.1 % |
| humana | LSTM | 75 % | 0.472 | +0.018 | -0.010 a +0.043 | -3.9 % | -9.5 a +2.2 % |

### Diferencia de caidas relativas contra la CfC (F1 macro y kappa)

| referencia | comparacion | quitado | F1: diferencia | IC 95 % | remuestreos con la CfC cayendo menos | kappa: diferencia | IC 95 % |
|---|---|---|---|---|---|---|---|
| IA | GRU - CfC | 25 % | +2.3 pp | -4.6 a +9.7 pp | 77.1 % | +6.2 pp | -6.6 a +19.4 pp |
| IA | GRU - CfC | 50 % | +2.8 pp | -2.5 a +7.9 pp | 85.8 % | +4.6 pp | -5.2 a +18.7 pp |
| IA | GRU - CfC | 75 % | +6.0 pp | -7.6 a +19.2 pp | 80.7 % | +5.5 pp | -15.3 a +33.8 pp |
| IA | LSTM - CfC | 25 % | -0.4 pp | -5.0 a +4.6 pp | 42.4 % | -1.1 pp | -14.8 a +12.3 pp |
| IA | LSTM - CfC | 50 % | -2.3 pp | -7.4 a +2.7 pp | 17.6 % | -4.8 pp | -21.1 a +8.5 pp |
| IA | LSTM - CfC | 75 % | -0.9 pp | -9.5 a +7.8 pp | 39.6 % | -6.5 pp | -20.1 a +11.7 pp |
| humana | GRU - CfC | 25 % | +5.5 pp | -1.8 a +12.7 pp | 92.0 % | +13.4 pp | -0.6 a +27.5 pp |
| humana | GRU - CfC | 50 % | +2.7 pp | -3.2 a +8.3 pp | 81.6 % | +2.5 pp | -6.9 a +12.2 pp |
| humana | GRU - CfC | 75 % | -0.9 pp | -10.2 a +8.4 pp | 45.0 % | -7.6 pp | -24.5 a +7.3 pp |
| humana | LSTM - CfC | 25 % | -2.7 pp | -8.4 a +2.9 pp | 16.2 % | -5.9 pp | -19.9 a +5.3 pp |
| humana | LSTM - CfC | 50 % | -3.3 pp | -8.5 a +2.4 pp | 12.8 % | -4.1 pp | -18.0 a +8.9 pp |
| humana | LSTM - CfC | 75 % | -2.4 pp | -9.7 a +5.1 pp | 25.4 % | -6.3 pp | -18.8 a +9.6 pp |

## Parte 2 contra parte 1: reentrenado menos congelado, en cada fraccion

| referencia | modelo | quitado | F1 congelado | F1 reentrenado | diferencia | IC 95 % | kappa: diferencia | IC 95 % |
|---|---|---|---|---|---|---|---|---|
| IA | CfC | 0 % | 0.460 | 0.449 | -0.011 | -0.046 a +0.026 | +0.007 | -0.038 a +0.050 |
| IA | CfC | 25 % | 0.460 | 0.457 | -0.003 | -0.036 a +0.034 | +0.013 | -0.038 a +0.066 |
| IA | CfC | 50 % | 0.462 | 0.461 | -0.001 | -0.029 a +0.032 | +0.016 | -0.026 a +0.060 |
| IA | CfC | 75 % | 0.448 | 0.466 | +0.018 | -0.016 a +0.049 | +0.020 | -0.021 a +0.060 |
| IA | GRU | 0 % | 0.519 | 0.508 | -0.011 | -0.043 a +0.027 | -0.010 | -0.059 a +0.052 |
| IA | GRU | 25 % | 0.469 | 0.506 | +0.037 | +0.000 a +0.075 | +0.051 | -0.004 a +0.112 |
| IA | GRU | 50 % | 0.476 | 0.508 | +0.031 | -0.016 a +0.087 | +0.079 | -0.000 a +0.161 |
| IA | GRU | 75 % | 0.409 | 0.497 | +0.088 | +0.011 a +0.165 | +0.148 | +0.042 a +0.247 |
| IA | LSTM | 0 % | 0.460 | 0.454 | -0.006 | -0.032 a +0.013 | -0.008 | -0.040 a +0.024 |
| IA | LSTM | 25 % | 0.465 | 0.464 | -0.001 | -0.022 a +0.020 | -0.012 | -0.046 a +0.016 |
| IA | LSTM | 50 % | 0.460 | 0.476 | +0.016 | -0.005 a +0.041 | +0.010 | -0.026 a +0.046 |
| IA | LSTM | 75 % | 0.423 | 0.475 | +0.052 | +0.012 a +0.097 | +0.050 | -0.008 a +0.110 |
| humana | CfC | 0 % | 0.475 | 0.472 | -0.003 | -0.028 a +0.020 | +0.016 | -0.018 a +0.052 |
| humana | CfC | 25 % | 0.477 | 0.469 | -0.008 | -0.032 a +0.013 | +0.002 | -0.029 a +0.034 |
| humana | CfC | 50 % | 0.470 | 0.483 | +0.013 | -0.009 a +0.033 | +0.039 | +0.011 a +0.067 |
| humana | CfC | 75 % | 0.484 | 0.478 | -0.005 | -0.036 a +0.023 | -0.000 | -0.039 a +0.041 |
| humana | GRU | 0 % | 0.497 | 0.504 | +0.006 | -0.042 a +0.055 | +0.006 | -0.061 a +0.074 |
| humana | GRU | 25 % | 0.461 | 0.474 | +0.012 | -0.027 a +0.057 | +0.003 | -0.052 a +0.063 |
| humana | GRU | 50 % | 0.456 | 0.503 | +0.047 | -0.016 a +0.107 | +0.066 | -0.021 a +0.147 |
| humana | GRU | 75 % | 0.419 | 0.516 | +0.097 | +0.020 a +0.172 | +0.143 | +0.036 a +0.243 |
| humana | LSTM | 0 % | 0.460 | 0.454 | -0.006 | -0.031 a +0.021 | -0.003 | -0.032 a +0.028 |
| humana | LSTM | 25 % | 0.465 | 0.464 | -0.001 | -0.028 a +0.024 | -0.001 | -0.030 a +0.025 |
| humana | LSTM | 50 % | 0.457 | 0.480 | +0.024 | -0.000 a +0.045 | +0.018 | -0.019 a +0.051 |
| humana | LSTM | 75 % | 0.437 | 0.472 | +0.035 | -0.000 a +0.066 | +0.037 | -0.018 a +0.082 |

Figura: `resultados/robustez/f1_vs_quitados.png`.
