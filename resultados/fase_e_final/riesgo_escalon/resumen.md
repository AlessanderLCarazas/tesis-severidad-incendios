# Fase E final en validacion: histeresis, alerta conformal y segmentos

Generado por `python fase_e_final.py --pesos runs_fuego/modelos_final/riesgo/escalon`. La prueba no se usa. Media +- desviacion entre semillas.

- Nivel: umbrales de CORAL en 0.5 e histeresis causal de 2 s.
- Alerta conformal por video: umbral = cuantil conformal del riesgo maximo de los 52 videos de calibracion sin fuego o controlados; se apaga cuando el riesgo baja de 0.8 veces el umbral durante 2 s.
- Referencias sin tiempo: una prediccion por segundo; riesgo = 0.5 P(incipiente) + P(incontrolado).
- Validacion: 47 videos de fuego para el nivel, 28 sin fuego, 25 controlados, 22 con propagacion. Videos con cambio a incontrolado: 6 en validacion y 5 en calibracion (estos no entran en el umbral conformal).
- Adelanto = instante del cambio a incontrolado menos instante de la primera alerta; negativo = la alerta llega despues.

| modelo | corridas | etapa | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | cambios de nivel por video |
|---|---|---|---|---|---|---|---|---|---|
| GRU | 3 | antes (umbral 0.5) | 0.660 +- 0.008 | 0.760 +- 0.005 | 0.416 +- 0.017 | 0.805 +- 0.027 | 0.714 +- 0.008 | 0.641 +- 0.022 | 2.52 +- 0.35 |
| GRU | 3 | umbral 0.5 + histeresis (final) | 0.641 +- 0.009 | 0.738 +- 0.003 | 0.397 +- 0.023 | 0.789 +- 0.025 | 0.690 +- 0.008 | 0.615 +- 0.022 | 1.89 +- 0.16 |

| modelo | alfa | umbral de riesgo | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | adelanto mediano en validacion (s) | antes del cambio en validacion | adelanto mediano en validacion + calibracion (s) | antes del cambio en validacion + calibracion | sin alerta (validacion + calibracion) |
|---|---|---|---|---|---|---|---|---|---|---|
| GRU | 0.1 (final) | 0.768 +- 0.113 | 0.000 +- 0.000 | 0.147 +- 0.050 | 0.833 +- 0.057 | -2.7 +- 2.1 | 1.7 +- 0.5 de 6 | -2.2 +- 2.5 | 4.0 +- 0.8 de 11 | 1.7 +- 0.9 |
| GRU | 0.05 | 0.998 +- 0.001 | 0.000 +- 0.000 | 0.027 +- 0.019 | 0.606 +- 0.077 | -32.0 +- 11.4 | 0.0 +- 0.0 de 6 | -32.0 +- 11.4 | 1.0 +- 0.0 de 11 | 6.0 +- 0.8 |
