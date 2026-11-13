# Fase E final en validacion: histeresis, alerta conformal y segmentos

Generado por `python fase_e_final.py --pesos runs_fuego/modelos_final/riesgo/anticipado`. La prueba no se usa. Media +- desviacion entre semillas.

- Nivel: umbrales de CORAL en 0.5 e histeresis causal de 2 s.
- Alerta conformal por video: umbral = cuantil conformal del riesgo maximo de los 52 videos de calibracion sin fuego o controlados; se apaga cuando el riesgo baja de 0.8 veces el umbral durante 2 s.
- Referencias sin tiempo: una prediccion por segundo; riesgo = 0.5 P(incipiente) + P(incontrolado).
- Validacion: 47 videos de fuego para el nivel, 28 sin fuego, 25 controlados, 22 con propagacion. Videos con cambio a incontrolado: 6 en validacion y 5 en calibracion (estos no entran en el umbral conformal).
- Adelanto = instante del cambio a incontrolado menos instante de la primera alerta; negativo = la alerta llega despues.

| modelo | corridas | etapa | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | cambios de nivel por video |
|---|---|---|---|---|---|---|---|---|---|
| GRU | 3 | antes (umbral 0.5) | 0.682 +- 0.001 | 0.789 +- 0.018 | 0.432 +- 0.002 | 0.823 +- 0.020 | 0.737 +- 0.005 | 0.678 +- 0.004 | 2.66 +- 0.36 |
| GRU | 3 | umbral 0.5 + histeresis (final) | 0.665 +- 0.003 | 0.772 +- 0.018 | 0.414 +- 0.005 | 0.810 +- 0.020 | 0.716 +- 0.006 | 0.657 +- 0.003 | 1.87 +- 0.20 |

| modelo | alfa | umbral de riesgo | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | adelanto mediano en validacion (s) | antes del cambio en validacion | adelanto mediano en validacion + calibracion (s) | antes del cambio en validacion + calibracion | sin alerta (validacion + calibracion) |
|---|---|---|---|---|---|---|---|---|---|---|
| GRU | 0.1 (final) | 0.890 +- 0.082 | 0.000 +- 0.000 | 0.093 +- 0.019 | 0.788 +- 0.057 | -5.2 +- 4.2 | 1.0 +- 0.8 de 6 | -4.2 +- 3.4 | 2.7 +- 1.7 de 11 | 2.0 +- 1.4 |
| GRU | 0.05 | 0.998 +- 0.001 | 0.000 +- 0.000 | 0.027 +- 0.019 | 0.652 +- 0.057 | -15.7 +- 4.8 | 0.0 +- 0.0 de 6 | -15.4 +- 5.0 | 1.0 +- 0.0 de 11 | 5.0 +- 0.8 |
