# Fase E en validacion con la configuracion elegida por modelo

Generado por `python fase_e_elegida.py`. La prueba no se usa. Media +- desviacion entre semillas.

- Nivel: umbrales de CORAL en 0.5 e histeresis causal de 2 s.
- Alerta conformal por video: umbral = cuantil conformal del riesgo maximo de los 52 videos de calibracion sin fuego o controlados; se apaga cuando el riesgo baja de 0.8 veces el umbral durante 2 s.
- Referencias sin tiempo: una prediccion por segundo; riesgo = 0.5 P(incipiente) + P(incontrolado).
- Validacion: 47 videos de fuego para el nivel, 28 sin fuego, 25 controlados, 22 con propagacion. Videos con cambio a incontrolado: 6 en validacion y 5 en calibracion (estos no entran en el umbral conformal).
- Adelanto = instante del cambio a incontrolado menos instante de la primera alerta; negativo = la alerta llega despues.

| modelo | corridas | etapa | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | cambios de nivel por video |
|---|---|---|---|---|---|---|---|---|---|
| CfC | 1 | antes (umbral 0.5) | 0.500 | 0.646 | 0.134 | 0.721 | 0.596 | 0.467 | 8.17 |
| CfC | 1 | umbral 0.5 + histeresis (final) | 0.483 | 0.629 | 0.124 | 0.696 | 0.568 | 0.440 | 3.34 |

| modelo | alfa | umbral de riesgo | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | adelanto mediano en validacion (s) | antes del cambio en validacion | adelanto mediano en validacion + calibracion (s) | antes del cambio en validacion + calibracion | sin alerta (validacion + calibracion) |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.1 (final) | 0.743 | 0.000 | 0.160 | 0.773 | -1.6 | 2.0 de 6 | -1.1 | 4.0 de 11 | 1.0 |
| CfC | 0.05 | 0.983 | 0.000 | 0.000 | 0.500 | -11.7 | 1.0 de 6 | -22.0 | 1.0 de 11 | 7.0 |
