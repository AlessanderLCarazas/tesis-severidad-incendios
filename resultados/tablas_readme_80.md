# Tablas del README del 80 %

Generado por `python figuras_readme_80.py --solo tablas` desde `resultados/prueba/metricas.json`, `resultados/prueba/analisis.md` y `resultados/robustez/analisis.json`.

## Resultados en la prueba

| Modelo | Referencia | F1 macro [IC 95 %] | Kappa lineal [IC 95 %] | mAP tIoU 0.5 | Falsas alarmas sin fuego | Falsas alarmas en controlados | Adelanto mediano (alerta antes del cambio) |
|---|---|---|---|---|---|---|---|
| CfC | IA | 0.460 [0.393, 0.522] | 0.319 [0.183, 0.442] | 0.216 [0.151, 0.303] | 0 de 32 | 7 de 30 (23 %) | +0.7 s (8.3 de 21) |
| CfC | humana | 0.475 [0.410, 0.540] | 0.345 [0.241, 0.449] | 0.103 [0.066, 0.199] | 0 de 32 | 5.3 de 29 (18 %) | +23.6 s (12.3 de 16) |
| GRU | IA | 0.519 [0.433, 0.603] | 0.371 [0.225, 0.516] | 0.299 [0.219, 0.400] | 0 de 32 | 6.7 de 30 (22 %) | -2.7 s (8.3 de 21) |
| GRU | humana | 0.497 [0.416, 0.579] | 0.352 [0.238, 0.472] | 0.136 [0.071, 0.290] | 0 de 32 | 6 de 29 (21 %) | +20.5 s (11.3 de 16) |
| LSTM | IA | 0.460 [0.396, 0.526] | 0.323 [0.204, 0.438] | 0.258 | 0 de 32 | 7 de 30 (23 %) | +4.4 s (9 de 21) |
| LSTM | humana | 0.460 [0.391, 0.528] | 0.340 [0.244, 0.440] | 0.120 | 0 de 32 | 6.7 de 29 (23 %) | +29.7 s (10.7 de 16) |
| gradient boosting | IA | 0.442 [0.368, 0.525] | 0.292 [0.178, 0.413] | 0.204 [0.140, 0.305] | 1 de 32 | 8 de 30 (27 %) | +3.8 s (12 de 21) |
| gradient boosting | humana | 0.451 [0.362, 0.531] | 0.320 [0.218, 0.426] | 0.125 [0.066, 0.257] | 1 de 32 | 6 de 29 (21 %) | +26.6 s (13 de 16) |
| regresión logística | IA | 0.491 [0.407, 0.570] | 0.311 [0.158, 0.449] | 0.204 | 0 de 32 | 3 de 30 (10 %) | -10.5 s (5 de 21) |
| regresión logística | humana | 0.527 [0.433, 0.604] | 0.365 [0.220, 0.485] | 0.069 | 0 de 32 | 2 de 29 (7 %) | +13.2 s (10 de 16) |

## Robustez: F1 macro con 0 y 75 % de fotogramas quitados

| Modelo | Referencia | Congelado, 0 % | Congelado, 75 % | Reentrenado, 0 % | Reentrenado, 75 % |
|---|---|---|---|---|---|
| CfC | IA | 0.460 | 0.448 | 0.449 | 0.466 |
| CfC | humana | 0.475 | 0.484 | 0.472 | 0.478 |
| GRU | IA | 0.519 | 0.409 | 0.508 | 0.497 |
| GRU | humana | 0.497 | 0.419 | 0.504 | 0.516 |
| LSTM | IA | 0.460 | 0.423 | 0.454 | 0.475 |
| LSTM | humana | 0.460 | 0.437 | 0.454 | 0.472 |

Diferencia de caidas relativas del F1 macro a 75 % (GRU menos CfC; positiva = la CfC cae menos):

- parte1, ia: +18.5 pp [+1.7, +32.3]
- parte1, humano: +17.6 pp [+0.6, +33.6]
- parte2, ia: +6.0 pp [-7.6, +19.2]
- parte2, humano: -0.9 pp [-10.2, +8.4]
