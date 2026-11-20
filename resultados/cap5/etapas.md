# Experimentos por etapa (bloque del asesor, despues de la fase F)

Generado por `python experimentos_etapas.py`. Solo ajuste y validacion; nada de esto se uso para elegir el sistema y la configuracion congelada no cambia.

## E6. Cortes de escena: doble umbral contra un solo umbral

Distancias h guardadas en `resultados/cortes/indicadores/indicadores.csv` (13 videos de ajuste y 11 de validacion; 19 y 18 cortes revisados). Regla de un solo umbral: corte si h >= T, sin usar la camara. Misma rejilla (0.25 a 1.0, paso 0.025), misma separacion minima de 1 s, mismo emparejamiento (0.5 s) y mismo criterio: mas cortes detectados en ajuste sin falsos, sin detecciones sin revisar y sin duplicados. En ajuste empatan T = 0.675 (12 cortes); se toma el centro, T = 0.675, como se hizo con la regla doble.

| regla | T_alto | T_bajo | ajuste: detectados | ajuste: falsos | validacion: detectados | validacion: falsos | validacion: sin revisar |
|---|---|---|---|---|---|---|---|
| doble umbral (elegida) | 0.850 | 0.650 | 12 de 19 | 0 | 14 de 18 | 0 | 0 |
| un solo umbral | 0.675 | 0.675 | 12 de 19 | 0 | 14 de 18 | 2 | 2 |

## E8. Decision en validacion: alerta y histeresis

Salidas de las semillas 42, 43 y 44 en validacion (control: la configuracion elegida reproduce `resultados/fase_e_elegida/` con diferencia 0.0e+00). Referencia de la IA. 47 videos de fuego para el nivel, 25 controlados, 28 sin fuego y 6 con cambio a incontrolado. Cambios por minuto: cambios del nivel por segundo dentro del tramo util, sobre el total de minutos. Umbral de alerta fijo: misma regla de activacion y desactivacion (0.8 veces el umbral durante 2 s). La histeresis solo cambia el nivel, no la alerta.

Cambios por minuto de la referencia de la IA en estos videos: 0.13.

### GRU

| variante | F1 macro | F1 incipiente | falsas alarmas en controlados | falsas alarmas sin fuego | cambios por minuto | EWT mediana (s) | antes del cambio | sin alerta |
|---|---|---|---|---|---|---|---|---|
| conformal (alfa 0.10), histéresis 2 s (elegida) | 0.641 +- 0.009 | 0.397 +- 0.023 | 0.147 +- 0.050 (de 25) | 0.000 +- 0.000 (de 28) | 1.93 +- 0.17 | -2.7 +- 2.1 | 1.7 +- 0.5 de 6 | 0.3 +- 0.5 |
| umbral fijo 0.5, histéresis 2 s | 0.641 +- 0.009 | 0.397 +- 0.023 | 0.387 +- 0.068 (de 25) | 0.000 +- 0.000 (de 28) | 1.93 +- 0.17 | -1.8 +- 2.0 | 2.0 +- 0.8 de 6 | 0.0 +- 0.0 |
| conformal, histéresis 0 s | 0.660 +- 0.008 | 0.416 +- 0.017 | 0.147 +- 0.050 (de 25) | 0.000 +- 0.000 (de 28) | 2.58 +- 0.36 | -2.7 +- 2.1 | 1.7 +- 0.5 de 6 | 0.3 +- 0.5 |
| conformal, histéresis 1 s | 0.652 +- 0.008 | 0.410 +- 0.017 | 0.147 +- 0.050 (de 25) | 0.000 +- 0.000 (de 28) | 2.34 +- 0.35 | -2.7 +- 2.1 | 1.7 +- 0.5 de 6 | 0.3 +- 0.5 |
| conformal, histéresis 3 s | 0.636 +- 0.010 | 0.397 +- 0.022 | 0.147 +- 0.050 (de 25) | 0.000 +- 0.000 (de 28) | 1.65 +- 0.12 | -2.7 +- 2.1 | 1.7 +- 0.5 de 6 | 0.3 +- 0.5 |

### CfC

| variante | F1 macro | F1 incipiente | falsas alarmas en controlados | falsas alarmas sin fuego | cambios por minuto | EWT mediana (s) | antes del cambio | sin alerta |
|---|---|---|---|---|---|---|---|---|
| conformal (alfa 0.10), histéresis 2 s (elegida) | 0.536 +- 0.019 | 0.169 +- 0.030 | 0.187 +- 0.038 (de 25) | 0.000 +- 0.000 (de 28) | 2.21 +- 0.14 | -6.8 +- 1.8 | 1.3 +- 0.5 de 6 | 0.3 +- 0.5 |
| umbral fijo 0.5, histéresis 2 s | 0.536 +- 0.019 | 0.169 +- 0.030 | 0.307 +- 0.094 (de 25) | 0.012 +- 0.017 (de 28) | 2.21 +- 0.14 | 0.0 +- 3.1 | 2.7 +- 0.5 de 6 | 0.0 +- 0.0 |
| conformal, histéresis 0 s | 0.544 +- 0.015 | 0.170 +- 0.034 | 0.187 +- 0.038 (de 25) | 0.000 +- 0.000 (de 28) | 5.38 +- 0.69 | -6.8 +- 1.8 | 1.3 +- 0.5 de 6 | 0.3 +- 0.5 |
| conformal, histéresis 1 s | 0.536 +- 0.018 | 0.155 +- 0.036 | 0.187 +- 0.038 (de 25) | 0.000 +- 0.000 (de 28) | 3.38 +- 0.14 | -6.8 +- 1.8 | 1.3 +- 0.5 de 6 | 0.3 +- 0.5 |
| conformal, histéresis 3 s | 0.537 +- 0.020 | 0.179 +- 0.042 | 0.187 +- 0.038 (de 25) | 0.000 +- 0.000 (de 28) | 1.77 +- 0.06 | -6.8 +- 1.8 | 1.3 +- 0.5 de 6 | 0.3 +- 0.5 |

## E5. Poder separador de cada columna del vector de escena

Validacion, referencia de la IA, 47 videos de fuego y 2757 segundos (1724 en incipiente o incontrolado). Valor por segundo: media de los fotogramas del segundo, con el vector sin normalizar (la normalizacion es monotona salvo el recorte en p1 y p99) y solo en los tramos de 10 o mas fotogramas, como en el entrenamiento. AUC de Controlado contra Incipiente o Incontrolado; AUC orientada = max(AUC, 1 - AUC), para ordenar sin importar el sentido.

| posicion | columna | AUC | AUC orientada | sentido |
|---|---|---|---|---|
| 1 | área detectada de humo (`area_det_humo`) | 0.829 | 0.829 | mayor en propagacion |
| 2 | d11 área de humo (media) (`d11_media`) | 0.812 | 0.812 | mayor en propagacion |
| 3 | detecciones de humo (`n_det_humo`) | 0.811 | 0.811 | mayor en propagacion |
| 4 | d11 área de humo (máx.) (`d11_max`) | 0.811 | 0.811 | mayor en propagacion |
| 5 | d10 confianza de humo (media) (`d10_media`) | 0.808 | 0.808 | mayor en propagacion |
| 6 | d10 confianza de humo (máx.) (`d10_max`) | 0.805 | 0.805 | mayor en propagacion |
| 7 | d8 persistencia (media) (`d8_media`) | 0.297 | 0.703 | menor en propagacion |
| 8 | edad máxima de trayectoria (`edad_max`) | 0.305 | 0.695 | menor en propagacion |
| 9 | d6 centro y (media) (`d6_media`) | 0.313 | 0.687 | menor en propagacion |
| 10 | detecciones de fuego (`n_det_fuego`) | 0.654 | 0.654 | mayor en propagacion |
| 11 | d8 persistencia (máx.) (`d8_max`) | 0.349 | 0.651 | menor en propagacion |
| 12 | d6 centro y (máx.) (`d6_max`) | 0.350 | 0.650 | menor en propagacion |
| 13 | d12 tasa del humo (máx.) (`d12_max`) | 0.631 | 0.631 | mayor en propagacion |
| 14 | d5 centro x (media) (`d5_media`) | 0.397 | 0.603 | menor en propagacion |
| 15 | d4 crecimiento relativo (máx.) (`d4_max`) | 0.589 | 0.589 | mayor en propagacion |
| 16 | sin trayectoria (`sin_trayectoria`) | 0.585 | 0.585 | mayor en propagacion |
| 17 | d9 variación de forma (máx.) (`d9_max`) | 0.585 | 0.585 | mayor en propagacion |
| 18 | d13 regiones activas (máx.) (`d13_max`) | 0.424 | 0.576 | menor en propagacion |
| 19 | d13 regiones activas (media) (`d13_media`) | 0.424 | 0.576 | menor en propagacion |
| 20 | trayectorias de al menos 1 s (`n_tray_1s`) | 0.424 | 0.576 | menor en propagacion |
| 21 | d9 variación de forma (media) (`d9_media`) | 0.572 | 0.572 | mayor en propagacion |
| 22 | d1 confianza (media) (`d1_media`) | 0.430 | 0.570 | menor en propagacion |
| 23 | d2 área (media) (`d2_media`) | 0.434 | 0.566 | menor en propagacion |
| 24 | d3 tasa de crecimiento (máx.) (`d3_max`) | 0.556 | 0.556 | mayor en propagacion |
| 25 | d5 centro x (máx.) (`d5_max`) | 0.446 | 0.554 | menor en propagacion |
| 26 | d2 área (máx.) (`d2_max`) | 0.454 | 0.546 | menor en propagacion |
| 27 | d1 confianza (máx.) (`d1_max`) | 0.461 | 0.539 | menor en propagacion |
| 28 | trayectorias activas (`n_tray`) | 0.536 | 0.536 | mayor en propagacion |
| 29 | d7 velocidad del centro (máx.) (`d7_max`) | 0.525 | 0.525 | mayor en propagacion |
| 30 | d4 crecimiento relativo (media) (`d4_media`) | 0.513 | 0.513 | mayor en propagacion |
| 31 | d12 tasa del humo (media) (`d12_media`) | 0.513 | 0.513 | mayor en propagacion |
| 32 | d3 tasa de crecimiento (media) (`d3_media`) | 0.511 | 0.511 | mayor en propagacion |
| 33 | d7 velocidad del centro (media) (`d7_media`) | 0.508 | 0.508 | mayor en propagacion |
| 34 | área detectada de fuego (`area_det_fuego`) | 0.494 | 0.506 | menor en propagacion |

