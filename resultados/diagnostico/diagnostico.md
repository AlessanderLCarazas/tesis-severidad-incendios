# Diagnostico antes de la fase E

Generado por `python diagnostico_escena.py`. No cambia el modelo, no reentrena los modelos temporales y no usa la prueba.

## 1. Segundos de validacion sin ninguna trayectoria de 1 s o mas

- Total: 378 s de 2757 s de escena.
- Cada segundo va a una sola causa, en este orden de prioridad: corte, trayectorias cortas, humo, sin deteccion, otra.
- Las detecciones de humo y de fuego sin trayectoria salen de `curvas/` (YOLO a 5 fps con confianza 0.25, de `preanotar.py`); el seguimiento usa confianza 0.1, asi que no son exactamente las mismas detecciones.
- Hueco: racha de 3 s o menos sin trayectoria, con trayectoria en el segundo anterior y en el siguiente.

| causa | controlado (s) | incipiente (s) | incontrolado (s) | total (s) | en huecos de 3 s o menos |
|---|---|---|---|---|---|
| dentro de los 3 s posteriores a un corte de escena | 0 | 9 | 25 | 34 | 19 |
| solo trayectorias de fuego de menos de 1 s | 32 | 16 | 165 | 213 | 50 |
| solo detecciones de humo | 13 | 0 | 12 | 25 | 3 |
| sin ninguna deteccion | 9 | 6 | 12 | 27 | 0 |
| otra: deteccion de fuego que no forma trayectoria | 20 | 8 | 51 | 79 | 17 |
| total | 74 | 39 | 265 | 378 | 89 |

En huecos de 3 s o menos, por nivel real: controlado 7, incipiente 7, incontrolado 75.
De los 34 s posteriores a un corte, 30 tienen ademas trayectorias cortas.

## 2 y 3. Referencias sin modelo temporal y acuerdo con la IA

- Vector por segundo de escena: maximo y media de cada uno de los 13 descriptores sobre las muestras de las trayectorias de 1 s o mas vivas en ese segundo, y numero de trayectorias (27 valores). Descriptores sin normalizar; la regresion logistica los estandariza.
- Entrenamiento: 15680 s con trayectoria (controlado 3345, incipiente 2075, incontrolado 10260), incluidos los segundos con trayectoria de los videos sin fuego como controlado. Pesos por clase balanceados. Cada segundo pesa igual (no hay muestreo por video).
- Validacion: los mismos 2757 s de escena de los modelos temporales; los 378 s sin trayectoria se cuentan como controlado, igual que en ellos.
- Kappa ponderado lineal por segundo contra las etiquetas de la IA. Referencia: humano contra IA 0.426 (otros videos: los 29 de la prueba de cambios; no es el mismo conjunto).

| estimador | F1 macro escena | exactitud escena | kappa lineal | F1 controlado | F1 incipiente | F1 incontrolado |
|---|---|---|---|---|---|---|
| regresion logistica | 0.526 | 0.563 | 0.396 | 0.570 | 0.332 | 0.676 |
| gradient boosting | 0.590 | 0.634 | 0.429 | 0.602 | 0.440 | 0.727 |
| CFC | 0.513 | 0.610 | 0.461 | 0.637 | 0.169 | 0.732 |
| GRU | 0.516 | 0.615 | 0.477 | 0.643 | 0.161 | 0.744 |
| LSTM | 0.526 | 0.567 | 0.419 | 0.591 | 0.303 | 0.683 |
| clase mayoritaria de entrenamiento | 0.227 | 0.515 | 0.000 | 0.000 | 0.000 | 0.680 |

Solo en los 2379 s de validacion con trayectoria (sin la regla de controlado por defecto):

| estimador | F1 macro | exactitud | kappa lineal |
|---|---|---|---|
| regresion logistica | 0.582 | 0.621 | 0.532 |
| gradient boosting | 0.649 | 0.703 | 0.574 |

Matriz de confusion por segundo de escena, regresion logistica (filas: etiqueta de la IA; columnas: referencia):

| real / estimado | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 608 | 393 | 32 |
| incipiente | 95 | 193 | 15 |
| incontrolado | 397 | 274 | 750 |

Matriz de confusion por segundo de escena, gradient boosting (filas: etiqueta de la IA; columnas: referencia):

| real / estimado | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 653 | 247 | 133 |
| incipiente | 82 | 186 | 35 |
| incontrolado | 403 | 110 | 908 |

## 4. Trayectorias vivas por segundo en entrenamiento

- Segundos de escena de los videos de fuego de entrenamiento, dentro del tramo util, con al menos una trayectoria de 1 s o mas. El area de una trayectoria en un segundo es la media de d2 en ese segundo.

| nivel | segundos | mediana de trayectorias vivas | percentil 90 | segundos con mas de una trayectoria | trayectorias-segundo | con area menor a un cuarto de la mayor |
|---|---|---|---|---|---|---|
| controlado | 3322 | 1 | 4 | 49.4 % | 6493 | 25.5 % |
| incipiente | 2075 | 1 | 3 | 46.3 % | 3761 | 18.0 % |
| incontrolado | 10260 | 2 | 6 | 65.0 % | 29916 | 41.5 % |
