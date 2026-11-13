# Modelos a nivel de escena en validacion

Generado por `python comparar_escena.py`. Mejor epoca de cada corrida en validacion; la prueba no se usa. Segundos de escena: 2757 (2379 con alguna trayectoria de 1 s o mas y 378 sin). Con varias semillas se da media +- desviacion.

| estimador | corridas | F1 macro | F1 con trayectoria | F1 sin trayectoria | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | MAE tau (s) | cambios de nivel por video |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CFC de escena | 3 | 0.544 +- 0.015 | 0.561 +- 0.007 | 0.402 +- 0.053 | 0.711 +- 0.051 | 0.170 +- 0.034 | 0.751 +- 0.009 | 0.644 +- 0.014 | 0.543 +- 0.040 | 37.5 +- 3.6 | 5.3 |
| GRU de escena | 3 | 0.621 +- 0.008 | 0.630 +- 0.016 | 0.558 +- 0.044 | 0.788 +- 0.015 | 0.208 +- 0.052 | 0.866 +- 0.018 | 0.753 +- 0.012 | 0.698 +- 0.022 | 27.9 +- 1.5 | 3.4 |
| LSTM de escena | 3 | 0.552 +- 0.048 | 0.560 +- 0.034 | 0.481 +- 0.132 | 0.738 +- 0.035 | 0.137 +- 0.061 | 0.782 +- 0.063 | 0.685 +- 0.042 | 0.583 +- 0.074 | 31.3 +- 1.4 | 4.8 |
| regresion logistica (sin tiempo) | 1 | 0.482 | 0.508 | 0.299 | 0.468 | 0.285 | 0.691 | 0.514 | 0.357 | - | 12.3 |
| gradient boosting (sin tiempo) | 1 | 0.622 | 0.645 | 0.432 | 0.623 | 0.444 | 0.799 | 0.679 | 0.519 | - | 12.8 |
| clase mayoritaria | 1 | 0.227 | 0.218 | 0.275 | 0.000 | 0.000 | 0.680 | 0.515 | 0.000 | - | 0.0 |

Cambios de nivel por video segun la etiqueta: 0.13 (47 videos de fuego).

| estimador | exactitud con trayectoria | kappa con trayectoria | exactitud sin trayectoria | kappa sin trayectoria |
|---|---|---|---|---|
| CFC de escena | 0.668 +- 0.009 | 0.582 +- 0.032 | 0.494 +- 0.054 | 0.254 +- 0.081 |
| GRU de escena | 0.763 +- 0.010 | 0.714 +- 0.020 | 0.693 +- 0.064 | 0.563 +- 0.073 |
| LSTM de escena | 0.696 +- 0.027 | 0.603 +- 0.054 | 0.616 +- 0.145 | 0.429 +- 0.174 |
| regresion logistica (sin tiempo) | 0.539 | 0.424 | 0.360 | 0.149 |
| gradient boosting (sin tiempo) | 0.697 | 0.562 | 0.569 | 0.313 |
| clase mayoritaria | 0.486 | 0.000 | 0.701 | 0.000 |

| modelo | semilla | parametros | s por epoca | epocas corridas | mejor epoca | minutos en total | F1 macro | kappa |
|---|---|---|---|---|---|---|---|---|
| CFC | 42 | 46133 | 27.8 | 60 | 26 | 29.7 | 0.530 | 0.488 |
| CFC | 43 | 46133 | 18.2 | 60 | 40 | 19.6 | 0.538 | 0.562 |
| CFC | 44 | 46133 | 18.2 | 60 | 60 | 19.6 | 0.564 | 0.580 |
| GRU | 42 | 19829 | 1.5 | 60 | 58 | 1.8 | 0.630 | 0.668 |
| GRU | 43 | 19829 | 1.1 | 60 | 57 | 1.3 | 0.610 | 0.720 |
| GRU | 44 | 19829 | 1.2 | 60 | 24 | 1.4 | 0.623 | 0.706 |
| LSTM | 42 | 26293 | 1.5 | 60 | 21 | 1.8 | 0.495 | 0.487 |
| LSTM | 43 | 26293 | 1.2 | 60 | 47 | 1.4 | 0.549 | 0.598 |
| LSTM | 44 | 26293 | 1.1 | 60 | 48 | 1.3 | 0.613 | 0.665 |

Primera version (modelo por trayectoria, semilla 42), mismos segundos:

| modelo | F1 macro | exactitud | kappa lineal |
|---|---|---|---|
| CFC por trayectoria | 0.513 | 0.610 | 0.461 |
| GRU por trayectoria | 0.516 | 0.615 | 0.477 |
| LSTM por trayectoria | 0.526 | 0.567 | 0.419 |

Matriz de confusion por segundo de escena, CFC (suma de 3 corridas; filas: etiqueta de la IA; columnas: modelo):

| real / modelo | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 2394 | 477 | 228 |
| incipiente | 671 | 226 | 12 |
| incontrolado | 566 | 987 | 2710 |

Matriz de confusion por segundo de escena, GRU (suma de 3 corridas; filas: etiqueta de la IA; columnas: modelo):

| real / modelo | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 2668 | 341 | 90 |
| incipiente | 663 | 218 | 28 |
| incontrolado | 339 | 579 | 3345 |

Matriz de confusion por segundo de escena, LSTM (suma de 3 corridas; filas: etiqueta de la IA; columnas: modelo):

| real / modelo | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 2671 | 301 | 127 |
| incipiente | 738 | 141 | 30 |
| incontrolado | 745 | 661 | 2857 |

Matriz de confusion por segundo de escena, regresion logistica (sin tiempo):

| real / estimado | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 438 | 553 | 42 |
| incipiente | 85 | 196 | 22 |
| incontrolado | 314 | 323 | 784 |

Matriz de confusion por segundo de escena, gradient boosting (sin tiempo):

| real / estimado | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 609 | 280 | 144 |
| incipiente | 70 | 199 | 34 |
| incontrolado | 243 | 114 | 1064 |
