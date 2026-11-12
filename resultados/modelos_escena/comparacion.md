# Modelos a nivel de escena en validacion

Generado por `python comparar_escena.py`. Mejor epoca de cada corrida en validacion; la prueba no se usa. Segundos de escena: 2757 (2379 con alguna trayectoria de 1 s o mas y 378 sin). Con varias semillas se da media +- desviacion.

| estimador | corridas | F1 macro | F1 con trayectoria | F1 sin trayectoria | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | MAE tau (s) | cambios de nivel por video |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CFC de escena | 1 | 0.493 | 0.512 | 0.363 | 0.698 | 0.029 | 0.752 | 0.679 | 0.499 | 43.6 | 4.0 |
| GRU de escena | 3 | 0.556 +- 0.019 | 0.579 +- 0.017 | 0.405 +- 0.042 | 0.725 +- 0.018 | 0.133 +- 0.033 | 0.809 +- 0.014 | 0.710 +- 0.015 | 0.582 +- 0.024 | 42.4 +- 1.4 | 4.5 |
| LSTM de escena | 3 | 0.526 +- 0.018 | 0.548 +- 0.013 | 0.379 +- 0.053 | 0.723 +- 0.024 | 0.069 +- 0.054 | 0.787 +- 0.035 | 0.704 +- 0.026 | 0.557 +- 0.050 | 44.0 +- 3.4 | 4.9 |
| regresion logistica (sin tiempo) | 1 | 0.480 | 0.508 | 0.287 | 0.465 | 0.285 | 0.690 | 0.513 | 0.355 | - | 12.8 |
| gradient boosting (sin tiempo) | 1 | 0.608 | 0.632 | 0.417 | 0.600 | 0.425 | 0.798 | 0.664 | 0.506 | - | 13.0 |
| clase mayoritaria | 1 | 0.227 | 0.218 | 0.275 | 0.000 | 0.000 | 0.680 | 0.515 | 0.000 | - | 0.0 |

Cambios de nivel por video segun la etiqueta: 0.13 (47 videos de fuego).

| estimador | exactitud con trayectoria | kappa con trayectoria | exactitud sin trayectoria | kappa sin trayectoria |
|---|---|---|---|---|
| CFC de escena | 0.704 | 0.541 | 0.526 | 0.269 |
| GRU de escena | 0.736 +- 0.013 | 0.628 +- 0.016 | 0.547 +- 0.045 | 0.326 +- 0.061 |
| LSTM de escena | 0.731 +- 0.017 | 0.605 +- 0.034 | 0.536 +- 0.096 | 0.299 +- 0.116 |
| regresion logistica (sin tiempo) | 0.540 | 0.426 | 0.344 | 0.131 |
| gradient boosting (sin tiempo) | 0.686 | 0.557 | 0.532 | 0.276 |
| clase mayoritaria | 0.486 | 0.000 | 0.701 | 0.000 |

| modelo | semilla | parametros | s por epoca | epocas corridas | mejor epoca | minutos en total | F1 macro | kappa |
|---|---|---|---|---|---|---|---|---|
| CFC | 42 | 46133 | 21.9 | 28 | 17 | 11.0 | 0.493 | 0.499 |
| GRU | 42 | 19829 | 0.8 | 60 | 50 | 0.9 | 0.552 | 0.568 |
| GRU | 43 | 19829 | 0.8 | 36 | 25 | 0.6 | 0.534 | 0.562 |
| GRU | 44 | 19829 | 0.8 | 48 | 37 | 0.8 | 0.581 | 0.615 |
| LSTM | 42 | 26293 | 0.8 | 27 | 16 | 0.4 | 0.529 | 0.503 |
| LSTM | 43 | 26293 | 0.7 | 19 | 8 | 0.3 | 0.503 | 0.543 |
| LSTM | 44 | 26293 | 0.7 | 27 | 16 | 0.4 | 0.547 | 0.624 |

Primera version (modelo por trayectoria, semilla 42), mismos segundos:

| modelo | F1 macro | exactitud | kappa lineal |
|---|---|---|---|
| CFC por trayectoria | 0.513 | 0.610 | 0.461 |
| GRU por trayectoria | 0.516 | 0.615 | 0.477 |
| LSTM por trayectoria | 0.526 | 0.567 | 0.419 |

Matriz de confusion por segundo de escena, CFC (suma de 1 corridas; filas: etiqueta de la IA; columnas: modelo):

| real / modelo | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 966 | 9 | 58 |
| incipiente | 279 | 5 | 19 |
| incontrolado | 491 | 28 | 902 |

Matriz de confusion por segundo de escena, GRU (suma de 3 corridas; filas: etiqueta de la IA; columnas: modelo):

| real / modelo | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 2765 | 206 | 128 |
| incipiente | 777 | 98 | 34 |
| incontrolado | 988 | 267 | 3008 |

Matriz de confusion por segundo de escena, LSTM (suma de 3 corridas; filas: etiqueta de la IA; columnas: modelo):

| real / modelo | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 2915 | 87 | 97 |
| incipiente | 825 | 44 | 40 |
| incontrolado | 1233 | 167 | 2863 |

Matriz de confusion por segundo de escena, regresion logistica (sin tiempo):

| real / estimado | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 435 | 555 | 43 |
| incipiente | 85 | 196 | 22 |
| incontrolado | 316 | 322 | 783 |

Matriz de confusion por segundo de escena, gradient boosting (sin tiempo):

| real / estimado | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 582 | 318 | 133 |
| incipiente | 75 | 201 | 27 |
| incontrolado | 249 | 123 | 1049 |
