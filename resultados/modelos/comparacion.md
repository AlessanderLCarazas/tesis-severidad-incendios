# Modelos temporales en validacion (fase D)

Generado por `python figura_modelos.py`. Metricas de la mejor epoca de cada modelo en el conjunto de validacion; la prueba no se usa.

| modelo | parametros | s por epoca (entrenar) | s por epoca (validar) | epocas corridas | mejor epoca | F1 macro escena | exactitud escena | F1 macro muestra | exactitud muestra | MAE tau (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFC | 43298 | 96.1 | 1.1 | 18 | 8 | 0.513 | 0.610 | 0.546 | 0.657 | 49.9 |
| GRU | 15650 | 3.0 | 0.2 | 16 | 6 | 0.516 | 0.615 | 0.539 | 0.667 | 32.6 |
| LSTM | 20770 | 3.0 | 0.2 | 26 | 16 | 0.526 | 0.567 | 0.535 | 0.591 | 36.6 |

| modelo | F1 escena controlado | F1 escena incipiente | F1 escena incontrolado | F1 muestra controlado | F1 muestra incipiente | F1 muestra incontrolado |
|---|---|---|---|---|---|---|
| CFC | 0.637 | 0.169 | 0.732 | 0.718 | 0.178 | 0.742 |
| GRU | 0.643 | 0.161 | 0.744 | 0.710 | 0.111 | 0.796 |
| LSTM | 0.591 | 0.303 | 0.683 | 0.680 | 0.202 | 0.722 |

Matriz de confusion por segundo de escena, CFC (filas: etiqueta de la IA; columnas: modelo; 2757 s):

| real / modelo | controlado | incipiente | incontrolado | total |
|---|---|---|---|---|
| controlado | 746 | 226 | 61 | 1033 |
| incipiente | 214 | 68 | 21 | 303 |
| incontrolado | 348 | 206 | 867 | 1421 |

Matriz de confusion por segundo de escena, GRU (filas: etiqueta de la IA; columnas: modelo; 2757 s):

| real / modelo | controlado | incipiente | incontrolado | total |
|---|---|---|---|---|
| controlado | 763 | 231 | 39 | 1033 |
| incipiente | 234 | 65 | 4 | 303 |
| incontrolado | 345 | 209 | 867 | 1421 |

Matriz de confusion por segundo de escena, LSTM (filas: etiqueta de la IA; columnas: modelo; 2757 s):

| real / modelo | controlado | incipiente | incontrolado | total |
|---|---|---|---|---|
| controlado | 608 | 361 | 64 | 1033 |
| incipiente | 116 | 181 | 6 | 303 |
| incontrolado | 300 | 348 | 773 | 1421 |

Frecuencia de niveles con el muestreo por video: controlado 0.567, incipiente 0.113, incontrolado 0.319. Pesos de la perdida de nivel: controlado 0.587, incipiente 2.944, incontrolado 1.044.
