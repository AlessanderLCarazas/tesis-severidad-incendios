# Secuencias por trayectoria (paso 11)

- Generado por `python secuencias.py`. Descriptores: `runs_fuego/descriptores_v2_noche/descriptores.csv`. Metas: `resultados/particion/metas.csv`, fuente `ia`.
- Una secuencia por video, tramo y trayectoria de fuego. Se descartan las de menos de 10 muestras. Ventanas de 300 muestras con paso de 150.
- Riesgo continuo (paso 9): 0 en controlado o sin fuego, 0.5 en incipiente y 1 en incontrolado.
- Normalizacion ajustada solo con entrenamiento: 323877 filas de 214 videos.
- `tau_mascara`: 0 sin tau, 1 el video pasa a incontrolado, 2 el video empieza en incontrolado (tau 0).

| conjunto | videos | de fuego | sin fuego | videos con trayectorias | sin fuego con trayectorias | trayectorias | trayectorias cortas descartadas | muestras | ventanas |
|---|---|---|---|---|---|---|---|---|---|
| entrenamiento | 343 | 215 | 128 | 214 | 5 | 5135 | 6153 | 323877 | 5486 |
| validacion | 75 | 47 | 28 | 48 | 1 | 683 | 1438 | 35545 | 713 |
| calibracion | 72 | 45 | 27 | 45 | 1 | 792 | 1017 | 71179 | 927 |
| prueba | 102 | 70 | 32 | 68 | 0 | 1150 | 1176 | 76877 | 1207 |

Segundos de escena por nivel (linea de tiempo de cada video dentro del tramo util, segun la fuente principal):

| conjunto | controlado (s) | incipiente (s) | incontrolado (s) | videos que pasan a incontrolado |
|---|---|---|---|---|
| entrenamiento | 4134 | 2255 | 11026 | 25 |
| validacion | 1031 | 304 | 1420 | 6 |
| calibracion | 936 | 1056 | 2524 | 5 |
| prueba | 1462 | 1494 | 1745 | 21 |

Muestras de trayectoria por nivel (cada muestra son unos 0.1 s; una escena con varias trayectorias cuenta varias veces):

| conjunto | validas | controlado | incipiente | incontrolado | con tau por cambio | con tau 0 por empezar en incontrolado |
|---|---|---|---|---|---|---|
| entrenamiento | 316887 | 50338 | 29313 | 237236 | 73601 | 180289 |
| validacion | 35373 | 12988 | 3129 | 19256 | 5560 | 16295 |
| calibracion | 70625 | 11926 | 12865 | 45834 | 33530 | 22652 |
| prueba | 76065 | 15256 | 24573 | 36236 | 28727 | 13402 |
