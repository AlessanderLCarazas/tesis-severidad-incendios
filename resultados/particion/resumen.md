# Particion por origen (paso 10)

- Generado por `python particion.py` (semilla 42). Videos de la lista: 604 (389 de fuego y 215 sin fuego).
- Grupos de origen: 582. Videos con union manual (`uniones.csv`): 22.
- Etiquetas de la IA: `ia_v2` en 210 videos y `ia_v1` en 179 (los que no pasaron por la segunda fase).
- Fracciones: validacion 0.15, calibracion 0.15, prueba de los videos sin fuego 0.15; el sorteo es por grupo de origen dentro de cada tipo.
- Comprobado: ningun grupo de origen esta en dos conjuntos y ningun video del origen de la prueba queda en entrenamiento, validacion o calibracion.

| tipo (segun la IA) | entrenamiento | validacion | calibracion | prueba | fuera | total |
|---|---|---|---|---|---|---|
| sin_fuego | 128 | 28 | 27 | 32 | 0 | 215 |
| controlado | 116 | 25 | 25 | 30 | 1 | 197 |
| incipiente | 20 | 4 | 4 | 5 | 1 | 34 |
| incontrolado | 54 | 12 | 11 | 9 | 1 | 87 |
| controlado_a_incipiente | 0 | 0 | 0 | 5 | 0 | 5 |
| controlado_a_incontrolado | 3 | 1 | 1 | 6 | 0 | 11 |
| incipiente_a_incontrolado | 22 | 5 | 4 | 15 | 3 | 49 |
| descartado | 0 | 0 | 0 | 0 | 6 | 6 |
| total | 343 | 75 | 72 | 102 | 12 | 604 |

| conjunto | videos | de fuego | sin fuego | grupos de origen | horas | cuentan para el umbral conformal |
|---|---|---|---|---|---|---|
| entrenamiento | 343 | 215 | 128 | 340 | 5.26 | - |
| validacion | 75 | 47 | 28 | 63 | 0.84 | - |
| calibracion | 72 | 45 | 27 | 72 | 1.34 | 52 |
| prueba | 102 | 70 | 32 | 101 | 1.42 | - |

Videos fuera de todos los conjuntos:

| video | tipo | motivo |
|---|---|---|
| 14035541_2048_1080_30fps.mp4 | descartado | descartado por la ia |
| 14296658_3840_2160_24fps - Copy.mp4 | incontrolado | duplicado exacto (excluidos.csv) |
| 19539603-hd_1920_1080_30fps.mp4 | descartado | descartado por la ia |
| 2016 April - Fire Investigation 1B - Milpitas CA.mp4 | incipiente_a_incontrolado | mismo origen que un video de prueba |
| 240_F_1127423393_SkYvCwswz8CN5OGhVUABadvMxKqWBRg9_ST.mp4 | descartado | descartado por la ia |
| 240_F_1564726580_ud86crnXW8Ye7CsKMejMN9XEwAae9ygL_ST.mp4 | controlado | mismo origen que un video de prueba |
| 700_F_1623872802_4LtdWtd26rIt1HXjOpyqheHprGXjcJ6X_ST.mp4 | descartado | descartado por la ia |
| Die Hamburger Feuerkasse klärt auf – Brandgefahr E-Zigaretten.mp4 | incipiente | mismo origen que un video de prueba |
| Die Hamburger Feuerkasse klärt auf – Brandgefahr Hoverboards.mp4 | incipiente_a_incontrolado | mismo origen que un video de prueba |
| Living room fires with and without a fire sprinkler (Timecode).mp4 | descartado | descartado por la ia |
| March 2016 - Fire Investigation 1A - Milpitas CA.mp4 | incipiente_a_incontrolado | mismo origen que un video de prueba |
| videoplayback (5).mp4 | descartado | descartado por la ia |
