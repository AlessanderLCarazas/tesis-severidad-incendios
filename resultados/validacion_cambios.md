# Validacion humana de los cambios de nivel (paso 8)

- Lista: `resultados/prueba_cambios.csv` (30 videos). Anotador humano: `aleCa_cambios`, a ciegas con `etiquetar_v3.py --ciego`.
- Etiquetas de la IA: `resultados/etiquetas_ia_v2.csv`.
- Descartados por el humano: 1. Descartados por la IA: 0. Comparables en cambios, instantes y tramo util: 29.
- Este archivo lo genera `python validar_cambios.py`. No modifica ninguna etiqueta.
- Error de un instante = instante de la IA menos instante del humano, en segundos. Negativo: la IA marca el cambio antes que el humano.

## 1. Nivel inicial

**Los 30 videos:** acuerdo 22 de 30 (73.3 %), kappa ponderado lineal 0.469.

| humano / IA | controlado | incipiente | incontrolado | total |
|---|---|---|---|---|
| controlado | 7 | 1 | 0 | 8 |
| incipiente | 6 | 15 | 1 | 22 |
| incontrolado | 0 | 0 | 0 | 0 |
| total | 13 | 16 | 1 | 30 |

**Sin los descartados (29 videos):** acuerdo 21 de 29 (72.4 %), kappa ponderado lineal 0.460.

| humano / IA | controlado | incipiente | incontrolado | total |
|---|---|---|---|---|
| controlado | 7 | 1 | 0 | 8 |
| incipiente | 6 | 14 | 1 | 21 |
| incontrolado | 0 | 0 | 0 | 0 |
| total | 13 | 15 | 1 | 29 |

Videos con distinto nivel inicial:

| video | humano | IA | motivo de la IA |
|---|---|---|---|
| videoplayback1.mp4 | controlado | incipiente | Incendio de vehiculo que se extiende a camion; humo denso y llamas grandes desde la foto 11. |
| Fire Investigation 2A Burn Cell 1 - Toaster Pop Tart Fire.mp4 | incipiente | controlado | Llama en tostadora que sube a cortinas y pared, luego la habitacion arde con humo negro. |
| Brennender Christbaum _ Weihnachtsbaum → so schnell kann gehen.mp4 | incipiente | controlado | Bengalas pequenas en el arbol, luego fuego en la base y explosion de llamas que llena toda la sala. |
| 700_F_1193365801_Vq9ZwN74WallFA9y1947785GEEYilGqp_ST.mp4 | incipiente | incontrolado | Vegetacion en llamas de noche con humo denso que ocupa gran parte de la escena; silueta de surfista. |
| videoplayback (3).mp4 | incipiente | controlado | Ensayo de fuego sobre pila de madera: llama baja al inicio y luego llamas altas sostenidas de un solo objeto. |
| Launch of Fire Investigations by BRE and FIUK.mp4 | incipiente | controlado | Llama pequena sobre una tostadora crece a la cocina y luego a toda la sala con humo denso. |
| Har du en fungerande brandvarnare_.mp4 | incipiente | controlado | Lampara de aceite estable al inicio; tras cortes las cortinas arden y el fuego pasa a la mesa y cocina, objetos limitados. |
| 13871980_3840_2160_24fps.mp4 | incipiente | controlado | Quema en un campo que crece de forma sostenida hasta llamas enormes sobre el rastrojo y humo. |

## 2. Deteccion de cambios

Sobre los 29 videos no descartados.

| cambio | ambos | solo la IA | solo el humano | ninguno |
|---|---|---|---|---|
| a incipiente | 3 | 7 | 1 | 18 |
| a incontrolado | 15 | 5 | 1 | 8 |
| algun cambio en el video | 17 | 7 | 0 | 5 |

- De los 7 cambios a incipiente que marca solo la IA, en 6 el humano puso un nivel inicial de incipiente o mas (para el humano el video ya empieza en ese nivel) y en 1 el humano dejo el video en controlado.
- De los 24 videos no descartados con cambio segun la IA, el humano marca algun cambio en 17 y ninguno en 7.
- **Videos sin cambio segun la IA (5):** el humano marca cambio en 0 y coincide en el nivel inicial en 4.

| video sin cambio segun la IA | nivel IA | nivel humano | cambios del humano |
|---|---|---|---|
| 2301425156.mp4 | controlado | controlado | sin cambio |
| 2354872182.mp4 | controlado | controlado | sin cambio |
| 700_F_199061911_W4h9LVdrne9VAzuLYrxuZAA68JuuaXHw_ST.mp4 | controlado | controlado | sin cambio |
| 700_F_1193365801_Vq9ZwN74WallFA9y1947785GEEYilGqp_ST.mp4 | incontrolado | incipiente | sin cambio |
| Box Alarm Structure Fire Bradley Beach 3_19_23.mp4 | incipiente | incipiente | sin cambio |

Cambio a incipiente marcado por solo la IA:

| video | humano | IA |
|---|---|---|
| Fire Investigation 2A Burn Cell 1 - Toaster Pop Tart Fire.mp4 | incipiente; a incontrolado en 114.97 s | controlado; a incipiente en 2.00 s, a incontrolado en 66.57 s |
| 700_F_1633820181_qwiNmqTW57MMB3OTqVnQuTKS6YMSBRHG_ST.mp4 | controlado; sin cambio | controlado; a incipiente en 25.63 s |
| Brennender Christbaum _ Weihnachtsbaum → so schnell kann gehen.mp4 | incipiente; a incontrolado en 94.07 s | controlado; a incipiente en 70.00 s, a incontrolado en 73.90 s |
| videoplayback (3).mp4 | incipiente; sin cambio | controlado; a incipiente en 34.00 s |
| Launch of Fire Investigations by BRE and FIUK.mp4 | incipiente; a incontrolado en 198.83 s | controlado; a incipiente en 27.10 s, a incontrolado en 63.93 s |
| Har du en fungerande brandvarnare_.mp4 | incipiente; a incontrolado en 20.53 s | controlado; a incipiente en 16.33 s |
| 13871980_3840_2160_24fps.mp4 | incipiente; sin cambio | controlado; a incipiente en 5.13 s, a incontrolado en 12.68 s |

Cambio a incipiente marcado por solo el humano:

| video | humano | IA |
|---|---|---|
| videoplayback1.mp4 | controlado; a incipiente en 24.97 s, a incontrolado en 73.43 s | incipiente; a incontrolado en 49.60 s |

Cambio a incontrolado marcado por solo la IA:

| video | humano | IA |
|---|---|---|
| Brandgefahren im Haushalt.mp4 | incipiente; sin cambio | incipiente; a incontrolado en 24.27 s |
| 700_F_124207480_7UfkXKXrfnslHPXWEvBvDbahs5IJRWjO_ST.mp4 | incipiente; sin cambio | incipiente; a incontrolado en 2.77 s |
| 14297190_3840_2160_24fps.mp4 | incipiente; sin cambio | incipiente; a incontrolado en 19.10 s |
| VID_20230612_142839.mp4 | incipiente; sin cambio | incipiente; a incontrolado en 20.90 s |
| 13871980_3840_2160_24fps.mp4 | incipiente; sin cambio | controlado; a incipiente en 5.13 s, a incontrolado en 12.68 s |

Cambio a incontrolado marcado por solo el humano:

| video | humano | IA |
|---|---|---|
| Har du en fungerande brandvarnare_.mp4 | incipiente; a incontrolado en 20.53 s | controlado; a incipiente en 16.33 s |

Descartados por el humano (fuera de las cuentas de cambios, instantes y tramo util):

| video | IA |
|---|---|
| Brennende Akkus.mp4 | incipiente; a incontrolado en 5.67 s |

## 3. Error en segundos cuando ambos marcan el mismo cambio

| grupo | instantes | mediana del error absoluto (s) | media del error absoluto (s) | mediana del error con signo (s) | dentro de 1 s | dentro de 2 s | dentro de 5 s | la IA marca antes |
|---|---|---|---|---|---|---|---|---|
| todos | 18 | 31.82 | 54.23 | -31.82 | 2 | 3 | 4 | 15 |
| a incipiente | 3 | 3.60 | 81.53 | -3.60 | 1 | 1 | 2 | 2 |
| a incontrolado | 15 | 39.80 | 48.77 | -39.80 | 1 | 2 | 2 | 13 |
| origen: pasada 2 (tira densa) | 11 | 39.80 | 59.99 | -39.80 | 1 | 1 | 2 | 10 |
| origen: pasada 1 (fotos separadas 1 s o menos) | 3 | 1.44 | 2.42 | +1.44 | 1 | 2 | 2 | 1 |
| origen: pasada 1 (la pasada 2 no ve el cambio) | 4 | 61.85 | 77.26 | -61.85 | 0 | 0 | 0 | 4 |

- Duracion de los videos con origen pasada 2 (tira densa): mediana 152 s (de 53 a 364 s).
- Duracion de los videos con origen pasada 1 (fotos separadas 1 s o menos): mediana 13 s (de 12 a 36 s).
- Duracion de los videos con origen pasada 1 (la pasada 2 no ve el cambio): mediana 167 s (de 144 a 260 s).

| video | cambio | humano (s) | IA (s) | error (s) | origen del instante de la IA | separacion de fotos en la pasada 1 (s) | duracion (s) |
|---|---|---|---|---|---|---|---|
| Die Provinzial klärt auf – Brandursache Fettbrand.mp4 | a incipiente | 27.77 | 28.07 | +0.30 | pasada2 | 2.74 | 129.0 |
| What Type Of Smoke Alarm Is Best For Your Home_.mp4 | a incontrolado | 4.33 | 3.67 | -0.66 | pasada1 | 0.24 | 11.6 |
| 700_F_132088641_URj7fxf00kOATXV1mNgjeBrQCXsMgfw1_ST.mp4 | a incontrolado | 9.20 | 10.64 | +1.44 | pasada1 | 0.28 | 13.2 |
| Weihnachtsbaumbrände - sind vermeidbar.mp4 | a incipiente | 61.77 | 58.17 | -3.60 | pasada2 | 3.88 | 182.5 |
| 1723634695.mp4 | a incontrolado | 27.06 | 32.23 | +5.17 | pasada1 | 0.77 | 36.1 |
| videoplayback (15).mp4 | a incontrolado | 113.53 | 100.73 | -12.80 | pasada1; pasada2 no ve el cambio | 3.05 | 143.5 |
| Brennender Christbaum _ Weihnachtsbaum → so schnell kann gehen.mp4 | a incontrolado | 94.07 | 73.90 | -20.17 | pasada2 | 2.94 | 138.0 |
| videoplayback (17).mp4 | a incontrolado | 33.87 | 10.33 | -23.54 | pasada2 | 1.13 | 53.0 |
| videoplayback1.mp4 | a incontrolado | 73.43 | 49.60 | -23.83 | pasada2 | 5.16 | 242.4 |
| Weihnachtsbaumbrände - sind vermeidbar.mp4 | a incontrolado | 118.90 | 79.10 | -39.80 | pasada2 | 3.88 | 182.5 |
| Fire Investigation 2A Burn Cell 1 - Toaster Pop Tart Fire.mp4 | a incontrolado | 114.97 | 66.57 | -48.40 | pasada2 | 3.23 | 152.1 |
| April 2016 - Fire Investigation 1B - Milpitas CA.mp4 | a incontrolado | 96.50 | 46.80 | -49.70 | pasada2 | 2.92 | 137.5 |
| videoplayback (12).mp4 | a incontrolado | 111.23 | 55.03 | -56.20 | pasada1; pasada2 no ve el cambio | 3.93 | 184.8 |
| Brannutvikling.mp4 | a incontrolado | 115.00 | 47.50 | -67.50 | pasada1; pasada2 no ve el cambio | 3.17 | 148.9 |
| Die Provinzial klärt auf – Brandursache Fettbrand.mp4 | a incontrolado | 107.00 | 32.03 | -74.97 | pasada2 | 2.74 | 129.0 |
| Launch of Fire Investigations by BRE and FIUK.mp4 | a incontrolado | 198.83 | 63.93 | -134.90 | pasada2 | 4.69 | 220.4 |
| videoplayback (1).mp4 | a incontrolado | 255.40 | 82.87 | -172.53 | pasada1; pasada2 no ve el cambio | 5.53 | 259.7 |
| Wooden Model Ship Battleship Jean Bart Burning And Sinking.mp4 | a incipiente | 263.77 | 23.07 | -240.70 | pasada2 | 7.74 | 363.9 |

## 4. Tramo util

Diferencia = instante del humano menos instante de la IA, en los videos no descartados en que ambos lo marcaron. La IA marca el tramo util con las fotos de la pasada 1, asi que su resolucion es la separacion entre fotos.

| extremo | videos | mediana de la diferencia absoluta (s) | media de la diferencia absoluta (s) | mediana con signo (s) | dentro de 1 s | dentro de 2 s | dentro de 5 s | dentro de una foto de la pasada 1 |
|---|---|---|---|---|---|---|---|---|
| inicio | 26 | 0.18 | 3.73 | +0.11 | 20 | 20 | 22 | 23 |
| fin | 25 | 2.10 | 1.99 | +0.00 | 10 | 12 | 24 | 16 |

- El humano dejo vacio el inicio util en 3 videos y el fin util en 4 (de 29); esos no entran en la tabla.

| video | inicio humano | inicio IA | diferencia | fin humano | fin IA | diferencia | duracion (s) |
|---|---|---|---|---|---|---|---|
| videoplayback1.mp4 | 0.1 | 0.0 | +0.10 | 242.33 | 242.33 | +0.00 | 242.4 |
| videoplayback (17).mp4 | 9.63 | 2.23 | +7.40 | 53.0 | 48.47 | +4.53 | 53.0 |
| 2301425156.mp4 | 0.16 | 0.0 | +0.16 | 25.88 | 25.88 | +0.00 | 25.9 |
| 2354872182.mp4 | 0.0 | 0.0 | +0.00 | 30.2 | 30.2 | +0.00 | 30.2 |
| videoplayback (1).mp4 | 5.37 | 5.5 | -0.13 | 259.67 | 254.13 | +5.54 | 259.7 |
| Fire Investigation 2A Burn Cell 1 - Toaster Pop Tart Fire.mp4 | 0.2 | 0.0 | +0.20 | 152.03 | 148.8 | +3.23 | 152.1 |
| 700_F_1633820181_qwiNmqTW57MMB3OTqVnQuTKS6YMSBRHG_ST.mp4 | - | 0.0 | - | - | 46.35 | - | 46.4 |
| videoplayback (15).mp4 | 1.17 | 33.57 | -32.40 | 141.1 | 140.43 | +0.67 | 143.5 |
| What Type Of Smoke Alarm Is Best For Your Home_.mp4 | 0.03 | 0.0 | +0.03 | 8.93 | 11.03 | -2.10 | 11.6 |
| April 2016 - Fire Investigation 1B - Milpitas CA.mp4 | 0.37 | 0.0 | +0.37 | 135.43 | 137.43 | -2.00 | 137.5 |
| videoplayback (12).mp4 | 0.4 | 3.9 | -3.50 | - | 184.8 | - | 184.8 |
| 700_F_199061911_W4h9LVdrne9VAzuLYrxuZAA68JuuaXHw_ST.mp4 | 0.17 | 0.0 | +0.17 | 40.61 | 40.61 | +0.00 | 40.6 |
| Brennender Christbaum _ Weihnachtsbaum → so schnell kann gehen.mp4 | 64.9 | 23.47 | +41.43 | 137.63 | 135.03 | +2.60 | 138.0 |
| Brandgefahren im Haushalt.mp4 | 0.2 | 0.0 | +0.20 | 65.93 | 68.47 | -2.54 | 68.5 |
| 700_F_1193365801_Vq9ZwN74WallFA9y1947785GEEYilGqp_ST.mp4 | - | 0.0 | - | - | 13.5 | - | 13.5 |
| Die Provinzial klärt auf – Brandursache Fettbrand.mp4 | 0.17 | 2.73 | -2.56 | 128.97 | 126.2 | +2.77 | 129.0 |
| Brannutvikling.mp4 | 0.27 | 0.0 | +0.27 | 147.9 | 145.7 | +2.20 | 148.9 |
| videoplayback (3).mp4 | 0.0 | 0.0 | +0.00 | 196.43 | 192.77 | +3.66 | 197.0 |
| 700_F_124207480_7UfkXKXrfnslHPXWEvBvDbahs5IJRWjO_ST.mp4 | 0.0 | 0.0 | +0.00 | 6.57 | 6.57 | +0.00 | 6.6 |
| Weihnachtsbaumbrände - sind vermeidbar.mp4 | 0.5 | 0.0 | +0.50 | 182.5 | 178.6 | +3.90 | 182.5 |
| Launch of Fire Investigations by BRE and FIUK.mp4 | - | 0.0 | - | 220.33 | 215.63 | +4.70 | 220.4 |
| 1723634695.mp4 | 0.1 | 0.0 | +0.10 | 36.07 | 36.07 | +0.00 | 36.1 |
| Box Alarm Structure Fire Bradley Beach 3_19_23.mp4 | 0.83 | 0.0 | +0.83 | 149.2 | 146.0 | +3.20 | 149.2 |
| 14297190_3840_2160_24fps.mp4 | 0.08 | 0.0 | +0.08 | 24.98 | 24.98 | +0.00 | 25.0 |
| Har du en fungerande brandvarnare_.mp4 | 0.13 | 0.0 | +0.13 | 27.33 | 28.47 | -1.14 | 28.5 |
| Wooden Model Ship Battleship Jean Bart Burning And Sinking.mp4 | 1.53 | 7.73 | -6.20 | 351.27 | 356.07 | -4.80 | 363.9 |
| VID_20230612_142839.mp4 | 0.2 | 0.0 | +0.20 | 37.77 | 37.83 | -0.06 | 37.9 |
| 13871980_3840_2160_24fps.mp4 | 0.0 | 0.0 | +0.00 | - | 18.64 | - | 18.7 |
| 700_F_132088641_URj7fxf00kOATXV1mNgjeBrQCXsMgfw1_ST.mp4 | 0.12 | 0.0 | +0.12 | 13.16 | 13.16 | +0.00 | 13.2 |

## 5. Tiempo de etiquetado

- Videos con registro: 30 de 30.
- Tiempo total con la ventana abierta: 215.8 min (12951 s).
- 2 videos pasan de 1000 s abiertos (3488 s, 8000 s); se toman como pausas con la ventana abierta.
- **Sin esos 2: 24.4 min en 28 videos** (mediana 20.6 s, media 52.2 s, minimo 4.8 s, maximo 382.7 s por video).
- Duracion de esos 28 videos: 51.7 min. Tiempo abierto entre duracion del video: mediana 0.29.
- **En 22 de 28 videos la ventana estuvo abierta menos tiempo que la duracion del video**, es decir, no se vio el video completo a velocidad normal (se avanzo con saltos o con la linea de tiempo).
- Videos con una sola reproduccion o ninguna: 18 de 30.
- Primer guardado: 2026-11-11 18:38:43. Ultimo guardado: 2026-11-11 22:08:13.

| video | duracion (s) | abierto (s) | abierto / duracion | reproducciones | cambios del humano |
|---|---|---|---|---|---|
| videoplayback1.mp4 | 242.4 | 382.7 | 1.58 | 6 | a incipiente en 24.97 s, a incontrolado en 73.43 s |
| videoplayback (17).mp4 | 53.0 | 3488.1 (pausa) | 65.78 | 4 | a incontrolado en 33.87 s |
| 2301425156.mp4 | 25.9 | 11.6 | 0.45 | 1 | sin cambio |
| 2354872182.mp4 | 30.2 | 8.7 | 0.29 | 1 | sin cambio |
| videoplayback (1).mp4 | 259.7 | 75.1 | 0.29 | 1 | a incontrolado en 255.40 s |
| Fire Investigation 2A Burn Cell 1 - Toaster Pop Tart Fire.mp4 | 152.1 | 95.3 | 0.63 | 2 | a incontrolado en 114.97 s |
| 700_F_1633820181_qwiNmqTW57MMB3OTqVnQuTKS6YMSBRHG_ST.mp4 | 46.4 | 5.2 | 0.11 | 1 | sin cambio |
| videoplayback (15).mp4 | 143.5 | 136.2 | 0.95 | 4 | a incontrolado en 113.53 s |
| What Type Of Smoke Alarm Is Best For Your Home_.mp4 | 11.6 | 33.8 | 2.92 | 2 | a incontrolado en 4.33 s |
| April 2016 - Fire Investigation 1B - Milpitas CA.mp4 | 137.5 | 22.0 | 0.16 | 1 | a incontrolado en 96.50 s |
| videoplayback (12).mp4 | 184.8 | 18.6 | 0.10 | 1 | a incontrolado en 111.23 s |
| 700_F_199061911_W4h9LVdrne9VAzuLYrxuZAA68JuuaXHw_ST.mp4 | 40.6 | 12.2 | 0.30 | 1 | sin cambio |
| Brennende Akkus.mp4 | 83.6 | 8.8 | 0.11 | 1 | sin cambio |
| Brennender Christbaum _ Weihnachtsbaum → so schnell kann gehen.mp4 | 138.0 | 28.9 | 0.21 | 1 | a incontrolado en 94.07 s |
| Brandgefahren im Haushalt.mp4 | 68.5 | 45.8 | 0.67 | 1 | sin cambio |
| 700_F_1193365801_Vq9ZwN74WallFA9y1947785GEEYilGqp_ST.mp4 | 13.5 | 4.8 | 0.35 | 1 | sin cambio |
| Die Provinzial klärt auf – Brandursache Fettbrand.mp4 | 129.0 | 36.9 | 0.29 | 3 | a incipiente en 27.77 s, a incontrolado en 107.00 s |
| Brannutvikling.mp4 | 148.9 | 15.9 | 0.11 | 1 | a incontrolado en 115.00 s |
| videoplayback (3).mp4 | 197.0 | 15.6 | 0.08 | 1 | sin cambio |
| 700_F_124207480_7UfkXKXrfnslHPXWEvBvDbahs5IJRWjO_ST.mp4 | 6.6 | 271.4 | 41.06 | 3 | sin cambio |
| Weihnachtsbaumbrände - sind vermeidbar.mp4 | 182.5 | 29.5 | 0.16 | 2 | a incipiente en 61.77 s, a incontrolado en 118.90 s |
| Launch of Fire Investigations by BRE and FIUK.mp4 | 220.4 | 17.6 | 0.08 | 2 | a incontrolado en 198.83 s |
| 1723634695.mp4 | 36.1 | 8000.3 (pausa) | 221.61 | 1 | a incontrolado en 27.06 s |
| Box Alarm Structure Fire Bradley Beach 3_19_23.mp4 | 149.2 | 19.2 | 0.13 | 1 | sin cambio |
| 14297190_3840_2160_24fps.mp4 | 25.0 | 9.8 | 0.39 | 1 | sin cambio |
| Har du en fungerande brandvarnare_.mp4 | 28.5 | 34.7 | 1.22 | 2 | a incontrolado en 20.53 s |
| Wooden Model Ship Battleship Jean Bart Burning And Sinking.mp4 | 363.9 | 65.6 | 0.18 | 1 | a incipiente en 263.77 s |
| VID_20230612_142839.mp4 | 37.9 | 6.7 | 0.18 | 1 | sin cambio |
| 13871980_3840_2160_24fps.mp4 | 18.7 | 34.7 | 1.86 | 6 | sin cambio |
| 700_F_132088641_URj7fxf00kOATXV1mNgjeBrQCXsMgfw1_ST.mp4 | 13.2 | 14.8 | 1.12 | 3 | a incontrolado en 9.20 s |
