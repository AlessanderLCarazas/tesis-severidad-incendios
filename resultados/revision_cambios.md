# Revision de las diferencias entre el humano y la IA (prueba de cambios)

- Lista: `resultados/prueba_cambios.csv`. Anotador humano: `aleCa_cambios`. Etiquetas de la IA: `resultados/etiquetas_ia_v2.csv`.
- Videos comparados: 29 (descartados: 1). Con alguna diferencia: 23.
- Diferencia: nivel inicial distinto, un cambio que marca solo uno, o un instante con mas de 5 s de diferencia.
- Orden: primero los videos con un instante que difiere mas del umbral, de la mayor diferencia a la menor; luego los que tienen un cambio marcado por uno solo; luego los que solo difieren en el nivel inicial; al final los que coinciden.
- Diferencia de un instante = instante de la IA menos instante del humano, en segundos.
- Hojas de comparacion (no versionadas): `runs_fuego/revision_cambios/`. Este archivo lo genera `python revisar_cambios.py`. No modifica ninguna etiqueta.

| n | video | inicial humano | inicial IA | a incipiente humano (s) | a incipiente IA (s) | a incontrolado humano (s) | a incontrolado IA (s) | mayor diferencia (s) | diferencias |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Wooden Model Ship Battleship Jean Bart Burning And Sinking.mp4 | controlado | controlado | 263.77 | 23.07 | - | - | 240.70 | a incipiente: -240.70 s |
| 2 | videoplayback (1).mp4 | incipiente | incipiente | - | - | 255.40 | 82.87 | 172.53 | a incontrolado: -172.53 s |
| 3 | Launch of Fire Investigations by BRE and FIUK.mp4 | incipiente | controlado | - | 27.10 | 198.83 | 63.93 | 134.90 | nivel inicial; a incipiente: solo la IA; a incontrolado: -134.90 s |
| 4 | Die Provinzial klärt auf – Brandursache Fettbrand.mp4 | controlado | controlado | 27.77 | 28.07 | 107.00 | 32.03 | 74.97 | a incontrolado: -74.97 s |
| 5 | Brannutvikling.mp4 | incipiente | incipiente | - | - | 115.00 | 47.50 | 67.50 | a incontrolado: -67.50 s |
| 6 | videoplayback (12).mp4 | incipiente | incipiente | - | - | 111.23 | 55.03 | 56.20 | a incontrolado: -56.20 s |
| 7 | April 2016 - Fire Investigation 1B - Milpitas CA.mp4 | incipiente | incipiente | - | - | 96.50 | 46.80 | 49.70 | a incontrolado: -49.70 s |
| 8 | Fire Investigation 2A Burn Cell 1 - Toaster Pop Tart Fire.mp4 | incipiente | controlado | - | 2.00 | 114.97 | 66.57 | 48.40 | nivel inicial; a incipiente: solo la IA; a incontrolado: -48.40 s |
| 9 | Weihnachtsbaumbrände - sind vermeidbar.mp4 | controlado | controlado | 61.77 | 58.17 | 118.90 | 79.10 | 39.80 | a incontrolado: -39.80 s |
| 10 | videoplayback1.mp4 | controlado | incipiente | 24.97 | - | 73.43 | 49.60 | 23.83 | nivel inicial; a incipiente: solo el humano; a incontrolado: -23.83 s |
| 11 | videoplayback (17).mp4 | incipiente | incipiente | - | - | 33.87 | 10.33 | 23.54 | a incontrolado: -23.54 s |
| 12 | Brennender Christbaum _ Weihnachtsbaum → so schnell kann gehen.mp4 | incipiente | controlado | - | 70.00 | 94.07 | 73.90 | 20.17 | nivel inicial; a incipiente: solo la IA; a incontrolado: -20.17 s |
| 13 | videoplayback (15).mp4 | incipiente | incipiente | - | - | 113.53 | 100.73 | 12.80 | a incontrolado: -12.80 s |
| 14 | 1723634695.mp4 | incipiente | incipiente | - | - | 27.06 | 32.23 | 5.17 | a incontrolado: +5.17 s |
| 15 | Har du en fungerande brandvarnare_.mp4 | incipiente | controlado | - | 16.33 | 20.53 | - | - | nivel inicial; a incipiente: solo la IA; a incontrolado: solo el humano |
| 16 | 13871980_3840_2160_24fps.mp4 | incipiente | controlado | - | 5.13 | - | 12.68 | - | nivel inicial; a incipiente: solo la IA; a incontrolado: solo la IA |
| 17 | 700_F_1633820181_qwiNmqTW57MMB3OTqVnQuTKS6YMSBRHG_ST.mp4 | controlado | controlado | - | 25.63 | - | - | - | a incipiente: solo la IA |
| 18 | Brandgefahren im Haushalt.mp4 | incipiente | incipiente | - | - | - | 24.27 | - | a incontrolado: solo la IA |
| 19 | videoplayback (3).mp4 | incipiente | controlado | - | 34.00 | - | - | - | nivel inicial; a incipiente: solo la IA |
| 20 | 700_F_124207480_7UfkXKXrfnslHPXWEvBvDbahs5IJRWjO_ST.mp4 | incipiente | incipiente | - | - | - | 2.77 | - | a incontrolado: solo la IA |
| 21 | 14297190_3840_2160_24fps.mp4 | incipiente | incipiente | - | - | - | 19.10 | - | a incontrolado: solo la IA |
| 22 | VID_20230612_142839.mp4 | incipiente | incipiente | - | - | - | 20.90 | - | a incontrolado: solo la IA |
| 23 | 700_F_1193365801_Vq9ZwN74WallFA9y1947785GEEYilGqp_ST.mp4 | incipiente | incontrolado | - | - | - | - | - | nivel inicial |
| 24 | 700_F_132088641_URj7fxf00kOATXV1mNgjeBrQCXsMgfw1_ST.mp4 | incipiente | incipiente | - | - | 9.20 | 10.64 | 1.44 | ninguno |
| 25 | What Type Of Smoke Alarm Is Best For Your Home_.mp4 | incipiente | incipiente | - | - | 4.33 | 3.67 | 0.66 | ninguno |
| 26 | 2301425156.mp4 | controlado | controlado | - | - | - | - | - | ninguno |
| 27 | 2354872182.mp4 | controlado | controlado | - | - | - | - | - | ninguno |
| 28 | 700_F_199061911_W4h9LVdrne9VAzuLYrxuZAA68JuuaXHw_ST.mp4 | controlado | controlado | - | - | - | - | - | ninguno |
| 29 | Box Alarm Structure Fire Bradley Beach 3_19_23.mp4 | incipiente | incipiente | - | - | - | - | - | ninguno |

Descartados (fuera de la comparacion): Brennende Akkus.mp4.
