# Velocidad del sistema completo en vivo

Generado por `python sistema.py --modo fps`. 5 videos de prueba elegidos al azar (semilla 42). El sistema procesa fotograma a fotograma: lectura con muestreo a 10 fps, YOLOv8s (GPU), movimiento de camara y cortes, ByteTrack, descriptores, `VectorEscena`, modelo temporal (CPU, un paso por fotograma) y decision.

- fps = fotogramas procesados por segundo de computo. El sistema muestrea a 10 fps, asi que funciona en tiempo real si supera 10.
- Tiempo real = segundos de video por segundo de computo.
- La lectura incluye decodificar y descartar los fotogramas que el muestreo omite; por eso depende de la resolucion y de los fps de la fuente.

| video | resolucion | fps de la fuente | duracion (s) | fotogramas | fps con gru_s42 | tiempo real con gru_s42 | fps con cfc_s42 | tiempo real con cfc_s42 |
|---|---|---|---|---|---|---|---|---|
| 1. Brandgefahren im Haushalt.mp4 | 1920x1080 | 30.0 | 68.4 | 685 | 17.4 | 1.74x | 17.4 | 1.74x |
| 2. 14035491_2048_1080_30fps.mp4 | 2048x1080 | 29.97 | 59.79 | 599 | 11.1 | 1.11x | 11.2 | 1.11x |
| 3. 12588006_1920_1080_30fps.mp4 | 1920x1080 | 30.0 | 33.7 | 338 | 13.9 | 1.39x | 13.9 | 1.39x |
| 4. fire19_clip_001.mp4 | 320x240 | 10.0 | 9.9 | 100 | 48.3 | 4.78x | 48.5 | 4.80x |
| 5. 1828453162.mp4 | 640x360 | 25.0 | 26.52 | 266 | 37.0 | 3.69x | 37.2 | 3.71x |
| **todos (ponderado por fotogramas)** | | | 198.3 | 1988 | **15.7** | 1.56x | **15.7** | 1.57x |

Tiempo por etapa (ms por fotograma procesado):

| video | lectura | YOLO | camara y cortes | seguimiento | descriptores | vector de escena | modelo gru_s42 | decision gru_s42 | modelo cfc_s42 | decision cfc_s42 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.31 | 20.77 | 30.67 | 0.78 | 0.20 | 0.18 | 0.63 | 0.01 | 0.46 | 0.01 |
| 2 | 6.01 | 35.50 | 45.70 | 1.11 | 0.45 | 0.26 | 0.66 | 0.01 | 0.48 | 0.01 |
| 3 | 5.41 | 25.04 | 38.96 | 1.13 | 0.46 | 0.28 | 0.65 | 0.01 | 0.47 | 0.01 |
| 4 | 0.29 | 12.93 | 6.20 | 0.61 | 0.03 | 0.13 | 0.50 | 0.01 | 0.42 | 0.01 |
| 5 | 0.52 | 11.94 | 12.02 | 1.38 | 0.28 | 0.30 | 0.59 | 0.01 | 0.45 | 0.01 |
| **media** | **4.30** | **24.36** | **32.88** | **1.01** | **0.32** | **0.24** | **0.63** | **0.01** | **0.46** | **0.01** |
