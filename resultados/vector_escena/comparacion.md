# Vector de escena: datos del pipeline (v3) contra el vector anterior

Generado por `python probar_vector_escena.py`. Videos: 5. Vector anterior: descriptores de la corrida de la noche y detecciones de `curvas/`. Vector nuevo: `descriptores_v3.py` y `vector_escena.VectorEscena`.

| video | fotogramas | filas de descriptores (anterior / nueva) | filas distintas | fotogramas con trayectoria | diferencia maxima de los 29 agregados | diferencia maxima de la marca de tiempo (s) | fotogramas sin trayectoria con camara distinta de la identidad |
|---|---|---|---|---|---|---|---|
| videoplayback (8).mp4 | 251 | 375 / 375 | 0 | 176 | 2.22e-16 | 0.0000 | 72 de 75 |
| 2451511630.mp4 | 73 | 80 / 80 | 0 | 61 | 5.55e-17 | 0.0000 | 11 de 12 |
| fire3.avi | 174 | 444 / 444 | 0 | 174 | 2.22e-16 | 0.0000 | 0 de 0 |
| 2030845629_clip_003.mp4 | 100 | 142 / 142 | 0 | 100 | 0.00e+00 | 0.0000 | 0 de 0 |
| 1293332816_clip_001.mp4 | 100 | 0 / 0 | 0 | 0 | 0.00e+00 | 0.0200 | 83 de 100 |

**Agregados de descriptores, numero de trayectorias, edad y bandera sin_trayectoria: coinciden en los 5 videos** (tolerancia 1e-6).

Detecciones por fotograma: anterior (`curvas/`, 5 fps, confianza 0.25, se mantiene la ultima medida) contra las del propio seguimiento (10 fps, confianza 0.1) y contra las mismas filtradas a confianza 0.25.

| valor | media anterior | media nueva (conf 0.1) | diferencia absoluta media | fotogramas distintos | media nueva a conf 0.25 | diferencia absoluta media a conf 0.25 | fotogramas distintos a conf 0.25 |
|---|---|---|---|---|---|---|---|
| n_det_fuego | 1.6117 | 2.8009 | 1.2063 | 49.0 % | 1.6347 | 0.1375 | 11.7 % |
| n_det_humo | 0.0774 | 0.1476 | 0.0759 | 6.7 % | 0.0788 | 0.0100 | 1.0 % |
| area_det_fuego | 0.0507 | 0.0772 | 0.0298 | 39.5 % | 0.0513 | 0.0053 | 13.9 % |
| area_det_humo | 0.0342 | 0.0398 | 0.0063 | 9.0 % | 0.0338 | 0.0027 | 3.6 % |

Un fotograma cuenta como distinto si el numero de cajas cambia o si el area difiere en mas de 0.005 del fotograma.
