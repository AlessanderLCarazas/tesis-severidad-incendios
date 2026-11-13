# Vector de escena por fotograma, version 2 (datos del propio pipeline)

- Generado por `python escena_v2.py` desde `runs_fuego/descriptores_v3`. El vector lo calcula `vector_escena.VectorEscena`, la misma clase que usa el sistema en vivo.
- Detecciones del fotograma: las del propio pipeline, filtradas a confianza 0.25.
- Normalizacion ajustada solo con entrenamiento: 189292 fotogramas de 337 videos.
- Tramos de menos de 10 fotogramas descartados: 26.

| conjunto | videos | secuencias | fotogramas | dentro del tramo util | controlado | incipiente | incontrolado | sin trayectoria | con tau por cambio | ventanas |
|---|---|---|---|---|---|---|---|---|---|---|
| entrenamiento | 337 | 460 | 189292 | 184113 | 51254 | 22555 | 110304 | 34489 | 40674 | 1267 |
| validacion | 74 | 110 | 30301 | 29738 | 12498 | 3041 | 14199 | 6496 | 5634 | 208 |
| calibracion | 71 | 82 | 48375 | 47460 | 11651 | 10565 | 25244 | 7092 | 20991 | 311 |
| prueba | 100 | 140 | 51076 | 49545 | 17131 | 14950 | 17464 | 9105 | 19966 | 328 |
