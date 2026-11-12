# Vector de escena por fotograma

- Generado por `python escena.py`. Una secuencia por video y tramo (se reinicia en cada corte de escena), a 10 fps, con todos los fotogramas procesados, tengan o no trayectorias.
- Vector de 34 valores: maximo y media de los 13 descriptores sobre las trayectorias de fuego vivas (tambien las de menos de 1 s), numero de trayectorias, numero con edad de al menos 1 s, edad de la mas antigua, numero de cajas y area total de fuego y de humo, y la bandera `sin_trayectoria`.
- Las marcas de tiempo de los fotogramas sin trayectoria se interpolan con el indice del fotograma (en los que si tienen, la marca real difiere de 0.1 k en 0.03 s como maximo).
- Detecciones del fotograma: de las curvas guardadas (YOLO a 5 fps, confianza 0.25); cada fotograma toma la ultima medida anterior. Videos sin curva: 0.
- Contexto de camara en los fotogramas sin trayectoria: no se guardo; se usa la identidad y la ultima proporcion de inliers conocida.
- Normalizacion ajustada solo con entrenamiento: 189291 fotogramas de 337 videos.
- Tramos de menos de 10 fotogramas descartados: 26. Fotogramas cuyo tramo calculado con los cortes no coincide con el guardado: 0.

| conjunto | videos | secuencias | fotogramas | dentro del tramo util | controlado | incipiente | incontrolado | sin trayectoria | sin trayectoria de 1 s o mas | con tau por cambio | ventanas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entrenamiento | 337 | 460 | 189291 | 184111 | 51252 | 22555 | 110304 | 34488 | 39026 | 40674 | 1267 |
| validacion | 74 | 110 | 30301 | 29738 | 12498 | 3041 | 14199 | 6496 | 8175 | 5634 | 208 |
| calibracion | 71 | 82 | 48375 | 47460 | 11651 | 10565 | 25244 | 7092 | 8030 | 20991 | 311 |
| prueba | 100 | 140 | 51075 | 49544 | 17130 | 14950 | 17464 | 9104 | 9923 | 19966 | 328 |
