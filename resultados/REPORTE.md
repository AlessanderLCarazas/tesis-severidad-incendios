# Reporte de avances

## Paso 5 (meta del 50 %): preanotacion semiautomatica y herramienta de etiquetado v2

Fecha: 2026-11-04. Rama: `avance-50`.

### Que se hizo

- `preanotar.py` (nuevo; el borrador no existia en la carpeta): YOLOv8s (`fasdd_v8s_full`) a 5 fps sobre cada video, area de fuego y de humo como union de cajas en una grilla 64x36, curva por video en `curvas/<video>.csv` y propuesta de etiqueta en `propuestas.csv`.
- La medicion (YOLO) y las reglas estan separadas: `--solo_reglas` recalcula `propuestas.csv` desde `curvas/` en segundos, sin GPU.
- `etiquetar_v2.py` (nuevo; `etiquetar.py` no se modifico): muestra la tira, la propuesta y la curva de area; las propuestas vienen precargadas (el anotador acepta con `n` o corrige); ordena claras primero y dudosas al final; mismo formato de `etiquetas.csv` y mismas teclas, mas `p` (volver a la propuesta) y clic en la curva para marcar un instante exacto.

### Reglas de propuesta (valores finales)

| Elemento | Regla |
|---|---|
| Tramo util | presencia de fuego (mediana 1 s) sostenida >= 1 s; si el fuego sigue hasta el final, `t_fin_util` = duracion del video |
| Duracion del video | `min(duracion_s de la seleccion, ultima muestra + 1/fps)` |
| Linea base | `a0`, `h0` = mediana del area de fuego y de humo en los primeros 2 s del tramo util |
| Nivel inicial | Incontrolado si `a0 >= 0.25` o `h0 >= 0.4`; si no, Controlado (sin umbral de area para Incipiente: Controlado = llama en su fuente con tamano estable, no fuego pequeno) |
| `t_inc` | area >= max(2*a0, a0+0.02) sostenida 2 s |
| `t_unc` | area >= max(5*a0, 0.25) o humo >= 0.4, sostenida 2 s; si ocurre antes que `t_inc`, `t_inc = t_unc` (paso directo) |

Motivos de duda (una propuesta es `dudosa` si tiene al menos uno). `cv_area` y los cruces se calculan sobre una curva de estabilidad (mediana de 2 s del area cruda) para no reaccionar al parpadeo de las llamas:

| Motivo | Condicion | Parametro |
|---|---|---|
| `sin_fuego` | no hay presencia sostenida | `--presencia_min 1.0` |
| `conf_baja` | confianza media de fuego < 0.40 | `--conf_duda 0.40` |
| `intermitente` | fraccion de muestras con fuego < 0.7 | `--presencia_duda 0.7` |
| `util_corto` | tramo util < 5 s | `--util_min 5.0` |
| `inestable` | coeficiente de variacion del area > 0.7 | `--cv_duda 0.7`, `--ventana_estabilidad 2.0` |
| `nivel_borde` | `a0` o `h0` a menos de x1.1 de su umbral | `--margen_borde 1.1` |
| `oscila` | mas de 3 cruces de umbral cuyo nuevo estado dura >= 2 s | `--cruces_max 3`, `--cruce_min 2.0` |
| `cambio_debil` | despues de un cambio propuesto el area no supera 1.1x el umbral | `--margen_borde 1.1` |

Los umbrales de duda se eligieron por barrido para que las dudosas queden entre 30 y 40 %, no se validaron contra etiquetas humanas. Al terminar el etiquetado conviene medir cuantas propuestas claras se aceptaron sin cambios.

### Comandos

```
python preanotar.py --limite 200
python preanotar.py --limite 200 --solo_reglas
python etiquetar_v2.py --anotador meral
```

### Numeros (200 primeros videos de `seleccion_etiquetado.csv`)

- Duracion total de video: 11 558 s (3.21 h). Tiempo de proceso de `preanotar.py`: 1 595 s (27 min), unos 0.14 s por segundo de video (mediana por video: 0.07 en resoluciones menores a 1080p, 0.13 en 1080p, 0.28 en 4K). Correccion del 2026-11-04: una version anterior de este reporte decia 9 401 s y 12 103 s porque la suma se hizo separando por comas y los nombres de video con comas desplazaban las columnas.
- Claridad: 126 claras (63 %), 74 dudosas (37 %). Con las reglas iniciales eran 106 dudosas (53 %).
- Motivos: inestable 32, conf_baja 20, intermitente 16, nivel_borde 13, oscila 9, cambio_debil 6, sin_fuego 6, util_corto 3.
- Nivel inicial propuesto: controlado 124, incontrolado 70, descartar 6. Con `t_inc`: 29; con `t_unc`: 7; con paso directo (`t_inc = t_unc`): 5.
- `t_fin_util` = duracion del video en 164 de 194 videos con tramo util.

### Archivos generados

- `propuestas.csv` (200 filas): `archivo, nivel_inicial, t_incipiente, t_incontrolado, t_inicio_util, t_fin_util, descartar, a0, h0, area_max, humo_max, conf_media, presencia, cv_area, cruces, claridad, motivos, duracion_s, muestras, seg_proceso`.
- `curvas/<video>.csv` (200 archivos, no versionados): `t, fuego, humo, fuego_suave, humo_suave, conf_fuego, conf_humo, n_fuego, n_humo`.

### Decisiones

- Se elimino el umbral de area 0.06 para Incipiente: con el, chimeneas filmadas de cerca salian como Incipiente.
- La duracion del video se toma acotada por lo que el decoder realmente lee (en un video el metadato decia 1.1 s mas).
- ~~Ninguna propuesta entra al dataset sin verificacion humana en `etiquetar_v2.py`.~~ **Reemplazada el 2026-11-04** por el protocolo de etiquetado con IA: entrenamiento con etiquetas de IA sin revision humana y 40 videos de prueba etiquetados por un humano a ciegas (ver "Etiquetado automatico con IA y lista de prueba humana").

## Etiquetado automatico con IA y lista de prueba humana

Fecha: 2026-11-04. Rama: `avance-50`. Reemplaza el etiquetado semiautomatico: las etiquetas rapidas previas (`aleCa`, 200 filas) quedaron en `etiquetas_rapidas_respaldo.csv` y no se usan.

### Metodo

- Modelo: Claude Sonnet (alias `sonnet` de los subagentes de Claude Code, Sonnet 5.5), con 10 subagentes nuevos sin contexto previo, un lote de 20 videos cada uno, en paralelo.
- Prompt fijo: `prompts/prompt_severidad.txt`. Mensaje identico para cada subagente, salvo el numero de lote: "La carpeta del proyecto es: <ruta>. Lee el archivo prompts\prompt_severidad.txt de esa carpeta y sigue exactamente sus instrucciones. Tu lote es prompts\lotes\lote_XX.csv."
- Entrada por video: solo su tira de 24 fotos (`tiras/`). El prompt prohibe abrir `propuestas.csv`, `curvas/`, `etiquetas*.csv` y ejecutar comandos.
- Salida cruda: `resultados/ia_crudo/<video>.json`, con numeros de foto. `python etiquetar_ia.py juntar` los convierte a segundos con `tiras/tiempos.csv` y escribe `resultados/etiquetas_ia.csv` (anotador `ia_sonnet`).
- Limitacion: con subagentes no se fija la temperatura; se guardan el prompt y las respuestas crudas.
- Costo aproximado: unos 98 000 tokens por subagente (unos 980 000 en total) y de 76 a 118 s por lote.

### Comandos

```
python etiquetar_ia.py preparar
python etiquetar_ia.py juntar
python etiquetar_ia.py prueba
python etiquetar_v2.py --anotador aleCa --lista resultados/prueba_humana.csv --ciego
```

### Numeros (200 videos)

- Respuestas: 200 de 200 validas, ninguna faltante ni invalida.
- Nivel inicial: controlado 138, incontrolado 45, incipiente 15, descartados 2 (`14035541_2048_1080_30fps.mp4`: solo humo ligero; `19539603-hd_1920_1080_30fps.mp4`: chispas de una sierra, sin fuego).
- Cambios de nivel: solo 2 videos con `t_incipiente` y 2 con `t_incontrolado` (las reglas de `preanotar.py` habian propuesto 29 con `t_incipiente`).
- Tramo util recortado (no empieza en la foto 1 o no termina en la 24): 19 videos.

### Lista de prueba humana

- `resultados/prueba_humana.csv`: 40 videos, solo la columna `archivo`, en orden aleatorio.
- Estratificada por el nivel maximo que alcanza el video segun la IA, con asignacion proporcional, minimo 5 por nivel y semilla 42: controlado 27, incipiente 5, incontrolado 8. Solo 1 de los 40 tiene un cambio de nivel segun la IA.
- Se excluyeron los 2 videos descartados por la IA y la copia exacta `14296658_3840_2160_24fps - Copy.mp4` (mismo MD5 que el original; en `seleccion_etiquetado.csv` su origen tenia un espacio al final y no se agrupo con el original). El original esta en la prueba, asi que la copia no debe usarse para entrenar.

## Comparacion humano contra IA en los 40 videos de prueba

Fecha: 2026-11-04. Comando: `python comparar_etiquetas.py` (salida completa en `resultados/comparacion_humano_ia.md`). Anotador humano: `aleCa`, en modo ciego (`etiquetar_v2.py --ciego`).

### Calidad de las etiquetas humanas

- 40 de 40 videos etiquetados, todos con nivel inicial y ninguno descartado.
- Tiempo entre guardados: minimo 1 s, mediana 4 s, media 6 s, maximo 49 s. 35 de 39 intervalos duran menos de 10 s; la sesion completa duro 3.8 min. Con la duracion de estos videos (mediana 33 s, minimo 26.6 s), esto indica que los niveles se asignaron mirando la tira y no reproduciendo los videos.
- No se marco tramo util en ningun video (se toma el video completo) ni ningun cambio de nivel (0 `t_incipiente`, 0 `t_incontrolado`).

### Acuerdo

| Medida | Nivel inicial | Nivel maximo del video |
|---|---|---|
| Acuerdo | 85.0 % (34/40) | 82.5 % (33/40) |
| Kappa ponderado lineal | 0.801 | 0.770 |
| Kappa ponderado cuadratico | 0.880 | 0.860 |
| Kappa sin ponderar | 0.704 | 0.662 |

Matriz de confusion del nivel inicial (filas: humano, columnas: IA):

| humano \ ia | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 25 | 0 | 0 |
| incipiente | 3 | 3 | 2 |
| incontrolado | 0 | 1 | 6 |

- Cambios de nivel: no se pueden comparar en segundos. El humano no marco ninguno y la IA marco uno (`12399227_3840_2160_60fps.mp4`, `t_incipiente` = 8.29 s).
- La muestra esta estratificada por el nivel de la IA, asi que el acuerdo y el kappa describen estos 40 videos y no son una estimacion sin sesgo para los 200.
- Los 6 desacuerdos estan en `resultados/comparacion_humano_ia.md`, con el motivo que dio la IA. En 3 de ellos el humano dijo incipiente y la IA controlado (fogatas); en los otros 3 la diferencia es entre incipiente e incontrolado en incendios de vegetacion.

## Paso 1 del plan del 50 %: preprocesamiento (lector con marcas de tiempo, dt y muestreo)

Fecha: 2026-11-04. Rama: `avance-50`. Script: `preprocesamiento.py` (clase `LectorVideo`, reutilizable por las etapas siguientes).

### Que hace

- Lee cada fotograma con `cap.grab()` y toma su marca de tiempo real de `cv2.CAP_PROP_POS_MSEC`. Solo decodifica la imagen (`cap.retrieve()`) de los fotogramas elegidos.
- Si la marca no es valida (0 o negativa despues del primer fotograma) usa `indice / fps`; si no es creciente la corrige a `t_anterior + 1/fps` y la cuenta en `corregidos`.
- Muestreo a ~10 fps sobre una rejilla de 100 ms anclada en el primer fotograma: se toma el primer fotograma con `t >= siguiente - tolerancia` (tolerancia = medio periodo de la fuente) y la rejilla avanza 100 ms. Se conserva la marca real del fotograma tomado, por eso `dt_k` no es constante. Si la fuente tiene 10 fps o menos, se toman todos los fotogramas.
- `dt_k = t_k - t_{k-1}` en segundos; para el primer fotograma (`k = 0`) se usa el paso nominal (0.1 s).
- `--quitar p` elimina al azar una fraccion `p` de los fotogramas muestreados (nunca el primero), con semilla `semilla + crc32(nombre del video)`, para las pruebas de robustez.
- Letterbox y normalizacion: los hace Ultralytics. Se verifico en el codigo instalado (8.4.142, `engine/predictor.py`): con `rect=False` (valor por defecto) el letterbox es cuadrado de 640x640, con relleno gris 114, y la imagen se divide entre 255.

### Comandos

```
python preprocesamiento.py
python preprocesamiento.py --quitar 0.5 --mosaico 0
```

### Numeros (14 videos de prueba: un video por fps frecuente, uno vertical 4K y uno sin fuego; lista en `resultados/preprocesamiento/videos_prueba.csv`)

| Video | fps fuente | fps efectivo | dt medio (ms) | dt desv. (ms) | dt min-max (ms) | lectura (fps) |
|---|---|---|---|---|---|---|
| 700_F_560569279_8ZejcDQzkVJ4 | 23.976 | 10.003 | 99.97 | 20.41 | 83.42-125.13 | 2844.2 |
| 14333044_2160_3840_24fps.mp4 | 24.0 | 9.992 | 100.08 | 20.43 | 83.33-125.0 | 89.6 |
| 5284fda1-b4c45d7c.mp4 | 25.0 | 9.99 | 100.1 | 20.0 | 80.0-120.0 | 4025.6 |
| 13695196_2160_3840_30fps.mp4 | 29.97 | 10.007 | 99.93 | 2.35 | 66.73-100.1 | 125.8 |
| 700_F_1144746294_DuYt5cvyX8H | 30.0 | 10.0 | 100.0 | 0.0 | 100.0-100.0 | 2928.9 |
| 700_F_87416564_E2wf7R9qR7vqz | 50.0 | 10.0 | 100.0 | 0.0 | 100.0-100.0 | 4227.2 |
| 12856131_1080_1920_60fps.mp4 | 59.94 | 9.998 | 100.02 | 1.18 | 83.42-100.1 | 460.7 |
| 700_F_329703997_6yh0UcoMTWJW | 60.0 | 10.0 | 100.0 | 0.0 | 100.0-100.0 | 3533.5 |
| 11998918_2160_3840_60fps.mp4 | 60.0 | 10.0 | 100.0 | 0.0 | 100.0-100.0 | 129.8 |
| 1293332816_clip_001.mp4 | 25.0 | 10.02 | 99.8 | 20.0 | 80.0-120.0 | 4323.3 |
| fire23_clip_001.mp4 | 7.0 | 7.0 | 142.86 | 0.0 | 142.86-142.86 | 773.9 |
| fire20_clip_001.mp4 | 9.0 | 9.0 | 111.11 | 0.0 | 111.11-111.11 | 4766.3 |
| fire16_clip_001.mp4 | 10.0 | 10.0 | 100.0 | 0.0 | 100.0-100.0 | 5206.0 |
| 1673111603_clip_001.mp4 | 29.0 | 10.003 | 99.97 | 10.39 | 68.97-103.45 | 5640.5 |

Con --quitar 0.5: fps efectivo medio 4.95, dt maximo entre 500 y 1293 ms, fotogramas quitados por video entre 37 y 117.
- Ninguna marca de tiempo tuvo que corregirse en estos 14 videos.
- La lectura de video 4K va a unos 90 a 130 fotogramas por segundo (decodificacion en CPU), frente a miles por segundo en resoluciones bajas: en 4K la decodificacion sera el cuello de botella del sistema completo.

### Archivos generados

- `resultados/preprocesamiento/marcas.csv` y `marcas_quitar50.csv`: `archivo, k, frame, t_k, dt_k`.
- `resultados/preprocesamiento/resumen.csv` y `resumen_quitar50.csv`: estadisticas por video.
- `resultados/preprocesamiento/mosaicos/`: los 8 primeros fotogramas muestreados de cada video despues del letterbox, con `k`, `frame`, `t_k` y `dt_k`. **No versionado desde el 2026-11-05** (contienen fotogramas de videos de terceros, 4 de ellos de Adobe Stock con marca de agua): se sacaron del repositorio y del historial y estan en `.gitignore`; se regeneran con `python preprocesamiento.py`.

### Decisiones

- `fps_objetivo = 10`, rejilla anclada en el primer fotograma, tolerancia de medio periodo de la fuente.
- `dt_0 = 0.1 s` (paso nominal).

## Curvas de los videos restantes (corrida nocturna del 2026-11-04)

- Plan inicial: `python listar_sin_curva.py` con `--duracion_min 20` daba 380 videos de fuego de mas de 20 s sin curva (8.11 h de video), que al ritmo medido (0.14 s por segundo de video) se procesaban en unas 1.1 h.
- **Lo que se corrio**: como cabian en la noche, se uso `python listar_sin_curva.py --duracion_min 0`, que genero `lista_sin_curva.csv` con **1 507 videos** de fuego sin curva (todos, de cualquier duracion), primero un video por origen y despues los demas clips, del mas largo al mas corto, sin copias por nombre. `preanotar.py --seleccion lista_sin_curva.csv --salida propuestas_extra.csv` termino con los 1 507 (ver "Curvas de todos los videos de fuego y candidatos con crecimiento sostenido").
- Salida separada: `propuestas_extra.csv` y las curvas en `curvas/`.

## Paso 2 del plan del 50 %: deteccion + ByteTrack con dt real en el filtro de Kalman

Fecha: 2026-11-04. Rama: `avance-50`. Script: `seguimiento.py`.

### Que se hizo

- Deteccion con `fasdd_v8s_full` sobre los fotogramas de `LectorVideo` (10 fps, marcas reales), con `conf = 0.1` para que ByteTrack tenga detecciones de baja confianza en su segunda etapa.
- Un tracker ByteTrack por clase (fuego y humo), para que una caja de humo no continue una trayectoria de fuego. Parametros de `bytetrack.yaml` de Ultralytics: `track_high_thresh 0.25`, `track_low_thresh 0.1`, `new_track_thresh 0.25`, `track_buffer 30` (en esta version se cuenta en muestras: 3 s a 10 fps), `match_thresh 0.8`, `fuse_score True`.
- Modificacion propia, sin tocar el codigo instalado (subclases):
  - `KalmanFilterDT(KalmanFilterXYAH)`: la matriz de transicion usa `dt_rel = dt_k / 0.1 s` en lugar de 1, y el ruido de proceso se multiplica por `dt_rel` (varianza proporcional al tiempo transcurrido). Con muestreo regular (`dt_k = 0.1 s`) es identico al original.
  - `STrackDT(STrack)`: su propio filtro compartido, porque `STrack.multi_predict` usa `STrack.shared_kalman` escrito a mano.
  - `TrackerDT(BYTETracker)`: recibe `dt` en cada `update` y lo fija en los filtros antes de predecir.
- Tres variantes con las mismas detecciones: `original` (BYTETracker de Ultralytics, `dt = 1`), `dt_constante` (las subclases con `dt_rel = 1`, como control de la implementacion) y `dt_real`.
- Cada video se corrio con todos los fotogramas muestreados y quitando el 50 % al azar (`--quitar 0.5`), que es donde `dt` deja de ser constante.
- Medidas por clase, sin verdad de terreno: numero de trayectorias; cortes (una trayectoria nueva cuya primera caja tiene IoU >= 0.3 con la ultima caja de otra trayectoria de la misma clase que termino en los 3 s anteriores; aproxima cuantas veces se corta una misma llama); trayectorias cortas (menos de 3 muestras); duracion media; cobertura (fraccion de detecciones con confianza >= 0.25 asignadas a una trayectoria).
- Nota del entorno: al importar el tracker, Ultralytics instalo por su cuenta la dependencia `lap` 0.5.14 (la usa ByteTrack para la asignacion).

### Comando

```
python seguimiento.py
```

### Videos de prueba (`resultados/seguimiento/videos_prueba.csv`)

- `Berkeley Township Structure Fire 2_12_21 (Heavy Flames  Drone Angle) New Jersey.mp4`: camara movil (dron), 39 s, 30 fps; tiene cortes de escena.
- `14294880_3840_2160_24fps.mp4`: vista aerea 4K, 27 s, 23.98 fps (dt irregular).
- `12021934_3840_2160_30fps.mp4`: horno con llamas que parpadean, 4K, 33 s, 30 fps.
- `5284fda1-b4c45d7c.mp4`: 20 s, 25 fps (dt irregular).

### Numeros (clase fuego; la tabla completa con humo esta en `resultados/seguimiento/comparacion.csv`)

| Video | Quitar | Variante | Trayectorias | Cortes | Cortas (<3 muestras) | Duracion media (s) | Cobertura |
|---|---|---|---|---|---|---|---|
| Berkeley Township Stru | 0.0 | original | 78 | 19 | 20 | 1.23 | 0.361 |
| Berkeley Township Stru | 0.0 | dt_real | 78 | 19 | 20 | 1.23 | 0.361 |
| Berkeley Township Stru | 0.5 | original | 38 | 6 | 12 | 1.65 | 0.274 |
| Berkeley Township Stru | 0.5 | dt_real | 39 | 6 | 17 | 1.36 | 0.261 |
| 14294880_3840_2160_24f | 0.0 | original | 9 | 0 | 0 | 8.91 | 0.6 |
| 14294880_3840_2160_24f | 0.0 | dt_real | 9 | 0 | 0 | 8.24 | 0.616 |
| 14294880_3840_2160_24f | 0.5 | original | 6 | 0 | 0 | 10.23 | 0.456 |
| 14294880_3840_2160_24f | 0.5 | dt_real | 6 | 1 | 0 | 5.58 | 0.404 |
| 12021934_3840_2160_30f | 0.0 | original | 15 | 3 | 3 | 2.51 | 0.693 |
| 12021934_3840_2160_30f | 0.0 | dt_real | 15 | 3 | 3 | 2.51 | 0.693 |
| 12021934_3840_2160_30f | 0.5 | original | 10 | 2 | 2 | 3.85 | 0.664 |
| 12021934_3840_2160_30f | 0.5 | dt_real | 13 | 4 | 3 | 2.61 | 0.664 |
| 5284fda1-b4c45d7c.mp4 | 0.0 | original | 12 | 4 | 1 | 3.11 | 0.533 |
| 5284fda1-b4c45d7c.mp4 | 0.0 | dt_real | 12 | 4 | 1 | 3.11 | 0.533 |
| 5284fda1-b4c45d7c.mp4 | 0.5 | original | 8 | 1 | 1 | 4.91 | 0.478 |
| 5284fda1-b4c45d7c.mp4 | 0.5 | dt_real | 10 | 2 | 3 | 2.62 | 0.442 |

- `dt_constante` dio exactamente las mismas trayectorias que `original` en los 8 casos (4 videos, con y sin quitar fotogramas): la implementacion de las subclases es correcta.
- Sin quitar fotogramas, `dt_real` es identico a `original` en 3 de 4 videos; en el de 23.98 fps tiene las mismas 9 trayectorias de fuego, con cobertura 0.616 frente a 0.600 y duracion media 8.24 s frente a 8.91 s.
- Quitando el 50 % de los fotogramas, `dt_real` empeora el seguimiento en los 4 videos: mas trayectorias y mas cortas, y menor cobertura (en el dron, humo: 15 trayectorias y 7 cortes frente a 12 y 5).

### Diagnostico (quitar 0.5, fuego, suma de los 4 videos)

| Variante | Trayectorias | Cortes | Cortas | Cobertura media |
|---|---|---|---|---|
| original (`dt = 1`) | 62 | 9 | 15 | 0.468 |
| `dt` real en transicion y ruido | 68 | 13 | 23 | 0.443 |
| `dt` real con tope 2 | 64 | 7 | 17 | 0.465 |
| `dt` real con tope 1.5 | 64 | 10 | 16 | 0.465 |
| `dt` solo en el ruido de proceso | 60 | 8 | 15 | 0.468 |

- La perdida viene de la matriz de transicion: la velocidad estimada de una caja de fuego es casi todo ruido (la llama parpadea), y multiplicarla por un `dt` grande aleja la caja predicha y baja el IoU con la deteccion. ByteTrack asocia por IoU con la media predicha, asi que el crecimiento de la covarianza no compensa.
- Escalar solo el ruido de proceso, o poner un tope de 2 al `dt` relativo, recupera el rendimiento del original.
- **Decision (2026-11-05): el sistema usa `dt` real con tope 2 (`dt_rel = min(dt_k / 0.1, 2)`, variante `dt_tope2`, parametro `--dt_max 2.0`).** Mantiene el `dt` real en la matriz de transicion, como plantea la tesis, y evita extrapolar la velocidad ruidosa de la llama mas alla de 2 periodos nominales (0.2 s).
- Las variantes quedaron dentro de `seguimiento.py` (la tabla anterior salio de un script de diagnostico; esta es la corrida oficial, con los mismos numeros para fuego). Suma de los 4 videos por variante (para el Cap5; `dt_constante` se omite porque es identica a `original`):

| Fotogramas | Clase | Variante | Trayectorias | Cortes | Cortas | Duracion media (s) | Cobertura media |
|---|---|---|---|---|---|---|---|
| todos | fuego | original (`dt` = 1) | 114 | 26 | 24 | 3.94 | 0.547 |
| todos | fuego | `dt` real en transicion y ruido | 114 | 26 | 24 | 3.77 | 0.551 |
| todos | fuego | `dt` real con tope 2 (sistema) | 114 | 26 | 24 | 3.77 | 0.551 |
| todos | fuego | `dt` solo en el ruido de proceso | 114 | 26 | 24 | 3.94 | 0.547 |
| todos | humo | original (`dt` = 1) | 21 | 5 | 2 | 15.15 | 0.917 |
| todos | humo | `dt` real en transicion y ruido | 21 | 5 | 2 | 15.15 | 0.917 |
| todos | humo | `dt` real con tope 2 (sistema) | 21 | 5 | 2 | 15.15 | 0.917 |
| todos | humo | `dt` solo en el ruido de proceso | 21 | 5 | 2 | 15.15 | 0.917 |
| quitando 50 % | fuego | original (`dt` = 1) | 62 | 9 | 15 | 5.16 | 0.468 |
| quitando 50 % | fuego | `dt` real en transicion y ruido | 68 | 13 | 23 | 3.04 | 0.443 |
| quitando 50 % | fuego | `dt` real con tope 2 (sistema) | 64 | 7 | 17 | 4.84 | 0.465 |
| quitando 50 % | fuego | `dt` solo en el ruido de proceso | 60 | 8 | 15 | 5.08 | 0.468 |
| quitando 50 % | humo | original (`dt` = 1) | 13 | 5 | 3 | 15.38 | 0.868 |
| quitando 50 % | humo | `dt` real en transicion y ruido | 16 | 7 | 3 | 14.86 | 0.857 |
| quitando 50 % | humo | `dt` real con tope 2 (sistema) | 15 | 6 | 3 | 15.12 | 0.866 |
| quitando 50 % | humo | `dt` solo en el ruido de proceso | 14 | 6 | 3 | 15.24 | 0.866 |


### Archivos generados

- `resultados/seguimiento/trayectorias.csv`: `archivo, quitar, variante, k, frame, t, dt, id, clase, conf, x1, y1, x2, y2, det` (cajas en pixeles del fotograma original).
- `resultados/seguimiento/comparacion.csv`: medidas por video, `quitar`, variante y clase.
- `runs_fuego/seguimiento/<video>_dt_tope2.mp4` (no versionado): video a 10 fps con las cajas, el ID de cada trayectoria, `t` y `dt`, con la variante del sistema.

## Curvas de todos los videos de fuego y candidatos con crecimiento sostenido

Fecha: 2026-11-05. Rama: `avance-50`. Solo estadisticas; no se etiqueto nada.

- Corrida nocturna: `python listar_sin_curva.py --duracion_min 0` y `preanotar.py --seleccion lista_sin_curva.csv --salida propuestas_extra.csv` (prioridad de CPU por debajo de lo normal). Termino con los 1 507 videos de la lista (todos con propuesta, ninguna fila incompleta). Con los 200 anteriores hay **1 707 videos de fuego con curva** (`propuestas.csv` + `propuestas_extra.csv`).
- Todas las propuestas: 1 099 claras, 608 dudosas; 96 sin fuego sostenido (`descartar = 1`).
- **Con crecimiento sostenido segun las reglas (`t_incipiente` o `t_incontrolado`): 218 videos (12.8 %)**, de 206 origenes distintos.
  - Con `t_incipiente`: 218; con `t_incontrolado`: 115; paso directo (`t_incipiente = t_incontrolado`): 57.
  - Claros: 67 (de 67 origenes); dudosos: 151. Motivos en los dudosos: inestable 82, oscila 77, cambio_debil 67, nivel_borde 11, conf_baja 8, intermitente 7, util_corto 2.
  - `t_incipiente`: mediana 8.8 s desde el inicio del video; solo 5 antes de 2 s.
- Duracion de los 218 candidatos: minimo 5.9 s, percentil 25 17.5 s, mediana 30.2 s, percentil 75 113.4 s, maximo 1 767.8 s; 5.93 h en total.

| Duracion (s) | 0-10 | 10-20 | 20-30 | 30-60 | 60-120 | mas de 120 |
|---|---|---|---|---|---|---|
| Candidatos | 19 | 50 | 39 | 36 | 19 | 55 |

- Los 67 claros tienen una mediana de 18.6 s; 57 duran mas de 10 s.
- Comparacion: la IA con tiras de 24 fotos marco cambios en 4 de 200 videos y el humano en 0 de 40; las reglas sobre la curva marcan crecimiento en el 12.8 %. Son candidatos para el etiquetado con mas fotogramas, no etiquetas.

## Paso 3 del plan del 50 %: movimiento de la camara (Shi-Tomasi + LK + RANSAC) y compensacion en el seguimiento

Fecha: 2026-11-05. Rama: `avance-50`. Script: `camara.py` (clase `MovimientoCamara`); compensacion en `seguimiento.py` (`gmc_xyah` y opcion `compensar` de `TrackerDT`).

### Que se hizo

- `MovimientoCamara.actualizar(imagen, cajas)`, por cada par de muestras consecutivas:
  - imagen en gris reducida a 960 px de ancho;
  - puntos Shi-Tomasi (`cv2.goodFeaturesToTrack`: 400 puntos, calidad 0.01, distancia minima 1, bloque 3, como el `sparseOptFlow` de BoT-SORT) **fuera de las cajas detectadas** en la muestra anterior;
  - seguimiento con Lucas-Kanade piramidal (`cv2.calcOpticalFlowPyrLK`);
  - similitud con RANSAC (`cv2.estimateAffinePartial2D`, umbral 3 px, confianza 0.99);
  - salida en coordenadas del fotograma original: matriz 2x3, escala `s`, rotacion, `tx`, `ty`, proporcion de inliers y numero de puntos;
  - si hay menos de 10 puntos o menos del 50 % de inliers, se usa la identidad y la muestra se marca como no valida (`valido = 0`; por ejemplo, cortes de escena o movimientos bruscos).
- Verificacion de la estimacion: error absoluto medio entre la muestra actual y la anterior, fuera de las cajas, sin alinear y alineando la anterior con la similitud estimada.
- Compensacion en ByteTrack (como BoT-SORT, despues de predecir y antes de asociar): el `multi_gmc` de Ultralytics trata el estado como cuatro pares (x,y) y su documentacion indica que los filtros XYAH deben reemplazarlo; `gmc_xyah` aplica la similitud a la posicion y la velocidad del centro, escala la altura y su velocidad por `s` y deja igual la relacion de aspecto.
- Dos mascaras para los puntos: `todas` (fuera de las cajas de fuego y de humo, como dice la tesis) y `fuego` (solo fuera de las cajas de fuego).

### Comandos

```
python camara.py --mascara todas
python camara.py --mascara fuego
```

### Estimacion del movimiento (mismos 4 videos de prueba, todos los fotogramas, mascara todas)

| Video | Muestras validas | Desplazamiento medio (px) | Inliers medio | Error de alineacion |
|---|---|---|---|---|
| Berkeley (dron) | 382/391 | 14.1 | 0.91 | 66.6 % menor |
| 14294880 (aerea) | 272/272 | 2.7 | 0.97 | 18.7 % menor |
| 12021934 (horno, camara en mano) | 323/332 | 19.8 | 0.84 | 35.2 % menor |
| 5284fda1 (camara fija) | 199/199 | 0.1 | 0.95 | 2.1 % mayor (sin movimiento que corregir) |

- En el dron las cajas de humo dejan libre en promedio el 40 % de la imagen (minimo 1 %). Con la mascara `fuego` el error de alineacion baja 57.2 % (se mide en una zona distinta, que incluye el humo, asi que no es comparable directamente).
- La grafica `resultados/camara/movimiento_Berkeley Township Structure Fire 2_12_21_fuego.png` muestra un alejamiento de zoom hacia los 13-14 s (`s` hasta 0.87) con paneo, mas movimiento desde los 25 s y 9 muestras no validas (inliers por debajo de 0.5).

### Efecto en el seguimiento (variante del sistema `dt_tope2`)

| Video | Fotogramas | Clase | dt_tope2 sin compensar | con compensacion, mascara todas | con compensacion, mascara fuego |
|---|---|---|---|---|---|
| Berkeley Township Stru | todos | fuego | 78 tr / 19 cortes / 20 cortas / cob 0.361 | 82 tr / 19 cortes / 17 cortas / cob 0.395 | 74 tr / 13 cortes / 15 cortas / cob 0.409 |
| Berkeley Township Stru | todos | humo | 20 tr / 5 cortes / 2 cortas / cob 0.838 | 21 tr / 4 cortes / 2 cortas / cob 0.846 | 21 tr / 4 cortes / 2 cortas / cob 0.838 |
| Berkeley Township Stru | quitando 50 % | fuego | 39 tr / 5 cortes / 14 cortas / cob 0.277 | 41 tr / 8 cortes / 16 cortas / cob 0.303 | 37 tr / 8 cortes / 10 cortas / cob 0.322 |
| Berkeley Township Stru | quitando 50 % | humo | 14 tr / 6 cortes / 3 cortas / cob 0.74 | 15 tr / 7 cortes / 3 cortas / cob 0.725 | 15 tr / 7 cortes / 3 cortas / cob 0.737 |
| 14294880_3840_2160_24f | todos | fuego | 9 tr / 0 cortes / 0 cortas / cob 0.616 | 7 tr / 0 cortes / 0 cortas / cob 0.59 | 7 tr / 0 cortes / 0 cortas / cob 0.59 |
| 14294880_3840_2160_24f | todos | humo | 1 tr / 0 cortes / 0 cortas / cob 0.996 | 1 tr / 0 cortes / 0 cortas / cob 0.996 | 1 tr / 0 cortes / 0 cortas / cob 0.996 |
| 14294880_3840_2160_24f | quitando 50 % | fuego | 6 tr / 0 cortes / 0 cortas / cob 0.447 | 6 tr / 1 cortes / 0 cortas / cob 0.408 | 5 tr / 0 cortes / 0 cortas / cob 0.417 |
| 14294880_3840_2160_24f | quitando 50 % | humo | 1 tr / 0 cortes / 0 cortas / cob 0.992 | 1 tr / 0 cortes / 0 cortas / cob 0.992 | 1 tr / 0 cortes / 0 cortas / cob 0.992 |
| 12021934_3840_2160_30f | todos | fuego | 15 tr / 3 cortes / 3 cortas / cob 0.693 | 13 tr / 2 cortes / 1 cortas / cob 0.722 | 13 tr / 2 cortes / 1 cortas / cob 0.722 |
| 12021934_3840_2160_30f | todos | humo | 0 tr / 0 cortes / 0 cortas / cob 0.0 | 0 tr / 0 cortes / 0 cortas / cob 0.0 | 0 tr / 0 cortes / 0 cortas / cob 0.0 |
| 12021934_3840_2160_30f | quitando 50 % | fuego | 11 tr / 2 cortes / 2 cortas / cob 0.668 | 13 tr / 1 cortes / 3 cortas / cob 0.678 | 13 tr / 1 cortes / 3 cortas / cob 0.678 |
| 12021934_3840_2160_30f | quitando 50 % | humo | 0 tr / 0 cortes / 0 cortas / cob 0.0 | 0 tr / 0 cortes / 0 cortas / cob 0.0 | 0 tr / 0 cortes / 0 cortas / cob 0.0 |
| 5284fda1-b4c45d7c.mp4 | todos | fuego | 12 tr / 4 cortes / 1 cortas / cob 0.533 | 12 tr / 4 cortes / 1 cortas / cob 0.533 | 12 tr / 4 cortes / 1 cortas / cob 0.533 |
| 5284fda1-b4c45d7c.mp4 | todos | humo | 0 tr / 0 cortes / 0 cortas / cob 0.0 | 0 tr / 0 cortes / 0 cortas / cob 0.0 | 0 tr / 0 cortes / 0 cortas / cob 0.0 |
| 5284fda1-b4c45d7c.mp4 | quitando 50 % | fuego | 8 tr / 0 cortes / 1 cortas / cob 0.469 | 8 tr / 0 cortes / 1 cortas / cob 0.469 | 8 tr / 0 cortes / 1 cortas / cob 0.469 |
| 5284fda1-b4c45d7c.mp4 | quitando 50 % | humo | 0 tr / 0 cortes / 0 cortas / cob 0.0 | 0 tr / 0 cortes / 0 cortas / cob 0.0 | 0 tr / 0 cortes / 0 cortas / cob 0.0 |

| Fotogramas | Clase | Variante | Trayectorias | Cortes | Cortas | Cobertura media |
|---|---|---|---|---|---|---|
| todos | fuego | dt_tope2 sin compensar | 114 | 26 | 24 | 0.551 |
| todos | fuego | dt_tope2 + compensacion (mascara todas) | 114 | 25 | 19 | 0.560 |
| todos | fuego | dt_tope2 + compensacion (mascara fuego) | 106 | 19 | 17 | 0.564 |
| todos | humo | dt_tope2 sin compensar | 21 | 5 | 2 | 0.917 |
| todos | humo | dt_tope2 + compensacion (mascara todas) | 22 | 4 | 2 | 0.921 |
| todos | humo | dt_tope2 + compensacion (mascara fuego) | 22 | 4 | 2 | 0.917 |
| quitando 50 % | fuego | dt_tope2 sin compensar | 64 | 7 | 17 | 0.465 |
| quitando 50 % | fuego | dt_tope2 + compensacion (mascara todas) | 68 | 10 | 20 | 0.465 |
| quitando 50 % | fuego | dt_tope2 + compensacion (mascara fuego) | 63 | 9 | 14 | 0.472 |
| quitando 50 % | humo | dt_tope2 sin compensar | 15 | 6 | 3 | 0.866 |
| quitando 50 % | humo | dt_tope2 + compensacion (mascara todas) | 16 | 7 | 3 | 0.859 |
| quitando 50 % | humo | dt_tope2 + compensacion (mascara fuego) | 16 | 7 | 3 | 0.865 |

- Con todos los fotogramas, la compensacion con mascara `fuego` reduce los cortes de fuego de 26 a 19 (de 19 a 13 en el dron y de 3 a 2 en el horno), las trayectorias cortas de 24 a 17 y sube la cobertura media de 0.551 a 0.564. Con mascara `todas` casi no cambia (26 a 25 cortes).
- Quitando el 50 % de los fotogramas, la compensacion no mejora los cortes (7 sin compensar, 9 con mascara `fuego`, 10 con `todas`); son pocos casos y el desplazamiento entre muestras es mayor (22 px de media en el dron).
- En el video de camara fija la compensacion no cambia nada, como se esperaba.
- La medida de cortes compara cajas en coordenadas de la imagen; con camara movil es una aproximacion.
- Pendiente de decision: mascara para los puntos (`todas`, como la tesis, o `fuego`) y si el sistema usa la compensacion en el seguimiento.

### Archivos generados

- `resultados/camara/movimiento_<mascara>.csv`: `archivo, quitar, k, frame, t, dt, s, rot, tx, ty, inliers, puntos, valido, libre, err_sin, err_con` por muestra.
- `resultados/camara/comparacion_<mascara>.csv`: medidas de seguimiento por video, `quitar`, variante (`original`, `original_cmc`, `dt_tope2`, `dt_tope2_cmc`) y clase.
- `resultados/camara/movimiento_<video>_<mascara>.png`: `tx`, `ty`, `s` e inliers en el tiempo (en rojo, las muestras no validas).

### Decisiones del 2026-11-05 sobre el paso 3

- Mascara de puntos: **solo las cajas de fuego** (`--mascara fuego`, ahora el valor por defecto de `camara.py`). Los puntos sobre el humo los descarta RANSAC como inliers que no siguen la camara.
- **El sistema usa la compensacion de camara en el seguimiento**: variante `dt_tope2_cmc` (`dt_rel = min(dt_k / 0.1, 2)` + `gmc_xyah` con la similitud estimada).
- Con el 50 % de fotogramas quitados la compensacion no reduce los cortes (7 sin compensar, 9 con compensacion): se reporta como hallazgo, no como mejora.

## Paso 4 del plan del 50 %: los 13 descriptores por trayectoria y fotograma

Fecha: 2026-11-05. Rama: `avance-50`. Script: `descriptores.py` (clase `Descriptores`).

### Que se hizo

Pipeline completo por video: `LectorVideo` (10 fps, marcas reales) -> YOLOv8s (`conf 0.1`) -> `MovimientoCamara` (mascara fuego) -> ByteTrack `dt_tope2_cmc` (un tracker por clase) -> descriptores de cada trayectoria de fuego activa.

| N | Descriptor | Calculo |
|---|---|---|
| 1 | confianza | confianza de la deteccion de fuego asociada a la trayectoria |
| 2 | area | `A_k` = area de la caja / area del fotograma |
| 3 | tasa de crecimiento | `(A_k - s^2 A_prev) / dt`, con `A_prev` de la ultima observacion de la trayectoria, `s` la escala de la similitud compuesta entre esa observacion y la actual, y `dt` el tiempo entre ambas; 0 en la primera observacion |
| 4 | crecimiento relativo | d3 / `A_k` |
| 5-6 | centro | `(x/W, y/H)` |
| 7 | velocidad del centro | norma de (centro actual - centro anterior llevado al fotograma actual con la similitud compuesta), en coordenadas normalizadas (x/W, y/H), dividida entre `dt`; 0 en la primera observacion |
| 8 | persistencia | muestras de los ultimos 3 s con la trayectoria detectada / muestras del video en esos 3 s |
| 9 | variacion de forma | varianza de `w/h` de la caja en los ultimos 3 s |
| 10 | confianza de humo | maxima confianza de las detecciones de humo (confianza >= 0.25) que tocan la region ampliada |
| 11 | area de humo | fraccion de la region ampliada cubierta por la union de esas cajas de humo (rejilla 48x48) |
| 12 | tasa del humo | cambio de d11 desde la ultima observacion / `dt` |
| 13 | regiones activas | trayectorias de fuego activas en el fotograma |

- Region ampliada del humo: la caja de fuego escalada **2 veces** en ancho y alto alrededor de su centro, recortada al fotograma (`--factor_humo 2.0`).
- Contexto por fila: `s_k`, `rot` (radianes), `tx/W`, `ty/H` e `inliers` de la similitud entre la muestra anterior y la actual; identidad cuando la estimacion no es valida.
- Parametros: `--ventana 3.0` (d8 y d9), `--conf_humo 0.25`, `--variante dt_tope2_cmc`, `--mascara fuego`.

### Comando

```
python descriptores.py
```

### Numeros (4 videos de prueba, todos los fotogramas)

| Video | Muestras | Trayectorias de fuego | Filas | Tiempo (s) |
|---|---|---|---|---|
| Berkeley (dron) | 392 | 74 | 731 | 22.4 |
| 14294880 (aerea 4K) | 273 | 7 | 607 | 29.5 |
| 12021934 (horno 4K) | 333 | 13 | 332 | 30.9 |
| 5284fda1 | 200 | 12 | 285 | 5.5 |

- Las trayectorias coinciden con las de `camara.py` para la misma variante (74, 7, 13, 12).
- 1 955 filas, ninguna con NaN o infinito.

| Descriptor | Media | Desv. | Min | Mediana | Max |
|---|---|---|---|---|---|
| d1 confianza | 0.427 | 0.208 | 0.101 | 0.389 | 0.873 |
| d2 area | 0.059 | 0.117 | 0.0004 | 0.014 | 0.563 |
| d3 tasa de crecimiento (1/s) | 0.004 | 0.084 | -1.132 | 0.0001 | 0.601 |
| d4 crecimiento relativo (1/s) | 0.005 | 1.822 | -9.926 | 0.012 | 7.267 |
| d5 centro x | 0.450 | 0.145 | 0.053 | 0.438 | 0.971 |
| d6 centro y | 0.536 | 0.139 | 0.118 | 0.549 | 0.844 |
| d7 velocidad del centro (1/s) | 0.089 | 0.121 | 0.000 | 0.039 | 0.949 |
| d8 persistencia | 0.588 | 0.364 | 0.032 | 0.633 | 1.000 |
| d9 variacion de forma | 0.005 | 0.011 | 0.000 | 0.001 | 0.083 |
| d10 confianza de humo | 0.362 | 0.339 | 0.000 | 0.359 | 0.832 |
| d11 area de humo | 0.421 | 0.425 | 0.000 | 0.354 | 1.000 |
| d12 tasa del humo (1/s) | -0.027 | 1.311 | -10.0 | 0.000 | 10.0 |
| d13 regiones activas | 2.68 | 1.70 | 1 | 2 | 10 |

- Contexto: `s_k` entre 0.871 y 1.023, `rot` entre -0.035 y 0.040 rad, `dt` entre 0.080 y 0.125 s.
- Grafico de los 13 descriptores de la trayectoria mas larga de la vista aerea (id 1, 229 muestras): `resultados/descriptores/descriptores_14294880_3840_2160_24fps_id1.png`. El centro de la caja se desplaza lento en la imagen (paneo de la camara aerea) mientras la velocidad compensada d7 queda cerca de 0; d11 esta saturada en 1 hasta los 17 s (la caja de humo cubre toda la region ampliada de una llama pequena) y luego baja.
- Observaciones para el modelo: d4 y d12 tienen colas largas (d12 llega a +-10 cuando d11 salta de 0 a 1 en 0.1 s, por el parpadeo de las cajas de humo); conviene normalizarlos o recortarlos antes de la CfC.

### Archivos generados

- `resultados/descriptores/descriptores.csv`: `archivo, id, k, t, dt, d1..d13, s_k, rot, tx, ty, inliers`.
- `resultados/descriptores/descriptores_14294880_3840_2160_24fps_id1.png`.

## Paso 5 del plan del 50 %: demo

Fecha: 2026-11-05. Rama: `avance-50`. Script: `demo.py`.

### Que se hizo

- Pipeline del sistema (`LectorVideo` 10 fps, YOLOv8s, camara con mascara fuego, ByteTrack `dt_tope2_cmc`, `Descriptores`) sobre dos videos que contrastan:
  - controlado: `19151618-hd_1920_1080_30fps.mp4` (vela, 30 s; en `propuestas.csv`: controlado, clara, CV del area 0.068). Se eligio un video de Pexels sin marca de agua; se descartaron una fogata con marca de agua de Adobe Stock y dos videos que parecen piras de cremacion.
  - crecimiento: `12399227_3840_2160_60fps.mp4` (Pexels, 38 s, 4K, 60 fps; uno de los 67 candidatos claros: `t_incipiente` 2.8 s, area de 0.018 a 0.17). Una mano acerca un papel a una vela; el papel arde, la llama crece y despues se retira o se apaga, y al final queda solo la vela.
  - **Retirado el 2026-11-05**: el ID 12399227 no existe en Pexels (verificado por el usuario en pexels.com), asi que su GIF se elimino del repositorio y del historial de git y no se publica ninguna imagen de ese video. La demo de crecimiento queda pendiente de un video con licencia verificada; `demo.py` ya no tiene video de crecimiento por defecto.
  - **Cambio del 2026-11-05**: la primera version de la demo usaba `Jeder dritte Brand entsteht durch Elektrizitat.mp4` (campana de prevencion de terceros incluida en DetectiumFire, 21 s, `t_incipiente` 12.6 s): un aparato sobre un refrigerador se enciende hacia los 5 s y el fuego termina ocupando la escena (region principal: ids 3, 8 y 30, 9 cambios; GIF de 6.3 MB). Se reemplazo por el video de Pexels para poder publicar el GIF en el README con una licencia clara.
- Video de salida (mp4 H.264 con Media Foundation de Windows, 10 fps, 1400x540): cajas de fuego y humo con el ID de cada trayectoria, `t`, `dt` y `k`, y un panel lateral con la evolucion de d2 (area), d3 (tasa de crecimiento) y d11 (area de humo) hasta el instante actual.
- Region principal del panel: la trayectoria de fuego con mayor area en cada instante (coherente con "el nivel de la escena es el de la trayectoria mas severa"); las lineas rojas del panel marcan los cambios de ID. Se definio asi porque en el video del lavadero una sola trayectoria no alcanzaba: el fuego aparecia en las trayectorias 3 y 8 (5.4 a 11.9 s) y despues de un cambio de plano continuaba en la 30 (12.3 a 21.1 s).
- GIF de cada video a 5 fps y 640 px de ancho, para el README.

### Comando

```
python demo.py
```

### Numeros

| Video | Muestras | Region principal | Cambios de ID | mp4 | GIF |
|---|---|---|---|---|---|
| vela (controlado) | 298 | id 1 | 0 | 18.4 MB | 149 cuadros, 5.6 MB |
| vela y papel (crecimiento) | 383 | 19 ids (1, 2, 3, 6, 7, 11, 18, 26, 46, 50, 54, 57, 86, 103, 104, 114, 115, 119, 124) | 68 | 33.6 MB | 192 cuadros, 7.1 MB |

- Vela: d2 plana cerca de 0.055, con un pico hacia los 15 s cuando la llama se inclina; d3 cerca de 0; d11 = 0 (sin humo).
- Vela y papel: d2 sube de ~0.04 a ~0.2 entre los 2 y los 9 s, oscila mientras el papel arde en pedazos y baja al final cuando queda solo la vela; d3 positiva durante el crecimiento; d11 = 0 (sin humo). Es crecimiento seguido de decaimiento, un ejemplo mas debil que el del lavadero.
- Fragmentacion: el papel se rompe en llamas separadas y la region principal cambia de ID 68 veces (tarea (e) del 80 %).
- En el lavadero (version anterior) el detector no veia la primera llama sobre la pared blanca hasta los 5.4 s.

### Archivos generados

- `runs_fuego/demo/demo_controlado.mp4` y `demo_crecimiento.mp4` (no versionados).
- `resultados/demo/demo_controlado.gif` y `resultados/demo/demo_crecimiento.gif` (version final, ver "Cierre de la demo para el README").
- Los GIF anteriores de crecimiento (lavadero y 12399227) se retiraron del repositorio y del historial.

## Cierre de la demo para el README (2026-11-05)

### Licencia y criterio

- Segun la Tabla 5 (apendice B.1) del paper DetectiumFire (arXiv:2511.02495), los videos de busqueda web del dataset se publican bajo CC BY-NC-SA 4.0: se pueden mostrar fotogramas con credito, sin fines comerciales y con la misma licencia. Los IDs con patron de nombre de Pexels no corresponden necesariamente a videos de Pexels (12399227 no existe alli); solo la vela 19151618 se verifico en pexels.com y conserva el credito a Pexels.
- Criterio para cualquier fotograma publicado: sin marca de agua, sin logo de canal, TikTok o noticiero, sin texto superpuesto y sin caras reconocibles. Todo lo que sea `700_F_*` (Adobe Stock) queda fuera. Cada imagen se reviso a ojo.

### Revision de los videos

| Video | Resultado | Motivo |
|---|---|---|
| `19151618-hd_1920_1080_30fps.mp4` (vela) | pasa | sin texto, logos ni personas |
| `13871980_3840_2160_24fps.mp4` (quema en un campo) | no pasa | entre los 5 y los 8.4 s un hombre gira la cabeza y se le ve el perfil; ademas, hacia los 11 s hay un cambio de plano y gran parte del "crecimiento" del area es por el acercamiento de la camara |
| `12487581_1920_1080_30fps.mp4` (basural) | pasa | la patente de una camioneta (9-10 s) solo se lee en el original 1080p; en la salida (video a 640 px, GIF a ~370 px) es ilegible; motociclistas lejanos con casco, sin caras distinguibles; los textos son de bolsas, no superpuestos |
| `Living room fires with and without a fire sprinkler (Timecode).mp4` (prueba del usuario) | no pasa | codigo de tiempo sobreimpreso en el video original; sus capturas estan solo en `runs_fuego/prueba_ale/` (no versionado) |

### Demo final

- Comando:

```
python demo.py --crecimiento 12487581_1920_1080_30fps.mp4 --alto 360 --ancho_panel 330 --fps_mp4 5 --fps_gif 2.5 --ancho_gif 560 --colores_gif 96 --salida_mp4 "../videos_readme_tesis" --salida_gif resultados/demo
```

- Nuevas opciones de `demo.py`: `--fps_mp4` (el codificador H.264 de Media Foundation ignora la calidad y su tamano depende sobre todo del numero de cuadros), `--colores_gif`; el paso de las marcas del eje de tiempo ahora depende de la duracion; `--crecimiento` sin valor por defecto.

| Video | Muestras | Region principal | Cambios de ID | mp4 (5 fps, 970x360 / 990x360) | GIF (2.5 fps, 560 px, 96 colores) |
|---|---|---|---|---|---|
| vela (controlado) | 298 | id 1 | 0 | 5.7 MB | 75 cuadros, 2.3 MB |
| basural (crecimiento) | 285 | 12 ids | 37 | 6.3 MB | 72 cuadros, 4.4 MB |

- Basural: el fuego se propaga a lo largo de la basura y aparece humo (d11 hasta ~0.9); d2 de la region principal es pequena (~0.02) con un pico de ~0.3 hacia los 17 s, en buena parte por el acercamiento de la camara en mano. Es un ejemplo de propagacion mas que de crecimiento de una sola llama.
- Los mp4 (H.264, menos de 10 MB) estan fuera del repositorio, en `../videos_readme_tesis/`, para subirlos a GitHub arrastrandolos en el editor.

### Repositorio

- `CLAUDE.md` y `AGENTS.md` se sacaron del repositorio y del historial y estan en `.gitignore` (quedan solo en el disco del autor).
- Los 14 mosaicos de `resultados/preprocesamiento/mosaicos/` se sacaron del repositorio y del historial y estan en `.gitignore`.

## README: evidencias del avance (2026-11-05)

- `docs/img/pipeline_estado.png`: figura de la tesis con las etapas 1 a 4 en verde y 5 a 8 en gris (aportada por el autor).
- `python figuras_readme.py` genera en `docs/img/` las figuras nuevas e imprime las tablas y conteos, todo calculado desde los CSV:

| Figura | Fuente de los datos | Fotogramas |
|---|---|---|
| `preprocesamiento_dt.png` | `5284fda1-b4c45d7c.mp4`: marcas leidas del archivo y `resultados/preprocesamiento/marcas.csv`; 500 fotogramas a 25 fps -> 200 muestras, `dt_k` de 80 a 120 ms, media 100.1 ms | no (grafica) |
| `etiquetado_curva.png` | `curvas/12487581_1920_1080_30fps.csv` y `propuestas.csv`: tramo util 0 a 28.47 s, paso a incipiente 9.4 s, dudosa (inestable) | no (grafica) |
| `etiquetado_tira_ia.png` | `tiras/12857999_3840_2160_24fps.jpg`, `resultados/ia_crudo/12857999_3840_2160_24fps.json` y `etiquetas.csv`: IA incipiente, autor incipiente | si, DetectiumFire |
| `etiquetar_v2_ciego.png` y `.gif` | `etiquetar_v2.dibujar` en modo ciego con la etiqueta guardada por el autor; videos 12857999, 14294952, 14035491 y la vela 19151618; GIF de 4 cuadros, 0.56 MB | si, DetectiumFire y Pexels (vela) |
| `matriz_confusion.png` | `etiquetas.csv` y `resultados/etiquetas_ia.csv`: 40 videos, acuerdo 85.0 %, kappa ponderado lineal 0.801 | no (grafica) |

- Revision a ojo con el criterio de imagenes: 12857999 (fuego al pie de un arbol), 14294952 (frente de fuego en pasto) y 14035491 (rama ardiendo) no tienen marcas de agua, logos, texto superpuesto ni personas; los tres estan entre los 40 validados y el autor y la IA coinciden (incipiente, incontrolado, incipiente).
- Conteos (desde los CSV): 200 videos etiquetados por la IA, 40 validados por el autor, 1 707 curvas de area, 218 candidatos de crecimiento.
- El PDF de la tesis (borrador) se uso solo como fuente para los nombres de las etapas y del Cuadro 4.4; no se versiona (`.gitignore`).

### Diferencias entre el capitulo 4 del borrador en PDF y lo implementado (para actualizar la tesis)

| Seccion del PDF | Dice | Implementado |
|---|---|---|
| 4.2.3 Etiquetado de la severidad | etiquetado semiautomatico: el anotador acepta o corrige cada propuesta y "ninguna etiqueta entra al conjunto sin verificacion humana" | etiquetas de IA sin revision humana (200 videos) y 40 videos validados por el autor a ciegas desde la tira (decision del 2026-11-04) |
| 4.2.3 | se seleccionaron 400 videos | etiquetados por la IA los 200 primeros de `seleccion_etiquetado.csv`; curvas de area de 1 707 videos |
| 4.2.2 | se eliminan los duplicados | encontrada la copia `14296658_3840_2160_24fps - Copy.mp4`; excluirla del entrenamiento sigue pendiente (tarea del 80 %) |
| 4.4.4 Etapa 3 | la matriz `F_k` usa el `dt_k` real | `dt_rel = min(dt_k / 0.1, 2)` (tope de 2 periodos), compensacion del movimiento de la camara antes de asociar (como BoT-SORT, version XYAH) y un tracker por clase |
| 4.4.5 Etapa 4 | puntos de Shi-Tomasi fuera de las cajas detectadas | fuera de las cajas de fuego (los puntos sobre el humo los descarta RANSAC) |
| 4.4.5 | region ampliada alrededor de la caja de fuego | factor 2 (`--factor_humo 2.0`) |
| Figura 4.1 frente a 4.4.8 | la figura dice "Calibracion y decisiones"; la seccion, "Calibracion y decision" | el README usa "Calibracion y decision" |

## Tareas para el 80 %

- **Normalizacion de los descriptores antes de la CfC**: d4 (crecimiento relativo, de -9.9 a 7.3 en la prueba) y d12 (tasa del humo, hasta +-10) tienen colas largas. Recortarlos por percentiles (por ejemplo 1 y 99, calculados solo en el conjunto de entrenamiento) y luego estandarizar todos los descriptores con la media y la desviacion del entrenamiento; guardar esos parametros para aplicarlos igual en validacion, calibracion, prueba y la demo.
- Etiquetado con IA de los 218 candidatos con crecimiento sostenido, mostrando mas fotogramas alrededor del crecimiento.
- Excluir del entrenamiento la copia `14296658_3840_2160_24fps - Copy.mp4` (su original esta en la prueba humana).
- Observaciones de la prueba con `Living room fires with and without a fire sprinkler (Timecode).mp4` (108 s, 1080p, 30 fps; corrida por el usuario con `descriptores.py` y `demo.py` en `runs_fuego/prueba_ale/`):
  - (a) Calcular d3, d4 y d12 sobre el area suavizada con mediana de 1 s, no sobre el area de cada muestra, para que el parpadeo no domine las tasas.
  - (b) Dar a la region de humo un tamano minimo del 20 % del lado del fotograma, porque d11 se satura con llamas pequenas (una sola caja de humo cubre toda la region ampliada).
  - (c) En d13 contar solo trayectorias con al menos 1 s de duracion.
  - (d) Videos editados: el video del rociador comprime unos 7 min de ensayo en 1:48 con cortes, asi que `dt` no es tiempo real. Contar los cortes de escena con las muestras no validas de `camara.py` y excluir los videos editados del entrenamiento del tiempo al peligro.
  - (e) Fragmentacion: en ese video la trayectoria principal llega al id 415 en 108 s. Evaluar describir la escena como una sola secuencia y reiniciar el seguimiento en cada corte de escena.
  - (f) Humo difuso: con la sala llena de humo, d11 = 0 porque no hay cajas de humo cerca de la llama. Evaluar una medida global de neblina de la escena o reportarlo como limitacion.
