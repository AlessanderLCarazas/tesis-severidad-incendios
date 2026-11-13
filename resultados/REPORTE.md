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

- `Berkeley Township Structure Fire 2_12_21 (Heavy Flames  Drone Angle) New Jersey.mp4`: camara movil en tierra (en mano, con zoom y paneos), 39 s, 30 fps; sin cortes de escena.
  - **Correccion del 2026-11-05.** Este video se habia descrito como "dron" y con "cortes de escena". Se reviso un fotograma cada 5 s y las muestras con la camara no valida o con mayor diferencia de histograma:
    - es una camara en tierra: planos a la altura de las personas, con el camion de bomberos en primer plano; el "Drone Angle" del titulo no corresponde;
    - la escala estimada por `camara.py` muestra dos alejamientos continuos de zoom (de 1.0 a 0.44 entre los 12 y los 16 s, y otro hacia los 28-30 s);
    - las 9 muestras no validas (27.4 s y 34.1-35.1 s) son paneos rapidos, y la mayor diferencia de histograma entre muestras es 0.171;
    - no tiene cortes de escena.
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
- Quitando el 50 % de los fotogramas, `dt_real` empeora el seguimiento en los 4 videos: mas trayectorias y mas cortas, y menor cobertura (en Berkeley, camara en mano, humo: 15 trayectorias y 7 cortes frente a 12 y 5).

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
| Berkeley (camara en mano con zoom) | 382/391 | 14.1 | 0.91 | 66.6 % menor |
| 14294880 (aerea) | 272/272 | 2.7 | 0.97 | 18.7 % menor |
| 12021934 (horno, camara en mano) | 323/332 | 19.8 | 0.84 | 35.2 % menor |
| 5284fda1 (camara fija) | 199/199 | 0.1 | 0.95 | 2.1 % mayor (sin movimiento que corregir) |

- En Berkeley las cajas de humo dejan libre en promedio el 40 % de la imagen (minimo 1 %). Con la mascara `fuego` el error de alineacion baja 57.2 % (se mide en una zona distinta, que incluye el humo, asi que no es comparable directamente).
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

- Con todos los fotogramas, la compensacion con mascara `fuego` reduce los cortes de fuego de 26 a 19 (de 19 a 13 en Berkeley y de 3 a 2 en el horno), las trayectorias cortas de 24 a 17 y sube la cobertura media de 0.551 a 0.564. Con mascara `todas` casi no cambia (26 a 25 cortes).
- Quitando el 50 % de los fotogramas, la compensacion no mejora los cortes (7 sin compensar, 9 con mascara `fuego`, 10 con `todas`); son pocos casos y el desplazamiento entre muestras es mayor (22 px de media en Berkeley).
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
| Berkeley (camara en mano con zoom) | 392 | 74 | 731 | 22.4 |
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

## Fase A del 80 %, pasos 1 a 4: descriptores corregidos (v2) y normalizacion

Fecha: 2026-11-05. Rama: `avance-80`. Scripts nuevos: `descriptores_v2.py` (clase `DescriptoresV2`), `normalizacion.py` (clase `Normalizador`) y `comparar_descriptores.py`. `descriptores.py` y `demo.py` no se modificaron.

### Que se hizo

- `descriptores_v2.py` usa el mismo pipeline de v1 (`LectorVideo` a 10 fps, YOLOv8s con `conf 0.1`, `MovimientoCamara` con mascara fuego, ByteTrack `dt_tope2_cmc`) y cambia cuatro cosas:
  1. **d3, d4 y d12 sobre la serie suavizada.** Para cada trayectoria se guarda la ventana de sus observaciones en `(t_k - 1 s, t_k]`. `d2_med` es la mediana de las areas de esa ventana, cada una llevada a la escala del fotograma actual con la **escala acumulada desde esa muestra hasta ahora**: `A_j * (s_{j+1} * ... * s_k)^2`. Es exacto porque la composicion de similitudes multiplica las escalas. `d11_med` es la mediana de d11 en la misma ventana; no necesita escala porque es una fraccion de la region.
     - `d3 = (d2_med_k - S^2 * d2_med_prev) / dt`, con `S` la escala acumulada desde la observacion anterior de la trayectoria.
     - `d4 = d3 / d2_med_k`.
     - `d12 = (d11_med_k - d11_med_prev) / dt`.
     - d2 y d11 siguen siendo el valor instantaneo.
     - El borde de la ventana tiene una tolerancia de 1e-6 s. Sin ella, por el redondeo de punto flotante, la ventana alternaba entre 10 y 11 muestras y d3 oscilaba entre 0.001 y 0.003 en una rampa de pendiente 0.002 (lo detecto la prueba sintetica).
  2. **Region de humo con tamano minimo:** ancho = `max(2 w, 0.2 W)` y alto = `max(2 h, 0.2 H)`, centrada en la caja de fuego y recortada al fotograma. En 4K la region minima es de 768 x 432 px. Cambian d10 y d11, que usan esa region, y por lo tanto d12.
  3. **d13 cuenta solo las trayectorias de fuego con al menos 1 s de edad.** La edad es `t_k - t` de la primera observacion de la trayectoria; usa solo el pasado. La trayectoria que se esta describiendo tambien cuenta si ya cumplio 1 s; si es mas joven, d13 cuenta solo a las demas que ya cumplieron 1 s. Por eso todas las filas de un mismo fotograma tienen el mismo d13, y d13 puede valer 0 (por ejemplo, en el primer segundo del video o cuando todas las trayectorias son nuevas).
  4. **Normalizacion (`normalizacion.py`):**
     - `Normalizador.ajustar(X)` calcula los percentiles 1 y 99 de cada columna, recorta y calcula media y desviacion (ddof 0) sobre los datos recortados; si la desviacion es 0 se usa 1.
     - `transformar(X)` recorta con esos limites y calcula `(x - media) / desv`.
     - `guardar` y `cargar` usan un JSON con columnas, percentiles, limites, media, desviacion, numero de filas y la lista de videos usados en el ajuste.
     - Por defecto normaliza d1 a d13.
     - **Todavia no se aplica a los videos:** las estadisticas se calcularan solo con los videos de entrenamiento cuando exista la particion (Fase C).
- Parametros nuevos de `descriptores_v2.py`: `--mediana 1.0`, `--min_humo 0.2`, `--min_duracion 1.0`. Con los tres en 0 se reproduce v1.
- Columnas de salida: las mismas de v1 mas `d2_med` y `d11_med` al final.
- Para la corrida sobre todos los videos, `descriptores_v2.py` escribe cada video al terminarlo, registra los videos terminados en `procesados.csv` y con `--reanudar` salta los que ya estan.

### Comandos

```
python descriptores_v2.py
python descriptores_v2.py --mediana 0 --min_humo 0 --min_duracion 0 --salida <carpeta temporal>
python comparar_descriptores.py --despues <carpeta temporal>/descriptores.csv --solo_igualdad
python comparar_descriptores.py
python normalizacion.py ajustar --lista <videos de entrenamiento>.csv --salida resultados/normalizacion.json
python normalizacion.py aplicar --parametros resultados/normalizacion.json --entrada <csv> --salida <csv normalizado>
```

### Pruebas sinteticas (sin YOLO): 13 de 13 correctas

| Prueba | Resultado |
|---|---|
| llama constante con un parpadeo de 1 muestra (+50 % de ancho) | max \|d3\|: v1 0.0241 1/s, v2 0 |
| rampa de area con pendiente 0.002 1/s | v2 da d3 = 0.00200 en todas las muestras desde 1.2 s; retraso de `d2_med` respecto del area real: 0.45 s |
| zoom continuo de 3 % por muestra durante 40 muestras (el area en la imagen crece x10, el area real es constante) | max \|d3\| v2 = 7e-17. Con solo la escala del fotograma actual en vez de la acumulada, d3 llega a 0.0024 1/s |
| zoom irregular (0.97 a 1.05 por muestra) con parpadeo | media \|d4\|: v1 1.771 1/s, v2 0.173 1/s |
| tamano de la region | caja de 100 x 100 px en 4K -> 768 x 432 px; caja de 1000 x 600 px -> 2000 x 1200 px (factor 2) |
| llama de 100 x 100 px con una caja de humo de 400 x 350 px | d11: v1 1.000 (saturada), v2 0.423 |
| caja de humo mas confiable fuera de la region 2x pero dentro de la region minima | d10: v1 0.70, v2 0.90 |
| trayectoria 1 desde 0 s y trayectoria 2 desde 1.5 s | d13 = 0 hasta 0.9 s, 1 desde 1.0 s (tambien en la fila de la trayectoria 2 mientras es joven), 2 desde 2.5 s; v1 da 2 a los 1.5 s |
| v1 contra v2 con los tres parametros en 0, 3 trayectorias, zoom aleatorio y humo | diferencia maxima 2e-14 |
| normalizador | media 0 y desviacion 1 en los datos de ajuste; valores nuevos recortados a p1 y p99 del ajuste; guardar y cargar da el mismo resultado |

### Modo compatibilidad (4 videos reales)

`descriptores_v2.py --mediana 0 --min_humo 0 --min_duracion 0` reproduce exactamente `resultados/descriptores/descriptores.csv`: 1 955 filas con las mismas claves `(archivo, id, k)` y 0 diferencias en d1 a d13, `s_k`, `rot`, `tx`, `ty`, `inliers`, `t` y `dt`. Las diferencias de v2 se deben solo a los tres cambios.

### Numeros: v1 contra v2 (4 videos de prueba, 1 955 filas)

- En v2, d1, d2, d5 a d9 y el contexto son identicos a v1. Cambian d3 (1 835 filas), d4 (1 836), d10 (34), d11 (559), d12 (674) y d13 (973).
- Tiempos de v2: 28.7, 36.3, 38.1 y 7.7 s por video. En el modo compatibilidad, 14294880 marco 1 579 s porque la laptop se suspendio a mitad de la corrida; no es un costo del metodo.

| Descriptor | Version | Media | Desv. | p1 | Mediana | p99 | Min | Max | Variacion media entre muestras |
|---|---|---|---|---|---|---|---|---|---|
| d3 (1/s) | v1 | 0.0037 | 0.0837 | -0.217 | 0.0001 | 0.290 | -1.132 | 0.601 | 0.0582 |
| d3 (1/s) | v2 | 0.0020 | 0.0215 | -0.067 | 0.0000 | 0.075 | -0.182 | 0.188 | 0.0109 |
| d4 (1/s) | v1 | 0.005 | 1.822 | -6.24 | 0.012 | 4.91 | -9.93 | 7.27 | 1.627 |
| d4 (1/s) | v2 | 0.095 | 0.673 | -1.89 | 0.000 | 2.40 | -9.18 | 3.56 | 0.393 |
| d10 | v1 | 0.362 | 0.339 | 0 | 0.359 | 0.824 | 0 | 0.831 | 0.037 |
| d10 | v2 | 0.368 | 0.337 | 0 | 0.377 | 0.824 | 0 | 0.831 | 0.035 |
| d11 | v1 | 0.421 | 0.425 | 0 | 0.354 | 1 | 0 | 1 | 0.041 |
| d11 | v2 | 0.407 | 0.406 | 0 | 0.354 | 1 | 0 | 1 | 0.038 |
| d12 (1/s) | v1 | -0.027 | 1.311 | -4.78 | 0 | 4.02 | -10.0 | 10.0 | 0.628 |
| d12 (1/s) | v2 | -0.025 | 0.415 | -0.93 | 0 | 0.77 | -5.42 | 3.65 | 0.106 |
| d13 | v1 | 2.68 | 1.70 | 1 | 2 | 8 | 1 | 10 | 0.400 |
| d13 | v2 | 1.77 | 1.14 | 0 | 2 | 5 | 0 | 6 | 0.268 |

"Variacion media entre muestras" = media de \|valor_k - valor_anterior\| entre observaciones consecutivas de una misma trayectoria; mide el parpadeo.

| Medida | v1 | v2 |
|---|---|---|
| filas con d11 = 1 (saturada), total | 23.7 % | 14.3 % |
| d11 = 1 en 14294880 (aerea 4K) | 35.7 % | 8.6 % |
| d11 = 1 en Berkeley (camara en mano) | 33.7 % | 31.2 % |
| d11 = 1 en la trayectoria 1 de 14294880 | 172 de 229 | 10 de 229 |
| filas con \|d12\| >= 5 | 1.8 % | 0.2 % |
| filas con \|d4\| >= 5 | 3.0 % | 0.1 % |
| filas con d13 = 0 | 0 % | 9.8 % |
| media de d13 en Berkeley (74 trayectorias, 34 llegan a 1 s) | 4.00 | 2.10 |

- Trayectoria 1 de 14294880 (`resultados/descriptores_v2/comparacion_14294880_3840_2160_24fps_id1.png`):
  - `d2_med` sigue la tendencia del area sin los picos de una muestra;
  - d3 y d4 quedan cerca de 0 mientras el area es estable, y el pico de -6.5 1/s de d4 (caida brusca de la caja hacia los 20.3 s) baja a -2.4 1/s y aparece unos 0.5 s despues;
  - d11 deja de estar saturada: en v2 vale ~0.93 hasta los 10 s y luego baja de forma gradual hasta ~0.52, mientras que v1 estaba en 1 hasta los 17 s;
  - d12 pasa de oscilar entre -1 y 0.8 despues de los 17 s a valores de +-0.1;
  - d13 sube un segundo despues de que aparece cada trayectoria nueva;
  - d10 no cambia en esta trayectoria.
- En Berkeley el humo es denso y extenso, asi que la region minima casi no cambia la saturacion (33.7 % -> 31.2 %): ahi d11 = 1 refleja humo real alrededor del fuego.
- Siguen quedando colas: d4 llega a -9.18 en Berkeley (id 378, t = 10.5 s). Al inicio de una trayectoria la ventana tiene 2 muestras y la mediana es su promedio, asi que una caida de la caja todavia pesa; el recorte por percentiles de la normalizacion se encarga de estos casos.
- La mediana de 1 s **retrasa unos 0.5 s** la respuesta de d3, d4 y d12 (0.45 s medido en la rampa sintetica). Elimina los parpadeos de menos de ~0.5 s y conserva los cambios sostenidos.

### Ensayo de la normalizacion (no son las estadisticas del modelo)

- Ajuste con 3 videos (1 348 filas; se excluye 14294880) y aplicacion al cuarto (607 filas, ninguna usada en el ajuste).
- En los datos de ajuste: media 0 y desviacion 1 en las 13 columnas.
- En el video excluido, los valores quedan dentro de los limites del ajuste. Por ejemplo, d10 llega a 2.283, que es el p99 del ajuste ya normalizado; d1 baja hasta -1.968, que es el p1.
- p1 y p99 del ajuste para las colas: d3 -0.073 / 0.086 1/s, d4 -2.22 / 2.53 1/s, d12 -2.74 / 0.99 1/s, d13 0 / 6.
- El JSON del ensayo quedo fuera del repositorio. Las estadisticas reales se calcularan en la Fase C, solo con los videos de entrenamiento: `python normalizacion.py ajustar --lista <entrenamiento>.csv`.

### Cambios para la tesis (Cuadro 4.4)

| N | Descriptor | Definicion nueva |
|---|---|---|
| 3 | tasa de crecimiento | `(A~_k - S^2 A~_prev) / dt`, donde `A~_k` es la mediana de las areas de la trayectoria en el ultimo segundo `(t_k - 1, t_k]`, cada area `A_j` llevada al fotograma actual con la escala acumulada `(s_{j+1} ... s_k)^2`; `A~_prev` es el mismo valor en la observacion anterior de la trayectoria, `S` la escala acumulada desde ella y `dt` el tiempo transcurrido. 0 en la primera observacion |
| 4 | crecimiento relativo | d3 / `A~_k` |
| 10 | confianza de humo | maxima confianza de las detecciones de humo (>= 0.25) que tocan la region de humo: la caja de fuego escalada 2 veces alrededor de su centro, con un tamano minimo del 20 % del ancho y del 20 % del alto del fotograma, recortada al fotograma |
| 11 | area de humo | fraccion de esa region cubierta por la union de las cajas de humo |
| 12 | tasa del humo | `(H~_k - H~_prev) / dt`, donde `H~_k` es la mediana de d11 de la trayectoria en el ultimo segundo |
| 13 | regiones activas | numero de trayectorias de fuego presentes en el fotograma con al menos 1 s desde su primera observacion. Incluye a la trayectoria descrita si ya cumplio 1 s; si no, cuenta solo a las demas |

- Retraso: la mediana de 1 s usa solo el pasado (el sistema funciona en linea), asi que las tasas d3, d4 y d12 responden unos 0.5 s despues del cambio real del area. Esto suma al tiempo de alerta y debe mencionarse junto al EWT.
- Normalizacion antes de la CfC: cada descriptor se recorta a los percentiles 1 y 99 y se estandariza con la media y la desviacion del conjunto de entrenamiento; los mismos parametros se aplican en validacion, calibracion, prueba y la demo.

### Archivos generados

- `resultados/descriptores_v2/descriptores.csv`: columnas de v1 mas `d2_med` y `d11_med`.
- `resultados/descriptores_v2/procesados.csv`: `archivo, muestras, trayectorias, filas, segundos`.
- `resultados/descriptores_v2/comparacion.csv`: estadisticas de d3, d4, d10, d11, d12 y d13 por video y en total, v1 y v2.
- `resultados/descriptores_v2/comparacion_14294880_3840_2160_24fps_id1.png`: v1 contra v2 en la trayectoria 1.
- `resultados/descriptores_v2/descriptores_14294880_3840_2160_24fps_id1.png`: los 13 descriptores v2 de la misma trayectoria, en el mismo formato que la grafica de v1.

## Fase A del 80 %, paso 5: videos editados (cortes de escena)

Fecha: 2026-11-05. Rama: `avance-80`. Script nuevo: `cortes.py`. Cambio en `descriptores_v2.py` (con permiso del usuario): opcion de cortes activada por defecto; `--sin_cortes` reproduce el CSV anterior.

### Detector

- Dos indicadores entre muestras consecutivas (10 fps):
  - **`h_k`**: distancia de Bhattacharyya entre histogramas HSV (16 x 4 x 4 bins) del fotograma reducido a 160 px de ancho; va de 0 a 1.
  - **camara**: `valido` de `MovimientoCamara` (mascara de fuego), que es 0 si hay menos de 10 puntos o menos de 50 % de inliers.
- Regla:
  - corte si `h_k >= T_alto` (motivo `histograma`);
  - corte si `T_bajo <= h_k < T_alto` y la camara no es valida (motivo `histograma_y_camara`);
  - la camara sola nunca marca un corte, porque tambien falla con humo denso o poca textura;
  - separacion minima de 1 s entre cortes.
- **Umbrales elegidos: `T_alto = 0.85`, `T_bajo = 0.65`.**
- Al detectar un corte se crean de nuevo los trackers, la camara y `DescriptoresV2`. El CSV de descriptores tiene una columna `tramo` (0, 1, 2...) y cada `(archivo, tramo, id)` es una secuencia aparte. `t` sigue siendo el tiempo del video.
- Video editado = al menos un corte.

### Tiempo al peligro por tramo (`cortes.tau_valido`, `cortes.estado_tramos`)

El peligro es `t_incontrolado`. Cada tramo va desde su inicio hasta el inicio del siguiente.

| Caso | Tramo | tau |
|---|---|---|
| `t_incontrolado` cae dentro del tramo | `normal` | `max(0, t_incontrolado - t_k)` |
| `t_incontrolado` cayo en un tramo anterior | `peligro` | 0 en todo el tramo (valido) |
| `t_incontrolado` cae en un tramo posterior | `enmascarado` | sin tau en todo el tramo |
| el video no llega a Incontrolado | `enmascarado` | sin tau (como en la tesis) |

### Pruebas sinteticas: 12 de 12

- Corte entre dos escenas con textura: h 0.833, detectado; dentro de la toma h <= 0.033.
- Destello de brillo +35 con la camara valida: h 0.387, sin corte.
- Camara no valida sin salto de histograma: sin corte.
- Salto medio con la camara no valida: `histograma_y_camara`; el mismo salto con la camara valida: sin corte.
- Separacion minima de 1 s.
- Los cuatro casos de tau, mas `t_incontrolado` justo en el corte, tau por muestra y un video sin cortes.
- Las primeras imagenes sinteticas (colores planos) daban h ~1 con un cambio de brillo porque sus histogramas son picos. Se reemplazaron por texturas suavizadas, mas parecidas a imagenes reales.

### Busqueda de ejemplos y revision

1. **Busqueda.** `python cortes.py seleccionar` eligio al azar (semilla 42) 60 de los 263 videos "con nombre" (no Pexels ni hash) de hasta 300 s, entre los 389 etiquetados o candidatos (1.00 h de video). `python cortes.py explorar` calculo solo el histograma (248 s en total) y guardo pares antes/despues de los 3 saltos mayores de cada video con `h >= 0.25` en `runs_fuego/cortes/revision/` (no se versiona ni se publica). 40 de los 60 videos no tienen ningun salto de 0.25 o mas.
2. **Videos de ajuste.** Se eligieron 9 videos (8 con cortes y 1 negativo dificil: una luz que se enciende) y se generaron pares de **todos** sus candidatos (`resultados/cortes/lista_ajuste.csv`, `runs_fuego/cortes/revision_ajuste/`).
3. **Revision.** 79 pares, en `resultados/cortes/confirmados.csv` (`ref, archivo, t_corte, h, tipo, nota`). Claude propuso una clasificacion y el usuario la reviso:
   - #5, A13 y A14: no son cortes.
   - **Los dudosos #6, #15 y #24 se resolvieron mirando el video:**
     - #6 es corte: termina una toma y empieza otra desde mas atras;
     - #15 no es corte: es la misma toma y el humo sale con mas fuerza (Claude lo habia marcado como corte desde una tira de fotogramas);
     - #24 es corte: un fundido hacia otra escena con pantalla azul.
   - El paso a negro (#9, #21) cuenta como corte.
4. **Validacion.** Las 4 detecciones de validacion que no se habian revisado (Die Provinzial 6.3, 11.0 y 97.1 s; videoplayback (5) 81.9 s) son cortes reales: Claude los propuso como cortes y el usuario lo confirmo (`ref` B1 a B4, `runs_fuego/cortes/hojas/sin_revisar_A.jpg`).
5. **Total:** 37 cortes y 46 pares sin corte.

### Indicadores y eleccion de umbrales

- Comando: `python descriptores_v2.py --lista resultados/cortes/lista_indicadores.csv --indicadores --salida_cortes resultados/cortes/indicadores`.
- 24 videos:
  - **ajuste:** los 9 de ajuste y los 4 de prueba, 13 063 muestras;
  - **validacion:** los otros 11 videos con pares revisados, 9 654 muestras.
- `python cortes.py umbrales` simula la regla sobre los indicadores (es exacto, porque los indicadores no dependen de los umbrales) para `T_alto` y `T_bajo` de 0.25 a 1.0 cada 0.025. Una deteccion se empareja con el par revisado mas cercano a 0.5 s o menos; si no hay ninguno, cuenta como "sin revisar".

| Grupo (ajuste) | Muestras | h mediana | h p99 | h max | Camara no valida |
|---|---|---|---|---|---|
| dentro de toma (sin revisar) | 12 995 | 0.072 | 0.213 | 0.910 | 7.5 % |
| pares sin corte | 36 | 0.316 | 0.636 | 0.650 | 38.9 % |
| pares con corte | 19 | 0.755 | 0.995 | 0.996 | 100 % |

- El histograma solo no separa los grupos:
  - hay cortes con h de 0.27, sobre todo cambios de encuadre en el mismo lugar;
  - hay pares sin corte con h de 0.81, por luces de emergencia (camara valida), un destello de explosion o humo que sale con fuerza.
- La camara no es valida en casi todos los cortes (100 % en ajuste y 13 de 14 en validacion), pero tampoco en las llamaradas y explosiones.
- Se probo exigir inliers por debajo de 0.20 a 0.40 en vez de `valido = 0`:
  - con `T_bajo` bajo reduce los falsos (de 15 a 5 con `T_bajo` 0.30);
  - no cambia el punto sin falsos, porque #15 (h 0.59, inliers 0.09) y A18 (h 0.61, inliers 0.13) tienen inliers de corte.
  - Se mantuvo la regla con `valido`.

| `T_alto` | `T_bajo` | Ajuste: cortes / falsos / sin revisar | Validacion: cortes / falsos / sin revisar | Videos con cortes marcados (ajuste + validacion) | Videos sin cortes marcados como editados |
|---|---|---|---|---|---|
| 0.85 | 0.25 | 18 de 19 / 15 / 3 | 14 de 14 / 2 / 64 | - | - |
| 0.85 | 0.40 | 16 de 19 / 5 / 0 | 14 de 14 / 0 / 37 | 13 de 13 | 2 de 11 |
| 0.85 | 0.50 | 13 de 19 / 4 / 0 | 11 de 14 / 0 / 22 | - | - |
| **0.85** | **0.65** | **12 de 19 / 0 / 0** | **14 de 18 / 0 / 0** (con B1-B4) | **10 de 13** | **0 de 11** |

**Resultado con los umbrales elegidos (confirmados por el usuario el 2026-11-05):**

| Conjunto | Cortes revisados | Detectados | Perdidos | Falsos | Videos con cortes marcados como editados | Videos sin cortes marcados como editados |
|---|---|---|---|---|---|---|
| Ajuste (13 videos) | 19 | 12 | 7 | 0 | 5 de 7 | 0 de 6 |
| Validacion (11 videos) | 18 | 14 | 4 | 0 | 5 de 6 | 0 de 5 |
| **Total (24 videos)** | **37** | **26** | **11** | **0** | **10 de 13** | **0 de 11** |

- **Precision: 26 de 26 detecciones son cortes reales (100 %, 0 falsos).**
- **Exhaustividad total: 26 de 37 cortes (70.3 %).** Por video: 10 de 13 videos editados quedan marcados (76.9 %), y ningun video sin cortes queda marcado.

- Con 0 falsos en ajuste, el maximo es 12 de 19 cortes. Se logra con `T_bajo` de 0.625 a 0.675 y `T_alto` de 0.70 a 1.0. En validacion los falsos desaparecen con `T_alto >= 0.825`, porque las luces de emergencia de BRANDWEER llegan a h 0.812 con la camara valida.
  - Se eligio el centro: `T_alto = 0.85` (margen sobre 0.812) y `T_bajo = 0.65`.
- **Por que se prioriza no tener falsos:** los falsos cortes caen en explosiones, destellos y humo que sale con fuerza, que son justo los momentos en que el fuego pasa a Incontrolado. Un falso corte ahi reiniciaria el seguimiento en el momento clave y dejaria `t_incontrolado` en un tramo posterior, con lo que se enmascararia el tiempo al peligro de todo lo anterior.
- **Limitacion que se declara:** se pierden los cortes con h bajo:
  - en ajuste, 7: cambios de encuadre en el mismo lugar (h 0.27 a 0.51) y dos cambios de toma en Brannutvikling (h 0.40 y 0.33);
  - en validacion, 4: tres en el recuadro de videoplayback (15) (h 0.41 a 0.49) y uno con h 0.64;
  - 3 de los 13 videos con cortes no quedan marcados como editados (Early Video, PRE-ARRIVAL Dumpster y videoplayback (15)).

### Pruebas con los umbrales elegidos

- **Regresion:** `python descriptores_v2.py --sin_cortes` sobre los 4 videos de prueba da un CSV **identico byte a byte** a `resultados/descriptores_v2/descriptores.csv`.
- **Punta a punta (4 videos de prueba, Brennende Akkus y video (4309))**, salida en `resultados/cortes/prueba/`:
  - 4 videos de prueba: 0 cortes, `editado = no`;
  - video (4309) (la luz que se enciende): 0 cortes;
  - Brennende Akkus: 5 cortes (3.7, 10.6, 13.8, 24.5 y 83.0 s), los mismos 5 confirmados en ese video. Queda en 6 tramos y el seguimiento se reinicia en cada uno.
- Tiempo: igual que sin cortes (el histograma cuesta ~1 ms por muestra).

### Correccion sobre Berkeley

Se reviso un fotograma cada 5 s: es una camara en tierra con zoom y paneos, no un dron, y no tiene cortes de escena. Ver la correccion en el paso 2 del 50 %. Tambien se corrigio la columna `motivo` de `resultados/seguimiento/videos_prueba.csv` y la Evidencia 2 del `README.md`.

### Corrida de la noche (la lanza el usuario)

- Lista: `resultados/cortes/lista_noche.csv` (`python cortes.py listas`):
  - los 389 videos de fuego (200 etiquetados por la IA y 218 candidatos, 29 en comun; 8.69 h);
  - **215 videos sin fuego, uno por cada origen** (semilla 42; 0.47 h; `resultados/cortes/lista_non_fire.csv`), para la calibracion conformal.
  - Incluye la copia `14296658_3840_2160_24fps - Copy.mp4`, que se excluye en la particion (paso 6).
- Ritmos medidos (segundos de proceso por segundo de video): 720p o menos 0.37; 1080p 0.73; 4K 1.12.
- Estimacion: **~6.7 h** (6.3 h los de fuego y 0.35 h los sin fuego).
- Los descriptores de la noche (~0.5 millones de filas, del orden de 100 MB) van a `runs_fuego/` (no se versiona, por el limite de 100 MB de GitHub). `cortes.csv` y `resumen.csv` van a `resultados/cortes/`.

### Archivos generados

- `resultados/cortes/lista_exploracion.csv`, `exploracion.csv`, `exploracion_resumen.csv`: busqueda con solo el histograma.
- `resultados/cortes/lista_ajuste.csv`, `ajuste/`: candidatos de los videos de ajuste.
- `resultados/cortes/confirmados.csv`: 83 pares revisados.
- `resultados/cortes/lista_indicadores.csv`, `indicadores/indicadores.csv` (h, `valido`, inliers y puntos por muestra), `indicadores/umbrales_rejilla.csv`, `indicadores/umbrales_elegidos.txt`.
- `resultados/cortes/prueba/cortes.csv` y `resumen.csv`: corrida de punta a punta.
- `resultados/cortes/lista_non_fire.csv` y `lista_noche.csv`.
- `runs_fuego/cortes/` (no se versiona): pares y hojas de revision.

### Resultado de la corrida de la noche (2026-11-05 20:40 a 2026-11-06 02:48)

- Comando: `python descriptores_v2.py --lista resultados/cortes/lista_noche.csv --salida runs_fuego/descriptores_v2_noche --salida_cortes resultados/cortes --grafico ninguno`.
- **604 de 604 videos procesados**, sin errores, sin repetidos y sin faltantes. Tiempo de proceso: **6.13 h** (6.02 h los de fuego y 0.12 h los sin fuego; se habian estimado 6.7 h).
- `runs_fuego/descriptores_v2_noche/descriptores.csv` (no se versiona): **118.9 MB**, 561 799 filas.

| | Videos | Muestras | Filas de fuego | Trayectorias de fuego `(tramo, id)` | Videos editados | Cortes | Tramos |
|---|---|---|---|---|---|---|---|
| Fuego | 389 | 312 837 | 561 331 | 17 968 | 60 (15.4 %) | 234 | 623 |
| Sin fuego | 215 | 16 855 | 468 | 19 | 4 | 7 | 222 |

- Sin la copia excluida en el paso 6 (que no tenia cortes): 388 videos de fuego, 60 editados, 234 cortes y 622 tramos.
- Motivos de los 241 cortes: 173 `histograma_y_camara` y 68 `histograma`.
- Cortes por video editado:
  - 21 videos con 1 corte, 14 con 2 y 8 con 3;
  - el maximo es `This Can Create a House Inferno!.mp4` (26 cortes en 155 s), seguido de `DUTCH FIREFIGHTERS - CLUBHUIS BRAND AF. 15.mp4` (19 cortes).
- **Falsos del detector en videos sin fuego:** 14 de los 215 videos sin fuego (6.5 %) tienen alguna trayectoria de fuego: 19 trayectorias y 468 filas, en 1 116 de las 16 855 muestras de esos videos. El mayor es `2030845629_clip_003.mp4`, con 142 filas.
- **Cortes en videos sin fuego** (revisados a ojo por Claude, no por el usuario; `runs_fuego/cortes/hojas/non_fire_cortes.jpg`): de 7, solo 1 es un corte (paso desde negro al inicio de `1898933216_clip_001.mp4`). Los otros 6 son cambios de luz en escenas que no estaban en el ajuste:
  - luces de un concierto (`1694496253_clip_002.mp4`, 2);
  - relampagos (`2382573000_clip_001.mp4`, 2);
  - una camara dentro del agua (`2568999255_clip_001.mp4`, 2).
  - La precision del 100 % medida en videos de fuego no se mantiene con destellos fuertes de otro tipo. En los videos de calibracion el efecto es menor, porque se usa el riesgo maximo por video.
- **Videos de fuego sin ninguna fila (6):** 13567986, 14196334, 14283144, 17469008, 18153408 y 19539603. El detector no ve fuego en ellos; sus curvas de `preanotar.py` ya tenian una presencia de 0 a 0.07. La IA los etiqueto como Controlado (5) o los descarto (1). No aportan secuencias.
- **Videos con menos muestras que 10 por segundo (4), sin falla:**
  - `fire20_clip_013`, `fire23_clip_044` y `fire26_clip_004` son clips de 7 a 9 fps;
  - `12220604_1080_1920_30fps.mp4` tiene 830 fotogramas aunque los metadatos dicen 861.

## Fase A del 80 %, paso 6: duplicados excluidos

Fecha: 2026-11-06. Rama: `avance-80`. Script nuevo: `duplicados.py`.

### Que se hizo

- La copia `14296658_3840_2160_24fps - Copy.mp4` es **identica byte a byte** al original `14296658_3840_2160_24fps.mp4`: 81 521 877 bytes y el mismo SHA-256 (`df2be7afb31634d1...`). Sus 1 034 filas de descriptores de la corrida de la noche tambien son identicas.
  - El original esta en la prueba humana (`resultados/prueba_humana.csv`).
  - Ambos estan en las etiquetas de la IA y en `seleccion_etiquetado.csv`.
- `duplicados.py` busca duplicados exactos en los 2 468 videos de DetectiumFire (`fire` y `non_fire`): agrupa por tamano y calcula SHA-256 solo dentro de cada grupo.
  - En cada grupo conserva el que esta en la prueba humana; si no hay ninguno, el que no tiene nombre de copia y despues el de numero menor.
  - Escribe `resultados/excluidos.csv` (`archivo, conservado, motivo, sha256, tamano, en_lista`).
- Resultado: **10 grupos de duplicados exactos y 34 archivos excluidos.**
  - El unico excluido que esta en los 604 de la noche es la copia.
  - Los otros 9 grupos son clips de fuego de 6 s (`video (4209)` a `video (6700)`, hasta 7 copias por grupo) que no estan en ninguna lista actual. Quedan registrados por si se agregan despues.
- **Regla para el paso 10:** los archivos de `excluidos.csv` no entran en ningun conjunto (entrenamiento, validacion, calibracion ni prueba). La funcion `duplicados.cargar_excluidos()` devuelve ese conjunto. El original de la copia sigue en la prueba.

### Comando

```
python duplicados.py
```

### Pendiente para el paso 10 (particion por origen)

El campo `origen` de `metadatos_videos.csv` no sirve tal cual para particionar:

- **La copia y su original tienen origenes distintos:** `"14296658_3840_2160_24fps "` (con un espacio al final) y `"14296658_3840_2160_24fps"`. Sin la exclusion, la copia y el original de prueba habrian podido quedar en conjuntos distintos.
- **27 videos (26 con nombre y la copia) tienen el origen cortado en el primer " - " y con espacios al final.**
  - Videos de un mismo canal quedan en el mismo origen sin ser el mismo video (por ejemplo, `"BRANDWEER "` agrupa 5 videos distintos).
  - Un mismo canal queda partido en dos origenes (`"DUTCH FIREFIGHTERS "` y `"DUTCH FIREFIGHTERS  "`, con dos espacios).
- **Casi duplicados con origenes distintos:**
  - `2016 April - Fire Investigation 1B - Milpitas CA.mp4` (278 s) y `April 2016 - Fire Investigation 1B - Milpitas CA.mp4` (137 s) parecen el mismo ensayo con otra edicion; sus origenes son `"2016 April "` y `"April 2016 "`;
  - `March 2016 - Fire Investigation 1A - Milpitas CA.mp4` y su version `(1)` no son identicas byte a byte.
- Hay que decidir en el paso 10 como se agrupan estos origenes, normalizandolos y uniendo a mano los casos como el de Milpitas.

## Cierre de la Fase A del 80 % (pasos 1 a 6)

| Paso | Que se hizo | Resultado principal | Archivos |
|---|---|---|---|
| 1 | d3, d4 y d12 sobre la mediana de 1 s (solo pasado, con escala acumulada de la camara) | desviacion de d3: 0.084 -> 0.0215; d4: 1.82 -> 0.67; d12: 1.31 -> 0.41; retraso de 0.45 s | `descriptores_v2.py` |
| 2 | region de humo minima del 20 % del ancho y del alto | d11 saturada: 23.7 % -> 14.3 % (vista aerea: 35.7 % -> 8.6 %; humo denso real: 33.7 % -> 31.2 %) | `descriptores_v2.py` |
| 3 | d13 cuenta solo trayectorias de al menos 1 s | media de d13: 2.68 -> 1.77 | `descriptores_v2.py` |
| 4 | normalizacion: recorte por p1 y p99 y estandarizacion con estadisticas de entrenamiento | funcion lista y probada; se ajustara en la Fase C | `normalizacion.py` |
| 5 | cortes de escena (histograma HSV + camara; 0.85 y 0.65), tramos y tau por tramo | 26 de 37 cortes revisados, 0 falsos en los videos de fuego; 60 de 388 videos de fuego editados, 622 tramos | `cortes.py`, `descriptores_v2.py` |
| 6 | duplicados exactos excluidos | la copia de 14296658 y 33 clips repetidos; solo la copia estaba en los 604 | `duplicados.py`, `resultados/excluidos.csv` |

- **Verificacion de que no se rompio nada:**
  - con las correcciones en 0, v2 reproduce v1 (1 955 filas, 0 diferencias);
  - `--sin_cortes` reproduce v2 byte a byte;
  - 13 pruebas sinteticas de los descriptores y 12 de los cortes y tau.
- **Datos listos para la Fase C:**
  - descriptores v2 de 603 videos (388 de fuego, despues de excluir la copia, y 215 sin fuego), con contexto de camara, `tramo` y cortes;
  - `runs_fuego/descriptores_v2_noche/descriptores.csv`, que no se versiona;
  - `resultados/cortes/resumen.csv`, `cortes.csv` y `resultados/excluidos.csv`.
- **Limitaciones que se declaran:**
  - la mediana retrasa las tasas unos 0.5 s;
  - el detector de cortes pierde cortes entre encuadres parecidos (exhaustividad 26 de 37) y da falsos con destellos fuertes en escenas sin fuego (6 de 7 en los clips sin fuego, revision de Claude);
  - 6 videos de fuego no tienen trayectorias;
  - 14 de 215 videos sin fuego tienen falsos del detector.
- **Pendiente para la Fase C (paso 10):** normalizar y agrupar los origenes antes de particionar.

## Fase B del 80 %, paso 7: etiquetado con IA de los candidatos con crecimiento (en curso)

Fecha: 2026-11-06. Rama: `avance-80`. Script nuevo: `etiquetar_ia_v2.py`. Prompt nuevo: `prompts/prompt_severidad_v2.txt`; el de la fase 1 (`prompts/prompt_severidad.txt`) no se modifica.

### Decisiones (2026-11-06)

- **Videos:** los 218 candidatos con crecimiento sostenido, menos los 8 que estan en `resultados/prueba_humana.csv` y menos los de `resultados/excluidos.csv` (ninguno). Quedan **210**.
- **Pasada 1:**
  - 48 fotos uniformes en todo el video, sin usar las reglas, en dos imagenes de 24 (`tiras_v2/<video>_a.jpg` y `_b.jpg`, 6 x 4, fotos de 320 px de ancho);
  - la IA da el nivel inicial y la primera foto de cada cambio;
  - lotes de 10 videos (20 imagenes), en orden barajado con semilla 42 (`prompts/lotes_v2/p1_lote_XX.csv`).
- **Pasada 2:**
  - solo para los cambios marcados en la pasada 1 y solo si la separacion entre fotos es mayor de 1 s;
  - tira densa (una foto cada 0.5 s, maximo 24) entre las fotos `j-2` y `j+1`, mas 2 s;
  - se reporta cuantas veces la IA responde que no se ve el cambio.
- **Modelo:** el mismo de la fase 1, Claude Sonnet 5.5, con subagentes nuevos sin contexto.
  - Cada JSON crudo guarda el identificador del modelo, y `etiquetas_ia_v2.csv` tiene una columna con el modelo.
  - Las respuestas crudas van a `resultados/ia_crudo_v2/`.
- **Consistencia con la fase 1:** se mide en los 21 videos que tienen las dos etiquetas (los otros 8 de los 29 estan en la prueba humana), solo para el nivel inicial y el nivel maximo. Los instantes los mide el paso 8. `resultados/etiquetas_ia.csv` no se modifica.
- **Paso 8 (prueba de la localizacion temporal):**
  - 30 videos elegidos al azar con semilla fija entre los 210: 25 con al menos un cambio segun la IA v2 y 5 sin cambio, mezclados en una sola lista (`resultados/prueba_cambios.csv`), para que el anotador no sepa cuales tienen cambio y para medir si la IA se pierde cambios;
  - se etiquetan a ciegas con `etiquetar_v3.py --ciego` (nuevo; `etiquetar_v2.py` no se modifica porque con el se hizo la validacion de los 40).

### Ensayo de la tanda 1 y cambios al protocolo (2026-11-06)

- **Ensayo:** la tanda 1 (lotes 1 a 5, 50 videos) se corrio con el agente general de Claude Code (alias `sonnet`).
  - Los 50 JSON fueron validos y los 50 reportaron el modelo `claude-sonnet-5-5`.
  - **4 de 5 subagentes ejecutaron comandos prohibidos por el prompt:** `ls`, copias de tiras a una carpeta temporal y un script de Python para escribir los JSON (este ultimo tambien listo la carpeta de respuestas, donde se ven los nombres de las respuestas de los otros subagentes).
  - Parte del problema vino del script: 11 de las 210 tiras con nombres no ASCII se guardaron con el nombre mal codificado (`cv2.imwrite` en Windows). Se corrigio (`cv2.imencode` y escritura binaria) y se renombraron las 22 imagenes.
  - No se pudo comprobar que cada subagente abriera las 2 imagenes de cada video: los registros de los subagentes no quedaron disponibles, y en 3 lotes el numero de llamadas a herramientas fue menor que el minimo esperado.
  - Las 50 respuestas del ensayo se movieron a `resultados/ia_crudo_v2_ensayo/` y **no se usan como etiquetas**.
- **Cambios respecto de la fase 1, y su motivo:**
  - **Tipo de agente:** en lugar del agente general, un agente propio, `anotador_severidad` (`.claude/agents/anotador_severidad.md`, modelo `sonnet`), **con solo las herramientas Read y Write**. Asi el subagente no puede ejecutar comandos ni listar carpetas. Su mensaje de sistema solo dice que siga el archivo de prompt que se le indique y que solo puede leer y escribir archivos. Motivo: en el ensayo, 4 de 5 subagentes ejecutaron comandos prohibidos. `.claude/` no se versiona (igual que `CLAUDE.md`).
  - **Prueba de lectura:** el JSON de la pasada 1 agrega `t_foto_24` y `t_foto_48`, los instantes escritos en la ultima foto de cada imagen. Se comparan con `tiras_v2/tiempos.csv` con una tolerancia de 0.1 s; si no coinciden, el video se repite una vez y, si vuelve a fallar, se marca. Mientras corren los subagentes, `tiempos.csv` se saca de la carpeta del proyecto.
  - Las definiciones de los niveles y las 7 reglas del prompt v2 son identicas, letra por letra, a las del prompt de la fase 1 (verificado con `diff`).
- La tanda 1 se repite con el protocolo final, y se reporta el acuerdo entre el ensayo y la repeticion.

### Estado de la pasada 1 (2026-11-06)

- Lanzamiento por tandas de 5 lotes, con el agente `anotador_severidad` y `tiras_v2/tiempos.csv` fuera del proyecto mientras corren los subagentes. Cada tanda se revisa con `python etiquetar_ia_v2.py revisar --archivos_lote ...`.
- **Repeticion de la tanda 1** (lotes 1 a 5), **tanda 2** (6 a 10) y **tanda 3** (11 a 15):
  - 150 de 150 JSON validos;
  - modelo `claude-sonnet-5-5` en los 150;
  - `t_foto_24` y `t_foto_48` correctos en los 150;
  - cada subagente uso 33 herramientas (prompt, lote, 20 imagenes, 10 escrituras y la respuesta final).
- **Tanda 4** (lotes 16 a 20): los 5 subagentes se detuvieron al alcanzar el limite de uso de 5 horas de Claude (error 429).
  - Quedaron 48 JSON, todos validos y con los controles correctos.
  - En el lote 19 faltan 2 videos (`Brannutvikling.mp4` y `240_F_1735655681_pTyPr0KLjIg3vG6BCfyzokPM3pPYMI5B_ST.mp4`). No quedaron archivos incompletos.
  - Los 2 que faltan estan en `prompts/lotes_v2/p1_repetir_faltantes_t4.csv`.
- **Pendiente:** esos 2 videos y el lote 21 (10 videos). En total faltan **12 de 210**.
- **Acuerdo entre el ensayo y la repeticion de la tanda 1** (48 videos no descartados en ambos):

| Medida | Acuerdo |
|---|---|
| nivel inicial | 45 de 48 (93.8 %); 2 de incipiente pasan a controlado y 1 de incontrolado a incipiente |
| nivel maximo | 45 de 48 (93.8 %) |
| hay cambio | 46 de 48 (95.8 %) |
| foto del paso a incontrolado (11 marcados por ambos) | 8 en la misma foto, 2 a 1-2 fotos y 1 a 6 fotos |
| foto del paso a incipiente | 1 marcado por ambos, en la misma foto |

- **Resultado parcial: 198 videos validos de los lotes 1 a 20**
  - 4 descartados:
    - `videoplayback (5).mp4`: compilacion de ensayos con cortes y mosaicos;
    - `240_F_1127423393...`: colada de metal con aspecto de render 3D;
    - `Living room fires with and without a fire sprinkler (Timecode).mp4`: escenas cambiantes con cortes;
    - `700_F_1623872802...`: erupcion volcanica.
  - Nivel inicial de los 194 no descartados: controlado 82, incipiente 71, incontrolado 41.
  - Cambios:
    - 131 sin cambio;
    - 48 solo a incontrolado (desde incipiente);
    - 4 solo a incipiente;
    - 10 con ambos cambios;
    - 1 directo de controlado a incontrolado.
  - **En total, 63 de 194 videos (32.5 %) tienen algun cambio.**
- El resumen final de los 210 y la pasada 2 se haran cuando terminen los 12 que faltan.

### Cierre de la pasada 1, pasada 2 y etiquetas v2 (2026-11-11)

- **Pasada 1 completa: 210 de 210 JSON validos.**
  - Los 2 videos que faltaban del lote 19 y el lote 21 se corrieron con el mismo agente y los mismos controles.
  - Los 210 tienen el modelo `claude-sonnet-5-5` y `t_foto_24` y `t_foto_48` correctos.

| Resultado (210 videos) | Videos |
|---|---|
| descartados | 4 (compilacion con mosaicos, colada de metal con aspecto de render, escenas con cortes del video del rociador, erupcion volcanica) |
| nivel inicial (206 no descartados) | controlado 89, incipiente 73, incontrolado 44 |
| sin cambio | 142 |
| controlado -> incipiente (solo) | 4 |
| incipiente -> incontrolado | 49 |
| controlado -> incontrolado (con paso por incipiente: 10; directo: 1) | 11 |
| **con algun cambio** | **64 (31.1 % de 206)** |

- **Pasada 2:**
  - **Cambios:** de los 74 cambios marcados en la pasada 1, 15 estan en videos con separacion de 1 s o menos y conservan el instante de la pasada 1. Para los otros 59 se armaron tiras densas: mediana de 0.53 s entre fotos, entre 11 y 24 fotos por tira, en `tiras_v2/precision/` (no se versiona).
  - **Lotes:** 3 lotes de 20 (`prompts/lotes_v2/p2_lote_XX.csv`, sin los instantes de la ventana para que la prueba de lectura sea valida). Mismo agente; tiempos de las tiras fuera del proyecto mientras corrian.
  - **Prueba de lectura:** el JSON agrega `t_foto_primera` y `t_foto_ultima`. Las definiciones y las reglas del prompt no cambiaron.
  - **Resultado:** 59 de 59 respuestas validas, con el modelo y los tiempos correctos. **En 47 se preciso el instante; en 12 (20 %) la IA respondio que no se ve el cambio** y se conserva el instante de la pasada 1.
  - **Incidencia:** un subagente escribio un archivo de mas con el nombre mal escrito y el mismo contenido. Se saco de `resultados/ia_crudo_v2/precision/`.
- **`resultados/etiquetas_ia_v2.csv`** (anotador `ia_sonnet_v2`), con columnas:
  - `modelo` (reportado) y `alias` (`sonnet`);
  - `agente` (`anotador_severidad`);
  - `sep_p1_s`, `sep_p2_incipiente_s` y `sep_p2_incontrolado_s`;
  - `origen_t_incipiente` y `origen_t_incontrolado`.
- **Origen de los 75 instantes:** 48 de la pasada 2 (47 tiras; el cambio directo a incontrolado da los dos instantes), 15 de la pasada 1 por separacion corta y 12 de la pasada 1 porque la pasada 2 no vio el cambio.
- `resultados/etiquetas_ia.csv` no se modifico.

### Consistencia con la fase 1 (`resultados/consistencia_ia_v1_v2.md`)

- 21 videos con las dos etiquetas. v2 descarta 1 (la colada de metal), asi que quedan **20 sin descartar en ambos**.

| Medida | Acuerdo | Kappa ponderado lineal |
|---|---|---|
| nivel inicial | 15 de 20 (75.0 %) | 0.615 |
| nivel maximo | 16 de 20 (80.0 %) | 0.750 |

- Videos con algun cambio: 2 en v1 y 2 en v2 (de 20). Los instantes no se comparan; eso lo mide el paso 8.

## Fase B del 80 %, paso 8: preparacion (2026-11-11)

- **Lista:** `python etiquetar_ia_v2.py prueba_cambios` genera `resultados/prueba_cambios.csv` (solo la columna `archivo`, orden aleatorio, semilla 42).
  - 25 videos con cambio segun la IA v2, estratificados por tipo de cambio (minimo 5 por tipo si hay; el resto, proporcional), mas 5 sin cambio.
- **Primera lista, sin tope de duracion: 98.2 min** (mediana de 146 s, maximo de 1 084 s; 4 videos pasaban de 300 s y sumaban 43.7 min). Pasaba de 1 h y no se uso.
- **Regla de duracion (decision del usuario, 2026-11-11): tope de 300 s por video, salvo en el estrato de controlado a incipiente, que se incluye completo por ser el mas escaso (solo hay 4).**
  - Parametros: `--tope_s 300` y `--sin_tope controlado_a_incipiente` (valores por defecto). El tope se aplica antes del sorteo: en los demas estratos solo se sortea entre los videos de 300 s o menos. Misma semilla (42) y mismos estratos.
  - **Limitacion que se declara: el tope favorece los videos cortos en los demas estratos** (incipiente a incontrolado, controlado a incontrolado y sin cambio). El error en segundos que se mida vale para videos de hasta 300 s en esos estratos.
  - Control: con `--tope_s 0` el script reproduce byte a byte la primera lista.
- **Lista final: 30 videos, 53.1 min** (mediana de 76 s, maximo de 364 s).

| Estrato | Disponibles | Con el tope | Elegidos | Duracion | Maximo |
|---|---|---|---|---|---|
| controlado a incipiente (sin tope) | 4 | 4 | 4 | 10.6 min | 364 s |
| incipiente a incontrolado | 49 | 46 | 15 | 24.2 min | 260 s |
| controlado a incontrolado | 11 | 8 | 6 | 14.0 min | 220 s |
| sin cambio | 142 | 137 | 5 | 4.3 min | 149 s |
| **Total** | 206 | 195 | **30** | **53.1 min** | 364 s |

  - Un solo video pasa de 300 s (364 s), del estrato de controlado a incipiente.
  - El tope deja fuera del sorteo a 11 videos: 3 de incipiente a incontrolado, 3 de controlado a incontrolado y 5 sin cambio.
  - Verificado: 30 videos distintos, todos existen en disco, ninguno esta en `resultados/prueba_humana.csv` y ninguno esta descartado por la IA.
- Comando para etiquetar: `python etiquetar_v3.py --anotador aleCa_cambios --lista resultados/prueba_cambios.csv --ciego`.
- **`etiquetar_v3.py --ciego` (nuevo; `etiquetar_v2.py` no se modifica):**
  - reproduce el video completo: espacio para reproducir y pausar, flechas para saltar 1 s y 5 s, `,` y `.` para retroceder o avanzar un fotograma, y clic en la linea de tiempo para ir a un instante;
  - marca el nivel inicial (1/2/3), el instante exacto de cada cambio (i/u) y el tramo util (a/f);
  - no carga nada de la IA, de las reglas ni de las curvas;
  - escribe en `etiquetas.csv` con un anotador propio. Rechaza un anotador que ya tenga etiquetas de videos fuera de la lista. Las filas existentes quedan identicas (probado en una copia: las 40 filas sin cambios);
  - registra el tiempo por video en `resultados/tiempos_etiquetado.csv` (`archivo, anotador, segundos_abierto, reproducciones, guardado, fecha`);
  - no necesita tiras.

### Regla para el paso 10 (particion)

- Los 30 videos de `resultados/prueba_cambios.csv` y los 40 de `resultados/prueba_humana.csv` son prueba.
- **Los videos que compartan origen con cualquiera de esos 70 tampoco entran en entrenamiento, validacion ni calibracion.**
- El origen se toma ya normalizado y agrupado (ver "Pendiente para el paso 10" en el paso 6).

## Fase B del 80 %, paso 8: validacion humana de los cambios de nivel (2026-11-11)

Fecha: 2026-11-11. Rama: `avance-80`. Script nuevo: `validar_cambios.py` (importa `leer`, `num` y `kappa` de `comparar_etiquetas.py`, que no se modifica).

### Que se hizo

- El usuario etiqueto a ciegas los 30 videos de `resultados/prueba_cambios.csv` con `python etiquetar_v3.py --anotador aleCa_cambios --lista resultados/prueba_cambios.csv --ciego`. Las 30 filas estan en `etiquetas.csv`; las 40 filas de `aleCa` quedaron iguales.
- `validar_cambios.py` compara esas etiquetas con `resultados/etiquetas_ia_v2.csv` y escribe `resultados/validacion_cambios.md` (tablas completas, video por video).
- **No se modifico ninguna etiqueta.** Se comprobo con SHA-256 antes y despues: `etiquetas.csv`, `resultados/etiquetas_ia_v2.csv`, `resultados/etiquetas_ia.csv` y `resultados/tiempos_etiquetado.csv` quedaron identicos.
- El humano descarto 1 video (`Brennende Akkus.mp4`; la IA lo deja en incipiente con cambio a incontrolado en 5.67 s). Los cambios, los instantes y el tramo util se comparan en los otros 29.

### Comando

```
python validar_cambios.py
```

Parametros: `--anotador aleCa_cambios`, `--lista resultados/prueba_cambios.csv`, `--ia resultados/etiquetas_ia_v2.csv`, `--pausa_s 1000` (un video abierto mas de 1 000 s se cuenta como pausa).

### Numeros

**Nivel inicial**

| Videos | Acuerdo | Kappa ponderado lineal |
|---|---|---|
| los 30 | 22 de 30 (73.3 %) | 0.469 |
| sin el descartado (29) | 21 de 29 (72.4 %) | 0.460 |

- De los 8 desacuerdos, en 6 la IA pone controlado y el humano incipiente; en 1 la IA pone incipiente y el humano controlado; en 1 la IA pone incontrolado y el humano incipiente.
- El humano no uso incontrolado como nivel inicial en ningun video (8 controlado, 22 incipiente). La IA: 13 controlado, 16 incipiente, 1 incontrolado.

**Deteccion de cambios (29 videos)**

| Cambio | Ambos | Solo la IA | Solo el humano | Ninguno |
|---|---|---|---|---|
| a incipiente | 3 | 7 | 1 | 18 |
| a incontrolado | 15 | 5 | 1 | 8 |
| algun cambio en el video | 17 | 7 | 0 | 5 |

- De los 24 videos no descartados con cambio segun la IA, el humano marca algun cambio en 17 (71 %) y ninguno en 7.
- De los 7 cambios a incipiente que marca solo la IA, en 6 el humano ya puso incipiente como nivel inicial: es el mismo desacuerdo del nivel inicial, no un cambio inventado.
- **Los 5 videos sin cambio segun la IA: el humano tampoco marca cambio en ninguno.** En 4 coincide ademas el nivel inicial. La IA no perdio ningun cambio en esta muestra (son solo 5 videos).

**Error en segundos cuando ambos marcan el mismo cambio (18 instantes)**

Error = instante de la IA menos instante del humano.

| Grupo | Instantes | Mediana del error absoluto | Media del error absoluto | Mediana con signo | Dentro de 1 s | Dentro de 2 s | Dentro de 5 s | La IA marca antes |
|---|---|---|---|---|---|---|---|---|
| todos | 18 | 31.8 s | 54.2 s | -31.8 s | 2 | 3 | 4 | 15 |
| a incipiente | 3 | 3.6 s | 81.5 s | -3.6 s | 1 | 1 | 2 | 2 |
| a incontrolado | 15 | 39.8 s | 48.8 s | -39.8 s | 1 | 2 | 2 | 13 |
| origen: pasada 2 (tira densa) | 11 | 39.8 s | 60.0 s | -39.8 s | 1 | 1 | 2 | 10 |
| origen: pasada 1 (fotos separadas 1 s o menos) | 3 | 1.4 s | 2.4 s | +1.4 s | 1 | 2 | 2 | 1 |
| origen: pasada 1 (la pasada 2 no ve el cambio) | 4 | 61.9 s | 77.3 s | -61.9 s | 0 | 0 | 0 | 4 |

- Los 3 instantes de la pasada 1 vienen de videos de 12 a 36 s, asi que su error esta acotado por la duracion. No sirven para decir que la pasada 1 es mas precisa que la pasada 2.
- Los instantes de la pasada 2 vienen de videos de 53 a 364 s (mediana de 152 s).
- Error maximo: 240.7 s (cambio a incipiente en un video de 364 s).

**Tramo util (diferencia = humano menos IA)**

| Extremo | Videos | Mediana de la diferencia absoluta | Media de la diferencia absoluta | Dentro de 1 s | Dentro de 2 s | Dentro de 5 s | Dentro de una foto de la pasada 1 |
|---|---|---|---|---|---|---|---|
| inicio | 26 | 0.18 s | 3.73 s | 20 | 20 | 22 | 23 |
| fin | 25 | 2.10 s | 1.99 s | 10 | 12 | 24 | 16 |

- El humano dejo vacio el inicio util en 3 videos y el fin util en 4 (de 29).
- Las diferencias grandes del inicio son 3 videos: +41.4 s, -32.4 s y +7.4 s.

**Tiempo de etiquetado (`resultados/tiempos_etiquetado.csv`)**

- Tiempo total con la ventana abierta: 215.8 min. Dos videos estuvieron abiertos 3 488 s y 8 000 s (pausas con la ventana abierta).
- **Sin esos dos: 24.4 min en 28 videos** (mediana de 20.6 s por video, media de 52.2 s, de 4.8 s a 382.7 s). Esos 28 videos duran 51.7 min.
- **En 22 de los 28 la ventana estuvo abierta menos tiempo que la duracion del video** (mediana de tiempo abierto entre duracion: 0.29). Esos videos no se vieron completos a velocidad normal; se recorrieron con saltos o con la linea de tiempo.
- De los 15 videos con cambio del humano y sin pausa, 11 estan en ese caso.
- 18 de los 30 videos tienen una sola reproduccion.
- Sesion: primer guardado a las 18:38:43 y ultimo a las 22:08:13.

### Lectura de los resultados

- **Los niveles coinciden de forma moderada** (73 %, kappa 0.47) y la IA no pierde cambios en los videos que dejo sin cambio.
- **Los instantes no coinciden.** La mediana del error es de 32 s y solo 4 de 18 quedan dentro de 5 s. El desacuerdo es sistematico: en 13 de 15 cambios a incontrolado la IA marca antes que el humano, con una mediana de 40 s.
- **No es un problema de resolucion.** La pasada 2 usa una foto cada 0.5 s y aun asi su mediana es de 40 s. Es una diferencia de criterio: la IA llama incontrolado a un estado que el humano todavia considera incipiente. Coincide con los otros dos resultados: la IA pone controlado al inicio en 6 videos donde el humano pone incipiente, y marca un cambio a incontrolado en 5 videos que el humano deja en incipiente.
- **Las 12 respuestas de "no se ve el cambio" de la pasada 2 eran una senal valida:** los 4 instantes de ese grupo que se pudieron comparar tienen el peor error (mediana de 62 s, ninguno dentro de 5 s).

### Limitaciones que se declaran

- Un solo anotador humano y 18 instantes comparados (solo 3 a incipiente).
- La lista tiene un tope de 300 s por video en tres de los cuatro estratos.
- El protocolo pedia reproducir cada video. El registro de tiempos muestra que 22 de 28 videos no se vieron completos, asi que el instante humano tampoco es una referencia exacta. Con estos datos no se puede decir cual de los dos instantes es el correcto; solo que difieren.
- Las etiquetas humanas no se modifican despues de esta comparacion.

### Consecuencias para la Fase C (pendiente de decision del usuario)

- El tiempo al peligro (`tau = t_inc - t`) y los segmentos se entrenarian con los instantes de la IA y se evaluarian con los del humano. Con un adelanto mediano de 40 s en el cambio a incontrolado, el MAE de `tau` y el error de los cambios tendrian ese piso aunque el modelo fuera perfecto.
- El mismo adelanto haria que la alerta parezca mas anticipada frente a la etiqueta humana (EWT inflado). Hay que declararlo o corregirlo.
- Opciones: (1) mantener el protocolo (entrenar con la IA, probar con el humano) y declarar el sesgo; (2) que el humano etiquete tambien los videos de entrenamiento con cambio (39 llegan a incontrolado desde otro nivel); (3) reportar las metricas de prueba contra las dos referencias, humano e IA.

### Archivos

- `validar_cambios.py` (nuevo).
- `resultados/validacion_cambios.md` (nuevo).
- `etiquetas.csv` (30 filas nuevas del anotador `aleCa_cambios`).
- `resultados/tiempos_etiquetado.csv` (nuevo, lo escribe `etiquetar_v3.py`).

## Fase B del 80 %, paso 8: revision de las diferencias entre el humano y la IA (2026-11-12)

Fecha: 2026-11-12. Rama: `avance-80`. Script nuevo: `revisar_cambios.py`. Script modificado a pedido del usuario: `etiquetar_v3.py` (opcion `--revisar`).

### Que se hizo

- `revisar_cambios.py` compara las etiquetas de `aleCa_cambios` con `resultados/etiquetas_ia_v2.csv` en los 29 videos no descartados de `resultados/prueba_cambios.csv` y genera una hoja de comparacion por cada video con alguna diferencia: nivel inicial distinto, un cambio que marca solo uno, o un instante con mas de 5 s de diferencia (`--umbral_s 5`).
- Cada hoja lleva arriba el nivel inicial, los instantes y el tramo util de cada uno y el motivo de la IA, y una fila de 4 fotogramas por diferencia, cada uno con su tiempo real y quien marco ese instante:
  - instante que difiere: instante del humano, instante de la IA y 2 puntos intermedios;
  - cambio que marca solo uno: un punto antes, el instante marcado, un punto despues y el final del tramo util;
  - nivel inicial distinto: 4 fotogramas desde el inicio del tramo util comun hasta el primer cambio marcado por cualquiera de los dos.
- `etiquetar_v3.py --revisar`: modo de solo lectura. Muestra en la linea de tiempo dos bandas (humano e IA) con sus marcas en colores distintos y una leyenda arriba con los niveles, los instantes y el motivo de la IA. No llama a `guardar_etiquetas` ni a `registrar_tiempo`; las teclas de etiquetar y de guardar no hacen nada. Tecla nueva `m`: ir a la siguiente marca. Solo abre videos que el anotador ya etiqueto, para no romper el etiquetado a ciegas. No se combina con `--ciego`.
- **No se modifico ninguna etiqueta.** SHA-256 de `etiquetas.csv`, `resultados/etiquetas_ia_v2.csv`, `resultados/etiquetas_ia.csv` y `resultados/tiempos_etiquetado.csv` identicos antes y despues.
- **El modo `--ciego` de `etiquetar_v3.py` no cambio de comportamiento.** Se corrio la misma sesion simulada (42 teclas y clics sobre 3 videos, con las funciones de ventana de OpenCV reemplazadas) en la version anterior y en la nueva: las 42 imagenes dibujadas son identicas pixel a pixel y las filas escritas en las etiquetas y en los tiempos son iguales. En el codigo de ese modo solo cambio la organizacion: el dibujo del fotograma, de la banda de niveles y de los segundos paso a las funciones `cuadro`, `banda` y `segundos`, y la busqueda de videos a `buscar_videos`.

### Comandos

```
python revisar_cambios.py
python etiquetar_v3.py --anotador aleCa_cambios --lista resultados/revision_cambios.csv --revisar
```

### Numeros

- Videos comparados: 29 (1 descartado por el humano). Con alguna diferencia: 23. Coinciden en todo: 6.
- Con un instante que difiere mas de 5 s: 14 videos (14 instantes). Solo con cambios marcados por uno: 8. Solo con nivel inicial distinto: 1.
- Nivel inicial distinto: 8 videos. Cambios marcados por uno solo: 14.
- La tabla completa, ordenada por la diferencia mas grande, esta en `resultados/revision_cambios.md`.

### Decisiones

- Las hojas no se versionan ni se publican: llevan fotogramas con marcas de agua y 4 videos `700_F_*`.
- Las etiquetas humanas no se modifican despues de la comparacion. La revision sirve para entender la diferencia de criterio, no para corregir etiquetas.

### Archivos

- `revisar_cambios.py` (nuevo).
- `etiquetar_v3.py` (opcion `--revisar`).
- `resultados/revision_cambios.md` (tabla de los 29 videos).
- `resultados/revision_cambios.csv` (los 23 videos con diferencias, en orden; sirve como `--lista`).
- `runs_fuego/revision_cambios/` (23 hojas, no versionadas).

## Fase B del 80 %, paso 8: nivel segundo a segundo (2026-11-12)

- `validar_cambios.py` tiene una seccion nueva (6) en `resultados/validacion_cambios.md`; las secciones 1 a 5 quedaron identicas. Parametros nuevos: `--paso_s 1` y `--margen_s 5`.
- Tramo util comun: del mayor de los dos inicios al menor de los dos fines; un inicio o fin vacio es el inicio o el fin del video.
- **2 911 s comparados en 29 videos. Mismo nivel en 1 707 s (58.6 %). Kappa ponderado lineal por segundo: 0.426.** Las muestras de un mismo video no son independientes.
- La IA pone un nivel mayor que el humano en 1 101 s (37.8 %) y un nivel menor en 103 s (3.5 %).
- Acuerdo por video: media 62.1 %, mediana 64.3 %; 5 de 29 videos con 90 % o mas.

| humano / IA (s) | controlado | incipiente | incontrolado | total |
|---|---|---|---|---|
| controlado | 220 | 289 | 0 | 509 |
| incipiente | 89 | 951 | 812 | 1852 |
| incontrolado | 0 | 14 | 536 | 550 |
| total | 309 | 1254 | 1348 | 2911 |

- **De las 8 diferencias de nivel inicial, 0 se explican porque los tramos utiles empiezan en instantes distintos.** En el inicio del tramo comun los dos niveles siguen siendo distintos en las 8. En 1 (la tostadora) coinciden antes de 5 s; en `Brennender Christbaum` a los 5.1 s; en 1 no coinciden nunca.
- `resultados/revision_cambios_notas.csv` creado con `n, video, causa, nota` (23 filas, `causa` y `nota` vacias). Lo llena el usuario: A = la IA se adelanto, B = el humano se atraso, C = ambiguo, D = no es una diferencia real.
- `.codex/` ya estaba en `.gitignore`.

## Fase C del 80 %, pasos 9 a 11: riesgo continuo, particion por origen y secuencias (2026-11-12)

Fecha: 2026-11-12. Rama: `avance-80`. Scripts nuevos: `casi_duplicados.py`, `particion.py`, `secuencias.py`.

### Decisiones del usuario (2026-11-12)

- Entrenamiento, validacion y calibracion usan solo etiquetas de la IA; el usuario no etiqueta mas videos. Las tablas de metas llevan la columna `fuente` (`ia`, `humano` o `dataset` para los videos sin fuego).
- En la prueba, todas las metricas se reportan contra las dos referencias (humano e IA). **Correccion del 2026-11-13:** aqui decia que para la alerta temprana la referencia principal era la IA porque los instantes humanos tienden a ser tardios. No se sostiene: en la revision de las 23 diferencias (`resultados/revision_cambios_notas.csv`) la IA tuvo razon en 13 y el humano en 10, asi que ninguna referencia es superior. En la prueba las dos referencias pesan igual, tambien para el adelanto de la alerta.
- Riesgo continuo (paso 9): escalon 0 (controlado o sin fuego), 0.5 (incipiente), 1 (incontrolado).
- 15 % de los 215 videos sin fuego reservados para la prueba, por origen, semilla 42.
- Las trayectorias heredan el nivel de la escena. Las detecciones falsas en videos sin fuego son controlado, riesgo 0 y tau enmascarado.
- Tramo util: un inicio o un fin vacio significa el inicio o el fin del video.

### Paso 10: particion por origen

- **Grupos de origen:** origen de `metadatos_videos.csv` normalizado (sin espacios sobrantes, en minusculas) mas las uniones manuales de `resultados/particion/uniones.csv` (22 videos en 4 grupos):
  - serie Fire Investigation de Milpitas (4 videos, 2 de ellos en la prueba);
  - `14035491` (prueba) y `240_F_1564726580` (mismo clip con otra resolucion y marca de agua);
  - serie "klart auf" (1 en la prueba y 2 fuera);
  - canal Dutch Firefighters (13 videos, ninguno en la prueba).
- **Casi duplicados:** `casi_duplicados.py` compara las fotos de las tiras (`tiras/` y `tiras_v2/`) con una huella de 64 bits. De 27 pares candidatos, revisados a ojo por Claude, solo 2 son reales: la copia ya excluida y el par de `14035491`. Los demas son escenas distintas con fondo oscuro. Solo cubre los videos de fuego.
- **Etiquetas de la IA:** `ia_v2` donde existe y `ia_v1` en los videos que no pasaron por la segunda fase (columna `version_ia`).
- **Prueba:** los 40 de `prueba_humana.csv`, los 30 de `prueba_cambios.csv` y 32 sin fuego.
- **Fuera de todo conjunto (12):** 6 descartados por la IA, 1 duplicado exacto y 5 del mismo origen que un video de prueba.
- **Resto:** sorteo por grupo de origen dentro de cada tipo, 70 / 15 / 15, del tipo mas escaso al mas comun, asignando cada grupo al conjunto con mas deficit.
- Comprobado: ningun grupo de origen esta en dos conjuntos.

| Tipo segun la IA | Entrenamiento | Validacion | Calibracion | Prueba | Fuera |
|---|---|---|---|---|---|
| sin fuego | 128 | 28 | 27 | 32 | 0 |
| controlado | 116 | 25 | 25 | 30 | 1 |
| incipiente | 20 | 4 | 4 | 5 | 1 |
| incontrolado | 54 | 12 | 11 | 9 | 1 |
| controlado a incipiente | 0 | 0 | 0 | 5 | 0 |
| controlado a incontrolado | 3 | 1 | 1 | 6 | 0 |
| incipiente a incontrolado | 22 | 5 | 4 | 15 | 3 |
| descartado | 0 | 0 | 0 | 0 | 6 |
| total | 343 | 75 | 72 | 102 | 12 |

- En calibracion cuentan para el umbral conformal 52 videos (27 sin fuego y 25 controlados), columna `calibra_umbral`.
- `metas.csv`: 662 filas (377 `ia`, 215 `dataset`, 70 `humano`). Los 70 de prueba tienen dos filas. La columna `valida` dice si la fila humana valida cambios (los 30) o solo el nivel inicial (los 40).

### Paso 11: secuencias por trayectoria

- Una secuencia por video, tramo y trayectoria de fuego, desde `runs_fuego/descriptores_v2_noche/descriptores.csv`.
- Por muestra: `x` (13 descriptores normalizados), `c` (contexto `s_k, rot, tx, ty, inliers, dt` normalizado), `dt` y `t` sin normalizar, `nivel`, `riesgo`, `mascara` (dentro del tramo util), `tau` en segundos y `tau_mascara`.
- `tau_mascara`: 0 sin tau; 1 el video pasa a incontrolado (reglas por tramo de `cortes.estado_tramos`); 2 el video empieza en incontrolado (tau 0). Asi se puede entrenar con o sin el caso 2.
- En la prueba van ademas `nivel_humano`, `riesgo_humano`, `mascara_humano`, `tau_humano`, `tau_mascara_humano` y `seq_ref_humano` (1 solo nivel inicial, 2 cambios, 3 descartado por el humano, 4 clase del dataset).
- Normalizacion (recorte p1 y p99 y estandarizacion) ajustada solo con entrenamiento: 323 877 filas de 214 videos.
- Se descartan las trayectorias de menos de 10 muestras (`--min_muestras 10`, 1 s), igual que el criterio de d13.
- Ventanas de 300 muestras con paso de 150 (`ventanas`: secuencia, inicio, fin); las secuencias completas tambien quedan guardadas.

| Conjunto | Videos | Con trayectorias | Trayectorias | Cortas descartadas | Muestras | Ventanas |
|---|---|---|---|---|---|---|
| entrenamiento | 343 | 214 | 5 135 | 6 153 | 323 877 | 5 486 |
| validacion | 75 | 48 | 683 | 1 438 | 35 545 | 713 |
| calibracion | 72 | 45 | 792 | 1 017 | 71 179 | 927 |
| prueba | 102 | 68 | 1 150 | 1 176 | 76 877 | 1 207 |

| Conjunto | Escena controlado (s) | Escena incipiente (s) | Escena incontrolado (s) | Videos que pasan a incontrolado | Muestras controlado | Muestras incipiente | Muestras incontrolado |
|---|---|---|---|---|---|---|---|
| entrenamiento | 4 134 | 2 255 | 11 026 | 25 | 50 338 | 29 313 | 237 236 |
| validacion | 1 031 | 304 | 1 420 | 6 | 12 988 | 3 129 | 19 256 |
| calibracion | 936 | 1 056 | 2 524 | 5 | 11 926 | 12 865 | 45 834 |
| prueba | 1 462 | 1 494 | 1 745 | 21 | 15 256 | 24 573 | 36 236 |

### Comandos

```
python casi_duplicados.py
python particion.py
python secuencias.py
```

### Limitaciones que se declaran

- **Desbalance:** en entrenamiento el 75 % de las muestras de trayectoria son de incontrolado (los incendios grandes tienen muchas trayectorias) y solo 25 videos pasan a incontrolado. Hay que ponderar la perdida o muestrear por video.
- **No hay ningun video de controlado a incipiente fuera de la prueba:** ese cambio no se ve en entrenamiento.
- Solo 7 videos sin fuego tienen trayectorias de 1 s o mas (5 en entrenamiento, 1 en validacion, 1 en calibracion, 0 en la prueba). El FPR por video se mide igual: un video sin trayectorias tiene riesgo 0.
- En la prueba, las metricas temporales contra el humano salen solo de los 29 videos con cambios validados; los 40 de `prueba_humana.csv` solo validan el nivel inicial.
- Las etiquetas son por escena y todas las trayectorias las heredan.
- La union de series y canales es manual y conservadora; `casi_duplicados.py` no cubre los videos sin fuego.

### Archivos

- `casi_duplicados.py`, `particion.py`, `secuencias.py` (nuevos); `validar_cambios.py` (seccion 6).
- `resultados/particion/`: `uniones.csv`, `casi_duplicados.csv`, `origenes.csv`, `particion.csv`, `metas.csv`, `resumen.md`.
- `resultados/secuencias/`: `indice.csv`, `videos.csv`, `resumen.md`, `normalizacion.json`, `normalizacion_contexto.json`.
- `runs_fuego/secuencias/`: `entrenamiento.npz`, `validacion.npz`, `calibracion.npz`, `prueba.npz` (17 MB, no versionados).
- `resultados/revision_cambios_notas.csv` (lo llena el usuario).

## Fase D del 80 %, pasos 12 a 14: compuerta + CfC, GRU y LSTM (2026-11-12)

Fecha: 2026-11-12. Rama: `avance-80`. Scripts nuevos: `modelo_temporal.py`, `entrenar_temporal.py`, `figura_modelos.py`, `hojas_casi_duplicados.py`. **El conjunto de prueba no se uso.**

### Decisiones del usuario (2026-11-12)

- Aceptadas las decisiones de la fase C (etiquetas v1 donde no hay v2, uniones manuales, descarte de trayectorias de menos de 1 s, reparto 70/15/15 de los videos sin fuego, ventanas de 300 con paso 150).
- Los videos que empiezan en incontrolado (`tau_mascara = 2`) no entran en la perdida de tau; si en las de nivel y riesgo.

### Que se hizo

- **Hojas de casi duplicados:** `hojas_casi_duplicados.py` arma una hoja por cada uno de los 27 pares candidatos en `runs_fuego/casi_duplicados/` (6 fotos de cada video, uno sobre otro, e `indice.csv`), para que el usuario confirme la revision.
- **Normalizacion:** las estadisticas de recorte y estandarizacion de los descriptores y del contexto las calcula `secuencias.py` solo con el entrenamiento de la particion nueva (323 877 filas de 214 videos; `resultados/secuencias/normalizacion.json` y `normalizacion_contexto.json`). No venian de la fase A: alli solo se hizo un ensayo y no quedo ningun archivo de estadisticas.
- **Modelo (`modelo_temporal.py`):**
  - compuerta `u_tilde = u * sigmoid(W c + b)`, con `c = [s_k, rot, tx, ty, inliers, dt_k]` normalizado;
  - CfC de 64 unidades con la celda `ncps.torch.CfCCell` (valores por defecto de `ncps`: una capa de 128 en el backbone);
  - tres salidas: CORAL con K = 3 (un peso compartido y 2 sesgos), riesgo con sigmoide y entropia cruzada binaria contra 0 / 0.5 / 1, y `log(1 + tau)` con Huber donde `tau_mascara = 1`;
  - ponderacion de Kendall: suma de `exp(-s_i) L_i + s_i` con tres `s_i` aprendidos.
- **Comparacion:** GRU y LSTM de 64 unidades con la misma compuerta, las mismas salidas y perdidas, y el intervalo de tiempo como entrada extra.
- **Dos diferencias con el pedido, que se declaran:**
  - `ncps.torch.CfC` (version 1.0.1) falla con lotes de mas de una secuencia cuando se le pasan `timespans` (aplana el intervalo a un vector que no coincide con las 64 unidades). Se usa su celda `CfCCell` con el bucle temporal propio; el calculo es el mismo.
  - El intervalo que recibe el modelo (`timespans` en el CfC, entrada extra en GRU y LSTM) es el tiempo real entre dos muestras consecutivas de la trayectoria. En el 96.1 % de los pasos es igual a `dt_k`; en el 3.9 % la trayectoria no fue detectada en algun fotograma y el intervalo es mayor (hasta 3.2 s).
- **Muestreo:** cada lote son 32 ventanas; para cada una se elige un video al azar con la misma probabilidad (214 videos de entrenamiento con trayectorias) y despues una ventana de ese video. Una epoca son 172 lotes.
- **Pesos de la perdida de nivel:** `1 / (3 f_k)` con la frecuencia esperada de cada nivel bajo ese muestreo (media 1 sobre las muestras) y tope 5.

| Nivel | Frecuencia con el muestreo por video | Peso |
|---|---|---|
| controlado | 0.567 | 0.587 |
| incipiente | 0.113 | 2.944 |
| incontrolado | 0.319 | 1.044 |

- **Entrenamiento:** semilla 42, AdamW (1e-3, weight decay 1e-4), recorte de gradiente 1.0, hasta 60 epocas, paciencia 10.
- **Eleccion de la mejor epoca:** F1 macro del nivel de escena por segundo en validacion, con las secuencias completas. El nivel de la escena en un segundo es el maximo de las trayectorias con muestras en ese segundo; si no hay ninguna trayectoria, el sistema da controlado. Solo cuentan los 47 videos de fuego de validacion, dentro del tramo util (2 757 s).

### Comandos

```
python hojas_casi_duplicados.py
python entrenar_temporal.py --modelo gru
python entrenar_temporal.py --modelo lstm
python entrenar_temporal.py --modelo cfc
python figura_modelos.py
```

### Numeros (validacion, mejor epoca de cada modelo, una sola semilla)

| Modelo | Parametros | s por epoca | Epocas corridas | Mejor epoca | F1 macro escena | Exactitud escena | F1 macro muestra | Exactitud muestra | MAE de tau (s) |
|---|---|---|---|---|---|---|---|---|---|
| CfC | 43 298 | 96.1 | 18 | 8 | 0.513 | 0.610 | 0.546 | 0.657 | 49.9 |
| GRU | 15 650 | 3.0 | 16 | 6 | 0.516 | 0.615 | 0.539 | 0.667 | 32.6 |
| LSTM | 20 770 | 3.0 | 26 | 16 | 0.526 | 0.567 | 0.535 | 0.591 | 36.6 |

- El MAE de tau se mide en las muestras de trayectoria anteriores al cambio a incontrolado (6 videos de validacion).
- F1 por nivel de escena (controlado, incipiente, incontrolado): CfC 0.637, 0.169, 0.732; GRU 0.643, 0.161, 0.744; LSTM 0.591, 0.303, 0.683.

Matriz de confusion por segundo de escena (filas: etiqueta de la IA; columnas: modelo):

| Modelo | Real | controlado | incipiente | incontrolado |
|---|---|---|---|---|
| CfC | controlado | 746 | 226 | 61 |
| CfC | incipiente | 214 | 68 | 21 |
| CfC | incontrolado | 348 | 206 | 867 |
| GRU | controlado | 763 | 231 | 39 |
| GRU | incipiente | 234 | 65 | 4 |
| GRU | incontrolado | 345 | 209 | 867 |
| LSTM | controlado | 608 | 361 | 64 |
| LSTM | incipiente | 116 | 181 | 6 |
| LSTM | incontrolado | 300 | 348 | 773 |

### Lectura de los resultados

- **Los tres modelos quedan empatados y con un resultado bajo: F1 macro de escena entre 0.51 y 0.53.** La diferencia entre ellos (0.013) es menor que la variacion de un mismo modelo entre epocas (de 0.46 a 0.53), con una sola semilla y 47 videos. No se puede decir que uno sea mejor.
- **El F1 de validacion no mejora con el entrenamiento.** La perdida de entrenamiento baja (de 2.3 a 0.6 - 1.4) y la de validacion sube despues de la epoca 12: hay sobreajuste. La mejor epoca se elige sobre una curva casi plana, asi que esa eleccion es en buena parte ruido.
- **El nivel incipiente casi no se reconoce** (F1 de 0.16 a 0.30). Es el nivel con menos datos (303 s en validacion) y el de frontera.
- **Parte del error no es del modelo temporal:** 378 de los 2 757 s de escena (13.7 %) no tienen ninguna trayectoria de fuego de 1 s o mas, y la regla les asigna controlado. De esos, 265 son de incontrolado: el 18.6 % de los segundos de incontrolado (265 de 1 421) se pierden antes del modelo, y explican la mayor parte de los 300 a 348 s de incontrolado clasificados como controlado.
- El CfC tarda 32 veces mas por epoca (bucle en Python sobre 300 pasos) y tiene mas parametros (43 298 contra 15 650 y 20 770) con el mismo tamano oculto.

### Limitaciones que se declaran

- Una sola semilla; la comparacion con 5 semillas es del 100 %.
- Validacion pequena: 47 videos de fuego, solo 6 con cambio a incontrolado.
- Etiquetas de la IA, por escena, heredadas por todas las trayectorias.
- No se ajusto ningun hiperparametro.

### Pendiente de decision del usuario antes de la fase E

- Con F1 de 0.51 a 0.53 y sobreajuste, conviene revisar el entrenamiento antes de calibrar: regularizacion (dropout, menos unidades), parada por perdida de validacion, y que hacer con los segundos sin trayectoria (por ejemplo, mantener el ultimo nivel durante unos segundos o bajar el minimo de 1 s).

### Archivos

- `modelo_temporal.py`, `entrenar_temporal.py`, `figura_modelos.py`, `hojas_casi_duplicados.py` (nuevos).
- `requirements.txt`: se agrego `ncps==1.0.1` (instalado con `pip install ncps --no-deps`).
- `resultados/modelos/`: `comparacion.md`, `curvas.png` y, por modelo, `historial.csv` y `metricas.json`.
- `runs_fuego/modelos/<modelo>/mejor.pt` y `registro_<modelo>.txt` (no versionados).
- `runs_fuego/casi_duplicados/` (27 hojas, no versionadas).

## Diagnostico antes de la fase E (2026-11-12)

Fecha: 2026-11-12. Rama: `avance-80`. Script nuevo: `diagnostico_escena.py` (`python diagnostico_escena.py`). No cambia el modelo, no reentrena los modelos temporales y no usa la prueba. Detalle en `resultados/diagnostico/diagnostico.md`.

### 1. Los 378 s de validacion sin ninguna trayectoria de 1 s o mas

Cada segundo va a una sola causa, en este orden: corte, trayectorias cortas, humo, sin deteccion, otra. Las detecciones de humo y de fuego sin trayectoria salen de `curvas/` (YOLO a 5 fps, confianza 0.25), no del seguimiento (confianza 0.1).

| Causa | Controlado (s) | Incipiente (s) | Incontrolado (s) | Total (s) | En huecos de 3 s o menos |
|---|---|---|---|---|---|
| dentro de los 3 s posteriores a un corte | 0 | 9 | 25 | 34 | 19 |
| solo trayectorias de fuego de menos de 1 s | 32 | 16 | 165 | 213 | 50 |
| solo detecciones de humo | 13 | 0 | 12 | 25 | 3 |
| sin ninguna deteccion | 9 | 6 | 12 | 27 | 0 |
| otra: deteccion de fuego que no forma trayectoria | 20 | 8 | 51 | 79 | 17 |
| total | 74 | 39 | 265 | 378 | 89 |

- **El 56 % (213 de 378) tiene fuego seguido, pero en trayectorias de menos de 1 s que se descartan.** Sumando los 30 s posteriores a un corte que tambien las tienen, son 243 (64 %).
- Solo 27 s (7 %) no tienen ninguna deteccion.
- Solo 89 s (24 %) estan en huecos de 3 s o menos entre segundos con trayectoria (75 de incontrolado). Mantener el ultimo nivel 3 s recuperaria como mucho esos; el resto son rachas largas.

### 2 y 3. Referencias sin modelo temporal y kappa contra la IA

- Vector por segundo de escena: maximo y media de cada descriptor sobre las trayectorias vivas y numero de trayectorias (27 valores). Entrenadas con 15 680 s de entrenamiento, pesos por clase balanceados, semilla 42. Los segundos sin trayectoria cuentan como controlado, igual que en los modelos temporales.

| Estimador | F1 macro escena | Exactitud escena | Kappa lineal | F1 controlado | F1 incipiente | F1 incontrolado |
|---|---|---|---|---|---|---|
| gradient boosting (sin tiempo) | 0.590 | 0.634 | 0.429 | 0.602 | 0.440 | 0.727 |
| regresion logistica (sin tiempo) | 0.526 | 0.563 | 0.396 | 0.570 | 0.332 | 0.676 |
| LSTM | 0.526 | 0.567 | 0.419 | 0.591 | 0.303 | 0.683 |
| GRU | 0.516 | 0.615 | 0.477 | 0.643 | 0.161 | 0.744 |
| CfC | 0.513 | 0.610 | 0.461 | 0.637 | 0.169 | 0.732 |
| clase mayoritaria | 0.227 | 0.515 | 0.000 | 0.000 | 0.000 | 0.680 |

- **Los modelos temporales no superan a una referencia sin tiempo.** El gradient boosting sobre agregados por segundo da F1 0.590, contra 0.513 a 0.526 de los tres modelos temporales; la regresion logistica iguala al LSTM.
- La ventaja de las referencias esta en el incipiente (F1 0.44 y 0.33 contra 0.16 a 0.30). En kappa lineal GRU y CfC quedan arriba (0.477 y 0.461) porque sus errores son mas entre niveles vecinos.
- Solo en los 2 379 s con trayectoria: gradient boosting F1 0.649, exactitud 0.703, kappa 0.574; regresion logistica 0.582, 0.621, 0.532. Ese es el orden de lo que permiten hoy los descriptores con etiquetas de la IA.
- **Kappa de los modelos contra la IA (0.40 a 0.48) frente al kappa humano contra IA (0.426):** estan en el mismo rango. No son los mismos videos (29 de la prueba de cambios contra 47 de validacion), asi que es una referencia, no una comparacion directa.
- Diferencias de protocolo que se declaran: las referencias ven cada segundo con el mismo peso (sin muestreo por video), no se ajustaron hiperparametros, y es una sola semilla.

### 4. Ruido de la herencia (entrenamiento)

| Nivel | Segundos con trayectoria | Mediana de trayectorias vivas | Percentil 90 | Segundos con mas de una | Trayectorias con area menor a un cuarto de la mayor |
|---|---|---|---|---|---|
| controlado | 3 322 | 1 | 4 | 49.4 % | 25.5 % |
| incipiente | 2 075 | 1 | 3 | 46.3 % | 18.0 % |
| incontrolado | 10 260 | 2 | 6 | 65.0 % | 41.5 % |

- En incontrolado, el 41.5 % de las trayectorias-segundo son regiones con menos de un cuarto del area de la mayor del mismo segundo, y todas heredan la etiqueta incontrolado. El modelo por trayectoria recibe muchas regiones pequenas etiquetadas con el nivel de la escena.

### Lectura

- El problema principal no es el tipo de red recurrente. Son (a) la etiqueta por escena heredada por cada trayectoria, que las referencias evitan al agregar por segundo, y (b) los segundos sin trayectoria de 1 s, que salen sobre todo de trayectorias cortas descartadas.
- Pendiente de decision del usuario: como cambiar la entrada o la etiqueta del modelo temporal antes de la fase E.

### Archivos

- `diagnostico_escena.py` (nuevo).
- `resultados/diagnostico/diagnostico.md` y `diagnostico.json`.

## Fase D, segunda version: modelo temporal a nivel de escena (2026-11-12)

Fecha: 2026-11-12. Rama: `avance-80`. Scripts nuevos: `escena.py`, `modelo_escena.py`, `entrenar_escena.py`, `comparar_escena.py`. **La prueba no se uso.** El modelo por trayectoria (`modelo_temporal.py`, `entrenar_temporal.py`, `resultados/modelos/`) se conserva sin cambios como primera version.

### Decisiones del usuario (2026-11-12)

- El modelo temporal pasa a nivel de escena; ya no hay herencia de la etiqueta de la escena en cada trayectoria.
- Casi duplicados: el usuario reviso las 27 hojas y no hay mas pares que unir, aparte de la copia exacta (ya excluida) y el par de `14035491` (ya unido). **La particion no cambia.**

### Vector de escena (`escena.py`)

- Una secuencia por video y tramo (se reinicia en cada corte), a 10 fps, con **todos** los fotogramas procesados.
- 34 valores por fotograma: maximo y media de los 13 descriptores sobre las trayectorias de fuego vivas (tambien las de menos de 1 s), numero de trayectorias, numero con edad de al menos 1 s, edad de la mas antigua, numero de cajas de fuego y de humo, area total de fuego y de humo, y la bandera `sin_trayectoria` (con los agregados en 0).
- Metas por fotograma: nivel, riesgo y tau de la escena, con las mismas mascaras (`tau_mascara = 2` no entra en la perdida de tau).
- Normalizacion (recorte p1 y p99 y estandarizacion) solo con entrenamiento: 189 291 fotogramas de 337 videos.
- **Con los datos guardados, y lo que eso implica:**
  - las marcas de tiempo de los fotogramas sin trayectoria no estaban guardadas; se interpolan con el indice del fotograma (donde si hay marca real, difiere de 0.1 k en 0.03 s como maximo);
  - el contexto de camara de esos fotogramas tampoco: se usa la identidad y la ultima proporcion de inliers conocida;
  - las detecciones por fotograma salen de `curvas/` (YOLO a 5 fps, confianza 0.25): cada fotograma toma la ultima medida anterior;
  - los 215 clips sin fuego no tenian curva: se corrio `preanotar.py` solo sobre ellos (36 min de video, unos 8 min de proceso; `runs_fuego/curvas_sin_fuego/`). No se reentreno nada;
  - tener las detecciones exactas a 10 fps y confianza 0.1, con marcas y camara de todos los fotogramas, exige repetir la corrida de descriptores de los 603 videos: unas 6 h. No se corrio.
- 10 clips sin fuego de menos de 1 s (2 a 9 fotogramas) no forman secuencia: 6 de entrenamiento, 1 de validacion, 1 de calibracion y 2 de prueba.

| Conjunto | Videos | Secuencias | Fotogramas | Controlado | Incipiente | Incontrolado | Sin trayectoria | Ventanas |
|---|---|---|---|---|---|---|---|---|
| entrenamiento | 337 | 460 | 189 291 | 51 252 | 22 555 | 110 304 | 34 488 | 1 267 |
| validacion | 74 | 110 | 30 301 | 12 498 | 3 041 | 14 199 | 6 496 | 208 |
| calibracion | 71 | 82 | 48 375 | 11 651 | 10 565 | 25 244 | 7 092 | 311 |
| prueba | 100 | 140 | 51 075 | 17 130 | 14 950 | 17 464 | 9 104 | 328 |

### Modelos y entrenamiento

- Misma compuerta con el contexto de camara y dt; CfC (`CfCCell`) con el intervalo real entre fotogramas; GRU y LSTM con dt como entrada; mismas tres salidas y ponderacion de Kendall.
- Dropout 0.2 en la entrada (despues de la compuerta) y antes de las salidas; AdamW con tasa 3e-4 y descenso coseno, weight decay 1e-3; lotes de 32 ventanas; recorte 1.0; hasta 60 epocas, paciencia 10.
- Muestreo por video: 337 videos de entrenamiento, **incluidos 122 sin fuego** (ahora tienen secuencia aunque no tengan trayectorias). Una epoca son 40 lotes.
- Frecuencia de niveles con ese muestreo: controlado 0.617, incipiente 0.116, incontrolado 0.268. Pesos: 0.541, 2.879 y 1.246.
- Mejor epoca: F1 macro de escena por segundo en validacion, suavizado con la media de la epoca anterior, la propia y la siguiente. El nivel de un segundo es el mas votado entre sus fotogramas.
- Semillas: GRU y LSTM tardan menos de 1 min, asi que van con 42, 43 y 44. **El CfC tarda 11 min (22 s por epoca), asi que solo va con la 42.**

### Numeros en validacion (2 757 s de escena de 47 videos de fuego; media +- desviacion entre semillas)

| Estimador | Corridas | F1 macro | F1 con trayectoria | F1 sin trayectoria | F1 controlado | F1 incipiente | F1 incontrolado | Exactitud | Kappa lineal | MAE tau (s) | Cambios de nivel por video |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CfC de escena | 1 | 0.493 | 0.512 | 0.363 | 0.698 | 0.029 | 0.752 | 0.679 | 0.499 | 43.6 | 4.0 |
| GRU de escena | 3 | 0.556 +- 0.019 | 0.579 +- 0.017 | 0.405 +- 0.042 | 0.725 +- 0.018 | 0.133 +- 0.033 | 0.809 +- 0.014 | 0.710 +- 0.015 | 0.582 +- 0.024 | 42.4 +- 1.4 | 4.5 |
| LSTM de escena | 3 | 0.526 +- 0.018 | 0.548 +- 0.013 | 0.379 +- 0.053 | 0.723 +- 0.024 | 0.069 +- 0.054 | 0.787 +- 0.035 | 0.704 +- 0.026 | 0.557 +- 0.050 | 44.0 +- 3.4 | 4.9 |
| regresion logistica (sin tiempo) | 1 | 0.480 | 0.508 | 0.287 | 0.465 | 0.285 | 0.690 | 0.513 | 0.355 | - | 12.8 |
| gradient boosting (sin tiempo) | 1 | 0.608 | 0.632 | 0.417 | 0.600 | 0.425 | 0.798 | 0.664 | 0.506 | - | 13.0 |
| clase mayoritaria | 1 | 0.227 | 0.218 | 0.275 | 0.000 | 0.000 | 0.680 | 0.515 | 0.000 | - | 0.0 |

- Cambios de nivel por video segun la etiqueta: 0.13.
- "Con trayectoria": 2 379 s con alguna trayectoria de 1 s o mas; "sin trayectoria": 378 s.
- Las referencias usan el mismo vector de escena promediado por segundo (18 433 s de entrenamiento, pesos por clase balanceados).

| Estimador | Exactitud con trayectoria | Kappa con trayectoria | Exactitud sin trayectoria | Kappa sin trayectoria |
|---|---|---|---|---|
| CfC de escena | 0.704 | 0.541 | 0.526 | 0.269 |
| GRU de escena | 0.736 +- 0.013 | 0.628 +- 0.016 | 0.547 +- 0.045 | 0.326 +- 0.061 |
| LSTM de escena | 0.731 +- 0.017 | 0.605 +- 0.034 | 0.536 +- 0.096 | 0.299 +- 0.116 |
| regresion logistica | 0.540 | 0.426 | 0.344 | 0.131 |
| gradient boosting | 0.686 | 0.557 | 0.532 | 0.276 |

| Modelo | Semilla | Parametros | s por epoca | Epocas corridas | Mejor epoca | F1 macro | Kappa |
|---|---|---|---|---|---|---|---|
| CfC | 42 | 46 133 | 21.9 | 28 | 17 | 0.493 | 0.499 |
| GRU | 42 | 19 829 | 0.8 | 60 | 50 | 0.552 | 0.568 |
| GRU | 43 | 19 829 | 0.8 | 36 | 25 | 0.534 | 0.562 |
| GRU | 44 | 19 829 | 0.8 | 48 | 37 | 0.581 | 0.615 |
| LSTM | 42 | 26 293 | 0.8 | 27 | 16 | 0.529 | 0.503 |
| LSTM | 43 | 26 293 | 0.7 | 19 | 8 | 0.503 | 0.543 |
| LSTM | 44 | 26 293 | 0.7 | 27 | 16 | 0.547 | 0.624 |

Primera version (por trayectoria, semilla 42), mismos segundos: CfC F1 0.513, exactitud 0.610, kappa 0.461; GRU 0.516, 0.615, 0.477; LSTM 0.526, 0.567, 0.419.

Las matrices de confusion estan en `resultados/modelos_escena/comparacion.md` y las curvas en `resultados/modelos_escena/curvas.png`.

### Lectura de los resultados

- **El paso a escena mejora la exactitud y el kappa.** GRU: exactitud de 0.615 a 0.710 y kappa de 0.477 a 0.582. En los segundos con trayectoria el kappa llega a 0.628.
- **El F1 macro mejora poco (GRU 0.556) porque el nivel incipiente casi desaparece** (F1 0.13 en GRU, 0.07 en LSTM, 0.03 en CfC). Los modelos temporales mandan el incipiente a controlado (777 de 909 s en las tres corridas de GRU).
- **El gradient boosting sin tiempo sigue teniendo el mejor F1 macro (0.608)** gracias al incipiente (0.425), pero con menos exactitud (0.664) y kappa (0.506) que el GRU, y cambia de nivel 13 veces por video.
- **Todos cambian de nivel muchas mas veces que la etiqueta** (4 a 5 por video los temporales, 13 las referencias, contra 0.13). Eso lo debe corregir la histeresis de la fase E.
- **El CfC queda por debajo del GRU** (F1 0.493 contra 0.556 +- 0.019; kappa 0.499 contra 0.582 +- 0.024), con una sola semilla y 28 epocas. Es el resultado contrario a la hipotesis de la tesis y hay que reportarlo asi o investigarlo.
- **Los segundos sin trayectoria siguen siendo los peores** (kappa 0.27 a 0.33), aunque ahora el modelo los ve.
- El MAE de tau (42 a 44 s) no mejora; solo 6 videos de validacion tienen cambio a incontrolado. No es comparable con el de la primera version, que se media por muestra de trayectoria.

### Limitaciones que se declaran

- CfC con una sola semilla; GRU y LSTM con tres. Validacion de 47 videos.
- Las detecciones por fotograma vienen de 5 fps y confianza 0.25, y los fotogramas sin trayectoria no tienen marca de tiempo ni contexto de camara propios.
- No se ajustaron hiperparametros mas alla de los pedidos.

### Comandos

```
python preanotar.py --seleccion resultados/cortes/lista_non_fire.csv --videos real_video/real_video/non_fire --salida <temporal> --curvas runs_fuego/curvas_sin_fuego
python escena.py
python entrenar_escena.py --modelo gru --semilla 42
python comparar_escena.py
```

### Archivos

- `escena.py`, `modelo_escena.py`, `entrenar_escena.py`, `comparar_escena.py` (nuevos).
- `requirements.txt`: se agregaron `scikit-learn==1.7.0` y `pandas==2.2.3` (ya estaban instalados y los usan `secuencias.py`, `escena.py` y las referencias).
- `resultados/escena/`: `resumen.md`, `videos.csv`, `normalizacion.json`, `normalizacion_contexto.json`.
- `resultados/modelos_escena/`: `comparacion.md`, `curvas.png`, `referencias.json` y, por corrida, `historial.csv` y `metricas.json`.
- No versionados: `runs_fuego/escena/*.npz`, `runs_fuego/modelos_escena/<modelo>_s<semilla>/mejor.pt`, `runs_fuego/curvas_sin_fuego/`.

## Vector de escena desde el propio pipeline: `descriptores_v3.py` y `vector_escena.py` (2026-11-12)

Fecha: 2026-11-12. Rama: `avance-80`. Scripts nuevos: `descriptores_v3.py`, `vector_escena.py`, `escena_v2.py`, `probar_vector_escena.py`. No se borro ni se sobrescribio nada anterior: `descriptores.py`, `descriptores_v2.py`, `escena.py`, `runs_fuego/descriptores_v2_noche/`, `runs_fuego/escena/` y los modelos quedan como estaban.

### Que se hizo

- **`descriptores_v3.py`:** misma corrida que `descriptores_v2.py` (mismo detector, seguimiento `dt_tope2_cmc`, camara, cortes y `DescriptoresV2`), y ademas escribe `fotogramas.csv` con una fila por **cada fotograma procesado**: `k`, marca real `t`, `dt`, `tramo`, contexto de camara (`s_k, rot, tx, ty, inliers, valido`), numero de trayectorias y las detecciones que entran al seguimiento (confianza 0.1): numero de cajas, area total (union en una grilla de 64x36) y confianza maxima de fuego y de humo. Guarda tambien los mismos conteos y areas filtrados a confianza 0.25.
  - Todo va a una carpeta propia (`--salida`, por defecto `runs_fuego/descriptores_v3`): `descriptores.csv`, `fotogramas.csv`, `cortes.csv`, `resumen.csv` y `procesados.csv`. No toca `resultados/cortes/`.
  - Deja fuera los videos de `resultados/excluidos.csv` (603 videos de los 604 de `lista_noche.csv`).
  - **Se retoma sola:** al arrancar lee `procesados.csv`, salta los videos ya hechos y quita de los CSV las filas de un video que hubiera quedado a medias. No hay opcion que sobrescriba; para empezar de cero hay que usar otra carpeta.
- **`vector_escena.py`:** clase `VectorEscena` con `paso(t, trayectorias, detecciones)`, que devuelve los 34 valores de un fotograma y guarda el nacimiento de cada trayectoria; `reiniciar()` en cada corte. La usan igual el entrenamiento (`vectores_video`, desde los CSV) y el sistema en vivo (fotograma a fotograma). Redondea como los CSV para que los dos caminos den el mismo numero.
- **`escena_v2.py`:** arma las secuencias de escena desde la carpeta de `descriptores_v3.py` con esa clase, sin `curvas/`, sin interpolar marcas y con el contexto de camara real. Salida en `runs_fuego/escena_v2/` y `resultados/escena_v2/`.

### Prueba con 5 videos (`python probar_vector_escena.py`)

Lista en `resultados/vector_escena/videos_prueba.csv`: 3 de fuego (uno con un corte, uno con huecos sin trayectoria, uno normal) y 2 sin fuego (uno con falsos del detector). Ninguno es del conjunto de prueba.

| Video | Fotogramas | Filas de descriptores (noche / v3) | Filas distintas | Diferencia maxima de los 29 agregados | Diferencia maxima de la marca de tiempo |
|---|---|---|---|---|---|
| videoplayback (8).mp4 | 251 | 375 / 375 | 0 | 2e-16 | 0.000 s |
| 2451511630.mp4 | 73 | 80 / 80 | 0 | 6e-17 | 0.000 s |
| fire3.avi | 174 | 444 / 444 | 0 | 2e-16 | 0.000 s |
| 2030845629_clip_003.mp4 | 100 | 142 / 142 | 0 | 0 | 0.000 s |
| 1293332816_clip_001.mp4 | 100 | 0 / 0 | 0 | 0 | 0.020 s |

- **Los descriptores por trayectoria de v3 son identicos a los de la corrida de la noche**, y los agregados, el numero de trayectorias, la edad y la bandera `sin_trayectoria` coinciden en los 5 videos.
- **Contexto de camara:** en los fotogramas sin trayectoria el vector anterior ponia la identidad; con los datos reales, 166 de 187 de esos fotogramas tienen movimiento de camara distinto de la identidad.
- **Detecciones (699 fotogramas), anterior (`curvas/`, 5 fps, confianza 0.25) contra las del seguimiento:**

| Valor | Media anterior | Media a confianza 0.1 | Fotogramas distintos | Media a confianza 0.25 | Fotogramas distintos a 0.25 |
|---|---|---|---|---|---|
| cajas de fuego | 1.61 | 2.80 | 49.0 % | 1.63 | 11.7 % |
| cajas de humo | 0.077 | 0.148 | 6.7 % | 0.079 | 1.0 % |
| area de fuego | 0.051 | 0.077 | 39.5 % | 0.051 | 13.9 % |
| area de humo | 0.034 | 0.040 | 9.0 % | 0.034 | 3.6 % |

  - La diferencia viene casi toda del umbral de confianza (0.1 contra 0.25): a 0.25 las medias coinciden y solo cambia del 1 al 14 % de los fotogramas, por el paso de 5 a 10 fps. A 0.1 hay un 74 % mas de cajas de fuego y un 52 % mas de area.
  - Como estas 4 columnas cambian de escala, **los modelos de escena hay que reentrenarlos con `escena_v2`**; los pesos actuales no sirven con el vector nuevo.
- **Retomar tras un corte:** se quito de `procesados.csv` el ultimo video y se trunco la ultima linea de `descriptores.csv`; al volver a correr, quito 344 filas incompletas, reproceso los 2 videos pendientes y los 4 CSV quedaron identicos byte a byte a los de la corrida sin corte.

### Comando de la corrida completa (la lanza el usuario)

```
python descriptores_v3.py
```

- 603 videos, salida en `runs_fuego/descriptores_v3/`. Tiempo estimado: unas 6.5 h (la corrida de la noche tardo 6.13 h con el mismo detector y seguimiento).
- Si se corta, se repite el mismo comando y sigue donde quedo.
- Al terminar: `python escena_v2.py`.

### Archivos

- `descriptores_v3.py`, `vector_escena.py`, `escena_v2.py`, `probar_vector_escena.py` (nuevos).
- `resultados/vector_escena/videos_prueba.csv` y `comparacion.md`.
- `runs_fuego/descriptores_v3_prueba/` (los 5 videos, no versionado).

## Vector de escena v2, reentrenamiento y fase E en validacion (2026-11-13)

Fecha: 2026-11-13. Rama: `avance-80`. Scripts nuevos: `reentrenar_escena.py`, `fase_e.py`, `entrenar_variantes.py`. Modificado: `escena_v2.py` (opcion `--detecciones`). **La prueba no se uso.** No se sobrescribio nada anterior (`runs_fuego/escena/`, `resultados/modelos_escena/` y el modelo por trayectoria siguen como estaban).

### Corrida completa de `descriptores_v3.py` y `escena_v2.py`

- La corrida la lanzo el usuario: 603 videos en 6.22 h, salida en `runs_fuego/descriptores_v3/`. Los conteos por video (fotogramas, trayectorias, filas) y los 241 cortes son iguales a los de la corrida de la noche.
- `python escena_v2.py` usa las detecciones del propio pipeline filtradas a confianza 0.25 (`--detecciones alta`, decision del usuario; `seguimiento` usa las de 0.1).
- **Comparacion con `resultados/escena/`:** mismos videos y mismas secuencias en los cuatro conjuntos. Validacion y calibracion son identicas en fotogramas, metas y fotogramas sin trayectoria. En entrenamiento y prueba hay 1 fotograma mas en cada uno (189 292 y 51 076): son 4 fotogramas de 3 clips sin fuego con cortes (`1898933216_clip_001`, `2568999255_clip_001`, `1694496253_clip_002`) que la version anterior asignaba al tramo equivocado por interpolar la marca de tiempo. La version nueva usa el tramo real.

### Reentrenamiento (`python reentrenar_escena.py`)

- 60 epocas completas (sin parada temprana), mejor epoca por F1 de escena suavizado. Resto igual que antes: dropout 0.2, weight decay 1e-3, descenso coseno, muestreo por video, pesos por clase.
- **Tasa de aprendizaje:** se probo 1e-4, 3e-4 y 1e-3 con la semilla 42 y se eligio por el F1 suavizado de validacion. La rejilla la eligio Claude. Las semillas 43 y 44 se entrenaron con la tasa elegida. El CfC en CPU (3 corridas en paralelo, 4 hilos cada una).

| Modelo | F1 suavizado con 1e-4 | con 3e-4 | con 1e-3 | Tasa elegida | Minutos por corrida |
|---|---|---|---|---|---|
| CfC | 0.482 | 0.524 | 0.529 | 1e-3 | 29.7 (busqueda), 19.6 (semillas 43 y 44) |
| GRU | 0.487 | 0.515 | 0.629 | 1e-3 | 1.3 a 1.9 |
| LSTM | 0.501 | 0.519 | 0.545 | 1e-3 | 1.3 a 2.0 |

- **Los tres eligen la tasa mas alta de la rejilla**, asi que el optimo puede estar por encima de 1e-3; no se probo.
- Tiempo total: 49.6 min (29.9 min la busqueda de tasa).

Validacion antes de la fase E (2 757 s de escena, media +- desviacion de 3 semillas):

| Estimador | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | Exactitud | Kappa lineal | MAE tau (s) | Cambios por video |
|---|---|---|---|---|---|---|---|---|
| CfC | 0.544 +- 0.015 | 0.711 | 0.170 | 0.751 | 0.644 +- 0.014 | 0.543 +- 0.040 | 37.5 | 5.3 |
| GRU | 0.621 +- 0.008 | 0.788 | 0.208 | 0.866 | 0.753 +- 0.012 | 0.698 +- 0.022 | 27.9 | 3.4 |
| LSTM | 0.552 +- 0.048 | 0.738 | 0.137 | 0.782 | 0.685 +- 0.042 | 0.583 +- 0.074 | 31.3 | 4.8 |
| gradient boosting (sin tiempo) | 0.622 | 0.623 | 0.444 | 0.799 | 0.679 | 0.519 | - | 12.8 |
| regresion logistica (sin tiempo) | 0.482 | 0.468 | 0.285 | 0.691 | 0.514 | 0.357 | - | 12.3 |

- En los segundos sin trayectoria de 1 s o mas, el GRU sube de kappa 0.326 a 0.563.
- Detalle por corrida, matrices de confusion y curvas: `resultados/modelos_escena_v2/final/comparacion.md` y `curvas.png`.

### Fase E (`python fase_e.py`)

- **Configuracion final (decision del usuario): umbrales de CORAL en 0.5 con histeresis de 2 s, y alerta con alfa 0.10.** Los umbrales ajustados en calibracion se reportan solo como comparacion: empeoran la validacion porque calibracion tiene otra proporcion de niveles.
- Histeresis: el nivel sube cuando el nivel crudo se mantiene por encima durante 2 s y baja cuando se mantiene por debajo durante 2 s (causal).
- Alerta conformal por video: puntuacion = riesgo maximo de cada uno de los 52 videos de calibracion sin fuego o controlados; umbral = cuantil conformal. Se activa cuando el riesgo llega al umbral y se apaga cuando baja de 0.8 veces el umbral durante 2 s (el factor 0.8 lo eligio Claude).
- Referencias sin tiempo: una prediccion por segundo, riesgo = 0.5 P(incipiente) + P(incontrolado), misma histeresis y misma alerta.
- Segmentos `(archivo, t_inicio, t_fin, nivel)` de la configuracion final en `resultados/fase_e/segmentos/`.

| Modelo | Etapa | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | Exactitud | Kappa lineal | Cambios por video |
|---|---|---|---|---|---|---|---|---|
| CfC | antes | 0.544 +- 0.015 | 0.711 | 0.170 | 0.751 | 0.644 | 0.543 +- 0.040 | 5.26 |
| CfC | 0.5 + histeresis (final) | 0.536 +- 0.019 | 0.706 | 0.169 | 0.734 | 0.629 | 0.529 +- 0.041 | 2.16 |
| CfC | solo umbrales ajustados | 0.477 +- 0.019 | 0.690 | 0.115 | 0.627 | 0.547 | 0.465 | 6.65 |
| CfC | umbrales ajustados + histeresis | 0.464 +- 0.020 | 0.678 | 0.113 | 0.602 | 0.529 | 0.444 | 2.20 |
| GRU | antes | 0.621 +- 0.008 | 0.788 | 0.208 | 0.866 | 0.753 | 0.698 +- 0.022 | 3.41 |
| GRU | 0.5 + histeresis (final) | 0.612 +- 0.002 | 0.771 | 0.209 | 0.855 | 0.732 | 0.679 +- 0.026 | 2.21 |
| GRU | solo umbrales ajustados | 0.579 +- 0.041 | 0.802 | 0.175 | 0.761 | 0.690 | 0.633 | 2.70 |
| GRU | umbrales ajustados + histeresis | 0.561 +- 0.043 | 0.787 | 0.155 | 0.740 | 0.666 | 0.607 | 1.67 |
| LSTM | antes | 0.552 +- 0.048 | 0.738 | 0.137 | 0.782 | 0.685 | 0.583 +- 0.074 | 4.76 |
| LSTM | 0.5 + histeresis (final) | 0.548 +- 0.048 | 0.726 | 0.156 | 0.763 | 0.663 | 0.563 +- 0.078 | 2.71 |
| LSTM | solo umbrales ajustados | 0.528 +- 0.050 | 0.735 | 0.139 | 0.708 | 0.633 | 0.536 | 4.80 |
| LSTM | umbrales ajustados + histeresis | 0.516 +- 0.047 | 0.720 | 0.138 | 0.691 | 0.611 | 0.514 | 2.75 |
| gradient boosting | antes | 0.622 | 0.623 | 0.444 | 0.799 | 0.679 | 0.519 | 12.81 |
| gradient boosting | histeresis (final) | 0.653 | 0.664 | 0.449 | 0.847 | 0.720 | 0.598 | 2.30 |
| regresion logistica | antes | 0.482 | 0.468 | 0.285 | 0.691 | 0.514 | 0.357 | 12.28 |
| regresion logistica | histeresis (final) | 0.479 | 0.488 | 0.268 | 0.681 | 0.509 | 0.371 | 3.13 |

Cambios de nivel por video segun la etiqueta: 0.13.

Alerta (28 videos sin fuego, 25 controlados, 22 con propagacion y 6 con cambio a incontrolado en validacion). Adelanto = `t_incontrolado` de la IA menos el instante de la primera alerta; negativo significa que la alerta llega despues del cambio.

| Modelo | Alfa | Umbral de riesgo | Falsas alarmas en videos sin fuego | Falsas alarmas en videos controlados | Videos con propagacion con alerta | Adelanto mediano (s) | Con alerta antes del cambio | Sin alerta |
|---|---|---|---|---|---|---|---|---|
| CfC | 0.10 (final) | 0.723 | 0.000 | 0.187 +- 0.038 | 0.803 | -6.8 | 1.3 de 6 | 0.3 |
| CfC | 0.05 | 0.912 | 0.000 | 0.040 | 0.591 | -10.7 | 0.3 de 6 | 3.0 |
| GRU | 0.10 (final) | 0.614 | 0.000 | 0.107 +- 0.038 | 0.848 | -4.0 | 2.0 de 6 | 0.3 |
| GRU | 0.05 | 0.968 | 0.000 | 0.013 | 0.606 | -19.8 | 0.0 de 6 | 2.7 |
| LSTM | 0.10 (final) | 0.599 | 0.000 | 0.093 +- 0.019 | 0.833 | -8.7 | 1.7 de 6 | 0.0 |
| LSTM | 0.05 | 0.978 | 0.000 | 0.027 | 0.530 | -24.1 | 0.0 de 6 | 3.3 |
| gradient boosting | 0.10 (final) | 0.869 | 0.071 | 0.360 | 0.864 | +0.4 | 3 de 6 | 0 |
| gradient boosting | 0.05 | 0.989 | 0.000 | 0.000 | 0.727 | -18.4 | 1 de 6 | 1 |
| regresion logistica | 0.10 (final) | 0.832 | 0.000 | 0.080 | 0.682 | -9.7 | 1 de 6 | 1 |
| regresion logistica | 0.05 | 0.971 | 0.000 | 0.000 | 0.409 | +1.6 | 1 de 6 | 5 |

### Lectura de los resultados

- **Con el vector del propio pipeline, la tasa 1e-3 y 60 epocas, el GRU sube a F1 0.621 y kappa 0.698** (antes 0.556 y 0.582). Su kappa contra la IA supera el kappa humano contra IA (0.426), con la salvedad de que son videos distintos.
- **La histeresis reduce los cambios de nivel de 3.4 - 5.3 a unos 2.2 por video y cuesta alrededor de 0.01 de F1** en los modelos temporales. La etiqueta tiene 0.13: los modelos siguen cambiando de nivel mucho mas que la etiqueta.
- **El gradient boosting con histeresis es el mejor en F1 macro (0.653)**, porque la histeresis le quita el parpadeo (12.8 a 2.3 cambios) y conserva el incipiente (0.449). El GRU es mejor en exactitud (0.732 contra 0.720) y kappa (0.679 contra 0.598), con 3 semillas contra 1 corrida.
- **El CfC sigue por debajo del GRU** con 3 semillas (F1 0.536 contra 0.612; kappa 0.529 contra 0.679). La LSTM queda entre los dos y es la mas variable.
- **La alerta no adelanta el cambio a incontrolado.** Con alfa 0.10 el adelanto mediano es negativo en los tres modelos temporales (de -4 a -9 s) y solo 1 a 2 de los 6 videos tienen la alerta antes del cambio. Son solo 6 videos y el instante es el de la IA.
- **Falsas alarmas:** 0 de 28 en los videos sin fuego para los modelos temporales; en los controlados, de 9 a 19 % con alfa 0.10 (el nivel nominal es 10 %). El gradient boosting da 7 % en sin fuego y 36 % en controlados con alfa 0.10.
- Con alfa 0.05 el umbral sube a 0.91 - 0.98 y la alerta se pierde en unos 3 de los 6 videos con cambio.

### Variantes de la GRU para el incipiente (`python entrenar_variantes.py`)

Misma GRU, tasa 1e-3, 60 epocas, semillas 42, 43 y 44. La variante base reproduce exactamente las corridas finales de la GRU. En la variante b el peso del incipiente pasa de 2.88 a 5.76 (se dobla despues del tope de 5).

| Variante | F1 macro antes | F1 macro con histeresis | F1 incipiente antes | F1 incipiente con histeresis | Kappa antes | Kappa con histeresis |
|---|---|---|---|---|---|---|
| base (CORAL, tres perdidas) | 0.621 +- 0.008 | 0.612 +- 0.002 | 0.208 +- 0.052 | 0.209 +- 0.040 | 0.698 +- 0.022 | 0.679 +- 0.026 |
| a. softmax con entropia cruzada | 0.623 +- 0.059 | 0.611 +- 0.057 | 0.365 +- 0.060 | 0.346 +- 0.060 | 0.605 +- 0.083 | 0.592 +- 0.080 |
| b. doble peso para el incipiente | 0.630 +- 0.022 | 0.615 +- 0.021 | 0.317 +- 0.073 | 0.303 +- 0.070 | 0.652 +- 0.028 | 0.633 +- 0.023 |
| c. solo la perdida de nivel | 0.609 +- 0.023 | 0.604 +- 0.018 | 0.225 +- 0.054 | 0.239 +- 0.042 | 0.658 +- 0.017 | 0.638 +- 0.016 |

- **a y b mejoran el incipiente (de 0.21 a 0.35 y 0.30 con histeresis) pero no el F1 macro, y bajan el kappa** (0.679 a 0.592 y 0.633). Lo que se gana en incipiente se pierde en los otros dos niveles.
- El softmax es mucho mas variable entre semillas (F1 de 0.560 a 0.701).
- **c no mejora nada:** quitar las salidas de riesgo y tau no ayuda al nivel (y baja el kappa de 0.698 a 0.658), asi que las tres perdidas no se estorban.
- Con 3 semillas y 303 s de incipiente en validacion, las diferencias de F1 macro entre variantes estan dentro del ruido.

### Limitaciones que se declaran

- Validacion pequena: 47 videos de fuego, 303 s de incipiente y 6 videos con cambio a incontrolado.
- La tasa elegida es el borde superior de la rejilla.
- Calibracion se uso para el umbral conformal y, solo como comparacion, para los umbrales de CORAL.
- Las referencias sin tiempo son una sola corrida.

### Comandos

```
python escena_v2.py
python reentrenar_escena.py
python comparar_escena.py --datos runs_fuego/escena_v2 --modelos resultados/modelos_escena_v2/final
python fase_e.py
python entrenar_variantes.py --variante softmax --semilla 42
```

### Archivos

- `reentrenar_escena.py`, `fase_e.py`, `entrenar_variantes.py` (nuevos); `escena_v2.py` (opcion `--detecciones`).
- `resultados/escena_v2/` (resumen, videos y normalizacion).
- `resultados/modelos_escena_v2/`: `tasas.json`, `tasas/lr_*/` (busqueda) y `final/` (9 corridas, `comparacion.md`, `curvas.png`, `referencias.json`).
- `resultados/fase_e/`: `resumen.md`, un `.json` por corrida y `segmentos/`.
- `resultados/variantes_gru/`: `resumen.md` y un `.json` por corrida.
- No versionados: `runs_fuego/descriptores_v3/`, `runs_fuego/escena_v2/`, `runs_fuego/modelos_escena_v2/`.

## Configuracion final congelada antes de la prueba (2026-11-13)

Todo en validacion y calibracion; la prueba no se uso. Scripts nuevos: `entrenar_final.py` (entrenamiento con `--factor_incipiente` y `--riesgo escalon|anticipado`), `fase_e_final.py` (fase E final, con el adelanto tambien en calibracion) y `configuracion_final.py` (etapas `tasas`, `riesgo`, `final` y `congelar`). No se modifico ningun script anterior.

Comandos:

```
python configuracion_final.py --etapa tasas
python configuracion_final.py --etapa riesgo
python fase_e_final.py --pesos runs_fuego/modelos_final/riesgo/escalon --salida resultados/fase_e_final/riesgo_escalon --sin_referencias
python fase_e_final.py --pesos runs_fuego/modelos_final/riesgo/anticipado --salida resultados/fase_e_final/riesgo_anticipado --sin_referencias
python configuracion_final.py --etapa final --riesgo escalon
python fase_e_final.py --guardar_referencias runs_fuego/modelos_final/referencias
python configuracion_final.py --etapa congelar --riesgo escalon
```

### 1. Peso del incipiente

Doble peso al incipiente en los tres modelos: pesos de la perdida de nivel 0.541 / 5.759 / 1.246 (controlado / incipiente / incontrolado); el factor 2 se aplica despues del tope de 5. La GRU con tasa 1e-3 y semilla 42 reproduce exactamente la variante b de `entrenar_variantes.py` (F1 suavizado 0.6542, epoca 41).

### 2. Tasa (semilla 42, 60 epocas, peso doble, riesgo en escalon)

| modelo | F1 suavizado con 1e-3 | con 3e-3 | elegida | F1 incipiente (1e-3 / 3e-3) | kappa (1e-3 / 3e-3) | minutos por corrida |
|---|---|---|---|---|---|---|
| CfC (CPU) | 0.4786 | 0.4875 | 3e-3 | 0.226 / 0.293 | 0.417 / 0.430 | 17.3 en la busqueda, 14.4 en las semillas 43 y 44 |
| GRU | 0.6542 | 0.6596 | 3e-3 | 0.392 / 0.436 | 0.658 / 0.618 | 1.1 |
| LSTM | 0.5373 | 0.5593 | 3e-3 | 0.255 / 0.336 | 0.494 / 0.452 | 1.1 |

Los tres eligen 3e-3, otra vez el valor mas alto probado. Las diferencias son pequenas (0.005 a 0.022 de F1 suavizado, con una sola semilla) y en la GRU y la LSTM el kappa baja con 3e-3.

### 3. Riesgo anticipado (solo GRU, 3 semillas, tasa 3e-3)

Meta anticipada: en los videos que llegan a incontrolado (`tau_mascara = 1`), antes del cambio el riesgo es `max(escalon, 1 - tau / 60)` recortado a [0, 1]. Alerta con alfa 0.10; 6 videos con cambio en validacion y 5 en calibracion.

| meta de riesgo | umbral | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | adelanto mediano en validacion (s) | antes del cambio en validacion | adelanto mediano en validacion + calibracion (s) | antes del cambio en validacion + calibracion |
|---|---|---|---|---|---|---|---|---|
| escalon | 0.768 +- 0.113 | 0.000 | 0.147 +- 0.050 | 0.833 | -2.7 +- 2.1 | 1.7 de 6 | -2.2 +- 2.5 | 4.0 de 11 |
| anticipado | 0.890 +- 0.082 | 0.000 | 0.093 +- 0.019 | 0.788 | -5.2 +- 4.2 | 1.0 de 6 | -4.2 +- 3.4 | 2.7 de 11 |

**Decision (regla fijada antes de mirar): el riesgo anticipado no adelanta la alerta, asi que se queda el escalon.** Con la meta anticipada el umbral conformal sube (0.77 a 0.89), bajan las falsas alarmas en controlados y la alerta llega mas tarde. Efecto lateral no buscado: el nivel de la GRU mejora con la meta anticipada (F1 macro con histeresis 0.665 +- 0.003 contra 0.641 +- 0.009; kappa 0.657 contra 0.615); no se adopta porque el criterio acordado era la alerta. Detalle en `resultados/fase_e_final/riesgo_escalon/` y `riesgo_anticipado/`.

### 4. Fase E final en validacion (umbral 0.5, histeresis de 2 s, alerta con alfa 0.10)

3 semillas (42, 43, 44) por modelo temporal. La etiqueta tiene 0.13 cambios de nivel por video.

| modelo | etapa | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | cambios por video |
|---|---|---|---|---|---|---|---|---|
| CfC | antes | 0.497 +- 0.024 | 0.467 | 0.299 | 0.726 | 0.526 | 0.436 | 5.11 |
| CfC | final | 0.485 +- 0.034 | 0.432 | 0.296 | 0.725 | 0.513 | 0.426 | 2.27 |
| GRU | antes | 0.660 +- 0.008 | 0.760 | 0.416 | 0.805 | 0.714 | 0.641 | 2.52 |
| GRU | final | 0.641 +- 0.009 | 0.738 | 0.397 | 0.789 | 0.690 | 0.615 | 1.89 |
| LSTM | antes | 0.552 +- 0.011 | 0.551 | 0.313 | 0.791 | 0.607 | 0.498 | 3.87 |
| LSTM | final | 0.536 +- 0.014 | 0.528 | 0.302 | 0.777 | 0.587 | 0.478 | 2.23 |
| gradient boosting | antes | 0.622 | 0.623 | 0.444 | 0.799 | 0.679 | 0.519 | 12.81 |
| gradient boosting | final | 0.653 | 0.664 | 0.449 | 0.847 | 0.720 | 0.598 | 2.30 |
| regresion logistica | antes | 0.482 | 0.468 | 0.285 | 0.691 | 0.514 | 0.357 | 12.28 |
| regresion logistica | final | 0.479 | 0.488 | 0.268 | 0.681 | 0.509 | 0.371 | 3.13 |

Alerta (28 videos sin fuego, 25 controlados y 22 con propagacion en validacion; 6 + 5 videos con cambio a incontrolado):

| modelo | alfa | umbral | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | adelanto mediano en validacion (s) | antes del cambio en validacion | adelanto mediano en validacion + calibracion (s) | antes del cambio en validacion + calibracion | sin alerta |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.10 (final) | 0.777 | 0.000 | 0.133 | 0.818 | -6.7 +- 4.1 | 2.0 de 6 | -3.2 +- 5.2 | 4.3 de 11 | 0.7 |
| GRU | 0.10 (final) | 0.768 | 0.000 | 0.147 | 0.833 | -2.7 +- 2.1 | 1.7 de 6 | -2.2 +- 2.5 | 4.0 de 11 | 1.7 |
| LSTM | 0.10 (final) | 0.907 | 0.000 | 0.093 | 0.758 | -11.7 +- 6.0 | 0.3 de 6 | -9.0 +- 6.2 | 2.3 de 11 | 1.0 |
| gradient boosting | 0.10 (final) | 0.869 | 0.071 | 0.360 | 0.864 | 0.4 | 3 de 6 | 1.3 | 6 de 11 | 0 |
| regresion logistica | 0.10 (final) | 0.832 | 0.000 | 0.080 | 0.682 | -9.7 | 1 de 6 | -10.4 | 2 de 11 | 3 |
| CfC | 0.05 | 0.973 | 0.000 | 0.080 | 0.621 | -14.6 | 0.3 de 6 | -13.4 | 1.3 de 11 | 4.0 |
| GRU | 0.05 | 0.998 | 0.000 | 0.027 | 0.606 | -32.0 | 0 de 6 | -32.0 | 1.0 de 11 | 6.0 |
| LSTM | 0.05 | 0.993 | 0.000 | 0.053 | 0.606 | -21.3 | 0 de 6 | -18.8 | 1.7 de 11 | 3.3 |

Comparacion con la configuracion anterior (peso simple, tasa 1e-3; valores finales con histeresis):

| modelo | F1 macro antes -> ahora | F1 incipiente | kappa |
|---|---|---|---|
| CfC | 0.536 -> 0.485 | 0.169 -> 0.296 | 0.529 -> 0.426 |
| GRU | 0.612 -> 0.641 | 0.209 -> 0.397 | 0.679 -> 0.615 |
| LSTM | 0.548 -> 0.536 | 0.156 -> 0.302 | 0.563 -> 0.478 |

Lectura:
- El incipiente sube en los tres modelos (de 0.16-0.21 a 0.30-0.40), que era el objetivo.
- Solo la GRU mejora en F1 macro; el CfC y la LSTM bajan en F1 macro, y el kappa baja en los tres (se pierde en controlado).
- El CfC queda casi empatado con la regresion logistica en F1 macro (0.485 contra 0.479 con histeresis) y lejos de la GRU.
- El gradient boosting con histeresis sigue con el mejor F1 macro (0.653 contra 0.641 de la GRU), pero con 36 % de falsas alarmas en controlados y 7 % en videos sin fuego.
- La alerta sigue sin adelantar el cambio a incontrolado: el adelanto mediano es negativo en los tres modelos temporales tambien con los 11 videos; con la GRU la alerta llega antes del cambio en 4 de 11.
- Las falsas alarmas en controlados (9 a 15 %) quedan cerca del 10 % nominal; ninguna en videos sin fuego.

### 5. Configuracion congelada

`resultados/configuracion_final.json`: tasas (3e-3 en los tres), pesos de nivel, meta de riesgo (escalon), umbrales de CORAL 0.5, histeresis 2 s, alfa 0.10 (y 0.05 como secundario), umbral conformal de cada corrida, factor de desactivacion 0.8, semillas 42/43/44, hiperparametros, y la ruta y el SHA-256 de cada archivo de pesos (9 modelos en `runs_fuego/modelos_final/final/` y las 2 referencias en `runs_fuego/modelos_final/referencias/`). Despues de esto no se cambia nada antes de la prueba.

Archivos generados: `resultados/modelos_final/` (`tasas.json`, `tasas/`, `riesgo/`, `final/`), `resultados/fase_e_final/` (`resumen.md`, `resumen.json`, un `.json` por corrida, `segmentos/`, `riesgo_escalon/`, `riesgo_anticipado/`), `resultados/configuracion_final.json`. Los pesos van en `runs_fuego/modelos_final/` (no se versiona).

## Configuracion final por modelo, congelada de nuevo antes de la prueba (2026-11-13)

Reemplaza a la configuracion congelada de la seccion anterior. Todo en validacion y calibracion; la prueba no se uso. Scripts nuevos: `entrenar_unidad_dt.py` (envuelve a `entrenar_final.py` y divide el intervalo del modelo entre `--unidad_dt`), `fase_e_elegida.py` (fase E con la corrida elegida de cada modelo) y `congelar_configuracion.py` (etapas `elegir` y `congelar`). No se modifico ningun script anterior.

Comandos:

```
python entrenar_unidad_dt.py --modelo cfc --semilla 42 --lr 1e-3 --factor_incipiente 1.0 --riesgo escalon --dispositivo cpu
python fase_e_elegida.py --carpetas "runs_fuego/modelos_final/cfc_dt/cfc_s*" --sin_referencias --salida resultados/fase_e_elegida/cfc_dt
python congelar_configuracion.py --etapa elegir
python fase_e_elegida.py --guardar_referencias runs_fuego/modelos_final/referencias
python congelar_configuracion.py --etapa congelar
```

### 1. Eleccion por modelo

Aplicar el doble peso y 3e-3 a los tres se habia decidido mirando solo la GRU. Ahora cada modelo elige entre las dos configuraciones que ya tenian 3 semillas, con el mismo criterio: F1 macro final (umbral 0.5 + histeresis) en validacion, media de las semillas 42, 43 y 44.

| modelo | A: peso simple, 1e-3 | B: doble peso, 3e-3 | elegida |
|---|---|---|---|
| CfC | 0.536 +- 0.019 (0.515, 0.532, 0.561) | 0.485 +- 0.034 (0.475, 0.531, 0.448) | A |
| GRU | 0.612 +- 0.002 (0.615, 0.611, 0.610) | 0.641 +- 0.009 (0.641, 0.630, 0.653) | B |
| LSTM | 0.548 +- 0.048 (0.487, 0.551, 0.606) | 0.536 +- 0.014 (0.548, 0.543, 0.516) | A |

Se confirma GRU = B, CfC = A y LSTM = A. En la LSTM la diferencia (0.012) es menor que la desviacion entre semillas de A (0.048). La eleccion queda en `resultados/modelos_final/eleccion.json`.

### 2. Unidad de tiempo del CfC

En `CfCCell` el intervalo multiplica el termino de tiempo de la compuerta de interpolacion; con dt de 0.1 s ese termino queda cerca de cero. Se probo el CfC con su configuracion elegida (A) y `timespans = dt / 0.1`, semilla 42, 60 epocas en CPU (11.0 min).

| CfC, semilla 42 | F1 macro antes | F1 macro final | kappa final | mejor epoca |
|---|---|---|---|---|
| dt en segundos (actual) | 0.530 | 0.515 | 0.473 | 26 |
| dt / 0.1 | 0.500 | 0.483 | 0.440 | 31 |

**No mejora el F1 macro final (0.483 contra 0.515), asi que el CfC se queda con dt en segundos** y no se corrieron las semillas 43 y 44. La corrida queda en `resultados/modelos_final/cfc_dt/` y `resultados/fase_e_elegida/cfc_dt/`.

### 3. Busqueda de tasa cerrada

La busqueda de tasa termina aqui. Las diferencias entre 1e-3 y 3e-3 con la semilla 42 (0.005 a 0.022 de F1 suavizado) estan dentro de la variacion entre semillas (desviacion de 0.002 a 0.048 en el F1 final). No se prueban mas tasas.

### 4. Riesgo

El riesgo se queda en escalon, por la regla acordada (el riesgo anticipado no adelanto la alerta). **Hallazgo, no adoptado:** en la GRU la meta anticipada `max(escalon, 1 - tau / 60)` mejoro el nivel (F1 macro final 0.665 +- 0.003 contra 0.641 +- 0.009; kappa 0.657 contra 0.615; 3 semillas), aunque atraso la alerta (adelanto mediano en validacion + calibracion -4.2 s contra -2.2 s).

### 5. Tabla final de validacion con la configuracion de cada modelo

3 semillas por modelo temporal. La etiqueta tiene 0.13 cambios de nivel por video.

| modelo | configuracion | etapa | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | cambios por video |
|---|---|---|---|---|---|---|---|---|---|
| CfC | A | antes | 0.544 +- 0.015 | 0.711 | 0.170 | 0.751 | 0.644 | 0.543 | 5.26 |
| CfC | A | final | 0.536 +- 0.019 | 0.706 | 0.169 | 0.734 | 0.629 | 0.529 | 2.16 |
| GRU | B | antes | 0.660 +- 0.008 | 0.760 | 0.416 | 0.805 | 0.714 | 0.641 | 2.52 |
| GRU | B | final | 0.641 +- 0.009 | 0.738 | 0.397 | 0.789 | 0.690 | 0.615 | 1.89 |
| LSTM | A | antes | 0.552 +- 0.048 | 0.738 | 0.137 | 0.782 | 0.685 | 0.583 | 4.76 |
| LSTM | A | final | 0.548 +- 0.048 | 0.726 | 0.156 | 0.763 | 0.663 | 0.563 | 2.71 |
| gradient boosting | - | antes | 0.622 | 0.623 | 0.444 | 0.799 | 0.679 | 0.519 | 12.81 |
| gradient boosting | - | final | 0.653 | 0.664 | 0.449 | 0.847 | 0.720 | 0.598 | 2.30 |
| regresion logistica | - | antes | 0.482 | 0.468 | 0.285 | 0.691 | 0.514 | 0.357 | 12.28 |
| regresion logistica | - | final | 0.479 | 0.488 | 0.268 | 0.681 | 0.509 | 0.371 | 3.13 |

Alerta (28 videos sin fuego, 25 controlados y 22 con propagacion en validacion; 6 + 5 videos con cambio a incontrolado):

| modelo | alfa | umbral | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | adelanto mediano en validacion (s) | antes del cambio en validacion | adelanto mediano en validacion + calibracion (s) | antes del cambio en validacion + calibracion | sin alerta |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.10 (final) | 0.723 | 0.000 | 0.187 | 0.803 | -6.8 +- 1.8 | 1.3 de 6 | -4.3 +- 3.6 | 3.3 de 11 | 0.3 |
| GRU | 0.10 (final) | 0.768 | 0.000 | 0.147 | 0.833 | -2.7 +- 2.1 | 1.7 de 6 | -2.2 +- 2.5 | 4.0 de 11 | 1.7 |
| LSTM | 0.10 (final) | 0.599 | 0.000 | 0.093 | 0.833 | -8.7 +- 8.4 | 1.7 de 6 | -1.7 +- 0.9 | 4.7 de 11 | 0.0 |
| gradient boosting | 0.10 (final) | 0.869 | 0.071 | 0.360 | 0.864 | 0.4 | 3 de 6 | 1.3 | 6 de 11 | 0 |
| regresion logistica | 0.10 (final) | 0.832 | 0.000 | 0.080 | 0.682 | -9.7 | 1 de 6 | -10.4 | 2 de 11 | 3 |
| CfC | 0.05 | 0.912 | 0.000 | 0.040 | 0.591 | -10.7 | 0.3 de 6 | -28.7 | 1.0 de 11 | 5.7 |
| GRU | 0.05 | 0.998 | 0.000 | 0.027 | 0.606 | -32.0 | 0 de 6 | -32.0 | 1.0 de 11 | 6.0 |
| LSTM | 0.05 | 0.978 | 0.000 | 0.027 | 0.530 | -24.1 | 0 de 6 | -27.8 | 0.3 de 11 | 6.7 |

Lectura:
- La GRU es el mejor modelo temporal en F1 macro (0.641) y kappa (0.615); el CfC (0.536, kappa 0.529) y la LSTM (0.548, kappa 0.563) quedan por debajo y cerca entre si.
- El gradient boosting con histeresis tiene el mejor F1 macro (0.653), con 36 % de falsas alarmas en controlados y 7 % en videos sin fuego.
- El incipiente sigue debil en el CfC y la LSTM (0.17 y 0.16), que usan peso simple; en la GRU es 0.40.
- La alerta sigue sin adelantar el cambio a incontrolado: adelanto mediano negativo en los tres modelos temporales, con alerta antes del cambio en 3 a 5 de 11 videos.
- Falsas alarmas: ninguna en videos sin fuego; en controlados, 9 a 19 % (nominal 10 %).

### 6. Configuracion congelada

`resultados/configuracion_final.json` (sobrescrito): por modelo, configuracion elegida, tasa, pesos de nivel, meta de riesgo (escalon), intervalo (dt en segundos), la prueba de la unidad de tiempo del CfC y el F1 de cada configuracion; decision (umbrales 0.5, histeresis 2 s, alfa 0.10 y 0.05, factor de desactivacion 0.8); por corrida, semilla, mejor epoca, umbral conformal, ruta y SHA-256 de los pesos. Pesos: CfC y LSTM en `runs_fuego/modelos_escena_v2/final/`, GRU en `runs_fuego/modelos_final/final/`, referencias en `runs_fuego/modelos_final/referencias/`. Despues de esto no se cambia nada antes de la prueba.

Archivos generados: `resultados/modelos_final/eleccion.json`, `resultados/modelos_final/cfc_dt/`, `resultados/fase_e_elegida/` (`resumen.md`, un `.json` por corrida, `segmentos/`, `cfc_dt/`), `resultados/configuracion_final.json`.

## Fase F: evaluacion en la prueba, velocidad y demo (2026-11-13)

La prueba se evaluo **una sola vez**, con la configuracion congelada en `resultados/configuracion_final.json` (CfC y LSTM con la configuracion A, GRU con la B; umbral 0.5, histeresis de 2 s, alerta con alfa 0.10). Antes de correrla, `evaluar_prueba.py` se ensayo en validacion y reprodujo exactamente los numeros de la fase E congelada. No se cambio nada despues de ver la prueba. `evaluar_prueba.py` se niega a correr de nuevo si ya existe `resultados/prueba/metricas.json`, y comprueba el SHA-256 de cada archivo de pesos contra la configuracion.

Scripts nuevos: `evaluar_prueba.py` y `sistema.py` (sistema en vivo: clases `Sistema` y `ModeloEnVivo`, modos `fps`, `demo` y `verificar`). No se modifico ningun script anterior.

Comandos:

```
python evaluar_prueba.py
python sistema.py --modo fps
python sistema.py --modo demo --corridas gru_s42 --archivos 19151618 12487581 --nombres vela basural --titulos "sistema completo (gru): vela" "sistema completo (gru): basural"
```

Correccion previa (punto 0): en la seccion de decisiones del 2026-11-12 se corrigio la frase que daba a la IA como referencia principal para la alerta. En la revision de las 23 diferencias la IA tuvo razon en 13 y el humano en 10; ninguna referencia es superior y las dos pesan igual.

### Videos y definiciones

- Prueba: 70 videos de fuego con referencia de la IA, 69 con referencia humana y 32 sin fuego. De los 69, 29 tienen tramo util y cambios marcados reproduciendo el video (`prueba_cambios`; el humano descarto 1 de los 30) y 40 tienen un solo nivel para todo el video (`prueba_humana`, etiquetados con la tira).
- Contra la referencia humana, el nivel por segundo usa los 69; la localizacion, los cambios, el tiempo al peligro y el adelanto usan solo los 29.
- Nivel por segundo: nivel final, mayoria de los fotogramas del segundo, dentro del tramo util de la referencia. Un segundo sin fotogramas cuenta como controlado.
- Clase mayoritaria: incontrolado (la mas frecuente por segundo en entrenamiento). No tiene alerta.
- mAP: segmentos de la referencia y del modelo construidos desde el nivel por segundo, por nivel; puntuacion de un segmento = probabilidad media de su nivel (en CORAL, P(controlado) = 1 - p1, P(incontrolado) = min(p1, p2), P(incipiente) = p1 - P(incontrolado)); AP con interpolacion de todos los puntos.
- Cambios: un cambio a un nivel esta detectado si el nivel final del modelo cruza hacia ese nivel o mas dentro del tramo util; el error es el cruce mas cercano menos el instante de la referencia. Cambios de mas = transiciones del modelo no emparejadas.
- Bootstrap: 1000 remuestreos de videos, semilla 42; en los modelos con 3 semillas se promedia la metrica de las semillas dentro de cada remuestreo; las diferencias usan los mismos remuestreos.
- Estas definiciones las fije yo antes de correr la prueba; no son parametros ajustados.

### a. Nivel por segundo (nivel final)

Referencia de la IA (70 videos, 4 703 s):

| modelo | F1 macro | IC 95 % | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | IC 95 % | cambios por video |
|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.460 +- 0.002 | 0.393 a 0.522 | 0.507 | 0.252 | 0.621 | 0.478 | 0.319 +- 0.036 | 0.183 a 0.442 | 2.35 |
| GRU | 0.519 +- 0.009 | 0.433 a 0.603 | 0.561 | 0.386 | 0.609 | 0.524 | 0.371 +- 0.029 | 0.225 a 0.516 | 1.79 |
| LSTM | 0.460 +- 0.021 | 0.396 a 0.526 | 0.536 | 0.232 | 0.611 | 0.491 | 0.323 +- 0.022 | 0.204 a 0.438 | 2.83 |
| gradient boosting | 0.442 | 0.368 a 0.525 | 0.510 | 0.212 | 0.605 | 0.476 | 0.292 | 0.178 a 0.413 | 3.11 |
| regresion logistica | 0.491 | 0.407 a 0.570 | 0.475 | 0.387 | 0.610 | 0.490 | 0.311 | 0.158 a 0.449 | 2.86 |
| clase mayoritaria | 0.180 | 0.135 a 0.219 | 0.000 | 0.000 | 0.541 | 0.371 | 0.000 | - | 0.00 |

Referencia humana (69 videos, 4 669 s):

| modelo | F1 macro | IC 95 % | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | IC 95 % |
|---|---|---|---|---|---|---|---|---|
| CfC | 0.475 +- 0.022 | 0.410 a 0.540 | 0.587 | 0.328 | 0.510 | 0.477 | 0.345 +- 0.057 | 0.241 a 0.449 |
| GRU | 0.497 +- 0.019 | 0.416 a 0.579 | 0.570 | 0.419 | 0.504 | 0.492 | 0.352 +- 0.040 | 0.238 a 0.472 |
| LSTM | 0.460 +- 0.007 | 0.391 a 0.528 | 0.607 | 0.282 | 0.493 | 0.475 | 0.340 +- 0.013 | 0.244 a 0.440 |
| gradient boosting | 0.451 | 0.362 a 0.531 | 0.597 | 0.293 | 0.463 | 0.461 | 0.320 | 0.218 a 0.426 |
| regresion logistica | 0.527 | 0.433 a 0.604 | 0.567 | 0.475 | 0.540 | 0.523 | 0.365 | 0.220 a 0.485 |
| clase mayoritaria | 0.100 | 0.064 a 0.138 | 0.001 | 0.000 | 0.299 | 0.176 | 0.000 | - |

Solo los 29 videos humanos con cambios marcados: CfC 0.428 +- 0.018 (kappa 0.293), GRU 0.446 +- 0.026 (0.269), LSTM 0.405 +- 0.026 (0.286), gradient boosting 0.429 (0.306), regresion logistica 0.525 (0.353).

Diferencias pareadas (bootstrap):

| diferencia | referencia | F1 macro | IC 95 % | kappa | IC 95 % |
|---|---|---|---|---|---|
| GRU - CfC | IA | +0.059 | -0.006 a +0.128 | +0.052 | -0.038 a +0.153 |
| GRU - gradient boosting | IA | +0.077 | -0.008 a +0.151 | +0.079 | -0.054 a +0.207 |
| GRU - CfC | humana | +0.023 | -0.037 a +0.088 | +0.007 | -0.078 a +0.094 |
| GRU - gradient boosting | humana | +0.046 | -0.039 a +0.137 | +0.032 | -0.084 a +0.158 |

Las matrices de confusion de cada modelo contra cada referencia estan en `resultados/prueba/resumen.md`.

### b. Localizacion temporal (mAP de segmentos, promedio de los tres niveles)

| modelo | IA, tIoU 0.3 | IA, 0.5 | IA, 0.7 | humana (29), tIoU 0.3 | humana, 0.5 | humana, 0.7 |
|---|---|---|---|---|---|---|
| CfC | 0.289 +- 0.033 | 0.216 +- 0.039 | 0.159 +- 0.036 | 0.187 +- 0.040 | 0.103 +- 0.048 | 0.044 +- 0.033 |
| GRU | 0.403 +- 0.023 | 0.299 +- 0.023 | 0.235 +- 0.006 | 0.296 +- 0.038 | 0.136 +- 0.015 | 0.066 +- 0.037 |
| LSTM | 0.329 +- 0.018 | 0.258 +- 0.022 | 0.194 +- 0.022 | 0.218 +- 0.026 | 0.120 +- 0.032 | 0.042 +- 0.009 |
| gradient boosting | 0.246 | 0.204 | 0.130 | 0.193 | 0.125 | 0.028 |
| regresion logistica | 0.277 | 0.204 | 0.147 | 0.172 | 0.069 | 0.021 |
| clase mayoritaria | 0.116 | 0.091 | 0.043 | 0.035 | 0.006 | 0.000 |

Por nivel (en `resumen.md`): el AP del incipiente es el mas bajo en todos (GRU, referencia de la IA: 0.207 / 0.060 / 0.015).

### c. Cambios de nivel

Cambio a incontrolado:

| modelo | IA: detectados de 21 | mediana del error absoluto (s) | mediana con signo (s) | dentro de 10 s | no llega | humana: detectados de 16 | mediana del error absoluto (s) | mediana con signo (s) | dentro de 10 s | no llega |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 14.3 | 16.9 | +16.5 | 4.0 | 5.0 | 12.7 | 13.6 | -5.1 | 6.3 | 2.3 |
| GRU | 16.0 | 27.8 | +11.2 | 5.0 | 5.0 | 13.7 | 17.5 | -0.5 | 4.0 | 2.0 |
| LSTM | 15.7 | 17.4 | +14.3 | 5.3 | 5.3 | 13.3 | 17.6 | -6.8 | 4.7 | 2.7 |
| gradient boosting | 19 | 8.3 | +4.2 | 10 | 0 | 14 | 4.8 | -3.1 | 9 | 1 |
| regresion logistica | 15 | 12.9 | +12.9 | 5 | 5 | 13 | 7.4 | -6.7 | 8 | 2 |

Cambio a incipiente (11 segun la IA, 4 segun el humano): GRU detecta 6.7 de 11 con mediana de 4.2 s y en 4.3 ya estaba en incipiente o mas desde el inicio; CfC 8.0 de 11 (11.6 s); LSTM 9.0 (14.8 s); gradient boosting 10 (9.3 s); regresion logistica 9 (16.5 s).

Cambios de mas por video (referencia de la IA / humana): CfC 2.17 / 3.98, GRU 1.51 / 2.32, LSTM 2.53 / 4.41, gradient boosting 2.71 / 5.10, regresion logistica 2.53 / 5.07.

Los 4 videos con paso de controlado a incipiente (instante del primer paso del modelo a incipiente o mas; semillas 42 / 43 / 44):

| video | IA | humano | CfC | GRU | LSTM | gradient boosting | regresion logistica |
|---|---|---|---|---|---|---|---|
| 1. `700_F_1633820181...` | controlado, incipiente en 25.6 s | controlado, sin cambio | desde el inicio (3) | desde el inicio (3) | desde el inicio (3) | 14.5 | desde el inicio |
| 2. `Har du en fungerande brandvarnare_` | controlado, incipiente en 16.3 s | incipiente, incontrolado en 20.5 s | 18.7 / nunca / desde el inicio | desde el inicio (3) | desde el inicio (3) | desde el inicio | desde el inicio |
| 3. `Wooden Model Ship Battleship Jean Bart...` | controlado, incipiente en 23.1 s | controlado, incipiente en 263.8 s | 17.0 / 251.9 / 108.3 | desde el inicio (3) | desde el inicio / 130.3 / desde el inicio | 4.5 | 4.5 |
| 4. `videoplayback (3)` | controlado, incipiente en 34.0 s | incipiente, sin cambio | 47.0 / 35.7 / desde el inicio | desde el inicio (3) | desde el inicio (3) | 8.5 | desde el inicio |

En los 4 la GRU marca incipiente o mas desde el inicio con las 3 semillas: no detecta el paso de controlado a incipiente. El humano y la IA solo coinciden en que hay ese cambio en 1 de los 4 (y difieren en 240 s).

### d. Tiempo al peligro (MAE de tau antes del cambio, en segundos)

| modelo | referencia de la IA (21 videos, 5 802 fotogramas) | referencia humana (16 videos, 10 293 fotogramas) |
|---|---|---|
| CfC | 18.9 +- 1.2 | 63.4 +- 2.8 |
| GRU | 25.7 +- 2.3 | 58.4 +- 0.7 |
| LSTM | 20.1 +- 1.9 | 60.1 +- 2.3 |
| constante (mediana de tau en entrenamiento, 52.7 s) | 30.4 | 48.1 |

Contra la referencia de la IA los tres modelos mejoran a la constante; contra la humana quedan peor que la constante.

### e. Alerta con alfa 0.10

| modelo | falsas alarmas sin fuego (de 32) | falsas alarmas en controlados, IA (de 30) | en controlados, humana (de 29) | propagacion con alerta, IA (de 40) | EWT mediana, IA (s) [cuartiles] | antes del cambio, IA (de 21) | sin alerta, IA | EWT mediana, humana (s) [cuartiles] | antes del cambio, humana (de 16) | sin alerta, humana |
|---|---|---|---|---|---|---|---|---|---|---|
| CfC | 0.000 | 0.233 +- 0.072 | 0.184 | 0.700 | +0.7 +- 3.8 [-34.1, +14.2] | 8.3 | 4.0 | +23.6 +- 8.7 [+4.0, +69.5] | 12.3 | 2.0 |
| GRU | 0.000 | 0.222 +- 0.031 | 0.207 | 0.750 | -2.7 +- 10.6 [-29.5, +14.4] | 8.3 | 3.3 | +20.5 +- 11.7 [-0.1, +67.1] | 11.3 | 1.3 |
| LSTM | 0.000 | 0.233 +- 0.072 | 0.230 | 0.708 | +4.4 +- 4.7 [-39.7, +14.0] | 9.0 | 4.7 | +29.7 +- 4.0 [+1.6, +69.6] | 10.7 | 2.3 |
| gradient boosting | 0.031 (1 de 32) | 0.267 | 0.207 | 0.775 | +3.8 [-15.8, +16.3] | 12 | 1 | +26.6 [+3.6, +94.5] | 13 | 1 |
| regresion logistica | 0.000 | 0.100 | 0.069 | 0.675 | -10.5 [-51.3, +1.7] | 5 | 5 | +13.2 [-3.3, +48.4] | 10 | 2 |

Alfa 0.05 (aparte): sin falsas alarmas en videos sin fuego; en controlados 2 a 9 % (IA); EWT mediana contra la IA de -9.6 s (CfC), -32.1 s (GRU) y -14.8 s (LSTM), con 9 a 13 de 21 videos sin alerta. Tabla completa en `resumen.md`.

### Lectura de la prueba

- **Todos los modelos bajan respecto a validacion**: GRU de 0.641 a 0.519 de F1 macro (kappa de 0.615 a 0.371), CfC de 0.536 a 0.460, LSTM de 0.548 a 0.460, gradient boosting de 0.653 a 0.442. La regresion logistica no baja (0.479 a 0.491).
- **La GRU es el mejor modelo contra la referencia de la IA** (F1 macro, kappa y mAP), pero ninguna diferencia pareada es significativa: los intervalos de GRU - CfC y GRU - gradient boosting incluyen el cero con las dos referencias.
- **Contra la referencia humana, la regresion logistica tiene el mejor F1 macro** (0.527 contra 0.497 de la GRU); los intervalos se solapan.
- **El CfC no supera a la GRU** en nivel ni en localizacion con ninguna referencia; si tiene el menor MAE de tau contra la IA (18.9 s).
- **Localizacion:** la GRU tiene el mejor mAP en los tres umbrales y con las dos referencias (0.299 con tIoU 0.5 contra la IA; 0.136 contra la humana). Contra la humana todos los valores son bajos.
- **Cambios:** el gradient boosting localiza mejor el cambio a incontrolado (mediana de 8.3 s y 4.8 s) que los modelos temporales (14 a 28 s), a costa de mas cambios de mas. Los modelos temporales no alcanzan el incontrolado en unos 5 de 21 videos.
- **Alerta:** ninguna falsa alarma en los 32 videos sin fuego con los modelos temporales. En controlados la tasa es de 18 a 23 %, el doble del 10 % nominal: la garantia conformal no se cumple en la prueba para los controlados (el umbral se calibro con 52 videos, mezcla de sin fuego y controlados).
- **El adelanto depende de la referencia:** contra la IA la alerta llega alrededor del cambio (mediana de -2.7 a +4.4 s; antes del cambio en 8 a 9 de 21); contra el humano llega antes (mediana de +20 a +30 s; antes en 11 a 12 de 16), porque los instantes humanos son mas tardios. Como ninguna referencia es superior, el adelanto real queda entre los dos valores.
- **Tiempo al peligro:** contra la referencia humana los modelos son peores que predecir una constante; el tau se entreno con los instantes de la IA.
- Limitaciones: un solo anotador humano, 29 videos con cambios humanos, 21 y 16 videos para el adelanto, y la referencia de entrenamiento es la de la IA.

### Velocidad del sistema completo en vivo

`sistema.py --modo fps`: 5 videos de prueba al azar (semilla 42), fotograma a fotograma con `VectorEscena`, YOLOv8s en GPU y modelo temporal en CPU (un paso por fotograma). Antes se verifico en 3 videos de validacion que el sistema en vivo reproduce los datos guardados: el vector de escena y el contexto coinciden exactamente (diferencia 0), las salidas del modelo paso a paso difieren en menos de 1e-6 de la inferencia por lotes, y el nivel y la alerta incrementales son iguales.

| video | resolucion | fps de la fuente | fotogramas | fps con GRU | fps con CfC | tiempo real |
|---|---|---|---|---|---|---|
| 1. `Brandgefahren im Haushalt` | 1920x1080 | 30 | 685 | 17.4 | 17.4 | 1.74x |
| 2. `14035491_2048_1080_30fps` | 2048x1080 | 29.97 | 599 | 11.1 | 11.2 | 1.11x |
| 3. `12588006_1920_1080_30fps` | 1920x1080 | 30 | 338 | 13.9 | 13.9 | 1.39x |
| 4. `fire19_clip_001` | 320x240 | 10 | 100 | 48.3 | 48.5 | 4.78x |
| 5. `1828453162` | 640x360 | 25 | 266 | 37.0 | 37.2 | 3.69x |
| **todos** | | | 1 988 | **15.7** | **15.7** | 1.56x |

Tiempo medio por etapa (ms por fotograma procesado): lectura 4.30, YOLO 24.36, camara y cortes 32.88, seguimiento 1.01, descriptores 0.32, vector de escena 0.24, modelo 0.63 (GRU) o 0.46 (CfC), decision 0.01.

- El sistema funciona en tiempo real (muestrea a 10 fps y procesa 15.7 por segundo de media; 11.1 en el peor video).
- El costo esta en el movimiento de camara (52 %) y en YOLO (38 %); el modelo temporal es menos del 1 %, asi que GRU y CfC dan los mismos fps.
- YOLO tarda aqui 24 ms por fotograma y no los 8.2 ms del detector solo, porque incluye el redimensionado de fotogramas de 1080p y la llamada por imagen.

### Demo

`runs_fuego/demo/sistema_vela.mp4` (16.9 MB) y `runs_fuego/demo/sistema_basural.mp4` (26.9 MB): GRU (semilla 42) con el sistema en vivo; cajas de fuego y humo, nivel, riesgo y alerta superpuestos, y un panel con la curva de riesgo, el umbral, la linea de niveles y la banda de alerta. No se publicaron ni se copiaron a `docs/`.

- Vela (`19151618`, 298 fotogramas): controlado en 275 fotogramas e incipiente en los primeros 23 (unos 2.3 s de arranque); riesgo maximo 0.279; sin alerta. Este video esta en la prueba.
- Basural (`12487581`, 285 fotogramas): incipiente 112 fotogramas e incontrolado 173; riesgo maximo 0.999; alerta desde los 7.1 s. La IA lo etiqueto como incipiente sin cambio; esta en calibracion (no entra en el umbral conformal).

### Archivos

`resultados/prueba/`: `resumen.md` (todas las tablas), `metricas.json`, `cambios.csv`, `alerta.csv`, `controlado_a_incipiente.csv`, `segmentos/` (segmentos `(t_inicio, t_fin, nivel)` de cada corrida), `velocidad.md`, `velocidad.json`. La salida de consola de la evaluacion esta en `runs_fuego/evaluar_prueba_salida.txt` (no se versiona).

## Analisis descriptivo de la prueba (2026-11-13)

`python analisis_prueba.py` (script nuevo). No cambia nada ni decide nada: repite la inferencia con la configuracion congelada (diferencia de F1 con `resultados/prueba/metricas.json`: 0) y desglosa los mismos resultados. Salida completa en `resultados/prueba/analisis.md`.

### a. Por subconjunto, referencia de la IA

| subconjunto | segundos (controlado / incipiente / incontrolado) | modelo | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | mAP tIoU 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| 40 al azar (`prueba_humana`) | 1112 / 219 / 315 | GRU | 0.541 +- 0.013 | 0.703 | 0.153 | 0.766 | 0.629 | 0.455 | 0.502 |
| 40 al azar | | CfC | 0.538 +- 0.028 | 0.678 | 0.184 | 0.753 | 0.612 | 0.418 | 0.444 |
| 40 al azar | | gradient boosting | 0.489 | 0.663 | 0.184 | 0.618 | 0.591 | 0.329 | 0.407 |
| 30 con cambios (`prueba_cambios`) | 352 / 1277 / 1428 | GRU | 0.435 +- 0.005 | 0.306 | 0.437 | 0.563 | 0.467 | 0.221 | 0.159 |
| 30 con cambios | | CfC | 0.387 +- 0.013 | 0.308 | 0.271 | 0.582 | 0.406 | 0.241 | 0.110 |
| 30 con cambios | | gradient boosting | 0.381 | 0.323 | 0.218 | 0.600 | 0.415 | 0.233 | 0.112 |

- En los 40 al azar la GRU y el CfC empatan en F1 macro (0.541 y 0.538); la ventaja de la GRU en los 70 viene de los 30 con cambios (0.435 contra 0.387), por el incipiente.
- Los 30 con cambios son mucho mas dificiles (F1 macro de 0.38 a 0.44, mAP de 0.11 a 0.16) y aportan el 65 % de los segundos de la prueba.

### b. Proporcion de niveles por segundo

| conjunto | referencia | videos de fuego | segundos | controlado | incipiente | incontrolado |
|---|---|---|---|---|---|---|
| entrenamiento | IA | 215 | 17 421 | 23.7 % | 13.0 % | 63.3 % |
| validacion | IA | 47 | 2 757 | 37.5 % | 11.0 % | 51.5 % |
| calibracion | IA | 45 | 4 515 | 20.7 % | 23.4 % | 55.9 % |
| prueba | IA | 70 | 4 703 | 31.1 % | 31.8 % | 37.1 % |
| prueba, solo los 40 al azar | IA | 40 | 1 646 | 67.6 % | 13.3 % | 19.1 % |
| prueba, solo los 30 con cambios | IA | 30 | 3 057 | 11.5 % | 41.8 % | 46.7 % |
| prueba | humana | 69 | 4 669 | 33.3 % | 49.1 % | 17.6 % |
| prueba, solo los 40 al azar | humana | 40 | 1 674 | 61.9 % | 23.5 % | 14.6 % |
| prueba, solo los 29 con cambios | humana | 29 | 2 995 | 17.4 % | 63.3 % | 19.3 % |

- La prueba tiene casi el triple de incipiente que validacion (31.8 % contra 11.0 %) y que entrenamiento (13.0 %), porque los 30 videos con cambios se eligieron entre los candidatos con crecimiento. El incipiente es el nivel que peor resuelven todos los modelos, asi que parte de la caida de validacion a prueba es por composicion.
- Segun el humano el incipiente es casi la mitad de la prueba (49.1 %) y el incontrolado solo el 17.6 %.

### c. Intervalo binomial del 95 % de las falsas alarmas (alfa 0.10, Clopper-Pearson, por corrida)

| corrida | sin fuego | IC 95 % | controlados (IA) | IC 95 % | controlados (humana) | IC 95 % |
|---|---|---|---|---|---|---|
| cfc_s42 | 0 de 32 | 0.0 a 10.9 % | 5 de 30 (16.7 %) | 5.6 a 34.7 % | 4 de 29 (13.8 %) | 3.9 a 31.7 % |
| cfc_s43 | 0 de 32 | 0.0 a 10.9 % | 6 de 30 (20.0 %) | 7.7 a 38.6 % | 4 de 29 (13.8 %) | 3.9 a 31.7 % |
| cfc_s44 | 0 de 32 | 0.0 a 10.9 % | 10 de 30 (33.3 %) | 17.3 a 52.8 % | 8 de 29 (27.6 %) | 12.7 a 47.2 % |
| gru_s42 | 0 de 32 | 0.0 a 10.9 % | 8 de 30 (26.7 %) | 12.3 a 45.9 % | 6 de 29 (20.7 %) | 8.0 a 39.7 % |
| gru_s43 | 0 de 32 | 0.0 a 10.9 % | 6 de 30 (20.0 %) | 7.7 a 38.6 % | 5 de 29 (17.2 %) | 5.8 a 35.8 % |
| gru_s44 | 0 de 32 | 0.0 a 10.9 % | 6 de 30 (20.0 %) | 7.7 a 38.6 % | 7 de 29 (24.1 %) | 10.3 a 43.5 % |
| lstm_s42 | 0 de 32 | 0.0 a 10.9 % | 5 de 30 (16.7 %) | 5.6 a 34.7 % | 5 de 29 (17.2 %) | 5.8 a 35.8 % |
| lstm_s43 | 0 de 32 | 0.0 a 10.9 % | 10 de 30 (33.3 %) | 17.3 a 52.8 % | 8 de 29 (27.6 %) | 12.7 a 47.2 % |
| lstm_s44 | 0 de 32 | 0.0 a 10.9 % | 6 de 30 (20.0 %) | 7.7 a 38.6 % | 7 de 29 (24.1 %) | 10.3 a 43.5 % |
| gradient boosting | 1 de 32 (3.1 %) | 0.1 a 16.2 % | 8 de 30 (26.7 %) | 12.3 a 45.9 % | 6 de 29 (20.7 %) | 8.0 a 39.7 % |
| regresion logistica | 0 de 32 | 0.0 a 10.9 % | 3 de 30 (10.0 %) | 2.1 a 26.5 % | 2 de 29 (6.9 %) | 0.8 a 22.8 % |

- Sin fuego: con 0 de 32 la tasa real puede llegar al 10.9 %.
- Controlados: el 10 % nominal queda dentro del intervalo en 6 de las 9 corridas de los modelos temporales contra la IA (todas menos `cfc_s44`, `gru_s42` y `lstm_s43`) y en 5 de 9 contra la humana. Con unos 30 videos solo se puede afirmar que la tasa supera el 10 % en esas corridas, no en general.

### d. Bootstrap de videos del mAP promedio con tIoU 0.5

| referencia | videos | modelo | mAP | IC 95 % |
|---|---|---|---|---|
| IA | 70 | GRU | 0.299 +- 0.023 | 0.219 a 0.400 |
| IA | 70 | CfC | 0.216 +- 0.039 | 0.151 a 0.303 |
| IA | 70 | gradient boosting | 0.204 | 0.140 a 0.305 |
| humana | 29 | GRU | 0.136 +- 0.015 | 0.071 a 0.290 |
| humana | 29 | CfC | 0.103 +- 0.048 | 0.066 a 0.199 |
| humana | 29 | gradient boosting | 0.125 | 0.066 a 0.257 |

Diferencia pareada GRU - CfC: +0.083 (IC 95 % de +0.002 a +0.162; positiva en el 97.9 % de los remuestreos) contra la IA; +0.033 (de -0.076 a +0.186; 68.9 %) contra la humana. Contra la IA es la unica diferencia entre modelos cuyo intervalo no incluye el cero, y por muy poco.

## Plan del experimento de robustez (registrado el 2026-11-13, antes de correr nada)

La configuracion sigue congelada (`resultados/configuracion_final.json`). Todo lo que sigue se decide aqui, antes de ver resultados.

### Parte 1: modelos tal como estan

1. **Fotogramas quitados en la entrada.** Se corre de nuevo `descriptores_v3.py` solo sobre los 102 videos de prueba (`resultados/robustez/lista_prueba.csv`), con `--quitar 0.25`, `0.50` y `0.75` y `--semilla 42` (cada fotograma muestreado, salvo el primero, se quita con esa probabilidad; la semilla efectiva es 42 + crc32 del nombre). Todo lo demas igual que en la corrida completa. Salidas: `runs_fuego/robustez/quitar_25`, `quitar_50` y `quitar_75`. La corrida la lanza el usuario.
2. **Evaluacion sin reentrenar ni recalibrar.** Con cada salida se arma el vector de escena con el mismo codigo (`escena_v2.vectores_carpeta` y `VectorEscena`, detecciones a confianza 0.25, edad minima de 1 s) y la normalizacion congelada (`resultados/escena_v2/normalizacion*.json`). Se descartan los tramos de menos de 10 fotogramas, como siempre. Se evaluan los 9 modelos temporales congelados, el gradient boosting y la regresion logistica, con los mismos pesos, umbrales de CORAL en 0.5, histeresis de 2 s y los umbrales conformales congelados (alfa 0.10). Metricas, contra las dos referencias: F1 macro y kappa del nivel por segundo, falsas alarmas por video (sin fuego y controlados) y adelanto de la alerta (EWT mediana y videos con alerta antes del cambio). Se reporta el valor y la caida respecto a la prueba completa.
   - Control: el mismo script aplicado a la corrida completa (`runs_fuego/descriptores_v3`, sin quitar) debe reproducir los numeros de la fase F.
   - Efecto conocido de antemano: la normalizacion congelada recorta `dt` del contexto al percentil 99 de entrenamiento, asi que con fotogramas quitados esa entrada se satura; el `dt` real si llega al modelo como intervalo (CfC) o como entrada (GRU y LSTM).
3. **Camara movil contra camara fija.** Con las predicciones de la fase F (sin correr nada nuevo). Movimiento por fotograma = norma de la traslacion (en fraccion del fotograma) + 0.5 (|s - 1| + |rotacion|), con la similitud ya guardada en `runs_fuego/descriptores_v3/fotogramas.csv`; puntaje del video = media por fotograma x 10 (fraccion del fotograma por segundo). **Camara movil si el puntaje es >= 0.01 (1 % del fotograma por segundo); fija si es menor.** El umbral se eligio mirando solo la distribucion del puntaje en los 102 videos de prueba (mediana 0.018, cuartiles 0.005 y 0.054), sin ver resultados de modelos por grupo. Se comparan los modelos en cada grupo con las mismas metricas.

### Parte 2: reentrenamiento con fotogramas quitados (decidida desde ahora, igual para los tres modelos)

- Se reentrenan CfC, GRU y LSTM, cada uno con su configuracion congelada (CfC y LSTM: peso simple, tasa 1e-3; GRU: doble peso al incipiente, tasa 3e-3; riesgo en escalon; 60 epocas; semillas 42, 43 y 44; CfC en CPU). Lo unico que cambia es el aumento de datos.
- **Aumento:** en cada ventana de entrenamiento se sortea una fraccion q uniforme entre 0 y 0.75 y cada fotograma de la ventana, salvo el primero, se quita con probabilidad q (el mismo mecanismo que `LectorVideo`). El `dt` de cada fotograma que queda es el tiempo transcurrido desde el anterior que quedo; la columna `dt` del contexto se recalcula con la normalizacion congelada. Las etiquetas (nivel, riesgo, tau) son las de los fotogramas que quedan. Es una aproximacion: los descriptores se calcularon con todos los fotogramas.
- La epoca se elige igual que antes: F1 macro de escena suavizado en la validacion completa (sin quitar fotogramas). La normalizacion no cambia.
- **Fase E:** se vuelve a calibrar el umbral conformal de cada modelo nuevo con los 52 videos de calibracion sin fuego o controlados, completos (alfa 0.10 y 0.05); umbrales de CORAL en 0.5, histeresis de 2 s y factor de desactivacion 0.8 sin cambios.
- **Evaluacion:** en la prueba completa y en las tres fracciones, con las mismas metricas de la parte 1, comparando con los modelos congelados. El gradient boosting y la regresion logistica no se reentrenan.
- Los modelos congelados siguen siendo el resultado principal; los de la parte 2 se guardan aparte (`runs_fuego/modelos_robustos/`, `resultados/robustez/`) y se reportan como experimento de robustez.
- La parte 2 se corre despues de la parte 1.

### Adenda al plan de robustez (2026-11-13, antes de cualquier resultado de las partes 1.1, 1.2 y 2)

Se agrega antes de que exista ningun resultado con fotogramas quitados (la corrida de `descriptores_v3.py` sobre las fracciones aun no se lanza):

- **Medida secundaria.** Con 75 % quitado, un segundo queda sin ningun fotograma con probabilidad 0.75^10 = 5.6 % (0.1 % con 50 %). La regla de la fase F lo cuenta como controlado. Ademas de la medida principal (la de la fase F), se reportan F1 macro y kappa con el segundo sin fotogramas tomando el ultimo nivel decidido, que es lo que muestra el sistema en vivo, y el numero de segundos sin fotogramas.
- **Descripcion de cada fraccion:** fotogramas procesados, cortes de escena detectados, videos con corte, tramos cortos descartados, y dt mediano y percentil 95.
- **Implementacion de la parte 2** (`entrenar_robusto.py`, `robustez_parte2.py`):
  - el aumento usa un generador aleatorio propio (semilla `[s, 1]`), asi que el sorteo de ventanas es el mismo que en los modelos congelados;
  - la columna dt del contexto solo se recalcula en los fotogramas cuyo anterior se quito; con 0 % quitado el lote es identico al original (verificado);
  - el dt recalculado coincide con la diferencia de marcas de tiempo (error maximo de 6e-6 s);
  - la recalibracion usa el mismo codigo que la fase E congelada (`fase_e_elegida.py`);
  - los hiperparametros de cada modelo se leen de `configuracion_final.json`.
- **Control de la parte 1** (`evaluar_robustez.py`): aplicado a la corrida completa (`runs_fuego/descriptores_v3`), reproduce la fase F con diferencia 0 en F1, kappa, falsas alarmas y adelanto, con las dos referencias. Si el control no pasa, el script no evalua las fracciones.

### Resultado de la parte 1.3: camara movil contra camara fija (2026-11-13)

`python analisis_prueba.py --partes camara`. Usa las predicciones de la fase F (configuracion congelada) y el criterio registrado en el plan (camara movil si el puntaje es >= 0.01). Salida en `resultados/prueba/camara.md` y la clasificacion de cada video en `resultados/prueba/camara_videos.csv`.

- 59 videos con camara movil y 43 con camara fija. De fuego: 39 y 31; sin fuego: 20 y 12.
- Segundos por nivel segun la IA: fija 888 / 718 / 775, movil 576 / 778 / 968 (controlado / incipiente / incontrolado).

F1 macro (kappa lineal) del nivel por segundo:

| modelo | IA, camara fija (31) | IA, camara movil (39) | humana, camara fija (30) | humana, camara movil (39) |
|---|---|---|---|---|
| CfC | 0.470 +- 0.023 (0.317) | 0.438 +- 0.027 (0.294) | 0.534 +- 0.028 (0.416) | 0.403 +- 0.022 (0.244) |
| GRU | 0.556 +- 0.026 (0.404) | 0.476 +- 0.049 (0.298) | 0.548 +- 0.023 (0.394) | 0.458 +- 0.038 (0.262) |
| LSTM | 0.444 +- 0.050 (0.274) | 0.472 +- 0.010 (0.342) | 0.473 +- 0.029 (0.348) | 0.429 +- 0.014 (0.294) |
| gradient boosting | 0.402 (0.190) | 0.472 (0.355) | 0.499 (0.316) | 0.404 (0.283) |
| regresion logistica | 0.435 (0.233) | 0.556 (0.376) | 0.551 (0.370) | 0.512 (0.353) |

Alerta con alfa 0.10:

| camara | modelo | falsas alarmas sin fuego | falsas alarmas en controlados (IA) | EWT mediana, IA (s) | antes del cambio, IA | EWT mediana, humana (s) | antes del cambio, humana |
|---|---|---|---|---|---|---|---|
| fija | CfC | 0 de 12 | 0.167 (de 18) | -42.5 | 0.7 de 7 | +18.1 | 3.7 de 6 |
| fija | GRU | 0 de 12 | 0.148 (de 18) | -23.0 | 0.7 de 7 | +3.7 | 3.7 de 6 |
| fija | LSTM | 0 de 12 | 0.167 (de 18) | -35.4 | 1.0 de 7 | +29.8 | 3.7 de 6 |
| fija | gradient boosting | 0 de 12 | 0.167 (de 18) | -18.8 | 2 de 7 | +13.1 | 4 de 6 |
| movil | CfC | 0 de 20 | 0.333 (de 12) | +10.3 | 7.7 de 14 | +36.5 | 8.7 de 10 |
| movil | GRU | 0 de 20 | 0.333 (de 12) | +9.2 | 7.7 de 14 | +38.6 | 7.7 de 10 |
| movil | LSTM | 0 de 20 | 0.333 (de 12) | +9.7 | 8.0 de 14 | +29.5 | 7.0 de 10 |
| movil | gradient boosting | 0.05 (1 de 20) | 0.417 (de 12) | +11.2 | 10 de 14 | +60.0 | 9 de 10 |

Lectura:
- La GRU y el CfC empeoran con camara movil con las dos referencias (GRU de 0.556 a 0.476 contra la IA y de 0.548 a 0.458 contra la humana). Contra la IA, la LSTM, el gradient boosting y la regresion logistica mejoran con camara movil; contra la humana empeoran todos.
- Las falsas alarmas en controlados se duplican con camara movil (33 % de 12 contra 15 a 17 % de 18), aunque con muy pocos videos.
- La alerta llega antes del cambio en los videos con camara movil y despues en los de camara fija. Los grupos no difieren solo en la camara. Segun la IA, los de camara movil tienen mas incontrolado (42 % de los segundos contra 33 %) y menos controlado (25 % contra 37 %; 12 videos controlados contra 18). Por eso la comparacion es descriptiva y no aisla el efecto de la camara.
