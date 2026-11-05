import os
import sys
import csv
import cv2
import numpy as np

carpeta_videos = sys.argv[1] if len(sys.argv) > 1 else os.path.join("real_video", "fire")
ruta_seleccion = sys.argv[2] if len(sys.argv) > 2 else "seleccion_etiquetado.csv"
carpeta_salida = sys.argv[3] if len(sys.argv) > 3 else "tiras"
n_fotos = int(sys.argv[4]) if len(sys.argv) > 4 else 24
columnas = 6
ancho_foto = 320

os.makedirs(carpeta_salida, exist_ok=True)

rutas = {}
for raiz, _, archivos in os.walk(carpeta_videos):
    for a in archivos:
        rutas[a] = os.path.join(raiz, a)

with open(ruta_seleccion, encoding="utf-8") as f:
    seleccion = list(csv.DictReader(f))

ruta_tiempos = os.path.join(carpeta_salida, "tiempos.csv")
ya_hechos = set()
if os.path.exists(ruta_tiempos):
    with open(ruta_tiempos, encoding="utf-8") as f:
        ya_hechos = {r["archivo"] for r in csv.DictReader(f)}
nuevo = not os.path.exists(ruta_tiempos)
archivo_tiempos = open(ruta_tiempos, "a", newline="", encoding="utf-8")
escritor = csv.writer(archivo_tiempos)
if nuevo:
    escritor.writerow(["archivo", "foto", "frame", "t_s"])


def poner_texto(img, texto, y):
    cv2.putText(img, texto, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, texto, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


faltantes = 0
errores = 0
for i, fila in enumerate(seleccion, 1):
    nombre = fila["archivo"]
    if nombre in ya_hechos:
        continue
    if nombre not in rutas:
        faltantes += 1
        print(f"no encontrado: {nombre}")
        continue
    cap = cv2.VideoCapture(rutas[nombre])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        errores += 1
        print(f"no se pudo leer: {nombre}")
        cap.release()
        continue
    objetivos = sorted(set(np.linspace(0, total - 1, min(n_fotos, total)).astype(int).tolist()))
    capturas = []
    idx = 0
    j = 0
    while j < len(objetivos):
        if not cap.grab():
            break
        if idx == objetivos[j]:
            ok, frame = cap.retrieve()
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = t_ms / 1000.0 if t_ms > 0 or idx == 0 else idx / fps
            if ok:
                capturas.append((idx, t, frame))
            j += 1
        idx += 1
    cap.release()
    if not capturas:
        errores += 1
        print(f"sin fotogramas: {nombre}")
        continue
    alto_foto = int(round(capturas[0][2].shape[0] * ancho_foto / capturas[0][2].shape[1]))
    fotos = []
    for k, (fr, t, frame) in enumerate(capturas, 1):
        img = cv2.resize(frame, (ancho_foto, alto_foto))
        poner_texto(img, f"{k}", 22)
        poner_texto(img, f"{t:.1f} s", alto_foto - 8)
        fotos.append(img)
        escritor.writerow([nombre, k, fr, round(t, 3)])
    while len(fotos) % columnas:
        fotos.append(np.zeros_like(fotos[0]))
    filas_img = [np.hstack(fotos[r:r + columnas]) for r in range(0, len(fotos), columnas)]
    tira = np.vstack(filas_img)
    base = os.path.splitext(nombre)[0]
    cv2.imwrite(os.path.join(carpeta_salida, base + ".jpg"), tira, [cv2.IMWRITE_JPEG_QUALITY, 85])
    archivo_tiempos.flush()
    if i % 20 == 0:
        print(f"procesados {i} de {len(seleccion)}")

archivo_tiempos.close()
print(f"listo. tiras en: {carpeta_salida}")
print(f"no encontrados: {faltantes}, con error: {errores}")