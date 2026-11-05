import os
import sys
import json
from collections import Counter

ruta = sys.argv[1] if len(sys.argv) > 1 else "."
ext_video = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv"}
ext_imagen = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ext_etiqueta = {".csv", ".json", ".txt", ".xml", ".yaml", ".yml", ".jsonl"}

print("=== estructura de carpetas (hasta 3 niveles) ===")
for raiz, dirs, archivos in os.walk(ruta):
    nivel = raiz[len(ruta):].count(os.sep)
    if nivel > 3:
        dirs[:] = []
        continue
    dirs.sort()
    sangria = "  " * nivel
    print(f"{sangria}{os.path.basename(raiz) or raiz}/  ({len(archivos)} archivos)")
    for a in sorted(archivos)[:3]:
        print(f"{sangria}  - {a}")
    if len(archivos) > 3:
        print(f"{sangria}  ...")

conteo = Counter()
etiquetas = []
videos = []
for raiz, _, archivos in os.walk(ruta):
    for a in archivos:
        ext = os.path.splitext(a)[1].lower()
        conteo[ext] += 1
        completo = os.path.join(raiz, a)
        if ext in ext_etiqueta:
            etiquetas.append(completo)
        if ext in ext_video:
            videos.append(completo)

print("\n=== cantidad de archivos por extension ===")
for ext, n in conteo.most_common():
    print(f"{ext or '(sin extension)'}: {n}")

total_videos = sum(n for e, n in conteo.items() if e in ext_video)
total_imagenes = sum(n for e, n in conteo.items() if e in ext_imagen)
print(f"\ntotal videos: {total_videos}")
print(f"total imagenes: {total_imagenes}")
print(f"total archivos de etiquetas: {len(etiquetas)}")

print("\n=== ejemplos de archivos de etiquetas ===")
vistos = set()
mostrados = 0
for e in sorted(etiquetas, key=lambda p: os.path.getsize(p), reverse=True):
    ext = os.path.splitext(e)[1].lower()
    if ext in vistos and mostrados >= 3:
        continue
    vistos.add(ext)
    mostrados += 1
    print(f"\n--- {e} ({os.path.getsize(e)} bytes) ---")
    try:
        if ext == ".json":
            with open(e, encoding="utf-8") as f:
                datos = json.load(f)
            if isinstance(datos, list):
                print(f"lista con {len(datos)} elementos, primer elemento:")
                print(json.dumps(datos[0], ensure_ascii=True, indent=2)[:1200] if datos else "vacia")
            elif isinstance(datos, dict):
                print(f"claves: {list(datos.keys())[:20]}")
                print(json.dumps(datos, ensure_ascii=True, indent=2)[:1200])
        else:
            with open(e, encoding="utf-8", errors="replace") as f:
                for i, linea in enumerate(f):
                    if i >= 6:
                        break
                    print(linea.rstrip()[:300])
    except Exception as ex:
        print(f"no se pudo leer: {ex}")
    if mostrados >= 6:
        break

print("\n=== informacion de un video de ejemplo ===")
if videos:
    try:
        import cv2
        v = videos[0]
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ancho = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        alto = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        print(f"video: {v}")
        print(f"fps: {fps:.2f}, frames: {int(frames)}, resolucion: {int(ancho)}x{int(alto)}")
        if fps > 0:
            print(f"duracion aprox: {frames / fps:.1f} s")
    except ImportError:
        print("opencv no instalado, ejecuta: pip install opencv-python")
else:
    print("no se encontraron videos")