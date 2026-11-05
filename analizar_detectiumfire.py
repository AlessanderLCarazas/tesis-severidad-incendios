import os
import sys
import csv
import re
import statistics
from collections import Counter, defaultdict

ruta = sys.argv[1] if len(sys.argv) > 1 else "."

print("=== 1. clases en etiquetas yolo de real_fire ===")
conteo_clases = Counter()
imagenes_por_split = Counter()
for raiz, _, archivos in os.walk(ruta):
    if "real_fire" not in raiz or os.path.basename(raiz) != "labels":
        continue
    split = os.path.basename(os.path.dirname(raiz))
    for a in archivos:
        if not a.endswith(".txt"):
            continue
        imagenes_por_split[split] += 1
        with open(os.path.join(raiz, a), encoding="utf-8", errors="replace") as f:
            for linea in f:
                partes = linea.split()
                if partes:
                    conteo_clases[partes[0]] += 1
for split, n in sorted(imagenes_por_split.items()):
    print(f"split {split}: {n} imagenes etiquetadas")
for clase, n in sorted(conteo_clases.items()):
    print(f"clase {clase}: {n} cajas")

print("\n=== 2. busqueda de archivos con severidad o descripciones ===")
palabras = re.compile(r"severity|risk|caption|prompt|description", re.IGNORECASE)
encontrados = 0
for raiz, _, archivos in os.walk(ruta):
    if "labels" in raiz or "preference_dataset" in raiz:
        continue
    for a in archivos:
        ext = os.path.splitext(a)[1].lower()
        if ext not in {".txt", ".json", ".csv", ".jsonl", ".yaml", ".yml", ".xlsx", ".tsv"}:
            continue
        p = os.path.join(raiz, a)
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                texto = f.read(200000)
        except Exception:
            continue
        if palabras.search(texto):
            encontrados += 1
            print(f"posible archivo de descripciones: {p}")
            print(texto[:500])
            print("...")
            if encontrados >= 5:
                break
    if encontrados >= 5:
        break
if encontrados == 0:
    print("no se encontraron archivos con severidad o descripciones fuera de las etiquetas yolo")

print("\n=== 3. estadisticas de videos ===")
try:
    import cv2
except ImportError:
    print("opencv no instalado, ejecuta: pip install opencv-python")
    sys.exit()

filas = []
for raiz, _, archivos in os.walk(ruta):
    nombre_carpeta = os.path.basename(raiz)
    if nombre_carpeta not in {"fire", "non_fire"} or "real_video" not in raiz:
        continue
    for a in sorted(archivos):
        if os.path.splitext(a)[1].lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            continue
        p = os.path.join(raiz, a)
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duracion = frames / fps if fps > 0 else 0
        base = os.path.splitext(a)[0]
        if "_clip_" in base:
            origen = base.split("_clip_")[0]
        elif "-" in base:
            origen = base.split("-")[0]
        else:
            origen = base
        filas.append({
            "archivo": a,
            "clase": nombre_carpeta,
            "origen": origen,
            "fps": round(fps, 2),
            "frames": int(frames),
            "duracion_s": round(duracion, 2),
            "ancho": ancho,
            "alto": alto,
        })

for clase in ["fire", "non_fire"]:
    sub = [f for f in filas if f["clase"] == clase]
    if not sub:
        continue
    duraciones = [f["duracion_s"] for f in sub if f["duracion_s"] > 0]
    origenes = Counter(f["origen"] for f in sub)
    print(f"\nclase: {clase}")
    print(f"videos: {len(sub)}")
    if duraciones:
        print(f"duracion min: {min(duraciones):.1f} s, mediana: {statistics.median(duraciones):.1f} s, max: {max(duraciones):.1f} s")
        print(f"duracion total: {sum(duraciones) / 3600:.2f} horas")
    print(f"fps mas comunes: {Counter(f['fps'] for f in sub).most_common(5)}")
    print(f"resoluciones mas comunes: {Counter((f['ancho'], f['alto']) for f in sub).most_common(5)}")
    print(f"videos de origen distintos (segun prefijo): {len(origenes)}")
    print(f"origenes con mas clips: {origenes.most_common(5)}")
    errores = [f["archivo"] for f in sub if f["duracion_s"] == 0]
    if errores:
        print(f"videos que no se pudieron leer: {len(errores)}, ejemplo: {errores[:3]}")

salida = os.path.join(ruta, "metadatos_videos.csv")
with open(salida, "w", newline="", encoding="utf-8") as f:
    escritor = csv.DictWriter(f, fieldnames=list(filas[0].keys()) if filas else ["archivo"])
    escritor.writeheader()
    escritor.writerows(filas)
print(f"\nmetadatos guardados en: {salida}")