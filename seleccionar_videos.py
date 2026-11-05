import sys
import csv
import re
import random
from collections import defaultdict

ruta_csv = sys.argv[1] if len(sys.argv) > 1 else "metadatos_videos.csv"
objetivo = int(sys.argv[2]) if len(sys.argv) > 2 else 400
duracion_min = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
max_por_origen = int(sys.argv[4]) if len(sys.argv) > 4 else 6
random.seed(42)


def numero_clip(nombre):
    m = re.search(r"_clip_(\d+)", nombre)
    if m:
        return int(m.group(1))
    m = re.search(r"-(\d+)\.\w+$", nombre)
    return int(m.group(1)) if m else 0


with open(ruta_csv, encoding="utf-8") as f:
    filas = list(csv.DictReader(f))

fuego = []
for r in filas:
    if r["clase"] != "fire":
        continue
    try:
        r["duracion_s"] = float(r["duracion_s"])
        r["fps"] = float(r["fps"])
    except ValueError:
        continue
    if r["duracion_s"] <= 0:
        continue
    r["orden_clip"] = numero_clip(r["archivo"])
    fuego.append(r)

print("=== videos con fuego ===")
print(f"total legibles: {len(fuego)}")
rangos = [(0, 5), (5, 10), (10, 20), (20, 60), (60, 1e9)]
for a, b in rangos:
    n = sum(1 for r in fuego if a <= r["duracion_s"] < b)
    etiqueta = f"{a:.0f}-{b:.0f} s" if b < 1e9 else f">= {a:.0f} s"
    print(f"duracion {etiqueta}: {n}")

por_origen = defaultdict(list)
for r in fuego:
    por_origen[r["origen"]].append(r)
for o in por_origen:
    por_origen[o].sort(key=lambda r: r["orden_clip"])

print(f"\norigenes distintos: {len(por_origen)}")
con_varios = sum(1 for v in por_origen.values() if len(v) > 1)
print(f"origenes con mas de un clip: {con_varios}")
print(f"origenes con un solo clip: {len(por_origen) - con_varios}")

candidatos = {o: [r for r in v if r["duracion_s"] >= duracion_min] for o, v in por_origen.items()}
candidatos = {o: v for o, v in candidatos.items() if v}
print(f"\norigenes con al menos un clip de {duracion_min:.0f} s o mas: {len(candidatos)}")

seleccion = []
orden_origenes = sorted(candidatos, key=lambda o: sum(r["duracion_s"] for r in candidatos[o]), reverse=True)
ronda = 0
while len(seleccion) < objetivo and ronda < max_por_origen:
    agregados = 0
    for o in orden_origenes:
        if len(seleccion) >= objetivo:
            break
        if ronda < len(candidatos[o]):
            seleccion.append(candidatos[o][ronda])
            agregados += 1
    if agregados == 0:
        break
    ronda += 1

seleccion.sort(key=lambda r: (r["origen"], r["orden_clip"]))
origenes_sel = {r["origen"] for r in seleccion}
duracion_total = sum(r["duracion_s"] for r in seleccion)

print("\n=== seleccion para etiquetar ===")
print(f"videos seleccionados: {len(seleccion)}")
print(f"origenes cubiertos: {len(origenes_sel)}")
print(f"duracion total: {duracion_total / 3600:.2f} horas")
print(f"tiempo estimado de etiquetado (20 s por video): {len(seleccion) * 20 / 3600:.1f} horas")

ids = sorted(origenes_sel)
random.shuffle(ids)
n_doble = max(1, round(len(ids) * 0.2))
doble = set(ids[:n_doble])
n_videos_doble = sum(1 for r in seleccion if r["origen"] in doble)
print(f"videos para el segundo anotador (20% de origenes): {n_videos_doble}")

salida = "seleccion_etiquetado.csv"
with open(salida, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["archivo", "origen", "orden_clip", "duracion_s", "fps", "segundo_anotador"])
    for r in seleccion:
        w.writerow([r["archivo"], r["origen"], r["orden_clip"], r["duracion_s"], r["fps"], int(r["origen"] in doble)])
print(f"\nseleccion guardada en: {salida}")