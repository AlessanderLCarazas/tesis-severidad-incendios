import os
import re
import sys
import shutil
import random
import argparse
from collections import Counter

p = argparse.ArgumentParser()
p.add_argument("--fasdd", default="FASDD")
p.add_argument("--salida", default="FASDD_yolo")
p.add_argument("--fuentes", default="CV")
p.add_argument("--negativos", default=os.path.join("real_images", "real_images", "real_non_fire"))
p.add_argument("--val_negativos", type=float, default=0.15)
a = p.parse_args()
random.seed(42)

fuentes = {f.strip().upper() for f in a.fuentes.split(",") if f.strip()}
patron_fuente = re.compile(r"_(CV|UAV|RS)\d+", re.IGNORECASE)
dir_yolo = os.path.join(a.fasdd, "annotations", "YOLO")
dir_labels = os.path.join(dir_yolo, "labels")
ext_img = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
modo_copia = {"enlace": 0, "copia": 0}


def enlazar(origen, destino):
    if os.path.exists(destino):
        return
    try:
        os.link(origen, destino)
        modo_copia["enlace"] += 1
    except OSError:
        shutil.copy2(origen, destino)
        modo_copia["copia"] += 1


def fuente_de(nombre):
    m = patron_fuente.search(nombre)
    return m.group(1).upper() if m else "OTRA"


print("=== deteccion de indices de clase ===")
conteo_fuego, conteo_humo = Counter(), Counter()
nombres_labels = os.listdir(dir_labels)
for nombre in nombres_labels:
    if nombre.startswith("fire_") and sum(conteo_fuego.values()) < 2000:
        destino = conteo_fuego
    elif nombre.startswith("smoke_") and sum(conteo_humo.values()) < 2000:
        destino = conteo_humo
    else:
        continue
    with open(os.path.join(dir_labels, nombre), encoding="utf-8") as f:
        for linea in f:
            partes = linea.split()
            if partes:
                destino[partes[0]] += 1
    if sum(conteo_fuego.values()) >= 2000 and sum(conteo_humo.values()) >= 2000:
        break
print(f"clases en imagenes fire_: {dict(conteo_fuego)}")
print(f"clases en imagenes smoke_: {dict(conteo_humo)}")
if not conteo_fuego or not conteo_humo:
    print("no se pudieron detectar las clases, revisa la carpeta de etiquetas")
    sys.exit(1)
id_fuego = int(conteo_fuego.most_common(1)[0][0])
id_humo = int(conteo_humo.most_common(1)[0][0])
if id_fuego == id_humo:
    print("fuego y humo tienen el mismo indice, revisa las etiquetas")
    sys.exit(1)
print(f"indice fuego: {id_fuego}, indice humo: {id_humo}")

resumen = {}
cajas = Counter()
sin_label = 0
for split in ["train", "val", "test"]:
    ruta_lista = os.path.join(dir_yolo, f"{split}.txt")
    with open(ruta_lista, encoding="utf-8") as f:
        rutas = [l.strip() for l in f if l.strip()]
    os.makedirs(os.path.join(a.salida, "images", split), exist_ok=True)
    os.makedirs(os.path.join(a.salida, "labels", split), exist_ok=True)
    por_fuente = Counter()
    usadas = 0
    for r in rutas:
        nombre = os.path.basename(r)
        fuente = fuente_de(nombre)
        por_fuente[fuente] += 1
        if fuente not in fuentes:
            continue
        origen_img = os.path.join(a.fasdd, r.replace("/", os.sep))
        if not os.path.exists(origen_img):
            origen_img = os.path.join(a.fasdd, "images", nombre)
        if not os.path.exists(origen_img):
            continue
        base = os.path.splitext(nombre)[0]
        enlazar(origen_img, os.path.join(a.salida, "images", split, nombre))
        origen_lbl = os.path.join(dir_labels, base + ".txt")
        if os.path.exists(origen_lbl):
            enlazar(origen_lbl, os.path.join(a.salida, "labels", split, base + ".txt"))
            if split == "train":
                with open(origen_lbl, encoding="utf-8") as f:
                    for linea in f:
                        partes = linea.split()
                        if partes:
                            cajas[int(partes[0])] += 1
        else:
            sin_label += 1
        usadas += 1
    resumen[split] = (usadas, dict(por_fuente))
    print(f"\nsplit {split}: {usadas} imagenes usadas")
    print(f"disponibles por fuente: {dict(por_fuente)}")

negativos = []
if os.path.isdir(a.negativos):
    for raiz, _, archivos in os.walk(a.negativos):
        for nombre in archivos:
            if os.path.splitext(nombre)[1].lower() in ext_img:
                negativos.append(os.path.join(raiz, nombre))
    random.shuffle(negativos)
    n_val = int(len(negativos) * a.val_negativos)
    for i, ruta in enumerate(negativos):
        split = "val" if i < n_val else "train"
        destino = os.path.join(a.salida, "images", split, "negdf_" + os.path.basename(ruta))
        enlazar(ruta, destino)
    print(f"\nnegativos de detectiumfire agregados: {len(negativos)} ({n_val} a val, {len(negativos) - n_val} a train)")
else:
    print(f"\nno se encontro la carpeta de negativos: {a.negativos}")

ruta_yaml = os.path.join(a.salida, "data.yaml")
with open(ruta_yaml, "w", encoding="utf-8") as f:
    f.write(f"path: {os.path.abspath(a.salida).replace(os.sep, '/')}\n")
    f.write("train: images/train\nval: images/val\ntest: images/test\n")
    f.write("names:\n")
    for i in sorted([id_fuego, id_humo]):
        f.write(f"  {i}: {'fuego' if i == id_fuego else 'humo'}\n")

print("\n=== resumen ===")
print(f"fuentes usadas: {sorted(fuentes)}")
print(f"cajas en train: fuego {cajas[id_fuego]}, humo {cajas[id_humo]}")
print(f"imagenes sin archivo de etiqueta (se toman como fondo): {sin_label}")
print(f"archivos enlazados: {modo_copia['enlace']}, copiados: {modo_copia['copia']}")
print(f"data.yaml creado en: {ruta_yaml}")