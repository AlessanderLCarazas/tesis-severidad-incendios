import os
import re
from collections import Counter

base = "FASDD"
dir_yolo = os.path.join(base, "annotations", "YOLO")
patron = re.compile(r"_(CV|UAV|RS)\d+", re.IGNORECASE)
imagenes = set(os.listdir(os.path.join(base, "images")))
listas = {}
for split in ["train", "val", "test"]:
    with open(os.path.join(dir_yolo, f"{split}.txt"), encoding="utf-8") as f:
        listas[split] = [os.path.basename(l.strip()) for l in f if l.strip()]

for split, nombres in listas.items():
    cv = [n for n in nombres if (m := patron.search(n)) and m.group(1).upper() == "CV"]
    unicos = set(cv)
    faltan = [n for n in unicos if n not in imagenes]
    print(f"\n=== {split} ===")
    print(f"lineas cv: {len(cv)}, unicas: {len(unicos)}, repetidas: {len(cv) - len(unicos)}")
    print(f"no existen en images: {len(faltan)}")
    for n in faltan[:5]:
        print(f"  ejemplo faltante: {n}")
    if faltan:
        print(f"  prefijos de faltantes: {Counter(n.split('_')[0] for n in faltan).most_common(4)}")

print("\n=== cruce entre splits (fuga de datos) ===")
sets = {s: set(v) for s, v in listas.items()}
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    print(f"{a} y {b} comparten: {len(sets[a] & sets[b])} imagenes")