import os
import re
import csv
import hashlib
import argparse
from collections import defaultdict

CAMPOS = ["archivo", "conservado", "motivo", "sha256", "tamano", "en_lista"]


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def es_copia(nombre):
    return bool(re.search(r" - copy|\(\d+\)\.\w+$|copia", nombre.lower())) and not nombre.lower().startswith("video (")


def clave_natural(nombre):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", nombre.lower())]


def cargar_excluidos(ruta=os.path.join("resultados", "excluidos.csv")):
    if not os.path.exists(ruta):
        return set()
    with open(ruta, encoding="utf-8") as f:
        return {r["archivo"] for r in csv.DictReader(f)}


def leer_columna(ruta, columna="archivo"):
    if not ruta or not os.path.exists(ruta):
        return set()
    with open(ruta, encoding="utf-8") as f:
        return {r[columna] for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="+", default=[os.path.join("real_video", "real_video", "fire"),
                                                   os.path.join("real_video", "real_video", "non_fire")])
    p.add_argument("--prueba", default=os.path.join("resultados", "prueba_humana.csv"))
    p.add_argument("--lista", default=os.path.join("resultados", "cortes", "lista_noche.csv"))
    p.add_argument("--salida", default=os.path.join("resultados", "excluidos.csv"))
    a = p.parse_args()

    rutas = {}
    for carpeta in a.videos:
        for raiz, _, archivos in os.walk(carpeta):
            for nombre in archivos:
                rutas.setdefault(nombre, os.path.join(raiz, nombre))
    por_tamano = defaultdict(list)
    for nombre, ruta in rutas.items():
        por_tamano[os.path.getsize(ruta)].append(nombre)
    grupos = defaultdict(list)
    for tamano, nombres in por_tamano.items():
        if len(nombres) > 1:
            for nombre in nombres:
                grupos[(sha256(rutas[nombre]), tamano)].append(nombre)
    prueba = leer_columna(a.prueba)
    lista = leer_columna(a.lista)
    filas = []
    for (h, tamano), nombres in sorted(grupos.items(), key=lambda x: clave_natural(min(x[1], key=clave_natural))):
        if len(nombres) < 2:
            continue
        orden = sorted(nombres, key=lambda n: (n not in prueba, es_copia(n), clave_natural(n)))
        conservado = orden[0]
        for nombre in orden[1:]:
            motivo = "duplicado exacto (sha256)"
            if conservado in prueba:
                motivo += "; el conservado esta en la prueba humana"
            filas.append({"archivo": nombre, "conservado": conservado, "motivo": motivo, "sha256": h[:16], "tamano": tamano,
                          "en_lista": "si" if nombre in lista else "no"})
        print(f"grupo de {len(nombres)} | se conserva {conservado[:45]} | se excluyen {len(nombres) - 1}")
    os.makedirs(os.path.dirname(a.salida) or ".", exist_ok=True)
    with open(a.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)
    print(f"videos revisados {len(rutas)} | grupos de duplicados {len(set(r['conservado'] for r in filas))}"
          f" | excluidos {len(filas)} | excluidos que estan en {os.path.basename(a.lista)}: {sum(r['en_lista'] == 'si' for r in filas)}")
    print(f"lista de excluidos en: {a.salida}")


if __name__ == "__main__":
    main()
