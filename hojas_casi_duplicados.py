import os
import re
import csv
import argparse
import unicodedata
import cv2
import numpy as np

COLUMNAS, FILAS = 6, 4


def sin_tildes(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def fotos_video(nombre, carpetas):
    base = os.path.splitext(nombre)[0]
    fotos = []
    for carpeta in carpetas:
        for sufijo in ("", "_a", "_b"):
            ruta = os.path.join(carpeta, base + sufijo + ".jpg")
            if not os.path.exists(ruta):
                continue
            img = cv2.imdecode(np.fromfile(ruta, np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            for i in range(FILAS):
                for j in range(COLUMNAS):
                    c = img[i * h // FILAS:(i + 1) * h // FILAS, j * w // COLUMNAS:(j + 1) * w // COLUMNAS]
                    if c.std() > 3:
                        fotos.append(c)
        if fotos:
            break
    return fotos


def fila(nombre, duracion, fotos, n, ancho, alto):
    idx = np.linspace(0, len(fotos) - 1, n).round().astype(int) if fotos else []
    celdas = [cv2.resize(fotos[i], (ancho, alto), interpolation=cv2.INTER_AREA) for i in idx]
    while len(celdas) < n:
        celdas.append(np.zeros((alto, ancho, 3), np.uint8))
    barra = np.full((30, n * ancho, 3), 40, np.uint8)
    cv2.putText(barra, f"{sin_tildes(nombre)[:110]} | {duracion} s", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([barra, np.hstack(celdas)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pares", default=os.path.join("resultados", "particion", "casi_duplicados.csv"))
    p.add_argument("--tiras", nargs="+", default=["tiras", "tiras_v2"])
    p.add_argument("--fotos", type=int, default=6)
    p.add_argument("--ancho", type=int, default=320)
    p.add_argument("--alto", type=int, default=180)
    p.add_argument("--salida", default=os.path.join("runs_fuego", "casi_duplicados"))
    a = p.parse_args()

    with open(a.pares, encoding="utf-8") as f:
        pares = list(csv.DictReader(f))
    os.makedirs(a.salida, exist_ok=True)
    indice = []
    for n, r in enumerate(pares, 1):
        fa, fb = fotos_video(r["archivo_a"], a.tiras), fotos_video(r["archivo_b"], a.tiras)
        cab = np.full((30, a.fotos * a.ancho, 3), 20, np.uint8)
        texto = (f"par {n} de {len(pares)} | fotos parecidas: {r['fotos_iguales_a']} de {r['fotos_a']} y"
                 f" {r['fotos_iguales_b']} de {r['fotos_b']} | distancia minima {r['distancia_min']}")
        cv2.putText(cab, texto, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        img = np.vstack([cab, fila(r["archivo_a"], r["duracion_a_s"], fa, a.fotos, a.ancho, a.alto),
                         fila(r["archivo_b"], r["duracion_b_s"], fb, a.fotos, a.ancho, a.alto)])
        base = re.sub(r"[^A-Za-z0-9]+", "_", sin_tildes(os.path.splitext(r["archivo_a"])[0]))[:40].strip("_")
        nombre = f"{n:02d}_{base}.jpg"
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(os.path.join(a.salida, nombre))
        indice.append({"n": n, "hoja": nombre, "archivo_a": r["archivo_a"], "archivo_b": r["archivo_b"]})
    with open(os.path.join(a.salida, "indice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "hoja", "archivo_a", "archivo_b"])
        w.writeheader()
        w.writerows(indice)
    print(f"hojas: {len(indice)} en {a.salida}")


if __name__ == "__main__":
    main()
