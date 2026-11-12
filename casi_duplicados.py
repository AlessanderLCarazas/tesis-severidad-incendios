import os
import csv
import argparse
import cv2
import numpy as np

COLUMNAS, FILAS = 6, 4
CAMPOS = ["archivo_a", "archivo_b", "fotos_iguales_a", "fotos_iguales_b", "fotos_a", "fotos_b", "distancia_min",
          "duracion_a_s", "duracion_b_s"]


def huellas_tira(ruta, desv_min):
    img = cv2.imdecode(np.fromfile(ruta, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    h, w = img.shape
    salida = []
    for i in range(FILAS):
        for j in range(COLUMNAS):
            c = img[i * h // FILAS:(i + 1) * h // FILAS, j * w // COLUMNAS:(j + 1) * w // COLUMNAS]
            c = c[int(c.shape[0] * 0.15):int(c.shape[0] * 0.8), int(c.shape[1] * 0.1):int(c.shape[1] * 0.9)]
            if c.std() < desv_min:
                continue
            s = cv2.resize(c, (9, 8), interpolation=cv2.INTER_AREA)
            salida.append((s[:, 1:] > s[:, :-1]).flatten())
    return salida


def huellas_video(nombre, carpetas, desv_min):
    base = os.path.splitext(nombre)[0]
    salida = []
    for carpeta in carpetas:
        for sufijo in ("", "_a", "_b"):
            ruta = os.path.join(carpeta, base + sufijo + ".jpg")
            if os.path.exists(ruta):
                salida += huellas_tira(ruta, desv_min)
    return np.array(salida, bool).reshape(-1, 64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lista", default=os.path.join("resultados", "cortes", "lista_noche.csv"))
    p.add_argument("--clase", default="fire")
    p.add_argument("--tiras", nargs="+", default=["tiras", "tiras_v2"])
    p.add_argument("--metadatos", default="metadatos_videos.csv")
    p.add_argument("--desv_min", type=float, default=20.0)
    p.add_argument("--distancia", type=int, default=4)
    p.add_argument("--min_iguales", type=int, default=4)
    p.add_argument("--salida", default=os.path.join("resultados", "particion", "casi_duplicados.csv"))
    a = p.parse_args()

    with open(a.lista, encoding="utf-8") as f:
        lista = sorted(r["archivo"] for r in csv.DictReader(f) if r["clase"] == a.clase)
    with open(a.metadatos, encoding="utf-8") as f:
        duracion = {r["archivo"]: r["duracion_s"] for r in csv.DictReader(f)}
    datos = {v: huellas_video(v, a.tiras, a.desv_min) for v in lista}
    sin_tira = [v for v in lista if len(datos[v]) == 0]
    nombres = [v for v in lista if len(datos[v])]
    pares = []
    for i, va in enumerate(nombres):
        ha = datos[va]
        for vb in nombres[i + 1:]:
            hb = datos[vb]
            d = (ha[:, None, :] != hb[None, :, :]).sum(2)
            na, nb = int((d.min(1) <= a.distancia).sum()), int((d.min(0) <= a.distancia).sum())
            if min(na, nb) >= a.min_iguales:
                pares.append({"archivo_a": va, "archivo_b": vb, "fotos_iguales_a": na, "fotos_iguales_b": nb,
                              "fotos_a": len(ha), "fotos_b": len(hb), "distancia_min": int(d.min()),
                              "duracion_a_s": duracion.get(va, ""), "duracion_b_s": duracion.get(vb, "")})
    pares.sort(key=lambda r: (-min(r["fotos_iguales_a"] / r["fotos_a"], r["fotos_iguales_b"] / r["fotos_b"]), r["archivo_a"]))
    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    with open(a.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(pares)
    print(f"videos con tira: {len(nombres)} de {len(lista)} | sin tira o sin fotos utiles: {len(sin_tira)}"
          f" | pares candidatos: {len(pares)} | salida: {a.salida}")


if __name__ == "__main__":
    main()
