import os
import csv
import json
import argparse
import numpy as np

COLUMNAS = [f"d{i}" for i in range(1, 14)]


class Normalizador:
    def __init__(self, columnas=None, p_bajo=1.0, p_alto=99.0):
        self.columnas = list(columnas or COLUMNAS)
        self.p_bajo, self.p_alto = float(p_bajo), float(p_alto)
        self.bajo = self.alto = self.media = self.desv = None
        self.n_filas = 0
        self.videos = []

    def ajustar(self, x, videos=None):
        x = np.asarray(x, np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.columnas) or len(x) == 0:
            raise ValueError("la matriz no coincide con las columnas")
        if not np.isfinite(x).all():
            raise ValueError("hay valores no finitos")
        self.bajo = np.percentile(x, self.p_bajo, axis=0)
        self.alto = np.percentile(x, self.p_alto, axis=0)
        r = np.clip(x, self.bajo, self.alto)
        self.media = r.mean(axis=0)
        desv = r.std(axis=0)
        self.desv = np.where(desv > 1e-12, desv, 1.0)
        self.n_filas = int(len(x))
        self.videos = sorted(set(videos)) if videos is not None else []
        return self

    def transformar(self, x):
        if self.media is None:
            raise ValueError("el normalizador no esta ajustado")
        return (np.clip(np.asarray(x, np.float64), self.bajo, self.alto) - self.media) / self.desv

    def guardar(self, ruta):
        d = {"columnas": self.columnas, "p_bajo": self.p_bajo, "p_alto": self.p_alto, "bajo": self.bajo.tolist(),
             "alto": self.alto.tolist(), "media": self.media.tolist(), "desv": self.desv.tolist(), "n_filas": self.n_filas,
             "n_videos": len(self.videos), "videos": self.videos}
        carpeta = os.path.dirname(ruta)
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1, ensure_ascii=False)

    @classmethod
    def cargar(cls, ruta):
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        n = cls(d["columnas"], d["p_bajo"], d["p_alto"])
        n.bajo, n.alto = np.array(d["bajo"]), np.array(d["alto"])
        n.media, n.desv = np.array(d["media"]), np.array(d["desv"])
        n.n_filas, n.videos = d["n_filas"], d["videos"]
        return n


def leer_lista(ruta):
    with open(ruta, encoding="utf-8") as f:
        return {r["archivo"] for r in csv.DictReader(f)}


def leer_filas(ruta, incluir=None, excluir=()):
    with open(ruta, encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f)]
    return [r for r in filas if (incluir is None or r["archivo"] in incluir) and r["archivo"] not in excluir]


def matriz(filas, columnas):
    return np.array([[float(r[c]) for c in columnas] for r in filas], np.float64).reshape(len(filas), len(columnas))


def resumen(x, columnas, titulo):
    print(titulo)
    for j, c in enumerate(columnas):
        print(f"   {c:8s} | media {x[:, j].mean():9.4f} | desv {x[:, j].std():8.4f} | min {x[:, j].min():9.4f}"
              f" | max {x[:, j].max():9.4f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="accion", required=True)
    pa = sub.add_parser("ajustar")
    pa.add_argument("--entrada", default=os.path.join("resultados", "descriptores_v2", "descriptores.csv"))
    pa.add_argument("--lista", default=None)
    pa.add_argument("--excluir", nargs="*", default=[])
    pa.add_argument("--columnas", nargs="+", default=COLUMNAS)
    pa.add_argument("--p_bajo", type=float, default=1.0)
    pa.add_argument("--p_alto", type=float, default=99.0)
    pa.add_argument("--salida", default=os.path.join("resultados", "normalizacion.json"))
    pp = sub.add_parser("aplicar")
    pp.add_argument("--entrada", default=os.path.join("resultados", "descriptores_v2", "descriptores.csv"))
    pp.add_argument("--parametros", default=os.path.join("resultados", "normalizacion.json"))
    pp.add_argument("--lista", default=None)
    pp.add_argument("--excluir", nargs="*", default=[])
    pp.add_argument("--salida", default=os.path.join("resultados", "descriptores_v2", "descriptores_normalizados.csv"))
    a = p.parse_args()

    incluir = leer_lista(a.lista) if a.lista else None
    filas = leer_filas(a.entrada, incluir, set(a.excluir))
    if not filas:
        raise SystemExit("no hay filas para procesar")
    videos = sorted({r["archivo"] for r in filas})
    if a.accion == "ajustar":
        if incluir is None:
            print("aviso: sin --lista se usan todos los videos de la entrada; para el modelo usar solo los de entrenamiento")
        norm = Normalizador(a.columnas, a.p_bajo, a.p_alto).ajustar(matriz(filas, a.columnas), [r["archivo"] for r in filas])
        norm.guardar(a.salida)
        print(f"ajustado con {norm.n_filas} filas de {len(videos)} videos")
        for j, c in enumerate(norm.columnas):
            print(f"   {c:8s} | p{a.p_bajo:g} {norm.bajo[j]:10.5f} | p{a.p_alto:g} {norm.alto[j]:10.5f}"
                  f" | media {norm.media[j]:10.5f} | desv {norm.desv[j]:10.5f}")
        resumen(norm.transformar(matriz(filas, a.columnas)), norm.columnas, "filas de ajuste ya normalizadas:")
        print(f"parametros en: {a.salida}")
    else:
        norm = Normalizador.cargar(a.parametros)
        x = norm.transformar(matriz(filas, norm.columnas))
        campos = list(filas[0].keys())
        with open(a.salida, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            for r, v in zip(filas, x):
                r = dict(r)
                for c, valor in zip(norm.columnas, v):
                    r[c] = round(float(valor), 6)
                w.writerow(r)
        comunes = set(videos) & set(norm.videos)
        print(f"aplicado a {len(filas)} filas de {len(videos)} videos | videos tambien usados en el ajuste: {len(comunes)}")
        resumen(x, norm.columnas, "filas normalizadas:")
        print(f"normalizado en: {a.salida}")


if __name__ == "__main__":
    main()
