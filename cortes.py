import os
import re
import csv
import time
import random
import argparse
import cv2
import numpy as np
from preprocesamiento import LectorVideo, buscar_rutas

CAMPOS_CORTES = ["archivo", "t_corte", "motivo", "k", "h", "valido", "inliers"]
CAMPOS_RESUMEN = ["archivo", "duracion_s", "muestras", "n_cortes", "editado", "n_tramos", "tramo_max_s"]
CAMPOS_INDICADORES = ["archivo", "k", "t", "h", "valido", "inliers", "puntos", "corte"]
PEXELS = re.compile(r"^\d+(_\d+_\d+_\d+fps|-(hd|uhd|sd)_\d+_\d+_\d+fps)")
HASH = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{8}\.mp4$")


def histograma(imagen, ancho=160, bins=(16, 4, 4)):
    f = ancho / imagen.shape[1]
    peq = cv2.resize(imagen, (ancho, max(1, int(round(imagen.shape[0] * f)))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(peq, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    return (h / max(h.sum(), 1e-9)).astype(np.float32)


def distancia(a, b):
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


class DetectorCortes:
    def __init__(self, t_alto=0.85, t_bajo=0.65, separacion=1.0, ancho=160):
        self.t_alto, self.t_bajo, self.separacion, self.ancho = t_alto, t_bajo, separacion, ancho
        self.previo = None
        self.t_ultimo = -np.inf

    def actualizar(self, imagen, t, valido):
        actual = histograma(imagen, self.ancho)
        h = distancia(self.previo, actual) if self.previo is not None else 0.0
        primero = self.previo is None
        self.previo = actual
        motivo = ""
        if not primero and t - self.t_ultimo >= self.separacion:
            if h >= self.t_alto:
                motivo = "histograma"
            elif h >= self.t_bajo and not valido:
                motivo = "histograma_y_camara"
        if motivo:
            self.t_ultimo = t
        return h, motivo


def estado_tramos(t_inc, inicios):
    if t_inc is None:
        return ["enmascarado"] * len(inicios)
    limites = list(inicios[1:]) + [np.inf]
    dentro = next(j for j, (a, b) in enumerate(zip(inicios, limites)) if a <= t_inc < b) if t_inc >= inicios[0] else 0
    return ["normal" if j == dentro else "peligro" if j > dentro else "enmascarado" for j in range(len(inicios))]


def tau_valido(t_k, tramo, t_inc, inicios):
    estado = estado_tramos(t_inc, inicios)[tramo]
    if estado == "enmascarado":
        return 0.0, False
    if estado == "peligro":
        return 0.0, True
    return max(0.0, t_inc - t_k), True


def resumen_video(archivo, duracion, muestras, cortes, inicios, t_fin):
    fines = list(inicios[1:]) + [t_fin]
    largos = [b - a for a, b in zip(inicios, fines)]
    return {"archivo": archivo, "duracion_s": round(duracion, 3), "muestras": muestras, "n_cortes": len(cortes),
            "editado": "si" if cortes else "no", "n_tramos": len(inicios), "tramo_max_s": round(max(largos), 3)}


def abrir_csv(ruta, campos, agregar):
    nuevo = not (agregar and os.path.exists(ruta))
    f = open(ruta, "w" if nuevo else "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=campos)
    if nuevo:
        w.writeheader()
    return f, w


def conjunto_389():
    ia = {r["archivo"] for r in csv.DictReader(open(os.path.join("resultados", "etiquetas_ia.csv"), encoding="utf-8"))}
    cand = set()
    for ruta in ["propuestas.csv", "propuestas_extra.csv"]:
        for r in csv.DictReader(open(ruta, encoding="utf-8")):
            if r.get("t_incipiente") or r.get("t_incontrolado"):
                cand.add(r["archivo"])
    return ia | cand


def seleccionar(a):
    meta = {r["archivo"]: r for r in csv.DictReader(open("metadatos_videos.csv", encoding="utf-8"))}
    nombres = sorted(v for v in conjunto_389() if not PEXELS.match(v) and not HASH.match(v)
                     and float(meta[v]["duracion_s"]) <= a.dur_max)
    elegidos = random.Random(a.semilla).sample(nombres, min(a.n, len(nombres)))
    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    with open(a.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "duracion_s"])
        for v in elegidos:
            w.writerow([v, meta[v]["duracion_s"]])
    total = sum(float(meta[v]["duracion_s"]) for v in elegidos)
    print(f"{len(elegidos)} de {len(nombres)} videos con nombre de hasta {a.dur_max:g} s | {total / 3600:.2f} h | lista en: {a.salida}")


def listas(a):
    meta = [r for r in csv.DictReader(open("metadatos_videos.csv", encoding="utf-8"))]
    por_origen = {}
    for r in meta:
        if r["clase"] == "non_fire":
            por_origen.setdefault(r["origen"], []).append(r)
    rng = random.Random(a.semilla)
    sin_fuego = [rng.choice(sorted(por_origen[o], key=lambda r: r["archivo"])) for o in sorted(por_origen)]
    os.makedirs(os.path.dirname(a.salida_non_fire), exist_ok=True)
    with open(a.salida_non_fire, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "origen", "duracion_s"])
        for r in sin_fuego:
            w.writerow([r["archivo"], r["origen"], r["duracion_s"]])
    dur = {r["archivo"]: float(r["duracion_s"]) for r in meta}
    fuego = sorted(conjunto_389())
    with open(a.salida_noche, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "clase", "duracion_s"])
        for v in fuego:
            w.writerow([v, "fire", dur[v]])
        for r in sin_fuego:
            w.writerow([r["archivo"], "non_fire", r["duracion_s"]])
    print(f"sin fuego: {len(sin_fuego)} videos de {len(por_origen)} origenes | {sum(float(r['duracion_s']) for r in sin_fuego) / 3600:.2f} h"
          f" | lista en: {a.salida_non_fire}")
    print(f"corrida completa: {len(fuego)} de fuego ({sum(dur[v] for v in fuego) / 3600:.2f} h) + {len(sin_fuego)} sin fuego"
          f" | lista en: {a.salida_noche}")


def par(img_a, img_b, texto, ancho=480):
    def red(i):
        return cv2.resize(i, (ancho, int(round(i.shape[0] * ancho / i.shape[1]))), interpolation=cv2.INTER_AREA)
    a, b = red(img_a), red(img_b)
    alto = max(a.shape[0], b.shape[0])
    lienzo = np.zeros((alto + 30, 2 * ancho + 10, 3), np.uint8)
    lienzo[30:30 + a.shape[0], :ancho] = a
    lienzo[30:30 + b.shape[0], ancho + 10:] = b
    cv2.putText(lienzo, texto, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return lienzo


def explorar(a):
    with open(a.lista, encoding="utf-8") as f:
        lista = [r["archivo"] for r in csv.DictReader(f)]
    rutas = buscar_rutas(a.videos)
    os.makedirs(a.salida, exist_ok=True)
    os.makedirs(a.revision, exist_ok=True)
    f_c, w_c = abrir_csv(os.path.join(a.salida, "exploracion.csv"), ["archivo", "k", "t", "h", "par"], False)
    f_r, w_r = abrir_csv(os.path.join(a.salida, "exploracion_resumen.csv"),
                         ["archivo", "muestras", "h_mediana", "h_p99", "h_max", "n_candidatos", "segundos"], False)
    for n, nombre in enumerate(lista, 1):
        if nombre not in rutas:
            print(f"no encontrado: {nombre}")
            continue
        inicio = time.time()
        previo_h, previo_img, hs, cands = None, None, [], []
        for m in LectorVideo(rutas[nombre], a.fps):
            actual = histograma(m.imagen)
            peq = cv2.resize(m.imagen, (480, int(round(m.imagen.shape[0] * 480 / m.imagen.shape[1]))), interpolation=cv2.INTER_AREA)
            if previo_h is not None:
                h = distancia(previo_h, actual)
                hs.append(h)
                if h >= a.umbral:
                    cands.append((h, m.k, m.t, previo_img, peq))
            previo_h, previo_img = actual, peq
        cands.sort(key=lambda c: -c[0])
        elegidos = []
        for c in cands:
            if all(abs(c[2] - e[2]) >= 1.0 for e in elegidos):
                elegidos.append(c)
        base = re.sub(r"[^0-9A-Za-z]+", "_", os.path.splitext(nombre)[0])[:50]
        for h, k, t, ia, ib in sorted(elegidos, key=lambda c: c[2]):
            ruta_par = ""
            if len([e for e in elegidos if e[0] >= h]) <= a.pares:
                ruta_par = os.path.join(a.revision, f"{base}_t{t:07.2f}_h{h:.2f}.jpg")
                cv2.imwrite(ruta_par, par(ia, ib, f"{nombre[:60]} | t {t:.2f} s | h {h:.3f}"))
            w_c.writerow({"archivo": nombre, "k": k, "t": round(t, 3), "h": round(h, 4), "par": ruta_par})
        hs = np.array(hs) if hs else np.zeros(1)
        seg = time.time() - inicio
        w_r.writerow({"archivo": nombre, "muestras": len(hs) + 1, "h_mediana": round(float(np.median(hs)), 4),
                      "h_p99": round(float(np.percentile(hs, 99)), 4), "h_max": round(float(hs.max()), 4),
                      "n_candidatos": len(elegidos), "segundos": round(seg, 1)})
        f_c.flush()
        f_r.flush()
        print(f"[{n}/{len(lista)}] {nombre[:50]} | muestras {len(hs) + 1} | h mediana {np.median(hs):.3f}"
              f" | h max {hs.max():.3f} | candidatos {len(elegidos)} | {seg:.1f} s")
    f_c.close()
    f_r.close()


def simular(filas, t_alto, t_bajo, separacion):
    detectados, t_ultimo = [], -np.inf
    for r in filas:
        if r["k"] == 0 or r["t"] - t_ultimo < separacion:
            continue
        if r["h"] >= t_alto or (r["h"] >= t_bajo and r["valido"] == 0):
            detectados.append(r)
            t_ultimo = r["t"]
    return detectados


def evaluar(por_video, revisados, t_alto, t_bajo, separacion, ventana):
    res = {"aciertos": [], "perdidos": [], "falsos": [], "sin_revisar": [], "duplicados": []}
    for v, filas in por_video.items():
        rev = revisados.get(v, [])
        usados = set()
        for d in simular(filas, t_alto, t_bajo, separacion):
            cerca = sorted((abs(d["t"] - t), t, tipo) for t, tipo in rev if abs(d["t"] - t) <= ventana)
            if not cerca:
                res["sin_revisar"].append((v, d))
                continue
            _, t, tipo = cerca[0]
            if tipo != "corte":
                res["falsos"].append((v, d))
            elif t in usados:
                res["duplicados"].append((v, d))
            else:
                usados.add(t)
                res["aciertos"].append((v, d))
        res["perdidos"] += [(v, t) for t, tipo in rev if tipo == "corte" and t not in usados]
    return res


def umbrales(a):
    grupos = {r["archivo"]: r["grupo"] for r in csv.DictReader(open(a.lista, encoding="utf-8"))}
    por_video = {}
    for r in csv.DictReader(open(a.indicadores, encoding="utf-8")):
        por_video.setdefault(r["archivo"], []).append({"k": int(r["k"]), "t": float(r["t"]), "h": float(r["h"]),
                                                       "valido": int(r["valido"]), "inliers": float(r["inliers"])})
    revisados = {}
    for r in csv.DictReader(open(a.confirmados, encoding="utf-8")):
        revisados.setdefault(r["archivo"], []).append((float(r["t_corte"]), r["tipo"]))
    conjuntos = {"ajuste": ["ajuste", "prueba"], "validacion": ["validacion"]}
    sub = {c: {v: f for v, f in por_video.items() if grupos.get(v) in gs} for c, gs in conjuntos.items()}
    for c, pv in sub.items():
        n_cortes = sum(1 for v in pv for _, tipo in revisados.get(v, []) if tipo == "corte")
        n_no = sum(1 for v in pv for _, tipo in revisados.get(v, []) if tipo != "corte")
        muestras = sum(len(f) for f in pv.values())
        print(f"{c}: {len(pv)} videos | {muestras} muestras | pares revisados: {n_cortes} cortes y {n_no} sin corte")
        tipos = {"corte": [], "no_corte": []}
        claves = set()
        for v, f in pv.items():
            for t, tipo in revisados.get(v, []):
                r = min(f, key=lambda x: abs(x["t"] - t))
                if abs(r["t"] - t) <= 0.06:
                    tipos["corte" if tipo == "corte" else "no_corte"].append(r)
                    claves.add((v, r["k"]))
        resto = [r for v, f in pv.items() for r in f if r["k"] > 0 and (v, r["k"]) not in claves]
        for nombre, filas in [("dentro de toma (sin revisar)", resto), ("pares sin corte", tipos["no_corte"]), ("pares con corte", tipos["corte"])]:
            if not filas:
                continue
            h = np.array([r["h"] for r in filas])
            inval = np.array([r["valido"] == 0 for r in filas])
            print(f"   {nombre:28s} | n {len(h):6d} | h p50 {np.median(h):.3f} | p99 {np.percentile(h, 99):.3f} | max {h.max():.3f}"
                  f" | camara no valida {100 * inval.mean():5.1f} %")
    filas_rejilla = []
    pasos = np.round(np.arange(0.25, 1.0001, 0.025), 3)
    for t_alto in pasos:
        for t_bajo in pasos[pasos <= t_alto]:
            fila = {"t_alto": t_alto, "t_bajo": t_bajo}
            for c, pv in sub.items():
                e = evaluar(pv, revisados, t_alto, t_bajo, a.separacion, a.ventana)
                for k in ["aciertos", "perdidos", "falsos", "sin_revisar", "duplicados"]:
                    fila[f"{c}_{k}"] = len(e[k])
            filas_rejilla.append(fila)
    ruta = os.path.join(os.path.dirname(a.indicadores), "umbrales_rejilla.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_rejilla[0].keys()))
        w.writeheader()
        w.writerows(filas_rejilla)
    limpias = [f for f in filas_rejilla if f["ajuste_falsos"] == 0 and f["ajuste_sin_revisar"] == 0 and f["ajuste_duplicados"] == 0]
    mejor = max(f["ajuste_aciertos"] for f in limpias) if limpias else 0
    print(f"rejilla en: {ruta}")
    print(f"ajuste sin falsos: maximo {mejor} aciertos; combinaciones que lo logran:")
    for f in limpias:
        if f["ajuste_aciertos"] == mejor:
            print(f"   t_alto {f['t_alto']:.3f} | t_bajo {f['t_bajo']:.3f} | validacion: aciertos {f['validacion_aciertos']}"
                  f" perdidos {f['validacion_perdidos']} falsos {f['validacion_falsos']} sin revisar {f['validacion_sin_revisar']}")
    if a.t_alto is not None:
        for c, pv in sub.items():
            e = evaluar(pv, revisados, a.t_alto, a.t_bajo, a.separacion, a.ventana)
            print(f"{c} con t_alto {a.t_alto} y t_bajo {a.t_bajo}: aciertos {len(e['aciertos'])} | perdidos {len(e['perdidos'])}"
                  f" | falsos {len(e['falsos'])} | sin revisar {len(e['sin_revisar'])} | duplicados {len(e['duplicados'])}")
            for v, d in e["aciertos"]:
                print(f"   acierto     {v[:40]:40s} t {d['t']:7.2f} | h {d['h']:.3f} | valido {d['valido']}")
            for v, t in e["perdidos"]:
                r = min(pv[v], key=lambda x: abs(x["t"] - t))
                print(f"   perdido     {v[:40]:40s} t {t:7.2f} | h {r['h']:.3f} | valido {r['valido']}")
            for k in ["falsos", "sin_revisar", "duplicados"]:
                for v, d in e[k]:
                    print(f"   {k:11s} {v[:40]:40s} t {d['t']:7.2f} | h {d['h']:.3f} | valido {d['valido']}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="accion", required=True)
    ps = sub.add_parser("seleccionar")
    ps.add_argument("--n", type=int, default=60)
    ps.add_argument("--semilla", type=int, default=42)
    ps.add_argument("--dur_max", type=float, default=300.0)
    ps.add_argument("--salida", default=os.path.join("resultados", "cortes", "lista_exploracion.csv"))
    pl = sub.add_parser("listas")
    pl.add_argument("--semilla", type=int, default=42)
    pl.add_argument("--salida_non_fire", default=os.path.join("resultados", "cortes", "lista_non_fire.csv"))
    pl.add_argument("--salida_noche", default=os.path.join("resultados", "cortes", "lista_noche.csv"))
    pe = sub.add_parser("explorar")
    pe.add_argument("--lista", default=os.path.join("resultados", "cortes", "lista_exploracion.csv"))
    pe.add_argument("--videos", nargs="+", default=[os.path.join("real_video", "real_video", "fire"),
                                                    os.path.join("real_video", "real_video", "non_fire")])
    pe.add_argument("--fps", type=float, default=10.0)
    pe.add_argument("--umbral", type=float, default=0.25)
    pe.add_argument("--pares", type=int, default=3)
    pe.add_argument("--salida", default=os.path.join("resultados", "cortes"))
    pe.add_argument("--revision", default=os.path.join("runs_fuego", "cortes", "revision"))
    pu = sub.add_parser("umbrales")
    pu.add_argument("--indicadores", default=os.path.join("resultados", "cortes", "indicadores", "indicadores.csv"))
    pu.add_argument("--confirmados", default=os.path.join("resultados", "cortes", "confirmados.csv"))
    pu.add_argument("--lista", default=os.path.join("resultados", "cortes", "lista_indicadores.csv"))
    pu.add_argument("--separacion", type=float, default=1.0)
    pu.add_argument("--ventana", type=float, default=0.5)
    pu.add_argument("--t_alto", type=float, default=None)
    pu.add_argument("--t_bajo", type=float, default=None)
    a = p.parse_args()
    {"seleccionar": seleccionar, "listas": listas, "explorar": explorar, "umbrales": umbrales}[a.accion](a)


if __name__ == "__main__":
    main()
