import os
import json
import argparse
import itertools
import numpy as np
from escena_v2 import vectores_carpeta
from vector_escena import COLUMNAS
from entrenar_temporal import verdad_escena, confusion, f1_macro
from entrenar_escena import resumen_conf
from fase_e import histeresis, por_segundo
from secuencias import leer_csv

CONJUNTOS = ["calibracion", "validacion", "prueba"]
I_FUEGO, I_HUMO = COLUMNAS.index("area_det_fuego"), COLUMNAS.index("area_det_humo")


def series(a, particion):
    videos = {c: sorted(v for v, r in particion.items() if r["conjunto"] == c) for c in CONJUNTOS}
    todos = set(v for vs in videos.values() for v in vs)
    vec = vectores_carpeta(a.descriptores_v3, todos, 1.0, 10, "_alta")
    salida = {c: {} for c in CONJUNTOS}
    for c, vs in videos.items():
        for v in vs:
            d = vec[v]
            sel = np.zeros(len(d["t"]), bool)
            for j in np.unique(d["tramo"]):
                idx = np.flatnonzero(d["tramo"] == j)
                if len(idx) >= a.min_fotogramas:
                    sel[idx] = True
            if sel.any():
                orden = np.flatnonzero(sel)
                orden = orden[np.argsort(d["t"][orden], kind="stable")]
                salida[c][v] = {"t": d["t"][orden].astype(np.float64), "fuego": d["x"][orden, I_FUEGO], "humo": d["x"][orden, I_HUMO]}
    return salida


def nivel(s, f1, f2, h1, h2):
    alto = (s["fuego"] >= f2) | (s["humo"] >= h2)
    medio = (s["fuego"] >= f1) | (s["humo"] >= h1)
    return np.where(alto, 2, np.where(medio, 1, 0)).astype(np.int64)


def riesgo(s, f2, h2):
    return np.maximum(s["fuego"] / f2 if f2 > 0 else np.where(s["fuego"] > 0, 1.0, 0.0),
                      s["humo"] / h2 if h2 > 0 else np.where(s["humo"] > 0, 1.0, 0.0))


def preparar_busqueda(ser, verdad):
    t_seg, real, idx_seg, fuego, humo = [], [], [], [], []
    n = 0
    for v, seg in verdad.items():
        s = ser.get(v)
        segundos = sorted(seg)
        mapa = {x: n + i for i, x in enumerate(segundos)}
        real += [seg[x] for x in segundos]
        if s is not None:
            sf = np.floor(s["t"]).astype(int)
            m = np.array([x in mapa for x in sf], bool)
            idx_seg.append(np.array([mapa[x] for x in sf[m]], np.int64))
            fuego.append(s["fuego"][m])
            humo.append(s["humo"][m])
        n += len(segundos)
    return np.array(real, np.int64), np.concatenate(idx_seg), np.concatenate(fuego), np.concatenate(humo), n


def f1_rapido(real, idx, fuego, humo, n, f1, f2, h1, h2):
    lvl = np.where((fuego >= f2) | (humo >= h2), 2, np.where((fuego >= f1) | (humo >= h1), 1, 0))
    cuenta = np.bincount(idx * 3 + lvl, minlength=3 * n).reshape(n, 3)
    est = 2 - np.argmax(cuenta[:, ::-1], axis=1)
    est[cuenta.sum(1) == 0] = 0
    return f1_macro(confusion(real, est))[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--descriptores_v3", default=os.path.join("runs_fuego", "descriptores_v3"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--min_fotogramas", type=int, default=10)
    p.add_argument("--percentiles", nargs="+", type=float, default=list(range(5, 100, 5)))
    p.add_argument("--finalistas", type=int, default=50)
    p.add_argument("--permanencia", type=float, default=2.0)
    p.add_argument("--factor_desactivar", type=float, default=0.8)
    p.add_argument("--alfas", nargs="+", type=float, default=[0.10, 0.05])
    p.add_argument("--salida", default=os.path.join("runs_fuego", "cap5", "regla"))
    p.add_argument("--resumen", default=os.path.join("resultados", "cap5", "regla.json"))
    a = p.parse_args()

    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    print("leyendo el vector de escena sin normalizar...", flush=True)
    ser = series(a, particion)
    metas = {m["archivo"]: m for m in leer_csv(a.metas) if m["conjunto"] == "validacion" and m["fuente"] == "ia"}
    verdad = verdad_escena(metas, sorted(metas))
    fuegos = np.concatenate([ser["validacion"][v]["fuego"] for v in verdad if v in ser["validacion"]])
    humos = np.concatenate([ser["validacion"][v]["humo"] for v in verdad if v in ser["validacion"]])
    cand_f = sorted(set([0.0] + [float(np.percentile(fuegos, q)) for q in a.percentiles]))
    cand_h = sorted(set([0.0] + [float(np.percentile(humos, q)) for q in a.percentiles]))
    real, idx, fuego, humo, n = preparar_busqueda(ser["validacion"], verdad)
    pares_f = [(x, y) for x, y in itertools.combinations_with_replacement(cand_f, 2)]
    pares_h = [(x, y) for x, y in itertools.combinations_with_replacement(cand_h, 2)]
    print(f"candidatos: fuego {len(cand_f)}, humo {len(cand_h)}, combinaciones {len(pares_f) * len(pares_h)}", flush=True)
    puntajes = []
    for f1, f2 in pares_f:
        for h1, h2 in pares_h:
            puntajes.append((f1_rapido(real, idx, fuego, humo, n, f1, f2, h1, h2), f1, f2, h1, h2))
    puntajes.sort(key=lambda x: -x[0])
    finalistas = []
    for sin_h, f1, f2, h1, h2 in puntajes[:a.finalistas]:
        niveles = {v: histeresis(nivel(s, f1, f2, h1, h2), s["t"], a.permanencia) for v, s in ser["validacion"].items()}
        tiempos = {v: s["t"] for v, s in ser["validacion"].items()}
        r = por_segundo(niveles, tiempos, verdad)
        finalistas.append({"f1_final": r["f1"], "f1_sin_histeresis": sin_h, "kappa": r["kappa"], "umbrales": [f1, f2, h1, h2]})
    mejor = max(finalistas, key=lambda x: x["f1_final"])
    f1, f2, h1, h2 = mejor["umbrales"]
    cal = sorted(v for v, r in particion.items() if r["conjunto"] == "calibracion" and r["calibra_umbral"] == "si")
    puntos = sorted(float(riesgo(ser["calibracion"][v], f2, h2).max()) if v in ser["calibracion"] else 0.0 for v in cal)
    umbral = {}
    for alfa in a.alfas:
        k = int(np.ceil((len(puntos) + 1) * (1.0 - alfa)))
        umbral[f"{alfa:g}"] = puntos[k - 1] if k <= len(puntos) else float("inf")
    os.makedirs(a.salida, exist_ok=True)
    for c in CONJUNTOS:
        filas = {"archivo": [], "t": [], "nivel": [], "riesgo": []}
        for v, s in ser[c].items():
            filas["archivo"] += [v] * len(s["t"])
            filas["t"].append(s["t"])
            filas["nivel"].append(nivel(s, f1, f2, h1, h2))
            filas["riesgo"].append(riesgo(s, f2, h2))
        np.savez_compressed(os.path.join(a.salida, f"salidas_{c}.npz"), archivo=np.array(filas["archivo"], dtype=str),
                            t=np.concatenate(filas["t"]), nivel=np.concatenate(filas["nivel"]), riesgo=np.concatenate(filas["riesgo"]))
    res = {"umbrales": {"fuego_incipiente": f1, "fuego_incontrolado": f2, "humo_incipiente": h1, "humo_incontrolado": h2},
           "umbral_alerta": umbral, "validacion": mejor, "candidatos_fuego": cand_f, "candidatos_humo": cand_h,
           "combinaciones": len(puntajes), "finalistas": finalistas, "videos_calibracion": len(puntos)}
    os.makedirs(os.path.dirname(a.resumen), exist_ok=True)
    with open(a.resumen, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"umbrales: fuego {f1:.4f} / {f2:.4f} | humo {h1:.4f} / {h2:.4f} | f1 final en validacion {mejor['f1_final']:.4f}"
          f" (sin histeresis {mejor['f1_sin_histeresis']:.4f}) | umbral de alerta {umbral['0.1']:.4f} | salidas en {a.salida}")


if __name__ == "__main__":
    main()
