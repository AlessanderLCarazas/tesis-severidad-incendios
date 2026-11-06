import os
import csv
import argparse
from collections import defaultdict
import numpy as np

DESCRIPTORES = [f"d{i}" for i in range(1, 14)]
CAMBIAN = ["d3", "d4", "d10", "d11", "d12", "d13"]
TITULOS = {"d2": "d2 area", "d3": "d3 tasa de crecimiento (1/s)", "d4": "d4 crecimiento relativo (1/s)",
           "d10": "d10 confianza de humo", "d11": "d11 area de humo", "d12": "d12 tasa del humo (1/s)",
           "d13": "d13 regiones activas"}
COLOR_ANTES, COLOR_DESPUES = "#eb6834", "#2a78d6"
CAMPOS = ["ambito", "descriptor", "version", "filas", "media", "desv", "min", "p1", "mediana", "p99", "max",
          "variacion_media", "frac_ceros", "frac_saturadas", "frac_abs_ge5"]


def leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return {(r["archivo"], int(r["id"]), int(r["k"])): r for r in csv.DictReader(f)}


def variacion(filas, col):
    por_id = defaultdict(list)
    for (archivo, tid, k), r in filas:
        por_id[(archivo, tid)].append((k, float(r[col])))
    difs = []
    for v in por_id.values():
        v.sort()
        difs.extend(abs(b[1] - a[1]) for a, b in zip(v, v[1:]))
    return float(np.mean(difs)) if difs else 0.0


def estadisticas(filas, col):
    x = np.array([float(r[col]) for _, r in filas])
    return {"filas": len(x), "media": x.mean(), "desv": x.std(), "min": x.min(), "p1": np.percentile(x, 1),
            "mediana": np.median(x), "p99": np.percentile(x, 99), "max": x.max(), "variacion_media": variacion(filas, col),
            "frac_ceros": float((x == 0).mean()), "frac_saturadas": float((x >= 0.999).mean()),
            "frac_abs_ge5": float((np.abs(x) >= 5).mean())}


def grafica(antes, despues, video, tid, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    claves = sorted((k for k in despues if k[0] == video and k[1] == tid), key=lambda c: c[2])
    if not claves:
        print(f"sin filas para {video} id {tid}")
        return
    t = np.array([float(despues[c]["t"]) for c in claves])
    paneles = ["d2", "d3", "d4", "d10", "d11", "d12", "d13"]
    fig, ejes = plt.subplots(len(paneles), 1, figsize=(10, 13), sharex=True)
    for eje, col in zip(ejes, paneles):
        if col == "d2":
            eje.plot(t, [float(antes[c]["d2"]) for c in claves], color=COLOR_ANTES, lw=1.2, label="d2 instantanea (v1 = v2)")
            eje.plot(t, [float(despues[c]["d2_med"]) for c in claves], color=COLOR_DESPUES, lw=1.6, label="d2_med, mediana de 1 s (v2)")
        else:
            eje.plot(t, [float(antes[c][col]) for c in claves], color=COLOR_ANTES, lw=1.2, label="v1 (antes)")
            eje.plot(t, [float(despues[c][col]) for c in claves], color=COLOR_DESPUES, lw=1.6, label="v2 (despues)")
        eje.set_title(TITULOS[col], fontsize=9, loc="left", color="#0b0b0b")
        eje.grid(alpha=0.25, lw=0.6)
        eje.tick_params(labelsize=8, colors="#52514e")
        for lado in ["top", "right"]:
            eje.spines[lado].set_visible(False)
        for lado in ["left", "bottom"]:
            eje.spines[lado].set_color("#b5b4ae")
        eje.legend(fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 0.98), frameon=False, ncol=2, borderaxespad=0.0)
    ejes[-1].set_xlabel("t (s)", fontsize=9)
    fig.suptitle(f"{video} | trayectoria {tid} | {len(claves)} muestras | v1 contra v2", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    print(f"grafica en: {ruta}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--antes", default=os.path.join("resultados", "descriptores", "descriptores.csv"))
    p.add_argument("--despues", default=os.path.join("resultados", "descriptores_v2", "descriptores.csv"))
    p.add_argument("--video", default="14294880_3840_2160_24fps.mp4")
    p.add_argument("--id", type=int, default=1)
    p.add_argument("--salida", default=os.path.join("resultados", "descriptores_v2"))
    p.add_argument("--solo_igualdad", action="store_true")
    a = p.parse_args()

    antes, despues = leer(a.antes), leer(a.despues)
    comunes = sorted(set(antes) & set(despues))
    print(f"filas: antes {len(antes)} | despues {len(despues)} | comunes {len(comunes)}"
          f" | solo antes {len(set(antes) - set(despues))} | solo despues {len(set(despues) - set(antes))}")
    for col in DESCRIPTORES + ["s_k", "rot", "tx", "ty", "inliers", "t", "dt"]:
        difs = np.array([abs(float(antes[c][col]) - float(despues[c][col])) for c in comunes])
        print(f"   {col:8s} | filas distintas {int((difs > 0).sum()):5d} | max diferencia {difs.max() if len(difs) else 0:.3g}")
    if a.solo_igualdad:
        return
    os.makedirs(a.salida, exist_ok=True)
    videos = sorted({c[0] for c in comunes})
    ruta_csv = os.path.join(a.salida, "comparacion.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        for ambito in videos + ["total"]:
            claves = [c for c in comunes if ambito == "total" or c[0] == ambito]
            for col in CAMBIAN:
                for version, datos in [("v1", antes), ("v2", despues)]:
                    e = estadisticas([(c, datos[c]) for c in claves], col)
                    w.writerow({"ambito": ambito, "descriptor": col, "version": version,
                                **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in e.items()}})
                    if ambito == "total":
                        print(f"total | {col:4s} | {version} | media {e['media']:8.4f} | desv {e['desv']:7.4f}"
                              f" | p1 {e['p1']:8.4f} | mediana {e['mediana']:7.4f} | p99 {e['p99']:7.4f}"
                              f" | min {e['min']:8.3f} | max {e['max']:7.3f} | variacion {e['variacion_media']:.4f}"
                              f" | ceros {e['frac_ceros']:.3f} | saturadas {e['frac_saturadas']:.3f} | abs>=5 {e['frac_abs_ge5']:.3f}")
    print(f"comparacion en: {ruta_csv}")
    grafica(antes, despues, a.video, a.id,
            os.path.join(a.salida, f"comparacion_{os.path.splitext(a.video)[0]}_id{a.id}.png"))


if __name__ == "__main__":
    main()
