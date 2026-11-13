import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from normalizacion import Normalizador
from evaluar_prueba import por_segundo_video
from analisis_prueba import especificacion, predecir, referencias, titulo
from evaluar_robustez import FRACCIONES, armar
from secuencias import leer_csv

PORCENTAJE = {"completa": 0, "quitar_25": 25, "quitar_50": 50, "quitar_75": 75}
TEMPORALES = ["cfc", "gru", "lstm"]
TODOS = ["cfc", "gru", "lstm", "gradient boosting", "regresion logistica"]
COLOR = {"cfc": "#2a78d6", "gru": "#eb6834", "lstm": "#1baf7a", "gradient boosting": "#eda100", "regresion logistica": "#e87ba4"}
MARCA = {"cfc": "o", "gru": "s", "lstm": "^", "gradient boosting": "D", "regresion logistica": "v"}
TINTA, TINTA_2, REJILLA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def f1_kappa(m):
    m = np.asarray(m, np.float64)
    tp = np.diagonal(m, axis1=-2, axis2=-1)
    den = m.sum(-1) + m.sum(-2)
    f1 = np.where(den > 0, 2.0 * tp / np.where(den > 0, den, 1.0), 0.0).mean(-1)
    n = m.sum((-2, -1))
    i, j = np.indices((3, 3))
    w = np.abs(i - j) / 2.0
    esperado = m.sum(-1)[..., :, None] * m.sum(-2)[..., None, :] / np.maximum(n, 1)[..., None, None]
    den_k = (w * esperado).sum((-2, -1))
    kappa = np.where(den_k > 0, 1.0 - (w * m).sum((-2, -1)) / np.where(den_k > 0, den_k, 1.0), np.nan)
    return f1, kappa


def confusiones(c, ref):
    salida = np.zeros((len(ref["nivel"]), 3, 3), np.int64)
    vacio = {"t": np.zeros(0), "p": np.zeros((0, 3))}
    for n, v in enumerate(ref["nivel"]):
        _, real, est, _ = por_segundo_video(c["pred"].get(v, vacio), c["final"].get(v, np.zeros(0, int)), ref["verdad"][v])
        np.add.at(salida[n], (real, est), 1)
    return salida


def ic(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def fila_ic(punto, muestras, formato="+.3f"):
    a, b = ic(muestras)
    return f"{format(punto, formato)} | {format(a, formato)} a {format(b, formato)}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    p.add_argument("--robustos", default=os.path.join("resultados", "robustez", "modelos_robustos.json"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--parte1", default=os.path.join("resultados", "robustez", "parte1.json"))
    p.add_argument("--parte2", default=os.path.join("resultados", "robustez", "parte2.json"))
    p.add_argument("--remuestreos", type=int, default=1000)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", default=os.path.join("resultados", "robustez"))
    a = p.parse_args()

    with open(a.config, encoding="utf-8") as f:
        config = json.load(f)
    with open(a.robustos, encoding="utf-8") as f:
        rob = json.load(f)
    guardado = {}
    for parte, ruta in (("parte1", a.parte1), ("parte2", a.parte2)):
        with open(ruta, encoding="utf-8") as f:
            guardado[parte] = json.load(f)["resultados"]
    dec = config["decision"]
    norm_x = Normalizador.cargar(config["datos"]["normalizacion"][0])
    norm_c = Normalizador.cargar(config["datos"]["normalizacion"][1])
    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    videos = sorted(v for v, r in particion.items() if r["conjunto"] == "prueba")
    refs, _ = referencias("prueba", particion, a.metas)
    temporales, sin_tiempo = especificacion(config)
    robustos = [{"corrida": n, "modelo": c["modelo"], "pesos": c["archivo_pesos"], "sha256": c["sha256"], "unidad": 1.0,
                 "umbral": c["umbral_alerta"]} for n, c in rob["corridas"].items()]

    conf = {"parte1": {}, "parte2": {}}
    modelo_de = {"parte1": {}, "parte2": {}}
    peor = 0.0
    for fr, carpeta in FRACCIONES:
        print(f"prediciendo: {fr}", flush=True)
        datos, info, _ = armar(carpeta, videos, norm_x, norm_c)
        for parte, corridas in (("parte1", predecir(datos, info, temporales, sin_tiempo, dec)), ("parte2", predecir(datos, info, robustos, [], dec))):
            for c in corridas:
                modelo_de[parte][c["corrida"]] = c["modelo"]
                for clave, ref in refs.items():
                    m = confusiones(c, ref)
                    conf[parte][(fr, c["corrida"], clave)] = m
                    f1, kappa = f1_kappa(m.sum(0))
                    g = guardado[parte][fr][c["corrida"]]["referencias"][clave]
                    peor = max(peor, abs(float(f1) - g["f1"]), abs(float(kappa) - g["kappa"]))
    print(f"diferencia maxima con parte1.json y parte2.json: {peor:.2e}", flush=True)
    if peor > 1e-9:
        raise SystemExit("las predicciones no coinciden con las evaluaciones guardadas")

    fracciones = [f for f, _ in FRACCIONES]
    puntos, muestras = {}, {}
    for clave, ref in refs.items():
        n = len(ref["nivel"])
        rng = np.random.default_rng(a.semilla)
        pesos = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(a.remuestreos)]).astype(np.float64)
        for parte in ("parte1", "parte2"):
            for m in set(modelo_de[parte].values()):
                corridas = [c for c, mm in modelo_de[parte].items() if mm == m]
                for fr in fracciones:
                    f1s, ks, f1p, kp = [], [], [], []
                    for c in corridas:
                        cv = conf[parte][(fr, c, clave)]
                        f, k = f1_kappa(np.tensordot(pesos, cv, axes=1))
                        f1s.append(f)
                        ks.append(k)
                        f0, k0 = f1_kappa(cv.sum(0))
                        f1p.append(float(f0))
                        kp.append(float(k0))
                    muestras[(clave, parte, m, fr)] = {"f1": np.mean(f1s, axis=0), "kappa": np.nanmean(ks, axis=0)}
                    puntos[(clave, parte, m, fr)] = {"f1": float(np.mean(f1p)), "kappa": float(np.mean(kp)), "f1_sd": float(np.std(f1p)),
                                                     "kappa_sd": float(np.std(kp)), "n": len(corridas)}

    def caida_rel(clave, parte, m, fr, k="f1"):
        p0, pq = puntos[(clave, parte, m, "completa")][k], puntos[(clave, parte, m, fr)][k]
        s0, sq = muestras[(clave, parte, m, "completa")][k], muestras[(clave, parte, m, fr)][k]
        return (p0 - pq) / p0, (s0 - sq) / s0

    nombre_ref = {"ia": "IA", "humano": "humana"}
    salida_json = {"remuestreos": a.remuestreos, "semilla": a.semilla, "control": peor, "puntos": {}, "pruebas": []}
    for (clave, parte, m, fr), v in puntos.items():
        salida_json["puntos"][f"{clave}|{parte}|{m}|{fr}"] = v
    lineas = ["# Robustez: caidas, pruebas pareadas y figura", ""]
    lineas.append(f"Generado por `python analisis_robustez.py`. Repite la inferencia de las partes 1 y 2 (diferencia maxima de F1 y kappa con"
                  f" `parte1.json` y `parte2.json`: {peor:.1e}) y hace un bootstrap pareado de videos ({a.remuestreos} remuestreos,"
                  f" semilla {a.semilla}; los mismos remuestreos para todos los modelos, fracciones y partes de cada referencia). En los modelos"
                  " con 3 semillas se promedia la metrica de las semillas dentro de cada remuestreo, como en la fase F. Medida principal"
                  " (la de la fase F).")
    lineas.append("")
    lineas.append("- Caida relativa = (F1 con todos los fotogramas - F1 con fotogramas quitados) / F1 con todos; positiva = el modelo empeora.")
    lineas.append("- Diferencia de caidas relativas = caida de la GRU (o la LSTM) menos caida de la CfC; positiva = la CfC cae menos.")
    lineas.append("- Significativa = el intervalo del 95 % no incluye el cero.")
    lineas.append("")
    for parte, nombre_parte in (("parte1", "Parte 1: modelos congelados"), ("parte2", "Parte 2: modelos reentrenados con fotogramas quitados")):
        modelos = TODOS if parte == "parte1" else TEMPORALES
        lineas.append(f"## {nombre_parte}")
        lineas.append("")
        lineas.append("### Caida de F1 macro respecto a la prueba completa")
        lineas.append("")
        lineas.append("| referencia | modelo | quitado | F1 macro | caida absoluta | IC 95 % | caida relativa | IC 95 % |")
        lineas.append("|---|---|---|---|---|---|---|---|")
        for clave in refs:
            for m in modelos:
                for fr in fracciones[1:]:
                    pq = puntos[(clave, parte, m, fr)]
                    d_p = pq["f1"] - puntos[(clave, parte, m, "completa")]["f1"]
                    d_s = muestras[(clave, parte, m, fr)]["f1"] - muestras[(clave, parte, m, "completa")]["f1"]
                    r_p, r_s = caida_rel(clave, parte, m, fr)
                    lineas.append(f"| {nombre_ref[clave]} | {titulo(m)} | {PORCENTAJE[fr]} % | {pq['f1']:.3f} | {fila_ic(d_p, d_s)} |"
                                  f" {100 * r_p:+.1f} % | {100 * ic(r_s)[0]:+.1f} a {100 * ic(r_s)[1]:+.1f} % |")
                    salida_json["pruebas"].append({"parte": parte, "referencia": clave, "prueba": "caida", "modelo": m, "fraccion": fr,
                                                   "metrica": "f1", "punto": d_p, "ic": ic(d_s), "caida_relativa": r_p, "ic_relativa": ic(r_s)})
        lineas.append("")
        lineas.append("### Diferencia de caidas relativas contra la CfC (F1 macro y kappa)")
        lineas.append("")
        lineas.append("| referencia | comparacion | quitado | F1: diferencia | IC 95 % | remuestreos con la CfC cayendo menos | kappa: diferencia | IC 95 % |")
        lineas.append("|---|---|---|---|---|---|---|---|")
        for clave in refs:
            for otro in ("gru", "lstm"):
                for fr in fracciones[1:]:
                    rc_p, rc_s = caida_rel(clave, parte, "cfc", fr)
                    ro_p, ro_s = caida_rel(clave, parte, otro, fr)
                    kc_p, kc_s = caida_rel(clave, parte, "cfc", fr, "kappa")
                    ko_p, ko_s = caida_rel(clave, parte, otro, fr, "kappa")
                    lineas.append(f"| {nombre_ref[clave]} | {titulo(otro)} - CfC | {PORCENTAJE[fr]} % | {100 * (ro_p - rc_p):+.1f} pp |"
                                  f" {100 * ic(ro_s - rc_s)[0]:+.1f} a {100 * ic(ro_s - rc_s)[1]:+.1f} pp | {100 * ((ro_s - rc_s) > 0).mean():.1f} % |"
                                  f" {100 * (ko_p - kc_p):+.1f} pp | {100 * ic(ko_s - kc_s)[0]:+.1f} a {100 * ic(ko_s - kc_s)[1]:+.1f} pp |")
                    for k, pp, ss in (("f1", ro_p - rc_p, ro_s - rc_s), ("kappa", ko_p - kc_p, ko_s - kc_s)):
                        salida_json["pruebas"].append({"parte": parte, "referencia": clave, "prueba": "diferencia_caida_relativa",
                                                       "comparacion": f"{otro} - cfc", "fraccion": fr, "metrica": k, "punto": pp, "ic": ic(ss),
                                                       "fraccion_positiva": float((ss > 0).mean())})
        lineas.append("")
    lineas.append("## Parte 2 contra parte 1: reentrenado menos congelado, en cada fraccion")
    lineas.append("")
    lineas.append("| referencia | modelo | quitado | F1 congelado | F1 reentrenado | diferencia | IC 95 % | kappa: diferencia | IC 95 % |")
    lineas.append("|---|---|---|---|---|---|---|---|---|")
    for clave in refs:
        for m in TEMPORALES:
            for fr in fracciones:
                p1, p2 = puntos[(clave, "parte1", m, fr)], puntos[(clave, "parte2", m, fr)]
                s1, s2 = muestras[(clave, "parte1", m, fr)], muestras[(clave, "parte2", m, fr)]
                lineas.append(f"| {nombre_ref[clave]} | {titulo(m)} | {PORCENTAJE[fr]} % | {p1['f1']:.3f} | {p2['f1']:.3f} |"
                              f" {fila_ic(p2['f1'] - p1['f1'], s2['f1'] - s1['f1'])} | {fila_ic(p2['kappa'] - p1['kappa'], s2['kappa'] - s1['kappa'])} |")
                for k in ("f1", "kappa"):
                    salida_json["pruebas"].append({"referencia": clave, "prueba": "reentrenado_menos_congelado", "modelo": m, "fraccion": fr,
                                                   "metrica": k, "punto": p2[k] - p1[k], "ic": ic(s2[k] - s1[k])})
    lineas.append("")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_2,
                         "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.titlecolor": TINTA})
    fig, ejes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True, facecolor=FONDO)
    xs = [PORCENTAJE[f] for f in fracciones]
    for fila, clave in enumerate(refs):
        valores = [puntos[(clave, parte, m, fr)]["f1"] for parte in ("parte1", "parte2")
                   for m in (TODOS if parte == "parte1" else TEMPORALES) for fr in fracciones]
        lo, hi = np.floor(min(valores) * 50 - 1) / 50, np.ceil(max(valores) * 50 + 1) / 50
        for col, parte in enumerate(("parte1", "parte2")):
            eje = ejes[fila, col]
            eje.set_facecolor(FONDO)
            for lado in ("top", "right"):
                eje.spines[lado].set_visible(False)
            eje.grid(axis="y", color=REJILLA, linewidth=0.8)
            eje.set_axisbelow(True)
            modelos = TODOS if parte == "parte1" else TEMPORALES
            if parte == "parte2":
                for m in TEMPORALES:
                    ys = [puntos[(clave, "parte1", m, fr)]["f1"] for fr in fracciones]
                    eje.plot(xs, ys, color=COLOR[m], linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.55, zorder=2)
            for m in modelos:
                ys = np.array([puntos[(clave, parte, m, fr)]["f1"] for fr in fracciones])
                sd = np.array([puntos[(clave, parte, m, fr)]["f1_sd"] for fr in fracciones])
                if puntos[(clave, parte, m, "completa")]["n"] > 1:
                    eje.fill_between(xs, ys - sd, ys + sd, color=COLOR[m], alpha=0.13, linewidth=0, zorder=1)
                eje.plot(xs, ys, color=COLOR[m], linewidth=2, marker=MARCA[m], markersize=6, markeredgecolor=FONDO, markeredgewidth=1.2, zorder=3)
            eje.set_ylim(lo, hi)
            eje.set_xticks(xs)
            eje.set_xticklabels([f"{x} %" for x in xs])
            titulo_col = "Parte 1: modelos congelados" if parte == "parte1" else "Parte 2: reentrenados con fotogramas quitados"
            eje.set_title(f"{titulo_col}\nreferencia {'de la IA' if clave == 'ia' else 'humana'}", fontsize=9.5, loc="left", color=TINTA)
            if col == 0:
                eje.set_ylabel("F1 macro (nivel por segundo)")
            if fila == 1:
                eje.set_xlabel("fotogramas quitados")
    from matplotlib.lines import Line2D
    leyenda = [Line2D([0], [0], color=COLOR[m], linewidth=2, marker=MARCA[m], markersize=6, markeredgecolor=FONDO, label=titulo(m)) for m in TODOS]
    leyenda.append(Line2D([0], [0], color=TINTA_2, linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.7, label="congelado (en la parte 2)"))
    fig.legend(handles=leyenda, loc="upper center", ncol=6, frameon=False, fontsize=9, labelcolor=TINTA, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, "Banda: desviacion entre 3 semillas (CfC, GRU, LSTM). Prueba: 70 videos con referencia de la IA y 69 con referencia humana.",
             fontsize=8, color=TINTA_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    ruta_figura = os.path.join(a.salida, "f1_vs_quitados.png")
    fig.savefig(ruta_figura, dpi=200, facecolor=FONDO)
    lineas.append(f"Figura: `{ruta_figura.replace(os.sep, '/')}`.")
    lineas.append("")
    with open(os.path.join(a.salida, "analisis.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    with open(os.path.join(a.salida, "analisis.json"), "w", encoding="utf-8") as f:
        json.dump(salida_json, f, indent=1)
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
