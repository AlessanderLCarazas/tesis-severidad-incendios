import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Circle
import evaluacion_cap5 as ev
from variantes_cap5 import VARIANTES, NOMBRE_VARIANTE, ABLACIONES_EXTRA

FIGURAS = ["pipeline_100", "map_tiou", "adelanto_alerta", "f1_vs_quitados", "confiabilidad_gru", "ablaciones"]
TINTA, TINTA2, BORDE, REJILLA = "#0b0b0b", "#52514e", "#c3c2b7", "#e4e3df"
CABECERA, FONDO_CAJA = "#2f3b48", "#ffffff"


def coma(eje, ejes="xy", formato="{:g}"):
    f = matplotlib.ticker.FuncFormatter(lambda v, _: formato.format(v).replace(".", ","))
    if "x" in ejes:
        eje.xaxis.set_major_formatter(f)
    if "y" in ejes:
        eje.yaxis.set_major_formatter(f)


def leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def estilo():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": BORDE, "axes.labelcolor": TINTA2,
                         "xtick.color": TINTA2, "ytick.color": TINTA2, "mathtext.fontset": "dejavusans"})


def limpiar(eje, rejilla="y"):
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    eje.grid(axis=rejilla, color=REJILLA, linewidth=0.8)
    eje.set_axisbelow(True)


def guardar(fig, nombre, a, pdf=True):
    rutas = [os.path.join(a.salida, f"{nombre}.png")]
    fig.savefig(rutas[0], dpi=a.dpi, facecolor="white")
    if pdf:
        rutas.append(os.path.join(a.tesis, f"{nombre}.pdf"))
        fig.savefig(rutas[1], facecolor="white")
    plt.close(fig)
    for r in rutas:
        print(f"figura: {r}")


def por_modelo(a):
    datos = leer(a.bloque2)["corridas"]
    base = {"cfc": "cfc_s", "gru": "gru_s", "lstm": "lstm_s", "mamba": "mamba_s"}
    salida = {}
    for m in ev.ORDEN:
        nombres = [n for n in datos if n.startswith(base[m]) and "__" not in n] if m in base else [n for n in datos if n == m]
        if nombres:
            salida[m] = [datos[n][a.referencia] for n in nombres]
    print("corridas por modelo: " + ", ".join(f"{m} {len(r)}" for m, r in salida.items()))
    return salida


def leyenda_modelos(modelos):
    return [Line2D([0], [0], color=ev.COLOR[m], linewidth=2, marker=ev.MARCA[m], markersize=6, markeredgecolor="white", label=ev.NOMBRE[m])
            for m in modelos]


def map_tiou(a, pm):
    estilo()
    modelos = [m for m in ev.ORDEN if m in pm and m != "clase mayoritaria"]
    xs = [float(u) for u in ev.TIOU]
    fig, eje = plt.subplots(figsize=(8.6, 4.2))
    limpiar(eje)
    if "clase mayoritaria" in pm:
        ys = [np.mean([r["map"][u]["promedio"] for r in pm["clase mayoritaria"]]) for u in ev.TIOU]
        eje.plot(xs, ys, color="#8a8984", linewidth=1.3, linestyle=(0, (4, 3)))
    for m in modelos:
        vals = np.array([[r["map"][u]["promedio"] for u in ev.TIOU] for r in pm[m]])
        if len(vals) > 1:
            eje.errorbar(xs, vals.mean(0), yerr=vals.std(0), color=ev.COLOR[m], linewidth=0, elinewidth=1.1, capsize=3)
        eje.plot(xs, vals.mean(0), color=ev.COLOR[m], linewidth=2, marker=ev.MARCA[m], markersize=6, markeredgecolor="white", markeredgewidth=1)
        print(f"map {m}: " + ", ".join(f"{u} {v:.3f}" for u, v in zip(ev.TIOU, vals.mean(0))))
    eje.set_xticks(xs)
    eje.set_xlabel("umbral de tIoU")
    eje.set_ylabel("mAP de segmentos")
    coma(eje)
    leyenda = leyenda_modelos(modelos)
    leyenda.append(Line2D([0], [0], color="#8a8984", linewidth=1.3, linestyle=(0, (4, 3)), label="clase mayoritaria"))
    fig.legend(handles=leyenda, loc="upper center", ncol=4, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    guardar(fig, "map_tiou", a)


def adelanto_alerta(a, pm):
    estilo()
    modelos = [m for m in ev.ORDEN if m in pm and m != "clase mayoritaria"]
    fig, eje = plt.subplots(figsize=(9.6, 4.4))
    rng = np.random.default_rng(0)
    limpiar(eje, "x")
    eje.axvspan(0, 200, color="#eaf2fc", zorder=0)
    eje.axvline(0, color=TINTA2, linewidth=1)
    for i, m in enumerate(modelos):
        y = len(modelos) - 1 - i
        ewt = np.concatenate([r["alerta"]["0.1"]["ewt"] for r in pm[m]]) if pm[m] else np.zeros(0)
        sin = np.mean([r["alerta"]["0.1"]["sin_alerta"] for r in pm[m]])
        eje.scatter(np.clip(ewt, -195, 195), y + rng.uniform(-0.22, 0.22, len(ewt)), s=14, color=ev.COLOR[m], alpha=0.75, edgecolor="none", zorder=3)
        if len(ewt):
            eje.plot([np.median(ewt)] * 2, [y - 0.33, y + 0.33], color=TINTA, linewidth=2.2, zorder=4)
            print(f"adelanto {m}: mediana {np.median(ewt):.1f} s, sin alerta {sin:.1f}")
        eje.text(1.01, (y + 0.6) / (len(modelos) + 0.3), f"{sin:.1f}".replace(".0", "").replace(".", ","), transform=eje.transAxes,
                 va="center", fontsize=8.5, color=TINTA2)
    eje.set_xlim(-200, 200)
    eje.set_xlabel("adelanto de la alerta respecto al cambio a incontrolado (s); a la derecha del 0, la alerta llega antes")
    eje.text(1.01, 1.01, "sin\nalerta", transform=eje.transAxes, va="bottom", fontsize=8, color=TINTA2)
    eje.set_yticks(range(len(modelos)))
    eje.set_yticklabels([ev.NOMBRE[m] for m in reversed(modelos)])
    eje.set_ylim(-0.6, len(modelos) - 0.4)
    coma(eje, "x")
    fig.tight_layout()
    guardar(fig, "adelanto_alerta", a)


def f1_vs_quitados(a):
    estilo()
    puntos = leer(os.path.join(os.path.dirname(a.robustos), "analisis.json"))["puntos"]
    fracciones = [("completa", 0), ("quitar_25", 25), ("quitar_50", 50), ("quitar_75", 75)]
    rob = ["cfc", "gru", "lstm", "gradient boosting", "regresion logistica"]
    clave = a.referencia
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    x = [p for _, p in fracciones]
    vals = [puntos[f"{clave}|{pa}|{m}|{fr}"]["f1"] for pa, ms in (("parte1", rob), ("parte2", rob[:3])) for m in ms for fr, _ in fracciones]
    lo, hi = np.floor(min(vals) * 50 - 1) / 50, np.ceil(max(vals) * 50 + 1) / 50
    for eje, parte in zip(ejes, ("parte1", "parte2")):
        limpiar(eje)
        ms = rob if parte == "parte1" else rob[:3]
        if parte == "parte2":
            for m in ms:
                eje.plot(x, [puntos[f"{clave}|parte1|{m}|{fr}"]["f1"] for fr, _ in fracciones], color=ev.COLOR[m], linewidth=1.2,
                         linestyle=(0, (4, 3)), alpha=0.55)
        for m in ms:
            ys = np.array([puntos[f"{clave}|{parte}|{m}|{fr}"]["f1"] for fr, _ in fracciones])
            sd = np.array([puntos[f"{clave}|{parte}|{m}|{fr}"]["f1_sd"] for fr, _ in fracciones])
            if puntos[f"{clave}|{parte}|{m}|completa"]["n"] > 1:
                eje.fill_between(x, ys - sd, ys + sd, color=ev.COLOR[m], alpha=0.13, linewidth=0)
            eje.plot(x, ys, color=ev.COLOR[m], linewidth=2, marker=ev.MARCA[m], markersize=6, markeredgecolor="white", markeredgewidth=1)
            print(f"robustez {parte} {m}: " + ", ".join(f"{v:.3f}" for v in ys))
        eje.set_ylim(lo, hi)
        eje.set_xticks(x)
        eje.set_xticklabels([f"{p} %" for p in x])
        eje.set_xlabel("fotogramas quitados")
        eje.set_title("Congelados" if parte == "parte1" else "Reentrenados con fotogramas quitados", fontsize=9.5, loc="left", color=TINTA)
        coma(eje, "y")
    ejes[0].set_ylabel("F1 macro")
    leyenda = leyenda_modelos(rob)
    leyenda.append(Line2D([0], [0], color=TINTA2, linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.7, label="congelado (a la derecha)"))
    fig.legend(handles=leyenda, loc="upper center", ncol=6, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.5)
    guardar(fig, "f1_vs_quitados", a)


def confiabilidad_gru(a):
    b4 = leer(a.sistema)["b4"]
    corridas = [n for n in b4 if n.startswith("gru_s")]
    estilo()
    fig, eje = plt.subplots(figsize=(8, 4.4))
    bins = []
    for b in range(10):
        filas = [b4[n][a.referencia]["bins"][b] for n in corridas]
        n = sum(f["n"] for f in filas)
        conf = sum(f["n"] * f["confianza"] for f in filas if f["n"]) / n if n else None
        acc = sum(f["n"] * f["exactitud"] for f in filas if f["n"]) / n if n else None
        bins.append({"n": n, "confianza": conf, "exactitud": acc})
    total = sum(b["n"] for b in bins)
    ece = sum(b["n"] / total * abs(b["exactitud"] - b["confianza"]) for b in bins if b["n"])
    alto = bins[9]
    print(f"confiabilidad gru: {len(corridas)} corridas, {total} segundos, ece {ece:.3f}, confianza > 0.9: {alto['n']} segundos con exactitud {alto['exactitud']:.3f}")
    centros = np.arange(0.05, 1.0, 0.1)
    eje.bar(centros, [b["exactitud"] if b["exactitud"] is not None else 0.0 for b in bins], width=0.09, color="#2a78d6", label="exactitud")
    eje.plot([0, 1], [0, 1], color=TINTA2, linewidth=1, linestyle=(0, (4, 3)), label="calibración perfecta")
    for x, b in zip(centros, bins):
        if b["confianza"] is not None:
            eje.plot([x - 0.045, x + 0.045], [b["confianza"]] * 2, color="#eb6834", linewidth=2)
        if b["n"]:
            eje.text(x, 1.02, f"{b['n']}", ha="center", fontsize=7, color=TINTA2)
    eje.plot([], [], color="#eb6834", linewidth=2, label="confianza media")
    eje.set_xlim(0, 1)
    eje.set_ylim(0, 1.08)
    eje.set_xlabel("confianza (probabilidad del nivel más probable)")
    eje.set_ylabel("exactitud en el intervalo")
    eje.set_title(f"ECE {ece:.3f}".replace(".", ","), fontsize=10, loc="left")
    limpiar(eje)
    coma(eje, formato="{:.1f}")
    eje.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=8)
    fig.text(0.01, 0.01, f"GRU, {len(corridas)} semillas juntas. Número sobre cada barra: segundos en el intervalo.", fontsize=8, color=TINTA2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    guardar(fig, "confiabilidad_gru", a)


def etiqueta_variante(v):
    s = NOMBRE_VARIANTE[v]
    return s if s[3:6] == "dt " else s[:3] + s[3].upper() + s[4:]


def ablaciones(a):
    pruebas = [x for x in leer(a.bloque2)["pruebas"] if x["bloque"] == "ablaciones" and x["referencia"] == a.referencia]
    orden = [v for v in VARIANTES[1:] + ABLACIONES_EXTRA if any(x["variante"] == v for x in pruebas)]
    estilo()
    fig, eje = plt.subplots(figsize=(9.2, 5.6))
    modelos = [("gru", "GRU", "#eb6834", "s", 0.17), ("cfc", "CfC", "#2a78d6", "o", -0.17)]
    xs = [v for x in pruebas for v in x["ic"]]
    lim = np.ceil(max(abs(min(xs)), abs(max(xs))) * 50 + 0.5) / 50
    eje.axvline(0, color=TINTA2, linewidth=1, zorder=1)
    for i in range(len(orden)):
        if i % 2 == 0:
            eje.axhspan(i - 0.5, i + 0.5, color="#f4f4f1", zorder=0, linewidth=0)
    for m, nombre, color, marca, dy in modelos:
        for i, v in enumerate(orden):
            y = len(orden) - 1 - i + dy
            x = next(p for p in pruebas if p["variante"] == v and p["modelo"] == m)
            eje.plot(x["ic"], [y, y], color=color, linewidth=1.6, solid_capstyle="round", zorder=2)
            relleno = color if x["significativa_sin_correccion"] else "white"
            eje.plot([x["punto"]], [y], marker=marca, markersize=7, color=color, markerfacecolor=relleno, markeredgewidth=1.6, zorder=3)
    eje.set_xlim(-lim, lim)
    eje.set_ylim(-0.6, len(orden) - 0.4)
    eje.grid(axis="x", color=REJILLA, linewidth=0.8)
    eje.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        eje.spines[lado].set_visible(False)
    eje.tick_params(axis="y", length=0)
    eje.set_xlabel("diferencia de F1 macro (variante − sistema completo)")
    eje.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.2f}".replace(".", ",")))
    eje.set_yticks(range(len(orden)))
    eje.set_yticklabels([etiqueta_variante(v) for v in reversed(orden)])
    leyenda = [Line2D([0], [0], color=c, linewidth=1.6, marker=mk, markersize=7, label=f"{n} (media de 3 semillas, IC 95 %)") for _, n, c, mk, _ in modelos]
    leyenda.append(Line2D([0], [0], color=TINTA2, linewidth=0, marker="o", markersize=7, markerfacecolor=TINTA2, label="IC sin el cero (sin corregir)"))
    leyenda.append(Line2D([0], [0], color=TINTA2, linewidth=0, marker="o", markersize=7, markerfacecolor="white", label="IC con el cero"))
    fig.legend(handles=leyenda, loc="upper center", ncol=2, frameon=False, fontsize=8.4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    n_sig = sum(x["significativa_sin_correccion"] for x in pruebas)
    n_holm = sum(x["significativa_holm"] for x in pruebas)
    print(f"ablaciones: {len(orden)} variantes, {len(pruebas)} diferencias, {sum(x['punto'] > 0 for x in pruebas)} positivas, "
          f"{n_sig} sin el cero sin corregir, {n_holm} significativas con holm")
    guardar(fig, "ablaciones", a)


ETAPAS = [
    ("Entrada", ["Video con marcas de tiempo", r"$(I_k,\ t_k)$", "Cualquier fps y resolución", "(DetectiumFire: 24 a 60 fps,", "hasta 4K)"], "E1 · E2"),
    ("Preprocesamiento", [r"$t_k$ real del archivo", r"$\Delta t_k = t_k - t_{k-1}$", "Muestreo a unos 10 fps", "conservando la marca real", "Letterbox: lado mayor de 640,", "relleno mínimo (640 × 384 en 16:9)", "píxeles en [0, 1]"], ""),
    ("Detección y seguimiento", ["YOLOv8s: fuego y humo", "ByteTrack con Δt real en el", "filtro de Kalman (tope de 2)", "Compensación de la cámara", "Cortes de escena"], "E3 · E4 · E6"),
    ("Representación", ["Cámara: Shi-Tomasi, Lucas-Kanade", "y RANSAC → s, rot, tx, ty, inliers", "13 descriptores por trayectoria", "de fuego (d1 a d13)", "Vector de escena de 34 valores:", "máximo y media de d1 a d13,", "trayectorias y detecciones"], "E5 · E7"),
    ("Procesamiento temporal", ["Compuerta de contexto", r"$\tilde{v} = v \odot \sigma(W c + b)$", r"$c = [s,\ rot,\ t_x,\ t_y,\ inliers,\ \Delta t]$", "Modelo temporal:"], "E9"),
    ("Estimación de riesgo", ["Nivel ordinal (CORAL):", "controlado, incipiente o", "incontrolado", "Riesgo continuo en [0, 1]", r"Tiempo al peligro: $\log(1+\tau)$", "Ponderación de Kendall"], "E9"),
    ("Calibración y decisión", ["Alerta con predicción conformal", "por video (α = 0,10)", "Desactivación bajo 0,8 × umbral", "Histéresis: al menos 2 s", "en cada nivel"], "E8"),
    ("Salida", ["Nivel, alerta, tiempo al", "peligro y caja de la región", "Segmentos", r"$(t_s,\ t_e,\ nivel)$", "15,7 fps en vivo"], ""),
]
MODELOS_CHIP = [("CfC", "#2a78d6"), ("GRU", "#eb6834"), ("LSTM", "#1baf7a"), ("Mamba", "#6250d6")]


def pipeline(a):
    estilo()
    fig = plt.figure(figsize=(13, 7.0))
    eje = fig.add_axes((0, 0, 1, 1))
    eje.set_xlim(0, 13)
    eje.set_ylim(0, 7.0)
    eje.axis("off")
    ancho, alto, cab = 2.9, 2.85, 0.62
    xs = [0.25 + 3.23 * i for i in range(4)]
    ys = [3.95, 0.75]
    cajas = []
    for n, (nombre, lineas, exp) in enumerate(ETAPAS):
        x0, y0 = xs[n % 4], ys[n // 4]
        cajas.append((x0, y0))
        eje.add_patch(FancyBboxPatch((x0, y0), ancho, alto, boxstyle="round,pad=0,rounding_size=0.12", facecolor=FONDO_CAJA,
                                     edgecolor=BORDE, linewidth=1.2))
        eje.add_patch(FancyBboxPatch((x0, y0 + alto - cab), ancho, cab, boxstyle="round,pad=0,rounding_size=0.12", facecolor=CABECERA,
                                     edgecolor=CABECERA, linewidth=1.2))
        eje.add_patch(plt.Rectangle((x0, y0 + alto - cab), ancho, 0.15, facecolor=CABECERA, edgecolor="none"))
        eje.add_patch(Circle((x0 + 0.36, y0 + alto - cab / 2), 0.21, facecolor="white", edgecolor="none"))
        eje.text(x0 + 0.36, y0 + alto - cab / 2, str(n + 1), ha="center", va="center", fontsize=11, fontweight="bold", color=CABECERA)
        eje.text(x0 + 0.68, y0 + alto - cab / 2, nombre, ha="left", va="center", fontsize=11, fontweight="bold", color="white")
        yy = y0 + alto - cab - 0.27
        for linea in lineas:
            matematica = linea.startswith("$")
            eje.text(x0 + 0.18, yy, linea, ha="left", va="center", fontsize=10.5 if matematica else 9.6,
                     color=TINTA if not matematica else CABECERA)
            yy -= 0.27
        if n == 4:
            cx = x0 + 0.18
            for texto, color in MODELOS_CHIP:
                w = 0.13 + 0.115 * len(texto)
                eje.add_patch(FancyBboxPatch((cx, yy - 0.15), w, 0.3, boxstyle="round,pad=0,rounding_size=0.08", facecolor=color, edgecolor="none"))
                eje.text(cx + w / 2, yy, texto, ha="center", va="center", fontsize=9, fontweight="bold", color="white")
                cx += w + 0.08
            eje.text(x0 + 0.18, yy - 0.34, r"CfC con $timespans = \Delta t$", ha="left", va="center", fontsize=9.2, color=TINTA2)
        if exp:
            eje.text(x0 + ancho - 0.12, y0 + 0.14, exp, ha="right", va="bottom", fontsize=8.6, color="#2a78d6", fontweight="bold")
    flecha = dict(arrowstyle="-|>", color=TINTA2, linewidth=1.6, mutation_scale=14, shrinkA=0, shrinkB=0)
    for n in range(7):
        if n == 3:
            continue
        x0, y0 = cajas[n]
        x1, y1 = cajas[n + 1]
        eje.annotate("", xy=(x1 - 0.02, y1 + alto / 2), xytext=(x0 + ancho + 0.02, y0 + alto / 2), arrowprops=flecha)
    x4, y4 = cajas[3]
    x5, y5 = cajas[4]
    ymed = (y4 + y5 + alto) / 2
    eje.plot([x4 + ancho / 2, x4 + ancho / 2, x5 + ancho / 2], [y4 - 0.02, ymed, ymed], color=TINTA2, linewidth=1.6, solid_capstyle="round")
    eje.annotate("", xy=(x5 + ancho / 2, y5 + alto + 0.02), xytext=(x5 + ancho / 2, ymed), arrowprops=flecha)
    eje.add_patch(FancyBboxPatch((0.25, 0.12), 12.59, 0.45, boxstyle="round,pad=0,rounding_size=0.1", facecolor="#eaf2fc", edgecolor="none"))
    eje.text(6.545, 0.345, "Sistema completo: E9 modelos · E10 ablaciones · E11 robustez · E12 eficiencia · E13 FIgLib        "
             "Datos: E1 partición · E2 etiquetado", ha="center", va="center", fontsize=9.6, color=CABECERA)
    eje.text(12.84, 6.88, "En azul, los experimentos que evalúan cada etapa", ha="right", va="center", fontsize=8.6, color="#2a78d6")
    guardar(fig, "pipeline_100", a, pdf=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figuras", nargs="+", choices=FIGURAS, default=FIGURAS)
    p.add_argument("--referencia", default="ia")
    p.add_argument("--bloque2", default=os.path.join("resultados", "cap5", "bloque2.json"))
    p.add_argument("--sistema", default=os.path.join("resultados", "cap5", "sistema.json"))
    p.add_argument("--robustos", default=os.path.join("resultados", "robustez", "modelos_robustos.json"))
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--salida", default=os.path.join("docs", "img"))
    p.add_argument("--tesis", default=os.path.join("resultados", "figuras_tesis"))
    a = p.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    os.makedirs(a.tesis, exist_ok=True)
    pm = por_modelo(a) if {"map_tiou", "adelanto_alerta"} & set(a.figuras) else None
    if "map_tiou" in a.figuras:
        map_tiou(a, pm)
    if "adelanto_alerta" in a.figuras:
        adelanto_alerta(a, pm)
    if "f1_vs_quitados" in a.figuras:
        f1_vs_quitados(a)
    if "confiabilidad_gru" in a.figuras:
        confiabilidad_gru(a)
    if "ablaciones" in a.figuras:
        ablaciones(a)
    if "pipeline_100" in a.figuras:
        pipeline(a)


if __name__ == "__main__":
    main()
