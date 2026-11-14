import os
import re
import csv
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

TINTA, TINTA_2, TINTA_3, REJILLA, FONDO = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
COLOR = {"cfc": "#2a78d6", "gru": "#eb6834", "lstm": "#1baf7a", "gradient boosting": "#eda100", "regresion logistica": "#e87ba4"}
MARCA = {"cfc": "o", "gru": "s", "lstm": "^", "gradient boosting": "D", "regresion logistica": "v"}
NOMBRE = {"cfc": "CfC", "gru": "GRU", "lstm": "LSTM", "gradient boosting": "gradient boosting", "regresion logistica": "regresión logística",
          "clase mayoritaria": "clase mayoritaria"}
MODELOS = ["cfc", "gru", "lstm", "gradient boosting", "regresion logistica"]
NIVELES = ["controlado", "incipiente", "incontrolado"]
REFERENCIA = {"ia": "referencia de la IA", "humano": "referencia humana"}
AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DESCRIPTORES = ["confianza", "área", "tasa de crecimiento", "crecimiento relativo", "centro x", "centro y", "velocidad del centro",
                "persistencia", "variación de forma", "confianza de humo", "área de humo", "tasa del humo", "regiones activas"]


def mayuscula(texto):
    return texto[:1].upper() + texto[1:]


def estilo():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_2,
                         "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.titlecolor": TINTA, "figure.facecolor": FONDO,
                         "axes.facecolor": FONDO, "savefig.facecolor": FONDO})


def limpiar(eje, rejilla="y"):
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    if rejilla:
        eje.grid(axis=rejilla, color=REJILLA, linewidth=0.8)
    eje.set_axisbelow(True)


def guardar(fig, ruta, dpi=150):
    fig.savefig(ruta, dpi=dpi)
    plt.close(fig)
    print(f"{ruta} | {os.path.getsize(ruta) / 1e3:.0f} kb", flush=True)


def leer_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def leer_csv(ruta):
    with open(ruta, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def figura_descriptores(a):
    df = pd.read_csv(a.descriptores_v2, dtype={"archivo": str}, keep_default_na=False, na_values=[])
    sel = df[(df["archivo"] == a.video_descriptores) & (df["id"] == 1)].sort_values("t")
    fig, ejes = plt.subplots(5, 3, figsize=(11, 9), sharex=True)
    for i, eje in enumerate(ejes.flat):
        if i >= 13:
            eje.axis("off")
            continue
        limpiar(eje)
        eje.plot(sel["t"], sel[f"d{i + 1}"], color=COLOR["cfc"], linewidth=1.3)
        eje.set_title(f"d{i + 1} · {DESCRIPTORES[i]}", fontsize=9.5, loc="left")
        eje.tick_params(labelsize=8)
        if i >= 10:
            eje.set_xlabel("t (s)")
            eje.tick_params(labelbottom=True)
    fig.tight_layout()
    guardar(fig, os.path.join(a.salida, "descriptores_v2.png"), 130)


def figura_instantes(a):
    metas = leer_csv(a.metas)
    humano = {r["archivo"]: r for r in metas if r["fuente"] == "humano" and r["version"] == "aleCa_cambios" and r["descartar"] != "1"}
    ia = {r["archivo"]: r for r in metas if r["fuente"] == "ia" and r["archivo"] in humano}
    revision = {r["orden"]: r["archivo"] for r in leer_csv(a.revision)}
    veredicto = {revision[n["n"]]: n["causa"] for n in leer_csv(a.notas)}
    grupos = {"B": ("tenía razón la IA", COLOR["cfc"]), "A": ("tenía razón el humano", COLOR["gru"]), "-": ("sin diferencia que revisar", TINTA_3)}
    tipo = {"t_incipiente": ("^", "cambio a incipiente"), "t_incontrolado": ("o", "cambio a incontrolado")}
    ambos, solo_ia, solo_humano = [], [], []
    for v, h in humano.items():
        for k in tipo:
            x, y = h[k], ia[v][k]
            g = veredicto.get(v, "-")
            if x and y:
                ambos.append((float(x), float(y), k, g))
            elif y:
                solo_ia.append((float(y), k, g))
            elif x:
                solo_humano.append((float(x), k, g))
    tope = max(max(x, y) for x, y, _, _ in ambos + [(0, t, 0, 0) for t, _, _ in solo_ia]) * 1.06
    fig = plt.figure(figsize=(8.4, 7))
    malla = fig.add_gridspec(2, 2, width_ratios=[1, 9], height_ratios=[9, 1], wspace=0.05, hspace=0.08)
    eje = fig.add_subplot(malla[0, 1])
    izq = fig.add_subplot(malla[0, 0], sharey=eje)
    abajo = fig.add_subplot(malla[1, 1], sharex=eje)
    limpiar(eje, "both")
    eje.plot([0, tope], [0, tope], color=TINTA_3, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    eje.text(tope * 0.97, tope * 0.93, "mismo instante", color=TINTA_2, fontsize=8.5, ha="right", rotation=0)
    eje.text(tope * 0.45, tope * 0.03, "debajo de la diagonal: la IA marca antes", color=TINTA_2, fontsize=8.5, ha="left")
    for x, y, k, g in ambos:
        eje.scatter(x, y, marker=tipo[k][0], s=60, color=grupos[g][1], edgecolor=FONDO, linewidth=1.2, zorder=3)
    for y, k, g in solo_ia:
        izq.scatter(0.5 + np.random.default_rng(int(y * 100)).uniform(-0.25, 0.25), y, marker=tipo[k][0], s=50, color=grupos[g][1],
                    edgecolor=FONDO, linewidth=1.2, zorder=3)
    for x, k, g in solo_humano:
        abajo.scatter(x, 0.5, marker=tipo[k][0], s=50, color=grupos[g][1], edgecolor=FONDO, linewidth=1.2, zorder=3)
    eje.set_xlim(0, tope)
    eje.set_ylim(0, tope)
    eje.tick_params(labelleft=False, labelbottom=False)
    izq.set_xlim(0, 1)
    izq.set_xticks([])
    izq.set_title("el humano\nno lo marca", fontsize=8, color=TINTA_2)
    izq.set_ylabel("instante marcado por la IA (s)")
    abajo.set_ylim(0, 1)
    abajo.set_yticks([0.5])
    abajo.set_yticklabels(["la IA no\nlo marca"], fontsize=8)
    abajo.set_xlabel("instante marcado por el humano (s)")
    for e in (izq, abajo):
        for lado in ("top", "right"):
            e.spines[lado].set_visible(False)
        e.set_facecolor("#f4f3f0")
    leyenda = [Line2D([0], [0], marker="s", linestyle="", color=c, markersize=8, label=f"{t} ({sum(1 for x in veredicto.values() if x == g) if g != '-' else len(humano) - len(veredicto)} videos)")
               for g, (t, c) in grupos.items()]
    leyenda += [Line2D([0], [0], marker=m, linestyle="", color=TINTA_2, markersize=7, label=t) for m, t in tipo.values()]
    eje.legend(handles=leyenda, loc="upper left", frameon=False, fontsize=8.5, labelcolor=TINTA)
    fig.suptitle(f"Instantes de cambio en los {len(humano)} videos de prueba etiquetados a ciegas: humano contra IA", x=0.12, ha="left", fontsize=11, color=TINTA)
    guardar(fig, os.path.join(a.salida, "instantes_humano_ia.png"))


def historiales(carpeta, modelos):
    salida = {}
    for m in modelos:
        salida[m] = pd.read_csv(os.path.join(carpeta, m, "historial.csv"))
    return salida


def figura_curvas(a):
    primera = historiales(a.modelos_v1, ["cfc", "gru", "lstm"])
    final = {}
    for m, carpeta in (("cfc", a.final_a), ("gru", a.final_b), ("lstm", a.final_a)):
        final[m] = [pd.read_csv(os.path.join(carpeta, f"{m}_s{s}", "historial.csv")) for s in (42, 43, 44)]
    columnas = [("perdida_ent", "pérdida en entrenamiento"), ("perdida_val", "pérdida en validación"), ("f1_escena", "F1 macro de escena en validación")]
    fig, ejes = plt.subplots(2, 3, figsize=(12, 6.6))
    for j, (col, titulo) in enumerate(columnas):
        eje = ejes[0, j]
        limpiar(eje)
        for m in ("cfc", "gru", "lstm"):
            h = primera[m]
            eje.plot(h["epoca"], h[col], color=COLOR[m], linewidth=1.8)
        eje.set_title(titulo, fontsize=9.5, loc="left")
        eje = ejes[1, j]
        limpiar(eje)
        for m in ("cfc", "gru", "lstm"):
            valores = np.array([h[col].to_numpy() for h in final[m]])
            epocas = final[m][0]["epoca"].to_numpy()
            media, desv = valores.mean(0), valores.std(0)
            eje.fill_between(epocas, media - desv, media + desv, color=COLOR[m], alpha=0.15, linewidth=0)
            eje.plot(epocas, media, color=COLOR[m], linewidth=1.8)
        eje.set_xlabel("época")
    ejes[0, 0].set_ylabel("primera versión\n(por trayectoria, semilla 42)", color=TINTA)
    ejes[1, 0].set_ylabel("configuración final\n(por escena, 3 semillas)", color=TINTA)
    leyenda = [Line2D([0], [0], color=COLOR[m], linewidth=2, label=NOMBRE[m]) for m in ("cfc", "gru", "lstm")]
    fig.legend(handles=leyenda, loc="upper center", ncol=3, frameon=False, labelcolor=TINTA, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    guardar(fig, os.path.join(a.salida, "curvas_entrenamiento.png"))


def figura_confusion(a, metricas):
    mapa = LinearSegmentedColormap.from_list("azul", ["#f3f7fd"] + AZUL)
    fig, ejes = plt.subplots(1, 2, figsize=(9.6, 4))
    for eje, clave in zip(ejes, ("ia", "humano")):
        conf = np.mean([np.array(metricas["corridas"][f"gru_s{s}"]["referencias"][clave]["final"]["conf"], float) for s in (42, 43, 44)], axis=0)
        filas = conf / conf.sum(1, keepdims=True)
        eje.imshow(filas, cmap=mapa, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                eje.text(j, i, f"{100 * filas[i, j]:.0f} %\n{conf[i, j]:.0f} s", ha="center", va="center", fontsize=9,
                         color="#ffffff" if filas[i, j] > 0.45 else TINTA)
        eje.set_xticks(range(3))
        eje.set_xticklabels(NIVELES)
        eje.set_yticks(range(3))
        eje.set_yticklabels(NIVELES)
        eje.set_xlabel("nivel de la GRU")
        eje.set_ylabel(mayuscula(REFERENCIA[clave]))
        n = metricas["videos"]["fuego_ia" if clave == "ia" else "humano"]
        eje.set_title(f"{mayuscula(REFERENCIA[clave])} ({n} videos)", fontsize=10, loc="left")
        for lado in eje.spines.values():
            lado.set_visible(False)
        eje.tick_params(length=0)
    fig.tight_layout()
    guardar(fig, os.path.join(a.salida, "confusion_gru.png"))


def figura_map(a, metricas):
    fig, ejes = plt.subplots(1, 2, figsize=(9.6, 4), sharey=True)
    umbrales = ["0.3", "0.5", "0.7"]
    xs = [float(u) for u in umbrales]
    for eje, clave in zip(ejes, ("ia", "humano")):
        limpiar(eje)
        for m in MODELOS + ["clase mayoritaria"]:
            corr = [c for n, c in metricas["corridas"].items() if c["modelo"] == m]
            valores = np.array([[c["referencias"][clave]["map"][u]["promedio"] for u in umbrales] for c in corr])
            if m == "clase mayoritaria":
                eje.plot(xs, valores.mean(0), color=TINTA_3, linewidth=1.3, linestyle=(0, (4, 3)), zorder=2)
                continue
            if len(corr) > 1:
                eje.errorbar(xs, valores.mean(0), yerr=valores.std(0), color=COLOR[m], linewidth=0, elinewidth=1.2, capsize=3, zorder=2)
            eje.plot(xs, valores.mean(0), color=COLOR[m], linewidth=2, marker=MARCA[m], markersize=6, markeredgecolor=FONDO,
                     markeredgewidth=1.2, zorder=3)
        eje.set_xticks(xs)
        eje.set_xticklabels([f"tIoU {u}" for u in umbrales])
        n = metricas["videos"]["fuego_ia" if clave == "ia" else "humano_cambios"]
        eje.set_title(f"{mayuscula(REFERENCIA[clave])} ({n} videos)", fontsize=10, loc="left")
    ejes[0].set_ylabel("mAP de segmentos (media de los 3 niveles)")
    leyenda = [Line2D([0], [0], color=COLOR[m], linewidth=2, marker=MARCA[m], markersize=6, markeredgecolor=FONDO, label=NOMBRE[m]) for m in MODELOS]
    leyenda.append(Line2D([0], [0], color=TINTA_3, linewidth=1.3, linestyle=(0, (4, 3)), label="clase mayoritaria"))
    fig.legend(handles=leyenda, loc="upper center", ncol=6, frameon=False, fontsize=8.8, labelcolor=TINTA, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    guardar(fig, os.path.join(a.salida, "map_tiou.png"))


def figura_adelanto(a, metricas):
    filas = [r for r in leer_csv(a.alerta) if r["alfa"] == "0.1"]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for eje, clave in zip(ejes, ("ia", "humano")):
        limpiar(eje, "x")
        eje.axvspan(0, 190, color="#eaf2fc", zorder=0)
        eje.axvline(0, color=TINTA_2, linewidth=1)
        for i, m in enumerate(MODELOS):
            corr = [n for n, c in metricas["corridas"].items() if c["modelo"] == m]
            sel = [r for r in filas if r["referencia"] == clave and r["corrida"] in corr]
            ewt = np.array([float(r["ewt_s"]) for r in sel if r["ewt_s"]])
            sin = sum(1 for r in sel if not r["ewt_s"]) / len(corr)
            y = len(MODELOS) - 1 - i
            eje.scatter(np.clip(ewt, -175, 185), y + rng.uniform(-0.22, 0.22, len(ewt)), s=18, color=COLOR[m], alpha=0.75, edgecolor="none", zorder=3)
            mediana = np.median(ewt)
            eje.plot([mediana, mediana], [y - 0.33, y + 0.33], color=TINTA, linewidth=2.2, zorder=4)
            eje.text(1.02, (y + 0.6) / (len(MODELOS) + 0.3), f"{sin:.1f}".replace(".0", ""), transform=eje.transAxes, va="center", ha="left",
                     fontsize=8.5, color=TINTA_2)
        n = metricas["videos"]["fuego_ia" if clave == "ia" else "humano_cambios"]
        cambios = sum(1 for r in filas if r["referencia"] == clave and r["corrida"] == "gru_s42")
        eje.set_title(f"{mayuscula(REFERENCIA[clave])} ({cambios} videos con cambio a incontrolado)", fontsize=10, loc="left")
        eje.set_xlim(-180, 190)
        eje.set_xlabel("adelanto de la alerta (s)")
        eje.text(4, len(MODELOS) - 0.45, "alerta antes del cambio", fontsize=8, color=COLOR["cfc"], va="center")
        eje.text(1.02, 1.0, "sin\nalerta", transform=eje.transAxes, va="bottom", ha="left", fontsize=8, color=TINTA_2)
    ejes[0].set_yticks(range(len(MODELOS)))
    ejes[0].set_yticklabels([NOMBRE[m] for m in reversed(MODELOS)])
    ejes[0].set_ylim(-0.6, len(MODELOS) - 0.3)
    fig.text(0.01, 0.01, "Adelanto = instante del cambio a incontrolado menos instante de la primera alerta. Cada punto es un video (y una semilla en CfC, GRU y LSTM);\n"
             "la barra negra es la mediana. A la derecha de cada panel, videos con cambio que no tuvieron alerta (media de las semillas).", fontsize=8, color=TINTA_2)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=3)
    guardar(fig, os.path.join(a.salida, "adelanto_alerta.png"))


def figura_etapas(a):
    v = leer_json(a.velocidad)
    etapas = [("lectura", "ms_lectura"), ("YOLO", "ms_yolo"), ("cámara y cortes", "ms_camara"), ("seguimiento", "ms_seguimiento"),
              ("descriptores", "ms_descriptores"), ("vector de escena", "ms_vector"), ("modelo (GRU)", "ms_modelo_gru_s42"),
              ("decisión", "ms_decision_gru_s42")]
    valores = [v["media_ms"][k] for _, k in etapas]
    total = sum(valores)
    fig, eje = plt.subplots(figsize=(8, 3.8))
    limpiar(eje, "x")
    ys = np.arange(len(etapas))[::-1]
    eje.barh(ys, valores, color=COLOR["cfc"], height=0.62)
    for y, x in zip(ys, valores):
        eje.text(x + 0.5, y, f"{x:.2f} ms" if x < 1 else f"{x:.1f} ms", va="center", fontsize=9, color=TINTA)
    eje.set_yticks(ys)
    eje.set_yticklabels([e for e, _ in etapas])
    eje.set_xlabel("milisegundos por fotograma procesado (media de 5 videos de prueba)")
    eje.set_xlim(0, max(valores) * 1.15)
    eje.set_title(f"Sistema completo en vivo: {total:.1f} ms por fotograma, {v['fps']['gru_s42']:.1f} fps", fontsize=10.5, loc="left")
    fig.tight_layout()
    guardar(fig, os.path.join(a.salida, "tiempo_etapas.png"))


def figura_robustez(a):
    puntos = leer_json(a.robustez)["puntos"]
    fracciones = [("completa", 0), ("quitar_25", 25), ("quitar_50", 50), ("quitar_75", 75)]
    xs = [x for _, x in fracciones]
    fig, ejes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True)
    for fila, clave in enumerate(("ia", "humano")):
        valores = [puntos[f"{clave}|{p}|{m}|{fr}"]["f1"] for p, ms in (("parte1", MODELOS), ("parte2", MODELOS[:3])) for m in ms for fr, _ in fracciones]
        lo, hi = np.floor(min(valores) * 50 - 1) / 50, np.ceil(max(valores) * 50 + 1) / 50
        for col, parte in enumerate(("parte1", "parte2")):
            eje = ejes[fila, col]
            limpiar(eje)
            modelos = MODELOS if parte == "parte1" else MODELOS[:3]
            if parte == "parte2":
                for m in modelos:
                    eje.plot(xs, [puntos[f"{clave}|parte1|{m}|{fr}"]["f1"] for fr, _ in fracciones], color=COLOR[m], linewidth=1.2,
                             linestyle=(0, (4, 3)), alpha=0.55, zorder=2)
            for m in modelos:
                ys = np.array([puntos[f"{clave}|{parte}|{m}|{fr}"]["f1"] for fr, _ in fracciones])
                sd = np.array([puntos[f"{clave}|{parte}|{m}|{fr}"]["f1_sd"] for fr, _ in fracciones])
                if puntos[f"{clave}|{parte}|{m}|completa"]["n"] > 1:
                    eje.fill_between(xs, ys - sd, ys + sd, color=COLOR[m], alpha=0.13, linewidth=0, zorder=1)
                eje.plot(xs, ys, color=COLOR[m], linewidth=2, marker=MARCA[m], markersize=6, markeredgecolor=FONDO, markeredgewidth=1.2, zorder=3)
            eje.set_ylim(lo, hi)
            eje.set_xticks(xs)
            eje.set_xticklabels([f"{x} %" for x in xs])
            eje.tick_params(labelbottom=True)
            titulo = "Modelos congelados" if parte == "parte1" else "Reentrenados con fotogramas quitados"
            eje.set_title(f"{titulo}, {REFERENCIA[clave]}", fontsize=9.8, loc="left")
            if col == 0:
                eje.set_ylabel("F1 macro (nivel por segundo)")
            if fila == 1:
                eje.set_xlabel("fotogramas quitados")
    leyenda = [Line2D([0], [0], color=COLOR[m], linewidth=2, marker=MARCA[m], markersize=6, markeredgecolor=FONDO, label=NOMBRE[m]) for m in MODELOS]
    leyenda.append(Line2D([0], [0], color=TINTA_2, linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.7, label="congelado (paneles de la derecha)"))
    fig.legend(handles=leyenda, loc="upper center", ncol=6, frameon=False, fontsize=8.8, labelcolor=TINTA, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    guardar(fig, os.path.join(a.salida, "f1_vs_quitados.png"))


def gif_vela(a):
    cap = cv2.VideoCapture(a.video_demo)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    inicio, fin = int(round(a.gif_inicio * fps)), int(round((a.gif_inicio + a.gif_segundos) * fps))
    paso = max(1, int(round(fps / a.gif_fps)))
    fotos, n = [], 0
    while True:
        ok, img = cap.read()
        if not ok or n >= fin:
            break
        if n >= inicio and (n - inicio) % paso == 0:
            alto = int(round(img.shape[0] * a.gif_ancho / img.shape[1]))
            img = cv2.resize(img, (a.gif_ancho, alto), interpolation=cv2.INTER_AREA)
            fotos.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        n += 1
    cap.release()
    paleta = [f.quantize(colors=a.gif_colores, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in fotos]
    ruta = os.path.join(a.salida, "demo_vela_80.gif")
    paleta[0].save(ruta, save_all=True, append_images=paleta[1:], duration=int(1000 / a.gif_fps), loop=0, optimize=True)
    print(f"{ruta} | {os.path.getsize(ruta) / 1e6:.2f} mb | {len(fotos)} cuadros de {fotos[0].size[0]}x{fotos[0].size[1]}", flush=True)


def ic_map(ruta):
    salida = {}
    for linea in open(ruta, encoding="utf-8"):
        g = re.match(r"\| (IA|humana) \| \d+ \| (\S+(?: \S+)?) \| [\d.]+(?: \+- [\d.]+)? \| ([\d.]+) a ([\d.]+) \|", linea)
        if g:
            clave = "ia" if g.group(1) == "IA" else "humano"
            modelo = {"GRU": "gru", "CfC": "cfc", "gradient boosting": "gradient boosting"}[g.group(2)]
            salida[(clave, modelo)] = (float(g.group(3)), float(g.group(4)))
    return salida


def tablas(a, metricas):
    boot = metricas["bootstrap"]["resultado"]
    ics = ic_map(a.analisis_prueba)
    lineas = ["# Tablas del README del 80 %", "", "Generado por `python figuras_readme_80.py --solo tablas` desde `resultados/prueba/metricas.json`,"
              " `resultados/prueba/analisis.md` y `resultados/robustez/analisis.json`.", "", "## Resultados en la prueba", ""]
    lineas.append("| Modelo | Referencia | F1 macro [IC 95 %] | Kappa lineal [IC 95 %] | mAP tIoU 0.5 | Falsas alarmas sin fuego | Falsas alarmas en controlados |"
          " Adelanto mediano (alerta antes del cambio) |")
    lineas.append("|---|---|---|---|---|---|---|---|")
    for m in MODELOS:
        corr = [c for c in metricas["corridas"].values() if c["modelo"] == m]
        for clave in ("ia", "humano"):
            rs = [c["referencias"][clave] for c in corr]
            f1 = np.mean([r["final"]["f1"] for r in rs])
            k = np.mean([r["final"]["kappa"] for r in rs])
            mp = np.mean([r["map"]["0.5"]["promedio"] for r in rs])
            al = [r["alerta"]["0.1"] for r in rs]
            b = boot[clave]["modelos"][m]
            texto_map = f"{mp:.3f}" + (f" [{ics[(clave, m)][0]:.3f}, {ics[(clave, m)][1]:.3f}]" if (clave, m) in ics else "")
            sin = np.mean([x["alarmas_sin_fuego"] for x in al])
            ctrl = np.mean([x["alarmas_controlado"] for x in al])
            ewt = np.mean([x["ewt_mediana"] for x in al])
            antes = np.mean([x["antes_del_cambio"] for x in al])
            fmt = lambda x: f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"
            lineas.append(f"| {NOMBRE[m]} | {'IA' if clave == 'ia' else 'humana'} | {f1:.3f} [{b['f1'][0]:.3f}, {b['f1'][1]:.3f}] |"
                  f" {k:.3f} [{b['kappa'][0]:.3f}, {b['kappa'][1]:.3f}] | {texto_map} | {fmt(sin)} de {al[0]['n_sin_fuego']} |"
                  f" {fmt(ctrl)} de {al[0]['n_controlado']} ({100 * ctrl / al[0]['n_controlado']:.0f} %) |"
                  f" {ewt:+.1f} s ({fmt(antes)} de {al[0]['n_cambio']}) |")
    rob = leer_json(a.robustez)
    p = rob["puntos"]
    lineas += ["", "## Robustez: F1 macro con 0 y 75 % de fotogramas quitados", ""]
    lineas.append("| Modelo | Referencia | Congelado, 0 % | Congelado, 75 % | Reentrenado, 0 % | Reentrenado, 75 % |")
    lineas.append("|---|---|---|---|---|---|")
    for m in MODELOS[:3]:
        for clave in ("ia", "humano"):
            lineas.append(f"| {NOMBRE[m]} | {'IA' if clave == 'ia' else 'humana'} | {p[f'{clave}|parte1|{m}|completa']['f1']:.3f} |"
                  f" {p[f'{clave}|parte1|{m}|quitar_75']['f1']:.3f} | {p[f'{clave}|parte2|{m}|completa']['f1']:.3f} |"
                  f" {p[f'{clave}|parte2|{m}|quitar_75']['f1']:.3f} |")
    lineas += ["", "Diferencia de caidas relativas del F1 macro a 75 % (GRU menos CfC; positiva = la CfC cae menos):", ""]
    for x in rob["pruebas"]:
        if x["prueba"] == "diferencia_caida_relativa" and x["comparacion"] == "gru - cfc" and x["fraccion"] == "quitar_75" and x["metrica"] == "f1":
            lineas.append(f"- {x['parte']}, {x['referencia']}: {100 * x['punto']:+.1f} pp [{100 * x['ic'][0]:+.1f}, {100 * x['ic'][1]:+.1f}]")
    with open(a.tablas, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas) + "\n")
    print(f"tablas en {a.tablas}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--salida", default=os.path.join("docs", "img"))
    p.add_argument("--descriptores_v2", default=os.path.join("resultados", "descriptores_v2", "descriptores.csv"))
    p.add_argument("--video_descriptores", default="14294880_3840_2160_24fps.mp4")
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--revision", default=os.path.join("resultados", "revision_cambios.csv"))
    p.add_argument("--notas", default=os.path.join("resultados", "revision_cambios_notas.csv"))
    p.add_argument("--modelos_v1", default=os.path.join("resultados", "modelos"))
    p.add_argument("--final_a", default=os.path.join("resultados", "modelos_escena_v2", "final"))
    p.add_argument("--final_b", default=os.path.join("resultados", "modelos_final", "final"))
    p.add_argument("--metricas", default=os.path.join("resultados", "prueba", "metricas.json"))
    p.add_argument("--analisis_prueba", default=os.path.join("resultados", "prueba", "analisis.md"))
    p.add_argument("--alerta", default=os.path.join("resultados", "prueba", "alerta.csv"))
    p.add_argument("--velocidad", default=os.path.join("resultados", "prueba", "velocidad.json"))
    p.add_argument("--robustez", default=os.path.join("resultados", "robustez", "analisis.json"))
    p.add_argument("--video_demo", default=os.path.join("runs_fuego", "demo", "sistema_vela.mp4"))
    p.add_argument("--gif_inicio", type=float, default=0.0)
    p.add_argument("--gif_segundos", type=float, default=8.0)
    p.add_argument("--gif_fps", type=float, default=10.0)
    p.add_argument("--gif_ancho", type=int, default=640)
    p.add_argument("--gif_colores", type=int, default=256)
    p.add_argument("--tablas", default=os.path.join("resultados", "tablas_readme_80.md"))
    p.add_argument("--solo", nargs="*", default=[])
    a = p.parse_args()

    estilo()
    os.makedirs(a.salida, exist_ok=True)
    metricas = leer_json(a.metricas)
    tareas = {"descriptores": lambda: figura_descriptores(a), "instantes": lambda: figura_instantes(a), "curvas": lambda: figura_curvas(a),
              "confusion": lambda: figura_confusion(a, metricas), "map": lambda: figura_map(a, metricas),
              "adelanto": lambda: figura_adelanto(a, metricas), "etapas": lambda: figura_etapas(a), "robustez": lambda: figura_robustez(a),
              "gif": lambda: gif_vela(a), "tablas": lambda: tablas(a, metricas)}
    for nombre, tarea in tareas.items():
        if not a.solo or nombre in a.solo:
            tarea()


if __name__ == "__main__":
    main()
