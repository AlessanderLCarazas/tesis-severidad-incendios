import os
import csv
import json
import argparse
import textwrap
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from etiquetar import cargar_tiempos, estado_desde, NIVELES
import etiquetar_v2 as ev
from comparar_etiquetas import kappa
from preprocesamiento import buscar_rutas


def leer_csv(ruta):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    return None if v in ("", None) else float(v)


def fig_dt(video, rutas, marcas, salida, ventana):
    cap = cv2.VideoCapture(rutas[video])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fuente = []
    idx = -1
    while cap.grab():
        idx += 1
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        fuente.append(max(0.0, t_ms / 1000.0 if (t_ms > 0 or idx == 0) else idx / fps))
    cap.release()
    filas = [r for r in leer_csv(marcas) if r["archivo"] == video]
    t = np.array([float(r["t_k"]) for r in filas])
    dt = np.array([float(r["dt_k"]) for r in filas]) * 1000
    fig, (e1, e2) = plt.subplots(2, 1, figsize=(10, 5.2), gridspec_kw={"height_ratios": [1, 1.4]})
    f = np.array(fuente)
    f = f[f <= ventana]
    e1.vlines(f, 0, 1, color="0.6", lw=1, label=f"fotogramas del archivo ({fps:.0f} fps)")
    m = t <= ventana
    e1.vlines(t[m], 0, 1, color="tab:red", lw=2.5, label="fotogramas muestreados (~10 fps)")
    for x in np.arange(0, ventana + 1e-9, 0.1):
        e1.axvline(x, color="tab:blue", lw=0.6, ls=":", alpha=0.6)
    e1.set_xlim(-0.02, ventana + 0.02)
    e1.set_ylim(0, 1.55)
    e1.set_yticks([])
    e1.set_xlabel("t (s)")
    e1.set_title(f"marcas de tiempo reales en los primeros {ventana:.0f} s (lineas punteadas: rejilla de 100 ms)", fontsize=10)
    e1.legend(loc="upper right", fontsize=8, ncol=2)
    e2.plot(t[1:], dt[1:], ".-", lw=0.8, ms=3, color="tab:red", label="dt_k real")
    e2.axhline(100, color="tab:blue", ls="--", lw=1, label="100 ms (10 fps nominal)")
    e2.set_xlabel("t_k (s)")
    e2.set_ylabel("dt_k (ms)")
    e2.set_ylim(min(60, dt[1:].min() - 10), max(140, dt[1:].max() + 10))
    e2.legend(loc="upper right", fontsize=8)
    e2.grid(alpha=0.3)
    fig.suptitle(f"{video}: {len(fuente)} fotogramas a {fps:.0f} fps -> {len(t)} muestras; dt_k entre {dt[1:].min():.0f} y {dt[1:].max():.0f} ms "
                 f"(media {dt[1:].mean():.1f} ms)", fontsize=10)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)
    return {"fotogramas": len(fuente), "fps": fps, "muestras": len(t), "dt_min": dt[1:].min(), "dt_max": dt[1:].max(),
            "dt_media": dt[1:].mean()}


def fig_curva(video, curvas, propuestas, salida):
    prop = {r["archivo"]: r for r in propuestas}[video]
    c = leer_csv(os.path.join(curvas, os.path.splitext(video)[0] + ".csv"))
    t = np.array([float(r["t"]) for r in c])
    fr = np.array([float(r["fuego"]) for r in c])
    fs = np.array([float(r["fuego_suave"]) for r in c])
    hs = np.array([float(r["humo_suave"]) for r in c])
    a0 = float(prop["a0"])
    u_inc = max(2 * a0, a0 + 0.02)
    u_unc = max(5 * a0, 0.25)
    fig, e = plt.subplots(figsize=(13, 4.2))
    t0, t1 = num(prop["t_inicio_util"]), num(prop["t_fin_util"])
    if t0 is not None:
        e.axvspan(t0, t1, color="tab:green", alpha=0.08, label=f"tramo util propuesto ({t0:.1f} a {t1:.1f} s)")
    e.plot(t, fr, color="tab:orange", alpha=0.3, lw=0.8, label="area de fuego (cada muestra)")
    e.plot(t, fs, color="tab:orange", lw=2, label="area de fuego (mediana de 1 s)")
    e.plot(t, hs, color="0.45", lw=1.5, label="area de humo (mediana de 1 s)")
    e.axhline(a0, color="tab:green", ls=":", lw=1, label=f"a0 = {a0:.3f} (primeros 2 s)")
    e.axhline(u_inc, color="goldenrod", ls="--", lw=1, label=f"umbral de incipiente = max(2 a0, a0 + 0.02) = {u_inc:.3f}")
    tope = max(fs.max(), hs.max(), u_inc) * 1.15
    if u_unc <= tope:
        e.axhline(u_unc, color="tab:red", ls="--", lw=1, label=f"umbral de incontrolado = {u_unc:.3f}")
    for campo, color, nombre in [("t_incipiente", "goldenrod", "incipiente"), ("t_incontrolado", "tab:red", "incontrolado")]:
        v = num(prop[campo])
        if v is not None:
            e.axvline(v, color=color, lw=2, label=f"paso a {nombre} propuesto: {v:.1f} s")
    e.set_ylim(0, tope)
    e.set_xlabel("t (s)")
    e.set_ylabel("fraccion del fotograma")
    e.grid(alpha=0.3)
    e.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    motivos = prop["motivos"].replace(";", ", ") or "-"
    e.set_title(f"{video}: propuesta de preanotar.py (nivel inicial {prop['nivel_inicial']}, {prop['claridad']}; motivos: {motivos})",
                fontsize=9.5)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)
    return prop


def fig_tira_ia(video, tiras, crudo, humanos, salida):
    base = os.path.splitext(video)[0]
    tira = cv2.cvtColor(cv2.imread(os.path.join(tiras, base + ".jpg")), cv2.COLOR_BGR2RGB)
    with open(os.path.join(crudo, base + ".json"), encoding="utf-8") as f:
        respuesta = json.load(f)
    lineas = ["Respuesta cruda de la IA", f"(resultados/ia_crudo/{base}.json)", ""]
    for k, v in respuesta.items():
        if k == "motivo":
            lineas += [f"{k}:"] + ["  " + x for x in textwrap.wrap(str(v), 34)]
        else:
            lineas.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    h = humanos.get(video)
    lineas += ["", "Etiqueta del autor a ciegas", "(etiquetas.csv)", ""]
    if h:
        lineas += [f"nivel_inicial: {h['nivel_inicial']}", f"t_incipiente: {h['t_incipiente'] or '-'}",
                   f"t_incontrolado: {h['t_incontrolado'] or '-'}",
                   f"tramo util: {h['t_inicio_util'] or 'inicio'} a {h['t_fin_util'] or 'fin'}"]
    else:
        lineas.append("no esta entre los 40 videos validados")
    fig = plt.figure(figsize=(14, 4.6))
    e1 = fig.add_axes([0.0, 0.0, 0.70, 1.0])
    e1.imshow(tira)
    e1.axis("off")
    e2 = fig.add_axes([0.71, 0.0, 0.29, 1.0])
    e2.axis("off")
    e2.text(0.0, 0.98, "\n".join(lineas), va="top", ha="left", family="monospace", fontsize=8.6)
    fig.savefig(salida, dpi=110)
    plt.close(fig)
    return respuesta, h


def fig_herramienta(videos, tiras, humanos, salida_png, salida_gif, ancho_gif):
    tiempos = cargar_tiempos(os.path.join(tiras, "tiempos.csv"))
    cuadros = []
    for i, video in enumerate(videos, 1):
        tira = cv2.imread(os.path.join(tiras, os.path.splitext(video)[0] + ".jpg"))
        ts = tiempos[video]
        h = humanos.get(video)
        est = estado_desde(h) if h else estado_desde(None)
        vacio = estado_desde(None)
        escala = min(1.0, 1500 / tira.shape[1], 600 / tira.shape[0])
        info = f"[{i}/{len(videos)}] ciego | {video[:60]} | instante: {max(0.0, ts[0]):.1f} s | ya guardado"
        vista = ev.dibujar(tira, ts, est, vacio, None, None, 0, ts[0], info, escala, 1100, True)[0]
        cuadros.append(vista)
    Image.fromarray(cv2.cvtColor(cuadros[0], cv2.COLOR_BGR2RGB)).quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(salida_png, optimize=True)
    alto = max(c.shape[0] for c in cuadros)
    ancho = max(c.shape[1] for c in cuadros)
    fotos = []
    for c in cuadros:
        lienzo = np.full((alto, ancho, 3), 22, np.uint8)
        lienzo[:c.shape[0], :c.shape[1]] = c
        g = cv2.resize(lienzo, (ancho_gif, int(round(alto * ancho_gif / ancho))), interpolation=cv2.INTER_AREA)
        fotos.append(Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)).quantize(colors=128, method=Image.Quantize.MEDIANCUT,
                                                                                  dither=Image.Dither.NONE))
    fotos[0].save(salida_gif, save_all=True, append_images=fotos[1:], duration=1800, loop=0, optimize=True)


def fig_matriz(humanos, ia, lista, salida):
    validos = [n for n in lista if n in humanos and n in ia and humanos[n]["nivel_inicial"] in NIVELES
               and ia[n]["nivel_inicial"] in NIVELES]
    h = [NIVELES.index(humanos[n]["nivel_inicial"]) for n in validos]
    a = [NIVELES.index(ia[n]["nivel_inicial"]) for n in validos]
    acuerdo = float(np.mean([x == y for x, y in zip(h, a)]))
    kl, m = kappa(h, a, 3, "lineal")
    fig, e = plt.subplots(figsize=(5.4, 4.6))
    e.imshow(m, cmap="Oranges")
    for i in range(3):
        for j in range(3):
            e.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=14,
                   color="white" if m[i, j] > m.max() / 2 else "black")
    e.set_xticks(range(3), NIVELES)
    e.set_yticks(range(3), NIVELES)
    e.set_xlabel("IA")
    e.set_ylabel("autor (a ciegas)")
    e.set_title(f"Nivel inicial, {len(validos)} videos\nacuerdo {acuerdo * 100:.1f} %, kappa ponderado lineal {kl:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(salida, dpi=110)
    plt.close(fig)
    return len(validos), acuerdo, kl, m


def tabla_kalman(ruta):
    r = leer_csv(ruta)
    nombres = [("original", "original (`dt` = 1)"), ("dt_real", "`dt` real"), ("dt_tope2", "`dt` real con tope 2 (sistema)"),
               ("dt_solo_ruido", "`dt` solo en el ruido de proceso")]
    lineas = ["| Variante | Todos los fotogramas: trayectorias / cortes | Quitando el 50 %: trayectorias / cortes |", "|---|---|---|"]
    for v, nombre in nombres:
        celdas = []
        for q in ["0.0", "0.5"]:
            xs = [x for x in r if x["variante"] == v and x["quitar"] == q and x["clase"] == "fuego"]
            celdas.append(f"{sum(int(x['trayectorias']) for x in xs)} / {sum(int(x['cortes']) for x in xs)}")
        lineas.append(f"| {nombre} | {celdas[0]} | {celdas[1]} |")
    return "\n".join(lineas)


def tabla_camara(ruta_todas, ruta_fuego):
    filas = []
    for etiqueta, ruta, v in [("sin compensar", ruta_todas, "dt_tope2"), ("compensando, puntos fuera de todas las cajas", ruta_todas, "dt_tope2_cmc"),
                              ("compensando, puntos fuera de las cajas de fuego (sistema)", ruta_fuego, "dt_tope2_cmc")]:
        xs = [x for x in leer_csv(ruta) if x["variante"] == v and x["quitar"] == "0.0" and x["clase"] == "fuego"]
        con = [x for x in xs if int(x["trayectorias"]) > 0]
        filas.append(f"| {etiqueta} | {sum(int(x['trayectorias']) for x in xs)} | {sum(int(x['cortes']) for x in xs)} | "
                     f"{sum(int(x['cortas']) for x in xs)} | {sum(float(x['cobertura']) for x in con) / len(con):.3f} |")
    return "\n".join(["| Seguimiento (fuego, 4 videos, todos los fotogramas) | Trayectorias | Cortes | Cortas | Cobertura media |",
                      "|---|---|---|---|---|"] + filas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--salida", default=os.path.join("docs", "img"))
    p.add_argument("--video_dt", default="5284fda1-b4c45d7c.mp4")
    p.add_argument("--ventana_dt", type=float, default=2.0)
    p.add_argument("--video_curva", default="12487581_1920_1080_30fps.mp4")
    p.add_argument("--video_tira", default="12857999_3840_2160_24fps.mp4")
    p.add_argument("--videos_herramienta", nargs="+", default=["12857999_3840_2160_24fps.mp4", "14294952_3840_2160_24fps.mp4",
                                                                 "14035491_2048_1080_30fps.mp4", "19151618-hd_1920_1080_30fps.mp4"])
    p.add_argument("--ancho_gif", type=int, default=760)
    p.add_argument("--videos", nargs="+", default=[os.path.join("real_video", "real_video", "fire"),
                                                   os.path.join("real_video", "real_video", "non_fire")])
    a = p.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    rutas = buscar_rutas(a.videos)
    propuestas = leer_csv("propuestas.csv") + leer_csv("propuestas_extra.csv")
    humanos = {r["archivo"]: r for r in leer_csv("etiquetas.csv") if r["anotador"] == "aleCa"}
    ia = {r["archivo"]: r for r in leer_csv(os.path.join("resultados", "etiquetas_ia.csv"))}
    lista = [r["archivo"] for r in leer_csv(os.path.join("resultados", "prueba_humana.csv"))]

    d = fig_dt(a.video_dt, rutas, os.path.join("resultados", "preprocesamiento", "marcas.csv"),
               os.path.join(a.salida, "preprocesamiento_dt.png"), a.ventana_dt)
    print(f"preprocesamiento_dt.png: {a.video_dt} | {d['fotogramas']} fotogramas a {d['fps']:.2f} fps, {d['muestras']} muestras,"
          f" dt {d['dt_min']:.0f}-{d['dt_max']:.0f} ms, media {d['dt_media']:.1f} ms")
    pr = fig_curva(a.video_curva, "curvas", propuestas, os.path.join(a.salida, "etiquetado_curva.png"))
    print(f"etiquetado_curva.png: {a.video_curva} | nivel {pr['nivel_inicial']}, util {pr['t_inicio_util']}-{pr['t_fin_util']},"
          f" inc {pr['t_incipiente'] or '-'}, unc {pr['t_incontrolado'] or '-'}, {pr['claridad']} {pr['motivos']}")
    resp, h = fig_tira_ia(a.video_tira, "tiras", os.path.join("resultados", "ia_crudo"), humanos,
                          os.path.join(a.salida, "etiquetado_tira_ia.png"))
    print(f"etiquetado_tira_ia.png: {a.video_tira} | ia {resp.get('nivel_inicial')} | autor {h['nivel_inicial'] if h else 'no validado'}")
    fig_herramienta(a.videos_herramienta, "tiras", humanos, os.path.join(a.salida, "etiquetar_v2_ciego.png"),
                    os.path.join(a.salida, "etiquetar_v2_ciego.gif"), a.ancho_gif)
    print(f"etiquetar_v2_ciego.png y .gif: {len(a.videos_herramienta)} videos, gif "
          f"{os.path.getsize(os.path.join(a.salida, 'etiquetar_v2_ciego.gif')) / 1e6:.2f} mb")
    n, acuerdo, kl, m = fig_matriz(humanos, ia, lista, os.path.join(a.salida, "matriz_confusion.png"))
    print(f"matriz_confusion.png: {n} videos, acuerdo {acuerdo * 100:.1f} %, kappa lineal {kl:.3f}")
    curvas = len(propuestas)
    crecimiento = sum(1 for r in propuestas if r["t_incipiente"] or r["t_incontrolado"])
    print(f"conteos: {len(ia)} videos etiquetados por ia, {len(humanos)} validados por el autor, {curvas} curvas de area,"
          f" {crecimiento} candidatos de crecimiento")
    print()
    print(tabla_kalman(os.path.join("resultados", "seguimiento", "comparacion.csv")))
    print()
    print(tabla_camara(os.path.join("resultados", "camara", "comparacion_todas.csv"),
                       os.path.join("resultados", "camara", "comparacion_fuego.csv")))


if __name__ == "__main__":
    main()
