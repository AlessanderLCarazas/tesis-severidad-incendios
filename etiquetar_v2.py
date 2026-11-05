import os
import csv
import argparse
from datetime import datetime
import cv2
import numpy as np
from etiquetar import (COLUMNAS, NIVELES, COLORES, GRIS, cargar_tiempos, cargar_etiquetas, guardar_etiquetas,
                       nivel_en, texto, fmt, reproducir, estado_desde, validar)

AYUDA = [
    "clic en foto o en la curva: elegir instante | 1/2/3: nivel inicial (controlado/incipiente/incontrolado)",
    "i: pasa a incipiente | u: pasa a incontrolado | a: inicio util | f: fin util | p: volver a la propuesta",
    "v: ver video desde el instante | x: descartar | r: reiniciar | n o espacio: guardar y siguiente | b: anterior | q: salir",
]
ALTO_PANEL = 176
ALTO_CURVA = 200
MARGEN_IZQ = 56
MARGEN_DER = 14
NARANJA = (0, 140, 255)
HUMO = (200, 200, 200)
FONDO = (30, 30, 30)


def cargar_propuestas(ruta):
    if not os.path.exists(ruta):
        return {}
    with open(ruta, encoding="utf-8") as f:
        return {r["archivo"]: r for r in csv.DictReader(f) if None not in r and None not in r.values()}


def cargar_curva(ruta):
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding="utf-8") as f:
        filas = [[float(r[k]) for k in ("t", "fuego", "fuego_suave", "humo_suave")] for r in csv.DictReader(f)]
    return np.array(filas, float) if filas else None


def estado_propuesta(prop):
    if prop is None:
        return estado_desde(None)
    est = estado_desde(prop)
    if est["nivel_inicial"] is None and not est["descartar"]:
        est["nivel_inicial"] = 0
    return est


def ordenar(lista, propuestas, orden):
    if orden == "original":
        return lista
    def clave(par):
        i, r = par
        prop = propuestas.get(r["archivo"])
        if prop is None:
            return (2, 0, i)
        motivos = [m for m in prop["motivos"].split(";") if m]
        return (0 if prop["claridad"] == "clara" else 1, len(motivos), i)
    return [r for _, r in sorted(enumerate(lista), key=clave)]


def grupo(prop):
    if prop is None:
        return "sin propuesta"
    return prop["claridad"]


def x_de(t, t_max, ancho):
    return int(MARGEN_IZQ + (ancho - MARGEN_IZQ - MARGEN_DER) * min(max(t / t_max, 0.0), 1.0))


def t_de(x, t_max, ancho):
    return max(0.0, min(t_max, (x - MARGEN_IZQ) / (ancho - MARGEN_IZQ - MARGEN_DER) * t_max))


def marca_vertical(img, x, y0, y1, color, grosor, etiqueta):
    cv2.line(img, (x, y0), (x, y1), color, grosor)
    cv2.putText(img, etiqueta, (x + 3, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def marcas_tiempo(img, ancho, t_max, est, t_cursor, y_banda0, y_top, y_bot):
    for campo, color, etiqueta in [("t_inicio_util", (255, 255, 255), "a"), ("t_fin_util", (255, 255, 255), "f"),
                                   ("t_incipiente", COLORES[1], "i"), ("t_incontrolado", COLORES[2], "u")]:
        if est[campo] is not None:
            marca_vertical(img, x_de(est[campo], t_max, ancho), y_top, y_bot, color, 1, etiqueta)
    xc = x_de(t_cursor, t_max, ancho)
    cv2.line(img, (xc, y_banda0), (xc, y_bot), (255, 255, 0), 1)
    for v in np.arange(0, t_max + 1e-6, 10 if t_max > 40 else 5):
        x = x_de(v, t_max, ancho)
        cv2.putText(img, f"{v:.0f}", (x - 6, ALTO_CURVA - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)


def dibujar_curva(ancho, curva, t_max, est, prop, t_cursor, ciego=False):
    img = np.full((ALTO_CURVA, ancho, 3), 22, np.uint8)
    y_banda0, y_banda1 = 6, 18
    y_top, y_bot = 38, ALTO_CURVA - 22
    x0, x1 = MARGEN_IZQ, ancho - MARGEN_DER
    for x in range(x0, x1):
        n = nivel_en(t_de(x, t_max, ancho), est)
        color = GRIS if n is None or n == -1 else COLORES[n]
        if n is None:
            color = (55, 55, 55)
        cv2.line(img, (x, y_banda0), (x, y_banda1), color, 1)
    if curva is None:
        cv2.rectangle(img, (x0, y_top), (x1, y_bot), (60, 60, 60), 1)
        if ciego:
            cv2.putText(img, "modo ciego: linea de tiempo (clic para elegir un instante)", (x0, y_top - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
        else:
            texto(img, "sin curva: falta correr preanotar.py para este video", x0 + 10, (y_top + y_bot) // 2, 0.55, NARANJA)
        marcas_tiempo(img, ancho, t_max, est, t_cursor, y_banda0, y_top, y_bot)
        return img
    t, fr, fs, hs = curva[:, 0], curva[:, 1], curva[:, 2], curva[:, 3]
    a0 = float(prop["a0"]) if prop and prop.get("a0") not in ("", None) else None
    umbrales = []
    if a0 is not None:
        umbrales.append((max(2 * a0, a0 + 0.02), COLORES[1], "inc"))
        umbrales.append((max(5 * a0, 0.25), COLORES[2], "unc"))
    y_max = max(0.05, 1.15 * float(max(fr.max(), hs.max())))
    if umbrales:
        y_max = max(y_max, 1.1 * umbrales[0][0])
    y_max = min(y_max, 1.0)

    def y_de(v):
        return int(y_bot - (y_bot - y_top) * min(v / y_max, 1.0))

    cv2.rectangle(img, (x0, y_top), (x1, y_bot), (60, 60, 60), 1)
    for v in np.linspace(0, y_max, 5)[1:]:
        cv2.line(img, (x0, y_de(v)), (x1, y_de(v)), (45, 45, 45), 1)
        cv2.putText(img, f"{v:.2f}", (4, y_de(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
    for u, color, nombre in umbrales + [(0.4, HUMO, "humo 0.4")]:
        if u <= y_max:
            cv2.line(img, (x0, y_de(u)), (x1, y_de(u)), color, 1, cv2.LINE_AA)
            cv2.putText(img, nombre, (x1 - 60, y_de(u) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    xs = np.array([x_de(v, t_max, ancho) for v in t])
    for serie, color, grosor in [(fr, (40, 70, 110), 1), (hs, HUMO, 1), (fs, NARANJA, 2)]:
        pts = np.column_stack([xs, [y_de(v) for v in serie]]).astype(np.int32)
        cv2.polylines(img, [pts], False, color, grosor, cv2.LINE_AA)
    if prop:
        for campo, color in [("t_incipiente", COLORES[1]), ("t_incontrolado", COLORES[2])]:
            if prop.get(campo) not in ("", None):
                x = x_de(float(prop[campo]), t_max, ancho)
                cv2.drawMarker(img, (x, y_bot + 8), color, cv2.MARKER_TRIANGLE_UP, 10, 2)
    marcas_tiempo(img, ancho, t_max, est, t_cursor, y_banda0, y_top, y_bot)
    cv2.putText(img, "area fuego (naranja: suave, oscura: cruda) | humo (gris) | triangulos: cambios propuestos",
                (x0, y_top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
    return img


def igual_a_propuesta(est, est_prop):
    return all(est[k] == est_prop[k] for k in est)


def dibujar(tira, tiempos, est, est_prop, prop, curva, seleccion, t_cursor, info, escala, ancho_min, ciego=False):
    img = cv2.resize(tira, None, fx=escala, fy=escala)
    filas = (len(tiempos) + COLUMNAS - 1) // COLUMNAS
    cw = img.shape[1] / COLUMNAS
    ch = img.shape[0] / filas
    for k, t in enumerate(tiempos):
        x0, y0 = int((k % COLUMNAS) * cw), int((k // COLUMNAS) * ch)
        x1, y1 = int(x0 + cw) - 1, int(y0 + ch) - 1
        nivel = nivel_en(t, est)
        if nivel is None:
            capa = img[y0:y1, x0:x1].copy()
            img[y0:y1, x0:x1] = (capa * 0.35).astype(np.uint8)
            color = GRIS
        elif nivel == -1:
            color = GRIS
        else:
            color = COLORES[nivel]
        cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), color, 4)
        if k == seleccion:
            cv2.rectangle(img, (x0 + 8, y0 + 8), (x1 - 8, y1 - 8), (255, 255, 255), 2)
    ancho = max(img.shape[1], ancho_min)
    if img.shape[1] < ancho:
        img = np.hstack([img, np.full((img.shape[0], ancho - img.shape[1], 3), 22, np.uint8)])
    t_max = max(tiempos[-1], float(curva[-1, 0]) if curva is not None else 0.0, 1.0)
    panel = np.full((ALTO_PANEL, ancho, 3), 30, np.uint8)
    texto(panel, info, 10, 24, 0.6, (0, 255, 255))
    if ciego:
        texto(panel, "modo ciego: sin propuesta ni curva. elige el nivel inicial con 1/2/3 y marca los cambios", 10, 52, 0.52, (180, 230, 180))
    elif prop is None:
        texto(panel, "sin propuesta para este video", 10, 52, 0.52, NARANJA)
    else:
        lin = (f"propuesta: {prop['nivel_inicial'] or 'descartar'} | incipiente: {fmt(est_prop['t_incipiente'])}"
               f" | incontrolado: {fmt(est_prop['t_incontrolado'])} | util: {fmt(est_prop['t_inicio_util'])} a {fmt(est_prop['t_fin_util'])}"
               f" | a0 {prop['a0'] or '-'} cv {prop.get('cv_area', '') or '-'} conf {prop['conf_media'] or '-'}")
        texto(panel, lin, 10, 52, 0.52, (180, 230, 180))
        if prop["claridad"] == "dudosa":
            texto(panel, f"dudosa: {prop['motivos'].replace(';', ', ')}", 10, 78, 0.52, NARANJA)
        else:
            texto(panel, "clara", 10, 78, 0.52, (80, 200, 80))
    ini = "-" if est["nivel_inicial"] is None else NIVELES[est["nivel_inicial"]]
    cambio = "" if ciego else (" (sin cambios)" if igual_a_propuesta(est, est_prop) else " (corregido)")
    estado = (f"actual{cambio}: {ini} | incipiente: {fmt(est['t_incipiente'])} | incontrolado: {fmt(est['t_incontrolado'])}"
              f" | util: {fmt(est['t_inicio_util'])} a {fmt(est['t_fin_util'])} | descartar: {'si' if est['descartar'] else 'no'}")
    texto(panel, estado, 10, 104, 0.52)
    for n, linea in enumerate(AYUDA):
        texto(panel, linea, 10, 128 + n * 19, 0.44, (200, 200, 200))
    graf = dibujar_curva(ancho, curva, t_max, est, prop, t_cursor, ciego)
    return np.vstack([panel, img, graf]), ch, cw, img.shape[0], t_max, ancho


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anotador", required=True)
    p.add_argument("--seleccion", default="seleccion_etiquetado.csv")
    p.add_argument("--limite", type=int, default=200)
    p.add_argument("--propuestas", default="propuestas.csv")
    p.add_argument("--curvas", default="curvas")
    p.add_argument("--tiras", default="tiras")
    p.add_argument("--videos", default=os.path.join("real_video", "real_video", "fire"))
    p.add_argument("--salida", default="etiquetas.csv")
    p.add_argument("--orden", choices=["claridad", "original"], default="claridad")
    p.add_argument("--solo_doble", action="store_true")
    p.add_argument("--alto_max", type=int, default=600)
    p.add_argument("--ancho_max", type=int, default=1500)
    p.add_argument("--ancho_min", type=int, default=1100)
    p.add_argument("--captura", default="")
    p.add_argument("--lista", default="")
    p.add_argument("--ciego", action="store_true")
    a = p.parse_args()

    if a.lista:
        with open(a.lista, encoding="utf-8") as f:
            lista = [r for r in csv.DictReader(f) if r.get("archivo")]
    else:
        with open(a.seleccion, encoding="utf-8") as f:
            lista = list(csv.DictReader(f))
        if a.limite > 0:
            lista = lista[:a.limite]
    if a.solo_doble:
        lista = [r for r in lista if r.get("segundo_anotador") == "1"]
    tiempos = cargar_tiempos(os.path.join(a.tiras, "tiempos.csv"))
    lista = [r for r in lista if r["archivo"] in tiempos]
    propuestas = {} if a.ciego else cargar_propuestas(a.propuestas)
    lista = ordenar(lista, propuestas, "original" if a.ciego else a.orden)
    etiquetas = cargar_etiquetas(a.salida)
    rutas = {}
    for raiz, _, archivos in os.walk(a.videos):
        for nombre in archivos:
            rutas[nombre] = os.path.join(raiz, nombre)

    def grupo_de(prop):
        return "ciego" if a.ciego else grupo(prop)

    hechos = sum(1 for r in lista if (r["archivo"], a.anotador) in etiquetas)
    if a.ciego:
        print(f"modo ciego. videos para etiquetar: {len(lista)}, ya etiquetados por {a.anotador}: {hechos}")
    else:
        grupos = {}
        for r in lista:
            g = grupo(propuestas.get(r["archivo"]))
            grupos[g] = grupos.get(g, 0) + 1
        print(f"videos para etiquetar: {len(lista)} (claras: {grupos.get('clara', 0)}, dudosas: {grupos.get('dudosa', 0)},"
              f" sin propuesta: {grupos.get('sin propuesta', 0)}), ya etiquetados por {a.anotador}: {hechos}")

    if a.captura:
        os.makedirs(a.captura, exist_ok=True)
        for pos, r in enumerate(lista):
            nombre = r["archivo"]
            tira = cv2.imread(os.path.join(a.tiras, os.path.splitext(nombre)[0] + ".jpg"))
            if tira is None:
                continue
            prop = propuestas.get(nombre)
            est_prop = estado_propuesta(prop)
            curva = None if a.ciego else cargar_curva(os.path.join(a.curvas, os.path.splitext(nombre)[0] + ".csv"))
            escala = min(1.0, a.ancho_max / tira.shape[1], a.alto_max / tira.shape[0])
            info = f"[{pos + 1}/{len(lista)}] {grupo_de(prop)} | {nombre[:70]} | foto 1: {tiempos[nombre][0]:.1f} s"
            vista = dibujar(tira, tiempos[nombre], dict(est_prop), est_prop, prop, curva, 0, tiempos[nombre][0], info, escala,
                            a.ancho_min, a.ciego)[0]
            cv2.imwrite(os.path.join(a.captura, f"{pos + 1:03d}_{os.path.splitext(nombre)[0][:40]}.jpg"), vista)
        print(f"capturas guardadas en: {a.captura}")
        return

    pos = next((i for i, r in enumerate(lista) if (r["archivo"], a.anotador) not in etiquetas), len(lista))
    if pos >= len(lista):
        print("no quedan videos pendientes; se abre el primero para revisar")
        pos = 0

    ventana = "etiquetado v2"
    cv2.namedWindow(ventana, cv2.WINDOW_AUTOSIZE)
    clic = {"pos": None}
    cv2.setMouseCallback(ventana, lambda ev, x, y, fl, prm: clic.update(pos=(x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)

    while 0 <= pos < len(lista):
        nombre = lista[pos]["archivo"]
        ts = tiempos[nombre]
        base = os.path.splitext(nombre)[0]
        tira = cv2.imread(os.path.join(a.tiras, base + ".jpg"))
        if tira is None:
            print(f"no se encontro la tira: {os.path.join(a.tiras, base + '.jpg')}")
            pos += 1
            continue
        escala = min(1.0, a.ancho_max / tira.shape[1], a.alto_max / tira.shape[0])
        if not a.ciego:
            propuestas.update(cargar_propuestas(a.propuestas))
        prop = propuestas.get(nombre)
        curva = None if a.ciego else cargar_curva(os.path.join(a.curvas, base + ".csv"))
        est_prop = estado_propuesta(prop)
        guardado = etiquetas.get((nombre, a.anotador))
        est = estado_desde(guardado) if guardado else dict(est_prop)
        seleccion = 0
        t_cursor = ts[0]
        mensaje = ""
        avance = 0
        while True:
            hechos = sum(1 for r in lista if (r["archivo"], a.anotador) in etiquetas)
            marca = " | ya guardado" if (nombre, a.anotador) in etiquetas else ""
            info = (f"[{pos + 1}/{len(lista)}] hechos: {hechos} | {grupo_de(prop)} | {nombre[:60]} | instante: {t_cursor:.1f} s"
                    f"{marca} {mensaje}")
            vista, ch, cw, alto_tira, t_max, ancho = dibujar(tira, ts, est, est_prop, prop, curva, seleccion, t_cursor,
                                                            info, escala, a.ancho_min, a.ciego)
            cv2.imshow(ventana, vista)
            k = cv2.waitKey(30) & 0xFF
            if clic["pos"] is not None:
                x, y = clic["pos"]
                clic["pos"] = None
                if ALTO_PANEL < y <= ALTO_PANEL + alto_tira and x < cw * COLUMNAS:
                    idx = int((y - ALTO_PANEL) // ch) * COLUMNAS + int(x // cw)
                    if 0 <= idx < len(ts):
                        seleccion = idx
                        t_cursor = ts[idx]
                        mensaje = ""
                elif y > ALTO_PANEL + alto_tira and x >= MARGEN_IZQ:
                    t_cursor = round(t_de(x, t_max, ancho), 2)
                    seleccion = int(np.argmin([abs(v - t_cursor) for v in ts]))
                    mensaje = ""
            if k == 255:
                continue
            c = chr(k).lower() if k < 128 else ""
            t_sel = round(t_cursor, 2)
            if c and c in "123":
                est["nivel_inicial"] = int(c) - 1
            elif c in ("i", "u", "a", "f"):
                campo = {"i": "t_incipiente", "u": "t_incontrolado", "a": "t_inicio_util", "f": "t_fin_util"}[c]
                est[campo] = None if est[campo] == t_sel else t_sel
            elif c == "p":
                est = dict(est_prop)
                mensaje = "| propuesta restaurada"
            elif c == "x":
                est["descartar"] = not est["descartar"]
            elif c == "r":
                est = estado_desde(None)
            elif c == "v":
                ruta_video = rutas.get(nombre)
                if ruta_video is None:
                    mensaje = "| video no encontrado, revisa --videos"
                else:
                    t0 = max(0.0, t_cursor - (ts[1] - ts[0] if len(ts) > 1 else 2.0))
                    t1 = t_cursor + (ts[1] - ts[0] if len(ts) > 1 else 2.0)
                    reproducir(ruta_video, t0, t1, est)
            elif c in ("n", " ", "b", "q"):
                if c == "b":
                    avance = -1
                    break
                error = validar(est)
                if error and c == "q":
                    print(f"salida sin guardar este video: {error}")
                    pos = -10
                    break
                if error:
                    mensaje = f"| error: {error}"
                    continue
                etiquetas[(nombre, a.anotador)] = {
                    "archivo": nombre, "anotador": a.anotador,
                    "nivel_inicial": "" if est["nivel_inicial"] is None else NIVELES[est["nivel_inicial"]],
                    "t_incipiente": "" if est["t_incipiente"] is None else est["t_incipiente"],
                    "t_incontrolado": "" if est["t_incontrolado"] is None else est["t_incontrolado"],
                    "t_inicio_util": "" if est["t_inicio_util"] is None else est["t_inicio_util"],
                    "t_fin_util": "" if est["t_fin_util"] is None else est["t_fin_util"],
                    "descartar": int(est["descartar"]),
                    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                guardar_etiquetas(a.salida, etiquetas)
                avance = 0 if c == "q" else 1
                if c == "q":
                    pos = -10
                break
        pos += avance
        if pos < -1:
            break
        pos = max(pos, 0)
    cv2.destroyAllWindows()
    hechos = sum(1 for r in lista if (r["archivo"], a.anotador) in etiquetas)
    print(f"guardado en {a.salida}. etiquetados por {a.anotador}: {hechos} de {len(lista)}")


if __name__ == "__main__":
    main()
