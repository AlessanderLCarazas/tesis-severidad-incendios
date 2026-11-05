import os
import sys
import csv
import argparse
from datetime import datetime
import cv2
import numpy as np

COLUMNAS = 6
NIVELES = ["controlado", "incipiente", "incontrolado"]
COLORES = [(80, 200, 80), (0, 210, 255), (40, 40, 230)]
GRIS = (130, 130, 130)
CAMPOS = ["archivo", "anotador", "nivel_inicial", "t_incipiente", "t_incontrolado",
          "t_inicio_util", "t_fin_util", "descartar", "fecha"]
AYUDA = [
    "clic: elegir foto | 1/2/3: nivel inicial (controlado/incipiente/incontrolado)",
    "i: pasa a incipiente en la foto | u: pasa a incontrolado | a: inicio util | f: fin util",
    "v: ver video desde la foto | x: descartar video | r: reiniciar | n o espacio: guardar y siguiente | b: anterior | q: salir",
]


def cargar_tiempos(ruta):
    tiempos = {}
    with open(ruta, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tiempos.setdefault(r["archivo"], []).append(float(r["t_s"]))
    return tiempos


def cargar_etiquetas(ruta):
    if not os.path.exists(ruta):
        return {}
    with open(ruta, encoding="utf-8") as f:
        return {(r["archivo"], r["anotador"]): r for r in csv.DictReader(f)}


def guardar_etiquetas(ruta, etiquetas):
    tmp = ruta + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        for clave in sorted(etiquetas):
            w.writerow(etiquetas[clave])
    os.replace(tmp, ruta)


def nivel_en(t, est):
    if est["descartar"]:
        return None
    if est["t_inicio_util"] is not None and t < est["t_inicio_util"] - 0.01:
        return None
    if est["t_fin_util"] is not None and t > est["t_fin_util"] + 0.01:
        return None
    nivel = est["nivel_inicial"]
    if nivel is None:
        return -1
    if est["t_incipiente"] is not None and t >= est["t_incipiente"] - 0.01:
        nivel = max(nivel, 1)
    if est["t_incontrolado"] is not None and t >= est["t_incontrolado"] - 0.01:
        nivel = 2
    return nivel


def texto(img, s, x, y, escala=0.55, color=(255, 255, 255)):
    (w, h), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + b + 2), (20, 20, 20), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)


def fmt(v):
    return "-" if v is None else f"{v:.1f}s"


def dibujar(tira, tiempos, est, seleccion, info, escala):
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
    alto_panel = 150
    panel = np.full((alto_panel, img.shape[1], 3), 30, np.uint8)
    texto(panel, info, 10, 24, 0.6, (0, 255, 255))
    ini = "-" if est["nivel_inicial"] is None else NIVELES[est["nivel_inicial"]]
    estado = (f"inicial: {ini} | incipiente: {fmt(est['t_incipiente'])} | incontrolado: {fmt(est['t_incontrolado'])}"
              f" | util: {fmt(est['t_inicio_util'])} a {fmt(est['t_fin_util'])} | descartar: {'si' if est['descartar'] else 'no'}")
    texto(panel, estado, 10, 54, 0.55)
    for n, linea in enumerate(AYUDA):
        texto(panel, linea, 10, 84 + n * 24, 0.47, (200, 200, 200))
    return np.vstack([panel, img]), ch, cw, alto_panel


def reproducir(ruta_video, t_desde, t_hasta, est):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"no se pudo abrir el video: {ruta_video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_desde) * 1000.0)
    pausa = False
    frame = None
    t = t_desde
    while True:
        if not pausa or frame is None:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t > t_hasta:
                pausa = True
        vista = frame.copy()
        h, w = vista.shape[:2]
        esc = min(1.0, 960 / w, 640 / h)
        vista = cv2.resize(vista, None, fx=esc, fy=esc)
        texto(vista, f"t = {t:.2f} s  {'pausa' if pausa else ''}", 10, 28, 0.8, (0, 255, 255))
        texto(vista, "espacio: pausa | i/u/a/f: marcar este instante | esc: volver", 10, vista.shape[0] - 14, 0.55)
        cv2.imshow("video", vista)
        k = cv2.waitKey(1 if pausa else max(1, int(1000 / fps))) & 0xFF
        if k == 27:
            break
        if k == ord(" "):
            pausa = not pausa
        elif k in (ord("i"), ord("u"), ord("a"), ord("f")):
            campo = {"i": "t_incipiente", "u": "t_incontrolado", "a": "t_inicio_util", "f": "t_fin_util"}[chr(k)]
            est[campo] = round(t, 2)
            print(f"{campo} = {t:.2f} s")
    cap.release()
    cv2.destroyWindow("video")


def estado_desde(fila):
    def num(v):
        return float(v) if v not in ("", None) else None
    est = {"nivel_inicial": None, "t_incipiente": None, "t_incontrolado": None,
           "t_inicio_util": None, "t_fin_util": None, "descartar": False}
    if fila:
        if fila["nivel_inicial"] in NIVELES:
            est["nivel_inicial"] = NIVELES.index(fila["nivel_inicial"])
        for c in ["t_incipiente", "t_incontrolado", "t_inicio_util", "t_fin_util"]:
            est[c] = num(fila[c])
        est["descartar"] = fila["descartar"] == "1"
    return est


def validar(est):
    if est["descartar"]:
        return None
    if est["nivel_inicial"] is None:
        return "falta el nivel inicial (1, 2 o 3)"
    if est["t_incipiente"] is not None and est["t_incontrolado"] is not None and est["t_incipiente"] > est["t_incontrolado"]:
        return "incipiente no puede ser despues de incontrolado"
    if est["nivel_inicial"] >= 1 and est["t_incipiente"] is not None:
        return "el nivel inicial ya es incipiente o mayor; borra la marca i"
    if est["nivel_inicial"] == 2 and est["t_incontrolado"] is not None:
        return "el nivel inicial ya es incontrolado; borra la marca u"
    if est["t_inicio_util"] is not None and est["t_fin_util"] is not None and est["t_inicio_util"] >= est["t_fin_util"]:
        return "el inicio util debe ser antes del fin util"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anotador", required=True)
    p.add_argument("--seleccion", default="seleccion_etiquetado.csv")
    p.add_argument("--tiras", default="tiras")
    p.add_argument("--videos", default=os.path.join("real_video", "real_video", "fire"))
    p.add_argument("--salida", default="etiquetas.csv")
    p.add_argument("--solo_doble", action="store_true")
    p.add_argument("--alto_max", type=int, default=820)
    p.add_argument("--ancho_max", type=int, default=1500)
    a = p.parse_args()

    with open(a.seleccion, encoding="utf-8") as f:
        lista = list(csv.DictReader(f))
    if a.solo_doble:
        lista = [r for r in lista if r.get("segundo_anotador") == "1"]
    tiempos = cargar_tiempos(os.path.join(a.tiras, "tiempos.csv"))
    lista = [r for r in lista if r["archivo"] in tiempos]
    etiquetas = cargar_etiquetas(a.salida)
    rutas = {}
    for raiz, _, archivos in os.walk(a.videos):
        for nombre in archivos:
            rutas[nombre] = os.path.join(raiz, nombre)

    hechos = sum(1 for r in lista if (r["archivo"], a.anotador) in etiquetas)
    print(f"videos para etiquetar: {len(lista)}, ya etiquetados por {a.anotador}: {hechos}")
    pos = next((i for i, r in enumerate(lista) if (r["archivo"], a.anotador) not in etiquetas), len(lista))
    if pos >= len(lista):
        print("no quedan videos pendientes; se abre el primero para revisar")
        pos = 0

    ventana = "etiquetado"
    cv2.namedWindow(ventana, cv2.WINDOW_AUTOSIZE)
    clic = {"pos": None}
    cv2.setMouseCallback(ventana, lambda ev, x, y, fl, prm: clic.update(pos=(x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)

    while 0 <= pos < len(lista):
        nombre = lista[pos]["archivo"]
        ts = tiempos[nombre]
        ruta_tira = os.path.join(a.tiras, os.path.splitext(nombre)[0] + ".jpg")
        tira = cv2.imread(ruta_tira)
        if tira is None:
            print(f"no se encontro la tira: {ruta_tira}")
            pos += 1
            continue
        escala = min(1.0, a.ancho_max / tira.shape[1], a.alto_max / tira.shape[0])
        est = estado_desde(etiquetas.get((nombre, a.anotador)))
        seleccion = 0
        mensaje = ""
        avance = 0
        while True:
            hechos = sum(1 for r in lista if (r["archivo"], a.anotador) in etiquetas)
            info = f"[{pos + 1}/{len(lista)}] hechos: {hechos} | {nombre[:70]} | foto {seleccion + 1}: {ts[seleccion]:.1f} s {mensaje}"
            vista, ch, cw, alto_panel = dibujar(tira, ts, est, seleccion, info, escala)
            cv2.imshow(ventana, vista)
            k = cv2.waitKey(30) & 0xFF
            if clic["pos"] is not None:
                x, y = clic["pos"]
                clic["pos"] = None
                if y > alto_panel:
                    idx = int((y - alto_panel) // ch) * COLUMNAS + int(x // cw)
                    if 0 <= idx < len(ts):
                        seleccion = idx
                        mensaje = ""
            if k == 255:
                continue
            c = chr(k).lower() if k < 128 else ""
            t_sel = round(ts[seleccion], 2)
            if c in "123" and c:
                est["nivel_inicial"] = int(c) - 1
            elif c in ("i", "u", "a", "f"):
                campo = {"i": "t_incipiente", "u": "t_incontrolado", "a": "t_inicio_util", "f": "t_fin_util"}[c]
                est[campo] = None if est[campo] == t_sel else t_sel
            elif c == "x":
                est["descartar"] = not est["descartar"]
            elif c == "r":
                est = estado_desde(None)
            elif c == "v":
                ruta_video = rutas.get(nombre)
                if ruta_video is None:
                    mensaje = "| video no encontrado, revisa --videos"
                else:
                    t0 = ts[seleccion - 1] if seleccion > 0 else ts[0]
                    t1 = ts[seleccion + 1] if seleccion + 1 < len(ts) else ts[-1]
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
