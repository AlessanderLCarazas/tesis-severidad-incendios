import os
import csv
import time
import argparse
import unicodedata
from datetime import datetime
import cv2
import numpy as np
from etiquetar import NIVELES, COLORES, GRIS, cargar_etiquetas, guardar_etiquetas, estado_desde, validar, texto, fmt, nivel_en

CAMPOS_TIEMPO = ["archivo", "anotador", "segundos_abierto", "reproducciones", "guardado", "fecha"]
IZQ, DER, ARR, ABA = 2424832, 2555904, 2490368, 2621440
ANCHO, ALTO_VIDEO, ALTO_PANEL, ALTO_LINEA, MARGEN = 1000, 562, 150, 70, 20
ALTO_LINEA_REV = 108
COLOR_H, COLOR_IA = (255, 255, 0), (255, 0, 255)
MARCAS = [("t_inicio_util", "a"), ("t_fin_util", "f"), ("t_incipiente", "i"), ("t_incontrolado", "u")]
AYUDA = [
    "espacio: reproducir/pausa | flechas izq/der: -1/+1 s | flechas arr/abajo: +5/-5 s | , .: fotograma atras/adelante | clic en la linea: ir",
    "1/2/3: nivel inicial | i/u: cambio a incipiente/incontrolado en el instante actual (otra vez: borrar) | a/f: inicio/fin util",
    "x: descartar | r: reiniciar | n o enter: guardar y seguir | b: anterior sin guardar | q: guardar y salir",
]
AYUDA_REV = [
    "solo lectura, no se guarda nada | espacio: reproducir/pausa | flechas izq/der: -1/+1 s | arr/abajo: +5/-5 s | , .: fotograma atras/adelante",
    "m: ir a la siguiente marca | clic en la linea: ir | n o enter: siguiente video | b: anterior | q: salir",
]


class Video:
    def __init__(self, ruta):
        self.cap = cv2.VideoCapture(ruta)
        if not self.cap.isOpened():
            raise IOError(f"no se pudo abrir el video: {ruta}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.dur = max(0.1, n / self.fps)
        self.frame, self.t = None, 0.0
        self.ir(0.0)

    def leer(self):
        ok, frame = self.cap.read()
        if ok:
            self.frame = frame
            self.t = max(0.0, self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            self.dur = max(self.dur, self.t)
        return ok

    def ir(self, t):
        t = float(np.clip(t, 0.0, max(0.0, self.dur - 1.0 / self.fps)))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        if not self.leer() and t > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - 1.0) * 1000.0)
            self.leer()

    def cerrar(self):
        self.cap.release()


def x_de(t, dur):
    return int(MARGEN + (ANCHO - 2 * MARGEN) * min(max(t / dur, 0.0), 1.0))


def t_de(x, dur):
    return (x - MARGEN) / (ANCHO - 2 * MARGEN) * dur


def sin_tildes(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")


def seg(v):
    return "-" if v is None else f"{v:.2f}s"


def resumen(quien, est):
    if est["descartar"]:
        return f"{quien} | descartado"
    ini = "-" if est["nivel_inicial"] is None else NIVELES[est["nivel_inicial"]]
    return (f"{quien} | inicial: {ini} | a incipiente: {seg(est['t_incipiente'])} | a incontrolado: {seg(est['t_incontrolado'])}"
            f" | util: {seg(est['t_inicio_util'])} a {seg(est['t_fin_util'])}")


def cuadro(video, reproduciendo):
    lienzo = np.zeros((ALTO_VIDEO, ANCHO, 3), np.uint8)
    if video.frame is not None:
        h, w = video.frame.shape[:2]
        esc = min(ANCHO / w, ALTO_VIDEO / h)
        img = cv2.resize(video.frame, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA)
        y0, x0 = (ALTO_VIDEO - img.shape[0]) // 2, (ANCHO - img.shape[1]) // 2
        lienzo[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    estado_rep = "reproduciendo" if reproduciendo else "pausa"
    texto(lienzo, f"t = {video.t:.2f} s de {video.dur:.1f} s | {estado_rep}", 10, 28, 0.7, (0, 255, 255))
    return lienzo


def banda(linea, est, dur, y_a, y_b):
    for x in range(MARGEN, ANCHO - MARGEN):
        nivel = nivel_en(t_de(x, dur), est)
        color = GRIS if nivel is None or nivel == -1 else COLORES[nivel]
        cv2.line(linea, (x, y_a), (x, y_b), color, 1)


def segundos(linea, dur, alto):
    for s in range(0, int(dur) + 1, max(1, int(dur // 10) or 1)):
        x = x_de(s, dur)
        cv2.putText(linea, f"{s}", (x - 6, alto - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)


def dibujar(video, est, info, reproduciendo):
    panel = np.full((ALTO_PANEL, ANCHO, 3), 30, np.uint8)
    texto(panel, info, 10, 24, 0.55, (0, 255, 255))
    ini = "-" if est["nivel_inicial"] is None else NIVELES[est["nivel_inicial"]]
    estado = (f"inicial: {ini} | incipiente: {fmt(est['t_incipiente'])} | incontrolado: {fmt(est['t_incontrolado'])}"
              f" | util: {fmt(est['t_inicio_util'])} a {fmt(est['t_fin_util'])} | descartar: {'si' if est['descartar'] else 'no'}")
    texto(panel, estado, 10, 54, 0.55)
    for n, linea in enumerate(AYUDA):
        texto(panel, linea, 10, 84 + n * 24, 0.42, (200, 200, 200))
    lienzo = cuadro(video, reproduciendo)
    linea = np.full((ALTO_LINEA, ANCHO, 3), 25, np.uint8)
    y_a, y_b = 18, 46
    banda(linea, est, video.dur, y_a, y_b)
    for campo, etiqueta, color in [("t_inicio_util", "a", (220, 220, 220)), ("t_fin_util", "f", (220, 220, 220)),
                                   ("t_incipiente", "i", COLORES[1]), ("t_incontrolado", "u", COLORES[2])]:
        if est[campo] is not None:
            x = x_de(est[campo], video.dur)
            cv2.line(linea, (x, y_a - 6), (x, y_b + 6), color, 2)
            cv2.putText(linea, etiqueta, (x - 4, y_a - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    xc = x_de(video.t, video.dur)
    cv2.line(linea, (xc, 6), (xc, ALTO_LINEA - 6), (255, 255, 255), 2)
    segundos(linea, video.dur, ALTO_LINEA)
    return np.vstack([panel, lienzo, linea])


def dibujar_revision(video, est_h, est_ia, info, motivo, reproduciendo):
    panel = np.full((ALTO_PANEL, ANCHO, 3), 30, np.uint8)
    texto(panel, info, 10, 22, 0.5, (0, 255, 255))
    texto(panel, resumen("humano", est_h), 10, 48, 0.5, COLOR_H)
    texto(panel, resumen("ia", est_ia), 10, 74, 0.5, COLOR_IA)
    texto(panel, f"motivo de la ia: {sin_tildes(motivo)}"[:140], 10, 98, 0.4, (220, 220, 220))
    for n, linea in enumerate(AYUDA_REV):
        texto(panel, linea, 10, 120 + n * 20, 0.4, (200, 200, 200))
    lienzo = cuadro(video, reproduciendo)
    linea = np.full((ALTO_LINEA_REV, ANCHO, 3), 25, np.uint8)
    for est, quien, color, y_a, y_b, y_letra in [(est_h, "h", COLOR_H, 20, 38, 12), (est_ia, "ia", COLOR_IA, 50, 68, 87)]:
        banda(linea, est, video.dur, y_a, y_b)
        cv2.putText(linea, quien, (2, y_b - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        for campo, letra in MARCAS:
            if est[campo] is not None:
                x = x_de(est[campo], video.dur)
                cv2.line(linea, (x, y_a - 5), (x, y_b + 5), (0, 0, 0), 4)
                cv2.line(linea, (x, y_a - 5), (x, y_b + 5), color, 2)
                cv2.putText(linea, letra, (x - 4, y_letra), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    xc = x_de(video.t, video.dur)
    cv2.line(linea, (xc, 4), (xc, ALTO_LINEA_REV - 18), (255, 255, 255), 1)
    segundos(linea, video.dur, ALTO_LINEA_REV)
    return np.vstack([panel, lienzo, linea])


def registrar_tiempo(ruta, fila):
    nuevo = not os.path.exists(ruta)
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_TIEMPO)
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def fila_etiqueta(nombre, anotador, est):
    return {"archivo": nombre, "anotador": anotador,
            "nivel_inicial": "" if est["nivel_inicial"] is None else NIVELES[est["nivel_inicial"]],
            "t_incipiente": "" if est["t_incipiente"] is None else est["t_incipiente"],
            "t_incontrolado": "" if est["t_incontrolado"] is None else est["t_incontrolado"],
            "t_inicio_util": "" if est["t_inicio_util"] is None else est["t_inicio_util"],
            "t_fin_util": "" if est["t_fin_util"] is None else est["t_fin_util"],
            "descartar": int(est["descartar"]), "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def buscar_videos(carpeta):
    rutas = {}
    for raiz, _, archivos in os.walk(carpeta):
        for n in archivos:
            rutas.setdefault(n, os.path.join(raiz, n))
    return rutas


def revisar(a, lista, etiquetas, rutas):
    if not os.path.exists(a.ia):
        raise SystemExit(f"no existe el archivo de etiquetas de la ia: {a.ia}")
    with open(a.ia, encoding="utf-8") as f:
        ia = {r["archivo"]: r for r in csv.DictReader(f)}
    con = [v for v in lista if (v, a.anotador) in etiquetas and v in ia]
    print(f"modo revisar, solo lectura. videos de la lista: {len(lista)}, con etiqueta de {a.anotador} y de la ia: {len(con)},"
          f" omitidos: {len(lista) - len(con)}")
    if not con:
        raise SystemExit("no hay videos para revisar")
    faltan = [v for v in con if v not in rutas]
    if faltan:
        raise SystemExit(f"videos no encontrados en {a.videos}: {faltan}")

    ventana = "etiquetar_v3 revisar"
    cv2.namedWindow(ventana, cv2.WINDOW_AUTOSIZE)
    clic = {"pos": None}
    cv2.setMouseCallback(ventana, lambda ev, x, y, fl, prm: clic.update(pos=(x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    pos, salir = 0, False
    while 0 <= pos < len(con) and not salir:
        nombre = con[pos]
        video = Video(rutas[nombre])
        est_h = estado_desde(etiquetas[(nombre, a.anotador)])
        est_ia = estado_desde(ia[nombre])
        motivo = ia[nombre].get("motivo", "")
        marcas = sorted({est[campo] for est in (est_h, est_ia) for campo, _ in MARCAS if est[campo] is not None})
        reproduciendo, avance = False, 0
        while True:
            if reproduciendo and not video.leer():
                reproduciendo = False
            info = f"revisar [{pos + 1}/{len(con)}] {sin_tildes(nombre)[:90]}"
            cv2.imshow(ventana, dibujar_revision(video, est_h, est_ia, info, motivo, reproduciendo))
            k = cv2.waitKeyEx(max(1, int(1000 / video.fps)) if reproduciendo else 30)
            if clic["pos"] is not None:
                x, y = clic["pos"]
                clic["pos"] = None
                if y > ALTO_PANEL + ALTO_VIDEO:
                    video.ir(t_de(x, video.dur))
            if k == -1:
                continue
            if k in (IZQ, DER, ARR, ABA):
                video.ir(video.t + {IZQ: -1.0, DER: 1.0, ARR: 5.0, ABA: -5.0}[k])
                continue
            c = chr(k & 0xFF).lower() if (k & 0xFF) < 128 and k < 256 else ""
            if c == " ":
                reproduciendo = not reproduciendo
            elif c == ".":
                reproduciendo = False
                video.leer()
            elif c == ",":
                reproduciendo = False
                video.ir(video.t - 1.0 / video.fps - 1e-3)
            elif c == "m" and marcas:
                reproduciendo = False
                video.ir(next((t for t in marcas if t > video.t + 0.05), marcas[0]))
            elif c in ("n", "\r"):
                avance = 1
                break
            elif c == "b":
                avance = -1 if pos > 0 else 0
                break
            elif c == "q":
                salir = True
                break
        video.cerrar()
        pos += avance
    cv2.destroyAllWindows()
    print("revision terminada. no se guardo nada")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anotador", required=True)
    p.add_argument("--lista", required=True)
    p.add_argument("--ciego", action="store_true")
    p.add_argument("--revisar", action="store_true")
    p.add_argument("--ia", default=os.path.join("resultados", "etiquetas_ia_v2.csv"))
    p.add_argument("--videos", default=os.path.join("real_video", "real_video", "fire"))
    p.add_argument("--salida", default="etiquetas.csv")
    p.add_argument("--tiempos", default=os.path.join("resultados", "tiempos_etiquetado.csv"))
    a = p.parse_args()
    if a.revisar and a.ciego:
        raise SystemExit("--revisar muestra las etiquetas de la ia y no se combina con --ciego")
    if not a.revisar and not a.ciego:
        raise SystemExit("etiquetar_v3.py solo funciona en modo ciego: agrega --ciego")

    with open(a.lista, encoding="utf-8") as f:
        lista = [r["archivo"] for r in csv.DictReader(f)]
    etiquetas = cargar_etiquetas(a.salida)
    if a.revisar:
        revisar(a, lista, etiquetas, buscar_videos(a.videos))
        return
    otros = sorted({v for (v, an) in etiquetas if an == a.anotador and v not in lista})
    if otros:
        raise SystemExit(f"el anotador {a.anotador} ya tiene etiquetas de {len(otros)} videos fuera de esta lista;"
                         " usa un anotador distinto para este protocolo")
    rutas = buscar_videos(a.videos)
    faltan = [v for v in lista if v not in rutas]
    if faltan:
        raise SystemExit(f"videos no encontrados en {a.videos}: {faltan}")
    hechos = sum(1 for v in lista if (v, a.anotador) in etiquetas)
    print(f"modo ciego. videos: {len(lista)}, ya etiquetados por {a.anotador}: {hechos}")

    ventana = "etiquetar_v3"
    cv2.namedWindow(ventana, cv2.WINDOW_AUTOSIZE)
    clic = {"pos": None}
    cv2.setMouseCallback(ventana, lambda ev, x, y, fl, prm: clic.update(pos=(x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    pos = next((i for i, v in enumerate(lista) if (v, a.anotador) not in etiquetas), 0)
    salir = False
    while 0 <= pos < len(lista) and not salir:
        nombre = lista[pos]
        video = Video(rutas[nombre])
        guardado = etiquetas.get((nombre, a.anotador))
        est = estado_desde(guardado)
        reproduciendo, reproducciones, mensaje = False, 0, ""
        inicio = time.time()
        avance, se_guardo = 0, False
        while True:
            if reproduciendo and not video.leer():
                reproduciendo = False
            hechos = sum(1 for v in lista if (v, a.anotador) in etiquetas)
            marca = " | ya guardado" if (nombre, a.anotador) in etiquetas else ""
            info = f"[{pos + 1}/{len(lista)}] hechos: {hechos} | {nombre[:70]}{marca} {mensaje}"
            cv2.imshow(ventana, dibujar(video, est, info, reproduciendo))
            k = cv2.waitKeyEx(max(1, int(1000 / video.fps)) if reproduciendo else 30)
            if clic["pos"] is not None:
                x, y = clic["pos"]
                clic["pos"] = None
                if y > ALTO_PANEL + ALTO_VIDEO:
                    video.ir(t_de(x, video.dur))
            if k == -1:
                continue
            if k in (IZQ, DER, ARR, ABA):
                video.ir(video.t + {IZQ: -1.0, DER: 1.0, ARR: 5.0, ABA: -5.0}[k])
                continue
            c = chr(k & 0xFF).lower() if (k & 0xFF) < 128 and k < 256 else ""
            t_act = round(video.t, 2)
            if c == " ":
                reproduciendo = not reproduciendo
                reproducciones += int(reproduciendo)
            elif c == ".":
                reproduciendo = False
                video.leer()
            elif c == ",":
                reproduciendo = False
                video.ir(video.t - 1.0 / video.fps - 1e-3)
            elif c and c in "123":
                est["nivel_inicial"] = int(c) - 1
            elif c in ("i", "u", "a", "f"):
                campo = {"i": "t_incipiente", "u": "t_incontrolado", "a": "t_inicio_util", "f": "t_fin_util"}[c]
                est[campo] = None if est[campo] is not None and abs(est[campo] - t_act) < 0.01 else t_act
            elif c == "x":
                est["descartar"] = not est["descartar"]
            elif c == "r":
                est = estado_desde(None)
            elif c in ("n", "\r", "b", "q"):
                if c == "b":
                    avance = -1
                    break
                error = validar(est)
                if error and c == "q":
                    print(f"salida sin guardar este video: {error}")
                    salir = True
                    break
                if error:
                    mensaje = f"| error: {error}"
                    continue
                etiquetas[(nombre, a.anotador)] = fila_etiqueta(nombre, a.anotador, est)
                guardar_etiquetas(a.salida, etiquetas)
                se_guardo = True
                if c == "q":
                    salir = True
                else:
                    avance = 1
                break
        video.cerrar()
        registrar_tiempo(a.tiempos, {"archivo": nombre, "anotador": a.anotador,
                                     "segundos_abierto": round(time.time() - inicio, 1), "reproducciones": reproducciones,
                                     "guardado": "si" if se_guardo else "no",
                                     "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        pos = max(0, pos + avance)
    cv2.destroyAllWindows()
    hechos = sum(1 for v in lista if (v, a.anotador) in etiquetas)
    print(f"guardado en {a.salida}. etiquetados por {a.anotador}: {hechos} de {len(lista)}. tiempos en {a.tiempos}")


if __name__ == "__main__":
    main()
