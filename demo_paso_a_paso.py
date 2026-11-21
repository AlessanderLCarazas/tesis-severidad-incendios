import os
import json
import pickle
import argparse
from functools import lru_cache
import cv2
import numpy as np
import torch
import matplotlib
from PIL import Image, ImageDraw, ImageFont
import sistema as sis
from camara import MovimientoCamara
from vector_escena import VectorEscena, COLUMNAS
from descriptores_v2 import DescriptoresV2
from descriptores_v3 import detecciones_fotograma
from preprocesamiento import buscar_rutas
from secuencias import leer_csv

ANCHO, ALTO = 1280, 720
NIVELES = ["controlado", "incipiente", "incontrolado"]
COLOR_NIVEL = ["#1baf7a", "#eda100", "#e5484d"]
C = {"fondo": "#15171c", "panel": "#1f232a", "borde": "#3a3f48", "rejilla": "#2c3139", "tinta": "#f2f2f0", "tinta2": "#a9adb5",
     "tinta3": "#7d828c", "acento": "#4b93ec", "fuego": "#ffb02e", "humo": "#c8ccd2", "riesgo": "#f2c94c", "alerta": "#e5484d",
     "inlier": "#2fd18f", "outlier": "#ff5c61"}
ETAPAS = ["Entrada", "Preproceso", "Detección", "Representación", "Temporal", "Riesgo", "Decisión", "Salida"]
COLS_VECTOR = [("d1_max", "d1 confianza (máx.)"), ("d2_max", "d2 área (máx.)"), ("d3_max", "d3 tasa de crecimiento (máx.)"),
               ("d7_max", "d7 velocidad del centro (máx.)"), ("d8_media", "d8 persistencia (media)"), ("d10_max", "d10 confianza de humo (máx.)"),
               ("d11_max", "d11 área de humo (máx.)"), ("n_tray", "trayectorias activas"), ("area_det_fuego", "área detectada de fuego"),
               ("area_det_humo", "área detectada de humo")]
CAPTURA = {}
ZONAS_DIFUMINAR = {"Jeder dritte Brand entsteht durch Elektrizität.mp4": [(1660, 806, 1864, 1080)]}
NOMBRE_CONJUNTO = {"prueba": "prueba", "validacion": "validación", "calibracion": "calibración"}


class CamaraDemo(MovimientoCamara):
    def actualizar(self, imagen, cajas):
        prev, prev_mascara = self.prev, self.prev_mascara
        res = super().actualizar(imagen, cajas)
        vis = None
        if prev is not None:
            pts = cv2.goodFeaturesToTrack(prev, mask=prev_mascara, **self.params)
            if pts is not None and len(pts) >= self.min_puntos:
                sig, estado, _ = cv2.calcOpticalFlowPyrLK(prev, self.prev, pts, None)
                ok = estado.reshape(-1) == 1
                p0, p1 = pts.reshape(-1, 2)[ok], sig.reshape(-1, 2)[ok]
                inl = np.zeros(len(p0), bool)
                if len(p0) >= self.min_puntos:
                    _, m = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=self.umbral_ransac,
                                                       confidence=self.confianza)
                    if m is not None:
                        inl = m.reshape(-1).astype(bool)
                vis = (p0 / self.factor, p1 / self.factor, inl)
        CAPTURA["camara"] = dict(res, vis=vis, cajas=[tuple(c) for c in cajas])
        return res


class VectorDemo(VectorEscena):
    def paso(self, t, trayectorias, detecciones):
        v = super().paso(t, trayectorias, detecciones)
        CAPTURA["x"] = v.copy()
        return v


class DescriptoresDemo(DescriptoresV2):
    def paso(self, k, t, dt, mov, fuego, det):
        filas = super().paso(k, t, dt, mov, fuego, det)
        CAPTURA["filas"] = [{"id": int(f["id"]), "d": list(f["d"])} for f in filas]
        return filas


def detecciones_demo(det, *args):
    CAPTURA["det"] = np.array(det, np.float32).reshape(-1, 6) if len(det) else np.zeros((0, 6), np.float32)
    return detecciones_fotograma(det, *args)


def jpg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def decod(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def difuminar(img, zonas, sigma, ancho_fuente):
    if not zonas:
        return img
    f = img.shape[1] / float(ancho_fuente)
    for x0, y0, x1, y1 in zonas:
        xa, ya = max(0, int(np.floor(x0 * f))), max(0, int(np.floor(y0 * f)))
        xb, yb = min(img.shape[1], int(np.ceil(x1 * f))), min(img.shape[0], int(np.ceil(y1 * f)))
        if xb > xa and yb > ya:
            img[ya:yb, xa:xb] = cv2.GaussianBlur(img[ya:yb, xa:xb], (0, 0), sigma * f)
    return img


def texto_conjunto(a):
    fila = next((r for r in leer_csv(a.particion) if r["archivo"] == a.video), None)
    if fila is not None and fila["conjunto"] != a.conjunto:
        raise SystemExit(f"el video esta en {fila['conjunto']}, no en {a.conjunto}")
    if a.conjunto == "prueba":
        return "no se usó para entrenar, validar ni calibrar"
    if a.conjunto == "validacion":
        return "no se usó para entrenar"
    if fila is not None and fila.get("calibra_umbral") == "si":
        return "no se usó para entrenar; sí para fijar el umbral de alerta"
    return "no se usó para entrenar ni para fijar el umbral de alerta"


def tiempos_fuente(ruta, n_imagenes):
    cap = cv2.VideoCapture(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS)
    paso = 1.0 / (fps if fps and 0 < fps <= 240 else 30.0)
    ts, imgs, idx, t_prev = [], [], -1, None
    while cap.grab():
        idx += 1
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = t_ms / 1000.0 if (t_ms > 0 or idx == 0) else idx * paso
        t = max(t, 0.0)
        if t_prev is not None and t <= t_prev:
            t = t_prev + paso
        t_prev = t
        ts.append(t)
        if idx < n_imagenes:
            ok, im = cap.retrieve()
            if ok:
                imgs.append(jpg(im))
    ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return np.array(ts), fps, imgs, ancho, alto


def capturar(a, config, ruta, nombre):
    sis.MovimientoCamara, sis.VectorEscena, sis.DescriptoresV2, sis.detecciones_fotograma = CamaraDemo, VectorDemo, DescriptoresDemo, detecciones_demo
    sistema = sis.Sistema(config, [a.corrida])
    modelo = sistema.modelos[0]
    muestras = []
    for m, s in sis.recorrer(sistema, ruta, nombre):
        r = s["resultados"][a.corrida]
        with torch.no_grad():
            g = torch.sigmoid(modelo.red.compuerta(torch.from_numpy(np.asarray(r["c"], np.float32)[None])))[0].numpy()
        h = modelo.h[0] if isinstance(modelo.h, tuple) else modelo.h
        cam = CAPTURA["camara"]
        muestras.append({"k": m.k, "frame": m.frame, "t": float(m.t), "dt": float(m.dt), "jpg": jpg(m.imagen),
                         "fuego": np.array(s["fuego"], np.float64).reshape(-1, 8) if len(s["fuego"]) else np.zeros((0, 8)),
                         "humo": np.array(s["humo"], np.float64).reshape(-1, 8) if len(s["humo"]) else np.zeros((0, 8)),
                         "det": CAPTURA["det"].copy(), "corte": s["corte"], "tramo": s["tramo"],
                         "camara": {k: cam[k] for k in ("s", "rot", "tx", "ty", "inliers", "valido", "puntos")}, "vis": cam["vis"], "cajas": cam["cajas"],
                         "filas": CAPTURA["filas"], "x": CAPTURA["x"].copy(), "xn": np.asarray(r["x"], np.float64).copy(),
                         "cn": np.asarray(r["c"], np.float64).copy(), "p1": r["p1"], "p2": r["p2"], "riesgo": r["riesgo"], "tau": r["tau"],
                         "crudo": int(r["crudo"]), "nivel": int(r["nivel"]), "alerta": bool(r["alerta"]), "g": g.astype(np.float64),
                         "h": h.reshape(-1).numpy().astype(np.float64).copy()})
        if len(muestras) % 50 == 0:
            print(f"{len(muestras)} fotogramas procesados", flush=True)
    p = sistema.yolo.predictor
    zancada = p.model.stride
    zancada = int(zancada.max()) if hasattr(zancada, "max") else int(zancada)
    ts, fps, imgs, ancho, alto = tiempos_fuente(ruta, a.fotogramas_entrada)
    return {"nombre": nombre, "corrida": a.corrida, "tipo": modelo.tipo, "muestras": muestras, "fuente_t": ts, "fps": fps, "fuente_jpg": imgs,
            "ancho": ancho, "alto": alto, "imgsz": list(p.imgsz), "lb_auto": bool(p.args.rect and p.model.format == "pt"), "zancada": zancada,
            "umbral": float(modelo.umbral), "factor": float(modelo.factor), "permanencia": float(modelo.permanencia)}


def control(a, D):
    ruta = os.path.join(a.salidas, f"{a.conjunto}.npz")
    if not os.path.exists(ruta):
        return {"comparado": False}
    d = np.load(ruta)
    i = np.flatnonzero(d["archivo"] == D["nombre"])
    i = i[np.argsort(d["t"][i], kind="stable")]
    vivo = {k: np.array([m[k] for m in D["muestras"]], np.float64) for k in ("t", "p1", "p2", "riesgo", "tau")}
    if len(i) != len(vivo["t"]):
        return {"comparado": False, "fotogramas_vivo": len(vivo["t"]), "fotogramas_guardados": int(len(i))}
    dif = {"t": float(np.abs(vivo["t"] - d["t"][i]).max())}
    for k in ("p1", "p2", "riesgo", "tau"):
        dif[k] = float(np.abs(vivo[k] - d[f"{D['corrida']}__{k}"][i].astype(np.float64)).max())
    return {"comparado": True, "fotogramas": int(len(i)), "diferencia_maxima": dif}


def segmentos(ts, niveles, t_fin):
    seg, inicio, nv = [], ts[0], niveles[0]
    for t, n in zip(ts[1:], niveles[1:]):
        if n != nv:
            seg.append((float(inicio), float(t), int(nv)))
            inicio, nv = t, n
    seg.append((float(inicio), float(t_fin), int(nv)))
    return seg


def num(x, d=1):
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def texto_segmentos(seg):
    return ", ".join(f"({num(t0)} s; {num(t1)} s; {NIVELES[n]})" for t0, t1, n in seg)


def rgb(h):
    h = C.get(h, h).lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def bgr(h):
    r, g, b = rgb(h)
    return (b, g, r)


def rampa(colores, n=256):
    pts = np.array([rgb(c)[::-1] for c in colores], np.float64)
    x = np.linspace(0, len(colores) - 1, n)
    j = np.minimum(x.astype(int), len(colores) - 2)
    f = (x - j)[:, None]
    return np.clip(pts[j] * (1 - f) + pts[j + 1] * f, 0, 255).astype(np.uint8)


@lru_cache(maxsize=None)
def fuente(tam, negrita=False):
    nombre = "DejaVuSans-Bold.ttf" if negrita else "DejaVuSans.ttf"
    return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", nombre), tam)


class Lienzo:
    def __init__(self, fondo="fondo"):
        self.img = np.full((ALTO, ANCHO, 3), bgr(fondo), np.uint8)
        self.textos = []

    def rect(self, x0, y0, x1, y1, color, grosor=-1):
        cv2.rectangle(self.img, (int(x0), int(y0)), (int(x1), int(y1)), bgr(color), grosor, cv2.LINE_AA)

    def velo(self, x0, y0, x1, y1, color, alfa):
        x0, y0, x1, y1 = [int(v) for v in (max(0, x0), max(0, y0), min(ANCHO, x1), min(ALTO, y1))]
        if x1 > x0 and y1 > y0:
            sub = self.img[y0:y1, x0:x1].astype(np.float32)
            self.img[y0:y1, x0:x1] = (sub * (1 - alfa) + np.array(bgr(color), np.float32) * alfa).astype(np.uint8)

    def linea(self, x0, y0, x1, y1, color, grosor=1):
        cv2.line(self.img, (int(x0), int(y0)), (int(x1), int(y1)), bgr(color), grosor, cv2.LINE_AA)

    def discontinua(self, x0, y0, x1, y1, color, grosor=1, guion=7, hueco=5):
        largo = max(1.0, float(np.hypot(x1 - x0, y1 - y0)))
        p = 0.0
        while p < largo:
            q = min(largo, p + guion)
            self.linea(x0 + (x1 - x0) * p / largo, y0 + (y1 - y0) * p / largo, x0 + (x1 - x0) * q / largo, y0 + (y1 - y0) * q / largo, color, grosor)
            p += guion + hueco

    def polilinea(self, pts, color, grosor=2):
        if len(pts) > 1:
            cv2.polylines(self.img, [np.round(np.array(pts)).astype(np.int32)], False, bgr(color), grosor, cv2.LINE_AA)

    def circulo(self, x, y, r, color, grosor=-1):
        cv2.circle(self.img, (int(round(x)), int(round(y))), int(r), bgr(color), grosor, cv2.LINE_AA)

    def flecha(self, x0, y0, x1, y1, color="tinta2", grosor=2):
        cv2.arrowedLine(self.img, (int(x0), int(y0)), (int(x1), int(y1)), bgr(color), grosor, cv2.LINE_AA, tipLength=0.3)

    def imagen(self, img, x, y, w, h):
        interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR
        self.img[y:y + h, x:x + w] = cv2.resize(img, (w, h), interpolation=interp)
        return w / img.shape[1]

    def texto(self, s, x, y, tam=18, color="tinta", negrita=False, ancla="la"):
        self.textos.append((s, int(x), int(y), tam, color, negrita, ancla))

    def largo(self, s, tam=18, negrita=False):
        return fuente(tam, negrita).getlength(s)

    def etiqueta(self, s, x, y, tam, color, fondo="#000000", alfa=0.65, pad=4):
        w = self.largo(s, tam)
        self.velo(x, y, x + w + 2 * pad, y + tam + 2 * pad, fondo, alfa)
        self.texto(s, x + pad, y + pad, tam, color)

    def chip(self, s, x, y, tam, color_fondo, color_texto="#ffffff", pad=8):
        w = self.largo(s, tam, True)
        self.rect(x, y, x + w + 2 * pad, y + tam + 2 * pad, color_fondo)
        self.texto(s, x + pad, y + pad, tam, color_texto, True)
        return x + w + 2 * pad

    def final(self):
        im = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(im)
        for s, x, y, tam, color, negrita, ancla in self.textos:
            d.text((x, y), s, font=fuente(tam, negrita), fill=rgb(color), anchor=ancla)
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def marco(lz, etapa, titulo, subtitulo, credito, pie=None):
    if etapa:
        lz.rect(24, 18, 152, 52, "acento")
        lz.texto(f"Etapa {etapa} de 8", 88, 35, 16, "#ffffff", True, "mm")
        lz.texto(titulo, 168, 35, 26, "tinta", True, "lm")
    elif titulo:
        lz.texto(titulo, 24, 35, 26, "tinta", True, "lm")
    if isinstance(subtitulo, (tuple, list)):
        for k, s in enumerate(subtitulo):
            lz.texto(s, 24, 63 + 20 * k, 15, "tinta2", False, "lm")
    elif subtitulo:
        lz.texto(subtitulo, 24, 72, 15, "tinta2", False, "lm")
    if etapa:
        for i, nombre in enumerate(ETAPAS):
            x0 = 24 + i * 153
            if i + 1 == etapa:
                lz.rect(x0, 646, x0 + 147, 678, "acento")
                lz.texto(f"{i + 1} {nombre}", x0 + 73, 662, 14, "#ffffff", True, "mm")
            else:
                lz.rect(x0, 646, x0 + 147, 678, "panel" if i + 1 < etapa else "borde", -1 if i + 1 < etapa else 1)
                lz.texto(f"{i + 1} {nombre}", x0 + 73, 662, 14, "tinta2" if i + 1 < etapa else "tinta3", False, "mm")
    elif pie:
        lz.texto(pie, 24, 662, 15, "tinta2", False, "lm")
    lz.texto("CfC (semilla 42), configuración congelada", 24, 703, 13, "tinta3", False, "lm")
    lz.texto(credito, 1256, 703, 13, "tinta3", False, "rm")


class Grafico:
    def __init__(self, lz, x0, y0, x1, y1, t_max, y_min, y_max, titulo=None, marcas_y=(), formato_y=None, eje_t=True, paso_t=5):
        self.lz, self.x0, self.y0, self.x1, self.y1 = lz, x0, y0, x1, y1
        self.t_max, self.y_min, self.y_max = t_max, y_min, y_max
        lz.rect(x0, y0, x1, y1, "panel")
        for v in marcas_y:
            y = self.py(v)
            lz.linea(x0, y, x1, y, "rejilla")
            lz.texto(formato_y(v) if formato_y else num(v, 1), x0 - 6, y, 13, "tinta2", False, "rm")
        if titulo:
            lz.texto(titulo, x0, y0 - 6, 15, "tinta", True, "lb")
        if eje_t:
            for s in range(0, int(t_max) + 1, paso_t):
                lz.texto(str(s), self.px(s), y1 + 5, 13, "tinta2", False, "mt")
            lz.texto("t (s)", x1, y1 + 22, 13, "tinta2", False, "rt")

    def px(self, t):
        return self.x0 + (self.x1 - self.x0) * float(np.clip(t, 0, self.t_max)) / self.t_max

    def py(self, v):
        return self.y1 - (self.y1 - self.y0) * (float(np.clip(v, self.y_min, self.y_max)) - self.y_min) / (self.y_max - self.y_min)

    def linea(self, ts, vs, color, grosor=2, hasta=None, discontinua=False):
        sel = np.ones(len(ts), bool) if hasta is None else ts <= hasta + 1e-9
        pts = [(self.px(t), self.py(v)) for t, v, s in zip(ts, vs, sel) if s and np.isfinite(v)]
        if discontinua:
            for (xa, ya), (xb, yb) in zip(pts[:-1:2], pts[1::2]):
                self.lz.linea(xa, ya, xb, yb, color, grosor)
        else:
            self.lz.polilinea(pts, color, grosor)

    def niveles(self, ts, vs, t_fin, grosor=5, hasta=None, color=None, conectores=True):
        lim = t_fin if hasta is None else hasta
        for j in range(len(ts)):
            if ts[j] > lim + 1e-9:
                break
            t_sig = min(ts[j + 1] if j + 1 < len(ts) else t_fin, lim + 0.1)
            y = self.py(vs[j])
            self.lz.linea(self.px(ts[j]), y, max(self.px(t_sig), self.px(ts[j]) + 1), y, color or COLOR_NIVEL[int(vs[j])], grosor)
            if conectores and j + 1 < len(ts) and vs[j + 1] != vs[j] and ts[j + 1] <= lim + 1e-9:
                self.lz.linea(self.px(ts[j + 1]), y, self.px(ts[j + 1]), self.py(vs[j + 1]), "tinta3", 1)

    def horizontal(self, v, color, grosor=1, discontinua=False, etiqueta=None):
        y = self.py(v)
        if discontinua:
            self.lz.discontinua(self.x0, y, self.x1, y, color, grosor)
        else:
            self.lz.linea(self.x0, y, self.x1, y, color, grosor)
        if etiqueta:
            self.lz.texto(etiqueta, self.x1 - 4, y - 3, 12, color, False, "rb")

    def vertical(self, t, color, grosor=1, discontinua=False):
        x = self.px(t)
        if discontinua:
            self.lz.discontinua(x, self.y0, x, self.y1, color, grosor)
        else:
            self.lz.linea(x, self.y0, x, self.y1, color, grosor)


def dibujar_cajas(lz, m, x, y, esc, etiquetas=True, ancho=ANCHO, alto=ALTO, reservado=()):
    ocupadas = list(reservado)

    def caja(f):
        return max(x, x + f[0] * esc), max(y, y + f[1] * esc), min(x + ancho - 1, x + f[2] * esc), min(y + alto - 1, y + f[3] * esc)

    def etiqueta(s, xe, ye, tam, color):
        w, h = lz.largo(s, tam) + 8, tam + 8
        while any(xe < b[2] and b[0] < xe + w and ye < b[3] and b[1] < ye + h for b in ocupadas) and ye + h < y + alto:
            ye += 2
        ocupadas.append((xe, ye, xe + w, ye + h))
        lz.etiqueta(s, xe, ye, tam, color)

    for f in m["humo"]:
        x1, y1, x2, y2 = caja(f)
        lz.rect(x1, y1, x2, y2, "humo", 1)
        if etiquetas:
            etiqueta(f"humo {int(f[4])}", x1 + 2, max(y, y2 - 22), 12, "humo")
    for f in m["fuego"]:
        x1, y1, x2, y2 = caja(f)
        lz.rect(x1, y1, x2, y2, "fuego", 2)
        if etiquetas:
            etiqueta(f"fuego {int(f[4])} · {num(f[5], 2)}", x1, max(y, y1 - 22), 13, "fuego")


def edades(muestras):
    nacimiento, salida = {}, []
    for m in muestras:
        e = {}
        for f in m["fuego"]:
            clave = (m["tramo"], int(f[4]))
            nacimiento.setdefault(clave, m["t"])
            e[int(f[4])] = m["t"] - nacimiento[clave]
        salida.append(e)
    return salida


class Demo:
    def __init__(self, D, a):
        self.D, self.a = D, a
        self.M = D["muestras"]
        self.T = len(self.M)
        self.t = np.array([m["t"] for m in self.M])
        self.t_fin = float(D["fuente_t"][-1] + 1.0 / D["fps"])
        self.p1 = np.array([m["p1"] for m in self.M])
        self.p2 = np.array([m["p2"] for m in self.M])
        self.P = np.column_stack([1 - self.p1, self.p1 - np.minimum(self.p1, self.p2), np.minimum(self.p1, self.p2)])
        self.riesgo = np.array([m["riesgo"] for m in self.M])
        self.tau = np.array([m["tau"] for m in self.M])
        self.crudo = np.array([m["crudo"] for m in self.M])
        self.nivel = np.array([m["nivel"] for m in self.M])
        self.alerta = np.array([m["alerta"] for m in self.M])
        self.G = np.array([m["g"] for m in self.M])
        self.H = np.array([m["h"] for m in self.M])
        self.hmax = float(np.percentile(np.abs(self.H), 99)) or 1.0
        self.seg = segmentos(self.t, self.nivel, self.t_fin)
        self.t_alerta = float(self.t[np.argmax(self.alerta)]) if self.alerta.any() else None
        self.edad = edades(self.M)
        self.cortes = sum(1 for m in self.M if m["corte"])
        self.n = int(round(a.escena_s * a.fps))
        self.cred = a.credito
        self.lut_g = rampa(["#22262e", "#2a78d6", "#bcd8fa"])
        self.lut_h = rampa(["#eb6834", "#2a2e36", "#4b93ec"])
        self.nombre_modelo = {"cfc": "CfC", "gru": "GRU", "lstm": "LSTM"}.get(D["tipo"], D["tipo"])

    def vis(self, b):
        return difuminar(decod(b), self.a.zonas, self.a.sigma_difuminado, self.D["ancho"])

    def img(self, j):
        return self.vis(self.M[j]["jpg"])

    def indice(self, t):
        return int(min(self.T - 1, np.searchsorted(self.t, t - 1e-9)))

    def resumen(self):
        partes = []
        niveles = [n for _, _, n in self.seg]
        if len(set(niveles)) == 1:
            partes.append(f"La {self.nombre_modelo} marca {NIVELES[niveles[0]]} durante todo el video")
        else:
            partes.append(f"La {self.nombre_modelo} cambia de nivel {len(self.seg) - 1} veces")
        if self.t_alerta is None:
            partes.append("y no alerta.")
        else:
            partes.append(f"y alerta por primera vez a los {num(self.t_alerta)} s.")
        return " ".join(partes)

    def video_pequeno(self, lz, j, x=24, y=96, w=400, h=225, etiqueta=None):
        esc = lz.imagen(self.img(j), x, y, w, h)
        if etiqueta:
            lz.etiqueta(etiqueta, x + 6, y + 6, 15, "tinta")
        return esc

    def titulo(self, n):
        j = self.indice(self.a.t_portada if self.a.t_portada >= 0 else 0.55 * self.t_fin)
        conjunto = NOMBRE_CONJUNTO[self.a.conjunto]
        for _ in range(n):
            lz = Lienzo()
            lz.texto("Detección temprana y clasificación de la severidad", 640, 112, 34, "tinta", True, "mm")
            lz.texto("de incendios en video mediante localización temporal", 640, 158, 34, "tinta", True, "mm")
            lz.texto(f"Demo: un video de {conjunto} por las 8 etapas del sistema", 640, 214, 22, "acento", True, "mm")
            lz.imagen(self.img(j), 400, 252, 480, 270)
            lz.texto(f"Modelo temporal: {self.nombre_modelo} (semilla 42), configuración congelada, la misma cadena del sistema en vivo (sistema.py)",
                     640, 550, 16, "tinta2", False, "mm")
            lz.texto(f"Video {os.path.splitext(self.D['nombre'])[0]}: {self.D['ancho']} × {self.D['alto']}, {num(self.D['fps'], 2)} fps, {num(self.t_fin)} s",
                     640, 578, 16, "tinta2", False, "mm")
            lz.texto(f"Conjunto de {conjunto}: {self.a.texto_conjunto}", 640, 604, 16, "tinta2", False, "mm")
            marco(lz, None, None, None, self.cred, "Parte 1: paso a paso · Parte 2: vista completa")
            yield lz.final()

    def tarjeta(self, n, linea1, linea2):
        for _ in range(n):
            lz = Lienzo()
            lz.texto(linea1, 640, 320, 34, "tinta", True, "mm")
            lz.texto(linea2, 640, 372, 18, "tinta2", False, "mm")
            marco(lz, None, None, None, self.cred)
            yield lz.final()

    def cierre(self, n):
        palabras, lineas, actual = self.resumen().split(), [], ""
        lz0 = Lienzo()
        for w in palabras:
            prueba = (actual + " " + w).strip()
            if lz0.largo(prueba, 24, True) > 1100 and actual:
                lineas.append(actual)
                actual = w
            else:
                actual = prueba
        lineas.append(actual)
        j = self.T - 1
        for _ in range(n):
            lz = Lienzo()
            lz.imagen(self.img(j), 460, 110, 360, 203)
            for r, s in enumerate(lineas):
                lz.texto(s, 640, 370 + r * 38, 24, "tinta", True, "mm")
            y = 370 + len(lineas) * 38 + 20
            lz.texto("Código: demo_paso_a_paso.py sobre la cadena de sistema.py", 640, y + 16, 16, "tinta3", False, "mm")
            marco(lz, None, None, None, self.cred)
            yield lz.final()

    def etapa1(self, n):
        ts, imgs = self.D["fuente_t"], self.D["fuente_jpg"]
        for i in range(n):
            j = min(i, len(imgs) - 1)
            lz = Lienzo()
            marco(lz, 1, "Entrada: fotogramas con su marca de tiempo",
                  "El sistema recibe pares (I_k, t_k): cada imagen con su marca de tiempo real, leída del archivo (CAP_PROP_POS_MSEC).", self.cred)
            lz.imagen(self.vis(imgs[j]), 24, 96, 720, 405)
            dt = ts[j] - ts[j - 1] if j > 0 else 0.0
            lz.etiqueta(f"fotograma {j} · t = {num(ts[j], 3)} s · Δt = {num(dt, 3)} s", 34, 106, 18, "tinta")
            lz.rect(768, 96, 1256, 501, "panel")
            lz.texto("Marcas de tiempo de la fuente", 788, 124, 18, "tinta", True, "lm")
            for x, s in ((800, "fotograma"), (1010, "t (s)"), (1200, "Δt (s)")):
                lz.texto(s, x, 162, 15, "tinta2", True, "lm" if x == 800 else "rm")
            filas = list(range(max(0, j - 8), j + 1))
            for r, f in enumerate(filas):
                y = 196 + r * 33
                if f == j:
                    lz.rect(784, y - 15, 1240, y + 15, "acento")
                color = "#ffffff" if f == j else "tinta"
                lz.texto(str(f), 800, y, 16, color, f == j, "lm")
                lz.texto(num(ts[f], 3), 1010, y, 16, color, f == j, "rm")
                lz.texto(num(ts[f] - ts[f - 1], 3) if f > 0 else "—", 1200, y, 16, color, f == j, "rm")
            lz.texto(f"Fuente: {self.D['ancho']} × {self.D['alto']} píxeles, {num(self.D['fps'], 2)} fps, {num(self.t_fin)} s ({len(ts)} fotogramas).",
                     24, 528, 16, "tinta", False, "lm")
            lz.texto("Δt se calcula con el tiempo real del archivo, no con un intervalo fijo: si faltan fotogramas, Δt crece.", 24, 558, 16, "tinta2", False, "lm")
            lz.texto("Cámara lenta: un fotograma de la fuente por cuadro de la demo.", 24, 588, 16, "tinta2", False, "lm")
            yield lz.final()

    def etapa2(self, n, t0):
        from ultralytics.data.augment import LetterBox
        letterbox = LetterBox(tuple(self.D["imgsz"]), auto=self.D["lb_auto"], stride=self.D["zancada"])
        j0 = self.indice(t0)
        ts = self.D["fuente_t"]
        for i in range(n):
            j = min(j0 + i, self.T - 1)
            m = self.M[j]
            img = self.img(j)
            lb = letterbox(image=img)
            lz = Lienzo()
            marco(lz, 2, "Preprocesamiento: muestreo a 10 fps y redimensión",
                  "Se procesa 1 de cada 3 fotogramas conservando su marca real; YOLO recibe la imagen con letterbox y valores en [0, 1].", self.cred)
            lz.texto("Fotograma procesado (original)", 24, 100, 16, "tinta", True, "la")
            lz.imagen(img, 24, 126, 352, 198)
            lz.texto(f"{img.shape[1]} × {img.shape[0]} píxeles", 24, 332, 15, "tinta2", False, "la")
            lz.flecha(388, 225, 418, 225)
            hh = int(round(400 * lb.shape[0] / lb.shape[1]))
            lz.texto("Entrada de YOLO (letterbox)", 430, 100, 16, "tinta", True, "la")
            lz.imagen(lb, 430, 126, 400, hh)
            lz.rect(430, 126, 830, 126 + hh, "tinta3", 1)
            lz.texto(f"{lb.shape[1]} × {lb.shape[0]}: lado mayor de 640 y relleno", 430, 134 + hh, 15, "tinta2", False, "la")
            lz.texto("gris mínimo hasta múltiplo de 32; píxeles / 255", 430, 156 + hh, 15, "tinta2", False, "la")
            lz.rect(860, 96, 1256, 396, "panel")
            lz.texto("Muestreo con la marca real", 878, 122, 17, "tinta", True, "lm")
            filas = [("fotograma de la fuente", str(m["frame"])), ("t del procesado", f"{num(m['t'], 3)} s"), ("Δt real", f"{num(m['dt'], 3)} s"),
                     ("fotogramas leídos", str(len(ts))), ("fotogramas procesados", str(self.T)), ("rejilla de muestreo", "100 ms")]
            for r, (k, v) in enumerate(filas):
                y = 164 + r * 36
                lz.texto(k, 878, y, 16, "tinta2", False, "lm")
                lz.texto(v, 1238, y, 16, "tinta", True, "rm")
            x0, x1 = 210, 1236
            ta, tb = m["t"] - 1.6, m["t"] + 0.4
            px = lambda t: x0 + (x1 - x0) * (t - ta) / (tb - ta)
            lz.rect(24, 420, 1256, 628, "panel")
            lz.texto("fuente (" + num(self.D["fps"], 2) + " fps)", 40, 476, 15, "tinta2", False, "lm")
            lz.texto("procesados (10 fps)", 40, 548, 15, "tinta2", False, "lm")
            g = np.ceil(ta * 10) / 10
            while g <= tb:
                lz.linea(px(g), 444, px(g), 584, "rejilla", 1)
                lz.texto(num(g, 1), px(g), 592, 12, "tinta3", False, "mt")
                g = round(g + 0.1, 6)
            for t in ts[(ts >= ta) & (ts <= tb)]:
                lz.linea(px(t), 464, px(t), 488, "tinta3", 2)
            for t in self.t[(self.t >= ta) & (self.t <= m["t"] + 1e-9)]:
                lz.linea(px(t), 464, px(t), 548, "riesgo", 2)
                lz.circulo(px(t), 548, 6, "riesgo")
            lz.linea(px(m["t"]), 440, px(m["t"]), 584, "acento", 2)
            lz.texto("t (s)", 1236, 610, 12, "tinta3", False, "rm")
            yield lz.final()

    def etapa3(self, n, t0):
        j0 = self.indice(t0)
        for i in range(n):
            j = min(j0 + i, self.T - 1)
            m = self.M[j]
            lz = Lienzo()
            marco(lz, 3, "Detección y seguimiento",
                  "YOLOv8s detecta fuego y humo (umbral 0,1) y ByteTrack une las detecciones en trayectorias con un identificador.", self.cred)
            esc = lz.imagen(self.img(j), 24, 96, 720, 405)
            for f in m["fuego"]:
                tid = int(f[4])
                pts = []
                for jj in range(max(0, j - 20), j + 1):
                    for g in self.M[jj]["fuego"]:
                        if int(g[4]) == tid and self.M[jj]["tramo"] == m["tramo"]:
                            pts.append((24 + (g[0] + g[2]) / 2 * esc, 96 + (g[1] + g[3]) / 2 * esc))
                lz.polilinea(pts, "fuego", 2)
            dibujar_cajas(lz, m, 24, 96, esc, ancho=720, alto=405)
            det = m["det"]
            nf, nh = int((det[:, 5] == 0).sum()), int((det[:, 5] == 1).sum())
            nfa, nha = int(((det[:, 5] == 0) & (det[:, 4] >= 0.25)).sum()), int(((det[:, 5] == 1) & (det[:, 4] >= 0.25)).sum())
            lz.rect(768, 96, 1256, 501, "panel")
            lz.texto(f"t = {num(m['t'], 1)} s", 788, 124, 18, "tinta", True, "lm")
            lineas = [(f"detecciones de fuego: {nf} ({nfa} con confianza ≥ 0,25)", "tinta"),
                      (f"detecciones de humo: {nh} ({nha} con confianza ≥ 0,25)", "tinta"),
                      (f"trayectorias activas: {len(m['fuego'])} de fuego y {len(m['humo'])} de humo", "tinta")]
            for r, (s, c) in enumerate(lineas):
                lz.texto(s, 788, 162 + r * 30, 15, c, False, "lm")
            lz.texto("Trayectorias de fuego (identificador · confianza · edad)", 788, 270, 15, "tinta2", True, "lm")
            for r, f in enumerate(sorted(m["fuego"], key=lambda f: -f[5])[:5]):
                tid = int(f[4])
                lz.rect(788, 290 + r * 30, 800, 302 + r * 30, "fuego")
                lz.texto(f"fuego {tid} · {num(f[5], 2)} · {num(self.edad[j].get(tid, 0.0), 1)} s", 810, 296 + r * 30, 15, "tinta", False, "lm")
            lz.rect(788, 452, 812, 468, "fuego", 2)
            lz.texto("fuego", 820, 460, 14, "tinta2", False, "lm")
            lz.rect(890, 452, 914, 468, "humo", 1)
            lz.texto("humo", 922, 460, 14, "tinta2", False, "lm")
            lz.linea(990, 460, 1020, 460, "fuego", 2)
            lz.texto("recorrido del centro (2 s)", 1028, 460, 14, "tinta2", False, "lm")
            lz.texto("ByteTrack por clase, con el Δt real en el filtro de Kalman (tope de 2 intervalos) y la cámara compensada.", 24, 528, 16, "tinta", False, "lm")
            lz.texto(f"Cortes de escena detectados en este video: {self.cortes} (en cada corte se reinicia el seguimiento).", 24, 558, 16, "tinta2", False, "lm")
            yield lz.final()

    def ventana_camara(self, n):
        mov = np.array([np.hypot(m["camara"]["tx"], m["camara"]["ty"]) * m["camara"]["valido"] for m in self.M])
        puntaje = np.convolve(mov, np.ones(n), "valid")
        return int(np.argmax(puntaje))

    def etapa4(self, n, j0):
        idx = {c: COLUMNAS.index(c) for c, _ in COLS_VECTOR}
        for i in range(n):
            j = min(j0 + i, self.T - 1)
            m = self.M[j]
            cam = m["camara"]
            lz = Lienzo()
            marco(lz, 4, "Representación: movimiento de la cámara y vector de escena",
                  "Puntos Shi-Tomasi fuera del fuego, seguidos con Lucas-Kanade; RANSAC estima la similitud entre fotogramas.", self.cred)
            esc = lz.imagen(self.img(j), 24, 96, 720, 405)
            for x1, y1, x2, y2 in m["cajas"]:
                lz.velo(24 + x1 * esc, 96 + y1 * esc, 24 + x2 * esc, 96 + y2 * esc, "#000000", 0.45)
                lz.rect(24 + x1 * esc, 96 + y1 * esc, 24 + x2 * esc, 96 + y2 * esc, "fuego", 1)
            if m["vis"] is not None:
                p0, p1, inl = m["vis"]
                for (xa, ya), (xb, yb), ok in zip(p0, p1, inl):
                    color = "inlier" if ok else "outlier"
                    xs, ys = 24 + xb * esc, 96 + yb * esc
                    lz.linea(xs - 4 * (xb - xa) * esc, ys - 4 * (yb - ya) * esc, xs, ys, color, 1)
                    lz.circulo(xs, ys, 2, color)
            lz.circulo(34, 528, 5, "inlier")
            lz.texto("puntos consistentes (inliers)", 46, 528, 15, "tinta2", False, "lm")
            lz.circulo(300, 528, 5, "outlier")
            lz.texto("descartados por RANSAC", 312, 528, 15, "tinta2", False, "lm")
            lz.texto("trazos ×4 · fuego excluido", 530, 528, 15, "tinta2", False, "lm")
            c = [cam["s"], cam["rot"], cam["tx"] / self.D["ancho"], cam["ty"] / self.D["alto"], cam["inliers"], m["dt"]]
            lz.texto("Contexto de la compuerta (etapa 5): c = [s, rot, tx, ty, inliers, Δt]", 24, 566, 16, "tinta", True, "lm")
            lz.texto(f"= [{num(c[0], 4)}; {num(c[1], 4)}; {num(c[2], 4)}; {num(c[3], 4)}; {num(c[4], 2)}; {num(c[5], 3)}]", 24, 594, 16, "tinta2", False, "lm")
            lz.rect(768, 96, 1256, 262, "panel")
            lz.texto("Movimiento de la cámara", 788, 120, 17, "tinta", True, "lm")
            filas = [("escala s", num(cam["s"], 4)), ("rotación", f"{num(np.degrees(cam['rot']), 2)}°"),
                     ("traslación", f"tx {num(cam['tx'], 1)} px · ty {num(cam['ty'], 1)} px"),
                     ("inliers", f"{num(100 * cam['inliers'], 0)} % de {cam['puntos']} puntos"),
                     ("estimación válida", "sí" if cam["valido"] else "no (identidad)")]
            for r, (k, v) in enumerate(filas):
                lz.texto(k, 788, 150 + r * 25, 15, "tinta2", False, "lm")
                lz.texto(v, 1240, 150 + r * 25, 15, "tinta", True, "rm")
            lz.rect(768, 276, 1256, 628, "panel")
            lz.texto("Vector de escena (34 valores): columnas principales", 788, 298, 16, "tinta", True, "lm")
            lz.texto("barra: valor normalizado (z) · número: valor sin normalizar", 788, 322, 13, "tinta2", False, "lm")
            cx, escala = 1100, 26.0
            lz.linea(cx, 338, cx, 618, "tinta3", 1)
            for r, (col, nombre) in enumerate(COLS_VECTOR):
                y = 352 + r * 27
                z = float(np.clip(m["xn"][idx[col]], -3, 3))
                crudo = m["x"][idx[col]]
                lz.texto(nombre, 788, y, 13, "tinta2", False, "lm")
                lz.rect(min(cx, cx + z * escala), y - 8, max(cx, cx + z * escala) + 1, y + 8, "acento")
                lz.texto(str(int(crudo)) if col.startswith("n_") else num(crudo, 3), 1244, y, 13, "tinta", False, "rm")
            yield lz.final()

    def pasos_acelerados(self):
        paso = int(self.a.aceleracion)
        idx = list(range(0, self.T, paso))
        if idx[-1] != self.T - 1:
            idx.append(self.T - 1)
        return idx

    def mapa(self, lz, M, x0, y0, x1, y1, lut, vmin, vmax, hasta):
        z = np.clip((M[: hasta + 1] - vmin) / (vmax - vmin) * 255, 0, 255).astype(int)
        img = np.zeros((M.shape[1], self.T, 3), np.uint8)
        img[:] = bgr("panel")
        img[:, : hasta + 1] = lut[z].transpose(1, 0, 2)
        lz.img[y0:y1, x0:x1] = cv2.resize(img, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)

    def etapa5(self):
        for j in self.pasos_acelerados():
            m = self.M[j]
            lz = Lienzo()
            marco(lz, 5, f"Procesamiento temporal: compuerta de contexto y {self.nombre_modelo}",
                  f"ṽ = v ⊙ σ(W·c + b) pondera cada columna según la cámara y Δt; la {self.nombre_modelo} actualiza su estado con el Δt real. Vista ×{self.a.aceleracion}.",
                  self.cred)
            self.video_pequeno(lz, j, etiqueta=f"×{self.a.aceleracion} · t = {num(m['t'], 1)} s")
            lz.rect(24, 334, 424, 628, "panel")
            lz.texto("Contexto c (sin normalizar)", 40, 356, 16, "tinta", True, "lm")
            cam = m["camara"]
            filas = [("s", num(cam["s"], 4)), ("rot", f"{num(cam['rot'], 4)} rad"), ("tx", num(cam["tx"] / self.D["ancho"], 4)),
                     ("ty", num(cam["ty"] / self.D["alto"], 4)), ("inliers", num(cam["inliers"], 2)), ("Δt", f"{num(m['dt'], 3)} s")]
            for r, (k, v) in enumerate(filas):
                lz.texto(k, 40, 388 + r * 27, 15, "tinta2", False, "lm")
                lz.texto(v, 408, 388 + r * 27, 15, "tinta", True, "rm")
            lz.texto(f"compuerta media: {num(m['g'].mean(), 2)}", 40, 562, 15, "tinta2", False, "lm")
            lz.texto(f"estado x: 64 unidades, |x| máx. {num(np.abs(m['h']).max(), 2)}", 40, 590, 15, "tinta2", False, "lm")
            gx0, gx1 = 568, 1240
            lz.texto("Compuerta σ(W·c + b): 34 columnas × tiempo", gx0, 106, 15, "tinta", True, "lm")
            self.mapa(lz, self.G, gx0, 118, gx1, 254, self.lut_g, 0.0, 1.0, j)
            for (a0, a1), s in (((0, 13), "máximos"), ((13, 26), "medias"), ((26, 34), "escena")):
                lz.texto(s, gx0 - 8, 118 + (a0 + a1) / 2 * 4, 13, "tinta2", False, "rm")
            lz.texto(f"Estado oculto x de la {self.nombre_modelo}: 64 unidades × tiempo", gx0, 284, 15, "tinta", True, "lm")
            self.mapa(lz, self.H, gx0, 296, gx1, 552, self.lut_h, -self.hmax, self.hmax, j)
            x = gx0 + (gx1 - gx0) * (j + 0.5) / self.T
            lz.linea(x, 114, x, 556, "tinta", 1)
            t_max = self.t_fin
            for s in range(0, int(t_max) + 1, 5):
                xs = gx0 + (gx1 - gx0) * s / t_max
                lz.texto(str(s), xs, 560, 13, "tinta2", False, "mt")
            lz.texto("t (s)", gx1, 578, 13, "tinta2", False, "rt")
            for k, (lut, a0, a1, xb) in enumerate(((self.lut_g, "0", "1", gx0), (self.lut_h, "−", "+", gx0 + 330))):
                barra = np.repeat(lut[None, :, :], 12, axis=0)
                lz.img[606:618, xb + 18:xb + 178] = cv2.resize(barra, (160, 12))
                lz.texto(a0, xb + 10, 612, 13, "tinta2", False, "rm")
                lz.texto(a1, xb + 186, 612, 13, "tinta2", False, "lm")
                lz.texto("compuerta" if k == 0 else "estado x", xb + 214, 612, 13, "tinta2", False, "lm")
            yield lz.final()

    def etapa6(self):
        for j in self.pasos_acelerados():
            m = self.M[j]
            lz = Lienzo()
            marco(lz, 6, "Estimación del riesgo: tres salidas",
                  f"Desde el estado oculto: nivel ordinal (CORAL), riesgo continuo en [0, 1] y tiempo al peligro. Vista ×{self.a.aceleracion}.", self.cred)
            self.video_pequeno(lz, j, etiqueta=f"×{self.a.aceleracion} · t = {num(m['t'], 1)} s")
            lz.rect(24, 334, 424, 628, "panel")
            lz.texto("En este instante", 40, 356, 16, "tinta", True, "lm")
            for r in range(3):
                lz.rect(40, 382 + r * 28, 52, 394 + r * 28, COLOR_NIVEL[r])
                lz.texto(f"P({NIVELES[r]})", 62, 388 + r * 28, 15, "tinta2", False, "lm")
                lz.texto(num(self.P[j, r], 2), 408, 388 + r * 28, 15, "tinta", True, "rm")
            lz.texto("riesgo", 40, 480, 15, "tinta2", False, "lm")
            lz.texto(num(self.riesgo[j], 2), 408, 480, 15, "tinta", True, "rm")
            lz.texto("tiempo al peligro estimado", 40, 508, 15, "tinta2", False, "lm")
            lz.texto(f"{num(self.tau[j], 1)} s", 408, 508, 15, "tinta", True, "rm")
            x0, x1 = 500, 1240
            g = Grafico(lz, x0, 116, x1, 236, self.t_fin, 0, 1, "Probabilidad de cada nivel", (0, 0.5, 1), eje_t=False)
            for r in range(3):
                g.linea(self.t, self.P[:, r], COLOR_NIVEL[r], 2, hasta=m["t"])
            g.vertical(m["t"], "tinta")
            g = Grafico(lz, x0, 276, x1, 386, self.t_fin, 0, 1, "Riesgo continuo", (0, 0.5, 1), eje_t=False)
            g.linea(self.t, self.riesgo, "riesgo", 2, hasta=m["t"])
            g.vertical(m["t"], "tinta")
            tmax = 60.0
            g = Grafico(lz, x0, 426, x1, 588, self.t_fin, 0, tmax, "Tiempo al peligro (s)", (0, 20, 40, 60), lambda v: str(int(v)))
            g.linea(self.t, self.tau, "acento", 2, hasta=m["t"])
            g.vertical(m["t"], "tinta")
            yield lz.final()

    def etapa7(self):
        u, fac = self.D["umbral"], self.D["factor"]
        for j in self.pasos_acelerados():
            m = self.M[j]
            lz = Lienzo()
            marco(lz, 7, "Calibración y decisión: histéresis y alerta conformal",
                  f"Umbral calibrado con predicción conformal por video (α = 0,10); el nivel cambia solo si el nuevo se mantiene 2 s. Vista ×{self.a.aceleracion}.",
                  self.cred)
            self.video_pequeno(lz, j, etiqueta=f"×{self.a.aceleracion} · t = {num(m['t'], 1)} s")
            lz.rect(24, 334, 424, 628, "panel")
            lz.texto("Decisión en este instante", 40, 356, 16, "tinta", True, "lm")
            lz.texto("nivel", 40, 392, 15, "tinta2", False, "lm")
            lz.chip(NIVELES[m["nivel"]], 120, 380, 15, COLOR_NIVEL[m["nivel"]])
            lz.texto("alerta", 40, 432, 15, "tinta2", False, "lm")
            if m["alerta"]:
                lz.chip("ALERTA", 120, 420, 15, "alerta")
            else:
                lz.texto("no", 128, 432, 15, "tinta", True, "lm")
            lz.texto(f"umbral de alerta: {num(u, 3)}", 40, 478, 15, "tinta2", False, "lm")
            lz.texto("(cuantil conformal del riesgo máximo", 40, 502, 14, "tinta3", False, "lm")
            lz.texto("de 52 videos sin fuego o controlados)", 40, 522, 14, "tinta3", False, "lm")
            lz.texto(f"se desactiva bajo {num(u * fac, 3)} durante 2 s", 40, 556, 15, "tinta2", False, "lm")
            if self.t_alerta is not None:
                lz.texto(f"primera alerta: {num(self.t_alerta, 1)} s", 40, 590, 15, "tinta", True, "lm")
            x0, x1 = 540, 1240
            nombres = lambda v: NIVELES[int(round(v))] if abs(v - round(v)) < 1e-6 else ""
            g = Grafico(lz, x0, 116, x1, 300, self.t_fin, -0.5, 2.5, "Nivel por fotograma (umbral 0,5) y con histéresis de 2 s", (0, 1, 2), nombres, eje_t=False)
            g.niveles(self.t, self.crudo, self.t_fin, 2, hasta=m["t"], color="tinta3", conectores=False)
            g.niveles(self.t, self.nivel, self.t_fin, 6, hasta=m["t"])
            g.vertical(m["t"], "tinta")
            lz.linea(x1 - 330, 318, x1 - 306, 318, "tinta3", 2)
            lz.texto("sin histéresis", x1 - 300, 318, 13, "tinta2", False, "lm")
            lz.linea(x1 - 170, 318, x1 - 146, 318, COLOR_NIVEL[2], 6)
            lz.texto("con histéresis", x1 - 140, 318, 13, "tinta2", False, "lm")
            g = Grafico(lz, x0, 350, x1, 588, self.t_fin, 0, 1, "Riesgo continuo y alerta", (0, 0.5, 1))
            for k in range(j + 1):
                if self.alerta[k]:
                    t_sig = self.t[k + 1] if k + 1 < self.T else self.t_fin
                    lz.rect(g.px(self.t[k]), 576, g.px(t_sig) + 1, 586, "alerta")
            lz.texto("alerta activa", x0 + 6, 573, 12, "alerta", False, "lb")
            g.horizontal(u, "alerta", 1, etiqueta=f"umbral {num(u, 3)}")
            g.horizontal(u * fac, "alerta", 1, discontinua=True, etiqueta=f"desactivación {num(u * fac, 3)}")
            g.linea(self.t, self.riesgo, "riesgo", 2, hasta=m["t"])
            g.vertical(m["t"], "tinta")
            yield lz.final()

    def fila_tiempo(self, lz, x0, x1, y, alto, hasta=None, tam=15):
        px = lambda t: x0 + (x1 - x0) * float(np.clip(t, 0, self.t_fin)) / self.t_fin
        lz.texto(f"{self.nombre_modelo} (salida)", x0 - 12, y + alto / 2, tam, "tinta", True, "rm")
        lz.rect(x0, y, x1, y + alto, "panel")
        for t0, t1, n in self.seg:
            lim = t1 if hasta is None else min(t1, hasta)
            if lim <= t0:
                continue
            lz.rect(px(t0) + 1, y + 2, px(lim) - 1, y + alto - 2, COLOR_NIVEL[n])
            if px(lim) - px(t0) > 110:
                lz.texto(NIVELES[n], (px(t0) + px(lim)) / 2, y + alto / 2, tam - 1, "#111111", True, "mm")
        return px

    def etapa8(self, n):
        for i in range(n):
            f = min(1.0, (i + 1) / max(1, n - 10))
            lz = Lienzo()
            marco(lz, 8, "Salida: segmentos (inicio, fin, nivel)",
                  "La secuencia de niveles, ya estabilizada con la histéresis, se convierte en segmentos.", self.cred)
            x0, x1 = 290, 1240
            px = self.fila_tiempo(lz, x0, x1, 150, 80, hasta=self.t_fin * f)
            for s in range(0, int(self.t_fin) + 1, 5):
                lz.texto(str(s), px(s), 240, 13, "tinta2", False, "mt")
            lz.texto("t (s)", x1, 258, 13, "tinta2", False, "rt")
            if self.t_alerta is not None and self.t_alerta <= self.t_fin * f:
                xa = px(self.t_alerta)
                lz.linea(xa, 142, xa, 232, "alerta", 2)
                cv2.fillPoly(lz.img, [np.array([[xa, 142], [xa - 8, 130], [xa + 8, 130]], np.int32)], bgr("alerta"), cv2.LINE_AA)
                lz.texto(f"primera alerta: {num(self.t_alerta, 1)} s", xa + 14, 134, 14, "alerta", True, "lm")
            lz.rect(24, 300, 1256, 520, "panel")
            lz.texto("Segmentos (t_s, t_e, nivel)", 44, 328, 17, "tinta", True, "lm")
            lz.texto(f"{self.nombre_modelo}: {texto_segmentos(self.seg)}", 44, 368, 15, "tinta", False, "lm")
            lz.texto(self.resumen(), 44, 438, 16, "tinta", True, "lm")
            lz.texto(f"Riesgo máximo: {num(self.riesgo.max(), 3)}", 44, 478, 15, "tinta2", False, "lm")
            yield lz.final()

    def vista_completa(self):
        u = self.D["umbral"]
        for j, m in enumerate(self.M):
            lz = Lienzo()
            marco(lz, None, f"Vista completa: el sistema en vivo con la {self.nombre_modelo}",
                  ("Cajas e identificadores de las trayectorias, riesgo y nivel en cada instante; abajo, los segmentos.",
                   "La alerta usa el riesgo continuo; el nivel espera 2 s para confirmar un cambio."),
                  self.cred, "Parte 2: tiempo real")
            esc = lz.imagen(self.img(j), 24, 96, 720, 405)
            dibujar_cajas(lz, m, 24, 96, esc, ancho=720, alto=405, reservado=[(24, 96, 744, 140)])
            lz.velo(24, 96, 744, 140, "#000000", 0.55)
            lz.texto("nivel:", 36, 118, 17, "tinta2", False, "lm")
            lz.chip(NIVELES[m["nivel"]], 96, 104, 16, COLOR_NIVEL[m["nivel"]])
            lz.texto(f"riesgo {num(m['riesgo'], 2)}   ·   t = {num(m['t'], 1)} s", 300, 118, 17, "tinta", False, "lm")
            if m["alerta"]:
                lz.chip("ALERTA", 640, 102, 18, "alerta")
                lz.rect(25, 97, 743, 500, "alerta", 3)
            g = Grafico(lz, 820, 118, 1240, 290, self.t_fin, 0, 1, "Riesgo continuo", (0, 0.5, 1), eje_t=False)
            g.horizontal(u, "alerta", 1, etiqueta=f"umbral {num(u, 3)}")
            g.linea(self.t, self.riesgo, "riesgo", 2, hasta=m["t"])
            g.vertical(m["t"], "tinta")
            nombres = lambda v: NIVELES[int(round(v))][:5] + "." if abs(v - round(v)) < 1e-6 else ""
            g = Grafico(lz, 820, 330, 1240, 470, self.t_fin, -0.5, 2.5, "Nivel (con histéresis)", (0, 1, 2), nombres)
            g.niveles(self.t, self.nivel, self.t_fin, 6, hasta=m["t"])
            g.vertical(m["t"], "tinta")
            px = self.fila_tiempo(lz, 230, 1240, 530, 44, hasta=m["t"], tam=14)
            lz.linea(px(m["t"]), 524, px(m["t"]), 580, "tinta", 2)
            yield lz.final()


class Escritor:
    def __init__(self, ruta, fps, crf, tasa_max):
        import imageio_ffmpeg
        self.gen = imageio_ffmpeg.write_frames(ruta, (ANCHO, ALTO), pix_fmt_in="bgr24", pix_fmt_out="yuv420p", fps=fps, quality=None,
                                               codec="libx264", macro_block_size=16, ffmpeg_log_level="error",
                                               output_params=["-crf", str(crf), "-maxrate", tasa_max, "-bufsize", tasa_max, "-preset", "slow",
                                                              "-movflags", "+faststart", "-profile:v", "high"])
        self.gen.send(None)
        self.n = 0

    def escribir(self, img):
        self.gen.send(np.ascontiguousarray(img))
        self.n += 1

    def cerrar(self):
        self.gen.close()


def guardar_gif(cuadros, ruta, fps_video, ancho, limite_mb):
    intentos = [(2, 128), (2, 96), (2, 64), (3, 96), (3, 64), (4, 64)]
    for paso, colores in intentos:
        sel = cuadros[::paso]
        alto = int(round(sel[0].shape[0] * ancho / sel[0].shape[1]))
        ims = [Image.fromarray(cv2.cvtColor(cv2.resize(c, (ancho, alto), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)) for c in sel]
        muestra = Image.fromarray(np.vstack([np.asarray(ims[k]) for k in np.linspace(0, len(ims) - 1, 6).astype(int)]))
        paleta = muestra.quantize(colors=colores, method=Image.Quantize.MEDIANCUT)
        q = [im.quantize(palette=paleta, dither=Image.Dither.NONE) for im in ims]
        q[0].save(ruta, save_all=True, append_images=q[1:], duration=int(round(1000 * paso / fps_video)), loop=0, optimize=True)
        mb = os.path.getsize(ruta) / 1e6
        print(f"gif: {len(q)} cuadros a {fps_video / paso:.2f} fps, {colores} colores, {mb:.1f} mb", flush=True)
        if mb < limite_mb:
            return {"cuadros": len(q), "fps": fps_video / paso, "colores": colores, "mb": mb, "ancho": ancho, "alto": alto}
    return {"cuadros": len(q), "fps": fps_video / paso, "colores": colores, "mb": mb, "ancho": ancho, "alto": alto, "excede": True}


def guardar_pasos(paneles, vista_final, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    salida = []
    for nombre, img in [(f"etapa_{k + 1}", p) for k, p in enumerate(paneles)] + [("vista_completa", vista_final)]:
        ruta = os.path.join(carpeta, f"{nombre}.png")
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(ruta, optimize=True)
        salida.append({"ruta": ruta, "mb": os.path.getsize(ruta) / 1e6})
    return salida


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="Jeder dritte Brand entsteht durch Elektrizität.mp4")
    p.add_argument("--videos", nargs="+", default=[os.path.join("real_video", "real_video", "fire"), os.path.join("real_video", "real_video", "non_fire")])
    p.add_argument("--corrida", default="cfc_s42")
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    p.add_argument("--salidas", default=os.path.join("runs_fuego", "salidas_congeladas"))
    p.add_argument("--conjunto", default="calibracion", choices=list(NOMBRE_CONJUNTO))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--difuminar", nargs="*", default=None)
    p.add_argument("--sigma_difuminado", type=float, default=24.0)
    p.add_argument("--t_portada", type=float, default=-1.0)
    p.add_argument("--cache", default=os.path.join("runs_fuego", "demo_paso_a_paso"))
    p.add_argument("--rehacer", action="store_true")
    p.add_argument("--fotogramas_entrada", type=int, default=40)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--escena_s", type=float, default=3.5)
    p.add_argument("--aceleracion", type=int, default=-1)
    p.add_argument("--t_preproceso", type=float, default=1.5)
    p.add_argument("--t_deteccion", type=float, default=6.0)
    p.add_argument("--t_camara", type=float, default=-1.0)
    p.add_argument("--crf", type=int, default=24)
    p.add_argument("--tasa_max", default="1400k")
    p.add_argument("--credito", default="")
    p.add_argument("--mp4", default=os.path.join("docs", "demo", "demo_cfc.mp4"))
    p.add_argument("--gif", default=os.path.join("docs", "img", "demo_cfc.gif"))
    p.add_argument("--pasos", default=os.path.join("docs", "demo", "pasos"))
    p.add_argument("--ancho_gif", type=int, default=720)
    p.add_argument("--limite_gif_mb", type=float, default=8.0)
    p.add_argument("--resumen", default=os.path.join("resultados", "demo", "demo_cfc.json"))
    a = p.parse_args()
    base = os.path.splitext(a.video)[0]
    if a.difuminar is None:
        a.zonas = ZONAS_DIFUMINAR.get(a.video, [])
    else:
        a.zonas = [tuple(float(v) for v in z.split(",")) for z in a.difuminar]
    a.credito = a.credito or (f"Video: DetectiumFire (arXiv:2511.02495), archivo {base}, CC BY-NC-SA 4.0"
                              + (", marca comercial difuminada" if a.zonas else ""))
    a.texto_conjunto = texto_conjunto(a)

    with open(a.config, encoding="utf-8") as f:
        config = json.load(f)
    rutas = buscar_rutas(a.videos)
    if a.video not in rutas:
        raise SystemExit(f"video no encontrado: {a.video}")
    os.makedirs(a.cache, exist_ok=True)
    cache = os.path.join(a.cache, f"captura_{base}_{a.corrida}.pkl")
    if os.path.exists(cache) and not a.rehacer:
        with open(cache, "rb") as f:
            D = pickle.load(f)
        print(f"captura reutilizada: {cache}")
    else:
        D = capturar(a, config, rutas[a.video], a.video)
        with open(cache, "wb") as f:
            pickle.dump(D, f)
        print(f"captura guardada: {cache}")
    ctl = control(a, D)
    print(f"control contra las salidas congeladas: {json.dumps(ctl)}")
    if a.aceleracion <= 0:
        a.aceleracion = max(1, int(round(len(D["muestras"]) / (a.escena_s * a.fps))))
    demo = Demo(D, a)
    print(f"{demo.T} fotogramas | segmentos {demo.seg} | primera alerta {demo.t_alerta} | cortes {demo.cortes}")

    for ruta in (a.mp4, a.gif, a.resumen):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
    n = demo.n
    j_cam = demo.indice(a.t_camara) if a.t_camara >= 0 else demo.ventana_camara(n)
    escenas = [("titulo", demo.titulo(int(2 * a.fps)), None), ("etapa 1", demo.etapa1(n), n // 2), ("etapa 2", demo.etapa2(n, a.t_preproceso), n // 2),
               ("etapa 3", demo.etapa3(n, a.t_deteccion), n // 2), ("etapa 4", demo.etapa4(n, j_cam), n // 2), ("etapa 5", demo.etapa5(), -1),
               ("etapa 6", demo.etapa6(), -1), ("etapa 7", demo.etapa7(), -1), ("etapa 8", demo.etapa8(n), -1),
               ("parte 2", demo.tarjeta(int(1.5 * a.fps), "Vista completa", "El mismo video en tiempo real, con todas las salidas a la vez"), None),
               ("vista", demo.vista_completa(), None),
               ("cierre", demo.cierre(int(3 * a.fps)), None)]
    escritor = Escritor(a.mp4, a.fps, a.crf, a.tasa_max)
    paneles, vista = [], []
    for nombre, gen, elegido in escenas:
        cuadros = 0
        ultimo = None
        for i, img in enumerate(gen):
            escritor.escribir(img)
            cuadros += 1
            ultimo = img
            if elegido is not None and elegido >= 0 and i == elegido:
                paneles.append(img)
            if nombre == "vista":
                vista.append(img)
        if elegido == -1:
            paneles.append(ultimo)
        print(f"{nombre}: {cuadros} cuadros", flush=True)
    escritor.cerrar()
    mp4 = {"cuadros": escritor.n, "segundos": escritor.n / a.fps, "mb": os.path.getsize(a.mp4) / 1e6}
    print(f"mp4: {a.mp4} | {mp4['cuadros']} cuadros | {mp4['segundos']:.1f} s | {mp4['mb']:.1f} mb")
    gif = guardar_gif(vista, a.gif, a.fps, a.ancho_gif, a.limite_gif_mb)
    print(f"gif: {a.gif} | {gif['mb']:.1f} mb")
    pasos = guardar_pasos(paneles, vista[-1], a.pasos)
    print(f"pasos: {len(pasos)} imagenes en {a.pasos} | {sum(x['mb'] for x in pasos):.1f} mb")
    resumen = {"video": a.video, "conjunto": a.conjunto, "uso_del_video": a.texto_conjunto, "corrida": a.corrida, "fotogramas": demo.T,
               "difuminado": {"zonas_px_fuente": a.zonas, "sigma_px_fuente": a.sigma_difuminado}, "aceleracion": a.aceleracion, "duracion_s": demo.t_fin, "control": ctl,
               "segmentos": demo.seg, "primera_alerta_s": demo.t_alerta, "umbral_alerta": D["umbral"], "riesgo_maximo": float(demo.riesgo.max()),
               "tau_inicio_s": float(demo.tau[0]), "cortes": demo.cortes, "resumen": demo.resumen(),
               "letterbox": {"imgsz": D["imgsz"], "auto": D["lb_auto"], "zancada": D["zancada"]},
               "ventanas": {"preproceso_t": a.t_preproceso, "deteccion_t": a.t_deteccion, "camara_t": float(demo.t[j_cam])},
               "mp4": dict(mp4, ruta=a.mp4), "gif": dict(gif, ruta=a.gif), "pasos": pasos, "credito": a.credito}
    with open(a.resumen, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=1, ensure_ascii=False)
    print(f"resumen: {a.resumen}")


if __name__ == "__main__":
    main()
