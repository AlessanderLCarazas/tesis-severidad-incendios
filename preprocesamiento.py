import os
import csv
import time
import random
import zlib
import argparse
from collections import namedtuple
import cv2
import numpy as np

Muestra = namedtuple("Muestra", ["k", "frame", "t", "dt", "imagen"])


class LectorVideo:
    def __init__(self, ruta, fps_objetivo=10.0, quitar=0.0, semilla=0):
        self.ruta = ruta
        self.fps_objetivo = fps_objetivo
        self.quitar = quitar
        self.semilla = semilla
        self.fps_nominal = 0.0
        self.ancho = 0
        self.alto = 0
        self.leidos = 0
        self.corregidos = 0
        self.quitados = 0

    def __iter__(self):
        cap = cv2.VideoCapture(self.ruta)
        if not cap.isOpened():
            raise IOError(f"no se pudo abrir el video: {self.ruta}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps_nominal = fps if fps and 0 < fps <= 240 else 0.0
        self.ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        paso_fuente = 1.0 / (self.fps_nominal or 30.0)
        paso = 1.0 / self.fps_objetivo
        tolerancia = 0.5 * paso_fuente
        rng = random.Random(self.semilla)
        siguiente = None
        t_prev = None
        t_ultimo = None
        k = 0
        idx = -1
        self.leidos = self.corregidos = self.quitados = 0
        try:
            while cap.grab():
                idx += 1
                self.leidos += 1
                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = t_ms / 1000.0 if (t_ms > 0 or idx == 0) else idx * paso_fuente
                t = max(t, 0.0)
                if t_prev is not None and t <= t_prev:
                    t = t_prev + paso_fuente
                    self.corregidos += 1
                t_prev = t
                if siguiente is not None and t < siguiente - tolerancia:
                    continue
                siguiente = t + paso if siguiente is None else siguiente + paso
                while siguiente <= t - tolerancia:
                    siguiente += paso
                if k > 0 and self.quitar > 0 and rng.random() < self.quitar:
                    self.quitados += 1
                    continue
                ok, imagen = cap.retrieve()
                if not ok:
                    continue
                dt = paso if t_ultimo is None else t - t_ultimo
                t_ultimo = t
                yield Muestra(k, idx, t, dt, imagen)
                k += 1
        finally:
            cap.release()


def buscar_rutas(carpetas):
    rutas = {}
    for c in carpetas:
        for raiz, _, archivos in os.walk(c):
            for nombre in archivos:
                rutas.setdefault(nombre, os.path.join(raiz, nombre))
    return rutas


def mosaico(muestras, nombre, lado=640, columnas=4):
    from ultralytics.data.augment import LetterBox
    lb = LetterBox(new_shape=(lado, lado), auto=False)
    fotos = []
    for m in muestras:
        img = lb(image=m.imagen)
        img = cv2.resize(img, (lado // 2, lado // 2))
        for y, txt in [(22, f"k={m.k} frame={m.frame}"), (46, f"t={m.t:.3f}s dt={m.dt * 1000:.0f}ms")]:
            cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        fotos.append(img)
    while len(fotos) % columnas:
        fotos.append(np.zeros_like(fotos[0]))
    filas = [np.hstack(fotos[i:i + columnas]) for i in range(0, len(fotos), columnas)]
    tablero = np.vstack(filas)
    cv2.putText(tablero, nombre[:70], (8, tablero.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return tablero


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lista", default=os.path.join("resultados", "preprocesamiento", "videos_prueba.csv"))
    p.add_argument("--videos", nargs="+", default=[os.path.join("real_video", "real_video", "fire"),
                                                   os.path.join("real_video", "real_video", "non_fire")])
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--quitar", type=float, default=0.0)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--limite", type=int, default=0)
    p.add_argument("--salida", default=os.path.join("resultados", "preprocesamiento"))
    p.add_argument("--mosaico", type=int, default=8)
    a = p.parse_args()

    with open(a.lista, encoding="utf-8") as f:
        lista = [r["archivo"] for r in csv.DictReader(f)]
    if a.limite > 0:
        lista = lista[:a.limite]
    rutas = buscar_rutas(a.videos)
    os.makedirs(a.salida, exist_ok=True)
    if a.mosaico:
        os.makedirs(os.path.join(a.salida, "mosaicos"), exist_ok=True)
    sufijo = "" if a.quitar == 0 else f"_quitar{int(a.quitar * 100)}"
    ruta_marcas = os.path.join(a.salida, f"marcas{sufijo}.csv")
    ruta_resumen = os.path.join(a.salida, f"resumen{sufijo}.csv")
    campos = ["archivo", "fps_nominal", "ancho", "alto", "leidos", "muestras", "duracion_s", "fps_efectivo",
              "dt_media_ms", "dt_std_ms", "dt_min_ms", "dt_max_ms", "corregidos", "quitados", "fps_lectura", "seg_proceso"]
    with open(ruta_marcas, "w", newline="", encoding="utf-8") as fm, open(ruta_resumen, "w", newline="", encoding="utf-8") as fr:
        wm = csv.writer(fm)
        wm.writerow(["archivo", "k", "frame", "t_k", "dt_k"])
        wr = csv.DictWriter(fr, fieldnames=campos)
        wr.writeheader()
        for n, nombre in enumerate(lista, 1):
            if nombre not in rutas:
                print(f"no encontrado: {nombre}")
                continue
            lector = LectorVideo(rutas[nombre], a.fps, a.quitar, a.semilla + zlib.crc32(nombre.encode("utf-8")))
            inicio = time.time()
            ts, dts, primeras = [], [], []
            for m in lector:
                wm.writerow([nombre, m.k, m.frame, round(m.t, 4), round(m.dt, 4)])
                ts.append(m.t)
                dts.append(m.dt)
                if len(primeras) < a.mosaico:
                    primeras.append(m)
            seg = time.time() - inicio
            if not ts:
                print(f"sin fotogramas: {nombre}")
                continue
            d = np.array(dts[1:]) * 1000 if len(dts) > 1 else np.array([0.0])
            dur = ts[-1] - ts[0]
            fila = {"archivo": nombre, "fps_nominal": round(lector.fps_nominal, 3), "ancho": lector.ancho, "alto": lector.alto,
                    "leidos": lector.leidos, "muestras": len(ts), "duracion_s": round(dur, 3),
                    "fps_efectivo": round((len(ts) - 1) / dur, 3) if dur > 0 else 0,
                    "dt_media_ms": round(float(d.mean()), 2), "dt_std_ms": round(float(d.std()), 2),
                    "dt_min_ms": round(float(d.min()), 2), "dt_max_ms": round(float(d.max()), 2),
                    "corregidos": lector.corregidos, "quitados": lector.quitados,
                    "fps_lectura": round(lector.leidos / seg, 1) if seg > 0 else 0, "seg_proceso": round(seg, 2)}
            wr.writerow(fila)
            if primeras:
                base = os.path.splitext(nombre)[0][:60]
                cv2.imwrite(os.path.join(a.salida, "mosaicos", base + sufijo + ".jpg"), mosaico(primeras, nombre),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"[{n}/{len(lista)}] {nombre[:45]} | fps {fila['fps_nominal']} -> {fila['fps_efectivo']} | dt {fila['dt_media_ms']}"
                  f" +- {fila['dt_std_ms']} ms (min {fila['dt_min_ms']}, max {fila['dt_max_ms']}) | corregidos {lector.corregidos}"
                  f" | quitados {lector.quitados} | lectura {fila['fps_lectura']} fps")
    print(f"marcas en: {ruta_marcas}")
    print(f"resumen en: {ruta_resumen}")


if __name__ == "__main__":
    main()
