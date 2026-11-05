import os
import csv
import time
import zlib
import argparse
from collections import deque
from types import SimpleNamespace
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from preprocesamiento import LectorVideo, buscar_rutas
from camara import MovimientoCamara
import seguimiento as sg

NOMBRES = ["confianza", "area", "tasa_crecimiento", "crecimiento_relativo", "centro_x", "centro_y", "velocidad_centro",
           "persistencia", "variacion_forma", "confianza_humo", "area_humo", "tasa_humo", "regiones_activas"]
CAMPOS = ["archivo", "id", "k", "t", "dt"] + [f"d{i}" for i in range(1, 14)] + ["s_k", "rot", "tx", "ty", "inliers"]


def a3(m):
    h = np.eye(3)
    h[:2, :] = m
    return h


def region_ampliada(caja, factor, ancho, alto):
    x1, y1, x2, y2 = caja
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * factor, (y2 - y1) * factor
    return max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(float(ancho), cx + w / 2), min(float(alto), cy + h / 2)


def humo_en_region(region, humo, rejilla):
    rx1, ry1, rx2, ry2 = region
    if len(humo) == 0 or rx2 <= rx1 or ry2 <= ry1:
        return 0.0, 0.0
    dentro = [(b, c) for b, c in humo if b[0] < rx2 and b[2] > rx1 and b[1] < ry2 and b[3] > ry1]
    if not dentro:
        return 0.0, 0.0
    xs = rx1 + (np.arange(rejilla) + 0.5) * (rx2 - rx1) / rejilla
    ys = ry1 + (np.arange(rejilla) + 0.5) * (ry2 - ry1) / rejilla
    cubierto = np.zeros((rejilla, rejilla), bool)
    for b, _ in dentro:
        cubierto |= np.outer((ys >= b[1]) & (ys <= b[3]), (xs >= b[0]) & (xs <= b[2]))
    return max(c for _, c in dentro), float(cubierto.mean())


class Descriptores:
    def __init__(self, ancho, alto, factor_humo=2.0, ventana=3.0, conf_humo=0.25, rejilla=48):
        self.ancho, self.alto = ancho, alto
        self.factor_humo, self.ventana, self.conf_humo, self.rejilla = factor_humo, ventana, conf_humo, rejilla
        self.acumulada = [np.eye(3)]
        self.tiempos = deque()
        self.historia = {}

    def paso(self, k, t, dt, mov, fuego, det):
        self.acumulada.append(a3(mov["M"]) @ self.acumulada[-1] if k > 0 else np.eye(3))
        self.tiempos.append(t)
        while self.tiempos and self.tiempos[0] <= t - self.ventana:
            self.tiempos.popleft()
        humo = [((d[0], d[1], d[2], d[3]), float(d[4])) for d in det if int(d[5]) == 1 and d[4] >= self.conf_humo]
        area_frame = float(self.ancho * self.alto)
        filas = []
        for x1, y1, x2, y2, tid, conf, _, _ in fuego:
            tid = int(tid)
            w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
            area = w * h / area_frame
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hist = self.historia.setdefault(tid, {"obs": deque(), "previa": None})
            conf_h, area_h = humo_en_region(region_ampliada((x1, y1, x2, y2), self.factor_humo, self.ancho, self.alto), humo, self.rejilla)
            tasa = rel = vel = tasa_h = 0.0
            if hist["previa"] is not None:
                kp, tp, ap, cxp, cyp, ahp = hist["previa"]
                dtp = max(t - tp, 1e-6)
                h_rel = self.acumulada[-1] @ np.linalg.inv(self.acumulada[kp + 1])
                s = float(np.hypot(h_rel[0, 0], h_rel[1, 0]))
                tasa = (area - s * s * ap) / dtp
                rel = tasa / max(area, 1e-9)
                px, py = (h_rel @ np.array([cxp, cyp, 1.0]))[:2]
                vel = float(np.hypot((cx - px) / self.ancho, (cy - py) / self.alto)) / dtp
                tasa_h = (area_h - ahp) / dtp
            hist["obs"].append((t, w / h))
            while hist["obs"] and hist["obs"][0][0] <= t - self.ventana:
                hist["obs"].popleft()
            persist = len(hist["obs"]) / max(1, len(self.tiempos))
            forma = float(np.var([r for _, r in hist["obs"]])) if len(hist["obs"]) > 1 else 0.0
            hist["previa"] = (k, t, area, cx, cy, area_h)
            filas.append({"id": tid, "d": [float(conf), area, tasa, rel, cx / self.ancho, cy / self.alto, vel, persist, forma,
                                           conf_h, area_h, tasa_h, float(len(fuego))]})
        return filas


def grafico(filas, ruta, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([f["t"] for f in filas])
    fig, ejes = plt.subplots(13, 1, figsize=(10, 20), sharex=True)
    for i, eje in enumerate(ejes):
        eje.plot(t, [f[f"d{i + 1}"] for f in filas], lw=1, marker=".", ms=2)
        eje.set_ylabel(f"d{i + 1}\n{NOMBRES[i]}", fontsize=7, rotation=0, ha="right", va="center")
        eje.grid(alpha=0.3)
        eje.tick_params(labelsize=7)
    ejes[-1].set_xlabel("t (s)")
    ejes[0].set_title(titulo, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lista", default=os.path.join("resultados", "seguimiento", "videos_prueba.csv"))
    p.add_argument("--videos", nargs="+", default=[os.path.join("real_video", "real_video", "fire"),
                                                   os.path.join("real_video", "real_video", "non_fire")])
    p.add_argument("--modelo", default=os.path.join("runs_fuego", "fasdd_v8s_full", "weights", "best.pt"))
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--img", type=int, default=640)
    p.add_argument("--quitar", type=float, default=0.0)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--variante", default="dt_tope2_cmc")
    p.add_argument("--dt_max", type=float, default=2.0)
    p.add_argument("--mascara", choices=["todas", "fuego", "ninguna"], default="fuego")
    p.add_argument("--factor_humo", type=float, default=2.0)
    p.add_argument("--conf_humo", type=float, default=0.25)
    p.add_argument("--ventana", type=float, default=3.0)
    p.add_argument("--salida", default=os.path.join("resultados", "descriptores"))
    p.add_argument("--grafico", default="14294880_3840_2160_24fps.mp4")
    a = p.parse_args()

    targs = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                            track_buffer=30, match_thresh=0.8, fuse_score=True)
    with open(a.lista, encoding="utf-8") as f:
        lista = [r["archivo"] for r in csv.DictReader(f)]
    rutas = buscar_rutas(a.videos)
    os.makedirs(a.salida, exist_ok=True)
    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    modelo = YOLO(a.modelo)
    ruta_csv = os.path.join(a.salida, "descriptores.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        for n, nombre in enumerate(lista, 1):
            if nombre not in rutas:
                print(f"no encontrado: {nombre}")
                continue
            inicio = time.time()
            semilla = a.semilla + zlib.crc32(nombre.encode("utf-8"))
            trackers = sg.crear_trackers(a.variante, targs, 1.0 / a.fps, a.dt_max)
            cam = MovimientoCamara()
            desc = None
            filas_video = []
            muestras = 0
            for m in LectorVideo(rutas[nombre], a.fps, a.quitar, semilla):
                muestras += 1
                alto, ancho = m.imagen.shape[:2]
                if desc is None:
                    desc = Descriptores(ancho, alto, a.factor_humo, a.ventana, a.conf_humo)
                r = modelo.predict(m.imagen, imgsz=a.img, conf=a.conf, device=dispositivo, verbose=False)[0]
                det = r.boxes.data.cpu().numpy() if r.boxes is not None else np.zeros((0, 6), np.float32)
                fuego_det = det[det[:, 5] == 0] if len(det) else det
                mov = cam.actualizar(m.imagen, [tuple(x[:4]) for x in (fuego_det if a.mascara == "fuego" else det if a.mascara == "todas" else [])])
                salidas = {}
                for c, tr in trackers.items():
                    datos = det[det[:, 5] == c] if len(det) else np.zeros((0, 6), np.float32)
                    salidas[c] = tr.update(Boxes(datos, (alto, ancho)), None, dt=m.dt, warp=mov["M"] if m.k > 0 else None)
                for fila in desc.paso(m.k, m.t, m.dt, mov, salidas[0], det):
                    reg = {"archivo": nombre, "id": fila["id"], "k": m.k, "t": round(m.t, 4), "dt": round(m.dt, 4),
                           "s_k": round(mov["s"], 5), "rot": round(mov["rot"], 5), "tx": round(mov["tx"] / ancho, 6),
                           "ty": round(mov["ty"] / alto, 6), "inliers": round(mov["inliers"], 4)}
                    for i, v in enumerate(fila["d"], 1):
                        reg[f"d{i}"] = round(v, 6)
                    w.writerow(reg)
                    filas_video.append(reg)
            ids = {f["id"] for f in filas_video}
            print(f"[{n}/{len(lista)}] {nombre[:50]} | muestras {muestras} | trayectorias de fuego {len(ids)}"
                  f" | filas {len(filas_video)} | {time.time() - inicio:.1f} s")
            if nombre == a.grafico and filas_video:
                por_id = {}
                for f_ in filas_video:
                    por_id.setdefault(f_["id"], []).append(f_)
                tid, sel = max(por_id.items(), key=lambda x: len(x[1]))
                ruta_png = os.path.join(a.salida, f"descriptores_{os.path.splitext(nombre)[0]}_id{tid}.png")
                grafico(sel, ruta_png, f"{nombre} | trayectoria {tid} | {len(sel)} muestras")
                print(f"   grafico de la trayectoria {tid} ({len(sel)} muestras): {ruta_png}")
    print(f"descriptores en: {ruta_csv}")


if __name__ == "__main__":
    main()
