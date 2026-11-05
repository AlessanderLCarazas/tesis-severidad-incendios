import os
import csv
import time
import zlib
import random
import argparse
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
from preprocesamiento import LectorVideo, buscar_rutas

CLASES = {0: "fuego", 1: "humo"}
VARIANTES = ["original", "dt_constante", "dt_real", "dt_tope2", "dt_solo_ruido"]
SISTEMA = "dt_tope2"


class KalmanFilterDT(KalmanFilterXYAH):
    def __init__(self):
        super().__init__()
        self.dt = 1.0

    def fijar_dt(self, dt, dt_transicion=None):
        self.dt = float(dt)
        f = self.dt if dt_transicion is None else float(dt_transicion)
        for i in range(4):
            self._motion_mat[i, 4 + i] = f

    def multi_predict(self, mean, covariance):
        h = mean[:, 3]
        uno = np.ones_like(h)
        wp, wv = self._std_weight_position, self._std_weight_velocity
        std = np.stack([wp * h, wp * h, 1e-2 * uno, wp * h, wv * h, wv * h, 1e-5 * uno, wv * h], axis=1)
        q = np.zeros((len(h), 8, 8))
        q[:, range(8), range(8)] = np.square(std) * self.dt
        f = self._motion_mat
        mean = np.dot(mean, f.T)
        covariance = np.matmul(np.matmul(f, covariance), f.T) + q
        return mean, covariance

    def predict(self, mean, covariance):
        m, c = self.multi_predict(mean[None], covariance[None])
        return m[0], c[0]


class STrackDT(STrack):
    shared_kalman = KalmanFilterDT()

    @staticmethod
    def multi_predict(stracks):
        if not stracks:
            return
        multi_mean = np.asarray([st.mean for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrackDT.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov


def gmc_xyah(stracks, m):
    if not stracks or m is None:
        return
    r = m[:2, :2]
    escala = float(np.hypot(r[0, 0], r[1, 0]))
    j = np.zeros((8, 8))
    j[0:2, 0:2] = r
    j[2, 2] = 1.0
    j[3, 3] = escala
    j[4:6, 4:6] = r
    j[6, 6] = 1.0
    j[7, 7] = escala
    for st in stracks:
        st.mean = j @ st.mean
        st.mean[:2] += m[:2, 2]
        st.covariance = j @ st.covariance @ j.T


class TrackerDT(BYTETracker):
    track_class = STrackDT

    def __init__(self, args, dt_nominal=0.1, modo="real", dt_max=None, solo_ruido=False, compensar=False):
        super().__init__(args)
        self.dt_nominal = dt_nominal
        self.modo = modo
        self.dt_max = dt_max
        self.solo_ruido = solo_ruido
        self.compensar = compensar
        self.warp = None

    def get_kalmanfilter(self):
        return KalmanFilterDT()

    def multi_predict(self, tracks):
        STrackDT.multi_predict(tracks)

    def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high):
        if self.warp is not None:
            gmc_xyah(strack_pool, self.warp)
            gmc_xyah(unconfirmed, self.warp)

    def update(self, results, img=None, feats=None, dt=None, warp=None, **kwargs):
        self.warp = warp if self.compensar else None
        rel = 1.0 if self.modo == "constante" or dt is None else dt / self.dt_nominal
        if self.dt_max is not None:
            rel = min(rel, self.dt_max)
        transicion = 1.0 if self.solo_ruido else None
        self.kalman_filter.fijar_dt(rel, transicion)
        STrackDT.shared_kalman.fijar_dt(rel, transicion)
        return super().update(results, img, feats, **kwargs)


def crear_trackers(variante, args, dt_nominal, dt_max=2.0):
    compensar = variante.endswith("_cmc")
    base = variante[:-4] if compensar else variante
    if base == "original" and not compensar:
        return {c: BYTETracker(args) for c in CLASES}
    opciones = {"original": dict(modo="constante"), "dt_constante": dict(modo="constante"), "dt_real": dict(modo="real"),
                "dt_tope2": dict(modo="real", dt_max=dt_max), "dt_solo_ruido": dict(modo="real", solo_ruido=True)}[base]
    return {c: TrackerDT(args, dt_nominal, compensar=compensar, **opciones) for c in CLASES}


def submuestrear(muestras, quitar, semilla):
    if quitar <= 0:
        return muestras
    rng = random.Random(semilla)
    res = []
    for m in muestras:
        if res and rng.random() < quitar:
            continue
        dt = m["dt"] if not res else m["t"] - res[-1]["t"]
        res.append(dict(m, dt=dt, k=len(res)))
    return res


def seguir(muestras, variante, args, dt_nominal, dt_max=2.0):
    trackers = crear_trackers(variante, args, dt_nominal, dt_max)
    filas = []
    for m in muestras:
        for c, tr in trackers.items():
            datos = m["det"][m["det"][:, 5] == c] if len(m["det"]) else np.zeros((0, 6), np.float32)
            cajas = Boxes(datos, m["forma"])
            salida = tr.update(cajas, None, dt=m["dt"], warp=m.get("warp")) if variante != "original" else tr.update(cajas, None)
            for x1, y1, x2, y2, tid, conf, cls, idx in salida:
                filas.append({"k": m["k"], "frame": m["frame"], "t": m["t"], "dt": m["dt"], "id": int(tid),
                              "clase": CLASES[int(cls)], "conf": float(conf), "x1": float(x1), "y1": float(y1),
                              "x2": float(x2), "y2": float(y2), "det": int(idx)})
    return filas


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def metricas(filas, muestras, clase, alto_conf, ventana_corte, iou_corte):
    tr = {}
    for f in filas:
        if f["clase"] == clase:
            tr.setdefault(f["id"], []).append(f)
    inicios = sorted((v[0]["t"], i) for i, v in tr.items())
    cortes = 0
    for t0, i in inicios:
        caja0 = [tr[i][0][k] for k in ("x1", "y1", "x2", "y2")]
        for j, v in tr.items():
            if j == i:
                continue
            t_fin = v[-1]["t"]
            if t0 - ventana_corte <= t_fin < t0 and iou([v[-1][k] for k in ("x1", "y1", "x2", "y2")], caja0) >= iou_corte:
                cortes += 1
                break
    duraciones = [v[-1]["t"] - v[0]["t"] for v in tr.values()]
    c = [k for k, v in CLASES.items() if v == clase][0]
    total = sum(int(((m["det"][:, 5] == c) & (m["det"][:, 4] >= alto_conf)).sum()) for m in muestras if len(m["det"]))
    usadas = len({(f["k"], f["det"]) for f in filas if f["clase"] == clase and f["conf"] >= alto_conf})
    return {"trayectorias": len(tr), "cortes": cortes, "cortas": sum(1 for v in tr.values() if len(v) < 3),
            "duracion_media_s": round(float(np.mean(duraciones)), 2) if duraciones else 0.0,
            "duracion_max_s": round(float(np.max(duraciones)), 2) if duraciones else 0.0,
            "detecciones": total, "cobertura": round(usadas / total, 3) if total else 0.0}


def color_id(i):
    r = random.Random(i * 7919)
    return (r.randint(60, 255), r.randint(60, 255), r.randint(60, 255))


def escribir_video(ruta_video, ruta_salida, filas, a, quitar, semilla, nombre):
    por_k = {}
    for f in filas:
        por_k.setdefault(f["k"], []).append(f)
    escritor = None
    for m in LectorVideo(ruta_video, a.fps, quitar, semilla):
        img = m.imagen
        esc = min(1.0, a.ancho_video / img.shape[1])
        if esc < 1.0:
            img = cv2.resize(img, None, fx=esc, fy=esc)
        for f in por_k.get(m.k, []):
            p1 = (int(f["x1"] * esc), int(f["y1"] * esc))
            p2 = (int(f["x2"] * esc), int(f["y2"] * esc))
            col = color_id(f["id"])
            cv2.rectangle(img, p1, p2, col, 2)
            txt = f"{f['clase']} id {f['id']} {f['conf']:.2f}"
            cv2.putText(img, txt, (p1[0], max(14, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (p1[0], max(14, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        info = f"{nombre[:50]} | k {m.k} | t {m.t:.2f} s | dt {m.dt * 1000:.0f} ms | bytetrack {SISTEMA}"
        cv2.putText(img, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        if escritor is None:
            escritor = cv2.VideoWriter(ruta_salida, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (img.shape[1], img.shape[0]))
        escritor.write(img)
    if escritor is not None:
        escritor.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lista", default=os.path.join("resultados", "seguimiento", "videos_prueba.csv"))
    p.add_argument("--videos", nargs="+", default=[os.path.join("real_video", "real_video", "fire"),
                                                   os.path.join("real_video", "real_video", "non_fire")])
    p.add_argument("--modelo", default=os.path.join("runs_fuego", "fasdd_v8s_full", "weights", "best.pt"))
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--img", type=int, default=640)
    p.add_argument("--quitar", type=float, nargs="+", default=[0.0, 0.5])
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--track_high_thresh", type=float, default=0.25)
    p.add_argument("--track_low_thresh", type=float, default=0.1)
    p.add_argument("--new_track_thresh", type=float, default=0.25)
    p.add_argument("--track_buffer", type=int, default=30)
    p.add_argument("--dt_max", type=float, default=2.0)
    p.add_argument("--match_thresh", type=float, default=0.8)
    p.add_argument("--ventana_corte", type=float, default=3.0)
    p.add_argument("--iou_corte", type=float, default=0.3)
    p.add_argument("--salida", default=os.path.join("resultados", "seguimiento"))
    p.add_argument("--salida_videos", default=os.path.join("runs_fuego", "seguimiento"))
    p.add_argument("--ancho_video", type=int, default=1280)
    a = p.parse_args()

    targs = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=a.track_high_thresh,
                            track_low_thresh=a.track_low_thresh, new_track_thresh=a.new_track_thresh,
                            track_buffer=a.track_buffer, match_thresh=a.match_thresh, fuse_score=True)
    with open(a.lista, encoding="utf-8") as f:
        lista = [r["archivo"] for r in csv.DictReader(f)]
    rutas = buscar_rutas(a.videos)
    os.makedirs(a.salida, exist_ok=True)
    os.makedirs(a.salida_videos, exist_ok=True)
    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    modelo = YOLO(a.modelo)
    dt_nominal = 1.0 / a.fps

    campos_tr = ["archivo", "quitar", "variante", "k", "frame", "t", "dt", "id", "clase", "conf", "x1", "y1", "x2", "y2", "det"]
    campos_cmp = ["archivo", "quitar", "variante", "clase", "muestras", "trayectorias", "cortes", "cortas",
                  "duracion_media_s", "duracion_max_s", "detecciones", "cobertura", "ms_por_muestra"]
    f_tr = open(os.path.join(a.salida, "trayectorias.csv"), "w", newline="", encoding="utf-8")
    f_cmp = open(os.path.join(a.salida, "comparacion.csv"), "w", newline="", encoding="utf-8")
    w_tr = csv.DictWriter(f_tr, fieldnames=campos_tr)
    w_cmp = csv.DictWriter(f_cmp, fieldnames=campos_cmp)
    w_tr.writeheader()
    w_cmp.writeheader()
    for n, nombre in enumerate(lista, 1):
        if nombre not in rutas:
            print(f"no encontrado: {nombre}")
            continue
        semilla = a.semilla + zlib.crc32(nombre.encode("utf-8"))
        inicio = time.time()
        base_muestras = []
        for m in LectorVideo(rutas[nombre], a.fps, 0.0, semilla):
            r = modelo.predict(m.imagen, imgsz=a.img, conf=a.conf, device=dispositivo, verbose=False)[0]
            det = r.boxes.data.cpu().numpy() if r.boxes is not None else np.zeros((0, 6), np.float32)
            base_muestras.append({"k": m.k, "frame": m.frame, "t": m.t, "dt": m.dt, "det": det, "forma": m.imagen.shape[:2]})
        seg_det = time.time() - inicio
        print(f"[{n}/{len(lista)}] {nombre[:50]} | muestras {len(base_muestras)} | deteccion {seg_det:.1f} s")
        for quitar in a.quitar:
            muestras = submuestrear(base_muestras, quitar, semilla)
            resultados = {}
            for variante in VARIANTES:
                t0 = time.time()
                filas = seguir(muestras, variante, targs, dt_nominal, a.dt_max)
                ms = (time.time() - t0) * 1000 / max(1, len(muestras))
                resultados[variante] = filas
                for f in filas:
                    w_tr.writerow(dict(f, archivo=nombre, quitar=quitar, variante=variante,
                                       t=round(f["t"], 4), dt=round(f["dt"], 4), conf=round(f["conf"], 3),
                                       x1=round(f["x1"], 1), y1=round(f["y1"], 1), x2=round(f["x2"], 1), y2=round(f["y2"], 1)))
                for clase in CLASES.values():
                    met = metricas(filas, muestras, clase, a.track_high_thresh, a.ventana_corte, a.iou_corte)
                    w_cmp.writerow(dict(met, archivo=nombre, quitar=quitar, variante=variante, clase=clase,
                                        muestras=len(muestras), ms_por_muestra=round(ms, 2)))
                    print(f"   quitar {quitar} | {variante:12s} | {clase:5s} | trayectorias {met['trayectorias']:3d}"
                          f" | cortes {met['cortes']:3d} | cortas {met['cortas']:3d} | duracion media {met['duracion_media_s']} s"
                          f" | cobertura {met['cobertura']}")
            clave = lambda fs: sorted((f["k"], f["id"], f["clase"], round(f["x1"], 3), round(f["y1"], 3)) for f in fs)
            igual = clave(resultados["original"]) == clave(resultados["dt_constante"])
            print(f"   quitar {quitar} | dt_constante identico a original: {'si' if igual else 'no'}")
            if quitar == 0.0:
                ruta_salida = os.path.join(a.salida_videos, os.path.splitext(nombre)[0][:60] + f"_{SISTEMA}.mp4")
                escribir_video(rutas[nombre], ruta_salida, resultados[SISTEMA], a, 0.0, semilla, nombre)
                print(f"   video: {ruta_salida}")
        f_tr.flush()
        f_cmp.flush()
    f_tr.close()
    f_cmp.close()
    print(f"trayectorias en: {os.path.join(a.salida, 'trayectorias.csv')}")
    print(f"comparacion en: {os.path.join(a.salida, 'comparacion.csv')}")


if __name__ == "__main__":
    main()
