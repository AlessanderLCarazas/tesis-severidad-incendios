import os
import csv
import zlib
import argparse
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from preprocesamiento import LectorVideo, buscar_rutas
import seguimiento as sg

IDENTIDAD = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class MovimientoCamara:
    def __init__(self, ancho_proc=960, max_puntos=400, calidad=0.01, distancia=1, bloque=3, umbral_ransac=3.0,
                 confianza=0.99, min_puntos=10, min_inliers=0.5):
        self.ancho_proc = ancho_proc
        self.params = dict(maxCorners=max_puntos, qualityLevel=calidad, minDistance=distancia, blockSize=bloque,
                           useHarrisDetector=False, k=0.04)
        self.umbral_ransac = umbral_ransac
        self.confianza = confianza
        self.min_puntos = min_puntos
        self.min_inliers = min_inliers
        self.prev = None
        self.prev_mascara = None
        self.factor = 1.0

    def _preparar(self, imagen, cajas):
        self.factor = min(1.0, self.ancho_proc / imagen.shape[1])
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        if self.factor < 1.0:
            gris = cv2.resize(gris, None, fx=self.factor, fy=self.factor, interpolation=cv2.INTER_AREA)
        mascara = np.full(gris.shape, 255, np.uint8)
        for x1, y1, x2, y2 in cajas:
            c0, f0 = max(0, int(x1 * self.factor)), max(0, int(y1 * self.factor))
            c1, f1 = int(np.ceil(x2 * self.factor)), int(np.ceil(y2 * self.factor))
            mascara[f0:f1, c0:c1] = 0
        return gris, mascara

    def actualizar(self, imagen, cajas):
        gris, mascara = self._preparar(imagen, cajas)
        res = {"M": IDENTIDAD.copy(), "s": 1.0, "rot": 0.0, "tx": 0.0, "ty": 0.0, "inliers": 0.0, "puntos": 0,
               "valido": 0, "libre": float((mascara > 0).mean())}
        if self.prev is not None:
            pts = cv2.goodFeaturesToTrack(self.prev, mask=self.prev_mascara, **self.params)
            if pts is not None and len(pts) >= self.min_puntos:
                sig, estado, _ = cv2.calcOpticalFlowPyrLK(self.prev, gris, pts, None)
                ok = estado.reshape(-1) == 1
                p0, p1 = pts.reshape(-1, 2)[ok], sig.reshape(-1, 2)[ok]
                res["puntos"] = int(ok.sum())
                if len(p0) >= self.min_puntos:
                    m, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                                         ransacReprojThreshold=self.umbral_ransac, confidence=self.confianza)
                    if m is not None:
                        res["inliers"] = float(inl.mean()) if inl is not None else 0.0
                        if res["inliers"] >= self.min_inliers:
                            m = m.astype(np.float64)
                            m[:, 2] /= self.factor
                            res.update(M=m, s=float(np.hypot(m[0, 0], m[1, 0])), rot=float(np.arctan2(m[1, 0], m[0, 0])),
                                       tx=float(m[0, 2]), ty=float(m[1, 2]), valido=1)
            res["err_sin"], res["err_con"] = self._error(gris, mascara, res["M"])
        self.prev, self.prev_mascara = gris, mascara
        return res

    def _error(self, gris, mascara, m):
        mp = m.copy()
        mp[:, 2] *= self.factor
        h, w = gris.shape
        alineada = cv2.warpAffine(self.prev, mp, (w, h))
        cubre = cv2.warpAffine(np.full_like(self.prev, 255), mp, (w, h))
        zona = (mascara > 0) & (self.prev_mascara > 0) & (cubre > 0)
        if zona.sum() < 100:
            return float("nan"), float("nan")
        a, b, c = gris.astype(np.float32), self.prev.astype(np.float32), alineada.astype(np.float32)
        return float(np.abs(a - b)[zona].mean()), float(np.abs(a - c)[zona].mean())


def cajas_de(det, mascara):
    if len(det) == 0 or mascara == "ninguna":
        return []
    d = det if mascara == "todas" else det[det[:, 5] == 0]
    return [tuple(x[:4]) for x in d]


def grafico(filas, ruta, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([f["t"] for f in filas])
    fig, ejes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for eje, clave, nombre in zip(ejes, ["tx", "ty", "s", "inliers"], ["tx (px)", "ty (px)", "escala s", "inliers"]):
        eje.plot(t, [f[clave] for f in filas], lw=1)
        eje.set_ylabel(nombre)
        eje.grid(alpha=0.3)
    malos = t[[f["valido"] == 0 for f in filas]]
    for eje in ejes:
        for x in malos[1:]:
            eje.axvline(x, color="red", alpha=0.25, lw=1)
    ejes[-1].set_xlabel("t (s)")
    ejes[0].set_title(titulo)
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
    p.add_argument("--quitar", type=float, nargs="+", default=[0.0, 0.5])
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--mascara", choices=["todas", "fuego", "ninguna"], default="fuego")
    p.add_argument("--ancho_proc", type=int, default=960)
    p.add_argument("--max_puntos", type=int, default=400)
    p.add_argument("--min_inliers", type=float, default=0.5)
    p.add_argument("--dt_max", type=float, default=2.0)
    p.add_argument("--variantes", nargs="+", default=["original", "original_cmc", "dt_tope2", "dt_tope2_cmc"])
    p.add_argument("--salida", default=os.path.join("resultados", "camara"))
    a = p.parse_args()

    targs = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                            track_buffer=30, match_thresh=0.8, fuse_score=True)
    with open(a.lista, encoding="utf-8") as f:
        lista = [r["archivo"] for r in csv.DictReader(f)]
    rutas = buscar_rutas(a.videos)
    os.makedirs(a.salida, exist_ok=True)
    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    modelo = YOLO(a.modelo)
    campos_mov = ["archivo", "quitar", "k", "frame", "t", "dt", "s", "rot", "tx", "ty", "inliers", "puntos", "valido",
                  "libre", "err_sin", "err_con"]
    campos_cmp = ["archivo", "quitar", "variante", "clase", "muestras", "trayectorias", "cortes", "cortas",
                  "duracion_media_s", "duracion_max_s", "detecciones", "cobertura"]
    f_mov = open(os.path.join(a.salida, f"movimiento_{a.mascara}.csv"), "w", newline="", encoding="utf-8")
    f_cmp = open(os.path.join(a.salida, f"comparacion_{a.mascara}.csv"), "w", newline="", encoding="utf-8")
    w_mov, w_cmp = csv.DictWriter(f_mov, fieldnames=campos_mov), csv.DictWriter(f_cmp, fieldnames=campos_cmp)
    w_mov.writeheader()
    w_cmp.writeheader()
    for n, nombre in enumerate(lista, 1):
        if nombre not in rutas:
            print(f"no encontrado: {nombre}")
            continue
        semilla = a.semilla + zlib.crc32(nombre.encode("utf-8"))
        dets = {}
        for quitar in a.quitar:
            cam = MovimientoCamara(a.ancho_proc, a.max_puntos, min_inliers=a.min_inliers)
            muestras, filas_mov = [], []
            for m in LectorVideo(rutas[nombre], a.fps, quitar, semilla):
                if m.frame not in dets:
                    r = modelo.predict(m.imagen, imgsz=a.img, conf=a.conf, device=dispositivo, verbose=False)[0]
                    dets[m.frame] = r.boxes.data.cpu().numpy() if r.boxes is not None else np.zeros((0, 6), np.float32)
                det = dets[m.frame]
                mov = cam.actualizar(m.imagen, cajas_de(det, a.mascara))
                muestras.append({"k": m.k, "frame": m.frame, "t": m.t, "dt": m.dt, "det": det, "forma": m.imagen.shape[:2],
                                 "warp": mov["M"] if m.k > 0 else None})
                fila = {k: mov.get(k, "") for k in campos_mov if k in mov}
                fila.update(archivo=nombre, quitar=quitar, k=m.k, frame=m.frame, t=round(m.t, 4), dt=round(m.dt, 4))
                for k in ["s", "rot", "tx", "ty", "inliers", "libre", "err_sin", "err_con"]:
                    if isinstance(fila.get(k), float):
                        fila[k] = round(fila[k], 5)
                filas_mov.append(fila)
                w_mov.writerow(fila)
            v = [f for f in filas_mov[1:]]
            validos = sum(f["valido"] for f in v)
            pares = [(f["err_sin"], f["err_con"]) for f in v if f["valido"] and not np.isnan(f["err_sin"])]
            mejora = 100 * (1 - np.mean([c for _, c in pares]) / np.mean([s for s, _ in pares])) if pares else float("nan")
            desp = np.mean([np.hypot(f["tx"], f["ty"]) for f in v if f["valido"]]) if validos else 0.0
            print(f"[{n}/{len(lista)}] {nombre[:45]} | quitar {quitar} | muestras {len(muestras)} | validas {validos}/{len(v)}"
                  f" | desplazamiento medio {desp:.1f} px | inliers medio {np.mean([f['inliers'] for f in v]):.2f}"
                  f" | error de alineacion {mejora:.1f} % menor")
            if quitar == 0.0:
                grafico(filas_mov, os.path.join(a.salida, f"movimiento_{os.path.splitext(nombre)[0][:40]}_{a.mascara}.png"),
                        nombre[:70])
            for variante in a.variantes:
                filas = sg.seguir(muestras, variante, targs, 1.0 / a.fps, a.dt_max)
                for clase in sg.CLASES.values():
                    met = sg.metricas(filas, muestras, clase, 0.25, 3.0, 0.3)
                    w_cmp.writerow(dict(met, archivo=nombre, quitar=quitar, variante=variante, clase=clase, muestras=len(muestras)))
                    if clase == "fuego":
                        print(f"   {variante:13s} | fuego | trayectorias {met['trayectorias']:3d} | cortes {met['cortes']:3d}"
                              f" | cortas {met['cortas']:3d} | duracion media {met['duracion_media_s']} s | cobertura {met['cobertura']}")
        f_mov.flush()
        f_cmp.flush()
    f_mov.close()
    f_cmp.close()
    print(f"movimiento en: {os.path.join(a.salida, f'movimiento_{a.mascara}.csv')}")
    print(f"comparacion en: {os.path.join(a.salida, f'comparacion_{a.mascara}.csv')}")


if __name__ == "__main__":
    main()
