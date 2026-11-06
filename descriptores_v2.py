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
from descriptores import NOMBRES, CAMPOS, a3, humo_en_region, grafico
from cortes import DetectorCortes, CAMPOS_CORTES, CAMPOS_RESUMEN, CAMPOS_INDICADORES, resumen_video, abrir_csv
import seguimiento as sg

CAMPOS_V2 = CAMPOS + ["d2_med", "d11_med"]
CAMPOS_LOG = ["archivo", "muestras", "trayectorias", "filas", "segundos"]
EPS = 1e-6


def region_humo(caja, factor, minimo, ancho, alto):
    x1, y1, x2, y2 = caja
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = max((x2 - x1) * factor, minimo * ancho)
    h = max((y2 - y1) * factor, minimo * alto)
    return max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(float(ancho), cx + w / 2), min(float(alto), cy + h / 2)


class DescriptoresV2:
    def __init__(self, ancho, alto, factor_humo=2.0, min_humo=0.2, ventana=3.0, mediana=1.0, min_duracion=1.0,
                 conf_humo=0.25, rejilla=48):
        self.ancho, self.alto = ancho, alto
        self.factor_humo, self.min_humo = factor_humo, min_humo
        self.ventana, self.mediana, self.min_duracion = ventana, mediana, min_duracion
        self.conf_humo, self.rejilla = conf_humo, rejilla
        self.acumulada = [np.eye(3)]
        self.escala = [1.0]
        self.tiempos = deque()
        self.historia = {}

    def escala_desde(self, kj):
        return self.escala[-1] / self.escala[kj + 1]

    def paso(self, k, t, dt, mov, fuego, det):
        if k > 0:
            self.acumulada.append(a3(mov["M"]) @ self.acumulada[-1])
            self.escala.append(self.escala[-1] * float(mov["s"]))
        else:
            self.acumulada.append(np.eye(3))
            self.escala.append(1.0)
        self.tiempos.append(t)
        while self.tiempos and self.tiempos[0] <= t - self.ventana:
            self.tiempos.popleft()
        humo = [((d[0], d[1], d[2], d[3]), float(d[4])) for d in det if int(d[5]) == 1 and d[4] >= self.conf_humo]
        area_frame = float(self.ancho * self.alto)
        for fila in fuego:
            self.historia.setdefault(int(fila[4]), {"obs": deque(), "med": deque(), "previa": None, "inicio": t})
        activas = sum(1 for fila in fuego if t - self.historia[int(fila[4])]["inicio"] >= self.min_duracion - EPS)
        filas = []
        for x1, y1, x2, y2, tid, conf, _, _ in fuego:
            tid = int(tid)
            w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
            area = w * h / area_frame
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hist = self.historia[tid]
            conf_h, area_h = humo_en_region(region_humo((x1, y1, x2, y2), self.factor_humo, self.min_humo, self.ancho, self.alto),
                                            humo, self.rejilla)
            while hist["med"] and hist["med"][0][0] <= t - self.mediana + EPS:
                hist["med"].popleft()
            hist["med"].append((t, k, area, area_h))
            a_med = float(np.median([a * self.escala_desde(kj) ** 2 for _, kj, a, _ in hist["med"]]))
            h_med = float(np.median([ah for _, _, _, ah in hist["med"]]))
            tasa = rel = vel = tasa_h = 0.0
            if hist["previa"] is not None:
                kp, tp, amp, cxp, cyp, hmp = hist["previa"]
                dtp = max(t - tp, 1e-6)
                tasa = (a_med - self.escala_desde(kp) ** 2 * amp) / dtp
                rel = tasa / max(a_med, 1e-9)
                h_rel = self.acumulada[-1] @ np.linalg.inv(self.acumulada[kp + 1])
                px, py = (h_rel @ np.array([cxp, cyp, 1.0]))[:2]
                vel = float(np.hypot((cx - px) / self.ancho, (cy - py) / self.alto)) / dtp
                tasa_h = (h_med - hmp) / dtp
            hist["obs"].append((t, w / h))
            while hist["obs"] and hist["obs"][0][0] <= t - self.ventana:
                hist["obs"].popleft()
            persist = len(hist["obs"]) / max(1, len(self.tiempos))
            forma = float(np.var([r for _, r in hist["obs"]])) if len(hist["obs"]) > 1 else 0.0
            hist["previa"] = (k, t, a_med, cx, cy, h_med)
            filas.append({"id": tid, "d": [float(conf), area, tasa, rel, cx / self.ancho, cy / self.alto, vel, persist, forma,
                                           conf_h, area_h, tasa_h, float(activas)], "d2_med": a_med, "d11_med": h_med})
        return filas


def procesar_video(ruta, nombre, a, modelo, dispositivo, targs):
    semilla = a.semilla + zlib.crc32(nombre.encode("utf-8"))
    detector = None if a.sin_cortes else DetectorCortes(a.t_alto, a.t_bajo, a.separacion)
    trackers = cam = desc = None
    filas, cortes, indicadores, inicios = [], [], [], []
    muestras, k0, tramo, t_fin = 0, 0, -1, 0.0
    for m in LectorVideo(ruta, a.fps, a.quitar, semilla):
        muestras += 1
        t_fin = m.t
        alto, ancho = m.imagen.shape[:2]
        r = modelo.predict(m.imagen, imgsz=a.img, conf=a.conf, device=dispositivo, verbose=False)[0]
        det = r.boxes.data.cpu().numpy() if r.boxes is not None else np.zeros((0, 6), np.float32)
        fuego_det = det[det[:, 5] == 0] if len(det) else det
        cajas = [tuple(x[:4]) for x in (fuego_det if a.mascara == "fuego" else det if a.mascara == "todas" else [])]
        h, motivo, mov_det = 0.0, "", None
        if trackers is None:
            reiniciar = True
            if detector is not None:
                detector.actualizar(m.imagen, m.t, 0)
        else:
            mov = cam.actualizar(m.imagen, cajas)
            mov_det = mov
            if detector is not None:
                h, motivo = detector.actualizar(m.imagen, m.t, mov["valido"])
            reiniciar = bool(motivo)
            if motivo:
                cortes.append({"archivo": nombre, "t_corte": round(m.t, 4), "motivo": motivo, "k": m.k, "h": round(h, 4),
                               "valido": mov["valido"], "inliers": round(mov["inliers"], 4)})
        if reiniciar:
            tramo += 1
            inicios.append(m.t)
            k0 = m.k
            trackers = sg.crear_trackers(a.variante, targs, 1.0 / a.fps, a.dt_max)
            cam = MovimientoCamara()
            mov = cam.actualizar(m.imagen, cajas)
            desc = DescriptoresV2(ancho, alto, a.factor_humo, a.min_humo, a.ventana, a.mediana, a.min_duracion, a.conf_humo)
        if detector is not None and a.indicadores:
            indicadores.append({"archivo": nombre, "k": m.k, "t": round(m.t, 4), "h": round(h, 4),
                                "valido": mov_det["valido"] if mov_det else 0,
                                "inliers": round(mov_det["inliers"], 4) if mov_det else 0.0,
                                "puntos": mov_det["puntos"] if mov_det else 0, "corte": motivo})
        kl = m.k - k0
        salidas = {}
        for c, tr in trackers.items():
            datos = det[det[:, 5] == c] if len(det) else np.zeros((0, 6), np.float32)
            salidas[c] = tr.update(Boxes(datos, (alto, ancho)), None, dt=m.dt, warp=mov["M"] if kl > 0 else None)
        for fila in desc.paso(kl, m.t, m.dt, mov, salidas[0], det):
            reg = {"archivo": nombre, "id": fila["id"], "k": m.k, "t": round(m.t, 4), "dt": round(m.dt, 4),
                   "s_k": round(mov["s"], 5), "rot": round(mov["rot"], 5), "tx": round(mov["tx"] / ancho, 6),
                   "ty": round(mov["ty"] / alto, 6), "inliers": round(mov["inliers"], 4),
                   "d2_med": round(fila["d2_med"], 6), "d11_med": round(fila["d11_med"], 6)}
            if detector is not None:
                reg["tramo"] = tramo
            for i, v in enumerate(fila["d"], 1):
                reg[f"d{i}"] = round(v, 6)
            filas.append(reg)
    resumen = resumen_video(nombre, t_fin, muestras, cortes, inicios, t_fin) if detector is not None and inicios else None
    return filas, muestras, cortes, indicadores, resumen


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
    p.add_argument("--min_humo", type=float, default=0.2)
    p.add_argument("--conf_humo", type=float, default=0.25)
    p.add_argument("--ventana", type=float, default=3.0)
    p.add_argument("--mediana", type=float, default=1.0)
    p.add_argument("--min_duracion", type=float, default=1.0)
    p.add_argument("--sin_cortes", action="store_true")
    p.add_argument("--t_alto", type=float, default=0.85)
    p.add_argument("--t_bajo", type=float, default=0.65)
    p.add_argument("--separacion", type=float, default=1.0)
    p.add_argument("--indicadores", action="store_true")
    p.add_argument("--salida", default=os.path.join("resultados", "descriptores_v2"))
    p.add_argument("--salida_cortes", default=os.path.join("resultados", "cortes"))
    p.add_argument("--grafico", default="14294880_3840_2160_24fps.mp4")
    p.add_argument("--reanudar", action="store_true")
    a = p.parse_args()

    targs = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                            track_buffer=30, match_thresh=0.8, fuse_score=True)
    with open(a.lista, encoding="utf-8") as f:
        lista = [r["archivo"] for r in csv.DictReader(f)]
    rutas = buscar_rutas(a.videos)
    os.makedirs(a.salida, exist_ok=True)
    ruta_csv = os.path.join(a.salida, "descriptores.csv")
    ruta_log = os.path.join(a.salida, "procesados.csv")
    hechos = set()
    if a.reanudar and os.path.exists(ruta_log) and os.path.exists(ruta_csv):
        with open(ruta_log, encoding="utf-8") as f:
            hechos = {r["archivo"] for r in csv.DictReader(f)}
        print(f"reanudando: {len(hechos)} videos ya procesados")
    agregar = bool(hechos)
    campos = CAMPOS_V2 + ([] if a.sin_cortes else ["tramo"])
    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    modelo = YOLO(a.modelo)
    print(f"parametros: mediana {a.mediana} s | min_humo {a.min_humo} | min_duracion {a.min_duracion} s"
          f" | factor_humo {a.factor_humo} | variante {a.variante}"
          f" | cortes {'no' if a.sin_cortes else f'si (t_alto {a.t_alto}, t_bajo {a.t_bajo}, separacion {a.separacion} s)'}")
    archivos = []
    f, w = abrir_csv(ruta_csv, campos, agregar)
    g, wl = abrir_csv(ruta_log, CAMPOS_LOG, agregar)
    archivos += [f, g]
    if not a.sin_cortes:
        os.makedirs(a.salida_cortes, exist_ok=True)
        fc, wc = abrir_csv(os.path.join(a.salida_cortes, "cortes.csv"), CAMPOS_CORTES, agregar)
        fr, wr = abrir_csv(os.path.join(a.salida_cortes, "resumen.csv"), CAMPOS_RESUMEN, agregar)
        archivos += [fc, fr]
        if a.indicadores:
            fi, wi = abrir_csv(os.path.join(a.salida_cortes, "indicadores.csv"), CAMPOS_INDICADORES, agregar)
            archivos.append(fi)
    try:
        for n, nombre in enumerate(lista, 1):
            if nombre in hechos:
                continue
            if nombre not in rutas:
                print(f"no encontrado: {nombre}")
                continue
            inicio = time.time()
            filas, muestras, cortes, indicadores, resumen = procesar_video(rutas[nombre], nombre, a, modelo, dispositivo, targs)
            w.writerows(filas)
            if not a.sin_cortes:
                wc.writerows(cortes)
                if resumen:
                    wr.writerow(resumen)
                if a.indicadores:
                    wi.writerows(indicadores)
            for x in archivos[2:]:
                x.flush()
            f.flush()
            ids = {(r.get("tramo", 0), r["id"]) for r in filas}
            seg = time.time() - inicio
            wl.writerow({"archivo": nombre, "muestras": muestras, "trayectorias": len(ids), "filas": len(filas),
                         "segundos": round(seg, 1)})
            g.flush()
            extra = "" if a.sin_cortes else f" | cortes {len(cortes)}"
            print(f"[{n}/{len(lista)}] {nombre[:50]} | muestras {muestras} | trayectorias de fuego {len(ids)}"
                  f" | filas {len(filas)}{extra} | {seg:.1f} s")
            if nombre == a.grafico and filas:
                por_id = {}
                for r in filas:
                    por_id.setdefault((r.get("tramo", 0), r["id"]), []).append(r)
                (tr, tid), sel = max(por_id.items(), key=lambda x: len(x[1]))
                ruta_png = os.path.join(a.salida, f"descriptores_{os.path.splitext(nombre)[0]}_id{tid}.png")
                grafico(sel, ruta_png, f"{nombre} | trayectoria {tid} | {len(sel)} muestras | descriptores v2")
                print(f"   grafico de la trayectoria {tid} ({len(sel)} muestras): {ruta_png}")
    finally:
        for x in archivos:
            x.close()
    print(f"descriptores en: {ruta_csv}")


if __name__ == "__main__":
    main()
