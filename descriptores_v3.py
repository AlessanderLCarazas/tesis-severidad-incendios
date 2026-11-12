import os
import csv
import time
import zlib
import argparse
from types import SimpleNamespace
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from preprocesamiento import LectorVideo, buscar_rutas
from camara import MovimientoCamara
from cortes import DetectorCortes, CAMPOS_CORTES, CAMPOS_RESUMEN, resumen_video
from descriptores_v2 import DescriptoresV2, CAMPOS_V2, CAMPOS_LOG
from preanotar import area_union
import seguimiento as sg

CAMPOS_DESC = CAMPOS_V2 + ["tramo"]
CAMPOS_FOTOGRAMA = ["archivo", "k", "t", "dt", "tramo", "s_k", "rot", "tx", "ty", "inliers", "valido", "n_tray",
                    "n_det_fuego", "n_det_humo", "area_det_fuego", "area_det_humo", "conf_det_fuego", "conf_det_humo",
                    "n_det_fuego_alta", "n_det_humo_alta", "area_det_fuego_alta", "area_det_humo_alta"]
ARCHIVOS = {"descriptores.csv": CAMPOS_DESC, "fotogramas.csv": CAMPOS_FOTOGRAMA, "cortes.csv": CAMPOS_CORTES,
            "resumen.csv": CAMPOS_RESUMEN, "procesados.csv": CAMPOS_LOG}


def detecciones_fotograma(det, ancho, alto, conf_alta, gx, gy):
    salida = {}
    for clase, nombre in ((0, "fuego"), (1, "humo")):
        d = det[det[:, 5] == clase] if len(det) else det
        cajas = d[:, :4] / np.array([ancho, alto, ancho, alto], np.float32) if len(d) else np.zeros((0, 4), np.float32)
        altas = d[:, 4] >= conf_alta if len(d) else np.zeros(0, bool)
        salida[f"n_det_{nombre}"] = int(len(d))
        salida[f"area_det_{nombre}"] = round(area_union(cajas, gx, gy), 6)
        salida[f"conf_det_{nombre}"] = round(float(d[:, 4].max()), 4) if len(d) else 0.0
        salida[f"n_det_{nombre}_alta"] = int(altas.sum())
        salida[f"area_det_{nombre}_alta"] = round(area_union(cajas[altas], gx, gy), 6)
    return salida


def procesar_video(ruta, nombre, a, modelo, dispositivo, targs):
    semilla = a.semilla + zlib.crc32(nombre.encode("utf-8"))
    detector = DetectorCortes(a.t_alto, a.t_bajo, a.separacion)
    trackers = cam = desc = None
    filas, fotogramas, cortes, inicios = [], [], [], []
    k0, tramo, t_fin = 0, -1, 0.0
    for m in LectorVideo(ruta, a.fps, a.quitar, semilla):
        t_fin = m.t
        alto, ancho = m.imagen.shape[:2]
        r = modelo.predict(m.imagen, imgsz=a.img, conf=a.conf, device=dispositivo, verbose=False)[0]
        det = r.boxes.data.cpu().numpy() if r.boxes is not None else np.zeros((0, 6), np.float32)
        fuego_det = det[det[:, 5] == 0] if len(det) else det
        cajas = [tuple(x[:4]) for x in (fuego_det if a.mascara == "fuego" else det if a.mascara == "todas" else [])]
        motivo = ""
        if trackers is None:
            reiniciar = True
            detector.actualizar(m.imagen, m.t, 0)
        else:
            mov = cam.actualizar(m.imagen, cajas)
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
        kl = m.k - k0
        salidas = {}
        for c, tr in trackers.items():
            datos = det[det[:, 5] == c] if len(det) else np.zeros((0, 6), np.float32)
            salidas[c] = tr.update(Boxes(datos, (alto, ancho)), None, dt=m.dt, warp=mov["M"] if kl > 0 else None)
        camara = {"s_k": round(mov["s"], 5), "rot": round(mov["rot"], 5), "tx": round(mov["tx"] / ancho, 6),
                  "ty": round(mov["ty"] / alto, 6), "inliers": round(mov["inliers"], 4)}
        n_tray = 0
        for fila in desc.paso(kl, m.t, m.dt, mov, salidas[0], det):
            reg = {"archivo": nombre, "id": fila["id"], "k": m.k, "t": round(m.t, 4), "dt": round(m.dt, 4),
                   "d2_med": round(fila["d2_med"], 6), "d11_med": round(fila["d11_med"], 6), "tramo": tramo}
            reg.update(camara)
            for i, v in enumerate(fila["d"], 1):
                reg[f"d{i}"] = round(v, 6)
            filas.append(reg)
            n_tray += 1
        foto = {"archivo": nombre, "k": m.k, "t": round(m.t, 4), "dt": round(m.dt, 4), "tramo": tramo,
                "valido": mov["valido"], "n_tray": n_tray}
        foto.update(camara)
        foto.update(detecciones_fotograma(det, ancho, alto, a.conf_alta, a.gx, a.gy))
        fotogramas.append(foto)
    resumen = resumen_video(nombre, t_fin, len(fotogramas), cortes, inicios, t_fin) if inicios else None
    return filas, fotogramas, cortes, resumen


def limpiar(ruta, campos, hechos):
    if not os.path.exists(ruta):
        return 0
    tmp, quitadas = ruta + ".tmp", 0
    with open(ruta, encoding="utf-8", newline="") as f, open(tmp, "w", encoding="utf-8", newline="") as g:
        w = csv.DictWriter(g, fieldnames=campos)
        w.writeheader()
        for r in csv.DictReader(f):
            if r.get("archivo") in hechos and set(r) == set(campos):
                w.writerow(r)
            else:
                quitadas += 1
    if quitadas:
        os.replace(tmp, ruta)
    else:
        os.remove(tmp)
    return quitadas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lista", default=os.path.join("resultados", "cortes", "lista_noche.csv"))
    p.add_argument("--videos", nargs="+", default=[os.path.join("real_video", "real_video", "fire"),
                                                   os.path.join("real_video", "real_video", "non_fire")])
    p.add_argument("--modelo", default=os.path.join("runs_fuego", "fasdd_v8s_full", "weights", "best.pt"))
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--conf_alta", type=float, default=0.25)
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
    p.add_argument("--t_alto", type=float, default=0.85)
    p.add_argument("--t_bajo", type=float, default=0.65)
    p.add_argument("--separacion", type=float, default=1.0)
    p.add_argument("--gx", type=int, default=64)
    p.add_argument("--gy", type=int, default=36)
    p.add_argument("--excluir", default=os.path.join("resultados", "excluidos.csv"))
    p.add_argument("--salida", default=os.path.join("runs_fuego", "descriptores_v3"))
    a = p.parse_args()

    targs = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                            track_buffer=30, match_thresh=0.8, fuse_score=True)
    with open(a.lista, encoding="utf-8") as f:
        lista = [r["archivo"] for r in csv.DictReader(f)]
    if a.excluir and os.path.exists(a.excluir):
        with open(a.excluir, encoding="utf-8") as f:
            fuera = {r["archivo"] for r in csv.DictReader(f)}
        lista = [v for v in lista if v not in fuera]
    rutas = buscar_rutas(a.videos)
    os.makedirs(a.salida, exist_ok=True)
    ruta_log = os.path.join(a.salida, "procesados.csv")
    hechos = set()
    if os.path.exists(ruta_log):
        with open(ruta_log, encoding="utf-8") as f:
            hechos = {r["archivo"] for r in csv.DictReader(f)}
    if hechos or any(os.path.exists(os.path.join(a.salida, n)) for n in ARCHIVOS):
        quitadas = {n: limpiar(os.path.join(a.salida, n), campos, hechos) for n, campos in ARCHIVOS.items() if n != "procesados.csv"}
        print(f"reanudando: {len(hechos)} videos ya procesados | filas incompletas quitadas: {sum(quitadas.values())}")
    pendientes = [v for v in lista if v not in hechos]
    print(f"videos en la lista: {len(lista)} | pendientes: {len(pendientes)} | salida: {a.salida}")
    if not pendientes:
        print("no hay nada pendiente")
        return
    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    modelo = YOLO(a.modelo)
    abiertos, escritores = {}, {}
    for n, campos in ARCHIVOS.items():
        ruta = os.path.join(a.salida, n)
        nuevo = not os.path.exists(ruta)
        abiertos[n] = open(ruta, "a", newline="", encoding="utf-8")
        escritores[n] = csv.DictWriter(abiertos[n], fieldnames=campos)
        if nuevo:
            escritores[n].writeheader()
    inicio_total, hechos_ahora, seg_video = time.time(), 0, 0.0
    try:
        for n, nombre in enumerate(pendientes, 1):
            if nombre not in rutas:
                print(f"no encontrado: {nombre}")
                continue
            inicio = time.time()
            filas, fotogramas, cortes, resumen = procesar_video(rutas[nombre], nombre, a, modelo, dispositivo, targs)
            escritores["descriptores.csv"].writerows(filas)
            escritores["fotogramas.csv"].writerows(fotogramas)
            escritores["cortes.csv"].writerows(cortes)
            if resumen:
                escritores["resumen.csv"].writerow(resumen)
            for nombre_csv in ARCHIVOS:
                if nombre_csv != "procesados.csv":
                    abiertos[nombre_csv].flush()
            seg = time.time() - inicio
            ids = {(r["tramo"], r["id"]) for r in filas}
            escritores["procesados.csv"].writerow({"archivo": nombre, "muestras": len(fotogramas), "trayectorias": len(ids),
                                                   "filas": len(filas), "segundos": round(seg, 1)})
            abiertos["procesados.csv"].flush()
            os.fsync(abiertos["procesados.csv"].fileno())
            hechos_ahora += 1
            transcurrido = time.time() - inicio_total
            falta = transcurrido / hechos_ahora * (len(pendientes) - n)
            print(f"[{n}/{len(pendientes)}] {nombre[:50].encode('ascii', 'replace').decode()} | fotogramas {len(fotogramas)}"
                  f" | trayectorias {len(ids)} | cortes {len(cortes)} | {seg:.1f} s | llevo {transcurrido / 60:.0f} min"
                  f" | faltan unos {falta / 60:.0f} min", flush=True)
    finally:
        for x in abiertos.values():
            x.close()
    print(f"terminado: {hechos_ahora} videos en {(time.time() - inicio_total) / 3600:.2f} h | salida en: {a.salida}")


if __name__ == "__main__":
    main()
