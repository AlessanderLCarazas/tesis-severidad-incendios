import os
import zlib
import random
import argparse
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from preprocesamiento import LectorVideo, buscar_rutas
from camara import MovimientoCamara
from descriptores import Descriptores
import seguimiento as sg

PANEL = [(2, "d2 area", (0, 165, 255)), (3, "d3 crecimiento (1/s)", (80, 220, 80)), (11, "d11 area de humo", (200, 200, 200))]


def color_id(i):
    r = random.Random(i * 7919)
    return (r.randint(60, 255), r.randint(60, 255), r.randint(60, 255))


def procesar(ruta, nombre, a, modelo, dispositivo):
    targs = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                            track_buffer=30, match_thresh=0.8, fuse_score=True)
    trackers = sg.crear_trackers(a.variante, targs, 1.0 / a.fps, a.dt_max)
    cam = MovimientoCamara()
    desc = None
    cuadros = []
    for m in LectorVideo(ruta, a.fps, 0.0, a.semilla + zlib.crc32(nombre.encode("utf-8"))):
        alto, ancho = m.imagen.shape[:2]
        if desc is None:
            desc = Descriptores(ancho, alto, a.factor_humo)
        r = modelo.predict(m.imagen, imgsz=a.img, conf=a.conf, device=dispositivo, verbose=False)[0]
        det = r.boxes.data.cpu().numpy() if r.boxes is not None else np.zeros((0, 6), np.float32)
        fuego_det = det[det[:, 5] == 0] if len(det) else det
        mov = cam.actualizar(m.imagen, [tuple(x[:4]) for x in fuego_det])
        salidas = {}
        for c, tr in trackers.items():
            datos = det[det[:, 5] == c] if len(det) else np.zeros((0, 6), np.float32)
            salidas[c] = tr.update(Boxes(datos, (alto, ancho)), None, dt=m.dt, warp=mov["M"] if m.k > 0 else None)
        filas = desc.paso(m.k, m.t, m.dt, mov, salidas[0], det)
        cuadros.append({"k": m.k, "t": m.t, "dt": m.dt, "fuego": salidas[0], "humo": salidas[1],
                        "desc": {f["id"]: f["d"] for f in filas}})
    return cuadros


def principal_por_cuadro(cuadros):
    return [max(c["desc"].items(), key=lambda x: x[1][1])[0] if c["desc"] else None for c in cuadros]


def texto(img, s, x, y, escala=0.5, color=(255, 255, 255), grosor=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def posicion_etiqueta(y):
    return int(y) - 4 if y - 4 > 74 else max(int(y) + 18, 76)


def dibujar_panel(ancho, alto, ts, series, t_max, t_actual, tid, cambios):
    panel = np.full((alto, ancho, 3), 24, np.uint8)
    texto(panel, f"region principal (mayor area): id {'-' if tid is None else tid}", 12, 24, 0.5, (0, 255, 255))
    margen_sup, margen_inf, izq, der = 40, 28, 58, 12
    h_graf = (alto - margen_sup - margen_inf) // len(PANEL)
    x0, x1 = izq, ancho - der
    for i, (d, nombre, color) in enumerate(PANEL):
        y0 = margen_sup + i * h_graf + 18
        y1 = margen_sup + (i + 1) * h_graf - 12
        v = series[d]
        validos = v[~np.isnan(v)]
        if d == 3:
            tope = max(1e-4, float(np.nanmax(np.abs(validos))) if len(validos) else 1e-4)
            lo, hi = -tope, tope
        else:
            lo, hi = 0.0, max(1e-4, float(np.nanmax(validos)) * 1.1 if len(validos) else 1e-4)
        cv2.rectangle(panel, (x0, y0), (x1, y1), (70, 70, 70), 1)
        for tc in cambios:
            if tc <= t_actual:
                xm = int(x0 + (x1 - x0) * tc / t_max)
                cv2.line(panel, (xm, y0), (xm, y1), (90, 90, 160), 1)
        actual = v[np.searchsorted(ts, t_actual, side="right") - 1] if len(ts) and t_actual >= ts[0] else np.nan
        valor = "-" if np.isnan(actual) else f"{actual:.4f}"
        texto(panel, f"{nombre}: {valor}", x0, y0 - 5, 0.45, color)
        for yv in [lo, hi] + ([0.0] if d == 3 else []):
            yy = int(y1 - (y1 - y0) * (yv - lo) / (hi - lo))
            cv2.putText(panel, f"{yv:.3g}", (4, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
            if d == 3 and yv == 0.0:
                cv2.line(panel, (x0, yy), (x1, yy), (60, 60, 60), 1)
        puntos = []
        for t, y in zip(ts, v):
            if t > t_actual:
                break
            if np.isnan(y):
                if len(puntos) > 1:
                    cv2.polylines(panel, [np.array(puntos, np.int32)], False, color, 2, cv2.LINE_AA)
                puntos = []
                continue
            puntos.append((int(x0 + (x1 - x0) * t / t_max), int(y1 - (y1 - y0) * (min(max(y, lo), hi) - lo) / (hi - lo))))
        if len(puntos) > 1:
            cv2.polylines(panel, [np.array(puntos, np.int32)], False, color, 2, cv2.LINE_AA)
        xc = int(x0 + (x1 - x0) * min(t_actual, t_max) / t_max)
        cv2.line(panel, (xc, y0), (xc, y1), (255, 255, 0), 1)
    paso = 5 if t_max <= 40 else 10 if t_max <= 90 else 20
    for s in range(0, int(t_max) + 1, paso):
        x = int(x0 + (x1 - x0) * s / t_max)
        if x > x1 - 45:
            continue
        cv2.putText(panel, f"{s}", (x - 5, alto - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(panel, "t (s)", (x1 - 30, alto - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    return panel


def generar(ruta, nombre, titulo, salida_mp4, salida_gif, cuadros, a):
    principal = principal_por_cuadro(cuadros)
    ts = np.array([c["t"] for c in cuadros])
    series = {d: np.array([c["desc"][i][d - 1] if i is not None else np.nan for c, i in zip(cuadros, principal)])
              for d, _, _ in PANEL}
    cambios = [c["t"] for c, i, j in zip(cuadros[1:], principal[1:], principal[:-1]) if i is not None and j is not None and i != j]
    ids = [i for i in dict.fromkeys(principal) if i is not None]
    t_max = max(float(ts[-1]), 1.0)
    escritor = None
    fotos_gif = []
    paso_gif = max(1, int(round(a.fps / a.fps_gif)))
    paso_mp4 = max(1, int(round(a.fps / a.fps_mp4)))
    for m, c, tid in zip(LectorVideo(ruta, a.fps, 0.0, a.semilla + zlib.crc32(nombre.encode("utf-8"))), cuadros, principal):
        img = m.imagen
        esc = a.alto / img.shape[0]
        img = cv2.resize(img, (int(round(img.shape[1] * esc / 2)) * 2, a.alto))
        for x1, y1, x2, y2, i, conf, _, _ in c["humo"]:
            cv2.rectangle(img, (int(x1 * esc), int(y1 * esc)), (int(x2 * esc), int(y2 * esc)), (170, 170, 170), 1)
            texto(img, f"humo {int(i)}", int(x1 * esc), posicion_etiqueta(y1 * esc), 0.4, (200, 200, 200))
        for x1, y1, x2, y2, i, conf, _, _ in c["fuego"]:
            principal = int(i) == tid
            col = (0, 255, 255) if principal else color_id(int(i))
            cv2.rectangle(img, (int(x1 * esc), int(y1 * esc)), (int(x2 * esc), int(y2 * esc)), col, 3 if principal else 2)
            texto(img, f"fuego id {int(i)} {conf:.2f}", int(x1 * esc), posicion_etiqueta(y1 * esc), 0.45, col)
        franja = img[:58].copy()
        img[:58] = (franja * 0.35).astype(np.uint8)
        texto(img, titulo, 10, 24, 0.6, (0, 255, 255))
        texto(img, f"t = {m.t:.2f} s   dt = {m.dt * 1000:.0f} ms   k = {m.k}", 10, 48, 0.55)
        panel = dibujar_panel(a.ancho_panel, a.alto, ts, series, t_max, m.t, tid, cambios)
        cuadro = np.hstack([img, panel])
        if escritor is None:
            escritor = cv2.VideoWriter(salida_mp4, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), a.fps / paso_mp4,
                                       (cuadro.shape[1], cuadro.shape[0]))
        if m.k % paso_mp4 == 0:
            escritor.write(cuadro)
        if m.k % paso_gif == 0:
            g = cv2.resize(cuadro, (a.ancho_gif, int(round(cuadro.shape[0] * a.ancho_gif / cuadro.shape[1]))),
                           interpolation=cv2.INTER_AREA)
            fotos_gif.append(Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)))
    if escritor is not None:
        escritor.release()
    paleta = [f.quantize(colors=a.colores_gif, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in fotos_gif]
    paleta[0].save(salida_gif, save_all=True, append_images=paleta[1:], duration=int(1000 / a.fps_gif), loop=0, optimize=True)
    return ids, len(cambios), len(cuadros), len(fotos_gif)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controlado", default="19151618-hd_1920_1080_30fps.mp4")
    p.add_argument("--crecimiento", default="")
    p.add_argument("--videos", nargs="+", default=[os.path.join("real_video", "real_video", "fire")])
    p.add_argument("--modelo", default=os.path.join("runs_fuego", "fasdd_v8s_full", "weights", "best.pt"))
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--img", type=int, default=640)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--variante", default="dt_tope2_cmc")
    p.add_argument("--dt_max", type=float, default=2.0)
    p.add_argument("--factor_humo", type=float, default=2.0)
    p.add_argument("--alto", type=int, default=540)
    p.add_argument("--ancho_panel", type=int, default=440)
    p.add_argument("--fps_mp4", type=float, default=10.0)
    p.add_argument("--fps_gif", type=float, default=5.0)
    p.add_argument("--ancho_gif", type=int, default=640)
    p.add_argument("--colores_gif", type=int, default=128)
    p.add_argument("--salida_mp4", default=os.path.join("runs_fuego", "demo"))
    p.add_argument("--salida_gif", default=os.path.join("resultados", "demo"))
    a = p.parse_args()

    rutas = buscar_rutas(a.videos)
    os.makedirs(a.salida_mp4, exist_ok=True)
    os.makedirs(a.salida_gif, exist_ok=True)
    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    modelo = YOLO(a.modelo)
    for clave, nombre, titulo in [("controlado", a.controlado, "fuego controlado (vela)"),
                                  ("crecimiento", a.crecimiento, "fuego con crecimiento sostenido")]:
        if not nombre:
            continue
        if nombre not in rutas:
            print(f"no encontrado: {nombre}")
            continue
        cuadros = procesar(rutas[nombre], nombre, a, modelo, dispositivo)
        mp4 = os.path.join(a.salida_mp4, f"demo_{clave}.mp4")
        gif = os.path.join(a.salida_gif, f"demo_{clave}.gif")
        ids, n_cambios, n, n_gif = generar(rutas[nombre], nombre, titulo, mp4, gif, cuadros, a)
        print(f"{clave}: {nombre[:50]} | muestras {n} | region principal: ids {ids}, {n_cambios} cambios"
              f" | mp4 {os.path.getsize(mp4) / 1e6:.1f} mb"
              f" | gif {n_gif} cuadros, {os.path.getsize(gif) / 1e6:.1f} mb")
        print(f"   {mp4}")
        print(f"   {gif}")


if __name__ == "__main__":
    main()
