import os
import json
import time
import zlib
import random
import argparse
from collections import deque, defaultdict
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from preprocesamiento import LectorVideo, buscar_rutas
from camara import MovimientoCamara
from cortes import DetectorCortes
from descriptores_v2 import DescriptoresV2
from descriptores_v3 import detecciones_fotograma
from normalizacion import Normalizador
from vector_escena import VectorEscena, DETECCIONES, contexto
from modelo_escena import ModeloEscena
from secuencias import leer_csv
import seguimiento as sg

NIVELES = ["controlado", "incipiente", "incontrolado"]
COLORES = [(80, 200, 80), (0, 165, 255), (40, 40, 230)]
ETAPAS = ["lectura", "yolo", "camara", "seguimiento", "descriptores", "vector", "modelo", "decision"]
PARAMETROS = dict(fps=10.0, conf=0.1, conf_alta=0.25, img=640, variante="dt_tope2_cmc", dt_max=2.0, factor_humo=2.0, min_humo=0.2,
                  conf_humo=0.25, ventana=3.0, mediana=1.0, min_duracion=1.0, t_alto=0.85, t_bajo=0.65, separacion=1.0, gx=64, gy=36)


class ModeloEnVivo:
    def __init__(self, nombre, config, norm_x, norm_c):
        c = config["corridas"][nombre]
        estado = torch.load(c["archivo_pesos"], map_location="cpu", weights_only=False)
        self.nombre, self.tipo = nombre, estado["tipo"]
        self.red = ModeloEscena(estado["tipo"], n_x=estado["n_x"], unidades=estado["unidades"], dropout=estado["dropout"])
        self.red.load_state_dict(estado["modelo"])
        self.red.eval()
        self.unidad = config["por_modelo"][c["modelo"]]["unidad_dt_s"] or 1.0
        self.norm_x, self.norm_c = norm_x, norm_c
        dec = config["decision"]
        self.u1, self.u2 = dec["umbral_coral"]
        self.permanencia, self.factor = dec["histeresis_s"], dec["factor_desactivacion"]
        self.umbral = c["umbral_alerta"][f"{dec['alfa']:g}"]
        self.reiniciar_video()

    def reiniciar_tramo(self):
        self.h = None

    def reiniciar_video(self):
        self.h = None
        self.nivel = None
        self.t0 = None
        self.ventana = deque()
        self.alerta, self.t_bajo = False, None

    @torch.no_grad()
    def inferir(self, x, c, dt):
        red = self.red
        x = torch.from_numpy(self.norm_x.transformar(x[None]).astype(np.float32))
        c = torch.from_numpy(self.norm_c.transformar(c[None]).astype(np.float32))
        ts = torch.tensor([[dt / self.unidad]], dtype=torch.float32)
        u = x * torch.sigmoid(red.compuerta(c))
        if self.tipo == "cfc":
            h = torch.zeros(1, red.unidades) if self.h is None else self.h
            _, self.h = red.celda(u, h, ts)
            salida = self.h
        else:
            y, self.h = red.rnn(torch.cat([u, ts], -1).unsqueeze(1), self.h)
            salida = y[:, 0]
        p = torch.sigmoid(red.coral(salida) + red.sesgos)[0]
        return (float(p[0]), float(p[1]), float(torch.sigmoid(red.riesgo(salida))[0, 0]),
                float(torch.expm1(red.tau(salida)).clamp(min=0)[0, 0]), x[0].numpy(), c[0].numpy())

    def decidir(self, t, p1, p2, riesgo):
        crudo = int(p1 > self.u1) + int(p1 > self.u1 and p2 > self.u2)
        if self.nivel is None:
            self.nivel, self.t0 = crudo, t
        self.ventana.append((t, crudo))
        while self.ventana and self.ventana[0][0] < t - self.permanencia - 1e-6:
            self.ventana.popleft()
        if t - self.t0 >= self.permanencia - 1e-6:
            valores = [n for _, n in self.ventana]
            if min(valores) > self.nivel:
                self.nivel = min(valores)
            elif max(valores) < self.nivel:
                self.nivel = max(valores)
        if not self.alerta:
            if riesgo >= self.umbral:
                self.alerta, self.t_bajo = True, None
        elif riesgo < self.umbral * self.factor:
            self.t_bajo = t if self.t_bajo is None else self.t_bajo
            if t - self.t_bajo >= self.permanencia - 1e-6:
                self.alerta = False
        else:
            self.t_bajo = None
        return crudo, self.nivel, self.alerta


class Sistema:
    def __init__(self, config, corridas, yolo=os.path.join("runs_fuego", "fasdd_v8s_full", "weights", "best.pt"), **cambios):
        self.p = SimpleNamespace(**dict(PARAMETROS, **cambios))
        norm_x = Normalizador.cargar(config["datos"]["normalizacion"][0])
        norm_c = Normalizador.cargar(config["datos"]["normalizacion"][1])
        self.modelos = [ModeloEnVivo(n, config, norm_x, norm_c) for n in corridas]
        self.dispositivo = 0 if torch.cuda.is_available() else "cpu"
        self.yolo = YOLO(yolo)
        self.targs = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                                     track_buffer=30, match_thresh=0.8, fuse_score=True)
        self.yolo.predict(np.zeros((480, 640, 3), np.uint8), imgsz=self.p.img, conf=self.p.conf, device=self.dispositivo, verbose=False)
        self.iniciar()

    def iniciar(self):
        p = self.p
        self.detector = DetectorCortes(p.t_alto, p.t_bajo, p.separacion)
        self.trackers = self.cam = self.desc = None
        self.vector = VectorEscena(p.min_duracion)
        self.k0, self.tramo = 0, -1
        for m in self.modelos:
            m.reiniciar_video()

    def paso(self, m):
        p = self.p
        tiempos = {}
        reloj = time.perf_counter()
        alto, ancho = m.imagen.shape[:2]
        r = self.yolo.predict(m.imagen, imgsz=p.img, conf=p.conf, device=self.dispositivo, verbose=False)[0]
        det = r.boxes.data.cpu().numpy() if r.boxes is not None else np.zeros((0, 6), np.float32)
        tiempos["yolo"] = time.perf_counter() - reloj
        reloj = time.perf_counter()
        fuego_det = det[det[:, 5] == 0] if len(det) else det
        cajas = [tuple(x[:4]) for x in fuego_det]
        corte = ""
        if self.trackers is None:
            reiniciar = True
            self.detector.actualizar(m.imagen, m.t, 0)
        else:
            mov = self.cam.actualizar(m.imagen, cajas)
            _, corte = self.detector.actualizar(m.imagen, m.t, mov["valido"])
            reiniciar = bool(corte)
        if reiniciar:
            self.tramo += 1
            self.k0 = m.k
            self.trackers = sg.crear_trackers(p.variante, self.targs, 1.0 / p.fps, p.dt_max)
            self.cam = MovimientoCamara()
            mov = self.cam.actualizar(m.imagen, cajas)
            self.desc = DescriptoresV2(ancho, alto, p.factor_humo, p.min_humo, p.ventana, p.mediana, p.min_duracion, p.conf_humo)
            self.vector.reiniciar()
            for modelo in self.modelos:
                modelo.reiniciar_tramo()
        tiempos["camara"] = time.perf_counter() - reloj
        reloj = time.perf_counter()
        kl = m.k - self.k0
        salidas = {}
        for c, tr in self.trackers.items():
            datos = det[det[:, 5] == c] if len(det) else np.zeros((0, 6), np.float32)
            salidas[c] = tr.update(Boxes(datos, (alto, ancho)), None, dt=m.dt, warp=mov["M"] if kl > 0 else None)
        tiempos["seguimiento"] = time.perf_counter() - reloj
        reloj = time.perf_counter()
        filas = self.desc.paso(kl, m.t, m.dt, mov, salidas[0], det)
        tiempos["descriptores"] = time.perf_counter() - reloj
        reloj = time.perf_counter()
        camara = {"s_k": round(mov["s"], 5), "rot": round(mov["rot"], 5), "tx": round(mov["tx"] / ancho, 6),
                  "ty": round(mov["ty"] / alto, 6), "inliers": round(mov["inliers"], 4)}
        d = detecciones_fotograma(det, ancho, alto, p.conf_alta, p.gx, p.gy)
        x = self.vector.paso(round(m.t, 4), sorted(((f["id"], f["d"]) for f in filas), key=lambda f: f[0]),
                             {c: d[c + "_alta"] for c in DETECCIONES})
        c = contexto(camara, round(m.dt, 4))
        tiempos["vector"] = time.perf_counter() - reloj
        resultados = {}
        for modelo in self.modelos:
            reloj = time.perf_counter()
            p1, p2, riesgo, tau, xn, cn = modelo.inferir(x, c, round(m.dt, 4))
            tiempos[f"modelo:{modelo.nombre}"] = time.perf_counter() - reloj
            reloj = time.perf_counter()
            crudo, nivel, alerta = modelo.decidir(round(m.t, 4), p1, p2, riesgo)
            tiempos[f"decision:{modelo.nombre}"] = time.perf_counter() - reloj
            resultados[modelo.nombre] = {"p1": p1, "p2": p2, "riesgo": riesgo, "tau": tau, "crudo": crudo, "nivel": nivel,
                                         "alerta": alerta, "x": xn, "c": cn}
        return {"k": m.k, "t": m.t, "dt": m.dt, "tramo": self.tramo, "corte": corte, "fuego": salidas[0], "humo": salidas[1],
                "resultados": resultados, "tiempos": tiempos}


def recorrer(sistema, ruta, nombre):
    sistema.iniciar()
    lector = iter(LectorVideo(ruta, sistema.p.fps, 0.0, 42 + zlib.crc32(nombre.encode("utf-8"))))
    while True:
        reloj = time.perf_counter()
        try:
            m = next(lector)
        except StopIteration:
            return
        lectura = time.perf_counter() - reloj
        salida = sistema.paso(m)
        salida["tiempos"]["lectura"] = lectura
        yield m, salida


def texto(img, s, x, y, escala=0.5, color=(255, 255, 255), grosor=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def panel(ancho, alto, historia, t_max, umbral):
    img = np.full((alto, ancho, 3), 24, np.uint8)
    x0, x1 = 50, ancho - 14
    ts = np.array([h[0] for h in historia])
    px = lambda t: int(x0 + (x1 - x0) * min(t, t_max) / t_max)
    y0, y1 = 50, alto // 2 - 20
    texto(img, f"riesgo: {historia[-1][1]:.2f}   umbral de alerta: {umbral:.2f}", x0, y0 - 14, 0.5, (0, 255, 255))
    cv2.rectangle(img, (x0, y0), (x1, y1), (70, 70, 70), 1)
    yu = int(y1 - (y1 - y0) * min(umbral, 1.0))
    cv2.line(img, (x0, yu), (x1, yu), (60, 60, 200), 1)
    for v in (0.0, 0.5, 1.0):
        cv2.putText(img, f"{v:g}", (14, int(y1 - (y1 - y0) * v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    puntos = np.array([(px(t), int(y1 - (y1 - y0) * r)) for t, r, _, _ in historia], np.int32)
    if len(puntos) > 1:
        cv2.polylines(img, [puntos], False, (0, 255, 255), 2, cv2.LINE_AA)
    z0, z1 = alto // 2 + 40, alto - 70
    texto(img, f"nivel: {NIVELES[historia[-1][2]]}", x0, z0 - 14, 0.5, COLORES[historia[-1][2]])
    cv2.rectangle(img, (x0, z0), (x1, z1), (70, 70, 70), 1)
    for n, nombre in enumerate(NIVELES):
        y = int(z1 - (z1 - z0) * (n + 0.5) / 3)
        cv2.putText(img, nombre[:5], (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORES[n], 1, cv2.LINE_AA)
    for (t, _, n, _), t_sig in zip(historia, list(ts[1:]) + [ts[-1] + 0.1]):
        y = int(z1 - (z1 - z0) * (n + 0.5) / 3)
        cv2.line(img, (px(t), y), (max(px(t_sig), px(t) + 1), y), COLORES[n], 6)
    a0, a1 = alto - 52, alto - 36
    cv2.putText(img, "alerta", (4, a1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, a0), (x1, a1), (70, 70, 70), 1)
    for (t, _, _, al), t_sig in zip(historia, list(ts[1:]) + [ts[-1] + 0.1]):
        if al:
            cv2.rectangle(img, (px(t), a0 + 1), (max(px(t_sig), px(t) + 1), a1 - 1), (40, 40, 230), -1)
    paso = 5 if t_max <= 40 else 10 if t_max <= 90 else 20
    for s in range(0, int(t_max) + 1, paso):
        if px(s) < x1 - 40:
            cv2.putText(img, f"{s}", (px(s) - 5, alto - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(img, "t (s)", (x1 - 30, alto - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def demo(sistema, ruta, nombre, titulo, salida, alto=540, ancho_panel=440):
    cap = cv2.VideoCapture(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t_max = max(1.0, cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    cap.release()
    modelo = sistema.modelos[0]
    escritor, historia = None, []
    for m, s in recorrer(sistema, ruta, nombre):
        r = s["resultados"][modelo.nombre]
        historia.append((m.t, r["riesgo"], r["nivel"], r["alerta"]))
        img = m.imagen
        esc = alto / img.shape[0]
        img = cv2.resize(img, (int(round(img.shape[1] * esc / 2)) * 2, alto))
        for x1, y1, x2, y2, i, conf, _, _ in s["humo"]:
            cv2.rectangle(img, (int(x1 * esc), int(y1 * esc)), (int(x2 * esc), int(y2 * esc)), (170, 170, 170), 1)
        for x1, y1, x2, y2, i, conf, _, _ in s["fuego"]:
            cv2.rectangle(img, (int(x1 * esc), int(y1 * esc)), (int(x2 * esc), int(y2 * esc)), (0, 255, 255), 2)
            texto(img, f"fuego {int(i)} {conf:.2f}", int(x1 * esc), max(int(y1 * esc) - 4, 96), 0.45, (0, 255, 255))
        franja = img[:84].copy()
        img[:84] = (franja * 0.35).astype(np.uint8)
        texto(img, titulo, 10, 22, 0.55, (255, 255, 255))
        texto(img, f"nivel: {NIVELES[r['nivel']]}", 10, 50, 0.75, COLORES[r["nivel"]], 2)
        texto(img, f"riesgo {r['riesgo']:.2f}   t = {m.t:.1f} s", 10, 74, 0.5)
        if r["alerta"]:
            cv2.rectangle(img, (img.shape[1] - 150, 26), (img.shape[1] - 10, 62), (40, 40, 230), -1)
            texto(img, "ALERTA", img.shape[1] - 138, 52, 0.75, (255, 255, 255), 2)
            cv2.rectangle(img, (1, 1), (img.shape[1] - 2, img.shape[0] - 2), (40, 40, 230), 3)
        cuadro = np.hstack([img, panel(ancho_panel, alto, historia, t_max, modelo.umbral)])
        if escritor is None:
            escritor = cv2.VideoWriter(salida, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), sistema.p.fps, (cuadro.shape[1], cuadro.shape[0]))
        escritor.write(cuadro)
    if escritor is not None:
        escritor.release()
    return historia


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modo", choices=["fps", "demo", "verificar"], default="fps")
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    p.add_argument("--corridas", nargs="+", default=["gru_s42", "cfc_s42"])
    p.add_argument("--videos", nargs="+", default=[os.path.join("real_video", "real_video", "fire"),
                                                   os.path.join("real_video", "real_video", "non_fire")])
    p.add_argument("--archivos", nargs="*", default=[])
    p.add_argument("--titulos", nargs="*", default=[])
    p.add_argument("--nombres", nargs="*", default=[])
    p.add_argument("--conjunto", default="prueba")
    p.add_argument("--cantidad", type=int, default=5)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--salida", default=os.path.join("resultados", "prueba"))
    p.add_argument("--salida_demo", default=os.path.join("runs_fuego", "demo"))
    a = p.parse_args()

    with open(a.config, encoding="utf-8") as f:
        config = json.load(f)
    rutas = buscar_rutas(a.videos)
    archivos = a.archivos
    if not archivos:
        del_conjunto = sorted(r["archivo"] for r in leer_csv(a.particion) if r["conjunto"] == a.conjunto)
        archivos = random.Random(a.semilla).sample(del_conjunto, a.cantidad)
    archivos = [next((v for v in rutas if v == x or v.startswith(x)), x) for x in archivos]
    faltan = [v for v in archivos if v not in rutas]
    if faltan:
        raise SystemExit(f"videos no encontrados: {faltan}")
    sistema = Sistema(config, a.corridas if a.modo != "demo" else a.corridas[:1])

    if a.modo == "demo":
        os.makedirs(a.salida_demo, exist_ok=True)
        for n, v in enumerate(archivos):
            nombre = a.nombres[n] if n < len(a.nombres) else os.path.splitext(v)[0][:20]
            titulo = a.titulos[n] if n < len(a.titulos) else "sistema completo"
            salida = os.path.join(a.salida_demo, f"sistema_{nombre}.mp4")
            historia = demo(sistema, rutas[v], v, titulo, salida)
            niveles = [h[2] for h in historia]
            alertas = [h[0] for h in historia if h[3]]
            print(f"{v[:50]}: {len(historia)} fotogramas | niveles " + " ".join(f"{NIVELES[k]} {niveles.count(k)}" for k in range(3))
                  + f" | riesgo maximo {max(h[1] for h in historia):.3f} | primera alerta: {alertas[0] if alertas else 'ninguna'}"
                  f" | {salida} ({os.path.getsize(salida) / 1e6:.1f} mb)", flush=True)
        return

    if a.modo == "verificar":
        d = np.load(os.path.join(a.datos, f"{a.conjunto}.npz"))
        ini = d["seq_inicio"]
        for v in archivos:
            vivos = defaultdict(list)
            for m, s in recorrer(sistema, rutas[v], v):
                vivos[s["tramo"]].append(s["resultados"][a.corridas[0]])
            for j in np.flatnonzero(d["seq_archivo"] == v):
                a0, a1 = int(ini[j]), int(ini[j + 1])
                filas = vivos[int(d["seq_tramo"][j])]
                if len(filas) != a1 - a0:
                    print(f"{v[:40]} tramo {d['seq_tramo'][j]}: {len(filas)} fotogramas en vivo y {a1 - a0} guardados")
                    continue
                dx = np.abs(np.array([f["x"] for f in filas]) - d["x"][a0:a1])
                dc = np.abs(np.array([f["c"] for f in filas]) - d["c"][a0:a1])
                print(f"{v[:40]} tramo {d['seq_tramo'][j]}: {a1 - a0} fotogramas | diferencia maxima del vector {dx.max():.2e}"
                      f" (fotogramas con diferencia > 1e-3: {int((dx.max(axis=1) > 1e-3).sum())}) | del contexto {dc.max():.2e}"
                      f" (fotogramas > 1e-3: {int((dc.max(axis=1) > 1e-3).sum())})", flush=True)
            np.savez(os.path.join(a.salida, f"vivo_{os.path.splitext(v)[0][:30]}.npz"),
                     **{k: np.array([f[k] for t in sorted(vivos) for f in vivos[t]]) for k in ("p1", "p2", "riesgo", "tau", "nivel", "alerta")})
        return

    filas = []
    meta = {r["archivo"]: r for r in leer_csv("metadatos_videos.csv")} if os.path.exists("metadatos_videos.csv") else {}
    for v in archivos:
        acum = defaultdict(float)
        n, inicio, t_fin, alto, ancho = 0, time.perf_counter(), 0.0, 0, 0
        for m, s in recorrer(sistema, rutas[v], v):
            n += 1
            t_fin = m.t
            alto, ancho = m.imagen.shape[:2]
            for k, x in s["tiempos"].items():
                acum[k] += x
        total = time.perf_counter() - inicio
        comun = sum(acum[k] for k in ("lectura", "yolo", "camara", "seguimiento", "descriptores", "vector"))
        fila = {"archivo": v, "resolucion": f"{ancho}x{alto}", "fps_fuente": meta.get(v, {}).get("fps", ""), "duracion_s": round(t_fin, 2),
                "fotogramas": n, "s_total": round(total, 2)}
        for k in ("lectura", "yolo", "camara", "seguimiento", "descriptores", "vector"):
            fila[f"ms_{k}"] = 1000.0 * acum[k] / n
        for c in a.corridas:
            fila[f"ms_modelo_{c}"] = 1000.0 * acum[f"modelo:{c}"] / n
            fila[f"ms_decision_{c}"] = 1000.0 * acum[f"decision:{c}"] / n
            propio = comun + acum[f"modelo:{c}"] + acum[f"decision:{c}"]
            fila[f"fps_{c}"] = n / propio
            fila[f"tiempo_real_{c}"] = t_fin / propio
        filas.append(fila)
        print(f"{v[:45]} | {fila['resolucion']} | {n} fotogramas | " + " | ".join(f"{c}: {fila[f'fps_{c}']:.1f} fps" for c in a.corridas), flush=True)
    os.makedirs(a.salida, exist_ok=True)
    total_n = sum(f["fotogramas"] for f in filas)
    medio = lambda k: sum(f[k] * f["fotogramas"] for f in filas) / total_n
    lineas = ["# Velocidad del sistema completo en vivo", ""]
    lineas.append(f"Generado por `python sistema.py --modo fps`. {len(filas)} videos de {a.conjunto} elegidos al azar (semilla {a.semilla})."
                  " El sistema procesa fotograma a fotograma: lectura con muestreo a 10 fps, YOLOv8s (GPU), movimiento de camara y cortes,"
                  " ByteTrack, descriptores, `VectorEscena`, modelo temporal (CPU, un paso por fotograma) y decision.")
    lineas.append("")
    lineas.append("- fps = fotogramas procesados por segundo de computo. El sistema muestrea a 10 fps, asi que funciona en tiempo real si supera 10.")
    lineas.append("- Tiempo real = segundos de video por segundo de computo.")
    lineas.append("- La lectura incluye decodificar y descartar los fotogramas que el muestreo omite; por eso depende de la resolucion y de los fps de la fuente.")
    lineas.append("")
    lineas.append("| video | resolucion | fps de la fuente | duracion (s) | fotogramas | " + " | ".join(f"fps con {c} | tiempo real con {c}" for c in a.corridas) + " |")
    lineas.append("|---|---|---|---|---|" + "---|---|" * len(a.corridas))
    for n, f in enumerate(filas, 1):
        lineas.append(f"| {n}. {f['archivo'][:40]} | {f['resolucion']} | {f['fps_fuente']} | {f['duracion_s']} | {f['fotogramas']} | "
                      + " | ".join(f"{f[f'fps_{c}']:.1f} | {f[f'tiempo_real_{c}']:.2f}x" for c in a.corridas) + " |")
    lineas.append(f"| **todos (ponderado por fotogramas)** | | | {sum(f['duracion_s'] for f in filas):.1f} | {total_n} | "
                  + " | ".join(f"**{total_n / sum(f['fotogramas'] / f[f'fps_{c}'] for f in filas):.1f}** |"
                               f" {sum(f['duracion_s'] for f in filas) / sum(f['fotogramas'] / f[f'fps_{c}'] for f in filas):.2f}x"
                               for c in a.corridas) + " |")
    lineas.append("")
    lineas.append("Tiempo por etapa (ms por fotograma procesado):")
    lineas.append("")
    etapas = ["lectura", "yolo", "camara", "seguimiento", "descriptores", "vector"]
    claves = [f"ms_{k}" for k in etapas] + [f"ms_{x}_{c}" for c in a.corridas for x in ("modelo", "decision")]
    lineas.append("| video | lectura | YOLO | camara y cortes | seguimiento | descriptores | vector de escena | "
                  + " | ".join(f"modelo {c} | decision {c}" for c in a.corridas) + " |")
    lineas.append("|---|" + "---|" * len(claves))
    for n, f in enumerate(filas, 1):
        lineas.append(f"| {n} | " + " | ".join(f"{f[k]:.2f}" for k in claves) + " |")
    lineas.append("| **media** | " + " | ".join(f"**{medio(k):.2f}**" for k in claves) + " |")
    lineas.append("")
    with open(os.path.join(a.salida, "velocidad.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    with open(os.path.join(a.salida, "velocidad.json"), "w", encoding="utf-8") as f:
        json.dump({"videos": filas, "media_ms": {k: medio(k) for k in claves},
                   "fps": {c: total_n / sum(x["fotogramas"] / x[f"fps_{c}"] for x in filas) for c in a.corridas}}, f, indent=1)
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
