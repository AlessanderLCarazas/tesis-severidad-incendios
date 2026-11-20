import os
import re
import csv
import json
import time
import random
import tarfile
import argparse
import urllib.request
from collections import namedtuple, defaultdict
import numpy as np
import cv2

BASE = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/"
Muestra = namedtuple("Muestra", ["k", "frame", "t", "dt", "imagen"])
AGENTE = {"User-Agent": "Mozilla/5.0 (tesis academica)"}


def bajar(url, ruta=None, intentos=3):
    for i in range(intentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENTE), timeout=120) as r:
                datos = r.read()
            if ruta:
                with open(ruta, "wb") as f:
                    f.write(datos)
            return datos
        except Exception as e:
            if i == intentos - 1:
                raise
            time.sleep(5)


def secuencias():
    t = bajar(BASE + "index.html").decode("utf-8", errors="replace")
    return sorted(set(re.findall(r"href=([^/>\s]+)/index\.html", t)))


def imagenes(seq):
    t = bajar(f"{BASE}{seq}/index.html").decode("utf-8", errors="replace")
    nombres = sorted(set(n.replace("%2B", "+") for n in re.findall(r"href=([0-9]+_[-+%0-9B]+\.jpg)", t)))
    return [(n, int(n.split("_")[0]), int(n.split("_")[1].split(".")[0])) for n in nombres]


def revisar_imagen(datos):
    img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR).astype(np.float32)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    color = float((np.abs(r - g) + np.abs(g - b)).mean() / 2.0)
    brillo = float(img.mean())
    return color, brillo


def elegir(a):
    todas = secuencias()
    orden = list(todas)
    random.Random(a.semilla).shuffle(orden)
    filas, aceptadas = [], []
    for seq in orden:
        if len(aceptadas) >= a.n:
            break
        imgs = imagenes(seq)
        if not imgs:
            filas.append({"secuencia": seq, "estado": "sin imagenes"})
            continue
        ref = min(imgs, key=lambda x: abs(x[2]))
        color, brillo = revisar_imagen(bajar(f"{BASE}{seq}/{ref[0].replace('+', '%2B')}"))
        motivo = "blanco y negro" if color < a.umbral_color else "nocturna" if brillo < a.umbral_brillo else "aceptada"
        filas.append({"secuencia": seq, "estado": motivo, "imagenes": len(imgs), "color": round(color, 2), "brillo": round(brillo, 1),
                      "desfase_min": min(x[2] for x in imgs), "desfase_max": max(x[2] for x in imgs)})
        if motivo == "aceptada":
            aceptadas.append(seq)
        print(f"{seq}: {motivo} | color {color:.2f} | brillo {brillo:.1f} | {len(imgs)} imagenes", flush=True)
    os.makedirs(os.path.dirname(a.lista), exist_ok=True)
    with open(a.lista, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["secuencia", "estado", "imagenes", "color", "brillo", "desfase_min", "desfase_max"])
        w.writeheader()
        w.writerows(filas)
    print(f"secuencias en el indice: {len(todas)} | revisadas: {len(filas)} | aceptadas: {len(aceptadas)} | lista en {a.lista}")


def descargar(a):
    os.makedirs(a.datos, exist_ok=True)
    lista = [r["secuencia"] for r in csv.DictReader(open(a.lista, encoding="utf-8")) if r["estado"] == "aceptada"]
    tiempos = {}
    for seq in lista:
        ruta = os.path.join(a.datos, f"{seq}.tgz")
        t0 = time.time()
        if not os.path.exists(ruta):
            bajar(f"{BASE}Tar/{seq}.tgz", ruta)
        destino = os.path.join(a.datos, seq)
        if not os.path.exists(destino):
            with tarfile.open(ruta) as tar:
                miembros = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".jpg") and ".." not in m.name]
                os.makedirs(destino, exist_ok=True)
                for m in miembros:
                    m.name = os.path.basename(m.name)
                    tar.extract(m, destino)
        tiempos[seq] = {"segundos_descarga": time.time() - t0, "mb_tgz": os.path.getsize(ruta) / 1e6,
                        "mb_imagenes": sum(os.path.getsize(os.path.join(destino, x)) for x in os.listdir(destino)) / 1e6,
                        "imagenes": len(os.listdir(destino))}
        print(f"{seq}: {tiempos[seq]['mb_tgz']:.1f} mb | {tiempos[seq]['imagenes']} imagenes | {tiempos[seq]['segundos_descarga']:.0f} s", flush=True)
    with open(os.path.join(os.path.dirname(a.lista), "piloto_descarga.json"), "w", encoding="utf-8") as f:
        json.dump(tiempos, f, indent=1)


def leer_secuencia(carpeta):
    nombres = sorted((n for n in os.listdir(carpeta) if n.lower().endswith(".jpg")), key=lambda n: int(n.split("_")[0]))
    marcas = [int(n.split("_")[0]) for n in nombres]
    desfases = [int(n.split("_")[1].split(".")[0]) for n in nombres]
    paso = float(np.median(np.diff(marcas))) if len(marcas) > 1 else 60.0
    return nombres, marcas, desfases, paso


def correr(a):
    import torch
    from sistema import Sistema
    from salidas_congeladas import leer_config
    config = leer_config(a.config)
    sistema = Sistema(config, a.corridas)
    lista = [r["secuencia"] for r in csv.DictReader(open(a.lista, encoding="utf-8")) if r["estado"] == "aceptada"]
    filas, por_imagen = [], []
    for seq in lista:
        carpeta = os.path.join(a.datos, seq)
        nombres, marcas, desfases, paso = leer_secuencia(carpeta)
        sistema.iniciar()
        t0 = time.time()
        tray = {"fuego": defaultdict(int), "humo": defaultdict(int)}
        cortes, alertas = 0, defaultdict(list)
        for k, (n, marca, desfase) in enumerate(zip(nombres, marcas, desfases)):
            img = cv2.imread(os.path.join(carpeta, n))
            t = float(marca - marcas[0])
            dt = paso if k == 0 else float(marca - marcas[k - 1])
            m = Muestra(k, k, t, dt, img)
            det = sistema.yolo.predict(img, imgsz=sistema.p.img, conf=0.1, device=sistema.dispositivo, verbose=False)[0]
            det = det.boxes.data.cpu().numpy() if det.boxes is not None else np.zeros((0, 6), np.float32)
            humo = det[det[:, 5] == 1] if len(det) else det
            s = sistema.paso(m)
            cortes += int(bool(s["corte"]))
            for clase in ("fuego", "humo"):
                for fila in s[clase]:
                    tray[clase][(s["tramo"], int(fila[4]))] += 1
            for c, r in s["resultados"].items():
                alertas[c].append((desfase, r["alerta"], r["nivel"], r["riesgo"]))
            por_imagen.append({"secuencia": seq, "imagen": n, "desfase_s": desfase, "humo_visible": int(desfase >= 0),
                               "humo_conf_max": round(float(humo[:, 4].max()), 4) if len(humo) else 0.0,
                               "cajas_humo_025": int((humo[:, 4] >= 0.25).sum()) if len(humo) else 0,
                               "cajas_fuego_025": int(((det[:, 5] == 0) & (det[:, 4] >= 0.25)).sum()) if len(det) else 0})
        seg = time.time() - t0
        fila = {"secuencia": seq, "imagenes": len(nombres), "paso_s": paso, "segundos_proceso": round(seg, 1), "cortes": cortes,
                "trayectorias_humo": len(tray["humo"]), "trayectorias_humo_2_o_mas": sum(v >= 2 for v in tray["humo"].values()),
                "trayectorias_fuego": len(tray["fuego"]), "trayectorias_fuego_2_o_mas": sum(v >= 2 for v in tray["fuego"].values()),
                "largo_max_trayectoria_humo": max(tray["humo"].values()) if tray["humo"] else 0}
        for c, v in alertas.items():
            antes = [x for x in v if x[0] < 0]
            despues = [x for x in v if x[0] >= 0]
            fila[f"{c}_alerta_antes"] = int(any(x[1] for x in antes))
            primera = next((x[0] for x in despues if x[1]), None)
            fila[f"{c}_min_hasta_alerta"] = None if primera is None else primera / 60.0
        filas.append(fila)
        print(f"{seq}: {len(nombres)} imagenes | {seg:.0f} s | cortes {cortes} | trayectorias de humo {fila['trayectorias_humo']}"
              f" ({fila['trayectorias_humo_2_o_mas']} con 2 o mas)", flush=True)
    with open(os.path.join(os.path.dirname(a.lista), "piloto_secuencias.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    with open(os.path.join(os.path.dirname(a.lista), "piloto_imagenes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(por_imagen[0].keys()))
        w.writeheader()
        w.writerows(por_imagen)
    print("listo")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("accion", choices=["elegir", "descargar", "correr"])
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--umbral_color", type=float, default=2.0)
    p.add_argument("--umbral_brillo", type=float, default=50.0)
    p.add_argument("--lista", default=os.path.join("resultados", "figlib", "piloto_lista.csv"))
    p.add_argument("--datos", default=os.path.join("..", "FIgLib_piloto"))
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    p.add_argument("--corridas", nargs="+", default=["gru_s42", "cfc_s42", "lstm_s42"])
    a = p.parse_args()
    {"elegir": elegir, "descargar": descargar, "correr": correr}[a.accion](a)


if __name__ == "__main__":
    main()
