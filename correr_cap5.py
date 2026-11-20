import os
import sys
import json
import time
import argparse
import subprocess
from variantes_cap5 import VARIANTES

CONFIG = {"cfc": (1e-3, 1.0), "gru": (3e-3, 2.0), "lstm": (1e-3, 1.0)}
PESOS_MAMBA = {"A": 1.0, "B": 2.0}
TASAS_MAMBA = [1e-3, 3e-3]


def leer(ruta):
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def metricas(a, grupo, modelo, semilla):
    return leer(os.path.join(a.salida, grupo, f"{modelo}_s{semilla}", "metricas.json"))


def lanzar(a, modelo, variante, semilla, lr, factor, grupo, dispositivo):
    if metricas(a, grupo, modelo, semilla):
        return None
    nombre = f"{grupo.replace('/', '__')}__{modelo}_s{semilla}"
    cmd = [sys.executable, "entrenar_cap5.py", "--modelo", modelo, "--variante", variante, "--semilla", str(semilla), "--lr", str(lr),
           "--factor_incipiente", str(factor), "--dispositivo", dispositivo, "--pesos", os.path.join(a.pesos, grupo),
           "--salida", os.path.join(a.salida, grupo)]
    entorno = dict(os.environ)
    if dispositivo == "cpu":
        entorno["OMP_NUM_THREADS"] = entorno["MKL_NUM_THREADS"] = str(a.hilos)
        entorno["CUDA_VISIBLE_DEVICES"] = ""
    registro = open(os.path.join(a.pesos, "registros", f"{nombre}.txt"), "w", encoding="utf-8")
    print(f"{time.strftime('%H:%M:%S')} inicia {nombre} ({dispositivo})", flush=True)
    return subprocess.Popen(cmd, stdout=registro, stderr=subprocess.STDOUT, env=entorno), registro, nombre


def esperar(proceso):
    p, registro, nombre = proceso
    p.wait()
    registro.close()
    estado = "ok" if p.returncode == 0 else f"error {p.returncode}"
    print(f"{time.strftime('%H:%M:%S')} termina {nombre}: {estado}", flush=True)
    if p.returncode:
        raise SystemExit(f"fallo {nombre}; ver el registro")


def secuencial(a, tareas):
    for t in tareas:
        r = lanzar(a, *t)
        if r:
            esperar(r)


def paralelo(a, tareas):
    pendientes, activos = list(tareas), []
    while pendientes or activos:
        while pendientes and len(activos) < a.procesos:
            r = lanzar(a, *pendientes.pop(0))
            if r:
                activos.append(r)
        for r in list(activos):
            if r[0].poll() is not None:
                esperar(r)
                activos.remove(r)
        time.sleep(5)


def cola_gpu(a):
    ablaciones = [("gru", v, s, *CONFIG["gru"], f"ablaciones/{v}", "cuda") for v in VARIANTES if v != "completo" for s in (42, 43, 44)]
    semillas = [(m, "completo", s, *CONFIG[m], "semillas", "cuda") for m in ("gru", "lstm") for s in (45, 46)]
    secuencial(a, ablaciones + semillas)
    secuencial(a, [("mamba", "completo", 42, lr, f, f"mamba_busqueda/{p}_lr{lr:g}", "cuda") for p, f in PESOS_MAMBA.items() for lr in TASAS_MAMBA])
    eleccion = {}
    for p, f in PESOS_MAMBA.items():
        f1s = {lr: metricas(a, f"mamba_busqueda/{p}_lr{lr:g}", "mamba", 42)["validacion"]["f1_suavizado"] for lr in TASAS_MAMBA}
        lr = max(f1s, key=f1s.get)
        grupo = f"mamba_busqueda/{p}_lr{lr:g}"
        secuencial(a, [("mamba", "completo", s, lr, f, grupo, "cuda") for s in (43, 44)])
        finales = [metricas(a, grupo, "mamba", s)["fase_e"]["validacion_final"]["f1"] for s in (42, 43, 44)]
        eleccion[p] = {"factor_incipiente": f, "f1_suavizado_semilla_42": {f"{k:g}": v for k, v in f1s.items()}, "tasa": lr, "grupo": grupo,
                       "f1_final_validacion": finales, "media": sum(finales) / 3.0}
    elegido = max(eleccion, key=lambda p: eleccion[p]["media"])
    e = eleccion[elegido]
    secuencial(a, [("mamba", "completo", s, e["tasa"], e["factor_incipiente"], e["grupo"], "cuda") for s in (45, 46)])
    with open(os.path.join(a.salida, "mamba_eleccion.json"), "w", encoding="utf-8") as f:
        json.dump({"opciones": eleccion, "elegido": elegido, "grupo": e["grupo"]}, f, indent=1)
    print(f"mamba: peso {elegido}, tasa {e['tasa']:g}", flush=True)


def cola_cpu(a):
    tareas = [("cfc", "completo", s, *CONFIG["cfc"], "semillas", "cpu") for s in (45, 46)]
    tareas += [("cfc", v, s, *CONFIG["cfc"], f"ablaciones/{v}", "cpu") for v in VARIANTES if v != "completo" for s in (42, 43, 44)]
    paralelo(a, tareas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cola", choices=["gpu", "cpu"], required=True)
    p.add_argument("--procesos", type=int, default=3)
    p.add_argument("--hilos", type=int, default=2)
    p.add_argument("--pesos", default=os.path.join("runs_fuego", "cap5"))
    p.add_argument("--salida", default=os.path.join("resultados", "cap5", "entrenamientos"))
    a = p.parse_args()
    os.makedirs(os.path.join(a.pesos, "registros"), exist_ok=True)
    inicio = time.time()
    (cola_gpu if a.cola == "gpu" else cola_cpu)(a)
    print(f"cola {a.cola} terminada en {(time.time() - inicio) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
