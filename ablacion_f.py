import argparse
import csv
import os
import subprocess
import sys
import time

from correr_cap5 import CONFIG


def registrar(a, texto):
    linea = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {texto}"
    print(linea, flush=True)
    with open(os.path.join(a.registros, "ablacion_f.txt"), "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def paso(a, nombre, cmd, intentos):
    entorno = dict(os.environ, PYTHONUNBUFFERED="1")
    for i in range(1, intentos + 1):
        registrar(a, f"inicia {nombre} (intento {i} de {intentos})")
        inicio = time.time()
        with open(os.path.join(a.registros, f"ablacion_f__{nombre}.txt"), "a", encoding="utf-8") as reg:
            reg.write(f"\n=== intento {i}: {' '.join(cmd)}\n")
            reg.flush()
            codigo = subprocess.call(cmd, stdout=reg, stderr=subprocess.STDOUT, env=entorno)
        registrar(a, f"termina {nombre}: codigo {codigo} en {(time.time() - inicio) / 60:.1f} min")
        if codigo == 0:
            return
        time.sleep(a.espera)
    registrar(a, f"fallo {nombre} despues de {intentos} intentos")
    sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--descriptores", default=os.path.join("runs_fuego", "descriptores_v3_sin_comp"))
    p.add_argument("--escena", default=os.path.join("runs_fuego", "escena_v2_sin_comp"))
    p.add_argument("--resumen", default=os.path.join("resultados", "escena_v2_sin_comp"))
    p.add_argument("--pesos", default=os.path.join("runs_fuego", "cap5", "ablaciones", "sin_compensacion"))
    p.add_argument("--salida", default=os.path.join("resultados", "cap5", "entrenamientos", "ablaciones", "sin_compensacion"))
    p.add_argument("--registros", default=os.path.join("runs_fuego", "cap5", "registros"))
    p.add_argument("--semillas", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--intentos", type=int, default=3)
    p.add_argument("--intentos_descriptores", type=int, default=6)
    p.add_argument("--espera", type=float, default=30.0)
    a = p.parse_args()
    os.makedirs(a.registros, exist_ok=True)
    inicio = time.time()
    py = sys.executable
    registrar(a, "inicia la ablacion f (sin compensacion de camara)")
    paso(a, "descriptores", [py, "descriptores_v3.py", "--sin_compensacion", "--salida", a.descriptores], a.intentos_descriptores)
    with open(os.path.join(a.descriptores, "procesados.csv"), encoding="utf-8") as f:
        registrar(a, f"videos procesados: {sum(1 for _ in csv.DictReader(f))}")
    if os.path.exists(os.path.join(a.resumen, "resumen.md")):
        registrar(a, "escena ya calculada")
    else:
        paso(a, "escena", [py, "escena_v2.py", "--entrada", a.descriptores, "--salida", a.escena, "--resumen", a.resumen], a.intentos)
    norm_c = os.path.join(a.resumen, "normalizacion_contexto.json")
    for m, dispositivo in (("gru", "cuda"), ("cfc", "cpu")):
        lr, factor = CONFIG[m]
        for s in a.semillas:
            nombre = f"{m}_s{s}"
            if os.path.exists(os.path.join(a.salida, nombre, "metricas.json")):
                registrar(a, f"ya esta {nombre}")
                continue
            paso(a, nombre, [py, "entrenar_cap5.py", "--modelo", m, "--variante", "completo", "--semilla", str(s), "--lr", str(lr),
                             "--factor_incipiente", str(factor), "--dispositivo", dispositivo, "--datos", a.escena,
                             "--normalizacion_contexto", norm_c, "--pesos", a.pesos, "--salida", a.salida], a.intentos)
    paso(a, "evaluacion", [py, "evaluacion_cap5.py", "--reusar_costo", "--sin_figuras"], a.intentos)
    registrar(a, f"ablacion f terminada en {(time.time() - inicio) / 3600:.2f} h")


if __name__ == "__main__":
    main()
