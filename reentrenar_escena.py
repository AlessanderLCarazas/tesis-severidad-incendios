import os
import sys
import json
import time
import shutil
import argparse
import subprocess


def leer_metricas(carpeta, nombre):
    ruta = os.path.join(carpeta, nombre, "metricas.json")
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def lanzar(a, modelo, semilla, tasa, sub, registros):
    nombre = f"{modelo}_s{semilla}"
    if leer_metricas(os.path.join(a.salida, sub), nombre):
        return None
    cmd = [sys.executable, "entrenar_escena.py", "--modelo", modelo, "--semilla", str(semilla), "--lr", str(tasa),
           "--epocas", str(a.epocas), "--paciencia", str(a.epocas), "--datos", a.datos,
           "--pesos", os.path.join(a.pesos, sub), "--salida", os.path.join(a.salida, sub)]
    entorno = dict(os.environ)
    if modelo in a.en_cpu:
        cmd += ["--dispositivo", "cpu"]
        entorno["OMP_NUM_THREADS"] = entorno["MKL_NUM_THREADS"] = str(a.hilos)
    registro = open(os.path.join(registros, f"{sub.replace(os.sep, '_').replace('/', '_')}_{nombre}.txt"), "w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=registro, stderr=subprocess.STDOUT, env=entorno), registro


def correr(tareas, a, registros):
    gpu = [t for t in tareas if t[0] not in a.en_cpu]
    cpu = [t for t in tareas if t[0] in a.en_cpu]
    activos = []
    for modelo, semilla, tasa, sub in cpu:
        r = lanzar(a, modelo, semilla, tasa, sub, registros)
        if r:
            activos.append(r)
    for modelo, semilla, tasa, sub in gpu:
        r = lanzar(a, modelo, semilla, tasa, sub, registros)
        if r:
            r[0].wait()
            r[1].close()
    for proceso, registro in activos:
        proceso.wait()
        registro.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelos", nargs="+", default=["gru", "lstm", "cfc"])
    p.add_argument("--tasas", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    p.add_argument("--semilla_tasa", type=int, default=42)
    p.add_argument("--semillas", nargs="+", type=int, default=[43, 44])
    p.add_argument("--epocas", type=int, default=60)
    p.add_argument("--en_cpu", nargs="*", default=["cfc"])
    p.add_argument("--hilos", type=int, default=4)
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--pesos", default=os.path.join("runs_fuego", "modelos_escena_v2"))
    p.add_argument("--salida", default=os.path.join("resultados", "modelos_escena_v2"))
    a = p.parse_args()

    registros = os.path.join(a.pesos, "registros")
    os.makedirs(registros, exist_ok=True)
    os.makedirs(a.salida, exist_ok=True)
    inicio = time.time()
    subs = {t: os.path.join("tasas", f"lr_{t:g}") for t in a.tasas}
    correr([(m, a.semilla_tasa, t, subs[t]) for m in a.modelos for t in a.tasas], a, registros)
    print(f"busqueda de tasa terminada en {(time.time() - inicio) / 60:.1f} min", flush=True)

    eleccion = {}
    for m in a.modelos:
        nombre = f"{m}_s{a.semilla_tasa}"
        puntajes = {}
        for t in a.tasas:
            r = leer_metricas(os.path.join(a.salida, subs[t]), nombre)
            if r:
                puntajes[t] = {"f1_suavizado": r["validacion"]["f1_suavizado"], "f1": r["validacion"]["f1_escena"],
                               "kappa": r["validacion"]["todo"]["kappa"], "mejor_epoca": r["mejor_epoca"],
                               "minutos": r["s_total"] / 60.0}
        if not puntajes:
            raise SystemExit(f"no hay corridas de {m}")
        mejor = max(puntajes, key=lambda t: puntajes[t]["f1_suavizado"])
        eleccion[m] = {"elegida": mejor, "tasas": {f"{t:g}": v for t, v in puntajes.items()}}
        for base in (a.salida, a.pesos):
            origen, destino = os.path.join(base, subs[mejor], nombre), os.path.join(base, "final", nombre)
            if os.path.exists(origen) and not os.path.exists(destino):
                shutil.copytree(origen, destino)
        print(f"{m}: tasa elegida {mejor:g} | " + " | ".join(f"{t:g}: {v['f1_suavizado']:.4f}" for t, v in puntajes.items()), flush=True)
    with open(os.path.join(a.salida, "tasas.json"), "w", encoding="utf-8") as f:
        json.dump(eleccion, f, indent=1)

    correr([(m, s, eleccion[m]["elegida"], "final") for m in a.modelos for s in a.semillas], a, registros)
    print(f"todo terminado en {(time.time() - inicio) / 60:.1f} min | resultados en {os.path.join(a.salida, 'final')}")


if __name__ == "__main__":
    main()
