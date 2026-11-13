import os
import sys
import glob
import json
import time
import hashlib
import argparse
import subprocess


def leer_json(ruta):
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def lanzar(a, config, modelo, semilla, registros):
    nombre = f"{modelo}_s{semilla}"
    if leer_json(os.path.join(a.salida, nombre, "metricas.json")):
        return None
    pm = config["por_modelo"][modelo]
    if pm["unidad_dt_s"] not in (None, 1.0):
        raise SystemExit(f"{modelo} usa otra unidad de tiempo; este script no la contempla")
    cmd = [sys.executable, "entrenar_robusto.py", "--modelo", modelo, "--semilla", str(semilla), "--lr", str(pm["tasa"]),
           "--factor_incipiente", str(pm["factor_incipiente"]), "--tope_peso", str(pm["tope_peso"]), "--epocas", str(pm["epocas"]),
           "--lote", str(pm["lote"]), "--decaimiento", str(pm["decaimiento"]), "--recorte", str(pm["recorte"]),
           "--unidades", str(pm["unidades"]), "--dropout", str(pm["dropout"]), "--q_max", str(a.q_max),
           "--pesos", a.pesos, "--salida", a.salida]
    entorno = dict(os.environ)
    if modelo in a.en_cpu:
        cmd += ["--dispositivo", "cpu"]
        entorno["OMP_NUM_THREADS"] = entorno["MKL_NUM_THREADS"] = str(a.hilos)
    registro = open(os.path.join(registros, f"{nombre}.txt"), "w", encoding="utf-8")
    print(f"entrenando {nombre} ({'cpu' if modelo in a.en_cpu else 'gpu'})", flush=True)
    return subprocess.Popen(cmd, stdout=registro, stderr=subprocess.STDOUT, env=entorno), registro


def entrenar(a, config):
    registros = os.path.join(a.pesos, "registros")
    os.makedirs(registros, exist_ok=True)
    tareas = [(m, s) for m in a.modelos for s in a.semillas]
    activos = []
    for m, s in [t for t in tareas if t[0] in a.en_cpu]:
        r = lanzar(a, config, m, s, registros)
        if r:
            activos.append(r)
    for m, s in [t for t in tareas if t[0] not in a.en_cpu]:
        r = lanzar(a, config, m, s, registros)
        if r:
            r[0].wait()
            r[1].close()
            if r[0].returncode:
                raise SystemExit(f"fallo el entrenamiento de {m}_s{s}; ver {registros}")
    for proceso, registro in activos:
        proceso.wait()
        registro.close()
        if proceso.returncode:
            raise SystemExit(f"fallo un entrenamiento en cpu; ver {registros}")
    faltan = [f"{m}_s{s}" for m, s in tareas if not leer_json(os.path.join(a.salida, f"{m}_s{s}", "metricas.json"))]
    if faltan:
        raise SystemExit(f"faltan corridas: {faltan}")


def calibrar(a):
    patron = os.path.join(a.pesos, "*_s*")
    if not glob.glob(os.path.join(patron, "mejor.pt")):
        raise SystemExit(f"no hay modelos en {a.pesos}")
    cmd = [sys.executable, "fase_e_elegida.py", "--carpetas", patron, "--sin_referencias", "--salida", a.fase_e]
    if subprocess.call(cmd):
        raise SystemExit("fallo la calibracion")


def registrar(a, config):
    corridas = {}
    for carpeta in sorted(glob.glob(os.path.join(a.pesos, "*_s*"))):
        nombre = os.path.basename(carpeta)
        pesos = os.path.join(carpeta, "mejor.pt")
        fase = leer_json(os.path.join(a.fase_e, f"{nombre}.json"))
        r = leer_json(os.path.join(a.salida, nombre, "metricas.json"))
        if fase is None or r is None:
            raise SystemExit(f"falta la calibracion o las metricas de {nombre}")
        if fase["pesos"] != pesos.replace(os.sep, "/"):
            raise SystemExit(f"la calibracion de {nombre} no corresponde a {pesos}")
        congelada = config["corridas"].get(nombre, {})
        corridas[nombre] = {"modelo": r["modelo"], "semilla": r["semilla"], "archivo_pesos": pesos.replace(os.sep, "/"), "sha256": sha256(pesos),
                            "mejor_epoca": r["mejor_epoca"], "tasa": r["hiperparametros"]["lr"],
                            "factor_incipiente": r["hiperparametros"]["factor_incipiente"], "q_max": r["aumento"]["q_max"],
                            "fraccion_quitada_media": r["aumento"]["fraccion_quitada_media"], "minutos": r["s_total"] / 60.0,
                            "umbral_alerta": {k: v["umbral"] for k, v in fase["alerta"].items()},
                            "validacion": {"f1_antes": fase["antes"]["f1"], "f1_final": fase["histeresis"]["f1"], "kappa_final": fase["histeresis"]["kappa"],
                                           "f1_incipiente_final": fase["histeresis"]["f1_clase"][1]},
                            "congelado": {"umbral_alerta": congelada.get("umbral_alerta"), "f1_final_validacion": congelada.get("f1_final_validacion")}}
    registro = {"fecha": time.strftime("%Y-%m-%d"),
                "nota": "modelos de la parte 2 de robustez (fotogramas quitados al entrenar); no reemplazan a la configuracion congelada",
                "aumento": f"en cada ventana q ~ uniforme(0, {a.q_max:g}); cada fotograma salvo el primero se quita con probabilidad q;"
                           " dt recalculado y columna dt del contexto renormalizada con la normalizacion congelada",
                "calibracion": "umbral conformal recalculado con los 52 videos de calibracion sin fuego o controlados, completos",
                "corridas": corridas}
    os.makedirs(os.path.dirname(a.registro), exist_ok=True)
    with open(a.registro, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=1)
    print(f"registro en {a.registro} | corridas: {len(corridas)}")
    for n, c in corridas.items():
        print(f"{n}: epoca {c['mejor_epoca']} | f1 final en validacion {c['validacion']['f1_final']:.3f}"
              f" (congelado {c['congelado']['f1_final_validacion'] or float('nan'):.3f}) | umbral 0.1 {c['umbral_alerta']['0.1']:.3f}"
              f" | {c['minutos']:.1f} min", flush=True)


def evaluar(a):
    cmd = [sys.executable, "evaluar_robustez.py", "--modelos", "robustos", "--robustos", a.registro]
    if subprocess.call(cmd):
        raise SystemExit("fallo la evaluacion")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--etapa", choices=["entrenar", "calibrar", "registrar", "evaluar", "todo"], default="todo")
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    p.add_argument("--modelos", nargs="+", default=["gru", "lstm", "cfc"])
    p.add_argument("--semillas", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--q_max", type=float, default=0.75)
    p.add_argument("--en_cpu", nargs="*", default=["cfc"])
    p.add_argument("--hilos", type=int, default=4)
    p.add_argument("--pesos", default=os.path.join("runs_fuego", "modelos_robustos"))
    p.add_argument("--salida", default=os.path.join("resultados", "robustez", "modelos"))
    p.add_argument("--fase_e", default=os.path.join("resultados", "robustez", "fase_e"))
    p.add_argument("--registro", default=os.path.join("resultados", "robustez", "modelos_robustos.json"))
    a = p.parse_args()

    with open(a.config, encoding="utf-8") as f:
        config = json.load(f)
    inicio = time.time()
    etapas = ["entrenar", "calibrar", "registrar", "evaluar"] if a.etapa == "todo" else [a.etapa]
    for e in etapas:
        print(f"etapa {e}", flush=True)
        if e == "entrenar":
            entrenar(a, config)
        elif e == "calibrar":
            calibrar(a)
        elif e == "registrar":
            registrar(a, config)
        else:
            evaluar(a)
        print(f"etapa {e} terminada | llevo {(time.time() - inicio) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
