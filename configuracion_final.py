import os
import sys
import glob
import json
import time
import shutil
import hashlib
import argparse
import subprocess

RAIZ_SUB = {"tasas": "tasas", "riesgo": "riesgo", "final": "final"}


def leer_json(ruta):
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def metricas(a, sub, nombre):
    return leer_json(os.path.join(a.salida, sub, nombre, "metricas.json"))


def copiar(a, sub_origen, sub_destino, nombre):
    for base in (a.salida, a.pesos):
        origen, destino = os.path.join(base, sub_origen, nombre), os.path.join(base, sub_destino, nombre)
        if os.path.exists(origen) and not os.path.exists(destino):
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            shutil.copytree(origen, destino)


def lanzar(a, modelo, semilla, tasa, riesgo, sub, registros):
    nombre = f"{modelo}_s{semilla}"
    if metricas(a, sub, nombre):
        return None
    cmd = [sys.executable, "entrenar_final.py", "--modelo", modelo, "--semilla", str(semilla), "--lr", str(tasa),
           "--epocas", str(a.epocas), "--riesgo", riesgo, "--horizonte", str(a.horizonte),
           "--factor_incipiente", str(a.factor_incipiente), "--datos", a.datos,
           "--pesos", os.path.join(a.pesos, sub), "--salida", os.path.join(a.salida, sub)]
    entorno = dict(os.environ)
    if modelo in a.en_cpu:
        cmd += ["--dispositivo", "cpu"]
        entorno["OMP_NUM_THREADS"] = entorno["MKL_NUM_THREADS"] = str(a.hilos)
    registro = open(os.path.join(registros, f"{sub.replace(os.sep, '_').replace('/', '_')}_{nombre}.txt"), "w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=registro, stderr=subprocess.STDOUT, env=entorno), registro


def correr(tareas, a, registros):
    activos = []
    for t in [t for t in tareas if t[0] in a.en_cpu]:
        r = lanzar(a, *t, registros)
        if r:
            activos.append(r)
    for t in [t for t in tareas if t[0] not in a.en_cpu]:
        r = lanzar(a, *t, registros)
        if r:
            r[0].wait()
            r[1].close()
    for proceso, registro in activos:
        proceso.wait()
        registro.close()


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def etapa_tasas(a, registros):
    subs = {t: os.path.join("tasas", f"lr_{t:g}") for t in a.tasas}
    correr([(m, a.semilla_tasa, t, "escalon", subs[t]) for m in a.modelos for t in a.tasas], a, registros)
    ruta = os.path.join(a.salida, "tasas.json")
    eleccion = leer_json(ruta) or {}
    for m in a.modelos:
        nombre = f"{m}_s{a.semilla_tasa}"
        puntajes = {}
        for t in a.tasas:
            r = metricas(a, subs[t], nombre)
            if r:
                puntajes[t] = {"f1_suavizado": r["validacion"]["f1_suavizado"], "f1": r["validacion"]["f1_escena"],
                               "f1_incipiente": r["validacion"]["todo"]["f1_clase"][1], "kappa": r["validacion"]["todo"]["kappa"],
                               "mejor_epoca": r["mejor_epoca"], "minutos": r["s_total"] / 60.0}
        if not puntajes:
            raise SystemExit(f"no hay corridas de {m}")
        mejor = max(puntajes, key=lambda t: puntajes[t]["f1_suavizado"])
        eleccion[m] = {"elegida": mejor, "tasas": {f"{t:g}": v for t, v in puntajes.items()}}
        print(f"{m}: tasa elegida {mejor:g} | " + " | ".join(f"{t:g}: {v['f1_suavizado']:.4f}" for t, v in puntajes.items()), flush=True)
    actual = leer_json(ruta) or {}
    actual.update({m: eleccion[m] for m in a.modelos})
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(actual, f, indent=1)


def tasa_de(a, modelo):
    eleccion = leer_json(os.path.join(a.salida, "tasas.json")) or {}
    if modelo not in eleccion:
        raise SystemExit(f"falta la tasa de {modelo}; corre primero la etapa tasas")
    return float(eleccion[modelo]["elegida"])


def etapa_riesgo(a, registros):
    m = a.modelo_riesgo
    tasa = tasa_de(a, m)
    copiar(a, os.path.join("tasas", f"lr_{tasa:g}"), os.path.join("riesgo", "escalon"), f"{m}_s{a.semilla_tasa}")
    semillas = [a.semilla_tasa] + a.semillas
    correr([(m, s, tasa, r, os.path.join("riesgo", r)) for r in ("escalon", "anticipado") for s in semillas], a, registros)


def etapa_final(a, registros):
    semillas = [a.semilla_tasa] + a.semillas
    tareas = []
    for m in a.modelos:
        tasa = tasa_de(a, m)
        for s in semillas:
            nombre = f"{m}_s{s}"
            if m == a.modelo_riesgo:
                copiar(a, os.path.join("riesgo", a.riesgo), "final", nombre)
            elif a.riesgo == "escalon" and s == a.semilla_tasa:
                copiar(a, os.path.join("tasas", f"lr_{tasa:g}"), "final", nombre)
            tareas.append((m, s, tasa, a.riesgo, "final"))
    correr(tareas, a, registros)


def etapa_congelar(a):
    tasas = leer_json(os.path.join(a.salida, "tasas.json"))
    modelos = {}
    for ruta in sorted(glob.glob(os.path.join(a.salida, "final", "*", "metricas.json"))):
        r = leer_json(ruta)
        nombre = os.path.basename(os.path.dirname(ruta))
        pesos = os.path.join(a.pesos, "final", nombre, "mejor.pt")
        fase = leer_json(os.path.join(a.fase_e, f"{nombre}.json"))
        modelos[nombre] = {"modelo": r["modelo"], "semilla": r["semilla"], "tasa": r["hiperparametros"]["lr"],
                           "mejor_epoca": r["mejor_epoca"], "riesgo": r["riesgo"], "pesos_nivel": r["pesos_nivel"],
                           "archivo_pesos": pesos.replace(os.sep, "/"), "sha256": sha256(pesos),
                           "umbral_alerta": {k: v["umbral"] for k, v in fase["alerta"].items()}}
        if r["riesgo"] != a.riesgo:
            raise SystemExit(f"{nombre} se entreno con riesgo {r['riesgo']} y no con {a.riesgo}")
    referencias = {}
    for ruta in sorted(glob.glob(os.path.join(a.pesos, "referencias", "*.joblib"))):
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        fase = leer_json(os.path.join(a.fase_e, f"{nombre}.json"))
        referencias[nombre] = {"archivo": ruta.replace(os.sep, "/"), "sha256": sha256(ruta),
                               "umbral_alerta": {k: v["umbral"] for k, v in fase["alerta"].items()}}
    primero = leer_json(sorted(glob.glob(os.path.join(a.salida, "final", "*", "metricas.json")))[0])
    config = {
        "fecha": time.strftime("%Y-%m-%d"),
        "nota": "configuracion congelada antes de la prueba; no se cambia nada despues",
        "datos": {"carpeta": a.datos.replace(os.sep, "/"), "vector_escena": "vector_escena.VectorEscena (34 valores, detecciones a confianza 0.25)",
                  "normalizacion": ["resultados/escena_v2/normalizacion.json", "resultados/escena_v2/normalizacion_contexto.json"],
                  "ventana": 300, "paso_ventana": 150, "particion": "resultados/particion/particion.csv",
                  "metas": "resultados/particion/metas.csv", "fuente_etiquetas": "ia"},
        "modelo": {"unidades": primero["unidades"], "dropout": primero["dropout"], "salidas": "coral k=3, riesgo, log(1 + tau)",
                   "ponderacion": "kendall"},
        "entrenamiento": {"epocas": primero["hiperparametros"]["epocas"], "lote": primero["hiperparametros"]["lote"],
                          "decaimiento": primero["hiperparametros"]["decaimiento"], "recorte": primero["hiperparametros"]["recorte"],
                          "plan_tasa": "coseno", "seleccion_epoca": "f1 macro de escena por segundo en validacion, suavizado con 3 epocas",
                          "tasas_probadas": a.tasas, "semilla_tasa": a.semilla_tasa, "semillas": [a.semilla_tasa] + a.semillas,
                          "tasa_elegida": {m: v["elegida"] for m, v in tasas.items()}, "busqueda_tasa": tasas,
                          "dispositivo": {"cfc": "cpu", "gru": "cuda", "lstm": "cuda"}},
        "pesos_nivel": {"formula": "min(1 / (3 f_k), tope) y luego factor para el incipiente", "tope": primero["hiperparametros"]["tope_peso"],
                        "factor_incipiente": primero["hiperparametros"]["factor_incipiente"], "valores": primero["pesos_nivel"],
                        "frecuencias": primero["frecuencia_niveles"]},
        "meta_riesgo": {"tipo": a.riesgo, "escalon": [0.0, 0.5, 1.0],
                        "anticipado": f"max(escalon, 1 - tau / {a.horizonte:g}) recortado a [0, 1] antes del cambio, solo en videos que llegan a incontrolado",
                        "horizonte_s": a.horizonte},
        "tau": {"meta": "log(1 + tau)", "perdida": "huber delta 1", "mascara": "solo videos con cambio a incontrolado (tau_mascara = 1)"},
        "decision": {"umbral_coral": [0.5, 0.5], "histeresis_s": 2.0, "alfa": 0.10, "alfa_secundario": 0.05,
                     "puntuacion_conformal": "riesgo maximo por video en los videos de calibracion sin fuego o controlados",
                     "factor_desactivacion": 0.8, "permanencia_alerta_s": 2.0,
                     "nivel_por_segundo": "mayoria de los fotogramas del segundo"},
        "referencias_sin_tiempo": {"entrada": "media por segundo del vector de escena", "semilla": 42, "class_weight": "balanced",
                                   "riesgo": "0.5 P(incipiente) + P(incontrolado)", "modelos": referencias},
        "modelos": modelos,
        "resultados_validacion": a.fase_e.replace(os.sep, "/") + "/resumen.md"}
    with open(a.config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=1)
    print(f"configuracion congelada en {a.config} | modelos: {len(modelos)} | referencias: {len(referencias)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--etapa", choices=["tasas", "riesgo", "final", "congelar"], required=True)
    p.add_argument("--modelos", nargs="+", default=["gru", "lstm", "cfc"])
    p.add_argument("--tasas", nargs="+", type=float, default=[1e-3, 3e-3])
    p.add_argument("--semilla_tasa", type=int, default=42)
    p.add_argument("--semillas", nargs="+", type=int, default=[43, 44])
    p.add_argument("--epocas", type=int, default=60)
    p.add_argument("--factor_incipiente", type=float, default=2.0)
    p.add_argument("--modelo_riesgo", default="gru")
    p.add_argument("--riesgo", choices=["escalon", "anticipado"], default="escalon")
    p.add_argument("--horizonte", type=float, default=60.0)
    p.add_argument("--en_cpu", nargs="*", default=["cfc"])
    p.add_argument("--hilos", type=int, default=4)
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--pesos", default=os.path.join("runs_fuego", "modelos_final"))
    p.add_argument("--salida", default=os.path.join("resultados", "modelos_final"))
    p.add_argument("--fase_e", default=os.path.join("resultados", "fase_e_final"))
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    a = p.parse_args()

    registros = os.path.join(a.pesos, "registros")
    os.makedirs(registros, exist_ok=True)
    os.makedirs(a.salida, exist_ok=True)
    inicio = time.time()
    if a.etapa == "tasas":
        etapa_tasas(a, registros)
    elif a.etapa == "riesgo":
        etapa_riesgo(a, registros)
    elif a.etapa == "final":
        etapa_final(a, registros)
    else:
        etapa_congelar(a)
    print(f"etapa {a.etapa} terminada en {(time.time() - inicio) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
