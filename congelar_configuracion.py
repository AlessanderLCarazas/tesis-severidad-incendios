import os
import glob
import json
import time
import hashlib
import argparse
import numpy as np

MODELOS = ["cfc", "gru", "lstm"]
OPCIONES = {
    "A": {"descripcion": "configuracion A: peso simple del incipiente y tasa 1e-3",
          "resultados": os.path.join("resultados", "modelos_escena_v2", "final"),
          "pesos": os.path.join("runs_fuego", "modelos_escena_v2", "final"), "fase_e": os.path.join("resultados", "fase_e")},
    "B": {"descripcion": "configuracion B: doble peso del incipiente y tasa 3e-3",
          "resultados": os.path.join("resultados", "modelos_final", "final"),
          "pesos": os.path.join("runs_fuego", "modelos_final", "final"), "fase_e": os.path.join("resultados", "fase_e_final")}}


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


def f1_final(carpeta, modelo, semillas):
    valores = []
    for s in semillas:
        r = leer_json(os.path.join(carpeta, f"{modelo}_s{s}.json"))
        valores.append(None if r is None else r["histeresis"]["f1"])
    return valores


def elegir(a):
    eleccion = {}
    for m in MODELOS:
        puntajes = {}
        for clave, o in OPCIONES.items():
            v = f1_final(o["fase_e"], m, a.semillas)
            if any(x is None for x in v):
                raise SystemExit(f"faltan corridas de {m} en {o['fase_e']}")
            puntajes[clave] = {"f1_final": v, "media": float(np.mean(v)), "desviacion": float(np.std(v))}
        mejor = max(puntajes, key=lambda k: puntajes[k]["media"])
        o = OPCIONES[mejor]
        e = {"elegida": mejor, "descripcion": o["descripcion"], "criterio": "f1 macro final (umbral 0.5 + histeresis) en validacion, media de 3 semillas",
             "opciones": puntajes, "unidad_dt": None, "carpeta_resultados": o["resultados"].replace(os.sep, "/"),
             "carpetas_pesos": [os.path.join(o["pesos"], f"{m}_s{s}").replace(os.sep, "/") for s in a.semillas]}
        if m == "cfc":
            dt = f1_final(a.cfc_dt_fase, m, a.semillas)
            base = puntajes[mejor]["f1_final"]
            prueba = {"unidad_dt": a.unidad_dt, "f1_final_semilla_42": dt[0], "f1_final_base_semilla_42": base[0],
                      "f1_final": dt, "adoptado": False}
            if dt[0] is not None and dt[0] > base[0] and all(x is not None for x in dt):
                prueba.update(adoptado=True, media=float(np.mean(dt)), desviacion=float(np.std(dt)))
                e.update(unidad_dt=a.unidad_dt, descripcion=o["descripcion"] + f", intervalo del cfc en unidades de {a.unidad_dt:g} s",
                         carpeta_resultados=a.cfc_dt_resultados.replace(os.sep, "/"),
                         carpetas_pesos=[os.path.join(a.cfc_dt_pesos, f"{m}_s{s}").replace(os.sep, "/") for s in a.semillas])
            e["prueba_unidad_dt"] = prueba
        eleccion[m] = e
        print(f"{m}: " + " | ".join(f"{k} {v['media']:.4f} +- {v['desviacion']:.4f}" for k, v in puntajes.items())
              + f" | elegida {mejor}" + (f" | unidad dt: {e['prueba_unidad_dt']}" if m == "cfc" else ""), flush=True)
    os.makedirs(os.path.dirname(a.eleccion), exist_ok=True)
    with open(a.eleccion, "w", encoding="utf-8") as f:
        json.dump(eleccion, f, indent=1)
    print(f"eleccion en {a.eleccion}")


def congelar(a):
    eleccion = leer_json(a.eleccion)
    tasas_a = leer_json(os.path.join("resultados", "modelos_escena_v2", "tasas.json"))
    tasas_b = leer_json(os.path.join("resultados", "modelos_final", "tasas.json"))
    por_modelo, corridas = {}, {}
    for m in MODELOS:
        e = eleccion[m]
        primero = None
        for carpeta in e["carpetas_pesos"]:
            nombre = os.path.basename(carpeta)
            r = leer_json(os.path.join(e["carpeta_resultados"], nombre, "metricas.json"))
            fase = leer_json(os.path.join(a.fase_e, f"{nombre}.json"))
            pesos = os.path.join(carpeta, "mejor.pt")
            if fase is None or fase["pesos"] != pesos.replace(os.sep, "/"):
                raise SystemExit(f"la fase e de {nombre} no corresponde a {pesos}; corre fase_e_elegida.py")
            primero = primero or r
            corridas[nombre] = {"modelo": m, "semilla": r["semilla"], "mejor_epoca": r["mejor_epoca"],
                                "archivo_pesos": pesos.replace(os.sep, "/"), "sha256": sha256(pesos),
                                "f1_final_validacion": fase["histeresis"]["f1"],
                                "umbral_alerta": {k: v["umbral"] for k, v in fase["alerta"].items()}}
        h = primero["hiperparametros"]
        por_modelo[m] = {"configuracion": e["elegida"], "descripcion": e["descripcion"], "tasa": h["lr"],
                         "factor_incipiente": h.get("factor_incipiente", 1.0), "tope_peso": h["tope_peso"],
                         "pesos_nivel": primero["pesos_nivel"], "meta_riesgo": primero.get("riesgo", "escalon"),
                         "unidad_dt_s": e["unidad_dt"],
                         "intervalo": "dt en segundos" if e["unidad_dt"] is None else f"dt / {e['unidad_dt']:g}",
                         "epocas": h["epocas"], "lote": h["lote"], "decaimiento": h["decaimiento"], "recorte": h["recorte"],
                         "unidades": primero["unidades"], "dropout": primero["dropout"],
                         "dispositivo": "cpu" if m == "cfc" else "cuda", "f1_final_por_configuracion": e["opciones"]}
        if m == "cfc":
            por_modelo[m]["prueba_unidad_dt"] = e["prueba_unidad_dt"]
    referencias = {}
    for ruta in sorted(glob.glob(os.path.join(a.referencias, "*.joblib"))):
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        fase = leer_json(os.path.join(a.fase_e, f"{nombre}.json"))
        referencias[nombre] = {"archivo": ruta.replace(os.sep, "/"), "sha256": sha256(ruta),
                               "f1_final_validacion": fase["histeresis"]["f1"],
                               "umbral_alerta": {k: v["umbral"] for k, v in fase["alerta"].items()}}
    config = {
        "fecha": time.strftime("%Y-%m-%d"),
        "nota": "configuracion congelada antes de la prueba; no se cambia nada despues",
        "eleccion": "cada modelo elige entre A (peso simple, 1e-3) y B (doble peso, 3e-3) por el f1 macro final en validacion,"
                    " media de 3 semillas; la busqueda de tasa queda cerrada",
        "datos": {"carpeta": "runs_fuego/escena_v2", "vector_escena": "vector_escena.VectorEscena (34 valores, detecciones a confianza 0.25)",
                  "normalizacion": ["resultados/escena_v2/normalizacion.json", "resultados/escena_v2/normalizacion_contexto.json"],
                  "ventana": 300, "paso_ventana": 150, "particion": "resultados/particion/particion.csv",
                  "metas": "resultados/particion/metas.csv", "fuente_etiquetas": "ia"},
        "modelo": {"salidas": "coral k=3, riesgo, log(1 + tau)", "ponderacion": "kendall",
                   "seleccion_epoca": "f1 macro de escena por segundo en validacion, suavizado con 3 epocas", "plan_tasa": "coseno"},
        "pesos_nivel": {"formula": "min(1 / (3 f_k), tope) y luego factor para el incipiente"},
        "meta_riesgo": {"tipo": "escalon", "valores": [0.0, 0.5, 1.0],
                        "descartado": "riesgo anticipado max(escalon, 1 - tau / 60): no adelanta la alerta"},
        "tau": {"meta": "log(1 + tau)", "perdida": "huber delta 1", "mascara": "solo videos con cambio a incontrolado (tau_mascara = 1)"},
        "semillas": a.semillas,
        "busqueda_tasa": {"configuracion_a": tasas_a, "configuracion_b": tasas_b},
        "decision": {"umbral_coral": [0.5, 0.5], "histeresis_s": 2.0, "alfa": 0.10, "alfa_secundario": 0.05,
                     "puntuacion_conformal": "riesgo maximo por video en los videos de calibracion sin fuego o controlados",
                     "factor_desactivacion": 0.8, "permanencia_alerta_s": 2.0,
                     "nivel_por_segundo": "mayoria de los fotogramas del segundo"},
        "por_modelo": por_modelo,
        "corridas": corridas,
        "referencias_sin_tiempo": {"entrada": "media por segundo del vector de escena", "semilla": 42, "class_weight": "balanced",
                                   "riesgo": "0.5 P(incipiente) + P(incontrolado)", "modelos": referencias},
        "resultados_validacion": a.fase_e.replace(os.sep, "/") + "/resumen.md"}
    with open(a.config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=1)
    print(f"configuracion congelada en {a.config} | corridas: {len(corridas)} | referencias: {len(referencias)}")
    for m in MODELOS:
        print(f"{m}: {por_modelo[m]['descripcion']} | tasa {por_modelo[m]['tasa']:g} | intervalo {por_modelo[m]['intervalo']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--etapa", choices=["elegir", "congelar"], required=True)
    p.add_argument("--semillas", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--unidad_dt", type=float, default=0.1)
    p.add_argument("--cfc_dt_resultados", default=os.path.join("resultados", "modelos_final", "cfc_dt"))
    p.add_argument("--cfc_dt_pesos", default=os.path.join("runs_fuego", "modelos_final", "cfc_dt"))
    p.add_argument("--cfc_dt_fase", default=os.path.join("resultados", "fase_e_elegida", "cfc_dt"))
    p.add_argument("--eleccion", default=os.path.join("resultados", "modelos_final", "eleccion.json"))
    p.add_argument("--fase_e", default=os.path.join("resultados", "fase_e_elegida"))
    p.add_argument("--referencias", default=os.path.join("runs_fuego", "modelos_final", "referencias"))
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    a = p.parse_args()
    if a.etapa == "elegir":
        elegir(a)
    else:
        congelar(a)


if __name__ == "__main__":
    main()
