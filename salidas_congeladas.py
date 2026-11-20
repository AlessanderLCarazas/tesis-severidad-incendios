import os
import json
import argparse
import numpy as np
import torch
from modelo_escena import ModeloEscena
from entrenar_escena import cargar
from evaluar_prueba import inferir, sha256
from fase_e import histeresis, alerta

CONJUNTOS = ["calibracion", "validacion", "prueba"]


def leer_config(ruta=os.path.join("resultados", "configuracion_final.json")):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def calcular(conjunto, config, datos_dir, salida, dispositivo="cpu", forzar=False):
    ruta = os.path.join(salida, f"{conjunto}.npz")
    if os.path.exists(ruta) and not forzar:
        return ruta
    datos, info = cargar(os.path.join(datos_dir, f"{conjunto}.npz"), dispositivo)
    largos = info["inicio"][1:] - info["inicio"][:-1]
    out = {"t": info["t"], "archivo": np.repeat(info["archivo"], largos)}
    for nombre, c in config["corridas"].items():
        if sha256(c["archivo_pesos"]) != c["sha256"]:
            raise SystemExit(f"los pesos de {nombre} no coinciden con la configuracion congelada")
        estado = torch.load(c["archivo_pesos"], map_location=dispositivo, weights_only=False)
        modelo = ModeloEscena(estado["tipo"], n_x=estado["n_x"], unidades=estado["unidades"], dropout=estado["dropout"]).to(dispositivo)
        modelo.load_state_dict(estado["modelo"])
        modelo.eval()
        unidad = config["por_modelo"][c["modelo"]]["unidad_dt_s"] or 1.0
        p1, p2, riesgo, tau = inferir(modelo, datos, info, unidad)
        for k, v in (("p1", p1), ("p2", p2), ("riesgo", riesgo), ("tau", tau)):
            out[f"{nombre}__{k}"] = v
        print(f"{conjunto}: {nombre} listo", flush=True)
    os.makedirs(salida, exist_ok=True)
    np.savez_compressed(ruta, **out)
    return ruta


def cargar_salidas(conjunto, config, salida=os.path.join("runs_fuego", "salidas_congeladas")):
    d = np.load(os.path.join(salida, f"{conjunto}.npz"))
    archivo, t = d["archivo"], d["t"]
    indices = {}
    for v in np.unique(archivo):
        sel = np.flatnonzero(archivo == v)
        indices[str(v)] = sel[np.argsort(t[sel], kind="stable")]
    u1, u2 = config["decision"]["umbral_coral"]
    corridas = {}
    for nombre, c in config["corridas"].items():
        p1, p2, riesgo, tau = (d[f"{nombre}__{k}"] for k in ("p1", "p2", "riesgo", "tau"))
        pred = {}
        for v, i in indices.items():
            alto = np.minimum(p1[i], p2[i])
            arriba = p1[i] > u1
            pred[v] = {"t": t[i].astype(np.float64), "p": np.column_stack([1.0 - p1[i], p1[i] - alto, alto]),
                       "nivel": arriba.astype(np.int64) + (arriba & (p2[i] > u2)).astype(np.int64),
                       "riesgo": riesgo[i], "tau": tau[i], "idx": i}
        corridas[nombre] = {"corrida": nombre, "modelo": c["modelo"], "pred": pred, "umbral": c["umbral_alerta"]}
    return corridas


def decidir(corrida, decision, permanencia=None, umbral=None):
    perm = decision["histeresis_s"] if permanencia is None else permanencia
    final = {v: histeresis(d["nivel"], d["t"], perm) for v, d in corrida["pred"].items()}
    u = corrida["umbral"][f"{decision['alfa']:g}"] if umbral is None else umbral
    t_alerta = {}
    for v, d in corrida["pred"].items():
        activa = alerta(d["riesgo"], d["t"], u, decision["factor_desactivacion"], decision["permanencia_alerta_s"])
        t_alerta[v] = float(d["t"][np.argmax(activa)]) if activa.any() else None
    return final, t_alerta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conjuntos", nargs="+", default=CONJUNTOS, choices=CONJUNTOS)
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--salida", default=os.path.join("runs_fuego", "salidas_congeladas"))
    p.add_argument("--forzar", action="store_true")
    a = p.parse_args()
    config = leer_config(a.config)
    for c in a.conjuntos:
        print(f"salidas en {calcular(c, config, a.datos, a.salida, forzar=a.forzar)}")


if __name__ == "__main__":
    main()
