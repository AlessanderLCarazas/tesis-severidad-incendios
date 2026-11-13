import os
import csv
import glob
import json
import argparse
from collections import defaultdict
import numpy as np
import torch
from modelo_escena import ModeloEscena
from entrenar_temporal import verdad_escena
from entrenar_escena import cargar
from fase_e import inferir, nivel_crudo, histeresis, alerta, por_segundo, segmentos, media_desv, NIVELES, ORDEN
from fase_e_final import referencias, adelantos, resumen_adelanto, ETAPAS
from secuencias import leer_csv, numero


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eleccion", default=os.path.join("resultados", "modelos_final", "eleccion.json"))
    p.add_argument("--carpetas", nargs="*", default=[])
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--alfas", nargs="+", type=float, default=[0.10, 0.05])
    p.add_argument("--umbral_nivel", type=float, default=0.5)
    p.add_argument("--permanencia", type=float, default=2.0)
    p.add_argument("--factor_desactivar", type=float, default=0.8)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--sin_referencias", action="store_true")
    p.add_argument("--guardar_referencias", default="")
    p.add_argument("--dispositivo", default="cpu")
    p.add_argument("--salida", default=os.path.join("resultados", "fase_e_elegida"))
    a = p.parse_args()

    descripcion = {}
    if a.carpetas:
        carpetas = [c for patron in a.carpetas for c in sorted(glob.glob(patron))]
    else:
        with open(a.eleccion, encoding="utf-8") as f:
            eleccion = json.load(f)
        carpetas = []
        for m in ("cfc", "gru", "lstm"):
            carpetas += eleccion[m]["carpetas_pesos"]
            descripcion[m] = eleccion[m]["descripcion"]
    if not carpetas:
        raise SystemExit("no hay corridas")

    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    metas = defaultdict(dict)
    for m in leer_csv(a.metas):
        if m["fuente"] in ("ia", "dataset") and m["conjunto"] != "prueba":
            metas[m["conjunto"]][m["archivo"]] = m
    verdad = verdad_escena(metas["validacion"], sorted(metas["validacion"]))
    cal_umbral = sorted(v for v, r in particion.items() if r["conjunto"] == "calibracion" and r["calibra_umbral"] == "si")
    val = {v: r for v, r in particion.items() if r["conjunto"] == "validacion"}
    sin_fuego = sorted(v for v, r in val.items() if r["clase"] == "sin_fuego")
    controlados = sorted(v for v, r in val.items() if r["clase"] == "fuego" and r["nivel_max"] == "controlado")
    propagacion = sorted(v for v, r in val.items() if r["clase"] == "fuego" and r["nivel_max"] in ("incipiente", "incontrolado"))
    cambios = {c: {v: numero(metas[c][v]["t_incontrolado"]) for v, r in particion.items()
                   if r["conjunto"] == c and r["tipo"].endswith("_a_incontrolado")} for c in ("validacion", "calibracion")}
    n_cambios = len(cambios["validacion"]) + len(cambios["calibracion"])
    os.makedirs(os.path.join(a.salida, "segmentos"), exist_ok=True)
    corridas = defaultdict(list)

    def aplicar(nombre, tipo, pred, extra):
        pv = pred["validacion"]
        tiempos = {v: d["t"] for v, d in pv.items()}
        crudo = {v: nivel_crudo(d, a.umbral_nivel, a.umbral_nivel) for v, d in pv.items()}
        final = {v: histeresis(crudo[v], d["t"], a.permanencia) for v, d in pv.items()}
        res = dict(extra, corrida=nombre, modelo=tipo, antes=por_segundo(crudo, tiempos, verdad),
                   histeresis=por_segundo(final, tiempos, verdad), alerta={})
        with open(os.path.join(a.salida, "segmentos", f"{nombre.replace(' ', '_')}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["archivo", "t_inicio", "t_fin", "nivel"])
            for v in sorted(final):
                for t0, t1, n in segmentos(final[v], pv[v]["t"]):
                    w.writerow([v, round(t0, 2), round(t1, 2), NIVELES[n]])
        res["segmentos_por_video"] = float(np.mean([len(segmentos(final[v], pv[v]["t"])) for v in final]))
        puntajes = sorted(float(pred["calibracion"][v]["riesgo"].max()) if v in pred["calibracion"] else 0.0 for v in cal_umbral)
        n = len(puntajes)
        for alfa in a.alfas:
            k = int(np.ceil((n + 1) * (1.0 - alfa)))
            umbral = puntajes[k - 1] if k <= n else float("inf")
            activa = {v: alerta(d["riesgo"], d["t"], umbral, a.factor_desactivar, a.permanencia) for v, d in pv.items()}
            con = lambda vs: [bool(activa[v].any()) if v in activa else False for v in vs]
            dv = adelantos(pv, cambios["validacion"], umbral, a.factor_desactivar, a.permanencia, "validacion")
            dc = adelantos(pred["calibracion"], cambios["calibracion"], umbral, a.factor_desactivar, a.permanencia, "calibracion")
            res["alerta"][f"{alfa:g}"] = {
                "umbral": umbral, "videos_calibracion": n, "fpr_sin_fuego": float(np.mean(con(sin_fuego))),
                "n_sin_fuego": len(sin_fuego), "alarmas_sin_fuego": int(np.sum(con(sin_fuego))),
                "fpr_controlado": float(np.mean(con(controlados))), "n_controlado": len(controlados),
                "alarmas_controlado": int(np.sum(con(controlados))),
                "deteccion_propagacion": float(np.mean(con(propagacion))), "n_propagacion": len(propagacion),
                "validacion": resumen_adelanto(dv), "calibracion": resumen_adelanto(dc), "ambos": resumen_adelanto(dv + dc),
                "detalle": dv + dc}
        with open(os.path.join(a.salida, f"{nombre.replace(' ', '_')}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        corridas[tipo].append(res)
        al = res["alerta"][f"{a.alfas[0]:g}"]
        print(f"{nombre}: f1 {res['antes']['f1']:.4f} -> {res['histeresis']['f1']:.4f} | kappa {res['histeresis']['kappa']:.3f}"
              f" | fpr sin fuego {al['fpr_sin_fuego']:.3f} controlado {al['fpr_controlado']:.3f}"
              f" | antes del cambio {al['ambos']['con_alerta_antes']} de {n_cambios}", flush=True)

    datos = {c: cargar(os.path.join(a.datos, f"{c}.npz"), a.dispositivo) for c in ("calibracion", "validacion")}
    for carpeta in carpetas:
        ruta = os.path.join(carpeta, "mejor.pt")
        estado = torch.load(ruta, map_location=a.dispositivo, weights_only=False)
        unidad = float(estado.get("unidad_dt", 1.0))
        modelo = ModeloEscena(estado["tipo"], n_x=estado["n_x"], unidades=estado["unidades"], dropout=estado["dropout"]).to(a.dispositivo)
        modelo.load_state_dict(estado["modelo"])
        modelo.eval()
        pred = {c: inferir(modelo, dict(datos[c][0], ts=datos[c][0]["ts"] / unidad), datos[c][1]) for c in datos}
        aplicar(os.path.basename(os.path.normpath(carpeta)), estado["tipo"], pred,
                {"pesos": ruta.replace(os.sep, "/"), "unidad_dt": unidad})
    if not a.sin_referencias:
        for nombre, pred in referencias(a, particion, metas).items():
            aplicar(nombre, nombre, pred, {})

    presentes = [m for m in ORDEN if corridas[m]]
    titulo = lambda m: {"cfc": "CfC", "gru": "GRU", "lstm": "LSTM"}.get(m, m)
    lineas = ["# Fase E en validacion con la configuracion elegida por modelo", ""]
    lineas.append("Generado por `python fase_e_elegida.py`. La prueba no se usa. Media +- desviacion entre semillas.")
    lineas.append("")
    for m in presentes:
        if m in descripcion:
            lineas.append(f"- {titulo(m)}: {descripcion[m]}.")
    lineas.append(f"- Nivel: umbrales de CORAL en {a.umbral_nivel:g} e histeresis causal de {a.permanencia:g} s.")
    lineas.append(f"- Alerta conformal por video: umbral = cuantil conformal del riesgo maximo de los {len(cal_umbral)} videos de"
                  f" calibracion sin fuego o controlados; se apaga cuando el riesgo baja de {a.factor_desactivar:g} veces el umbral"
                  f" durante {a.permanencia:g} s.")
    lineas.append("- Referencias sin tiempo: una prediccion por segundo; riesgo = 0.5 P(incipiente) + P(incontrolado).")
    lineas.append(f"- Validacion: {len(verdad)} videos de fuego para el nivel, {len(sin_fuego)} sin fuego, {len(controlados)} controlados,"
                  f" {len(propagacion)} con propagacion. Videos con cambio a incontrolado: {len(cambios['validacion'])} en validacion"
                  f" y {len(cambios['calibracion'])} en calibracion (estos no entran en el umbral conformal).")
    lineas.append("- Adelanto = instante del cambio a incontrolado menos instante de la primera alerta; negativo = la alerta llega despues.")
    lineas.append("")
    lineas.append("| modelo | corridas | etapa | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal |"
                  " cambios de nivel por video |")
    lineas.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in presentes:
        for clave, nombre in ETAPAS:
            cs = corridas[m]
            lineas.append(f"| {titulo(m)} | {len(cs)} | {nombre} | {media_desv([c[clave]['f1'] for c in cs])} | "
                          + " | ".join(media_desv([c[clave]["f1_clase"][k] for c in cs]) for k in range(3))
                          + f" | {media_desv([c[clave]['exactitud'] for c in cs])} | {media_desv([c[clave]['kappa'] for c in cs])} |"
                          f" {media_desv([c[clave]['cambios'] for c in cs], '.2f')} |")
    lineas.append("")
    lineas.append("| modelo | alfa | umbral de riesgo | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta |"
                  " adelanto mediano en validacion (s) | antes del cambio en validacion | adelanto mediano en validacion + calibracion (s) |"
                  " antes del cambio en validacion + calibracion | sin alerta (validacion + calibracion) |")
    lineas.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, alfa in enumerate(a.alfas):
        for m in presentes:
            cs = [c["alerta"][f"{alfa:g}"] for c in corridas[m]]
            lineas.append(f"| {titulo(m)} | {alfa:g}{' (final)' if i == 0 else ''} | {media_desv([c['umbral'] for c in cs])} |"
                          f" {media_desv([c['fpr_sin_fuego'] for c in cs])} | {media_desv([c['fpr_controlado'] for c in cs])} |"
                          f" {media_desv([c['deteccion_propagacion'] for c in cs])} |"
                          f" {media_desv([c['validacion']['adelanto_mediana'] for c in cs], '.1f')} |"
                          f" {media_desv([c['validacion']['con_alerta_antes'] for c in cs], '.1f')} de {len(cambios['validacion'])} |"
                          f" {media_desv([c['ambos']['adelanto_mediana'] for c in cs], '.1f')} |"
                          f" {media_desv([c['ambos']['con_alerta_antes'] for c in cs], '.1f')} de {n_cambios} |"
                          f" {media_desv([c['ambos']['sin_alerta'] for c in cs], '.1f')} |")
    lineas.append("")
    with open(os.path.join(a.salida, "resumen.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
