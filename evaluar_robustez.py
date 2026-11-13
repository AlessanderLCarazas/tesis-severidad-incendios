import os
import json
import argparse
from collections import defaultdict
import numpy as np
import torch
from normalizacion import Normalizador
from escena_v2 import vectores_carpeta
from evaluar_prueba import por_segundo_video, md
from analisis_prueba import especificacion, predecir, referencias, nivel_de, alerta_de, titulo, ORDEN
from entrenar_escena import resumen_conf
from secuencias import leer_csv

FRACCIONES = [("completa", os.path.join("runs_fuego", "descriptores_v3")),
              ("quitar_25", os.path.join("runs_fuego", "robustez", "quitar_25")),
              ("quitar_50", os.path.join("runs_fuego", "robustez", "quitar_50")),
              ("quitar_75", os.path.join("runs_fuego", "robustez", "quitar_75"))]
NOMBRE_FRACCION = {"completa": "completa", "quitar_25": "25 % quitado", "quitar_50": "50 % quitado", "quitar_75": "75 % quitado"}


def armar(carpeta, videos, norm_x, norm_c, min_fotogramas=10, edad_min=1.0, min_larga=10, dispositivo="cpu"):
    vectores = vectores_carpeta(carpeta, set(videos), edad_min, min_larga, "_alta")
    faltan = sorted(v for v in videos if v not in vectores)
    if faltan:
        raise SystemExit(f"faltan {len(faltan)} videos en {carpeta}; termina la corrida de descriptores_v3.py")
    xs, cs, dts, ts, archivos, largos = [], [], [], [], [], []
    descartados, fotogramas, tramos = 0, 0, 0
    for v in sorted(videos):
        d = vectores[v]
        fotogramas += len(d["t"])
        for j in np.unique(d["tramo"]):
            tramos += 1
            sel = np.flatnonzero(d["tramo"] == j)
            if len(sel) < min_fotogramas:
                descartados += 1
                continue
            xs.append(d["x"][sel])
            cs.append(d["c"][sel])
            dts.append(d["dt"][sel].astype(np.float32))
            ts.append(d["t"][sel].astype(np.float32))
            archivos.append(v)
            largos.append(len(sel))
    x = norm_x.transformar(np.concatenate(xs)).astype(np.float32)
    c = norm_c.transformar(np.concatenate(cs)).astype(np.float32)
    datos = {"x": torch.from_numpy(x).to(dispositivo), "c": torch.from_numpy(c).to(dispositivo),
             "ts": torch.from_numpy(np.concatenate(dts)).to(dispositivo)}
    info = {"inicio": np.concatenate([[0], np.cumsum(largos)]).astype(np.int64), "archivo": np.array(archivos, dtype=str),
            "t": np.concatenate(ts)}
    resumen = {"fotogramas": int(fotogramas), "fotogramas_usados": int(sum(largos)), "tramos": int(tramos),
               "tramos_cortos": int(descartados), "videos_con_corte": int(sum(len(np.unique(vectores[v]["tramo"])) > 1 for v in videos)),
               "cortes": int(sum(len(np.unique(vectores[v]["tramo"])) - 1 for v in videos)),
               "dt_mediana": float(np.median(np.concatenate(dts))), "dt_p95": float(np.percentile(np.concatenate(dts), 95))}
    return datos, info, resumen


def mantener(d, final, seg):
    salida = {}
    t = np.asarray(d["t"])
    for s in sorted(seg):
        antes = np.flatnonzero(t < s + 1)
        salida[s] = int(final[antes[-1]]) if len(antes) else 0
    return salida


def nivel_mantenido(c, ref, videos):
    conf = np.zeros((3, 3), int)
    vacio = {"t": np.zeros(0), "p": np.zeros((0, 3))}
    for v in videos:
        d = c["pred"].get(v, vacio)
        f = c["final"].get(v, np.zeros(0, int))
        segundos, real, est, _ = por_segundo_video(d, f, ref["verdad"][v])
        votos = dict(zip(segundos, est))
        con_datos = set(np.floor(np.asarray(d["t"])).astype(int).tolist())
        rellenado = mantener(d, f, ref["verdad"][v])
        for s, r in zip(segundos, real):
            conf[r, votos[s] if s in con_datos else rellenado[s]] += 1
    return resumen_conf(conf)


def segundos_vacios(c, ref):
    total, vacios = 0, 0
    for v in ref["nivel"]:
        d = c["pred"].get(v)
        con = set() if d is None else set(np.floor(np.asarray(d["t"])).astype(int).tolist())
        total += len(ref["verdad"][v])
        vacios += sum(s not in con for s in ref["verdad"][v])
    return vacios, total


def metricas(corridas, refs, sin_fuego, alfas):
    salida = {}
    for c in corridas:
        r = {}
        for clave, ref in refs.items():
            pv = {}
            vacio = {"t": np.zeros(0), "p": np.zeros((0, 3))}
            for v in ref["nivel"]:
                segundos, real, est, _ = por_segundo_video(c["pred"].get(v, vacio), c["final"].get(v, np.zeros(0, int)), ref["verdad"][v])
                conf = np.zeros((3, 3), int)
                np.add.at(conf, (real, est), 1)
                pv[v] = {"conf": conf}
            n = nivel_de(pv, ref["nivel"])
            m = nivel_mantenido(c, ref, ref["nivel"])
            vacios, total = segundos_vacios(c, ref)
            r[clave] = {"f1": n["f1"], "f1_clase": n["f1_clase"], "kappa": n["kappa"], "exactitud": n["exactitud"],
                        "f1_mantenido": m["f1"], "kappa_mantenido": m["kappa"], "segundos_vacios": vacios, "segundos": total,
                        "alerta": {alfa: alerta_de(c, ref, sin_fuego, alfa) for alfa in alfas}}
        salida[c["corrida"]] = {"modelo": c["modelo"], "referencias": r}
    return salida


def control(res, ruta):
    with open(ruta, encoding="utf-8") as f:
        guardado = json.load(f)["corridas"]
    peor = 0.0
    for nombre, r in res.items():
        for clave, x in r["referencias"].items():
            g = guardado[nombre]["referencias"][clave]
            peor = max(peor, abs(x["f1"] - g["final"]["f1"]), abs(x["kappa"] - g["final"]["kappa"]))
            for alfa, al in x["alerta"].items():
                ga = g["alerta"][alfa]
                for k in ("fpr_sin_fuego", "fpr_controlado", "propagacion_con_alerta", "ewt_mediana"):
                    if (al[k] is None) != (ga[k] is None):
                        peor = max(peor, 1.0)
                    elif al[k] is not None:
                        peor = max(peor, abs(al[k] - ga[k]))
                peor = max(peor, abs(al["antes_del_cambio"] - ga["antes_del_cambio"]), abs(al["sin_alerta"] - ga["sin_alerta"]))
    return peor


def cargar_modelos(a, config):
    if a.modelos == "congelados":
        temporales, sin_tiempo = especificacion(config)
        return temporales, sin_tiempo, "Modelos congelados (`resultados/configuracion_final.json`)"
    with open(a.robustos, encoding="utf-8") as f:
        rob = json.load(f)
    temporales = [{"corrida": n, "modelo": c["modelo"], "pesos": c["archivo_pesos"], "sha256": c["sha256"], "unidad": 1.0,
                   "umbral": c["umbral_alerta"]} for n, c in rob["corridas"].items()]
    return temporales, [], f"Modelos reentrenados con fotogramas quitados (`{a.robustos.replace(os.sep, '/')}`)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelos", choices=["congelados", "robustos"], default="congelados")
    p.add_argument("--fracciones", nargs="+", default=[f for f, _ in FRACCIONES], choices=[f for f, _ in FRACCIONES])
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    p.add_argument("--robustos", default=os.path.join("resultados", "robustez", "modelos_robustos.json"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--guardado", default=os.path.join("resultados", "prueba", "metricas.json"))
    p.add_argument("--base", default=os.path.join("resultados", "robustez", "parte1.json"))
    p.add_argument("--solo_control", action="store_true")
    p.add_argument("--salida", default=os.path.join("resultados", "robustez"))
    a = p.parse_args()

    with open(a.config, encoding="utf-8") as f:
        config = json.load(f)
    dec = config["decision"]
    alfas = [f"{dec['alfa']:g}", f"{dec['alfa_secundario']:g}"]
    norm_x = Normalizador.cargar(config["datos"]["normalizacion"][0])
    norm_c = Normalizador.cargar(config["datos"]["normalizacion"][1])
    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    videos = sorted(v for v, r in particion.items() if r["conjunto"] == "prueba")
    refs, sin_fuego = referencias("prueba", particion, a.metas)
    temporales, sin_tiempo, descripcion = cargar_modelos(a, config)
    carpetas = dict(FRACCIONES)
    fracciones = ["completa"] if a.solo_control else a.fracciones
    res, datos_fraccion = {}, {}
    for fr in fracciones:
        print(f"armando el vector de escena: {fr}", flush=True)
        datos, info, resumen = armar(carpetas[fr], videos, norm_x, norm_c)
        corridas = predecir(datos, info, temporales, sin_tiempo, dec)
        res[fr] = metricas(corridas, refs, sin_fuego, alfas)
        datos_fraccion[fr] = resumen
        if fr == "completa" and a.modelos == "congelados":
            peor = control(res[fr], a.guardado)
            datos_fraccion[fr]["control"] = peor
            print(f"control con la fase f: diferencia maxima {peor:.2e}", flush=True)
            if peor > 1e-9:
                raise SystemExit("la corrida completa no reproduce la fase f; no se evaluan las fracciones")
        for nombre, r in res[fr].items():
            x = r["referencias"]["ia"]
            print(f"{fr} | {nombre}: f1 {x['f1']:.3f} | kappa {x['kappa']:.3f}", flush=True)
    if a.solo_control:
        return

    base = None
    if a.modelos == "robustos" and os.path.exists(a.base):
        with open(a.base, encoding="utf-8") as f:
            base = json.load(f)["resultados"]
    grupos = defaultdict(list)
    for nombre, r in res[fracciones[0]].items():
        grupos[r["modelo"]].append(nombre)
    presentes = [m for m in ORDEN if grupos[m]]
    etiqueta = "parte 2" if a.modelos == "robustos" else "parte 1"
    lineas = [f"# Robustez, {etiqueta}: fotogramas quitados en la prueba", ""]
    lineas.append(f"Generado por `python evaluar_robustez.py --modelos {a.modelos}`. {descripcion}. Sin reentrenar ni recalibrar en la evaluacion:"
                  " umbrales de CORAL en 0.5, histeresis de 2 s y umbral conformal de cada corrida. Vector de escena armado con"
                  " `escena_v2.vectores_carpeta` y la normalizacion congelada del entrenamiento. Modelos temporales: media +- desviacion de 3 semillas.")
    lineas.append("")
    lineas.append("- Caida = valor con fotogramas quitados menos valor en la prueba completa (con el mismo modelo).")
    lineas.append("- Nivel por segundo como en la fase F: un segundo sin fotogramas cuenta como controlado. Columna aparte: el segundo sin"
                  " fotogramas toma el ultimo nivel decidido (lo que mostraria el sistema en vivo).")
    if "completa" in datos_fraccion and "control" in datos_fraccion["completa"]:
        lineas.append(f"- Control: la corrida completa reproduce la fase F (diferencia maxima {datos_fraccion['completa']['control']:.1e}).")
    lineas.append("")
    lineas.append("| fraccion | fotogramas | usados en tramos de 10 o mas | cortes de escena | videos con corte | tramos cortos descartados | dt mediano (s) | dt percentil 95 (s) |")
    lineas.append("|---|---|---|---|---|---|---|---|")
    for fr in fracciones:
        d = datos_fraccion[fr]
        lineas.append(f"| {NOMBRE_FRACCION[fr]} | {d['fotogramas']} | {d['fotogramas_usados']} | {d['cortes']} | {d['videos_con_corte']} |"
                      f" {d['tramos_cortos']} | {d['dt_mediana']:.2f} | {d['dt_p95']:.2f} |")
    lineas.append("")

    def valor(fr, m, clave, f):
        return [f(res[fr][n]["referencias"][clave]) for n in grupos[m]]

    def caida(fr, m, clave, f):
        fuente = res
        if "completa" not in fuente:
            return "-"
        actual = [f(res[fr][n]["referencias"][clave]) for n in grupos[m]]
        inicial = [f(fuente["completa"][n]["referencias"][clave]) for n in fuente["completa"] if fuente["completa"][n]["modelo"] == m]
        actual = [x for x in actual if x is not None]
        inicial = [x for x in inicial if x is not None]
        if not actual or not inicial:
            return "-"
        return f"{np.mean(actual) - np.mean(inicial):+.3f}"

    for clave, ref in refs.items():
        lineas.append(f"## Nivel por segundo, {ref['titulo']} ({len(ref['nivel'])} videos)")
        lineas.append("")
        lineas.append("| modelo | fraccion | F1 macro | caida | F1 incipiente | kappa lineal | caida | segundos sin fotogramas |"
                      " F1 macro manteniendo el ultimo nivel | kappa manteniendo el ultimo nivel |")
        lineas.append("|---|---|---|---|---|---|---|---|---|---|")
        for m in presentes:
            for fr in fracciones:
                vac = valor(fr, m, clave, lambda x: x["segundos_vacios"])
                tot = valor(fr, m, clave, lambda x: x["segundos"])[0]
                lineas.append(f"| {titulo(m)} | {NOMBRE_FRACCION[fr]} | {md(valor(fr, m, clave, lambda x: x['f1']))} |"
                              f" {caida(fr, m, clave, lambda x: x['f1'])} | {md(valor(fr, m, clave, lambda x: x['f1_clase'][1]))} |"
                              f" {md(valor(fr, m, clave, lambda x: x['kappa']))} | {caida(fr, m, clave, lambda x: x['kappa'])} |"
                              f" {np.mean(vac):.0f} de {tot} | {md(valor(fr, m, clave, lambda x: x['f1_mantenido']))} |"
                              f" {md(valor(fr, m, clave, lambda x: x['kappa_mantenido']))} |")
        lineas.append("")
    for i, alfa in enumerate(alfas):
        lineas.append(f"## Alerta con alfa {alfa}{' (configuracion final)' if i == 0 else ' (aparte)'}")
        lineas.append("")
        for clave, ref in refs.items():
            lineas.append(f"{ref['titulo'][0].upper() + ref['titulo'][1:]}:")
            lineas.append("")
            lineas.append("| modelo | fraccion | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta |"
                          " EWT mediana (s) | caida | con alerta antes del cambio | sin alerta |")
            lineas.append("|---|---|---|---|---|---|---|---|---|")
            for m in presentes:
                for fr in fracciones:
                    al = lambda x: x["alerta"][alfa]
                    rs = valor(fr, m, clave, al)
                    lineas.append(f"| {titulo(m)} | {NOMBRE_FRACCION[fr]} | {md([r['fpr_sin_fuego'] for r in rs])} (de {rs[0]['n_sin_fuego']}) |"
                                  f" {md([r['fpr_controlado'] for r in rs])} (de {rs[0]['n_controlado']}) |"
                                  f" {md([r['propagacion_con_alerta'] for r in rs])} (de {rs[0]['n_propagacion']}) |"
                                  f" {md([r['ewt_mediana'] for r in rs], '.1f')} | {caida(fr, m, clave, lambda x: x['alerta'][alfa]['ewt_mediana'])} |"
                                  f" {md([r['antes_del_cambio'] for r in rs], '.1f')} de {rs[0]['n_cambio']} | {md([r['sin_alerta'] for r in rs], '.1f')} |")
            lineas.append("")
    if base is not None:
        lineas.append("## Comparacion con los modelos congelados (parte 1)")
        lineas.append("")
        lineas.append("| modelo | fraccion | referencia | F1 macro congelado | F1 macro reentrenado | diferencia | kappa congelado | kappa reentrenado |"
                      " diferencia | falsas alarmas en controlados (congelado / reentrenado) | EWT mediana (congelado / reentrenado) |")
        lineas.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for m in presentes:
            for fr in fracciones:
                if fr not in base:
                    continue
                viejos = [n for n in base[fr] if base[fr][n]["modelo"] == m]
                for clave in refs:
                    f_v = [base[fr][n]["referencias"][clave]["f1"] for n in viejos]
                    f_n = valor(fr, m, clave, lambda x: x["f1"])
                    k_v = [base[fr][n]["referencias"][clave]["kappa"] for n in viejos]
                    k_n = valor(fr, m, clave, lambda x: x["kappa"])
                    c_v = [base[fr][n]["referencias"][clave]["alerta"][alfas[0]]["fpr_controlado"] for n in viejos]
                    c_n = valor(fr, m, clave, lambda x: x["alerta"][alfas[0]]["fpr_controlado"])
                    e_v = [base[fr][n]["referencias"][clave]["alerta"][alfas[0]]["ewt_mediana"] for n in viejos]
                    e_n = valor(fr, m, clave, lambda x: x["alerta"][alfas[0]]["ewt_mediana"])
                    lineas.append(f"| {titulo(m)} | {NOMBRE_FRACCION[fr]} | {'IA' if clave == 'ia' else 'humana'} | {md(f_v)} | {md(f_n)} |"
                                  f" {np.mean(f_n) - np.mean(f_v):+.3f} | {md(k_v)} | {md(k_n)} | {np.mean(k_n) - np.mean(k_v):+.3f} |"
                                  f" {md(c_v)} / {md(c_n)} | {md(e_v, '.1f')} / {md(e_n, '.1f')} |")
        lineas.append("")
    os.makedirs(a.salida, exist_ok=True)
    nombre = "parte2" if a.modelos == "robustos" else "parte1"
    with open(os.path.join(a.salida, f"{nombre}.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    with open(os.path.join(a.salida, f"{nombre}.json"), "w", encoding="utf-8") as f:
        json.dump({"modelos": a.modelos, "fracciones": datos_fraccion, "resultados": res}, f, indent=1)
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
