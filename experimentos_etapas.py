import os
import csv
import json
import argparse
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from cortes import evaluar as evaluar_cortes
from entrenar_temporal import verdad_escena
from entrenar_escena import resumen_conf
from evaluar_prueba import por_segundo_video, md
from escena_v2 import vectores_carpeta
from vector_escena import COLUMNAS
from secuencias import leer_csv, numero
from salidas_congeladas import leer_config, cargar_salidas, decidir
from tablas_latex import tabular, num, pm, pct, escapar

DESCRIPTORES = ["confianza", "área", "tasa de crecimiento", "crecimiento relativo", "centro x", "centro y", "velocidad del centro",
                "persistencia", "variación de forma", "confianza de humo", "área de humo", "tasa del humo", "regiones activas"]
EXTRA = ["trayectorias activas", "trayectorias de al menos 1 s", "edad máxima de trayectoria", "detecciones de fuego",
         "detecciones de humo", "área detectada de fuego", "área detectada de humo", "sin trayectoria"]
NOMBRE_COLUMNA = ([f"d{i + 1} {n} (máx.)" for i, n in enumerate(DESCRIPTORES)] + [f"d{i + 1} {n} (media)" for i, n in enumerate(DESCRIPTORES)]
                  + EXTRA)
TITULO = {"cfc": "CfC", "gru": "GRU", "lstm": "LSTM"}


def e6_cortes(a):
    grupos = {r["archivo"]: r["grupo"] for r in leer_csv(a.lista_cortes)}
    por_video = defaultdict(list)
    for r in leer_csv(a.indicadores):
        por_video[r["archivo"]].append({"k": int(r["k"]), "t": float(r["t"]), "h": float(r["h"]), "valido": int(r["valido"]),
                                        "inliers": float(r["inliers"])})
    revisados = defaultdict(list)
    for r in leer_csv(a.confirmados):
        revisados[r["archivo"]].append((float(r["t_corte"]), r["tipo"]))
    sub = {c: {v: f for v, f in por_video.items() if grupos.get(v) in gs} for c, gs in (("ajuste", ("ajuste", "prueba")), ("validacion", ("validacion",)))}
    cortes_reales = {c: sum(1 for v in pv for _, tipo in revisados.get(v, []) if tipo == "corte") for c, pv in sub.items()}

    def contar(c, t_alto, t_bajo):
        e = evaluar_cortes(sub[c], revisados, t_alto, t_bajo, a.separacion, a.ventana)
        return {k: len(e[k]) for k in ("aciertos", "perdidos", "falsos", "sin_revisar", "duplicados")}

    pasos = np.round(np.arange(0.25, 1.0001, 0.025), 3)
    rejilla = [(float(u), contar("ajuste", u, u)) for u in pasos]
    limpias = [(u, r) for u, r in rejilla if r["falsos"] == 0 and r["sin_revisar"] == 0 and r["duplicados"] == 0]
    mejor = max(r["aciertos"] for _, r in limpias)
    empatados = [u for u, r in limpias if r["aciertos"] == mejor]
    elegido = empatados[(len(empatados) - 1) // 2]
    reglas = {"doble": {"nombre": "doble umbral (elegida)", "t_alto": 0.85, "t_bajo": 0.65},
              "unico": {"nombre": "un solo umbral", "t_alto": elegido, "t_bajo": elegido}}
    for r in reglas.values():
        r["ajuste"] = contar("ajuste", r["t_alto"], r["t_bajo"])
        r["validacion"] = contar("validacion", r["t_alto"], r["t_bajo"])
    rango = {u: contar("validacion", u, u) for u in empatados}
    res = {"cortes_reales": cortes_reales, "videos": {c: len(pv) for c, pv in sub.items()}, "umbral_unico": elegido,
           "empatados_en_ajuste": empatados, "aciertos_ajuste_sin_falsos": mejor, "reglas": reglas, "validacion_empatados": rango,
           "rejilla_unico": [{"umbral": u, **r} for u, r in rejilla]}
    lineas = ["## E6. Cortes de escena: doble umbral contra un solo umbral", ""]
    lineas.append(f"Distancias h guardadas en `resultados/cortes/indicadores/indicadores.csv` ({res['videos']['ajuste']} videos de ajuste y"
                  f" {res['videos']['validacion']} de validacion; {cortes_reales['ajuste']} y {cortes_reales['validacion']} cortes revisados)."
                  " Regla de un solo umbral: corte si h >= T, sin usar la camara. Misma rejilla (0.25 a 1.0, paso 0.025), misma separacion"
                  " minima de 1 s, mismo emparejamiento (0.5 s) y mismo criterio: mas cortes detectados en ajuste sin falsos, sin detecciones"
                  f" sin revisar y sin duplicados. En ajuste empatan T = {', '.join(f'{u:.3f}' for u in empatados)} ({mejor} cortes);"
                  f" se toma el centro, T = {elegido:.3f}, como se hizo con la regla doble.")
    lineas.append("")
    lineas.append("| regla | T_alto | T_bajo | ajuste: detectados | ajuste: falsos | validacion: detectados | validacion: falsos | validacion: sin revisar |")
    lineas.append("|---|---|---|---|---|---|---|---|")
    filas_tex = []
    for r in reglas.values():
        lineas.append(f"| {r['nombre']} | {r['t_alto']:.3f} | {r['t_bajo']:.3f} | {r['ajuste']['aciertos']} de {cortes_reales['ajuste']} |"
                      f" {r['ajuste']['falsos']} | {r['validacion']['aciertos']} de {cortes_reales['validacion']} | {r['validacion']['falsos']} |"
                      f" {r['validacion']['sin_revisar']} |")
        filas_tex.append([escapar(r["nombre"][0].upper() + r["nombre"][1:]), num(r["t_alto"], 3), num(r["t_bajo"], 3),
                          f"{r['ajuste']['aciertos']} de {cortes_reales['ajuste']}", str(r["ajuste"]["falsos"]),
                          f"{r['validacion']['aciertos']} de {cortes_reales['validacion']}", str(r["validacion"]["falsos"]),
                          str(r["validacion"]["sin_revisar"])])
    lineas.append("")
    if len(empatados) > 1:
        lineas.append("En validacion, a lo largo de los umbrales empatados en ajuste: "
                      + "; ".join(f"T {u:.3f}: {r['aciertos']} detectados, {r['falsos']} falsos, {r['sin_revisar']} sin revisar" for u, r in rango.items()) + ".")
        lineas.append("")
    tabular(os.path.join(a.tablas, "e6_cortes.tex"), ["Regla", "$T_{alto}$", "$T_{bajo}$", "Ajuste: detectados", "Ajuste: falsos",
                                                       "Validación: detectados", "Validación: falsos", "Validación: sin revisar"], filas_tex)
    return res, lineas


def conjuntos_validacion(a):
    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    metas = {m["archivo"]: m for m in leer_csv(a.metas) if m["conjunto"] == "validacion" and m["fuente"] in ("ia", "dataset")}
    val = {v: r for v, r in particion.items() if r["conjunto"] == "validacion"}
    fuego = sorted(v for v, m in metas.items() if m["fuente"] == "ia")
    verdad = verdad_escena(metas, fuego)
    return {"verdad": verdad, "sin_fuego": sorted(v for v, r in val.items() if r["clase"] == "sin_fuego"),
            "controlados": sorted(v for v, r in val.items() if r["clase"] == "fuego" and r["nivel_max"] == "controlado"),
            "cambios": {v: numero(metas[v]["t_incontrolado"]) for v, r in val.items() if r["tipo"].endswith("_a_incontrolado")}}


def medir_variante(corrida, final, t_alerta, conj):
    vacio = {"t": np.zeros(0), "p": np.zeros((0, 3))}
    conf = np.zeros((3, 3), int)
    cambios, segundos, cambios_ref = 0, 0, 0
    for v, seg in conj["verdad"].items():
        _, real, est, _ = por_segundo_video(corrida["pred"].get(v, vacio), final.get(v, np.zeros(0, int)), seg)
        np.add.at(conf, (real, est), 1)
        cambios += int((np.diff(est) != 0).sum())
        cambios_ref += int((np.diff(real) != 0).sum())
        segundos += len(real)
    r = resumen_conf(conf)
    ewt = [conj["cambios"][v] - t_alerta[v] for v in sorted(conj["cambios"]) if t_alerta.get(v) is not None]
    return {"f1": r["f1"], "f1_clase": r["f1_clase"], "kappa": r["kappa"], "cambios_min": 60.0 * cambios / segundos,
            "cambios_min_referencia": 60.0 * cambios_ref / segundos, "segundos": segundos,
            "fa_controlado": float(np.mean([t_alerta.get(v) is not None for v in conj["controlados"]])),
            "fa_sin_fuego": float(np.mean([t_alerta.get(v) is not None for v in conj["sin_fuego"]])),
            "ewt_mediana": float(np.median(ewt)) if ewt else None, "antes_del_cambio": int(sum(x > 0 for x in ewt)),
            "sin_alerta": len(conj["cambios"]) - len(ewt), "n_cambios": len(conj["cambios"])}


def e8_decision(a, config):
    dec = config["decision"]
    conj = conjuntos_validacion(a)
    corridas = cargar_salidas("validacion", config, a.salidas)
    variantes = [("elegida", "conformal (alfa 0.10), histéresis 2 s (elegida)", None, 2.0), ("fijo", "umbral fijo 0.5, histéresis 2 s", 0.5, 2.0),
                 ("h0", "conformal, histéresis 0 s", None, 0.0), ("h1", "conformal, histéresis 1 s", None, 1.0),
                 ("h3", "conformal, histéresis 3 s", None, 3.0)]
    res, peor = {}, 0.0
    for nombre, c in corridas.items():
        if c["modelo"] not in a.modelos_e8:
            continue
        res[nombre] = {}
        for clave, _, umbral, perm in variantes:
            final, t_alerta = decidir(c, dec, perm, umbral)
            res[nombre][clave] = medir_variante(c, final, t_alerta, conj)
        with open(os.path.join(a.fase_e, f"{nombre}.json"), encoding="utf-8") as f:
            g = json.load(f)
        e = res[nombre]["elegida"]
        peor = max(peor, abs(e["f1"] - g["histeresis"]["f1"]), abs(e["fa_controlado"] - g["alerta"]["0.1"]["fpr_controlado"]),
                   abs(e["fa_sin_fuego"] - g["alerta"]["0.1"]["fpr_sin_fuego"]),
                   abs((e["ewt_mediana"] or 0) - (g["alerta"]["0.1"]["validacion"]["adelanto_mediana"] or 0)))
    print(f"control e8 contra la fase e guardada: diferencia maxima {peor:.2e}", flush=True)
    if peor > 1e-9:
        raise SystemExit("las salidas no reproducen la fase e guardada")
    lineas = ["## E8. Decision en validacion: alerta y histeresis", ""]
    lineas.append(f"Salidas de las semillas 42, 43 y 44 en validacion (control: la configuracion elegida reproduce `resultados/fase_e_elegida/`"
                  f" con diferencia {peor:.1e}). Referencia de la IA. {len(conj['verdad'])} videos de fuego para el nivel,"
                  f" {len(conj['controlados'])} controlados, {len(conj['sin_fuego'])} sin fuego y {len(conj['cambios'])} con cambio a incontrolado."
                  " Cambios por minuto: cambios del nivel por segundo dentro del tramo util, sobre el total de minutos. Umbral de alerta fijo:"
                  " misma regla de activacion y desactivacion (0.8 veces el umbral durante 2 s). La histeresis solo cambia el nivel, no la alerta.")
    lineas.append("")
    ref = next(iter(next(iter(res.values())).values()))["cambios_min_referencia"]
    lineas.append(f"Cambios por minuto de la referencia de la IA en estos videos: {ref:.2f}.")
    lineas.append("")
    for m in a.modelos_e8:
        nombres = [n for n in res if corridas[n]["modelo"] == m]
        lineas.append(f"### {TITULO[m]}")
        lineas.append("")
        lineas.append("| variante | F1 macro | F1 incipiente | falsas alarmas en controlados | falsas alarmas sin fuego | cambios por minuto |"
                      " EWT mediana (s) | antes del cambio | sin alerta |")
        lineas.append("|---|---|---|---|---|---|---|---|---|")
        filas_tex = []
        for clave, titulo, _, _ in variantes:
            rs = [res[n][clave] for n in nombres]
            ms = lambda k, d=3: md([r[k] for r in rs], f".{d}f")
            lineas.append(f"| {titulo} | {ms('f1')} | {md([r['f1_clase'][1] for r in rs])} | {ms('fa_controlado')} (de {len(conj['controlados'])})"
                          f" | {ms('fa_sin_fuego')} (de {len(conj['sin_fuego'])}) | {ms('cambios_min', 2)} | {ms('ewt_mediana', 1)} | {ms('antes_del_cambio', 1)} de {rs[0]['n_cambios']} |"
                          f" {ms('sin_alerta', 1)} |")
            media = lambda k: float(np.mean([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else None
            desv = lambda k: float(np.std([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else None
            titulo_tex = {"elegida": "Conformal ($\\alpha = 0{,}10$), histéresis 2 s (elegida)", "fijo": "Umbral fijo 0{,}5, histéresis 2 s",
                          "h0": "Conformal, histéresis 0 s", "h1": "Conformal, histéresis 1 s", "h3": "Conformal, histéresis 3 s"}[clave]
            filas_tex.append([titulo_tex, pm(media("f1"), desv("f1")), pct(media("fa_controlado")),
                              pct(media("fa_sin_fuego")), num(media("cambios_min"), 2), pm(media("ewt_mediana"), desv("ewt_mediana"), 1, True),
                              f"{num(media('antes_del_cambio'), 1)} de {rs[0]['n_cambios']}"])
        lineas.append("")
        tabular(os.path.join(a.tablas, f"e8_decision_{m}.tex"), ["Variante", "F1 macro", "Falsas alarmas en controlados", "Falsas alarmas sin fuego",
                                                                  "Cambios por minuto", "EWT mediana (s)", "Alerta antes del cambio"],
                filas_tex, "lcccccc")
    return {"control": peor, "cambios_min_referencia": ref, "corridas": res}, lineas


def e5_descriptores(a):
    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    metas = {m["archivo"]: m for m in leer_csv(a.metas) if m["conjunto"] == "validacion" and m["fuente"] == "ia"}
    verdad = verdad_escena(metas, sorted(metas))
    videos = sorted(verdad)
    vectores = vectores_carpeta(a.descriptores_v3, set(videos), 1.0, 10, "_alta")
    x, y = [], []
    for v in videos:
        d = vectores[v]
        sel = np.zeros(len(d["t"]), bool)
        for j in np.unique(d["tramo"]):
            idx = np.flatnonzero(d["tramo"] == j)
            if len(idx) >= a.min_fotogramas:
                sel[idx] = True
        segundos = np.floor(d["t"]).astype(int)
        for s, nivel in verdad[v].items():
            m = sel & (segundos == s)
            if m.any():
                x.append(d["x"][m].mean(axis=0))
                y.append(int(nivel > 0))
    x, y = np.array(x), np.array(y)
    filas = []
    for j, c in enumerate(COLUMNAS):
        auc = float(roc_auc_score(y, x[:, j])) if np.unique(x[:, j]).size > 1 else 0.5
        filas.append({"columna": c, "nombre": NOMBRE_COLUMNA[j], "auc": auc, "auc_orientada": max(auc, 1.0 - auc),
                      "sentido": "mayor en propagacion" if auc >= 0.5 else "menor en propagacion"})
    orden = sorted(filas, key=lambda f: -f["auc_orientada"])
    res = {"segundos": int(len(y)), "segundos_propagacion": int(y.sum()), "videos": len(videos), "columnas": orden}
    lineas = ["## E5. Poder separador de cada columna del vector de escena", ""]
    lineas.append(f"Validacion, referencia de la IA, {len(videos)} videos de fuego y {len(y)} segundos ({int(y.sum())} en incipiente o incontrolado)."
                  " Valor por segundo: media de los fotogramas del segundo, con el vector sin normalizar (la normalizacion es monotona salvo el"
                  " recorte en p1 y p99) y solo en los tramos de 10 o mas fotogramas, como en el entrenamiento. AUC de Controlado contra"
                  " Incipiente o Incontrolado; AUC orientada = max(AUC, 1 - AUC), para ordenar sin importar el sentido.")
    lineas.append("")
    lineas.append("| posicion | columna | AUC | AUC orientada | sentido |")
    lineas.append("|---|---|---|---|---|")
    for n, f in enumerate(orden, 1):
        lineas.append(f"| {n} | {f['nombre']} (`{f['columna']}`) | {f['auc']:.3f} | {f['auc_orientada']:.3f} | {f['sentido']} |")
    lineas.append("")
    filas_tex = []
    for grupo, sel in (("5 mejores", orden[:5]), ("5 peores", orden[-5:])):
        for f in sel:
            filas_tex.append([grupo, escapar(f["nombre"]), num(f["auc"]), num(f["auc_orientada"]),
                              "mayor en propagación" if f["auc"] >= 0.5 else "menor en propagación"])
    tabular(os.path.join(a.tablas, "e5_auc_columnas.tex"), ["Grupo", "Columna", "AUC", "AUC orientada", "Sentido"], filas_tex, "llccl")
    return res, lineas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partes", nargs="+", default=["e6", "e8", "e5"], choices=["e6", "e8", "e5"])
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    p.add_argument("--indicadores", default=os.path.join("resultados", "cortes", "indicadores", "indicadores.csv"))
    p.add_argument("--confirmados", default=os.path.join("resultados", "cortes", "confirmados.csv"))
    p.add_argument("--lista_cortes", default=os.path.join("resultados", "cortes", "lista_indicadores.csv"))
    p.add_argument("--separacion", type=float, default=1.0)
    p.add_argument("--ventana", type=float, default=0.5)
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--salidas", default=os.path.join("runs_fuego", "salidas_congeladas"))
    p.add_argument("--fase_e", default=os.path.join("resultados", "fase_e_elegida"))
    p.add_argument("--modelos_e8", nargs="+", default=["gru", "cfc"])
    p.add_argument("--descriptores_v3", default=os.path.join("runs_fuego", "descriptores_v3"))
    p.add_argument("--min_fotogramas", type=int, default=10)
    p.add_argument("--tablas", default=os.path.join("resultados", "tablas_tesis"))
    p.add_argument("--salida", default=os.path.join("resultados", "cap5"))
    a = p.parse_args()

    config = leer_config(a.config)
    os.makedirs(a.salida, exist_ok=True)
    resultados, lineas = {}, ["# Experimentos por etapa (bloque del asesor, despues de la fase F)", "",
                              "Generado por `python experimentos_etapas.py`. Solo ajuste y validacion; nada de esto se uso para elegir el sistema"
                              " y la configuracion congelada no cambia.", ""]
    for parte, fn in (("e6", lambda: e6_cortes(a)), ("e8", lambda: e8_decision(a, config)), ("e5", lambda: e5_descriptores(a))):
        if parte in a.partes:
            r, l = fn()
            resultados[parte] = r
            lineas += l
            print(f"{parte} listo", flush=True)
    with open(os.path.join(a.salida, "etapas.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas) + "\n")
    with open(os.path.join(a.salida, "etapas.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=1)
    print(f"resultados en {a.salida} y tablas en {a.tablas}")


if __name__ == "__main__":
    main()
