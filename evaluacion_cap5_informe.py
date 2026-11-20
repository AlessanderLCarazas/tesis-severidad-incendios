import os
import csv
import json
from collections import defaultdict
import numpy as np
from evaluacion_cap5 import (ORDEN, NOMBRE, TEMPORALES, REF, TIOU, Fuentes, corridas_disponibles, agregar_sin_tiempo, mayoritaria, decidir,
                             medir, pesos_bootstrap, muestras_modelo, ic, p_normal, holm, robustez_pruebas, costo, figuras, leer)
from analisis_prueba import referencias
from metricas_sistema import grupos_alerta, clopper
from experimentos_etapas import conjuntos_validacion, medir_variante
from salidas_congeladas import leer_config
from variantes_cap5 import VARIANTES, NOMBRE_VARIANTE, ABLACIONES_EXTRA
from secuencias import leer_csv
from tablas_latex import tabular, num, pm, pct, miles


def media(xs):
    v = [x for x in xs if x is not None]
    return (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else None) if v else (None, None)


def ic_tex(v):
    return f"[{num(v[0])}; {num(v[1])}]"


def ref_tex(clave):
    return "IA" if clave == "ia" else "Humana"


def informe(a):
    config = leer_config(a.config)
    dec = config["decision"]
    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    refs, sin_fuego = referencias("prueba", particion, a.metas)
    lista, faltan = corridas_disponibles(a, config)
    if faltan:
        print(f"aviso: faltan {len(faltan)} corridas: {', '.join(faltan[:8])}{' ...' if len(faltan) > 8 else ''}", flush=True)
    agregar_sin_tiempo(a, config, lista)
    fuentes = Fuentes(a, config)
    d = np.load(os.path.join(a.datos, "prueba.npz"))
    archivo_fr = np.repeat(d["seq_archivo"], d["seq_inicio"][1:] - d["seq_inicio"][:-1])
    tau_ref = {"ia": (d["tau"], d["tau_mascara"], archivo_fr), "humano": (d["tau_humano"], d["tau_mascara_humano"], archivo_fr)}
    base = fuentes.pred(lista[0], "prueba")
    lista.append({"nombre": "clase mayoritaria", "modelo": "clase mayoritaria", "grupo": "base", "semilla": None, "fuente": "memoria",
                  "pred": {"prueba": mayoritaria(a, refs, {v: x["t"] for v, x in base.items()})}, "umbral": None})
    res = {}
    for c in lista:
        pred = fuentes.pred(c, "prueba")
        final, t_alerta = decidir(pred, c["umbral"], dec)
        res[c["nombre"]] = medir(pred, final, t_alerta, refs, sin_fuego, tau_ref if c["modelo"] in TEMPORALES else None,
                                 c["modelo"] not in ("regla", "clase mayoritaria"))
    guardado = leer(a.metricas_prueba)["corridas"]
    peor = 0.0
    for n, g in guardado.items():
        if n in res:
            for clave in refs:
                peor = max(peor, abs(res[n][clave]["f1"] - g["referencias"][clave]["final"]["f1"]),
                           abs(res[n][clave]["map"]["0.5"]["promedio"] - g["referencias"][clave]["map"]["0.5"]["promedio"]))
                ga = g["referencias"][clave]["alerta"].get("0.1")
                if ga:
                    peor = max(peor, abs(res[n][clave]["alerta"]["0.1"]["fpr_controlado"] - ga["fpr_controlado"]))
    print(f"control contra la fase f (f1, map y falsas alarmas): diferencia maxima {peor:.2e}", flush=True)
    if peor > 1e-9:
        raise SystemExit("la evaluacion no reproduce la fase f")

    def grupo(m, semillas=None, g="base"):
        return [c["nombre"] for c in lista if c["modelo"] == m and c["grupo"] == g and (semillas is None or c["semilla"] in semillas)]

    por_modelo = {m: [res[n] for n in grupo(m)] for m in ORDEN if grupo(m)}
    n_semillas = {m: len(grupo(m)) for m in por_modelo}
    pesos = pesos_bootstrap(refs, a.remuestreos, a.semilla)
    boot = {}
    for clave in refs:
        for m in por_modelo:
            for etiqueta, sem in (("5", None), ("3", (42, 43, 44))):
                nombres = grupo(m, sem if m in TEMPORALES else None)
                if nombres:
                    boot[(clave, m, etiqueta)] = muestras_modelo([res[n][clave]["conf_videos"] for n in nombres], pesos[clave])
    fb = leer(a.metricas_prueba)["bootstrap"]["resultado"]
    peor = 0.0
    for clave in refs:
        for x, y in (("gru", "cfc"), ("gru", "gradient boosting")):
            for k in ("f1", "kappa"):
                bx, by = boot[(clave, x, "3")][k], boot[(clave, y, "3")][k]
                g = fb[clave]["diferencias"][f"{x} - {y}"][k]
                peor = max(peor, abs(bx[0] - by[0] - g["media"]), abs(ic(bx[1] - by[1])[0] - g["ic"][0]), abs(ic(bx[1] - by[1])[1] - g["ic"][1]))
    print(f"control de los ic pareados de la fase f: diferencia maxima {peor:.2e}", flush=True)
    if peor > 1e-9:
        raise SystemExit("el bootstrap no reproduce la fase f")

    pruebas = []

    def comparar(bloque, x, y, etiqueta, clave, k):
        bx, by = boot[(clave, x, etiqueta)][k], boot[(clave, y, etiqueta)][k]
        d_p, d_s = bx[0] - by[0], bx[1] - by[1]
        pruebas.append({"bloque": bloque, "descripcion": f"{NOMBRE[x]} - {NOMBRE[y]}, {k}, {REF[clave]} ({etiqueta} semillas)", "punto": d_p,
                        "ic": ic(d_s), "p": p_normal(d_p, d_s), "x": x, "y": y, "metrica": k, "referencia": clave, "semillas": etiqueta})

    for clave in refs:
        for k in ("f1", "kappa"):
            for x, y in (("gru", "cfc"), ("gru", "gradient boosting")):
                comparar("fase F (3 semillas)", x, y, "3", clave, k)
            for x, y in (("gru", "cfc"), ("gru", "gradient boosting"), ("mamba", "gru"), ("mamba", "cfc")):
                if x in por_modelo and y in por_modelo:
                    comparar("fase F (5 semillas y Mamba)", x, y, "5", clave, k)
    ablaciones = {}
    for v in VARIANTES[1:] + ABLACIONES_EXTRA:
        for m in ("gru", "cfc"):
            nombres = grupo(m, None, f"ablacion:{v}")
            if len(nombres) < 3:
                continue
            for clave in refs:
                bv = muestras_modelo([res[n][clave]["conf_videos"] for n in nombres], pesos[clave])
                bb = boot[(clave, m, "3")]
                d_p, d_s = bv["f1"][0] - bb["f1"][0], bv["f1"][1] - bb["f1"][1]
                x = {"bloque": "ablaciones", "descripcion": f"{NOMBRE_VARIANTE[v]}, {NOMBRE[m]}, F1, {REF[clave]}", "punto": d_p, "ic": ic(d_s),
                     "p": p_normal(d_p, d_s), "variante": v, "modelo": m, "referencia": clave, "f1_variante": bv["f1"][0], "f1_variante_sd": bv["f1_sd"]}
                pruebas.append(x)
                ablaciones[(v, m, clave)] = x
    if not a.sin_robustez:
        ruta = os.path.join(a.salida, "robustez_pruebas.json")
        if os.path.exists(ruta):
            with open(ruta, encoding="utf-8") as f:
                pruebas += json.load(f)
        else:
            rp = robustez_pruebas(a, config, refs)
            os.makedirs(a.salida, exist_ok=True)
            with open(ruta, "w", encoding="utf-8") as f:
                json.dump(rp, f, indent=1)
            pruebas += rp
    holm(pruebas)
    for bloque in sorted(set(x["bloque"] for x in pruebas)):
        sub = [dict(x) for x in pruebas if x["bloque"] == bloque]
        holm(sub)
        for x, y in zip([x for x in pruebas if x["bloque"] == bloque], sub):
            x["significativa_holm_bloque"] = y["significativa_holm"]

    umbral_fijo = {}
    for m in ("gru", "cfc"):
        for n in grupo(m, (42, 43, 44)):
            c = next(x for x in lista if x["nombre"] == n)
            pred = fuentes.pred(c, "prueba")
            final, t_al = decidir(pred, None, dec, umbral_fijo=0.5)
            umbral_fijo[n] = medir(pred, final, {"0.1": t_al["0.1"]}, refs, sin_fuego, None, False)

    grupos = grupos_alerta(particion, refs, sin_fuego)
    cobertura = defaultdict(dict)
    for c in lista:
        if c["grupo"] != "base" or not c["umbral"]:
            continue
        for conjunto, g in grupos.items():
            pred = fuentes.pred(c, conjunto.split(" ")[0])
            _, t_al = decidir(pred, c["umbral"], dec)
            fila = {}
            for k, vs in g.items():
                n_al = int(sum(t_al["0.1"].get(v) is not None for v in vs))
                fila[k] = {"alarmas": n_al, "videos": len(vs), "tasa": n_al / len(vs), "ic": clopper(n_al, len(vs))}
            cobertura[conjunto][c["nombre"]] = fila

    camara = {r["archivo"]: r["camara"] for r in leer_csv(a.camara)}
    cam = {}
    for clave, ref in refs.items():
        for g in ("movil", "fija"):
            idx = [i for i, v in enumerate(ref["nivel"]) if camara.get(v) == g]
            rng = np.random.default_rng(a.semilla)
            pesos_g = np.stack([np.bincount(rng.choice(idx, len(idx)), minlength=len(ref["nivel"])) for _ in range(a.remuestreos)]).astype(np.float64)
            for m in por_modelo:
                if m == "clase mayoritaria":
                    continue
                confs = [res[n][clave]["conf_videos"] for n in grupo(m)]
                r = muestras_modelo([np.where(np.isin(np.arange(len(ref["nivel"])), idx)[:, None, None], cf, 0) for cf in confs], pesos_g)
                cam[(clave, g, m)] = {"videos": len(idx), "f1": r["f1"][0], "f1_ic": ic(r["f1"][1]), "kappa": r["kappa"][0], "kappa_ic": ic(r["kappa"][1])}

    e8 = {}
    if "mamba" in por_modelo:
        conj = conjuntos_validacion(a)
        for n in grupo("mamba"):
            c = next(x for x in lista if x["nombre"] == n)
            corr = {"pred": fuentes.pred(c, "validacion")}
            e8[n] = {}
            for clave, umbral, perm in (("elegida", None, 2.0), ("fijo", 0.5, 2.0), ("h0", None, 0.0), ("h1", None, 1.0), ("h3", None, 3.0)):
                final, t_al = decidir(corr["pred"], c["umbral"], dec, perm, umbral)
                e8[n][clave] = medir_variante(corr, final, t_al["0.1"], conj)

    validacion = {}
    for c in lista:
        if c["grupo"] != "base":
            continue
        if c["fuente"] == "congelada" or c["modelo"] in ("gradient boosting", "regresion logistica"):
            g = leer(os.path.join(a.fase_e, f"{c['nombre'].replace(' ', '_')}.json"))
            f1 = g["histeresis"]["f1"] if g else None
        elif c["fuente"] == "archivo":
            f1 = (c.get("validacion_final") or {}).get("f1")
        elif c["fuente"] == "regla":
            f1 = leer(a.regla_json)["validacion"]["f1_final"]
        else:
            continue
        validacion.setdefault(c["modelo"], []).append(f1)
    if a.reusar_costo:
        costos = leer(os.path.join(a.salida, "bloque2.json"))["costos"]
        print("costo reutilizado de bloque2.json")
    else:
        costos = {} if a.sin_costo else costo(a, config, lista)
    if not a.sin_figuras:
        figuras(a, por_modelo, refs)
    escribir(a, refs, res, lista, por_modelo, n_semillas, boot, pruebas, ablaciones, umbral_fijo, cobertura, cam, e8, costos, grupo, faltan, validacion)


def escribir(a, refs, res, lista, por_modelo, n_semillas, boot, pruebas, ablaciones, umbral_fijo, cobertura, cam, e8, costos, grupo, faltan, validacion):
    T = a.tablas
    L = ["# Bloque pendiente del Cap. 5: semillas, Mamba, regla, ablaciones, Holm, camara y figuras", ""]
    L.append("Generado por `python evaluacion_cap5.py`. Prueba evaluada una sola vez; todo se decidio en validacion. Media +- desviacion entre"
             " semillas; IC 95 % por bootstrap pareado de videos (1000 remuestreos, semilla 42, los mismos de la fase F).")
    L.append("")
    if faltan:
        L.append(f"**Aviso:** faltan {len(faltan)} corridas sin terminar: {', '.join(faltan)}.")
        L.append("")
    L.append("Semillas por modelo: " + ", ".join(f"{NOMBRE[m]} {n_semillas[m]}" for m in por_modelo) + ".")
    L.append("")
    modelos = [m for m in ORDEN if m in por_modelo]

    L += ["## F1 macro final en validacion (umbral 0.5 + histeresis 2 s, referencia de la IA)", "", "| modelo | semillas | F1 macro final en validacion |", "|---|---|---|"]
    for m in modelos:
        v = [x for x in validacion.get(m, []) if x is not None]
        if v:
            L.append(f"| {NOMBRE[m]} | {len(v)} | {np.mean(v):.3f}{'' if len(v) < 2 else f' +- {np.std(v):.3f}'} |")
    L.append("")
    L += ["## Nivel por segundo en la prueba (5 semillas)", "",
          "| modelo | referencia | F1 macro [IC 95 %] | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa [IC 95 %] | mAP tIoU 0.3 | mAP 0.5 | mAP 0.7 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    filas_tex = []
    for m in modelos:
        for clave in refs:
            rs = [r[clave] for r in por_modelo[m]]
            b = boot[(clave, m, "5")]
            f1, k = media([r["f1"] for r in rs]), media([r["kappa"] for r in rs])
            mp = {u: media([r["map"][u]["promedio"] for r in rs]) for u in TIOU}
            L.append(f"| {NOMBRE[m]} | {REF[clave]} | {f1[0]:.3f}{'' if f1[1] is None else f' +- {f1[1]:.3f}'} [{ic(b['f1'][1])[0]:.3f}, {ic(b['f1'][1])[1]:.3f}] | "
                     + " | ".join(f"{media([r['f1_clase'][j] for r in rs])[0]:.3f}" for j in range(3))
                     + f" | {media([r['exactitud'] for r in rs])[0]:.3f} | {k[0]:.3f} [{ic(b['kappa'][1])[0]:.3f}, {ic(b['kappa'][1])[1]:.3f}] | "
                     + " | ".join(f"{mp[u][0]:.3f}" for u in TIOU) + " |")
            filas_tex.append([NOMBRE[m][0].upper() + NOMBRE[m][1:], ref_tex(clave), f"{pm(*f1)} {ic_tex(ic(b['f1'][1]))}",
                              f"{pm(*k)} {ic_tex(ic(b['kappa'][1]))}", pm(*mp["0.5"])])
    L.append("")
    tabular(os.path.join(T, "e9_nivel.tex"), ["Modelo", "Referencia", "F1 macro [IC 95\\,\\%]", "Kappa lineal [IC 95\\,\\%]", "mAP tIoU 0{,}5"],
            filas_tex, "llccc")

    L += ["## Alerta (alfa 0.10), tiempo al peligro y confiabilidad en la prueba (5 semillas)", "",
          "| modelo | falsas alarmas sin fuego | controlados IA | controlados humana | EWT mediana IA (s) | antes del cambio IA | EWT mediana humana (s) |"
          " antes del cambio humana | MAE tau IA (s) | MAE tau humana (s) | mediana tau IA (s) | mediana tau humana (s) | ECE IA | ECE humana | Brier IA |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    filas_tex = []
    for m in modelos:
        if m == "clase mayoritaria":
            continue
        rs = por_modelo[m]
        al = {c: [r[c]["alerta"]["0.1"] for r in rs] for c in refs}
        tau = {c: [r[c].get("tau") for r in rs] for c in refs}
        cf = {c: [r[c].get("confiabilidad") for r in rs] for c in refs}
        sf = media([x["fpr_sin_fuego"] for x in al["ia"]])
        ci, ch = media([x["fpr_controlado"] for x in al["ia"]]), media([x["fpr_controlado"] for x in al["humano"]])
        ei, eh = media([x["ewt_mediana"] for x in al["ia"]]), media([x["ewt_mediana"] for x in al["humano"]])
        ti = media([x["mae_s"] if x else None for x in tau["ia"]])
        th = media([x["mae_s"] if x else None for x in tau["humano"]])
        mi = media([x["mediana_s"] if x else None for x in tau["ia"]])
        mh = media([x["mediana_s"] if x else None for x in tau["humano"]])
        ece_i, ece_h = media([x["ece"] if x else None for x in cf["ia"]]), media([x["ece"] if x else None for x in cf["humano"]])
        bri = media([x["brier"] if x else None for x in cf["ia"]])
        f = lambda v, d=3: "-" if v[0] is None else f"{v[0]:.{d}f}"
        L.append(f"| {NOMBRE[m]} | {f(sf)} | {f(ci)} | {f(ch)} | {f(ei, 1)} | {media([x['antes_del_cambio'] for x in al['ia']])[0]:.1f} de {al['ia'][0]['n_cambio']} |"
                 f" {f(eh, 1)} | {media([x['antes_del_cambio'] for x in al['humano']])[0]:.1f} de {al['humano'][0]['n_cambio']} | {f(ti, 1)} | {f(th, 1)} |"
                 f" {f(mi, 1)} | {f(mh, 1)} | {f(ece_i)} | {f(ece_h)} | {f(bri)} |")
        filas_tex.append([NOMBRE[m][0].upper() + NOMBRE[m][1:], pct(sf[0]), f"{pct(ci[0])} / {pct(ch[0])}", pm(ei[0], ei[1], 1, True),
                          pm(eh[0], eh[1], 1, True), pm(ti[0], ti[1], 1), pm(th[0], th[1], 1), pm(*ece_i)])
    L.append("")
    tabular(os.path.join(T, "e9_alerta.tex"), ["Modelo", "Falsas alarmas sin fuego", "Falsas alarmas en controlados (IA / humana)", "EWT IA (s)",
                                               "EWT humana (s)", "MAE $\\tau$ IA (s)", "MAE $\\tau$ humana (s)", "ECE IA"], filas_tex, "lccccccc")

    L += ["## Alerta con alfa 0.05 (5 semillas)", "", "| modelo | referencia | sin fuego | controlados | EWT mediana (s) | antes del cambio | sin alerta |",
          "|---|---|---|---|---|---|---|"]
    for m in modelos:
        for clave in refs:
            al = [r[clave]["alerta"].get("0.05") for r in por_modelo[m] if r[clave]["alerta"].get("0.05")]
            if al:
                L.append(f"| {NOMBRE[m]} | {REF[clave]} | {media([x['fpr_sin_fuego'] for x in al])[0]:.3f} | {media([x['fpr_controlado'] for x in al])[0]:.3f} |"
                         f" {('-' if media([x['ewt_mediana'] for x in al])[0] is None else format(media([x['ewt_mediana'] for x in al])[0], '.1f'))} |"
                         f" {media([x['antes_del_cambio'] for x in al])[0]:.1f} de {al[0]['n_cambio']} | {media([x['sin_alerta'] for x in al])[0]:.1f} |")
    L.append("")

    L += ["## Cambios de nivel (5 semillas)", "", "| modelo | referencia | a incipiente: detectados | mediana del error (s) | a incontrolado: detectados |"
          " mediana del error (s) | cambios de mas por video | cambios por minuto | cambios por minuto de la referencia |", "|---|---|---|---|---|---|---|---|---|"]
    filas_tex = []
    for m in modelos:
        for clave in refs:
            rs = [r[clave] for r in por_modelo[m]]
            ci_ = rs[0]["cambios"]
            g = lambda t, k: media([r["cambios"][t][k] for r in rs])[0]
            cpm = media([r["cambios_min"] for r in rs])
            L.append(f"| {NOMBRE[m]} | {REF[clave]} | {g('a_incipiente', 'detectados'):.1f} de {ci_['a_incipiente']['cambios']} |"
                     f" {('-' if g('a_incipiente', 'mediana_abs') is None else format(g('a_incipiente', 'mediana_abs'), '.1f'))} |"
                     f" {g('a_incontrolado', 'detectados'):.1f} de {ci_['a_incontrolado']['cambios']} |"
                     f" {('-' if g('a_incontrolado', 'mediana_abs') is None else format(g('a_incontrolado', 'mediana_abs'), '.1f'))} |"
                     f" {media([r['cambios']['cambios_de_mas_por_video'] for r in rs])[0]:.2f} | {cpm[0]:.2f} | {rs[0]['cambios_min_referencia']:.2f} |")
            filas_tex.append([NOMBRE[m][0].upper() + NOMBRE[m][1:], ref_tex(clave), pm(cpm[0], cpm[1], 2), num(rs[0]["cambios_min_referencia"], 2)])
    L.append("")
    tabular(os.path.join(T, "b6_estabilidad_5s.tex"), ["Modelo", "Referencia", "Cambios por minuto", "Cambios por minuto de la referencia"], filas_tex, "llcc")

    filas_tex = []
    L += ["## B4. Confiabilidad (5 semillas)", "", "| modelo | referencia | ECE | Brier |", "|---|---|---|---|"]
    for m in modelos:
        for clave in refs:
            cf = [r[clave].get("confiabilidad") for r in por_modelo[m]]
            if all(cf):
                e, b = media([x["ece"] for x in cf]), media([x["brier"] for x in cf])
                L.append(f"| {NOMBRE[m]} | {REF[clave]} | {e[0]:.3f} | {b[0]:.3f} |")
                filas_tex.append([NOMBRE[m][0].upper() + NOMBRE[m][1:], ref_tex(clave), pm(*e), pm(*b)])
    L.append("")
    tabular(os.path.join(T, "b4_confiabilidad_5s.tex"), ["Modelo", "Referencia", "ECE", "Brier"], filas_tex, "llcc")

    L += ["## B5. Cobertura de la alerta conformal (alfa 0.10, por video, 5 semillas)", "",
          "| conjunto | modelo | controlados | sin fuego | juntos | corridas con el 10 % dentro del IC (juntos) |", "|---|---|---|---|---|---|"]
    filas_tex = []
    for conjunto, filas in cobertura.items():
        for m in modelos:
            nombres = [n for n in filas if next(c for c in lista if c["nombre"] == n)["modelo"] == m]
            if not nombres:
                continue
            celdas, celdas_tex = [], []
            for k in ("controlados", "sin fuego", "juntos"):
                tasas = [filas[n][k]["tasa"] for n in nombres]
                celdas.append(f"{100 * np.mean(tasas):.1f} % ({np.mean([filas[n][k]['alarmas'] for n in nombres]):.1f} de {filas[nombres[0]][k]['videos']})")
                celdas_tex.append(f"{pct(float(np.mean(tasas)))} ({num(float(np.mean([filas[n][k]['alarmas'] for n in nombres])), 1)} de {filas[nombres[0]][k]['videos']})")
            dentro = sum(filas[n]["juntos"]["ic"][0] <= 0.10 <= filas[n]["juntos"]["ic"][1] for n in nombres)
            L.append(f"| {conjunto} | {NOMBRE[m]} | " + " | ".join(celdas) + f" | {dentro} de {len(nombres)} |")
            nombre_c = {"calibracion": "Calibración", "validacion": "Validación", "prueba (IA)": "Prueba (IA)", "prueba (humana)": "Prueba (humana)"}[conjunto]
            filas_tex.append([nombre_c, NOMBRE[m][0].upper() + NOMBRE[m][1:]] + celdas_tex + [f"{dentro} de {len(nombres)}"])
    L.append("")
    tabular(os.path.join(T, "b5_cobertura_conformal_5s.tex"), ["Conjunto", "Modelo", "Controlados", "Sin fuego", "Juntos",
                                                                "Corridas con 10\\,\\% en el IC"], filas_tex, "llcccc")

    L += ["## B8. Tiempo al peligro (5 semillas)", "", "| modelo | referencia | videos | error absoluto mediano (s) | MAE (s) |", "|---|---|---|---|---|"]
    filas_tex = []
    for m in TEMPORALES:
        if m not in por_modelo:
            continue
        for clave in refs:
            tt = [r[clave]["tau"] for r in por_modelo[m]]
            me, ma = media([x["mediana_s"] for x in tt]), media([x["mae_s"] for x in tt])
            L.append(f"| {NOMBRE[m]} | {REF[clave]} | {tt[0]['videos']} | {me[0]:.1f} +- {me[1]:.1f} | {ma[0]:.1f} +- {ma[1]:.1f} |")
            filas_tex.append([NOMBRE[m], ref_tex(clave), str(tt[0]["videos"]), pm(me[0], me[1], 1), pm(ma[0], ma[1], 1)])
    L.append("")
    tabular(os.path.join(T, "b8_tiempo_peligro_5s.tex"), ["Modelo", "Referencia", "Videos", "Error absoluto mediano (s)", "MAE (s)"], filas_tex, "llccc")

    if costos:
        L += ["## Costo por paso (lote 1)", "", "| modelo | parametros | CPU mediana (ms) | CPU p95 (ms) | GPU mediana (ms) | GPU p95 (ms) |", "|---|---|---|---|---|---|"]
        filas_tex = []
        for m, r in costos.items():
            g = r.get("gpu", {})
            L.append(f"| {NOMBRE[m]} | {r['parametros']} | {r['cpu']['mediana_ms']:.3f} | {r['cpu']['p95_ms']:.3f} | {g.get('mediana_ms', float('nan')):.3f} |"
                     f" {g.get('p95_ms', float('nan')):.3f} |")
            filas_tex.append([NOMBRE[m], miles(r["parametros"]), num(r["cpu"]["mediana_ms"]), num(r["cpu"]["p95_ms"]), num(g.get("mediana_ms")),
                              num(g.get("p95_ms"))])
        L.append("")
        tabular(os.path.join(T, "e12_costo.tex"), ["Modelo", "Parámetros", "CPU, mediana (ms)", "CPU, p95 (ms)", "GPU, mediana (ms)", "GPU, p95 (ms)"],
                filas_tex)

    L += ["## Comparaciones pareadas de la prueba", "", "| bloque | comparacion | diferencia | IC 95 % | p | p de Holm (global) | significativa tras Holm |",
          "|---|---|---|---|---|---|---|"]
    for x in pruebas:
        if x["bloque"].startswith("fase F"):
            L.append(f"| {x['bloque']} | {x['descripcion']} | {x['punto']:+.3f} | {x['ic'][0]:+.3f} a {x['ic'][1]:+.3f} | {x['p']:.4f} | {x['p_holm']:.4f} |"
                     f" {'si' if x['significativa_holm'] else 'no'} |")
    L.append("")

    L += ["## Ablaciones (GRU y CfC, semillas 42 a 44)", "",
          "Diferencia de F1 macro frente al sistema completo (las mismas semillas congeladas). * = el IC sigue sin el cero despues de Holm (familia global).", "",
          "| variante | GRU IA | GRU humana | CfC IA | CfC humana |", "|---|---|---|---|---|"]
    filas_tex = []
    fila_md, fila_t = ["sistema completo (F1)"], ["Sistema completo (F1)"]
    for m in ("gru", "cfc"):
        for clave in refs:
            b = boot[(clave, m, "3")]
            fila_md.append(f"{b['f1'][0]:.3f}")
            fila_t.append(pm(b["f1"][0], b["f1_sd"]))
    L.append("| " + " | ".join(fila_md) + " |")
    filas_tex.append(fila_t)
    def etiqueta_tex(nombre):
        letra, resto = nombre.split(". ", 1)
        resto = resto.replace("0,1", "0{,}1")
        if resto.startswith("dt "):
            return f"{letra}. $dt$ {resto[3:]}"
        return f"{letra}. {resto[0].upper()}{resto[1:]}"

    orden = VARIANTES[1:4] + ["umbral_fijo"] + VARIANTES[4:] + ABLACIONES_EXTRA
    for v in orden:
        if v == "umbral_fijo":
            fila_md, fila_t = ["d. umbral de alerta fijo 0.5"], [etiqueta_tex("d. umbral de alerta fijo 0{,}5")]
            for m in ("gru", "cfc"):
                for clave in refs:
                    fila_md.append("+0.000 (no cambia el nivel)")
                    fila_t.append(num(0.0, 3))
            L.append("| " + " | ".join(fila_md) + " |")
            filas_tex.append(fila_t)
            continue
        fila_md, fila_t, hay = [NOMBRE_VARIANTE[v]], [etiqueta_tex(NOMBRE_VARIANTE[v])], False
        for m in ("gru", "cfc"):
            for clave in refs:
                x = ablaciones.get((v, m, clave))
                if x is None:
                    fila_md.append("-")
                    fila_t.append("--")
                    continue
                hay = True
                s = "*" if x["significativa_holm"] else ""
                fila_md.append(f"{x['punto']:+.3f} [{x['ic'][0]:+.3f}, {x['ic'][1]:+.3f}]{s}")
                fila_t.append(f"{num(x['punto'], 3, True)} {ic_tex(x['ic'])}{'$^{*}$' if s else ''}")
        if hay:
            L.append("| " + " | ".join(fila_md) + " |")
            filas_tex.append(fila_t)
    L.append("")
    tabular(os.path.join(T, "e10_ablaciones.tex"), ["Variante", "GRU IA", "GRU humana", "CfC IA", "CfC humana"], filas_tex, "lcccc")
    L += ["Variante d (umbral fijo 0.5, sin reentrenar), semillas 42 a 44:", "",
          "| modelo | referencia | falsas alarmas en controlados: conformal | umbral fijo | EWT mediana: conformal (s) | umbral fijo (s) | antes del cambio: conformal | umbral fijo |",
          "|---|---|---|---|---|---|---|---|"]
    for m in ("gru", "cfc"):
        nombres = grupo(m, (42, 43, 44))
        for clave in refs:
            c_ = [res[n][clave]["alerta"]["0.1"] for n in nombres]
            f_ = [umbral_fijo[n][clave]["alerta"]["0.1"] for n in nombres]
            g = lambda xs, k: media([x[k] for x in xs])[0]
            L.append(f"| {NOMBRE[m]} | {REF[clave]} | {g(c_, 'fpr_controlado'):.3f} | {g(f_, 'fpr_controlado'):.3f} | {g(c_, 'ewt_mediana'):.1f} |"
                     f" {g(f_, 'ewt_mediana'):.1f} | {g(c_, 'antes_del_cambio'):.1f} | {g(f_, 'antes_del_cambio'):.1f} |")
    L.append("")

    L += ["## Holm", "", f"Familia global: {len(pruebas)} comparaciones pareadas. Valor p por la aproximacion normal del bootstrap.", ""]
    sig_antes = [x for x in pruebas if x["significativa_sin_correccion"]]
    sig_despues = [x for x in pruebas if x["significativa_holm"]]
    L.append(f"- Con el IC sin corregir no incluyen el cero {len(sig_antes)} de {len(pruebas)}; despues de Holm siguen significativas {len(sig_despues)}.")
    for bloque in sorted(set(x["bloque"] for x in pruebas)):
        sub = [x for x in pruebas if x["bloque"] == bloque]
        L.append(f"- {bloque}: {len(sub)} comparaciones; sin corregir {sum(x['significativa_sin_correccion'] for x in sub)}; Holm global"
                 f" {sum(x['significativa_holm'] for x in sub)}; Holm dentro del bloque {sum(x['significativa_holm_bloque'] for x in sub)}.")
    L += ["", "| bloque | comparacion | diferencia | IC 95 % | p | p de Holm |", "|---|---|---|---|---|---|"]
    for x in sorted(sig_antes, key=lambda x: x["p"]):
        L.append(f"| {x['bloque']} | {x['descripcion']}{' **(Holm)**' if x['significativa_holm'] else ''} | {x['punto']:+.3f} |"
                 f" {x['ic'][0]:+.3f} a {x['ic'][1]:+.3f} | {x['p']:.2e} | {x['p_holm']:.2e} |")
    L.append("")

    L += ["## Camara movil contra fija (criterio del 2026-11-13; 5 semillas)", "",
          "| modelo | referencia | F1 movil [IC] | F1 fija [IC] | kappa movil [IC] | kappa fija [IC] |", "|---|---|---|---|---|---|"]
    filas_tex = []
    for m in modelos:
        if m == "clase mayoritaria":
            continue
        for clave in refs:
            mv, fj = cam.get((clave, "movil", m)), cam.get((clave, "fija", m))
            if not mv:
                continue
            L.append(f"| {NOMBRE[m]} | {REF[clave]} | {mv['f1']:.3f} [{mv['f1_ic'][0]:.3f}, {mv['f1_ic'][1]:.3f}] | {fj['f1']:.3f} [{fj['f1_ic'][0]:.3f}, {fj['f1_ic'][1]:.3f}] |"
                     f" {mv['kappa']:.3f} [{mv['kappa_ic'][0]:.3f}, {mv['kappa_ic'][1]:.3f}] | {fj['kappa']:.3f} [{fj['kappa_ic'][0]:.3f}, {fj['kappa_ic'][1]:.3f}] |")
            filas_tex.append([NOMBRE[m][0].upper() + NOMBRE[m][1:], ref_tex(clave), f"{num(mv['f1'])} {ic_tex(mv['f1_ic'])}", f"{num(fj['f1'])} {ic_tex(fj['f1_ic'])}",
                              f"{num(mv['kappa'])} {ic_tex(mv['kappa_ic'])}", f"{num(fj['kappa'])} {ic_tex(fj['kappa_ic'])}"])
    if cam:
        any_ = next(iter(cam.values()))
        L.append("")
        L.append("Videos por grupo: " + "; ".join(f"{REF[c]} {g} {cam[(c, g, 'gru')]['videos']}" for c in refs for g in ("movil", "fija") if (c, g, "gru") in cam) + ".")
    L.append("")
    tabular(os.path.join(T, "e11_camara.tex"), ["Modelo", "Referencia", "F1 cámara móvil", "F1 cámara fija", "Kappa cámara móvil", "Kappa cámara fija"],
            filas_tex, "llcccc")

    if e8:
        L += ["## E8 con Mamba (validacion, 5 semillas)", "", "| variante | F1 macro | falsas alarmas en controlados | sin fuego | cambios por minuto | EWT mediana (s) | antes del cambio |",
              "|---|---|---|---|---|---|---|"]
        for clave, titulo in (("elegida", "conformal, histeresis 2 s (elegida)"), ("fijo", "umbral fijo 0.5"), ("h0", "histeresis 0 s"), ("h1", "histeresis 1 s"),
                              ("h3", "histeresis 3 s")):
            rs = [e8[n][clave] for n in e8]
            L.append(f"| {titulo} | {media([r['f1'] for r in rs])[0]:.3f} | {media([r['fa_controlado'] for r in rs])[0]:.3f} | {media([r['fa_sin_fuego'] for r in rs])[0]:.3f} |"
                     f" {media([r['cambios_min'] for r in rs])[0]:.2f} | {('-' if media([r['ewt_mediana'] for r in rs])[0] is None else format(media([r['ewt_mediana'] for r in rs])[0], '.1f'))} |"
                     f" {media([r['antes_del_cambio'] for r in rs])[0]:.1f} de {rs[0]['n_cambios']} |")
        L.append("")
    eleccion = leer(os.path.join(a.entrenamientos, "mamba_eleccion.json"))
    if eleccion:
        L += ["## Eleccion de Mamba en validacion", ""]
        for p_, e in eleccion["opciones"].items():
            L.append(f"- Peso {p_} (factor {e['factor_incipiente']:g}): F1 suavizado con la semilla 42 " + ", ".join(f"{k}: {v:.4f}" for k, v in e["f1_suavizado_semilla_42"].items())
                     + f"; tasa elegida {e['tasa']:g}; F1 final en validacion (semillas 42 a 44) " + ", ".join(f"{x:.3f}" for x in e["f1_final_validacion"]) + f" (media {e['media']:.3f}).")
        L.append(f"- Elegido: peso {eleccion['elegido']}.")
        L.append("")
    regla = leer(a.regla_json)
    if regla:
        u = regla["umbrales"]
        L += ["## Regla sin aprendizaje", "", f"Umbrales elegidos en validacion: fuego {u['fuego_incipiente']:.4f} y {u['fuego_incontrolado']:.4f}; humo {u['humo_incipiente']:.4f}"
              f" y {u['humo_incontrolado']:.4f} (fraccion del fotograma). F1 macro final en validacion {regla['validacion']['f1_final']:.3f}. Umbral de alerta"
              f" (riesgo sin tope) {regla['umbral_alerta']['0.1']:.3f}.", ""]
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "bloque2.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(a.salida, "bloque2.json"), "w", encoding="utf-8") as f:
        json.dump({"pruebas": pruebas, "camara": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in cam.items()}, "costos": costos,
                   "corridas": {n: {c: {k: v for k, v in r.items() if k not in ("conf_videos",)} for c, r in x.items()} for n, x in res.items()
                                if not n.startswith(tuple(VARIANTES[1:]))}}, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"informe en {os.path.join(a.salida, 'bloque2.md')}")
