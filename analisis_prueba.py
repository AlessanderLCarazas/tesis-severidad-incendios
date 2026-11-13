import os
import json
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.stats import beta
from modelo_escena import ModeloEscena
from entrenar_temporal import verdad_escena
from entrenar_escena import cargar, resumen_conf
from fase_e import histeresis, alerta
from evaluar_prueba import inferir, por_segundo_video, tramos_de, precision_media, nivel_max, cambios_meta, sha256, md
from secuencias import leer_csv

NIVELES = ["controlado", "incipiente", "incontrolado"]
TITULO = {"cfc": "CfC", "gru": "GRU", "lstm": "LSTM"}
ORDEN = ["cfc", "gru", "lstm", "gradient boosting", "regresion logistica"]


def titulo(m):
    return TITULO.get(m, m)


def especificacion(config):
    temporales = [{"corrida": n, "modelo": c["modelo"], "pesos": c["archivo_pesos"], "sha256": c["sha256"],
                   "unidad": config["por_modelo"][c["modelo"]]["unidad_dt_s"] or 1.0, "umbral": c["umbral_alerta"]}
                  for n, c in config["corridas"].items()]
    sin_tiempo = [{"corrida": n.replace("_", " "), "modelo": n.replace("_", " "), "pesos": c["archivo"], "sha256": c["sha256"],
                   "umbral": c["umbral_alerta"]} for n, c in config["referencias_sin_tiempo"]["modelos"].items()]
    return temporales, sin_tiempo


def desde_npz(ruta, dispositivo="cpu"):
    datos, info = cargar(ruta, dispositivo)
    return {"x": datos["x"], "c": datos["c"], "ts": datos["ts"]}, {"inicio": info["inicio"], "archivo": info["archivo"], "t": info["t"]}


def media_por_segundo(x, archivo, t):
    suma, cuenta = {}, defaultdict(int)
    for i, clave in enumerate(zip(archivo.tolist(), np.floor(t).astype(np.int64).tolist())):
        if clave in suma:
            suma[clave] += x[i]
        else:
            suma[clave] = x[i].astype(np.float64).copy()
        cuenta[clave] += 1
    por_video = defaultdict(list)
    for (v, s), total in suma.items():
        por_video[v].append((s, total / cuenta[(v, s)]))
    return {v: sorted(filas, key=lambda f: f[0]) for v, filas in por_video.items()}


def predecir(datos, info, temporales, sin_tiempo, decision, dispositivo="cpu", comprobar=True):
    largos = info["inicio"][1:] - info["inicio"][:-1]
    archivo = np.repeat(info["archivo"], largos)
    indices = {}
    for v in np.unique(archivo):
        sel = np.flatnonzero(archivo == v)
        indices[str(v)] = sel[np.argsort(info["t"][sel], kind="stable")]
    tiempo = {v: info["t"][i].astype(np.float64) for v, i in indices.items()}
    u1, u2 = decision["umbral_coral"]
    corridas = []
    for e in temporales:
        if comprobar and e.get("sha256") and sha256(e["pesos"]) != e["sha256"]:
            raise SystemExit(f"los pesos de {e['corrida']} no coinciden con la configuracion")
        estado = torch.load(e["pesos"], map_location=dispositivo, weights_only=False)
        modelo = ModeloEscena(estado["tipo"], n_x=estado["n_x"], unidades=estado["unidades"], dropout=estado["dropout"]).to(dispositivo)
        modelo.load_state_dict(estado["modelo"])
        modelo.eval()
        p1, p2, riesgo, tau = inferir(modelo, datos, info, e["unidad"])
        pred = {}
        for v, i in indices.items():
            alto = np.minimum(p1[i], p2[i])
            arriba = p1[i] > u1
            pred[v] = {"t": tiempo[v], "p": np.column_stack([1.0 - p1[i], p1[i] - alto, alto]),
                       "nivel": arriba.astype(np.int64) + (arriba & (p2[i] > u2)).astype(np.int64), "riesgo": riesgo[i]}
        corridas.append({"corrida": e["corrida"], "modelo": e["modelo"], "pred": pred, "umbral": e["umbral"]})
    vec = media_por_segundo(datos["x"].cpu().numpy(), archivo, info["t"]) if sin_tiempo else {}
    for e in sin_tiempo:
        if comprobar and e.get("sha256") and sha256(e["pesos"]) != e["sha256"]:
            raise SystemExit(f"el modelo {e['corrida']} no coincide con la configuracion")
        modelo = joblib.load(e["pesos"])
        pred = {}
        for v, filas in vec.items():
            prob = modelo.predict_proba(np.array([f for _, f in filas]))
            pred[v] = {"t": np.array([s + 0.5 for s, _ in filas]), "p": prob, "nivel": prob.argmax(axis=1).astype(np.int64),
                       "riesgo": 0.5 * prob[:, 1] + prob[:, 2]}
        corridas.append({"corrida": e["corrida"], "modelo": e["modelo"], "pred": pred, "umbral": e["umbral"]})
    for c in corridas:
        c["final"] = {v: histeresis(d["nivel"], d["t"], decision["histeresis_s"]) for v, d in c["pred"].items()}
        c["t_alerta"] = {}
        for alfa, umbral in c["umbral"].items():
            ta = {}
            for v, d in c["pred"].items():
                activa = alerta(d["riesgo"], d["t"], umbral, decision["factor_desactivacion"], decision["histeresis_s"])
                ta[v] = float(d["t"][np.argmax(activa)]) if activa.any() else None
            c["t_alerta"][alfa] = ta
    return corridas


def referencias(conjunto, particion, ruta_metas):
    metas = defaultdict(dict)
    for m in leer_csv(ruta_metas):
        if m["conjunto"] == conjunto:
            metas[m["fuente"]][m["archivo"]] = m
    if conjunto != "prueba":
        metas["humano"] = dict(metas["ia"])
    refs = {}
    for clave, nombre in (("ia", "referencia de la IA"), ("humano", "referencia humana")):
        validos = sorted(v for v, m in metas[clave].items() if m["nivel_inicial"] in NIVELES and m["descartar"] != "1")
        verdad = verdad_escena(metas[clave], validos)
        validos = [v for v in validos if v in verdad]
        completa = validos if clave == "ia" else [v for v in validos if metas[clave][v]["valida"] == "cambios"]
        refs[clave] = {"titulo": nombre, "metas": metas[clave], "nivel": validos, "completa": completa, "verdad": verdad}
    sin_fuego = sorted(v for v, r in particion.items() if r["conjunto"] == conjunto and r["clase"] == "sin_fuego")
    return refs, sin_fuego


def por_video(c, ref, videos=None):
    salida = {}
    vacio = {"t": np.zeros(0), "p": np.zeros((0, 3))}
    for v in (ref["nivel"] if videos is None else videos):
        segundos, real, est, prob = por_segundo_video(c["pred"].get(v, vacio), c["final"].get(v, np.zeros(0, int)), ref["verdad"][v])
        conf = np.zeros((3, 3), int)
        np.add.at(conf, (real, est), 1)
        salida[v] = {"conf": conf, "real": tramos_de(segundos, real), "est": tramos_de(segundos, est, prob)}
    return salida


def nivel_de(pv, videos):
    return resumen_conf(np.sum([pv[v]["conf"] for v in videos], axis=0)) if videos else None


def map_de(pv, videos, umbral=0.5, nombres=None):
    nombres = nombres or videos
    aps = []
    for k in range(3):
        reales = {n: [(a, b) for a, b, nivel, _ in pv[v]["real"] if nivel == k] for n, v in zip(nombres, videos)}
        estimados = {n: [(a, b, p) for a, b, nivel, p in pv[v]["est"] if nivel == k] for n, v in zip(nombres, videos)}
        ap = precision_media({n: x for n, x in reales.items() if x}, {n: x for n, x in estimados.items() if x}, umbral)
        if ap is not None:
            aps.append(ap)
    return float(np.mean(aps)) if aps else None


def alerta_de(c, ref, sin_fuego, alfa):
    ta = c["t_alerta"][alfa]
    con = lambda vs: [ta.get(v) is not None for v in vs]
    controlados = [v for v in ref["nivel"] if nivel_max(ref["metas"][v]) == 0]
    propagacion = [v for v in ref["nivel"] if nivel_max(ref["metas"][v]) >= 1]
    cambio = {v: t for v in ref["completa"] for t, n in cambios_meta(ref["metas"][v]) if n == 2}
    ewt = [cambio[v] - ta[v] for v in sorted(cambio) if ta.get(v) is not None]
    return {"n_sin_fuego": len(sin_fuego), "alarmas_sin_fuego": int(np.sum(con(sin_fuego))),
            "fpr_sin_fuego": float(np.mean(con(sin_fuego))) if sin_fuego else None,
            "n_controlado": len(controlados), "alarmas_controlado": int(np.sum(con(controlados))),
            "fpr_controlado": float(np.mean(con(controlados))) if controlados else None,
            "propagacion_con_alerta": float(np.mean(con(propagacion))) if propagacion else None, "n_propagacion": len(propagacion),
            "n_cambio": len(cambio), "sin_alerta": len(cambio) - len(ewt), "antes_del_cambio": int(sum(x > 0 for x in ewt)),
            "ewt_mediana": float(np.median(ewt)) if ewt else None}


def clopper(k, n, nivel=0.95):
    a = (1.0 - nivel) / 2.0
    bajo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    alto = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return bajo, alto


def tabla_nivel(lineas, grupos, modelos, tablas, subconjuntos, con_map=True):
    lineas.append("| subconjunto | videos | segundos (controlado / incipiente / incontrolado) | modelo | F1 macro | F1 controlado | F1 incipiente |"
                  " F1 incontrolado | exactitud | kappa lineal |" + (" mAP tIoU 0.5 |" if con_map else ""))
    lineas.append("|---|---|---|---|---|---|---|---|---|---|" + ("---|" if con_map else ""))
    for nombre, videos in subconjuntos:
        for m in modelos:
            rs = [nivel_de(tablas[c["corrida"]], videos) for c in grupos[m]]
            if not videos or rs[0] is None:
                continue
            seg = np.array(rs[0]["conf"]).sum(axis=1)
            fila = (f"| {nombre} | {len(videos)} | {seg[0]} / {seg[1]} / {seg[2]} | {titulo(m)} | {md([r['f1'] for r in rs])} | "
                    + " | ".join(md([r["f1_clase"][k] for r in rs]) for k in range(3))
                    + f" | {md([r['exactitud'] for r in rs])} | {md([r['kappa'] for r in rs])} |")
            if con_map:
                fila += f" {md([map_de(tablas[c['corrida']], videos) for c in grupos[m]])} |"
            lineas.append(fila)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partes", nargs="+", default=["a", "b", "c", "d"], choices=["a", "b", "c", "d", "camara"])
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--fotogramas", default=os.path.join("runs_fuego", "descriptores_v3", "fotogramas.csv"))
    p.add_argument("--umbral_camara", type=float, default=0.01)
    p.add_argument("--remuestreos", type=int, default=1000)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", default=os.path.join("resultados", "prueba"))
    a = p.parse_args()

    with open(a.config, encoding="utf-8") as f:
        config = json.load(f)
    dec = config["decision"]
    alfa = f"{dec['alfa']:g}"
    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    refs, sin_fuego = referencias("prueba", particion, a.metas)
    datos, info = desde_npz(os.path.join(a.datos, "prueba.npz"))
    temporales, sin_tiempo = especificacion(config)
    corridas = predecir(datos, info, temporales, sin_tiempo, dec)
    grupos = {m: [c for c in corridas if c["modelo"] == m] for m in ORDEN}
    tablas = {clave: {c["corrida"]: por_video(c, ref) for c in corridas} for clave, ref in refs.items()}
    with open(os.path.join(a.salida, "metricas.json"), encoding="utf-8") as f:
        guardado = json.load(f)
    peor = max(abs(nivel_de(tablas[clave][c["corrida"]], refs[clave]["nivel"])["f1"]
                   - guardado["corridas"][c["corrida"]]["referencias"][clave]["final"]["f1"]) for c in corridas for clave in refs)
    print(f"diferencia maxima de f1 con la evaluacion guardada: {peor:.2e}", flush=True)
    if peor > 1e-9:
        raise SystemExit("las predicciones no coinciden con la evaluacion guardada")
    tres = ["gru", "cfc", "gradient boosting"]
    lineas = []

    if "a" in a.partes:
        lineas += ["# Analisis descriptivo de la prueba", "",
                   "Generado por `python analisis_prueba.py`. No cambia nada: repite la inferencia con la configuracion congelada"
                   f" (diferencia maxima de F1 con `metricas.json`: {peor:.1e}) y desglosa los mismos resultados.", "",
                   "## a. Metricas por subconjunto, referencia de la IA", ""]
        azar = [v for v in refs["ia"]["nivel"] if particion[v]["lista_prueba"] == "prueba_humana"]
        cambios = [v for v in refs["ia"]["nivel"] if particion[v]["lista_prueba"] == "prueba_cambios"]
        tabla_nivel(lineas, grupos, tres, tablas["ia"], [("40 al azar (`prueba_humana`)", azar), ("30 con cambios (`prueba_cambios`)", cambios),
                                                          ("los 70", refs["ia"]["nivel"])])
        lineas.append("")
        tipos = defaultdict(int)
        for v in azar:
            tipos[("azar", particion[v]["tipo"])] += 1
        for v in cambios:
            tipos[("cambios", particion[v]["tipo"])] += 1
        lineas.append("Tipos de video segun la IA: " + "; ".join(
            f"{g}: " + ", ".join(f"{t} {n}" for (gg, t), n in sorted(tipos.items()) if gg == g) for g in ("azar", "cambios")) + ".")
        lineas.append("")

    if "b" in a.partes:
        lineas += ["## b. Proporcion de niveles por segundo", "",
                   "| conjunto | referencia | videos de fuego | segundos | controlado | incipiente | incontrolado |", "|---|---|---|---|---|---|---|"]
        for conjunto in ("entrenamiento", "validacion", "calibracion", "prueba"):
            rs, _ = referencias(conjunto, particion, a.metas)
            for clave in (("ia", "humano") if conjunto == "prueba" else ("ia",)):
                cuenta = np.zeros(3, int)
                for seg in rs[clave]["verdad"].values():
                    cuenta += np.bincount(list(seg.values()), minlength=3)[:3]
                lineas.append(f"| {conjunto} | {'IA' if clave == 'ia' else 'humana'} | {len(rs[clave]['nivel'])} | {cuenta.sum()} | "
                              + " | ".join(f"{100.0 * x / cuenta.sum():.1f} % ({x})" for x in cuenta) + " |")
                if conjunto == "prueba":
                    for nombre in ("prueba_humana", "prueba_cambios"):
                        sub = np.zeros(3, int)
                        for v, seg in rs[clave]["verdad"].items():
                            if particion[v]["lista_prueba"] == nombre:
                                sub += np.bincount(list(seg.values()), minlength=3)[:3]
                        n = sum(particion[v]["lista_prueba"] == nombre for v in rs[clave]["nivel"])
                        lineas.append(f"| prueba, solo `{nombre}` | {'IA' if clave == 'ia' else 'humana'} | {n} | {sub.sum()} | "
                                      + " | ".join(f"{100.0 * x / sub.sum():.1f} % ({x})" for x in sub) + " |")
        lineas.append("")

    if "c" in a.partes:
        lineas += [f"## c. Intervalo binomial del 95 % de las falsas alarmas (alfa {alfa})", "",
                   "Intervalo exacto de Clopper-Pearson, por corrida (cada semilla es un modelo distinto sobre los mismos videos).", "",
                   "| corrida | sin fuego: alarmas | tasa | IC 95 % | controlados (IA): alarmas | tasa | IC 95 % | controlados (humana): alarmas | tasa | IC 95 % |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
        for c in corridas:
            celdas = []
            ai = alerta_de(c, refs["ia"], sin_fuego, alfa)
            ah = alerta_de(c, refs["humano"], sin_fuego, alfa)
            for k, n in ((ai["alarmas_sin_fuego"], ai["n_sin_fuego"]), (ai["alarmas_controlado"], ai["n_controlado"]),
                         (ah["alarmas_controlado"], ah["n_controlado"])):
                bajo, alto = clopper(k, n)
                celdas += [f"{k} de {n}", f"{100.0 * k / n:.1f} %", f"{100.0 * bajo:.1f} a {100.0 * alto:.1f} %"]
            lineas.append(f"| {c['corrida']} | " + " | ".join(celdas) + " |")
        lineas.append("")

    if "d" in a.partes:
        lineas += ["## d. Intervalo por bootstrap de videos del mAP promedio con tIoU 0.5", "",
                   f"{a.remuestreos} remuestreos de videos, semilla {a.semilla}; en los modelos con 3 semillas se promedia el mAP de las semillas"
                   " dentro de cada remuestreo; la diferencia usa los mismos remuestreos.", "",
                   "| referencia | videos | modelo | mAP tIoU 0.5 | IC 95 % |", "|---|---|---|---|---|"]
        diferencias = []
        for clave, ref in refs.items():
            videos = ref["completa"]
            rng = np.random.default_rng(a.semilla)
            muestras = rng.integers(0, len(videos), (a.remuestreos, len(videos)))
            valores = {}
            for m in tres:
                valores[m] = np.zeros(a.remuestreos)
                for b in range(a.remuestreos):
                    elegidos = [videos[i] for i in muestras[b]]
                    nombres = [f"{n}:{v}" for n, v in enumerate(elegidos)]
                    valores[m][b] = np.mean([map_de(tablas[clave][c["corrida"]], elegidos, 0.5, nombres) or 0.0 for c in grupos[m]])
                punto = [map_de(tablas[clave][c["corrida"]], videos) for c in grupos[m]]
                lineas.append(f"| {'IA' if clave == 'ia' else 'humana'} | {len(videos)} | {titulo(m)} | {md(punto)} |"
                              f" {np.percentile(valores[m], 2.5):.3f} a {np.percentile(valores[m], 97.5):.3f} |")
                print(f"bootstrap del map listo: {clave} {m}", flush=True)
            dif = valores["gru"] - valores["cfc"]
            punto = (np.mean([map_de(tablas[clave][c["corrida"]], videos) for c in grupos["gru"]])
                     - np.mean([map_de(tablas[clave][c["corrida"]], videos) for c in grupos["cfc"]]))
            diferencias.append(f"| {'IA' if clave == 'ia' else 'humana'} | GRU - CfC | {punto:+.3f} | {np.percentile(dif, 2.5):+.3f} a"
                               f" {np.percentile(dif, 97.5):+.3f} | {100.0 * (dif > 0).mean():.1f} % |")
        lineas += ["", "| referencia | diferencia | mAP tIoU 0.5 | IC 95 % | remuestreos con diferencia positiva |", "|---|---|---|---|---|"] + diferencias + [""]

    if "camara" in a.partes:
        fot = pd.read_csv(a.fotogramas, dtype={"archivo": str}, keep_default_na=False, na_values=[])
        fot = fot[fot["archivo"].isin({v for v, r in particion.items() if r["conjunto"] == "prueba"})]
        mov = np.sqrt(fot["tx"] ** 2 + fot["ty"] ** 2) + 0.5 * ((fot["s_k"] - 1.0).abs() + fot["rot"].abs())
        puntaje = (10.0 * mov.groupby(fot["archivo"]).mean()).to_dict()
        movil = {v for v, x in puntaje.items() if x >= a.umbral_camara}
        lineas += ["# Camara movil contra camara fija en la prueba (parte 1.3 de la robustez)", "",
                   "Generado por `python analisis_prueba.py --partes camara` con las predicciones de la fase F (configuracion congelada).", "",
                   "- Movimiento por fotograma = norma de la traslacion (fraccion del fotograma) + 0.5 (|s - 1| + |rotacion|), de la similitud"
                   " estimada por `camara.py`; puntaje del video = media por fotograma multiplicada por 10 (fraccion del fotograma por segundo).",
                   f"- Camara movil: puntaje >= {a.umbral_camara:g}; camara fija: menor. El umbral se fijo mirando solo la distribucion del"
                   " movimiento, antes de ver los resultados por grupo.",
                   f"- Videos de prueba: {len(movil)} con camara movil y {len(puntaje) - len(movil)} con camara fija"
                   f" (de fuego: {sum(v in movil for v in refs['ia']['nivel'])} y {sum(v not in movil for v in refs['ia']['nivel'])};"
                   f" sin fuego: {sum(v in movil for v in sin_fuego)} y {sum(v not in movil for v in sin_fuego)}).", ""]
        for clave, ref in refs.items():
            lineas += [f"## Nivel por segundo, {ref['titulo']}", ""]
            tabla_nivel(lineas, grupos, ORDEN, tablas[clave], [("camara fija", [v for v in ref["nivel"] if v not in movil]),
                                                               ("camara movil", [v for v in ref["nivel"] if v in movil])], con_map=False)
            lineas.append("")
        lineas += [f"## Alerta con alfa {alfa}", "",
                   "| camara | modelo | falsas alarmas sin fuego | falsas alarmas en controlados (IA) | propagacion con alerta (IA) |"
                   " EWT mediana, IA (s) | antes del cambio, IA | EWT mediana, humana (s) | antes del cambio, humana |",
                   "|---|---|---|---|---|---|---|---|---|"]
        for nombre, dentro in (("fija", lambda v: v not in movil), ("movil", lambda v: v in movil)):
            sub = {clave: dict(ref, nivel=[v for v in ref["nivel"] if dentro(v)], completa=[v for v in ref["completa"] if dentro(v)])
                   for clave, ref in refs.items()}
            sf = [v for v in sin_fuego if dentro(v)]
            for m in ORDEN:
                ai = [alerta_de(c, sub["ia"], sf, alfa) for c in grupos[m]]
                ah = [alerta_de(c, sub["humano"], sf, alfa) for c in grupos[m]]
                lineas.append(f"| {nombre} | {titulo(m)} | {md([x['fpr_sin_fuego'] for x in ai])} (de {ai[0]['n_sin_fuego']}) |"
                              f" {md([x['fpr_controlado'] for x in ai])} (de {ai[0]['n_controlado']}) |"
                              f" {md([x['propagacion_con_alerta'] for x in ai])} (de {ai[0]['n_propagacion']}) |"
                              f" {md([x['ewt_mediana'] for x in ai], '.1f')} | {md([x['antes_del_cambio'] for x in ai], '.1f')} de {ai[0]['n_cambio']} |"
                              f" {md([x['ewt_mediana'] for x in ah], '.1f')} | {md([x['antes_del_cambio'] for x in ah], '.1f')} de {ah[0]['n_cambio']} |")
        lineas.append("")
        with open(os.path.join(a.salida, "camara_videos.csv"), "w", encoding="utf-8", newline="") as f:
            f.write("archivo,puntaje_camara,camara\n")
            for v in sorted(puntaje):
                f.write(f"\"{v}\",{puntaje[v]:.5f},{'movil' if v in movil else 'fija'}\n")

    nombre = "camara.md" if a.partes == ["camara"] else "analisis.md"
    with open(os.path.join(a.salida, nombre), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
