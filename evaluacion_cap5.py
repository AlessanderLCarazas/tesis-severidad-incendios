import os
import csv
import json
import time
import argparse
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import norm
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from evaluar_prueba import por_segundo_video, tramos_de, precision_media, nivel_max, cambios_meta, evaluar_cambios, resumen_cambios
from analisis_prueba import referencias, especificacion, desde_npz, predecir
from analisis_robustez import f1_kappa, confusiones
from fase_e import histeresis, alerta
from entrenar_temporal import verdad_escena
from entrenar_escena import resumen_conf
from metricas_sistema import probs_por_segundo, confiabilidad, grupos_alerta, clopper
from experimentos_etapas import conjuntos_validacion, medir_variante
from salidas_congeladas import leer_config
from variantes_cap5 import VARIANTES, NOMBRE_VARIANTE, ABLACIONES_EXTRA
from secuencias import leer_csv
from tablas_latex import tabular, num, pm, pct, miles

ORDEN = ["cfc", "gru", "lstm", "mamba", "gradient boosting", "regresion logistica", "regla", "clase mayoritaria"]
NOMBRE = {"cfc": "CfC", "gru": "GRU", "lstm": "LSTM", "mamba": "Mamba", "gradient boosting": "gradient boosting",
          "regresion logistica": "regresión logística", "regla": "regla sin aprendizaje", "clase mayoritaria": "clase mayoritaria"}
COLOR = {"cfc": "#2a78d6", "gru": "#eb6834", "lstm": "#1baf7a", "mamba": "#6250d6", "gradient boosting": "#eda100",
         "regresion logistica": "#e87ba4", "regla": "#008300"}
MARCA = {"cfc": "o", "gru": "s", "lstm": "^", "mamba": "P", "gradient boosting": "D", "regresion logistica": "v", "regla": "X"}
TEMPORALES = ["cfc", "gru", "lstm", "mamba"]
REF = {"ia": "IA", "humano": "humana"}
TIOU = ["0.3", "0.5", "0.7"]
CONJUNTOS = ["calibracion", "validacion", "prueba"]


def leer(ruta):
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def indices_por_video(archivo, t):
    salida = {}
    for v in np.unique(archivo):
        sel = np.flatnonzero(archivo == v)
        salida[str(v)] = sel[np.argsort(t[sel], kind="stable")]
    return salida


def pred_modelo(t, archivo, p1, p2, riesgo, tau, cache):
    clave = (id(archivo), len(archivo))
    if clave not in cache:
        cache[clave] = indices_por_video(archivo, t)
    pred = {}
    for v, i in cache[clave].items():
        alto = np.minimum(p1[i], p2[i])
        arriba = p1[i] > 0.5
        pred[v] = {"t": t[i].astype(np.float64), "p": np.column_stack([1.0 - p1[i], p1[i] - alto, alto]),
                   "nivel": arriba.astype(np.int64) + (arriba & (p2[i] > 0.5)).astype(np.int64), "riesgo": riesgo[i], "tau": tau[i], "idx": i}
    return pred


class Fuentes:
    def __init__(self, a, config):
        self.a, self.config = a, config
        self.congeladas = {c: np.load(os.path.join(a.salidas, f"{c}.npz")) for c in CONJUNTOS}
        self.cache, self.preds = {}, {}

    def pred(self, corrida, conjunto):
        clave = (corrida["nombre"], conjunto)
        if clave in self.preds:
            return self.preds[clave]
        if corrida["fuente"] == "congelada":
            d = self.congeladas[conjunto]
            n = corrida["nombre"]
            pred = pred_modelo(d["t"], d["archivo"], d[f"{n}__p1"], d[f"{n}__p2"], d[f"{n}__riesgo"], d[f"{n}__tau"], self.cache)
        elif corrida["fuente"] == "archivo":
            d = np.load(os.path.join(corrida["carpeta"], f"salidas_{conjunto}.npz"))
            base = self.congeladas[conjunto]
            if len(d["t"]) != len(base["t"]) or not np.array_equal(d["archivo"], base["archivo"]):
                raise SystemExit(f"las salidas de {corrida['nombre']} no estan alineadas con las congeladas")
            pred = pred_modelo(base["t"], base["archivo"], d["p1"], d["p2"], d["riesgo"], d["tau"], self.cache)
        elif corrida["fuente"] == "regla":
            d = np.load(os.path.join(self.a.regla, f"salidas_{conjunto}.npz"))
            pred = {}
            for v, i in indices_por_video(d["archivo"], d["t"]).items():
                pred[v] = {"t": d["t"][i].astype(np.float64), "p": np.ones((len(i), 3)), "nivel": d["nivel"][i].astype(np.int64),
                           "riesgo": d["riesgo"][i], "tau": None, "idx": None}
        else:
            pred = corrida["pred"][conjunto]
        self.preds[clave] = pred
        return pred


def corridas_disponibles(a, config):
    lista = []
    for n, c in config["corridas"].items():
        lista.append({"nombre": n, "modelo": c["modelo"], "grupo": "base", "semilla": int(n.split("_s")[1]), "fuente": "congelada",
                      "umbral": c["umbral_alerta"]})
    faltan = []

    def nueva(nombre, modelo, grupo, semilla, carpeta):
        u = leer(os.path.join(carpeta, "umbral.json"))
        if u is None or not os.path.exists(os.path.join(carpeta, "salidas_prueba.npz")):
            faltan.append(nombre)
            return
        lista.append({"nombre": nombre, "modelo": modelo, "grupo": grupo, "semilla": semilla, "fuente": "archivo", "carpeta": carpeta,
                      "umbral": u["umbral_alerta"], "validacion_final": u.get("validacion_final")})

    for m in ("cfc", "gru", "lstm"):
        for s in (45, 46):
            nueva(f"{m}_s{s}", m, "base", s, os.path.join(a.pesos, "semillas", f"{m}_s{s}"))
    eleccion = leer(os.path.join(a.entrenamientos, "mamba_eleccion.json"))
    if eleccion:
        for s in (42, 43, 44, 45, 46):
            nueva(f"mamba_s{s}", "mamba", "base", s, os.path.join(a.pesos, eleccion["grupo"], f"mamba_s{s}"))
    else:
        faltan.append("mamba")
    for v in VARIANTES[1:] + ABLACIONES_EXTRA:
        for m in ("gru", "cfc"):
            for s in (42, 43, 44):
                antes = len(faltan)
                nueva(f"{v}__{m}_s{s}", m, f"ablacion:{v}", s, os.path.join(a.pesos, "ablaciones", v, f"{m}_s{s}"))
                if v in ABLACIONES_EXTRA and len(faltan) > antes:
                    faltan.pop()
    regla = leer(a.regla_json)
    if regla:
        lista.append({"nombre": "regla", "modelo": "regla", "grupo": "base", "semilla": None, "fuente": "regla", "umbral": regla["umbral_alerta"]})
    return lista, faltan


def agregar_sin_tiempo(a, config, lista):
    _, sin_tiempo = especificacion(config)
    preds = defaultdict(dict)
    for conjunto in CONJUNTOS:
        datos, info = desde_npz(os.path.join(a.datos, f"{conjunto}.npz"))
        for c in predecir(datos, info, [], sin_tiempo, config["decision"]):
            preds[c["corrida"]][conjunto] = c["pred"]
    for e in sin_tiempo:
        lista.append({"nombre": e["corrida"], "modelo": e["modelo"], "grupo": "base", "semilla": None, "fuente": "memoria",
                      "pred": preds[e["corrida"]], "umbral": e["umbral"]})


def mayoritaria(a, refs, t_prueba):
    metas = {m["archivo"]: m for m in leer_csv(a.metas) if m["conjunto"] == "entrenamiento" and m["fuente"] == "ia"}
    cuenta = np.zeros(3, int)
    for seg in verdad_escena(metas, sorted(metas)).values():
        cuenta += np.bincount(list(seg.values()), minlength=3)[:3]
    k = int(cuenta.argmax())
    return {v: {"t": t, "p": np.tile(np.eye(3)[k], (len(t), 1)), "nivel": np.full(len(t), k, np.int64), "riesgo": np.zeros(len(t)),
                "tau": None, "idx": None} for v, t in t_prueba.items()}


def decidir(pred, umbrales, dec, permanencia=None, umbral_fijo=None):
    perm = dec["histeresis_s"] if permanencia is None else permanencia
    final = {v: histeresis(d["nivel"], d["t"], perm) for v, d in pred.items()}
    t_alerta = {}
    for alfa in ("0.1", "0.05"):
        u = umbral_fijo if umbral_fijo is not None else (umbrales or {}).get(alfa)
        if u is None:
            continue
        ta = {}
        for v, d in pred.items():
            act = alerta(d["riesgo"], d["t"], u, dec["factor_desactivacion"], dec["permanencia_alerta_s"])
            ta[v] = float(d["t"][np.argmax(act)]) if act.any() else None
        t_alerta[alfa] = ta
    return final, t_alerta


def medir(pred, final, t_alerta, refs, sin_fuego, tau_ref, con_probas):
    res = {}
    vacio = {"t": np.zeros(0), "p": np.zeros((0, 3))}
    for clave, ref in refs.items():
        conf = np.zeros((len(ref["nivel"]), 3, 3), np.int64)
        reales, estimados = {k: {} for k in range(3)}, {k: {} for k in range(3)}
        cambios, segundos, cambios_ref = 0, 0, 0
        for n, v in enumerate(ref["nivel"]):
            segs, real, est, prob = por_segundo_video(pred.get(v, vacio), final.get(v, np.zeros(0, int)), ref["verdad"][v])
            np.add.at(conf[n], (real, est), 1)
            if v in ref["completa"]:
                for a0, a1, k, _ in tramos_de(segs, real):
                    reales[k].setdefault(v, []).append((a0, a1))
                for a0, a1, k, puntaje in tramos_de(segs, est, prob):
                    estimados[k].setdefault(v, []).append((a0, a1, puntaje))
                cambios += int((np.diff(est) != 0).sum())
                cambios_ref += int((np.diff(real) != 0).sum())
                segundos += len(real)
        r = resumen_conf(conf.sum(axis=0))
        r["conf_videos"] = conf
        r["map"] = {}
        for u in TIOU:
            aps = [precision_media(reales[k], estimados[k], float(u)) for k in range(3)]
            r["map"][u] = {"por_nivel": aps, "promedio": float(np.mean([x for x in aps if x is not None]))}
        r["cambios_min"] = 60.0 * cambios / segundos
        r["cambios_min_referencia"] = 60.0 * cambios_ref / segundos
        filas, extras = [], []
        for v in ref["completa"]:
            if v in pred:
                fs, extra, _ = evaluar_cambios(ref["metas"][v], pred[v], final[v])
                filas += fs
                extras.append(extra)
        r["cambios"] = resumen_cambios(filas, extras, len(ref["completa"]))
        controlados = [v for v in ref["nivel"] if nivel_max(ref["metas"][v]) == 0]
        propagacion = [v for v in ref["nivel"] if nivel_max(ref["metas"][v]) >= 1]
        con_cambio = {v: t for v in ref["completa"] for t, n in cambios_meta(ref["metas"][v]) if n == 2}
        r["alerta"] = {}
        for alfa, ta in t_alerta.items():
            con = lambda vs: [ta.get(v) is not None for v in vs]
            ewt = [con_cambio[v] - ta[v] for v in sorted(con_cambio) if ta.get(v) is not None]
            r["alerta"][alfa] = {"fpr_sin_fuego": float(np.mean(con(sin_fuego))), "alarmas_sin_fuego": int(np.sum(con(sin_fuego))),
                                 "n_sin_fuego": len(sin_fuego), "fpr_controlado": float(np.mean(con(controlados))),
                                 "alarmas_controlado": int(np.sum(con(controlados))), "n_controlado": len(controlados),
                                 "propagacion_con_alerta": float(np.mean(con(propagacion))), "n_cambio": len(con_cambio),
                                 "ewt": ewt, "ewt_mediana": float(np.median(ewt)) if ewt else None,
                                 "antes_del_cambio": int(sum(x > 0 for x in ewt)), "sin_alerta": len(con_cambio) - len(ewt)}
        if con_probas:
            pares = [x for v in ref["nivel"] for x in probs_por_segundo(pred.get(v), ref["verdad"][v])]
            r["confiabilidad"] = confiabilidad(pares)
        if tau_ref is not None and next(iter(pred.values()))["tau"] is not None:
            tr, tm, archivo = tau_ref[clave]
            sel = np.isin(archivo, list(ref["completa"])) & (tm == 1) & (tr > 0)
            est = np.zeros(len(tr), np.float32)
            for v, d in pred.items():
                est[d["idx"]] = d["tau"]
            err = np.abs(est[sel] - tr[sel])
            r["tau"] = {"mediana_s": float(np.median(err)), "mae_s": float(err.mean()), "fotogramas": int(sel.sum()),
                        "videos": int(len(set(archivo[sel])))}
        res[clave] = r
    return res


def pesos_bootstrap(refs, remuestreos, semilla):
    rng = np.random.default_rng(semilla)
    salida = {}
    for clave, ref in refs.items():
        n = len(ref["nivel"])
        muestras = rng.integers(0, n, (remuestreos, n))
        salida[clave] = np.stack([np.bincount(m, minlength=n) for m in muestras]).astype(np.float64)
    return salida


def muestras_modelo(confs, pesos):
    f1s, ks, f1p, kp = [], [], [], []
    for conf in confs:
        f, k = f1_kappa(np.tensordot(pesos, conf, axes=1))
        f1s.append(f)
        ks.append(k)
        f0, k0 = f1_kappa(conf.sum(0))
        f1p.append(float(f0))
        kp.append(float(k0))
    return {"f1": (float(np.mean(f1p)), np.mean(f1s, axis=0)), "kappa": (float(np.mean(kp)), np.nanmean(ks, axis=0)),
            "f1_sd": float(np.std(f1p)), "kappa_sd": float(np.std(kp))}


def ic(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def p_normal(punto, muestras):
    ee = float(np.std(muestras))
    return 1.0 if ee == 0 else float(2.0 * (1.0 - norm.cdf(abs(punto) / ee)))


def holm(pruebas, alfa=0.05):
    orden = sorted(range(len(pruebas)), key=lambda i: pruebas[i]["p"])
    m = len(pruebas)
    ajustado, maximo = [0.0] * m, 0.0
    for r, i in enumerate(orden):
        maximo = max(maximo, min(1.0, (m - r) * pruebas[i]["p"]))
        ajustado[i] = maximo
    for i, x in enumerate(pruebas):
        x["p_holm"] = ajustado[i]
        x["significativa_holm"] = ajustado[i] < alfa
        x["significativa_sin_correccion"] = not (x["ic"][0] <= 0.0 <= x["ic"][1])
    return pruebas


def robustez_pruebas(a, config, refs):
    from analisis_robustez import FRACCIONES, armar
    from normalizacion import Normalizador
    rob = leer(a.robustos)
    norm_x = Normalizador.cargar(config["datos"]["normalizacion"][0])
    norm_c = Normalizador.cargar(config["datos"]["normalizacion"][1])
    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    videos = sorted(v for v, r in particion.items() if r["conjunto"] == "prueba")
    temporales, sin_tiempo = especificacion(config)
    robustos = [{"corrida": n, "modelo": c["modelo"], "pesos": c["archivo_pesos"], "sha256": c["sha256"], "unidad": 1.0,
                 "umbral": c["umbral_alerta"]} for n, c in rob["corridas"].items()]
    conf, modelo_de = {"parte1": {}, "parte2": {}}, {"parte1": {}, "parte2": {}}
    for fr, carpeta in FRACCIONES:
        print(f"robustez: prediciendo {fr}", flush=True)
        datos, info, _ = armar(carpeta, videos, norm_x, norm_c)
        for parte, corr in (("parte1", predecir(datos, info, temporales, sin_tiempo, config["decision"])),
                            ("parte2", predecir(datos, info, robustos, [], config["decision"]))):
            for c in corr:
                modelo_de[parte][c["corrida"]] = c["modelo"]
                for clave, ref in refs.items():
                    conf[parte][(fr, c["corrida"], clave)] = confusiones(c, ref)
    fracciones = [f for f, _ in FRACCIONES]
    pts, smp = {}, {}
    for clave, ref in refs.items():
        n = len(ref["nivel"])
        rng = np.random.default_rng(a.semilla)
        pesos = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(a.remuestreos)]).astype(np.float64)
        for parte in ("parte1", "parte2"):
            for m in set(modelo_de[parte].values()):
                corr = [c for c, mm in modelo_de[parte].items() if mm == m]
                for fr in fracciones:
                    r = muestras_modelo([conf[parte][(fr, c, clave)] for c in corr], pesos)
                    pts[(clave, parte, m, fr)] = {"f1": r["f1"][0], "kappa": r["kappa"][0]}
                    smp[(clave, parte, m, fr)] = {"f1": r["f1"][1], "kappa": r["kappa"][1]}
    pruebas = []
    for clave in refs:
        for parte, modelos in (("parte1", ["cfc", "gru", "lstm", "gradient boosting", "regresion logistica"]), ("parte2", ["cfc", "gru", "lstm"])):
            for m in modelos:
                for fr in fracciones[1:]:
                    d_p = pts[(clave, parte, m, fr)]["f1"] - pts[(clave, parte, m, "completa")]["f1"]
                    d_s = smp[(clave, parte, m, fr)]["f1"] - smp[(clave, parte, m, "completa")]["f1"]
                    pruebas.append({"bloque": "robustez", "descripcion": f"{parte}, caida de F1 de {NOMBRE[m]} con {fr}, {REF[clave]}",
                                    "punto": d_p, "ic": ic(d_s), "p": p_normal(d_p, d_s)})
            for otro in ("gru", "lstm"):
                for fr in fracciones[1:]:
                    for k in ("f1", "kappa"):
                        rel = lambda m, t: ((pts if t == "p" else smp)[(clave, parte, m, "completa")][k] - (pts if t == "p" else smp)[(clave, parte, m, fr)][k]) / (pts if t == "p" else smp)[(clave, parte, m, "completa")][k]
                        d_p = rel(otro, "p") - rel("cfc", "p")
                        d_s = rel(otro, "s") - rel("cfc", "s")
                        pruebas.append({"bloque": "robustez", "descripcion": f"{parte}, caida relativa de {k} {NOMBRE[otro]} - CfC con {fr}, {REF[clave]}",
                                        "punto": d_p, "ic": ic(d_s), "p": p_normal(d_p, d_s)})
        for m in ("cfc", "gru", "lstm"):
            for fr in fracciones:
                for k in ("f1", "kappa"):
                    d_p = pts[(clave, "parte2", m, fr)][k] - pts[(clave, "parte1", m, fr)][k]
                    d_s = smp[(clave, "parte2", m, fr)][k] - smp[(clave, "parte1", m, fr)][k]
                    pruebas.append({"bloque": "robustez", "descripcion": f"reentrenado - congelado, {k} de {NOMBRE[m]} con {fr}, {REF[clave]}",
                                    "punto": d_p, "ic": ic(d_s), "p": p_normal(d_p, d_s)})
    guardado = leer(os.path.join(os.path.dirname(a.robustos), "analisis.json"))
    peor = 0.0
    for x in guardado["pruebas"]:
        if x["prueba"] == "reentrenado_menos_congelado":
            fr, m, clave, k = x["fraccion"], x["modelo"], x["referencia"], x["metrica"]
            peor = max(peor, abs(pts[(clave, "parte2", m, fr)][k] - pts[(clave, "parte1", m, fr)][k] - x["punto"]),
                       abs(ic(smp[(clave, "parte2", m, fr)][k] - smp[(clave, "parte1", m, fr)][k])[0] - x["ic"][0]))
    print(f"control de la robustez contra analisis.json: diferencia maxima {peor:.2e}", flush=True)
    if peor > 1e-9:
        raise SystemExit("la robustez no reproduce analisis.json")
    return pruebas


def costo(a, config, corridas):
    from metricas_sistema import medir as medir_paso
    from modelos_cap5 import cargar_modelo
    from modelo_escena import ModeloEscena
    res = {}

    @torch.no_grad()
    def medir_mamba(modelo, dispositivo, n_x):
        modelo = modelo.to(dispositivo).eval()
        x = torch.randn(1, n_x, device=dispositivo)
        c = torch.randn(1, 6, device=dispositivo)
        ts = torch.full((1, 1), 0.1, device=dispositivo)
        cache = modelo.cache_inicial(1, dispositivo)
        tiempos = []
        for i in range(a.calentamiento + a.pasos):
            if dispositivo != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            *_, cache = modelo.paso(x, c, ts, cache)
            if dispositivo != "cpu":
                torch.cuda.synchronize()
            if i >= a.calentamiento:
                tiempos.append(1000.0 * (time.perf_counter() - t0))
        t = np.array(tiempos)
        return {"mediana_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)), "media_ms": float(t.mean())}

    for m in TEMPORALES:
        c = next((x for x in corridas if x["modelo"] == m and x["semilla"] == 42 and x["grupo"] == "base"), None)
        if c is None:
            continue
        if c["fuente"] == "congelada":
            e = torch.load(config["corridas"][c["nombre"]]["archivo_pesos"], map_location="cpu", weights_only=False)
            modelo = ModeloEscena(e["tipo"], n_x=e["n_x"], unidades=e["unidades"], dropout=e["dropout"])
            modelo.load_state_dict(e["modelo"])
            n_x = e["n_x"]
            medidor = lambda d: medir_paso(modelo, d, n_x, a.pasos, a.calentamiento)
        else:
            modelo, e = cargar_modelo(os.path.join(c["carpeta"], "mejor.pt"))
            n_x = e["n_x"]
            medidor = lambda d: medir_mamba(modelo, d, n_x)
        r = {"parametros": int(sum(p.numel() for p in modelo.parameters())), "cpu": medidor("cpu")}
        if torch.cuda.is_available():
            r["gpu"] = medidor("cuda")
        res[m] = r
        print(f"costo {m}: {r['parametros']} parametros | cpu {r['cpu']['mediana_ms']:.3f} ms | gpu {r.get('gpu', {}).get('mediana_ms', float('nan')):.3f} ms", flush=True)
    return res


def coma(eje):
    f = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))
    eje.xaxis.set_major_formatter(f)
    eje.yaxis.set_major_formatter(f)


def limpiar(eje, rejilla="y"):
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    eje.grid(axis=rejilla, color="#e4e3df", linewidth=0.8)
    eje.set_axisbelow(True)


def figuras(a, por_modelo, refs):
    os.makedirs(a.figuras, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    modelos = [m for m in ORDEN if m in por_modelo and m != "clase mayoritaria"]
    fig, ejes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    xs = [float(u) for u in TIOU]
    for eje, clave in zip(ejes, refs):
        limpiar(eje)
        if "clase mayoritaria" in por_modelo:
            ys = [np.mean([r[clave]["map"][u]["promedio"] for r in por_modelo["clase mayoritaria"]]) for u in TIOU]
            eje.plot(xs, ys, color="#8a8984", linewidth=1.3, linestyle=(0, (4, 3)))
        for m in modelos:
            vals = np.array([[r[clave]["map"][u]["promedio"] for u in TIOU] for r in por_modelo[m]])
            if len(vals) > 1:
                eje.errorbar(xs, vals.mean(0), yerr=vals.std(0), color=COLOR[m], linewidth=0, elinewidth=1.1, capsize=3)
            eje.plot(xs, vals.mean(0), color=COLOR[m], linewidth=2, marker=MARCA[m], markersize=6, markeredgecolor="white", markeredgewidth=1)
        eje.set_xticks(xs)
        eje.set_xlabel("umbral de tIoU")
        eje.text(0.02, 0.98, f"Referencia {'de la IA' if clave == 'ia' else 'humana'}", transform=eje.transAxes, va="top", fontsize=9.5, color="#0b0b0b")
        coma(eje)
    ejes[0].set_ylabel("mAP de segmentos")
    leyenda = [Line2D([0], [0], color=COLOR[m], linewidth=2, marker=MARCA[m], markersize=6, markeredgecolor="white", label=NOMBRE[m]) for m in modelos]
    leyenda.append(Line2D([0], [0], color="#8a8984", linewidth=1.3, linestyle=(0, (4, 3)), label="clase mayoritaria"))
    fig.legend(handles=leyenda, loc="upper center", ncol=4, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(a.figuras, "map_tiou.pdf"))
    plt.close(fig)

    fig, ejes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    rng = np.random.default_rng(0)
    for eje, clave in zip(ejes, refs):
        limpiar(eje, "x")
        eje.axvspan(0, 200, color="#eaf2fc", zorder=0)
        eje.axvline(0, color="#52514e", linewidth=1)
        for i, m in enumerate(modelos):
            y = len(modelos) - 1 - i
            ewt = np.concatenate([r[clave]["alerta"]["0.1"]["ewt"] for r in por_modelo[m]]) if por_modelo[m] else np.zeros(0)
            sin = np.mean([r[clave]["alerta"]["0.1"]["sin_alerta"] for r in por_modelo[m]])
            eje.scatter(np.clip(ewt, -195, 195), y + rng.uniform(-0.22, 0.22, len(ewt)), s=14, color=COLOR[m], alpha=0.75, edgecolor="none", zorder=3)
            if len(ewt):
                eje.plot([np.median(ewt)] * 2, [y - 0.33, y + 0.33], color="#0b0b0b", linewidth=2.2, zorder=4)
            eje.text(1.02, (y + 0.6) / (len(modelos) + 0.3), f"{sin:.1f}".replace(".0", "").replace(".", ","), transform=eje.transAxes,
                     va="center", fontsize=8.5, color="#52514e")
        eje.set_xlim(-200, 200)
        eje.set_xlabel("adelanto de la alerta respecto al cambio a incontrolado (s)")
        eje.text(0.02, 1.01, f"Referencia {'de la IA' if clave == 'ia' else 'humana'}", transform=eje.transAxes, va="bottom", fontsize=9.5)
        eje.text(1.02, 1.01, "sin\nalerta", transform=eje.transAxes, va="bottom", fontsize=8, color="#52514e")
        coma(eje)
    ejes[0].set_yticks(range(len(modelos)))
    ejes[0].set_yticklabels([NOMBRE[m] for m in reversed(modelos)])
    ejes[0].set_ylim(-0.6, len(modelos) - 0.4)
    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(a.figuras, "adelanto_alerta.pdf"))
    plt.close(fig)

    puntos = leer(os.path.join(os.path.dirname(a.robustos), "analisis.json"))["puntos"]
    fracciones = [("completa", 0), ("quitar_25", 25), ("quitar_50", 50), ("quitar_75", 75)]
    rob_modelos = ["cfc", "gru", "lstm", "gradient boosting", "regresion logistica"]
    fig, ejes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    x = [p for _, p in fracciones]
    for fila, clave in enumerate(("ia", "humano")):
        vals = [puntos[f"{clave}|{pa}|{m}|{fr}"]["f1"] for pa, ms in (("parte1", rob_modelos), ("parte2", rob_modelos[:3])) for m in ms for fr, _ in fracciones]
        lo, hi = np.floor(min(vals) * 50 - 1) / 50, np.ceil(max(vals) * 50 + 1) / 50
        for col, parte in enumerate(("parte1", "parte2")):
            eje = ejes[fila, col]
            limpiar(eje)
            ms = rob_modelos if parte == "parte1" else rob_modelos[:3]
            if parte == "parte2":
                for m in ms:
                    eje.plot(x, [puntos[f"{clave}|parte1|{m}|{fr}"]["f1"] for fr, _ in fracciones], color=COLOR[m], linewidth=1.2,
                             linestyle=(0, (4, 3)), alpha=0.55)
            for m in ms:
                ys = np.array([puntos[f"{clave}|{parte}|{m}|{fr}"]["f1"] for fr, _ in fracciones])
                sd = np.array([puntos[f"{clave}|{parte}|{m}|{fr}"]["f1_sd"] for fr, _ in fracciones])
                if puntos[f"{clave}|{parte}|{m}|completa"]["n"] > 1:
                    eje.fill_between(x, ys - sd, ys + sd, color=COLOR[m], alpha=0.13, linewidth=0)
                eje.plot(x, ys, color=COLOR[m], linewidth=2, marker=MARCA[m], markersize=6, markeredgecolor="white", markeredgewidth=1)
            eje.set_ylim(lo, hi)
            eje.set_xticks(x)
            eje.set_xticklabels([f"{p} %" for p in x])
            eje.tick_params(labelbottom=True)
            eje.text(0.02, 0.98, f"{'Congelados' if parte == 'parte1' else 'Reentrenados con fotogramas quitados'}, referencia "
                     f"{'de la IA' if clave == 'ia' else 'humana'}", transform=eje.transAxes, va="top", fontsize=9)
            eje.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
            if col == 0:
                eje.set_ylabel("F1 macro")
            if fila == 1:
                eje.set_xlabel("fotogramas quitados")
    leyenda = [Line2D([0], [0], color=COLOR[m], linewidth=2, marker=MARCA[m], markersize=6, markeredgecolor="white", label=NOMBRE[m]) for m in rob_modelos]
    leyenda.append(Line2D([0], [0], color="#52514e", linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.7, label="congelado (a la derecha)"))
    fig.legend(handles=leyenda, loc="upper center", ncol=6, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(a.figuras, "f1_vs_quitados.pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--salidas", default=os.path.join("runs_fuego", "salidas_congeladas"))
    p.add_argument("--pesos", default=os.path.join("runs_fuego", "cap5"))
    p.add_argument("--entrenamientos", default=os.path.join("resultados", "cap5", "entrenamientos"))
    p.add_argument("--regla", default=os.path.join("runs_fuego", "cap5", "regla"))
    p.add_argument("--regla_json", default=os.path.join("resultados", "cap5", "regla.json"))
    p.add_argument("--robustos", default=os.path.join("resultados", "robustez", "modelos_robustos.json"))
    p.add_argument("--camara", default=os.path.join("resultados", "prueba", "camara_videos.csv"))
    p.add_argument("--fase_e", default=os.path.join("resultados", "fase_e_elegida"))
    p.add_argument("--metricas_prueba", default=os.path.join("resultados", "prueba", "metricas.json"))
    p.add_argument("--remuestreos", type=int, default=1000)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--pasos", type=int, default=2000)
    p.add_argument("--calentamiento", type=int, default=200)
    p.add_argument("--sin_robustez", action="store_true")
    p.add_argument("--sin_costo", action="store_true")
    p.add_argument("--reusar_costo", action="store_true")
    p.add_argument("--sin_figuras", action="store_true")
    p.add_argument("--tablas", default=os.path.join("resultados", "tablas_tesis"))
    p.add_argument("--figuras", default=os.path.join("resultados", "figuras_tesis"))
    p.add_argument("--salida", default=os.path.join("resultados", "cap5"))
    a = p.parse_args()
    from evaluacion_cap5_informe import informe
    informe(a)


if __name__ == "__main__":
    main()
