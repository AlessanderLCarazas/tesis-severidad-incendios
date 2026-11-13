import os
import csv
import json
import hashlib
import argparse
from collections import defaultdict
import numpy as np
import torch
import joblib
from modelo_escena import ModeloEscena
from entrenar_temporal import lote, verdad_escena
from entrenar_escena import cargar, resumen_conf
from fase_e import histeresis, alerta, segmentos, vector_por_segundo
from secuencias import leer_csv, instantes, RIESGO

NIVELES = ["controlado", "incipiente", "incontrolado"]
ORDEN = ["cfc", "gru", "lstm", "gradient boosting", "regresion logistica", "clase mayoritaria"]
TITULO = {"cfc": "CfC", "gru": "GRU", "lstm": "LSTM"}
TIOU = [0.3, 0.5, 0.7]


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def md(valores, formato=".3f"):
    v = np.array([x for x in valores if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if len(v) == 0:
        return "-"
    return format(v.mean(), formato) if len(v) == 1 else f"{format(v.mean(), formato)} +- {format(v.std(), formato)}"


def media(valores):
    v = [x for x in valores if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else None


@torch.no_grad()
def inferir(modelo, datos, info, unidad, tam_lote=64):
    ini = info["inicio"]
    orden = np.argsort(ini[1:] - ini[:-1])
    n = len(info["t"])
    p1, p2, riesgo, tau = (np.zeros(n, np.float32) for _ in range(4))
    d = dict(datos, ts=datos["ts"] / unidad)
    for i in range(0, len(orden), tam_lote):
        tramos = [(int(ini[j]), int(ini[j + 1])) for j in orden[i:i + tam_lote]]
        b = lote(d, tramos)
        logits, r, lt = modelo(b["x"], b["c"], b["ts"])
        pr = torch.sigmoid(logits).cpu().numpy()
        rr = torch.sigmoid(r).cpu().numpy()
        tt = torch.expm1(lt).clamp(min=0).cpu().numpy()
        for fila, (a0, a1) in enumerate(tramos):
            p1[a0:a1], p2[a0:a1] = pr[fila, :a1 - a0, 0], pr[fila, :a1 - a0, 1]
            riesgo[a0:a1], tau[a0:a1] = rr[fila, :a1 - a0], tt[fila, :a1 - a0]
    return p1, p2, riesgo, tau


def nivel_max(meta):
    ini, ti, tu, t0, t1 = instantes(meta)
    n = ini
    if ti is not None and t0 < ti < t1:
        n = max(n, 1)
    if tu is not None and t0 < tu < t1:
        n = 2
    return n


def cambios_meta(meta):
    ini, ti, tu, t0, t1 = instantes(meta)
    salida = []
    if ti is not None and ini < 1 and t0 < ti < t1 and (tu is None or ti < tu):
        salida.append((ti, 1))
    if tu is not None and ini < 2 and t0 < tu < t1:
        salida.append((tu, 2))
    return salida


def por_segundo_video(d, final, seg):
    votos = defaultdict(lambda: np.zeros(3, int))
    suma = defaultdict(lambda: np.zeros(3))
    for s, n, p in zip(np.floor(d["t"]).astype(int), final, d["p"]):
        votos[s][n] += 1
        suma[s] += p
    segundos = sorted(seg)
    est, prob = [], []
    for s in segundos:
        c = votos.get(s)
        if c is None:
            est.append(0)
            prob.append([1.0, 0.0, 0.0])
        else:
            est.append(int(np.flatnonzero(c == c.max())[-1]))
            prob.append(suma[s] / c.sum())
    return np.array(segundos, int), np.array([seg[s] for s in segundos], int), np.array(est, int), np.array(prob, float).reshape(-1, 3)


def tramos_de(segundos, nivel, prob=None):
    salida, a = [], 0
    for i in range(1, len(nivel) + 1):
        if i == len(nivel) or nivel[i] != nivel[a] or segundos[i] != segundos[i - 1] + 1:
            puntaje = 1.0 if prob is None else float(prob[a:i, nivel[a]].mean())
            salida.append((float(segundos[a]), float(segundos[i - 1] + 1), int(nivel[a]), puntaje))
            a = i
    return salida


def precision_media(reales, estimados, umbral):
    total = sum(len(x) for x in reales.values())
    if total == 0:
        return None
    filas = sorted(((p, v, a, b) for v, lista in estimados.items() for a, b, p in lista), key=lambda f: -f[0])
    usados = {v: np.zeros(len(x), bool) for v, x in reales.items()}
    tp = np.zeros(len(filas))
    for i, (_, v, a, b) in enumerate(filas):
        mejor, cual = 0.0, -1
        for j, (ra, rb) in enumerate(reales.get(v, [])):
            if usados[v][j]:
                continue
            inter = max(0.0, min(b, rb) - max(a, ra))
            iou = inter / (max(b, rb) - min(a, ra)) if inter > 0 else 0.0
            if iou > mejor:
                mejor, cual = iou, j
        if cual >= 0 and mejor >= umbral:
            usados[v][cual] = True
            tp[i] = 1.0
    if len(filas) == 0:
        return 0.0
    acum = np.cumsum(tp)
    recall = acum / total
    precision = acum / np.arange(1, len(filas) + 1)
    r = np.concatenate([[0.0], recall, [1.0]])
    q = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(q) - 2, -1, -1):
        q[i] = max(q[i], q[i + 1])
    cambia = np.flatnonzero(r[1:] != r[:-1])
    return float(((r[cambia + 1] - r[cambia]) * q[cambia + 1]).sum())


def evaluar_cambios(meta, d, final):
    ini, ti, tu, t0, t1 = instantes(meta)
    sel = (d["t"] >= t0 - 0.01) & (d["t"] <= t1 + 0.01)
    tt, ff = d["t"][sel], np.asarray(final)[sel]
    trans = [(float(tt[i]), int(ff[i - 1]), int(ff[i])) for i in range(1, len(ff)) if ff[i] != ff[i - 1]]
    usados, filas = set(), []
    for t_real, nivel in cambios_meta(meta):
        cruces = [(abs(t - t_real), n) for n, (t, a, b) in enumerate(trans) if a < nivel <= b]
        fila = {"t_real": t_real, "a_nivel": NIVELES[nivel], "t_modelo": None, "error_s": None}
        if cruces:
            n = min(cruces)[1]
            usados.add(n)
            fila.update(estado="detectado", t_modelo=trans[n][0], error_s=trans[n][0] - t_real)
        elif len(ff) == 0:
            fila["estado"] = "no_llega"
        else:
            fila["estado"] = "desde_el_inicio" if ff[0] >= nivel else "no_llega"
        filas.append(fila)
    return filas, len(trans) - len(usados), len(trans)


def primer_cruce(d, final, nivel=1):
    if len(final) == 0:
        return "sin datos"
    if final[0] >= nivel:
        return "desde el inicio"
    arriba = np.flatnonzero(np.asarray(final) >= nivel)
    return f"{d['t'][arriba[0]]:.1f}" if len(arriba) else "nunca"


def resumen_cambios(filas, extras, n_videos):
    r = {}
    for clave, nivel in (("a_incipiente", "incipiente"), ("a_incontrolado", "incontrolado"), ("todos", None)):
        fs = [f for f in filas if nivel is None or f["a_nivel"] == nivel]
        err = np.array([f["error_s"] for f in fs if f["estado"] == "detectado"], float)
        r[clave] = {"cambios": len(fs), "detectados": int(len(err)),
                    "mediana_abs": float(np.median(np.abs(err))) if len(err) else None,
                    "mediana_signo": float(np.median(err)) if len(err) else None,
                    "media_abs": float(np.mean(np.abs(err))) if len(err) else None,
                    "dentro_5s": int((np.abs(err) <= 5).sum()), "dentro_10s": int((np.abs(err) <= 10).sum()),
                    "desde_el_inicio": sum(f["estado"] == "desde_el_inicio" for f in fs),
                    "no_llega": sum(f["estado"] == "no_llega" for f in fs)}
    r["cambios_de_mas_por_video"] = float(np.mean(extras)) if extras else 0.0
    r["videos"] = n_videos
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conjunto", choices=["prueba", "validacion"], default="prueba")
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--remuestreos", type=int, default=1000)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--repetir", action="store_true")
    p.add_argument("--dispositivo", default="cpu")
    p.add_argument("--salida", default=os.path.join("resultados", "prueba"))
    a = p.parse_args()

    if a.conjunto == "prueba" and os.path.exists(os.path.join(a.salida, "metricas.json")) and not a.repetir:
        raise SystemExit(f"la prueba ya se evaluo ({a.salida}); se corre una sola vez")
    with open(a.config, encoding="utf-8") as f:
        config = json.load(f)
    dec = config["decision"]
    permanencia, factor = dec["histeresis_s"], dec["factor_desactivacion"]
    alfas = [f"{dec['alfa']:g}", f"{dec['alfa_secundario']:g}"]
    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    del_conjunto = {v: r for v, r in particion.items() if r["conjunto"] == a.conjunto}
    metas = defaultdict(dict)
    metas_ent = {}
    for m in leer_csv(a.metas):
        if m["conjunto"] == a.conjunto:
            metas[m["fuente"]][m["archivo"]] = m
        if m["conjunto"] == "entrenamiento" and m["fuente"] == "ia":
            metas_ent[m["archivo"]] = m
    if a.conjunto != "prueba":
        metas["humano"] = dict(metas["ia"])
    fuego_ia = sorted(v for v, m in metas["ia"].items() if m["nivel_inicial"] in NIVELES and m["descartar"] != "1")
    humano_ok = sorted(v for v, m in metas["humano"].items() if m["nivel_inicial"] in NIVELES and m["descartar"] != "1")
    humano_cambios = [v for v in humano_ok if metas["humano"][v]["valida"] == "cambios"]
    sin_fuego = sorted(v for v, r in del_conjunto.items() if r["clase"] == "sin_fuego")
    refs = {"ia": {"titulo": "referencia de la IA", "metas": metas["ia"], "nivel": fuego_ia, "completa": fuego_ia},
            "humano": {"titulo": "referencia humana", "metas": metas["humano"], "nivel": humano_ok, "completa": humano_cambios}}
    for r in refs.values():
        r["verdad"] = verdad_escena(r["metas"], r["nivel"])
        r["nivel"] = [v for v in r["nivel"] if v in r["verdad"]]
    uno_a_uno = sorted(v for v, r in del_conjunto.items() if r["tipo"] == "controlado_a_incipiente"
                       and (a.conjunto != "prueba" or r["lista_prueba"] == "prueba_cambios"))

    ruta = os.path.join(a.datos, f"{a.conjunto}.npz")
    datos, info = cargar(ruta, a.dispositivo)
    crudo = np.load(ruta)
    tau_ref = {"ia": (info["tau"], info["tau_mascara"])}
    if a.conjunto == "prueba":
        tau_ref["humano"] = (crudo["tau_humano"], crudo["tau_mascara_humano"])
    else:
        tau_ref["humano"] = tau_ref["ia"]
    largos = info["inicio"][1:] - info["inicio"][:-1]
    archivo = np.repeat(info["archivo"], largos)
    indices = {}
    for v in np.unique(archivo):
        sel = np.flatnonzero(archivo == v)
        indices[str(v)] = sel[np.argsort(info["t"][sel], kind="stable")]
    todos = sorted(indices)
    tiempo = {v: info["t"][indices[v]].astype(np.float64) for v in todos}

    corridas = []
    for nombre, c in config["corridas"].items():
        if sha256(c["archivo_pesos"]) != c["sha256"]:
            raise SystemExit(f"los pesos de {nombre} no coinciden con la configuracion congelada")
        estado = torch.load(c["archivo_pesos"], map_location=a.dispositivo, weights_only=False)
        modelo = ModeloEscena(estado["tipo"], n_x=estado["n_x"], unidades=estado["unidades"], dropout=estado["dropout"]).to(a.dispositivo)
        modelo.load_state_dict(estado["modelo"])
        modelo.eval()
        unidad = config["por_modelo"][c["modelo"]]["unidad_dt_s"] or 1.0
        p1, p2, riesgo, tau = inferir(modelo, datos, info, unidad)
        pred = {}
        for v in todos:
            i = indices[v]
            alto = np.minimum(p1[i], p2[i])
            arriba = p1[i] > dec["umbral_coral"][0]
            pred[v] = {"t": tiempo[v], "p": np.column_stack([1.0 - p1[i], p1[i] - alto, alto]),
                       "nivel": arriba.astype(np.int64) + (arriba & (p2[i] > dec["umbral_coral"][1])).astype(np.int64),
                       "riesgo": riesgo[i], "tau": tau[i], "idx": i}
        corridas.append({"corrida": nombre, "modelo": c["modelo"], "pred": pred, "umbral": c["umbral_alerta"]})
        print(f"inferencia de {nombre} lista", flush=True)
    vec = vector_por_segundo(ruta)
    for nombre, c in config["referencias_sin_tiempo"]["modelos"].items():
        if sha256(c["archivo"]) != c["sha256"]:
            raise SystemExit(f"el modelo {nombre} no coincide con la configuracion congelada")
        modelo = joblib.load(c["archivo"])
        pred = {}
        for v, filas in vec.items():
            prob = modelo.predict_proba(np.array([f for _, f in filas]))
            pred[v] = {"t": np.array([s + 0.5 for s, _ in filas]), "p": prob, "nivel": prob.argmax(axis=1).astype(np.int64),
                       "riesgo": 0.5 * prob[:, 1] + prob[:, 2], "tau": None, "idx": None}
        corridas.append({"corrida": nombre.replace("_", " "), "modelo": nombre.replace("_", " "), "pred": pred, "umbral": c["umbral_alerta"]})
    cuenta = np.zeros(3, int)
    for seg in verdad_escena(metas_ent, sorted(metas_ent)).values():
        cuenta += np.bincount(list(seg.values()), minlength=3)[:3]
    mayor = int(cuenta.argmax())
    pred = {v: {"t": tiempo[v], "p": np.tile(np.eye(3)[mayor], (len(tiempo[v]), 1)), "nivel": np.full(len(tiempo[v]), mayor, np.int64),
                "riesgo": np.full(len(tiempo[v]), RIESGO[mayor]), "tau": None, "idx": None} for v in todos}
    corridas.append({"corrida": "clase mayoritaria", "modelo": "clase mayoritaria", "pred": pred, "umbral": None})
    ent = np.load(os.path.join(a.datos, "entrenamiento.npz"))
    te = ent["tau"][(ent["tau_mascara"] == 1) & (ent["tau"] > 0)]
    tau_constante = float(np.median(te))

    os.makedirs(os.path.join(a.salida, "segmentos"), exist_ok=True)
    filas_cambios, filas_alerta, filas_uno = [], [], []
    for c in corridas:
        pred = c["pred"]
        final = {v: histeresis(d["nivel"], d["t"], permanencia) for v, d in pred.items()}
        c["final"] = final
        with open(os.path.join(a.salida, "segmentos", f"{c['corrida'].replace(' ', '_')}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["archivo", "t_inicio", "t_fin", "nivel"])
            for v in sorted(final):
                for t0, t1, n in segmentos(final[v], pred[v]["t"]):
                    w.writerow([v, round(t0, 2), round(t1, 2), NIVELES[n]])
        c["t_alerta"] = {}
        if c["umbral"] is not None:
            for alfa in alfas:
                ta = {}
                for v, d in pred.items():
                    activa = alerta(d["riesgo"], d["t"], c["umbral"][alfa], factor, permanencia)
                    ta[v] = float(d["t"][np.argmax(activa)]) if activa.any() else None
                c["t_alerta"][alfa] = ta
        c["res"] = {}
        for clave, ref in refs.items():
            res = {}
            for etapa, niveles in (("antes", {v: d["nivel"] for v, d in pred.items()}), ("final", final)):
                conf = np.zeros((len(ref["nivel"]), 3, 3), int)
                cambios_seg = []
                seg_real = {k: {} for k in range(3)}
                seg_est = {k: {} for k in range(3)}
                for n, v in enumerate(ref["nivel"]):
                    vacio = {"t": np.zeros(0), "p": np.zeros((0, 3))}
                    segundos, real, est, prob = por_segundo_video(pred.get(v, vacio), niveles.get(v, np.zeros(0, int)), ref["verdad"][v])
                    np.add.at(conf[n], (real, est), 1)
                    cambios_seg.append(int((np.diff(est) != 0).sum()) if len(est) > 1 else 0)
                    if etapa == "final" and v in ref["completa"]:
                        for a0, a1, k, _ in tramos_de(segundos, real):
                            seg_real[k].setdefault(v, []).append((a0, a1))
                        for a0, a1, k, puntaje in tramos_de(segundos, est, prob):
                            seg_est[k].setdefault(v, []).append((a0, a1, puntaje))
                r = resumen_conf(conf.sum(axis=0))
                r["cambios"] = float(np.mean(cambios_seg)) if cambios_seg else 0.0
                res[etapa] = r
                if etapa == "final":
                    res["conf_videos"] = conf
                    sub = np.array([v in ref["completa"] for v in ref["nivel"]], bool)
                    res["final_completa"] = resumen_conf(conf[sub].sum(axis=0))
                    res["final_completa"]["videos"] = int(sub.sum())
                    ap = {f"{u:g}": [precision_media(seg_real[k], seg_est[k], u) for k in range(3)] for u in TIOU}
                    res["map"] = {u: {"por_nivel": x, "promedio": media(x)} for u, x in ap.items()}
            filas, extras = [], []
            for v in ref["completa"]:
                d = pred.get(v)
                if d is None:
                    continue
                fs, extra, _ = evaluar_cambios(ref["metas"][v], d, final[v])
                extras.append(extra)
                for f in fs:
                    f.update(corrida=c["corrida"], referencia=clave, archivo=v)
                    filas_cambios.append(f)
                filas += fs
            res["cambios"] = resumen_cambios(filas, extras, len(ref["completa"]))
            tau_r, tau_m = tau_ref[clave]
            sel = np.zeros(len(tau_r), bool)
            for v in ref["completa"]:
                if v in indices:
                    sel[indices[v]] = True
            sel &= (tau_m == 1) & (tau_r > 0)
            res["tau"] = {"muestras": int(sel.sum()), "videos": len({str(x) for x in archivo[sel]}), "mae_s": None}
            if sel.any():
                if c["modelo"] in TITULO:
                    estimado = np.zeros(len(tau_r), np.float32)
                    for v, d in pred.items():
                        estimado[d["idx"]] = d["tau"]
                    res["tau"]["mae_s"] = float(np.abs(estimado[sel] - tau_r[sel]).mean())
                elif c["corrida"] == "clase mayoritaria":
                    res["tau"]["mae_s"] = float(np.abs(tau_constante - tau_r[sel]).mean())
            res["alerta"] = {}
            controlados = [v for v in ref["nivel"] if nivel_max(ref["metas"][v]) == 0]
            propagacion = [v for v in ref["nivel"] if nivel_max(ref["metas"][v]) >= 1]
            con_cambio = {v: t for v in ref["completa"] for t, n in cambios_meta(ref["metas"][v]) if n == 2}
            for alfa, ta in c["t_alerta"].items():
                con = lambda vs: [ta.get(v) is not None for v in vs]
                ewt = [con_cambio[v] - ta[v] for v in sorted(con_cambio) if ta.get(v) is not None]
                for v in sorted(con_cambio):
                    filas_alerta.append({"corrida": c["corrida"], "referencia": clave, "alfa": alfa, "archivo": v,
                                         "t_incontrolado": con_cambio[v], "t_alerta": ta.get(v),
                                         "ewt_s": None if ta.get(v) is None else con_cambio[v] - ta[v]})
                res["alerta"][alfa] = {
                    "umbral": c["umbral"][alfa], "n_sin_fuego": len(sin_fuego), "alarmas_sin_fuego": int(np.sum(con(sin_fuego))),
                    "fpr_sin_fuego": float(np.mean(con(sin_fuego))) if sin_fuego else None,
                    "n_controlado": len(controlados), "alarmas_controlado": int(np.sum(con(controlados))),
                    "fpr_controlado": float(np.mean(con(controlados))) if controlados else None,
                    "n_propagacion": len(propagacion), "propagacion_con_alerta": float(np.mean(con(propagacion))) if propagacion else None,
                    "n_cambio": len(con_cambio), "sin_alerta": len(con_cambio) - len(ewt), "antes_del_cambio": int(sum(x > 0 for x in ewt)),
                    "ewt_mediana": float(np.median(ewt)) if ewt else None, "ewt_q1": float(np.percentile(ewt, 25)) if ewt else None,
                    "ewt_q3": float(np.percentile(ewt, 75)) if ewt else None}
            c["res"][clave] = res
        for v in uno_a_uno:
            if v in pred:
                filas_uno.append({"corrida": c["corrida"], "modelo": c["modelo"], "archivo": v, "cruce": primer_cruce(pred[v], final[v])})
        r = c["res"]["ia"]["final"]
        print(f"{c['corrida']}: f1 {r['f1']:.3f} | kappa {r['kappa']:.3f}", flush=True)

    grupos = {m: [c for c in corridas if c["modelo"] == m] for m in ORDEN}
    presentes = [m for m in ORDEN if grupos[m]]
    rng = np.random.default_rng(a.semilla)
    boot = {}
    for clave, ref in refs.items():
        n = len(ref["nivel"])
        muestras = rng.integers(0, n, (a.remuestreos, n))
        valores = {m: {"f1": np.zeros(a.remuestreos), "kappa": np.zeros(a.remuestreos)} for m in presentes}
        for b in range(a.remuestreos):
            peso = np.bincount(muestras[b], minlength=n)
            for m in presentes:
                rs = [resumen_conf(np.tensordot(peso, c["res"][clave]["conf_videos"], axes=1)) for c in grupos[m]]
                valores[m]["f1"][b] = np.mean([r["f1"] for r in rs])
                valores[m]["kappa"][b] = np.mean([r["kappa"] for r in rs])
        ic = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        boot[clave] = {"videos": n, "modelos": {m: {k: ic(valores[m][k]) for k in ("f1", "kappa")} for m in presentes}, "diferencias": {}}
        for x, y in (("gru", "cfc"), ("gru", "gradient boosting")):
            if x in valores and y in valores:
                boot[clave]["diferencias"][f"{x} - {y}"] = {
                    k: {"media": float(np.mean([c["res"][clave]["final"][k] for c in grupos[x]])
                                       - np.mean([c["res"][clave]["final"][k] for c in grupos[y]])),
                        "ic": ic(valores[x][k] - valores[y][k]),
                        "fraccion_positiva": float(((valores[x][k] - valores[y][k]) > 0).mean())} for k in ("f1", "kappa")}

    nombre_m = lambda m: TITULO.get(m, m)
    lineas = [f"# Evaluacion en {'la prueba' if a.conjunto == 'prueba' else a.conjunto} con la configuracion congelada", ""]
    lineas.append(f"Generado por `python evaluar_prueba.py`. Configuracion: `{a.config.replace(os.sep, '/')}` ({config['fecha']})."
                  " Modelos temporales: media +- desviacion de 3 semillas.")
    lineas.append("")
    lineas.append(f"- Videos: {len(fuego_ia)} de fuego con referencia de la IA, {len(refs['humano']['nivel'])} con referencia humana"
                  f" ({len(humano_cambios)} con tramo util y cambios marcados reproduciendo el video; el resto solo tiene un nivel para todo"
                  f" el video) y {len(sin_fuego)} sin fuego.")
    lineas.append("- Nivel por segundo: nivel final (umbral 0.5 + histeresis de 2 s), mayoria de los fotogramas del segundo, dentro del tramo util"
                  " de cada referencia.")
    lineas.append("- Las dos referencias pesan igual. Localizacion, cambios, tiempo al peligro y adelanto contra la referencia humana usan solo"
                  f" los {len(humano_cambios)} videos con cambios marcados.")
    lineas.append(f"- Clase mayoritaria: {NIVELES[mayor]} (la mas frecuente por segundo en entrenamiento). No tiene alerta.")
    lineas.append("")
    for clave, ref in refs.items():
        lineas.append(f"## Nivel por segundo, {ref['titulo']} ({len(ref['nivel'])} videos)")
        lineas.append("")
        lineas.append("| modelo | F1 macro | IC 95 % | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal | IC 95 % |"
                      " cambios de nivel por video | F1 macro antes de la histeresis |")
        lineas.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for m in presentes:
            rs = [c["res"][clave] for c in grupos[m]]
            b = boot[clave]["modelos"][m]
            lineas.append(f"| {nombre_m(m)} | {md([r['final']['f1'] for r in rs])} | {b['f1'][0]:.3f} a {b['f1'][1]:.3f} | "
                          + " | ".join(md([r["final"]["f1_clase"][k] for r in rs]) for k in range(3))
                          + f" | {md([r['final']['exactitud'] for r in rs])} | {md([r['final']['kappa'] for r in rs])} |"
                          f" {b['kappa'][0]:.3f} a {b['kappa'][1]:.3f} | {md([r['final']['cambios'] for r in rs], '.2f')} |"
                          f" {md([r['antes']['f1'] for r in rs])} |")
        lineas.append("")
        if clave == "humano":
            lineas.append(f"Solo los {len(humano_cambios)} videos con cambios marcados:")
            lineas.append("")
            lineas.append("| modelo | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal |")
            lineas.append("|---|---|---|---|---|---|---|")
            for m in presentes:
                rs = [c["res"][clave]["final_completa"] for c in grupos[m]]
                lineas.append(f"| {nombre_m(m)} | {md([r['f1'] for r in rs])} | " + " | ".join(md([r["f1_clase"][k] for r in rs]) for k in range(3))
                              + f" | {md([r['exactitud'] for r in rs])} | {md([r['kappa'] for r in rs])} |")
            lineas.append("")
        lineas.append("Diferencias pareadas (bootstrap de videos, mismos remuestreos):")
        lineas.append("")
        lineas.append("| diferencia | F1 macro | IC 95 % | remuestreos con diferencia positiva | kappa | IC 95 % | remuestreos con diferencia positiva |")
        lineas.append("|---|---|---|---|---|---|---|")
        for nombre, dif in boot[clave]["diferencias"].items():
            x, y = nombre.split(" - ")
            lineas.append(f"| {nombre_m(x)} - {nombre_m(y)} | " + " | ".join(
                f"{dif[k]['media']:+.3f} | {dif[k]['ic'][0]:+.3f} a {dif[k]['ic'][1]:+.3f} | {100 * dif[k]['fraccion_positiva']:.1f} %"
                for k in ("f1", "kappa")) + " |")
        lineas.append("")
        lineas.append("Matrices de confusion en segundos (filas: referencia; columnas: modelo; media de las semillas):")
        lineas.append("")
        lineas.append("| modelo | referencia | controlado | incipiente | incontrolado |")
        lineas.append("|---|---|---|---|---|")
        for m in presentes:
            conf = np.mean([c["res"][clave]["final"]["conf"] for c in grupos[m]], axis=0)
            for k in range(3):
                lineas.append(f"| {nombre_m(m) if k == 0 else ''} | {NIVELES[k]} | " + " | ".join(f"{x:.0f}" for x in conf[k]) + " |")
        lineas.append("")
    for clave, ref in refs.items():
        lineas.append(f"## Localizacion temporal, {ref['titulo']} ({len(ref['completa'])} videos)")
        lineas.append("")
        lineas.append("mAP de segmentos por nivel. Puntuacion de un segmento: probabilidad media de su nivel. AP con interpolacion de todos los puntos.")
        lineas.append("")
        lineas.append("| modelo | " + " | ".join(f"mAP tIoU {u:g} | controlado | incipiente | incontrolado" for u in TIOU) + " |")
        lineas.append("|---|" + "---|" * (4 * len(TIOU)))
        for m in presentes:
            rs = [c["res"][clave]["map"] for c in grupos[m]]
            celdas = []
            for u in TIOU:
                celdas.append(md([r[f"{u:g}"]["promedio"] for r in rs]))
                celdas += [md([r[f"{u:g}"]["por_nivel"][k] for r in rs]) for k in range(3)]
            lineas.append(f"| {nombre_m(m)} | " + " | ".join(celdas) + " |")
        lineas.append("")
    for clave, ref in refs.items():
        lineas.append(f"## Cambios de nivel, {ref['titulo']} ({len(ref['completa'])} videos)")
        lineas.append("")
        lineas.append("Un cambio a un nivel se da por detectado si el nivel final del modelo cruza hacia ese nivel (o mas) dentro del tramo util;"
                      " el error es el instante del cruce mas cercano menos el de la referencia (positivo = el modelo llega tarde)."
                      " Desde el inicio = el modelo ya estaba en ese nivel o por encima; no llega = nunca lo alcanza.")
        lineas.append("")
        lineas.append("| modelo | cambio | cambios | detectados | mediana del error absoluto (s) | mediana con signo (s) | dentro de 5 s | dentro de 10 s |"
                      " desde el inicio | no llega | cambios de mas por video |")
        lineas.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for m in presentes:
            rs = [c["res"][clave]["cambios"] for c in grupos[m]]
            for tipo, nombre in (("a_incipiente", "a incipiente"), ("a_incontrolado", "a incontrolado")):
                lineas.append(f"| {nombre_m(m)} | {nombre} | {rs[0][tipo]['cambios']} | {md([r[tipo]['detectados'] for r in rs], '.1f')} |"
                              f" {md([r[tipo]['mediana_abs'] for r in rs], '.1f')} | {md([r[tipo]['mediana_signo'] for r in rs], '.1f')} |"
                              f" {md([r[tipo]['dentro_5s'] for r in rs], '.1f')} | {md([r[tipo]['dentro_10s'] for r in rs], '.1f')} |"
                              f" {md([r[tipo]['desde_el_inicio'] for r in rs], '.1f')} | {md([r[tipo]['no_llega'] for r in rs], '.1f')} |"
                              f" {md([r['cambios_de_mas_por_video'] for r in rs], '.2f')} |")
        lineas.append("")
    lineas.append(f"## Los {len(uno_a_uno)} videos con paso de controlado a incipiente, uno por uno")
    lineas.append("")
    lineas.append("Instante (s) del primer paso del nivel final del modelo a incipiente o mas, por semilla (42 / 43 / 44).")
    lineas.append("")
    lineas.append("| video | IA: inicial, a incipiente, a incontrolado | humano: inicial, a incipiente, a incontrolado | "
                  + " | ".join(nombre_m(m) for m in presentes if m != "clase mayoritaria") + " |")
    lineas.append("|---|---|---|" + "---|" * (len(presentes) - 1))
    for n, v in enumerate(uno_a_uno, 1):
        celdas = []
        for clave in ("ia", "humano"):
            mm = refs[clave]["metas"].get(v)
            celdas.append("-" if mm is None else "descartado" if mm["descartar"] == "1" else
                          f"{mm['nivel_inicial']}, {mm['t_incipiente'] or '-'}, {mm['t_incontrolado'] or '-'}")
        for m in presentes:
            if m != "clase mayoritaria":
                celdas.append(" / ".join(f["cruce"] for f in filas_uno if f["archivo"] == v and f["modelo"] == m))
        lineas.append(f"| {n}. {v[:40]} | " + " | ".join(celdas) + " |")
    lineas.append("")
    lineas.append("## Tiempo al peligro")
    lineas.append("")
    lineas.append("MAE de tau en segundos, en los fotogramas anteriores al cambio a incontrolado. La fila de la clase mayoritaria es una constante:"
                  f" la mediana de tau en entrenamiento ({tau_constante:.1f} s).")
    lineas.append("")
    lineas.append("| modelo | " + " | ".join(f"MAE, {r['titulo']} (s) | videos | fotogramas" for r in refs.values()) + " |")
    lineas.append("|---|" + "---|" * (3 * len(refs)))
    for m in presentes:
        if m in TITULO or m == "clase mayoritaria":
            celdas = []
            for clave in refs:
                rs = [c["res"][clave]["tau"] for c in grupos[m]]
                celdas += [md([r["mae_s"] for r in rs], ".1f"), str(rs[0]["videos"]), str(rs[0]["muestras"])]
            lineas.append(f"| {'constante (mediana de entrenamiento)' if m == 'clase mayoritaria' else nombre_m(m)} | " + " | ".join(celdas) + " |")
    lineas.append("")
    for i, alfa in enumerate(alfas):
        lineas.append(f"## Alerta con alfa {alfa}{' (configuracion final)' if i == 0 else ' (aparte)'}")
        lineas.append("")
        lineas.append("EWT = instante del cambio a incontrolado menos instante de la primera alerta; positivo = la alerta llega antes.")
        lineas.append("")
        for clave, ref in refs.items():
            lineas.append(f"{ref['titulo'][0].upper() + ref['titulo'][1:]}:")
            lineas.append("")
            lineas.append("| modelo | umbral | falsas alarmas sin fuego | falsas alarmas en controlados | propagacion con alerta | videos con cambio |"
                          " EWT mediana (s) | EWT cuartil 1 (s) | EWT cuartil 3 (s) | con alerta antes del cambio | sin alerta |")
            lineas.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for m in presentes:
                rs = [c["res"][clave]["alerta"].get(alfa) for c in grupos[m]]
                if rs[0] is None:
                    continue
                lineas.append(f"| {nombre_m(m)} | {md([r['umbral'] for r in rs])} |"
                              f" {md([r['fpr_sin_fuego'] for r in rs])} ({md([r['alarmas_sin_fuego'] for r in rs], '.1f')} de {rs[0]['n_sin_fuego']}) |"
                              f" {md([r['fpr_controlado'] for r in rs])} ({md([r['alarmas_controlado'] for r in rs], '.1f')} de {rs[0]['n_controlado']}) |"
                              f" {md([r['propagacion_con_alerta'] for r in rs])} (de {rs[0]['n_propagacion']}) | {rs[0]['n_cambio']} |"
                              f" {md([r['ewt_mediana'] for r in rs], '.1f')} | {md([r['ewt_q1'] for r in rs], '.1f')} |"
                              f" {md([r['ewt_q3'] for r in rs], '.1f')} | {md([r['antes_del_cambio'] for r in rs], '.1f')} |"
                              f" {md([r['sin_alerta'] for r in rs], '.1f')} |")
            lineas.append("")
    with open(os.path.join(a.salida, "resumen.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    for nombre, filas, campos in (("cambios.csv", filas_cambios, ["corrida", "referencia", "archivo", "a_nivel", "t_real", "t_modelo", "error_s", "estado"]),
                                  ("alerta.csv", filas_alerta, ["corrida", "referencia", "alfa", "archivo", "t_incontrolado", "t_alerta", "ewt_s"]),
                                  ("controlado_a_incipiente.csv", filas_uno, ["corrida", "modelo", "archivo", "cruce"])):
        with open(os.path.join(a.salida, nombre), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            for fila in filas:
                w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fila.items()})
    salida = {"conjunto": a.conjunto, "configuracion": a.config.replace(os.sep, "/"), "clase_mayoritaria": NIVELES[mayor],
              "tau_constante_s": tau_constante, "videos": {"fuego_ia": len(fuego_ia), "humano": len(refs["humano"]["nivel"]),
                                                             "humano_cambios": len(humano_cambios), "sin_fuego": len(sin_fuego)},
              "bootstrap": {"remuestreos": a.remuestreos, "semilla": a.semilla, "resultado": boot}, "corridas": {}}
    for c in corridas:
        salida["corridas"][c["corrida"]] = {"modelo": c["modelo"],
                                            "referencias": {k: {x: y for x, y in r.items() if x != "conf_videos"} for k, r in c["res"].items()}}
    with open(os.path.join(a.salida, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=1)
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
