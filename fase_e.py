import os
import csv
import glob
import json
import argparse
from collections import defaultdict, deque
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from modelo_escena import ModeloEscena
from entrenar_temporal import lote, confusion, verdad_escena
from entrenar_escena import cargar, resumen_conf
from secuencias import leer_csv, numero

NIVELES = ["controlado", "incipiente", "incontrolado"]
ETAPAS = [("antes", "antes (umbral 0.5)"), ("histeresis", "umbral 0.5 + histeresis (final)"),
          ("umbrales_solo", "solo umbrales ajustados"), ("despues", "umbrales ajustados + histeresis")]
ORDEN = ["cfc", "gru", "lstm", "gradient boosting", "regresion logistica"]


@torch.no_grad()
def inferir(modelo, datos, info, tam_lote=64):
    ini = info["inicio"]
    orden = np.argsort(ini[1:] - ini[:-1])
    n = len(info["t"])
    p1, p2, riesgo = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
    for i in range(0, len(orden), tam_lote):
        tramos = [(int(ini[j]), int(ini[j + 1])) for j in orden[i:i + tam_lote]]
        b = lote(datos, tramos)
        logits, r, _ = modelo(b["x"], b["c"], b["ts"])
        pr = torch.sigmoid(logits).cpu().numpy()
        rr = torch.sigmoid(r).cpu().numpy()
        for fila, (a0, a1) in enumerate(tramos):
            p1[a0:a1], p2[a0:a1], riesgo[a0:a1] = pr[fila, :a1 - a0, 0], pr[fila, :a1 - a0, 1], rr[fila, :a1 - a0]
    largos = ini[1:] - ini[:-1]
    archivo = np.repeat(info["archivo"], largos)
    por_video = {}
    for v in np.unique(archivo):
        sel = np.flatnonzero(archivo == v)
        sel = sel[np.argsort(info["t"][sel], kind="stable")]
        por_video[str(v)] = {"t": info["t"][sel].astype(np.float64), "p1": p1[sel], "p2": p2[sel], "riesgo": riesgo[sel]}
    return por_video


def nivel_crudo(d, u1, u2):
    if "nivel" in d:
        return d["nivel"]
    arriba = d["p1"] > u1
    return arriba.astype(np.int64) + (arriba & (d["p2"] > u2)).astype(np.int64)


def histeresis(nivel, t, permanencia):
    salida = np.zeros(len(nivel), np.int64)
    if len(nivel) == 0:
        return salida
    actual = int(nivel[0])
    ventana = deque()
    for i, (n, ti) in enumerate(zip(nivel, t)):
        ventana.append((ti, int(n)))
        while ventana and ventana[0][0] < ti - permanencia - 1e-6:
            ventana.popleft()
        if ti - t[0] >= permanencia - 1e-6:
            valores = [x for _, x in ventana]
            if min(valores) > actual:
                actual = min(valores)
            elif max(valores) < actual:
                actual = max(valores)
        salida[i] = actual
    return salida


def alerta(riesgo, t, umbral, factor, permanencia):
    salida = np.zeros(len(riesgo), bool)
    activa, t_bajo = False, None
    for i, (r, ti) in enumerate(zip(riesgo, t)):
        if not activa:
            if r >= umbral:
                activa, t_bajo = True, None
        else:
            if r < umbral * factor:
                t_bajo = ti if t_bajo is None else t_bajo
                if ti - t_bajo >= permanencia - 1e-6:
                    activa = False
            else:
                t_bajo = None
        salida[i] = activa
    return salida


def por_segundo(niveles, tiempos, verdad):
    real, est, cambios = [], [], []
    for v, seg in verdad.items():
        votos = defaultdict(lambda: np.zeros(3, int))
        if v in niveles:
            for s, n in zip(np.floor(tiempos[v]).astype(int), niveles[v]):
                votos[s][n] += 1
        ev = []
        for s in sorted(seg):
            c = votos.get(s)
            ev.append(0 if c is None else int(np.flatnonzero(c == c.max())[-1]))
            real.append(seg[s])
        est += ev
        cambios.append(int((np.diff(ev) != 0).sum()) if len(ev) > 1 else 0)
    r = resumen_conf(confusion(np.array(real, np.int64), np.array(est, np.int64)))
    r["cambios"] = float(np.mean(cambios)) if cambios else 0.0
    return r


def segmentos(nivel, t):
    salida, inicio = [], 0
    for i in range(1, len(nivel) + 1):
        if i == len(nivel) or nivel[i] != nivel[inicio]:
            salida.append((float(t[inicio]), float(t[i - 1]), int(nivel[inicio])))
            inicio = i
    return salida


def media_desv(valores, formato=".3f"):
    v = np.array([x for x in valores if x is not None and not np.isnan(x)], float)
    if len(v) == 0:
        return "-"
    return format(v.mean(), formato) if len(v) == 1 else f"{format(v.mean(), formato)} +- {format(v.std(), formato)}"


def vector_por_segundo(ruta):
    d = np.load(ruta)
    x = d["x"].astype(np.float64)
    largos = d["seq_inicio"][1:] - d["seq_inicio"][:-1]
    archivo = np.repeat(d["seq_archivo"], largos).tolist()
    seg = np.floor(d["t"]).astype(np.int64).tolist()
    suma, cuenta = {}, defaultdict(int)
    for i, clave in enumerate(zip(archivo, seg)):
        if clave in suma:
            suma[clave] += x[i]
        else:
            suma[clave] = x[i].copy()
        cuenta[clave] += 1
    por_video = defaultdict(list)
    for (v, s), total in suma.items():
        por_video[v].append((s, total / cuenta[(v, s)]))
    return {v: sorted(filas, key=lambda f: f[0]) for v, filas in por_video.items()}


def referencias(a, particion, metas):
    vec = {c: vector_por_segundo(os.path.join(a.datos, f"{c}.npz")) for c in ("entrenamiento", "calibracion", "validacion")}
    verdad_ent = verdad_escena(metas["entrenamiento"], sorted(metas["entrenamiento"]))
    x, y = [], []
    for v, filas in vec["entrenamiento"].items():
        for s, fila in filas:
            if particion[v]["clase"] == "sin_fuego":
                x.append(fila)
                y.append(0)
            elif s in verdad_ent.get(v, {}):
                x.append(fila)
                y.append(verdad_ent[v][s])
    x, y = np.array(x), np.array(y, int)
    modelos = {"gradient boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=a.semilla),
               "regresion logistica": LogisticRegression(class_weight="balanced", max_iter=3000, random_state=a.semilla)}
    salida = {}
    for nombre, modelo in modelos.items():
        modelo.fit(x, y)
        pred = {}
        for c in ("calibracion", "validacion"):
            pred[c] = {}
            for v, filas in vec[c].items():
                prob = modelo.predict_proba(np.array([f for _, f in filas]))
                pred[c][v] = {"t": np.array([s + 0.5 for s, _ in filas]), "nivel": prob.argmax(axis=1).astype(np.int64),
                              "riesgo": 0.5 * prob[:, 1] + prob[:, 2]}
        salida[nombre] = pred
    return salida


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pesos", default=os.path.join("runs_fuego", "modelos_escena_v2", "final"))
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--alfas", nargs="+", type=float, default=[0.10, 0.05])
    p.add_argument("--permanencia", type=float, default=2.0)
    p.add_argument("--factor_desactivar", type=float, default=0.8)
    p.add_argument("--paso_umbral", type=float, default=0.05)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--sin_referencias", action="store_true")
    p.add_argument("--dispositivo", default="cpu")
    p.add_argument("--salida", default=os.path.join("resultados", "fase_e"))
    a = p.parse_args()

    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    metas = defaultdict(dict)
    for m in leer_csv(a.metas):
        if m["fuente"] in ("ia", "dataset"):
            metas[m["conjunto"]][m["archivo"]] = m
    verdad = {c: verdad_escena(metas[c], sorted(metas[c])) for c in ("calibracion", "validacion")}
    cal_umbral = sorted(v for v, r in particion.items() if r["conjunto"] == "calibracion" and r["calibra_umbral"] == "si")
    val = {v: r for v, r in particion.items() if r["conjunto"] == "validacion"}
    sin_fuego = sorted(v for v, r in val.items() if r["clase"] == "sin_fuego")
    controlados = sorted(v for v, r in val.items() if r["clase"] == "fuego" and r["nivel_max"] == "controlado")
    propagacion = sorted(v for v, r in val.items() if r["clase"] == "fuego" and r["nivel_max"] in ("incipiente", "incontrolado"))
    con_cambio = {v: numero(metas["validacion"][v]["t_incontrolado"]) for v, r in val.items()
                  if r["tipo"].endswith("_a_incontrolado")}
    rejilla = np.round(np.arange(a.paso_umbral, 1.0, a.paso_umbral), 4)
    os.makedirs(os.path.join(a.salida, "segmentos"), exist_ok=True)
    corridas = defaultdict(list)

    def aplicar(nombre, tipo, pred, con_umbrales):
        tiempos = {c: {v: d["t"] for v, d in pred[c].items()} for c in pred}
        pv = pred["validacion"]
        crudo = {v: nivel_crudo(d, 0.5, 0.5) for v, d in pv.items()}
        final = {v: histeresis(crudo[v], d["t"], a.permanencia) for v, d in pv.items()}
        res = {"corrida": nombre, "modelo": tipo, "antes": por_segundo(crudo, tiempos["validacion"], verdad["validacion"]),
               "histeresis": por_segundo(final, tiempos["validacion"], verdad["validacion"]), "alerta": {}}
        if con_umbrales:
            mejor = (-1.0, 0.5, 0.5)
            for u1 in rejilla:
                for u2 in rejilla:
                    r = por_segundo({v: nivel_crudo(d, u1, u2) for v, d in pred["calibracion"].items()}, tiempos["calibracion"],
                                    verdad["calibracion"])
                    if r["f1"] > mejor[0] + 1e-9:
                        mejor = (r["f1"], float(u1), float(u2))
            _, u1, u2 = mejor
            ajustado = {v: nivel_crudo(d, u1, u2) for v, d in pv.items()}
            estable = {v: histeresis(ajustado[v], d["t"], a.permanencia) for v, d in pv.items()}
            res.update(umbrales=[u1, u2], f1_calibracion=mejor[0],
                       umbrales_solo=por_segundo(ajustado, tiempos["validacion"], verdad["validacion"]),
                       despues=por_segundo(estable, tiempos["validacion"], verdad["validacion"]))
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
            adelantos, detalle = [], []
            for v, tu in sorted(con_cambio.items()):
                t_al = float(pv[v]["t"][np.argmax(activa[v])]) if v in activa and activa[v].any() else None
                detalle.append({"archivo": v, "t_incontrolado": tu, "t_alerta": t_al,
                                "adelanto_s": None if t_al is None else tu - t_al})
                if t_al is not None:
                    adelantos.append(tu - t_al)
            res["alerta"][f"{alfa:g}"] = {
                "umbral": umbral, "videos_calibracion": n, "fpr_sin_fuego": float(np.mean(con(sin_fuego))),
                "n_sin_fuego": len(sin_fuego), "alarmas_sin_fuego": int(np.sum(con(sin_fuego))),
                "fpr_controlado": float(np.mean(con(controlados))), "n_controlado": len(controlados),
                "deteccion_propagacion": float(np.mean(con(propagacion))), "n_propagacion": len(propagacion),
                "con_alerta_antes": int(sum(x > 0 for x in adelantos)), "sin_alerta": len(con_cambio) - len(adelantos),
                "adelanto_mediana": float(np.median(adelantos)) if adelantos else None,
                "adelanto_media": float(np.mean(adelantos)) if adelantos else None, "detalle": detalle}
        with open(os.path.join(a.salida, f"{nombre.replace(' ', '_')}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        corridas[tipo].append(res)
        print(f"{nombre}: f1 antes {res['antes']['f1']:.3f} | con histeresis {res['histeresis']['f1']:.3f}"
              f" | cambios {res['antes']['cambios']:.1f} -> {res['histeresis']['cambios']:.1f}", flush=True)

    datos = {c: cargar(os.path.join(a.datos, f"{c}.npz"), a.dispositivo) for c in ("calibracion", "validacion")}
    for ruta in sorted(glob.glob(os.path.join(a.pesos, "*", "mejor.pt"))):
        estado = torch.load(ruta, map_location=a.dispositivo, weights_only=False)
        modelo = ModeloEscena(estado["tipo"], n_x=estado["n_x"], unidades=estado["unidades"], dropout=estado["dropout"]).to(a.dispositivo)
        modelo.load_state_dict(estado["modelo"])
        modelo.eval()
        aplicar(os.path.basename(os.path.dirname(ruta)), estado["tipo"], {c: inferir(modelo, *datos[c]) for c in datos}, True)
    if not a.sin_referencias:
        for nombre, pred in referencias(a, particion, metas).items():
            aplicar(nombre, nombre, pred, False)

    presentes = [m for m in ORDEN if corridas[m]]
    lineas = ["# Fase E en validacion: histeresis, alerta conformal y segmentos", ""]
    lineas.append("Generado por `python fase_e.py`. La prueba no se usa. Media +- desviacion entre semillas.")
    lineas.append("")
    lineas.append("- **Configuracion final: umbrales de CORAL en 0.5 con histeresis, y alerta con alfa 0.10.**")
    lineas.append(f"- Histeresis de nivel: el nivel sube cuando el nivel crudo se mantiene por encima durante {a.permanencia:g} s y"
                  f" baja cuando se mantiene por debajo durante {a.permanencia:g} s (causal).")
    lineas.append(f"- Umbrales ajustados (solo como comparacion): rejilla de {a.paso_umbral:g} sobre las dos probabilidades de CORAL,"
                  " eligiendo el mayor F1 macro de escena por segundo en los videos de fuego de calibracion.")
    lineas.append(f"- Alerta conformal por video: la puntuacion de cada video de calibracion sin fuego o controlado ({len(cal_umbral)})"
                  " es su riesgo maximo; el umbral es el cuantil conformal. La alerta se activa cuando el riesgo llega al umbral y"
                  f" se apaga cuando baja de {a.factor_desactivar:g} veces el umbral durante {a.permanencia:g} s.")
    lineas.append("- Referencias sin tiempo: una prediccion por segundo; riesgo = 0.5 P(incipiente) + P(incontrolado); misma"
                  " histeresis y misma alerta.")
    lineas.append(f"- Validacion: {len(verdad['validacion'])} videos de fuego para el nivel, {len(sin_fuego)} sin fuego,"
                  f" {len(controlados)} controlados, {len(propagacion)} con propagacion y {len(con_cambio)} con cambio a incontrolado.")
    lineas.append("")
    lineas.append("| modelo | corridas | etapa | F1 macro | F1 controlado | F1 incipiente | F1 incontrolado | exactitud | kappa lineal |"
                  " cambios de nivel por video |")
    lineas.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in presentes:
        for clave, nombre in ETAPAS:
            cs = [c for c in corridas[m] if clave in c]
            if cs:
                lineas.append(f"| {m.upper() if len(m) <= 4 else m} | {len(cs)} | {nombre} | {media_desv([c[clave]['f1'] for c in cs])} | "
                              + " | ".join(media_desv([c[clave]["f1_clase"][k] for c in cs]) for k in range(3))
                              + f" | {media_desv([c[clave]['exactitud'] for c in cs])} | {media_desv([c[clave]['kappa'] for c in cs])} |"
                              f" {media_desv([c[clave]['cambios'] for c in cs], '.2f')} |")
    lineas.append("")
    lineas.append("| modelo | alfa | umbral de riesgo | falsas alarmas en videos sin fuego | falsas alarmas en videos controlados |"
                  " videos con propagacion con alerta | adelanto mediano (s) | videos con alerta antes del cambio | sin alerta |")
    lineas.append("|---|---|---|---|---|---|---|---|---|")
    for m in presentes:
        for alfa in a.alfas:
            cs = [c["alerta"][f"{alfa:g}"] for c in corridas[m]]
            lineas.append(f"| {m.upper() if len(m) <= 4 else m} | {alfa:g}{' (final)' if alfa == a.alfas[0] else ''} |"
                          f" {media_desv([c['umbral'] for c in cs])} |"
                          f" {media_desv([c['fpr_sin_fuego'] for c in cs])} | {media_desv([c['fpr_controlado'] for c in cs])} |"
                          f" {media_desv([c['deteccion_propagacion'] for c in cs])} |"
                          f" {media_desv([c['adelanto_mediana'] for c in cs], '.1f')} |"
                          f" {media_desv([c['con_alerta_antes'] for c in cs], '.1f')} de {len(con_cambio)} |"
                          f" {media_desv([c['sin_alerta'] for c in cs], '.1f')} |")
    lineas.append("")
    lineas.append("| corrida | umbral > controlado | umbral > incipiente | F1 en calibracion | segmentos por video (final) |")
    lineas.append("|---|---|---|---|---|")
    for m in presentes:
        for c in corridas[m]:
            u = c.get("umbrales")
            lineas.append(f"| {c['corrida']} | {'-' if u is None else format(u[0], '.2f')} | {'-' if u is None else format(u[1], '.2f')} |"
                          f" {'-' if u is None else format(c['f1_calibracion'], '.3f')} | {c['segmentos_por_video']:.1f} |")
    lineas.append("")
    with open(os.path.join(a.salida, "resumen.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
