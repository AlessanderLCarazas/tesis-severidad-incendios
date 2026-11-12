import os
import csv
import glob
import json
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from entrenar_temporal import verdad_escena, confusion
from entrenar_escena import resumen_conf
from secuencias import leer_csv

NIVELES = ["controlado", "incipiente", "incontrolado"]
COLORES = {"cfc": "#c0392b", "gru": "#2471a3", "lstm": "#1e8449"}


def por_segundo(ruta):
    d = np.load(ruta)
    x = d["x"].astype(np.float64)
    con_larga = d["con_larga"]
    largos = d["seq_inicio"][1:] - d["seq_inicio"][:-1]
    archivo = np.repeat(d["seq_archivo"], largos)
    seg = np.floor(d["t"]).astype(np.int64)
    suma, cuenta, larga = {}, defaultdict(int), defaultdict(bool)
    for i, clave in enumerate(zip(archivo.tolist(), seg.tolist())):
        if clave in suma:
            suma[clave] += x[i]
        else:
            suma[clave] = x[i].copy()
        cuenta[clave] += 1
        if con_larga[i]:
            larga[clave] = True
    return {k: v / cuenta[k] for k, v in suma.items()}, larga


def media_desv(valores):
    v = np.array(valores, float)
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} +- {v.std():.3f}"


def fila_tabla(nombre, corridas):
    partes = [nombre, str(len(corridas))]
    for clave in ("todo", "con_trayectoria", "sin_trayectoria"):
        partes.append(media_desv([c[clave]["f1"] for c in corridas]))
    for k in range(3):
        partes.append(media_desv([c["todo"]["f1_clase"][k] for c in corridas]))
    partes.append(media_desv([c["todo"]["exactitud"] for c in corridas]))
    partes.append(media_desv([c["todo"]["kappa"] for c in corridas]))
    mae = [c["mae_tau_s"] for c in corridas if c.get("mae_tau_s") is not None]
    partes.append("-" if not mae else (f"{np.mean(mae):.1f}" if len(mae) == 1 else f"{np.mean(mae):.1f} +- {np.std(mae):.1f}"))
    partes.append(f"{np.mean([c['cambios_modelo'] for c in corridas]):.1f}")
    return "| " + " | ".join(partes) + " |"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--modelos", default=os.path.join("resultados", "modelos_escena"))
    p.add_argument("--anteriores", default=os.path.join("resultados", "modelos"))
    p.add_argument("--semilla", type=int, default=42)
    a = p.parse_args()

    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    metas = defaultdict(dict)
    for m in leer_csv(a.metas):
        if m["fuente"] in ("ia", "dataset"):
            metas[m["conjunto"]][m["archivo"]] = m
    x_ent_seg, _ = por_segundo(os.path.join(a.datos, "entrenamiento.npz"))
    x_val_seg, larga_val = por_segundo(os.path.join(a.datos, "validacion.npz"))
    verdad_ent = verdad_escena(metas["entrenamiento"], sorted(metas["entrenamiento"]))
    verdad_val = verdad_escena(metas["validacion"], sorted(metas["validacion"]))
    x_ent, y_ent = [], []
    for (v, s), x in x_ent_seg.items():
        if particion[v]["clase"] == "sin_fuego":
            x_ent.append(x)
            y_ent.append(0)
        elif s in verdad_ent.get(v, {}):
            x_ent.append(x)
            y_ent.append(verdad_ent[v][s])
    x_ent, y_ent = np.array(x_ent), np.array(y_ent, int)
    claves, y_val = [], []
    for v, seg in verdad_val.items():
        for s in sorted(seg):
            claves.append((v, s))
            y_val.append(seg[s])
    y_val = np.array(y_val, np.int64)
    dim = x_ent.shape[1]
    x_val = np.array([x_val_seg.get(k, np.zeros(dim)) for k in claves])
    con = np.array([larga_val[k] for k in claves], bool)
    cambios_real = float(np.mean([int((np.diff([seg[s] for s in sorted(seg)]) != 0).sum()) for seg in verdad_val.values()]))

    def evaluar(pred):
        cambios, pos = [], 0
        for v, seg in verdad_val.items():
            e = pred[pos:pos + len(seg)]
            pos += len(seg)
            cambios.append(int((np.diff(e) != 0).sum()) if len(e) > 1 else 0)
        return {"todo": resumen_conf(confusion(y_val, pred)), "con_trayectoria": resumen_conf(confusion(y_val[con], pred[con])),
                "sin_trayectoria": resumen_conf(confusion(y_val[~con], pred[~con])), "mae_tau_s": None,
                "cambios_modelo": float(np.mean(cambios))}

    referencias = {
        "regresion logistica (sin tiempo)": LogisticRegression(class_weight="balanced", max_iter=3000, random_state=a.semilla),
        "gradient boosting (sin tiempo)": HistGradientBoostingClassifier(class_weight="balanced", random_state=a.semilla),
    }
    grupos = {}
    for nombre, modelo in referencias.items():
        modelo.fit(x_ent, y_ent)
        grupos[nombre] = [evaluar(modelo.predict(x_val).astype(np.int64))]
    mayoria = int(np.bincount(y_ent, minlength=3).argmax())
    grupos["clase mayoritaria"] = [evaluar(np.full(len(y_val), mayoria, np.int64))]

    corridas = defaultdict(list)
    historiales = []
    for ruta in sorted(glob.glob(os.path.join(a.modelos, "*", "metricas.json"))):
        with open(ruta, encoding="utf-8") as f:
            r = json.load(f)
        corridas[r["modelo"]].append(r)
        with open(os.path.join(os.path.dirname(ruta), "historial.csv"), encoding="utf-8") as f:
            historiales.append((r["modelo"], r["semilla"], r["mejor_epoca"],
                                [{k: float(v) for k, v in h.items()} for h in csv.DictReader(f)]))

    lineas = ["# Modelos a nivel de escena en validacion", ""]
    lineas.append("Generado por `python comparar_escena.py`. Mejor epoca de cada corrida en validacion; la prueba no se usa."
                  f" Segundos de escena: {len(y_val)} ({int(con.sum())} con alguna trayectoria de 1 s o mas y {int((~con).sum())} sin)."
                  " Con varias semillas se da media +- desviacion.")
    lineas.append("")
    lineas.append("| estimador | corridas | F1 macro | F1 con trayectoria | F1 sin trayectoria | F1 controlado | F1 incipiente |"
                  " F1 incontrolado | exactitud | kappa lineal | MAE tau (s) | cambios de nivel por video |")
    lineas.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in ("cfc", "gru", "lstm"):
        if corridas[t]:
            lineas.append(fila_tabla(f"{t.upper()} de escena", [r["validacion"] for r in corridas[t]]))
    for nombre, c in grupos.items():
        lineas.append(fila_tabla(nombre, c))
    lineas.append("")
    lineas.append(f"Cambios de nivel por video segun la etiqueta: {cambios_real:.2f} ({len(verdad_val)} videos de fuego).")
    lineas.append("")
    lineas.append("| estimador | exactitud con trayectoria | kappa con trayectoria | exactitud sin trayectoria | kappa sin trayectoria |")
    lineas.append("|---|---|---|---|---|")
    todos = [(f"{t.upper()} de escena", [r["validacion"] for r in corridas[t]]) for t in ("cfc", "gru", "lstm") if corridas[t]]
    todos += list(grupos.items())
    for nombre, c in todos:
        lineas.append(f"| {nombre} | " + " | ".join(media_desv([x[parte][m] for x in c]) for parte in ("con_trayectoria", "sin_trayectoria")
                                                    for m in ("exactitud", "kappa")) + " |")
    lineas.append("")
    lineas.append("| modelo | semilla | parametros | s por epoca | epocas corridas | mejor epoca | minutos en total | F1 macro | kappa |")
    lineas.append("|---|---|---|---|---|---|---|---|---|")
    for t in ("cfc", "gru", "lstm"):
        for r in corridas[t]:
            lineas.append(f"| {t.upper()} | {r['semilla']} | {r['parametros']} | {r['s_por_epoca_ent']:.1f} | {r['epocas_corridas']} |"
                          f" {r['mejor_epoca']} | {r['s_total'] / 60:.1f} | {r['validacion']['todo']['f1']:.3f} |"
                          f" {r['validacion']['todo']['kappa']:.3f} |")
    lineas.append("")
    anteriores = []
    for t in ("cfc", "gru", "lstm"):
        ruta = os.path.join(a.anteriores, t, "metricas.json")
        if os.path.exists(ruta):
            with open(ruta, encoding="utf-8") as f:
                anteriores.append((t, resumen_conf(np.array(json.load(f)["validacion"]["conf_escena"]))))
    if anteriores:
        lineas.append("Primera version (modelo por trayectoria, semilla 42), mismos segundos:")
        lineas.append("")
        lineas.append("| modelo | F1 macro | exactitud | kappa lineal |")
        lineas.append("|---|---|---|---|")
        for t, r in anteriores:
            lineas.append(f"| {t.upper()} por trayectoria | {r['f1']:.3f} | {r['exactitud']:.3f} | {r['kappa']:.3f} |")
        lineas.append("")
    for t in ("cfc", "gru", "lstm"):
        if not corridas[t]:
            continue
        m = np.sum([np.array(r["validacion"]["todo"]["conf"]) for r in corridas[t]], axis=0)
        lineas.append(f"Matriz de confusion por segundo de escena, {t.upper()} (suma de {len(corridas[t])} corridas; filas: etiqueta"
                      " de la IA; columnas: modelo):")
        lineas.append("")
        lineas.append("| real / modelo | " + " | ".join(NIVELES) + " |")
        lineas.append("|---|---|---|---|")
        for i, n in enumerate(NIVELES):
            lineas.append(f"| {n} | " + " | ".join(str(int(x)) for x in m[i]) + " |")
        lineas.append("")
    for nombre in referencias:
        m = np.array(grupos[nombre][0]["todo"]["conf"])
        lineas.append(f"Matriz de confusion por segundo de escena, {nombre}:")
        lineas.append("")
        lineas.append("| real / estimado | " + " | ".join(NIVELES) + " |")
        lineas.append("|---|---|---|---|")
        for i, n in enumerate(NIVELES):
            lineas.append(f"| {n} | " + " | ".join(str(int(x)) for x in m[i]) + " |")
        lineas.append("")

    os.makedirs(a.modelos, exist_ok=True)
    if historiales:
        fig, ejes = plt.subplots(1, 3, figsize=(15, 4.2))
        vistos = set()
        for t, semilla, mejor, h in historiales:
            x = [f["epoca"] for f in h]
            etiqueta = t.upper() if t not in vistos else None
            vistos.add(t)
            ejes[0].plot(x, [f["perdida_ent"] for f in h], color=COLORES.get(t), label=etiqueta, linewidth=1)
            ejes[1].plot(x, [f["perdida_val"] for f in h], color=COLORES.get(t), label=etiqueta, linewidth=1)
            ejes[2].plot(x, [f["f1_escena"] for f in h], color=COLORES.get(t), label=etiqueta, linewidth=1)
            ejes[2].plot([mejor], [h[mejor - 1]["f1_escena"]], "o", color=COLORES.get(t), markersize=5)
        ref = grupos["gradient boosting (sin tiempo)"][0]["todo"]["f1"]
        ejes[2].axhline(ref, color="gray", linestyle="--", linewidth=1, label="gradient boosting sin tiempo")
        ejes[0].set_title("Perdida total en entrenamiento")
        ejes[1].set_title("Perdida total en validacion")
        ejes[2].set_title("F1 macro de escena por segundo en validacion\n(punto: epoca elegida)")
        for e in ejes:
            e.set_xlabel("epoca")
            e.grid(alpha=0.3)
            e.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(a.modelos, "curvas.png"), dpi=130)
    with open(os.path.join(a.modelos, "comparacion.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    with open(os.path.join(a.modelos, "referencias.json"), "w", encoding="utf-8") as f:
        json.dump({"referencias": grupos, "cambios_etiqueta": cambios_real, "segundos": int(len(y_val)),
                   "con_trayectoria": int(con.sum()), "entrenamiento_segundos": int(len(y_ent)),
                   "entrenamiento_por_nivel": np.bincount(y_ent, minlength=3).tolist()}, f, indent=1)
    print("\n".join(lineas))
    print(f"entrenamiento de las referencias: {len(y_ent)} s | por nivel {np.bincount(y_ent, minlength=3).tolist()}")


if __name__ == "__main__":
    main()
