import os
import csv
import json
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from entrenar_temporal import verdad_escena, confusion, f1_macro
from secuencias import leer_csv

NIVELES = ["controlado", "incipiente", "incontrolado"]
DESC = [f"d{i}" for i in range(1, 14)]
CAUSAS = [("corte", "dentro de los 3 s posteriores a un corte de escena"),
          ("cortas", "solo trayectorias de fuego de menos de 1 s"),
          ("humo", "solo detecciones de humo"),
          ("nada", "sin ninguna deteccion"),
          ("otra", "otra: deteccion de fuego que no forma trayectoria")]


def kappa_lineal(m):
    m = np.asarray(m, float)
    n = m.sum()
    k = m.shape[0]
    i, j = np.indices((k, k))
    w = np.abs(i - j) / (k - 1)
    esperado = np.outer(m.sum(1), m.sum(0)) / n
    return float(1.0 - (w * m).sum() / (w * esperado).sum())


def metricas(m):
    m = np.asarray(m)
    f1, por_clase = f1_macro(m)
    return {"f1": f1, "f1_clase": por_clase, "exactitud": float(np.trace(m) / m.sum()), "kappa": kappa_lineal(m), "conf": m.tolist()}


def tabla_conf(nombre, m, lineas):
    lineas.append(f"Matriz de confusion por segundo de escena, {nombre} (filas: etiqueta de la IA; columnas: referencia):")
    lineas.append("")
    lineas.append("| real / estimado | " + " | ".join(NIVELES) + " |")
    lineas.append("|---|---|---|---|")
    for i, n in enumerate(NIVELES):
        lineas.append(f"| {n} | " + " | ".join(str(int(x)) for x in m[i]) + " |")
    lineas.append("")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--descriptores", default=os.path.join("runs_fuego", "descriptores_v2_noche", "descriptores.csv"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--cortes", default=os.path.join("resultados", "cortes", "cortes.csv"))
    p.add_argument("--curvas", default="curvas")
    p.add_argument("--modelos", default=os.path.join("resultados", "modelos"))
    p.add_argument("--min_muestras", type=int, default=10)
    p.add_argument("--tras_corte", type=float, default=3.0)
    p.add_argument("--hueco", type=int, default=3)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", default=os.path.join("resultados", "diagnostico"))
    a = p.parse_args()

    particion = {r["archivo"]: r for r in leer_csv(a.particion) if r["conjunto"] in ("entrenamiento", "validacion")}
    metas = {m["archivo"]: m for m in leer_csv(a.metas) if m["fuente"] in ("ia", "dataset") and m["archivo"] in particion}
    conj = {v: r["conjunto"] for v, r in particion.items()}
    verdad = {c: verdad_escena(metas, sorted(v for v in particion if conj[v] == c)) for c in ("entrenamiento", "validacion")}
    cortes = defaultdict(list)
    for r in leer_csv(a.cortes):
        cortes[r["archivo"]].append(float(r["t_corte"]))

    print("leyendo descriptores...", flush=True)
    df = pd.read_csv(a.descriptores, usecols=["archivo", "id", "t", "tramo"] + DESC, dtype={"archivo": str},
                     keep_default_na=False, na_values=[])
    df = df[df["archivo"].isin(set(particion))].copy()
    df["seq"] = df["archivo"] + "\t" + df["tramo"].astype(str) + "\t" + df["id"].astype(str)
    df["larga"] = df.groupby("seq")["t"].transform("size") >= a.min_muestras
    df["seg"] = np.floor(df["t"]).astype(int)
    con_larga, con_corta = defaultdict(set), defaultdict(set)
    for v, s, larga in zip(df["archivo"], df["seg"], df["larga"]):
        (con_larga if larga else con_corta)[v].add(int(s))

    lineas = ["# Diagnostico antes de la fase E", ""]
    lineas.append("Generado por `python diagnostico_escena.py`. No cambia el modelo, no reentrena los modelos temporales y no usa"
                  " la prueba.")
    lineas.append("")

    cuenta = {c: np.zeros(3, int) for c, _ in CAUSAS}
    en_hueco = {c: np.zeros(3, int) for c, _ in CAUSAS}
    tambien_cortas = np.zeros(3, int)
    total_sin = 0
    for v, seg in verdad["validacion"].items():
        ruta = os.path.join(a.curvas, os.path.splitext(v)[0] + ".csv")
        curva = defaultdict(lambda: [0, 0, 0])
        if os.path.exists(ruta):
            for r in leer_csv(ruta):
                c = curva[int(float(r["t"]))]
                c[0] += 1
                c[1] += int(float(r["n_fuego"]) > 0)
                c[2] += int(float(r["n_humo"]) > 0)
        ultimo = int(np.ceil(float(metas[v]["duracion_s"])))
        tiene = [b in con_larga[v] for b in range(ultimo + 1)]
        hueco = set()
        b = 0
        while b <= ultimo:
            if tiene[b]:
                b += 1
                continue
            fin = b
            while fin <= ultimo and not tiene[fin]:
                fin += 1
            if b > 0 and fin <= ultimo and fin - b <= a.hueco:
                hueco.update(range(b, fin))
            b = fin
        for s, nivel in seg.items():
            if s in con_larga[v]:
                continue
            total_sin += 1
            centro = s + 0.5
            if any(tc <= centro <= tc + a.tras_corte for tc in cortes[v]):
                causa = "corte"
                tambien_cortas[nivel] += int(s in con_corta[v])
            elif s in con_corta[v]:
                causa = "cortas"
            elif s not in curva:
                causa = "otra"
            elif curva[s][1] > 0:
                causa = "otra"
            elif curva[s][2] > 0:
                causa = "humo"
            else:
                causa = "nada"
            cuenta[causa][nivel] += 1
            en_hueco[causa][nivel] += int(s in hueco)
    lineas.append("## 1. Segundos de validacion sin ninguna trayectoria de 1 s o mas")
    lineas.append("")
    lineas.append(f"- Total: {total_sin} s de {sum(len(s) for s in verdad['validacion'].values())} s de escena.")
    lineas.append("- Cada segundo va a una sola causa, en este orden de prioridad: corte, trayectorias cortas, humo, sin deteccion, otra.")
    lineas.append("- Las detecciones de humo y de fuego sin trayectoria salen de `curvas/` (YOLO a 5 fps con confianza 0.25,"
                  " de `preanotar.py`); el seguimiento usa confianza 0.1, asi que no son exactamente las mismas detecciones.")
    lineas.append(f"- Hueco: racha de {a.hueco} s o menos sin trayectoria, con trayectoria en el segundo anterior y en el siguiente.")
    lineas.append("")
    lineas.append("| causa | controlado (s) | incipiente (s) | incontrolado (s) | total (s) | en huecos de 3 s o menos |")
    lineas.append("|---|---|---|---|---|---|")
    for c, nombre in CAUSAS:
        lineas.append(f"| {nombre} | " + " | ".join(str(x) for x in cuenta[c]) + f" | {cuenta[c].sum()} | {en_hueco[c].sum()} |")
    tot = sum(cuenta.values())
    toth = sum(en_hueco.values())
    lineas.append(f"| total | " + " | ".join(str(x) for x in tot) + f" | {tot.sum()} | {toth.sum()} |")
    lineas.append("")
    lineas.append("En huecos de 3 s o menos, por nivel real: " + ", ".join(f"{n} {x}" for n, x in zip(NIVELES, toth)) + ".")
    lineas.append(f"De los {cuenta['corte'].sum()} s posteriores a un corte, {tambien_cortas.sum()} tienen ademas trayectorias cortas.")
    lineas.append("")

    largas = df[df["larga"]]
    g = largas.groupby(["archivo", "seg"])
    agregados = pd.concat([g[DESC].max().add_suffix("_max"), g[DESC].mean().add_suffix("_media"),
                           g["seq"].nunique().rename("n_tray")], axis=1)
    columnas = list(agregados.columns)
    indice = {clave: i for i, clave in enumerate(agregados.index)}
    valores = agregados.to_numpy(np.float64)

    def filas(conjunto, con_sin_fuego):
        idx, y, sin = [], [], []
        for v, seg in verdad[conjunto].items():
            for s, nivel in seg.items():
                i = indice.get((v, s))
                if i is None:
                    sin.append(nivel)
                else:
                    idx.append(i)
                    y.append(nivel)
        if con_sin_fuego:
            for (v, s), i in indice.items():
                if conj[v] == conjunto and particion[v]["clase"] == "sin_fuego":
                    idx.append(i)
                    y.append(0)
        return valores[idx], np.array(y, int), np.array(sin, int)

    x_ent, y_ent, _ = filas("entrenamiento", True)
    x_val, y_val, sin_val = filas("validacion", False)
    referencias = {
        "regresion logistica": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000,
                                                                                  random_state=a.semilla)),
        "gradient boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=a.semilla),
    }
    resultados = {}
    for nombre, modelo in referencias.items():
        modelo.fit(x_ent, y_ent)
        pred = modelo.predict(x_val)
        m = confusion(np.concatenate([y_val, sin_val]), np.concatenate([pred, np.zeros(len(sin_val), int)]))
        resultados[nombre] = metricas(m)
        resultados[nombre]["solo_con_trayectoria"] = metricas(confusion(y_val, pred))
    for t in ("cfc", "gru", "lstm"):
        ruta = os.path.join(a.modelos, t, "metricas.json")
        if os.path.exists(ruta):
            with open(ruta, encoding="utf-8") as f:
                resultados[t.upper()] = metricas(np.array(json.load(f)["validacion"]["conf_escena"]))
    mayoria = np.zeros((3, 3), int)
    real = np.concatenate([y_val, sin_val])
    np.add.at(mayoria, (real, np.full(len(real), np.bincount(y_ent, minlength=3).argmax())), 1)
    resultados["clase mayoritaria de entrenamiento"] = metricas(mayoria)

    lineas.append("## 2 y 3. Referencias sin modelo temporal y acuerdo con la IA")
    lineas.append("")
    lineas.append(f"- Vector por segundo de escena: maximo y media de cada uno de los 13 descriptores sobre las muestras de las"
                  f" trayectorias de 1 s o mas vivas en ese segundo, y numero de trayectorias ({len(columnas)} valores)."
                  f" Descriptores sin normalizar; la regresion logistica los estandariza.")
    lineas.append(f"- Entrenamiento: {len(y_ent)} s con trayectoria ("
                  + ", ".join(f"{n} {int((y_ent == k).sum())}" for k, n in enumerate(NIVELES))
                  + "), incluidos los segundos con trayectoria de los videos sin fuego como controlado. Pesos por clase"
                  " balanceados. Cada segundo pesa igual (no hay muestreo por video).")
    lineas.append(f"- Validacion: los mismos {len(real)} s de escena de los modelos temporales; los {len(sin_val)} s sin"
                  f" trayectoria se cuentan como controlado, igual que en ellos.")
    lineas.append("- Kappa ponderado lineal por segundo contra las etiquetas de la IA. Referencia: humano contra IA 0.426"
                  " (otros videos: los 29 de la prueba de cambios; no es el mismo conjunto).")
    lineas.append("")
    lineas.append("| estimador | F1 macro escena | exactitud escena | kappa lineal | F1 controlado | F1 incipiente | F1 incontrolado |")
    lineas.append("|---|---|---|---|---|---|---|")
    for nombre, r in resultados.items():
        lineas.append(f"| {nombre} | {r['f1']:.3f} | {r['exactitud']:.3f} | {r['kappa']:.3f} | "
                      + " | ".join(f"{x:.3f}" for x in r["f1_clase"]) + " |")
    lineas.append("")
    lineas.append(f"Solo en los {len(y_val)} s de validacion con trayectoria (sin la regla de controlado por defecto):")
    lineas.append("")
    lineas.append("| estimador | F1 macro | exactitud | kappa lineal |")
    lineas.append("|---|---|---|---|")
    for nombre in referencias:
        r = resultados[nombre]["solo_con_trayectoria"]
        lineas.append(f"| {nombre} | {r['f1']:.3f} | {r['exactitud']:.3f} | {r['kappa']:.3f} |")
    lineas.append("")
    for nombre in referencias:
        tabla_conf(nombre, np.array(resultados[nombre]["conf"]), lineas)

    lineas.append("## 4. Trayectorias vivas por segundo en entrenamiento")
    lineas.append("")
    por_tray = largas[largas["archivo"].map(conj) == "entrenamiento"].groupby(["archivo", "seg", "seq"])["d2"].mean().reset_index()
    por_tray["maximo"] = por_tray.groupby(["archivo", "seg"])["d2"].transform("max")
    por_tray["n"] = por_tray.groupby(["archivo", "seg"])["d2"].transform("size")
    por_tray["nivel"] = [verdad["entrenamiento"].get(v, {}).get(s, -1) for v, s in zip(por_tray["archivo"], por_tray["seg"])]
    por_tray = por_tray[por_tray["nivel"] >= 0]
    por_tray["pequena"] = por_tray["d2"] < por_tray["maximo"] / 4.0
    lineas.append("- Segundos de escena de los videos de fuego de entrenamiento, dentro del tramo util, con al menos una"
                  " trayectoria de 1 s o mas. El area de una trayectoria en un segundo es la media de d2 en ese segundo.")
    lineas.append("")
    lineas.append("| nivel | segundos | mediana de trayectorias vivas | percentil 90 | segundos con mas de una trayectoria |"
                  " trayectorias-segundo | con area menor a un cuarto de la mayor |")
    lineas.append("|---|---|---|---|---|---|---|")
    ruido = {}
    for k, n in enumerate(NIVELES):
        sub = por_tray[por_tray["nivel"] == k]
        por_seg = sub.groupby(["archivo", "seg"])["n"].first().to_numpy()
        prop = float(sub["pequena"].mean()) if len(sub) else float("nan")
        ruido[n] = {"segundos": int(len(por_seg)), "mediana": float(np.median(por_seg)), "p90": float(np.percentile(por_seg, 90)),
                    "varias": float((por_seg > 1).mean()), "tray_seg": int(len(sub)), "pequenas": prop}
        lineas.append(f"| {n} | {len(por_seg)} | {np.median(por_seg):.0f} | {np.percentile(por_seg, 90):.0f} |"
                      f" {100.0 * (por_seg > 1).mean():.1f} % | {len(sub)} | {100.0 * prop:.1f} % |")
    lineas.append("")

    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "diagnostico.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    with open(os.path.join(a.salida, "diagnostico.json"), "w", encoding="utf-8") as f:
        json.dump({"sin_trayectoria": {c: cuenta[c].tolist() for c, _ in CAUSAS},
                   "en_hueco": {c: en_hueco[c].tolist() for c, _ in CAUSAS}, "estimadores": resultados, "herencia": ruido}, f, indent=1)
    print("\n".join(lineas))
    print(f"salida en: {a.salida}")


if __name__ == "__main__":
    main()
