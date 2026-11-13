import os
import csv
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from normalizacion import Normalizador
from secuencias import metas_muestras, segundos_escena, ventanas_de, leer_csv, RIESGO, REF_HUMANO
from vector_escena import COLUMNAS, CONTEXTO, DESC, DETECCIONES, CAMARA, vectores_video

CONJUNTOS = ["entrenamiento", "validacion", "calibracion", "prueba"]
CAMPOS_VIDEOS = ["conjunto", "archivo", "clase", "tipo", "fotogramas", "tramos", "secuencias", "sin_trayectoria",
                 "s_controlado", "s_incipiente", "s_incontrolado"]


def vectores_carpeta(carpeta, videos=None, edad_min=1.0, min_larga=10, sufijo=""):
    fot = pd.read_csv(os.path.join(carpeta, "fotogramas.csv"), dtype={"archivo": str}, keep_default_na=False, na_values=[])
    des = pd.read_csv(os.path.join(carpeta, "descriptores.csv"), dtype={"archivo": str}, keep_default_na=False, na_values=[])
    if videos is not None:
        fot, des = fot[fot["archivo"].isin(set(videos))], des[des["archivo"].isin(set(videos))]
    des = des.sort_values(["archivo", "k", "id"], kind="stable")
    seq = des["archivo"] + "\t" + des["tramo"].astype(str) + "\t" + des["id"].astype(str)
    des = des.assign(larga=(des.groupby(seq)["t"].transform("size") >= min_larga).to_numpy())
    filas = defaultdict(lambda: defaultdict(list))
    largas = defaultdict(set)
    for v, k, i, larga, d in zip(des["archivo"], des["k"], des["id"], des["larga"], des[DESC].to_numpy(np.float64)):
        filas[v][int(k)].append((int(i), d))
        if larga:
            largas[v].add(int(k))
    salida = {}
    if sufijo:
        fot = fot.drop(columns=DETECCIONES).rename(columns={c + sufijo: c for c in DETECCIONES})
    columnas = ["k", "t", "dt", "tramo"] + CAMARA + DETECCIONES
    for v, sub in fot.groupby("archivo", sort=False):
        sub = sub.sort_values("k")
        fotogramas = [dict(zip(columnas, fila)) for fila in zip(*[sub[c].tolist() for c in columnas])]
        x, c = vectores_video(fotogramas, filas[v], edad_min)
        salida[v] = {"k": sub["k"].to_numpy(np.int64), "t": sub["t"].to_numpy(np.float64), "dt": sub["dt"].to_numpy(np.float64),
                     "tramo": sub["tramo"].to_numpy(np.int64), "x": x, "c": c,
                     "con_larga": np.array([int(k) in largas[v] for k in sub["k"]], bool)}
    return salida


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", default=os.path.join("runs_fuego", "descriptores_v3"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--fuente", default="ia")
    p.add_argument("--detecciones", choices=["alta", "seguimiento"], default="alta")
    p.add_argument("--edad_min", type=float, default=1.0)
    p.add_argument("--min_larga", type=int, default=10)
    p.add_argument("--min_fotogramas", type=int, default=10)
    p.add_argument("--ventana", type=int, default=300)
    p.add_argument("--paso_ventana", type=int, default=150)
    p.add_argument("--salida", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--resumen", default=os.path.join("resultados", "escena_v2"))
    a = p.parse_args()

    if not os.path.exists(os.path.join(a.entrada, "procesados.csv")):
        raise SystemExit(f"no hay corrida de descriptores_v3.py en {a.entrada}")
    particion = {r["archivo"]: r for r in leer_csv(a.particion) if r["conjunto"] in CONJUNTOS}
    principal, humano = {}, {}
    for m in leer_csv(a.metas):
        if m["fuente"] in (a.fuente, "dataset"):
            principal[m["archivo"]] = m
        if m["fuente"] == "humano":
            humano[m["archivo"]] = m
    inicios = defaultdict(lambda: [0.0])
    for r in leer_csv(os.path.join(a.entrada, "cortes.csv")):
        inicios[r["archivo"]].append(float(r["t_corte"]))
    print("calculando el vector de escena...", flush=True)
    vectores = vectores_carpeta(a.entrada, set(particion), a.edad_min, a.min_larga, "_alta" if a.detecciones == "alta" else "")
    faltan = sorted(v for v in particion if v not in vectores)
    if faltan:
        raise SystemExit(f"faltan {len(faltan)} videos de la particion en {a.entrada}; termina la corrida de descriptores_v3.py")

    crudo = {c: defaultdict(list) for c in CONJUNTOS}
    filas_video, cortas = [], 0
    for v in sorted(particion):
        c = particion[v]["conjunto"]
        d = vectores[v]
        t, tramo = d["t"], d["tramo"]
        ini = sorted(inicios[v])
        ini[0] = min(ini[0], float(t[0])) if len(t) else ini[0]
        nivel, mascara, tau, tau_m = metas_muestras(principal[v], t, tramo, ini)
        ref = "ninguna"
        if c == "prueba":
            if principal[v]["nivel_inicial"] == "sin_fuego":
                mh, ref = principal[v], "clase"
            else:
                mh = humano.get(v)
                ref = "ninguna" if mh is None else "descartado" if mh["descartar"] == "1" else mh["valida"]
            nh, mk, th, tm = metas_muestras(mh, t, tramo, ini)
        n_seq = 0
        for j in np.unique(tramo):
            sel = np.flatnonzero(tramo == j)
            if len(sel) < a.min_fotogramas:
                cortas += 1
                continue
            e = crudo[c]
            e["x"].append(d["x"][sel])
            e["c"].append(d["c"][sel])
            e["dt"].append(d["dt"][sel].astype(np.float32))
            e["t"].append(t[sel].astype(np.float32))
            e["nivel"].append(nivel[sel])
            e["riesgo"].append(np.array(RIESGO, np.float32)[nivel[sel]])
            e["mascara"].append(mascara[sel])
            e["tau"].append(tau[sel])
            e["tau_mascara"].append(tau_m[sel])
            e["con_larga"].append(d["con_larga"][sel])
            e["sin_tray"].append(d["x"][sel, 33] > 0)
            if c == "prueba":
                e["nivel_humano"].append(nh[sel])
                e["riesgo_humano"].append(np.array(RIESGO, np.float32)[nh[sel]])
                e["mascara_humano"].append(mk[sel])
                e["tau_humano"].append(th[sel])
                e["tau_mascara_humano"].append(tm[sel])
            e["largo"].append(len(sel))
            e["archivo"].append(v)
            e["tramo"].append(int(j))
            e["ref"].append(REF_HUMANO[ref])
            n_seq += 1
        seg = segundos_escena(principal[v])
        filas_video.append({"conjunto": c, "archivo": v, "clase": particion[v]["clase"], "tipo": particion[v]["tipo"],
                            "fotogramas": len(t), "tramos": len(np.unique(tramo)), "secuencias": n_seq,
                            "sin_trayectoria": int((d["x"][:, 33] > 0).sum()), "s_controlado": round(seg[0], 2),
                            "s_incipiente": round(seg[1], 2), "s_incontrolado": round(seg[2], 2)})

    ent = crudo["entrenamiento"]
    videos_ent = sorted(set(ent["archivo"]))
    norm_x = Normalizador(COLUMNAS).ajustar(np.concatenate(ent["x"]), videos_ent)
    norm_c = Normalizador(CONTEXTO).ajustar(np.concatenate(ent["c"]), videos_ent)
    os.makedirs(a.salida, exist_ok=True)
    os.makedirs(a.resumen, exist_ok=True)
    norm_x.guardar(os.path.join(a.resumen, "normalizacion.json"))
    norm_c.guardar(os.path.join(a.resumen, "normalizacion_contexto.json"))
    lineas = ["# Vector de escena por fotograma, version 2 (datos del propio pipeline)", ""]
    lineas.append(f"- Generado por `python escena_v2.py` desde `{a.entrada.replace(os.sep, '/')}`. El vector lo calcula"
                  " `vector_escena.VectorEscena`, la misma clase que usa el sistema en vivo.")
    lineas.append("- Detecciones del fotograma: las del propio pipeline, "
                  + ("filtradas a confianza 0.25." if a.detecciones == "alta" else "a la confianza del seguimiento (0.1)."))
    lineas.append(f"- Normalizacion ajustada solo con entrenamiento: {norm_x.n_filas} fotogramas de {len(norm_x.videos)} videos.")
    lineas.append(f"- Tramos de menos de {a.min_fotogramas} fotogramas descartados: {cortas}.")
    lineas.append("")
    lineas.append("| conjunto | videos | secuencias | fotogramas | dentro del tramo util | controlado | incipiente | incontrolado |"
                  " sin trayectoria | con tau por cambio | ventanas |")
    lineas.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CONJUNTOS:
        e = crudo[c]
        largos = np.array(e["largo"], np.int64)
        datos = {"x": norm_x.transformar(np.concatenate(e["x"])).astype(np.float32),
                 "c": norm_c.transformar(np.concatenate(e["c"])).astype(np.float32)}
        claves = ["dt", "t", "nivel", "riesgo", "mascara", "tau", "tau_mascara", "con_larga", "sin_tray"]
        if c == "prueba":
            claves += ["nivel_humano", "riesgo_humano", "mascara_humano", "tau_humano", "tau_mascara_humano"]
        for k in claves:
            datos[k] = np.concatenate(e[k])
        datos["seq_inicio"] = np.concatenate([[0], np.cumsum(largos)]).astype(np.int64)
        datos["seq_archivo"] = np.array(e["archivo"], dtype=str)
        datos["seq_tramo"] = np.array(e["tramo"], np.int32)
        datos["seq_ref_humano"] = np.array(e["ref"], np.int8)
        vent = [(n, i, j) for n, largo in enumerate(largos) for i, j in ventanas_de(int(largo), a.ventana, a.paso_ventana)]
        datos["ventanas"] = np.array(vent, np.int64).reshape(-1, 3)
        datos["columnas_x"] = np.array(COLUMNAS, dtype=str)
        datos["columnas_c"] = np.array(CONTEXTO, dtype=str)
        np.savez_compressed(os.path.join(a.salida, f"{c}.npz"), **datos)
        m = datos["mascara"]
        lineas.append(f"| {c} | {len(set(e['archivo']))} | {len(largos)} | {int(largos.sum())} | {int(m.sum())} | "
                      + " | ".join(str(int((m & (datos['nivel'] == k)).sum())) for k in range(3))
                      + f" | {int(datos['sin_tray'].sum())} | {int((datos['tau_mascara'] == 1).sum())} | {len(vent)} |")
        print(f"{c}: {len(set(e['archivo']))} videos | {len(largos)} secuencias | {int(largos.sum())} fotogramas | {len(vent)} ventanas",
              flush=True)
    lineas.append("")
    with open(os.path.join(a.resumen, "videos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_VIDEOS)
        w.writeheader()
        w.writerows(filas_video)
    with open(os.path.join(a.resumen, "resumen.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    print(f"datos en: {a.salida} | resumen en: {a.resumen}")


if __name__ == "__main__":
    main()
