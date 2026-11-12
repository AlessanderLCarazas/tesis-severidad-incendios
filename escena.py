import os
import csv
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from normalizacion import Normalizador
from secuencias import metas_muestras, segundos_escena, ventanas_de, t_peligro, leer_csv, RIESGO, REF_HUMANO

DESC = [f"d{i}" for i in range(1, 14)]
CAMARA = ["s_k", "rot", "tx", "ty", "inliers"]
CONTEXTO = CAMARA + ["dt"]
COLUMNAS = ([f"{d}_max" for d in DESC] + [f"{d}_media" for d in DESC]
            + ["n_tray", "n_tray_1s", "edad_max", "n_det_fuego", "n_det_humo", "area_det_fuego", "area_det_humo", "sin_trayectoria"])
CONJUNTOS = ["entrenamiento", "validacion", "calibracion", "prueba"]
CAMPOS_VIDEOS = ["conjunto", "archivo", "clase", "tipo", "fotogramas", "tramos", "secuencias", "sin_trayectoria", "con_curva",
                 "s_controlado", "s_incipiente", "s_incontrolado"]


def leer_curva(carpetas, nombre):
    base = os.path.splitext(nombre)[0] + ".csv"
    for carpeta in carpetas:
        ruta = os.path.join(carpeta, base)
        if os.path.exists(ruta):
            filas = leer_csv(ruta)
            if filas:
                return np.array([[float(r[k]) for k in ("t", "n_fuego", "n_humo", "fuego", "humo")] for r in filas], np.float64)
    return None


def tiempos(n, k_con, t_con, paso):
    k = np.arange(n, dtype=np.float64)
    if len(k_con) == 0:
        return k * paso
    t = np.interp(k, k_con, t_con)
    antes, despues = k < k_con[0], k > k_con[-1]
    t[antes] = t_con[0] - paso * (k_con[0] - k[antes])
    t[despues] = t_con[-1] + paso * (k[despues] - k_con[-1])
    return np.maximum(t, 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--descriptores", default=os.path.join("runs_fuego", "descriptores_v2_noche", "descriptores.csv"))
    p.add_argument("--procesados", default=os.path.join("runs_fuego", "descriptores_v2_noche", "procesados.csv"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--cortes", default=os.path.join("resultados", "cortes", "cortes.csv"))
    p.add_argument("--curvas", nargs="+", default=["curvas", os.path.join("runs_fuego", "curvas_sin_fuego")])
    p.add_argument("--fuente", default="ia")
    p.add_argument("--paso", type=float, default=0.1)
    p.add_argument("--edad_min", type=float, default=1.0)
    p.add_argument("--min_larga", type=int, default=10)
    p.add_argument("--min_fotogramas", type=int, default=10)
    p.add_argument("--ventana", type=int, default=300)
    p.add_argument("--paso_ventana", type=int, default=150)
    p.add_argument("--salida", default=os.path.join("runs_fuego", "escena"))
    p.add_argument("--resumen", default=os.path.join("resultados", "escena"))
    a = p.parse_args()

    particion = {r["archivo"]: r for r in leer_csv(a.particion) if r["conjunto"] in CONJUNTOS}
    principal, humano = {}, {}
    for m in leer_csv(a.metas):
        if m["fuente"] in (a.fuente, "dataset"):
            principal[m["archivo"]] = m
        if m["fuente"] == "humano":
            humano[m["archivo"]] = m
    muestras = {r["archivo"]: int(r["muestras"]) for r in leer_csv(a.procesados)}
    inicios = defaultdict(lambda: [0.0])
    for r in leer_csv(a.cortes):
        inicios[r["archivo"]].append(float(r["t_corte"]))
    inicios = defaultdict(lambda: [0.0], {v: sorted(x) for v, x in inicios.items()})

    print("leyendo descriptores...", flush=True)
    df = pd.read_csv(a.descriptores, dtype={"archivo": str}, keep_default_na=False, na_values=[])
    df = df[df["archivo"].isin(set(particion))].copy()
    seq = df["archivo"] + "\t" + df["tramo"].astype(str) + "\t" + df["id"].astype(str)
    df["edad"] = df["t"] - df.groupby(seq)["t"].transform("min")
    df["madura"] = (df["edad"] >= a.edad_min - 1e-6).astype(float)
    df["larga"] = (df.groupby(seq)["t"].transform("size") >= a.min_larga).astype(float)
    g = df.groupby(["archivo", "k"], sort=True)
    tabla = pd.concat([g[DESC].max().add_suffix("_max"), g[DESC].mean().add_suffix("_media"),
                       g["t"].size().rename("n_tray"), g["madura"].sum().rename("n_tray_1s"),
                       g["edad"].max().rename("edad_max"), g["larga"].max().rename("larga"),
                       g[["t", "tramo"] + CAMARA].first()], axis=1).reset_index()
    por_video = {v: sub for v, sub in tabla.groupby("archivo", sort=False)}
    agregados = COLUMNAS[:29]

    crudo = {c: defaultdict(list) for c in CONJUNTOS}
    filas_video, sin_curva, cortas, tramo_distinto = [], 0, 0, 0
    for v in sorted(particion):
        c = particion[v]["conjunto"]
        n = muestras.get(v, 0)
        if n == 0:
            continue
        sub = por_video.get(v)
        k_con = sub["k"].to_numpy(np.int64) if sub is not None else np.zeros(0, np.int64)
        t = tiempos(n, k_con, sub["t"].to_numpy(np.float64) if sub is not None else np.zeros(0), a.paso)
        dt = np.diff(t, prepend=t[0] - a.paso)
        dt[0] = a.paso
        dt = np.maximum(dt, 1e-3)
        x = np.zeros((n, len(COLUMNAS)), np.float64)
        cam = np.zeros((n, 5), np.float64)
        cam[:, 0] = 1.0
        cam[:, 4] = np.nan
        larga = np.zeros(n, bool)
        ini = np.array(inicios[v])
        tramo = np.searchsorted(ini, t + 1e-6, side="right") - 1
        if sub is not None:
            x[k_con, :29] = sub[agregados].to_numpy(np.float64)
            cam[k_con] = sub[CAMARA].to_numpy(np.float64)
            larga[k_con] = sub["larga"].to_numpy() > 0
            tramo_distinto += int((tramo[k_con] != sub["tramo"].to_numpy()).sum())
            tramo[k_con] = sub["tramo"].to_numpy()
        x[:, 33] = 1.0
        x[k_con, 33] = 0.0
        inl = pd.Series(cam[:, 4]).ffill().fillna(1.0).to_numpy()
        cam[:, 4] = inl
        curva = leer_curva(a.curvas, v)
        if curva is None:
            sin_curva += 1
        else:
            pos = np.searchsorted(curva[:, 0], t + 0.05, side="right") - 1
            ok = pos >= 0
            x[ok, 29:33] = curva[pos[ok], 1:5]
        nivel, mascara, tau, tau_m = metas_muestras(principal[v], t, tramo, list(ini))
        ref, mh = "ninguna", None
        if c == "prueba":
            if principal[v]["nivel_inicial"] == "sin_fuego":
                mh, ref = principal[v], "clase"
            else:
                mh = humano.get(v)
                ref = "ninguna" if mh is None else "descartado" if mh["descartar"] == "1" else mh["valida"]
            nh, mk, th, tm = metas_muestras(mh, t, tramo, list(ini))
        n_seq = 0
        for j in np.unique(tramo):
            sel = np.flatnonzero(tramo == j)
            if len(sel) < a.min_fotogramas:
                cortas += 1
                continue
            d = crudo[c]
            d["x"].append(x[sel])
            d["c"].append(np.column_stack([cam[sel], dt[sel]]))
            d["dt"].append(dt[sel].astype(np.float32))
            d["t"].append(t[sel].astype(np.float32))
            d["nivel"].append(nivel[sel])
            d["riesgo"].append(np.array(RIESGO, np.float32)[nivel[sel]])
            d["mascara"].append(mascara[sel])
            d["tau"].append(tau[sel])
            d["tau_mascara"].append(tau_m[sel])
            d["con_larga"].append(larga[sel])
            d["sin_tray"].append(x[sel, 33] > 0)
            if c == "prueba":
                d["nivel_humano"].append(nh[sel])
                d["riesgo_humano"].append(np.array(RIESGO, np.float32)[nh[sel]])
                d["mascara_humano"].append(mk[sel])
                d["tau_humano"].append(th[sel])
                d["tau_mascara_humano"].append(tm[sel])
            d["largo"].append(len(sel))
            d["archivo"].append(v)
            d["tramo"].append(int(j))
            d["ref"].append(REF_HUMANO[ref])
            n_seq += 1
        seg = segundos_escena(principal[v])
        filas_video.append({"conjunto": c, "archivo": v, "clase": particion[v]["clase"], "tipo": particion[v]["tipo"],
                            "fotogramas": n, "tramos": len(np.unique(tramo)), "secuencias": n_seq,
                            "sin_trayectoria": int((x[:, 33] > 0).sum()), "con_curva": "si" if curva is not None else "no",
                            "s_controlado": round(seg[0], 2), "s_incipiente": round(seg[1], 2), "s_incontrolado": round(seg[2], 2)})

    ent = crudo["entrenamiento"]
    videos_ent = sorted(set(ent["archivo"]))
    norm_x = Normalizador(COLUMNAS).ajustar(np.concatenate(ent["x"]), videos_ent)
    norm_c = Normalizador(CONTEXTO).ajustar(np.concatenate(ent["c"]), videos_ent)
    os.makedirs(a.salida, exist_ok=True)
    os.makedirs(a.resumen, exist_ok=True)
    norm_x.guardar(os.path.join(a.resumen, "normalizacion.json"))
    norm_c.guardar(os.path.join(a.resumen, "normalizacion_contexto.json"))

    totales = {}
    for c in CONJUNTOS:
        d = crudo[c]
        largos = np.array(d["largo"], np.int64)
        datos = {"x": norm_x.transformar(np.concatenate(d["x"])).astype(np.float32),
                 "c": norm_c.transformar(np.concatenate(d["c"])).astype(np.float32)}
        for k in ("dt", "t", "nivel", "riesgo", "mascara", "tau", "tau_mascara", "con_larga", "sin_tray"):
            datos[k] = np.concatenate(d[k])
        if c == "prueba":
            for k in ("nivel_humano", "riesgo_humano", "mascara_humano", "tau_humano", "tau_mascara_humano"):
                datos[k] = np.concatenate(d[k])
        datos["seq_inicio"] = np.concatenate([[0], np.cumsum(largos)]).astype(np.int64)
        datos["seq_archivo"] = np.array(d["archivo"], dtype=str)
        datos["seq_tramo"] = np.array(d["tramo"], np.int32)
        datos["seq_ref_humano"] = np.array(d["ref"], np.int8)
        vent = [(n, i, j) for n, largo in enumerate(largos) for i, j in ventanas_de(int(largo), a.ventana, a.paso_ventana)]
        datos["ventanas"] = np.array(vent, np.int64).reshape(-1, 3)
        datos["columnas_x"] = np.array(COLUMNAS, dtype=str)
        datos["columnas_c"] = np.array(CONTEXTO, dtype=str)
        np.savez_compressed(os.path.join(a.salida, f"{c}.npz"), **datos)
        m = datos["mascara"]
        totales[c] = {"videos": len(set(d["archivo"])), "secuencias": len(largos), "fotogramas": int(largos.sum()),
                      "validos": int(m.sum()), "nivel": [int((m & (datos["nivel"] == k)).sum()) for k in range(3)],
                      "sin_tray": int(datos["sin_tray"].sum()), "sin_larga": int((~datos["con_larga"]).sum()),
                      "tau": int((datos["tau_mascara"] == 1).sum()), "ventanas": len(vent)}
        print(f"{c}: {totales[c]['videos']} videos | {len(largos)} secuencias | {int(largos.sum())} fotogramas"
              f" | {len(vent)} ventanas", flush=True)

    with open(os.path.join(a.resumen, "videos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_VIDEOS)
        w.writeheader()
        w.writerows(filas_video)
    lineas = ["# Vector de escena por fotograma", ""]
    lineas.append(f"- Generado por `python escena.py`. Una secuencia por video y tramo (se reinicia en cada corte de escena),"
                  f" a 10 fps, con todos los fotogramas procesados, tengan o no trayectorias.")
    lineas.append(f"- Vector de {len(COLUMNAS)} valores: maximo y media de los 13 descriptores sobre las trayectorias de fuego vivas"
                  f" (tambien las de menos de 1 s), numero de trayectorias, numero con edad de al menos {a.edad_min:g} s, edad de la"
                  f" mas antigua, numero de cajas y area total de fuego y de humo, y la bandera `sin_trayectoria`.")
    lineas.append("- Las marcas de tiempo de los fotogramas sin trayectoria se interpolan con el indice del fotograma (en los que"
                  " si tienen, la marca real difiere de 0.1 k en 0.03 s como maximo).")
    lineas.append("- Detecciones del fotograma: de las curvas guardadas (YOLO a 5 fps, confianza 0.25); cada fotograma toma la"
                  f" ultima medida anterior. Videos sin curva: {sin_curva}.")
    lineas.append("- Contexto de camara en los fotogramas sin trayectoria: no se guardo; se usa la identidad y la ultima"
                  " proporcion de inliers conocida.")
    lineas.append(f"- Normalizacion ajustada solo con entrenamiento: {norm_x.n_filas} fotogramas de {len(norm_x.videos)} videos.")
    lineas.append(f"- Tramos de menos de {a.min_fotogramas} fotogramas descartados: {cortas}. Fotogramas cuyo tramo calculado con"
                  f" los cortes no coincide con el guardado: {tramo_distinto}.")
    lineas.append("")
    lineas.append("| conjunto | videos | secuencias | fotogramas | dentro del tramo util | controlado | incipiente | incontrolado |"
                  " sin trayectoria | sin trayectoria de 1 s o mas | con tau por cambio | ventanas |")
    lineas.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CONJUNTOS:
        x = totales[c]
        lineas.append(f"| {c} | {x['videos']} | {x['secuencias']} | {x['fotogramas']} | {x['validos']} | {x['nivel'][0]} |"
                      f" {x['nivel'][1]} | {x['nivel'][2]} | {x['sin_tray']} | {x['sin_larga']} | {x['tau']} | {x['ventanas']} |")
    lineas.append("")
    with open(os.path.join(a.resumen, "resumen.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    print(f"videos sin curva: {sin_curva} | tramos cortos descartados: {cortas} | tramo distinto: {tramo_distinto}")
    print(f"datos en: {a.salida} | resumen en: {a.resumen}")


if __name__ == "__main__":
    main()
