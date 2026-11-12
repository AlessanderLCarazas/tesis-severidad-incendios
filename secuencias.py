import os
import csv
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from cortes import estado_tramos
from normalizacion import Normalizador, COLUMNAS

CONTEXTO = ["s_k", "rot", "tx", "ty", "inliers", "dt"]
NIVELES = ["controlado", "incipiente", "incontrolado"]
CONJUNTOS = ["entrenamiento", "validacion", "calibracion", "prueba"]
RIESGO = [0.0, 0.5, 1.0]
REF_HUMANO = {"ninguna": 0, "nivel_inicial": 1, "cambios": 2, "descartado": 3, "clase": 4}
CAMPOS_INDICE = ["conjunto", "archivo", "clase", "tramo", "id", "muestras", "t_inicio", "t_fin", "muestras_validas",
                 "muestras_controlado", "muestras_incipiente", "muestras_incontrolado", "muestras_tau", "ventanas"]
CAMPOS_VIDEOS = ["conjunto", "archivo", "clase", "tipo", "fuente", "trayectorias", "trayectorias_cortas", "muestras",
                 "s_controlado", "s_incipiente", "s_incontrolado", "llega_a_incontrolado", "ref_humano"]


def numero(v):
    return None if v in ("", None) or (isinstance(v, float) and np.isnan(v)) else float(v)


def leer_csv(ruta):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def instantes(meta):
    ini = NIVELES.index(meta["nivel_inicial"]) if meta["nivel_inicial"] in NIVELES else 0
    return ini, numero(meta["t_incipiente"]), numero(meta["t_incontrolado"]), float(meta["t_inicio_util"]), float(meta["t_fin_util"])


def t_peligro(meta):
    ini, _, tu, t0, _ = instantes(meta)
    if meta["nivel_inicial"] == "sin_fuego":
        return None, 0
    if tu is not None and ini < 2:
        return tu, 1
    if ini == 2:
        return t0, 2
    return None, 0


def metas_muestras(meta, t, tramo, inicios):
    n = len(t)
    if meta is None or meta["descartar"] == "1":
        return np.zeros(n, np.int8), np.zeros(n, bool), np.zeros(n, np.float32), np.zeros(n, np.int8)
    ini, ti, tu, t0, t1 = instantes(meta)
    nivel = np.full(n, ini, np.int8)
    if ti is not None:
        nivel[t >= ti] = np.maximum(nivel[t >= ti], 1)
    if tu is not None:
        nivel[t >= tu] = 2
    mascara = (t >= t0 - 0.01) & (t <= t1 + 0.01)
    tau = np.zeros(n, np.float32)
    tau_m = np.zeros(n, np.int8)
    t_p, codigo = t_peligro(meta)
    if t_p is not None:
        estados = estado_tramos(t_p, inicios)
        for j, estado in enumerate(estados):
            sel = tramo == j
            if estado == "normal":
                tau[sel] = np.maximum(0.0, t_p - t[sel])
                tau_m[sel] = codigo
            elif estado == "peligro":
                tau_m[sel] = codigo
    tau_m[~mascara] = 0
    return nivel, mascara, tau, tau_m


def segundos_escena(meta):
    if meta["nivel_inicial"] == "sin_fuego" or meta["descartar"] == "1":
        return [0.0, 0.0, 0.0]
    ini, ti, tu, t0, t1 = instantes(meta)
    cortes = [t0]
    niveles = [ini]
    if ti is not None and ini < 1 and t0 < ti < t1 and (tu is None or ti < tu):
        cortes.append(ti)
        niveles.append(1)
    if tu is not None and ini < 2 and t0 < tu < t1:
        cortes.append(tu)
        niveles.append(2)
    cortes.append(t1)
    seg = [0.0, 0.0, 0.0]
    for nv, a, b in zip(niveles, cortes[:-1], cortes[1:]):
        seg[nv] += max(0.0, b - a)
    return seg


def ventanas_de(largo, ventana, paso):
    if largo <= ventana:
        return [(0, largo)]
    inicios = list(range(0, largo - ventana + 1, paso))
    if inicios[-1] != largo - ventana:
        inicios.append(largo - ventana)
    return [(i, i + ventana) for i in inicios]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--descriptores", default=os.path.join("runs_fuego", "descriptores_v2_noche", "descriptores.csv"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--cortes", default=os.path.join("resultados", "cortes", "cortes.csv"))
    p.add_argument("--fuente", default="ia")
    p.add_argument("--min_muestras", type=int, default=10)
    p.add_argument("--ventana", type=int, default=300)
    p.add_argument("--paso", type=int, default=150)
    p.add_argument("--salida", default=os.path.join("runs_fuego", "secuencias"))
    p.add_argument("--resumen", default=os.path.join("resultados", "secuencias"))
    a = p.parse_args()

    particion = {r["archivo"]: r for r in leer_csv(a.particion) if r["conjunto"] in CONJUNTOS}
    principal, humano = {}, {}
    for m in leer_csv(a.metas):
        if m["fuente"] in (a.fuente, "dataset"):
            principal[m["archivo"]] = m
        if m["fuente"] == "humano":
            humano[m["archivo"]] = m
    sin_meta = [v for v in particion if v not in principal]
    if sin_meta:
        raise SystemExit(f"videos de la particion sin meta de la fuente {a.fuente}: {len(sin_meta)}")
    inicios = defaultdict(lambda: [0.0])
    for r in leer_csv(a.cortes):
        inicios[r["archivo"]].append(float(r["t_corte"]))
    inicios = defaultdict(lambda: [0.0], {v: sorted(x) for v, x in inicios.items()})

    print("leyendo descriptores...", flush=True)
    df = pd.read_csv(a.descriptores, dtype={"archivo": str}, keep_default_na=False, na_values=[])
    df = df[df["archivo"].isin(set(particion))]
    df = df.sort_values(["archivo", "tramo", "id", "k"], kind="stable").reset_index(drop=True)
    valores = df[COLUMNAS + CONTEXTO].to_numpy(np.float64)
    if not np.isfinite(valores).all():
        raise SystemExit("hay valores no finitos en los descriptores")
    claves = df["archivo"] + "\t" + df["tramo"].astype(str) + "\t" + df["id"].astype(str)
    cambios = np.flatnonzero((claves != claves.shift()).to_numpy())
    limites = list(cambios) + [len(df)]
    archivo_col = df["archivo"].to_numpy()
    tramo_col = df["tramo"].to_numpy(np.int64)
    id_col = df["id"].to_numpy(np.int64)
    t_col = df["t"].to_numpy(np.float64)
    dt_col = df["dt"].to_numpy(np.float64)

    por_conjunto = {c: [] for c in CONJUNTOS}
    cortas = defaultdict(int)
    for a0, a1 in zip(limites[:-1], limites[1:]):
        v = archivo_col[a0]
        if a1 - a0 < a.min_muestras:
            cortas[v] += 1
            continue
        por_conjunto[particion[v]["conjunto"]].append((a0, a1))

    filas_ent = np.concatenate([np.arange(a0, a1) for a0, a1 in por_conjunto["entrenamiento"]])
    videos_ent = sorted({archivo_col[a0] for a0, _ in por_conjunto["entrenamiento"]})
    norm_x = Normalizador(COLUMNAS).ajustar(valores[filas_ent, :13], videos_ent)
    norm_c = Normalizador(CONTEXTO).ajustar(valores[filas_ent, 13:], videos_ent)
    os.makedirs(a.resumen, exist_ok=True)
    os.makedirs(a.salida, exist_ok=True)
    norm_x.guardar(os.path.join(a.resumen, "normalizacion.json"))
    norm_c.guardar(os.path.join(a.resumen, "normalizacion_contexto.json"))

    indice, filas_video, totales = [], [], {}
    for c in CONJUNTOS:
        secs = por_conjunto[c]
        partes = defaultdict(list)
        inicio_seq, ventanas = [0], []
        seq_archivo, seq_tramo, seq_id, seq_ref = [], [], [], []
        tray = defaultdict(int)
        muestras_video = defaultdict(int)
        for n, (a0, a1) in enumerate(secs):
            v = archivo_col[a0]
            t = t_col[a0:a1]
            tramo = tramo_col[a0:a1]
            nivel, mascara, tau, tau_m = metas_muestras(principal[v], t, tramo, inicios[v])
            partes["x"].append(norm_x.transformar(valores[a0:a1, :13]).astype(np.float32))
            partes["c"].append(norm_c.transformar(valores[a0:a1, 13:]).astype(np.float32))
            partes["dt"].append(dt_col[a0:a1].astype(np.float32))
            partes["t"].append(t.astype(np.float32))
            partes["nivel"].append(nivel)
            partes["riesgo"].append(np.array(RIESGO, np.float32)[nivel])
            partes["mascara"].append(mascara)
            partes["tau"].append(tau)
            partes["tau_mascara"].append(tau_m)
            ref = "ninguna"
            if c == "prueba":
                if principal[v]["nivel_inicial"] == "sin_fuego":
                    mh, ref = principal[v], "clase"
                else:
                    mh = humano.get(v)
                    ref = "ninguna" if mh is None else "descartado" if mh["descartar"] == "1" else mh["valida"]
                nh, mk, th, tm = metas_muestras(mh, t, tramo, inicios[v])
                partes["nivel_humano"].append(nh)
                partes["riesgo_humano"].append(np.array(RIESGO, np.float32)[nh])
                partes["mascara_humano"].append(mk)
                partes["tau_humano"].append(th)
                partes["tau_mascara_humano"].append(tm)
            vs = ventanas_de(a1 - a0, a.ventana, a.paso)
            ventanas += [(n, i, j) for i, j in vs]
            inicio_seq.append(inicio_seq[-1] + a1 - a0)
            seq_archivo.append(v)
            seq_tramo.append(int(tramo[0]))
            seq_id.append(int(id_col[a0]))
            seq_ref.append(REF_HUMANO[ref])
            tray[v] += 1
            muestras_video[v] += a1 - a0
            indice.append({"conjunto": c, "archivo": v, "clase": particion[v]["clase"], "tramo": int(tramo[0]),
                           "id": int(id_col[a0]), "muestras": a1 - a0, "t_inicio": round(float(t[0]), 2),
                           "t_fin": round(float(t[-1]), 2), "muestras_validas": int(mascara.sum()),
                           "muestras_controlado": int((mascara & (nivel == 0)).sum()),
                           "muestras_incipiente": int((mascara & (nivel == 1)).sum()),
                           "muestras_incontrolado": int((mascara & (nivel == 2)).sum()),
                           "muestras_tau": int((tau_m > 0).sum()), "ventanas": len(vs)})
        datos = {k: np.concatenate(x) for k, x in partes.items()}
        datos["seq_inicio"] = np.array(inicio_seq, np.int64)
        datos["seq_archivo"] = np.array(seq_archivo, dtype=str)
        datos["seq_tramo"] = np.array(seq_tramo, np.int32)
        datos["seq_id"] = np.array(seq_id, np.int32)
        datos["seq_ref_humano"] = np.array(seq_ref, np.int8)
        datos["ventanas"] = np.array(ventanas, np.int64).reshape(-1, 3)
        datos["columnas_x"] = np.array(COLUMNAS, dtype=str)
        datos["columnas_c"] = np.array(CONTEXTO, dtype=str)
        np.savez_compressed(os.path.join(a.salida, f"{c}.npz"), **datos)

        videos = sorted(v for v in particion if particion[v]["conjunto"] == c)
        seg_tot = np.zeros(3)
        for v in videos:
            seg = segundos_escena(principal[v])
            seg_tot += seg
            ref = ""
            if c == "prueba":
                mh = humano.get(v)
                ref = ("clase" if principal[v]["nivel_inicial"] == "sin_fuego" else "ninguna" if mh is None
                       else "descartado" if mh["descartar"] == "1" else mh["valida"])
            filas_video.append({"conjunto": c, "archivo": v, "clase": particion[v]["clase"], "tipo": particion[v]["tipo"],
                                "fuente": principal[v]["fuente"], "trayectorias": tray[v], "trayectorias_cortas": cortas[v],
                                "muestras": muestras_video[v], "s_controlado": round(seg[0], 2), "s_incipiente": round(seg[1], 2),
                                "s_incontrolado": round(seg[2], 2),
                                "llega_a_incontrolado": "si" if t_peligro(principal[v])[0] is not None else "no",
                                "ref_humano": ref})
        m = datos["mascara"]
        totales[c] = {"videos": len(videos), "fuego": sum(particion[v]["clase"] == "fuego" for v in videos),
                      "sin_fuego": sum(particion[v]["clase"] == "sin_fuego" for v in videos),
                      "con_tray": sum(tray[v] > 0 for v in videos),
                      "sin_fuego_con_tray": sum(tray[v] > 0 and particion[v]["clase"] == "sin_fuego" for v in videos),
                      "tray": len(secs), "cortas": sum(cortas[v] for v in videos), "muestras": int(len(m)),
                      "validas": int(m.sum()), "nivel": [int((m & (datos["nivel"] == k)).sum()) for k in range(3)],
                      "tau": [int((datos["tau_mascara"] == k).sum()) for k in (1, 2)], "ventanas": len(ventanas),
                      "escena": seg_tot.tolist(),
                      "llegan": sum(1 for v in videos if t_peligro(principal[v])[1] == 1)}
        print(f"{c}: {len(videos)} videos | {len(secs)} trayectorias | {len(m)} muestras | {len(ventanas)} ventanas", flush=True)

    with open(os.path.join(a.resumen, "indice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_INDICE)
        w.writeheader()
        w.writerows(indice)
    with open(os.path.join(a.resumen, "videos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_VIDEOS)
        w.writeheader()
        w.writerows(filas_video)

    lineas = ["# Secuencias por trayectoria (paso 11)", ""]
    lineas.append(f"- Generado por `python secuencias.py`. Descriptores: `{a.descriptores.replace(os.sep, '/')}`."
                  f" Metas: `{a.metas.replace(os.sep, '/')}`, fuente `{a.fuente}`.")
    lineas.append(f"- Una secuencia por video, tramo y trayectoria de fuego. Se descartan las de menos de {a.min_muestras}"
                  f" muestras. Ventanas de {a.ventana} muestras con paso de {a.paso}.")
    lineas.append("- Riesgo continuo (paso 9): 0 en controlado o sin fuego, 0.5 en incipiente y 1 en incontrolado.")
    lineas.append(f"- Normalizacion ajustada solo con entrenamiento: {norm_x.n_filas} filas de {len(norm_x.videos)} videos.")
    lineas.append("- `tau_mascara`: 0 sin tau, 1 el video pasa a incontrolado, 2 el video empieza en incontrolado (tau 0).")
    lineas.append("")
    lineas.append("| conjunto | videos | de fuego | sin fuego | videos con trayectorias | sin fuego con trayectorias | trayectorias |"
                  " trayectorias cortas descartadas | muestras | ventanas |")
    lineas.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in CONJUNTOS:
        x = totales[c]
        lineas.append(f"| {c} | {x['videos']} | {x['fuego']} | {x['sin_fuego']} | {x['con_tray']} | {x['sin_fuego_con_tray']} |"
                      f" {x['tray']} | {x['cortas']} | {x['muestras']} | {x['ventanas']} |")
    lineas.append("")
    lineas.append("Segundos de escena por nivel (linea de tiempo de cada video dentro del tramo util, segun la fuente principal):")
    lineas.append("")
    lineas.append("| conjunto | controlado (s) | incipiente (s) | incontrolado (s) | videos que pasan a incontrolado |")
    lineas.append("|---|---|---|---|---|")
    for c in CONJUNTOS:
        x = totales[c]
        lineas.append(f"| {c} | {x['escena'][0]:.0f} | {x['escena'][1]:.0f} | {x['escena'][2]:.0f} | {x['llegan']} |")
    lineas.append("")
    lineas.append("Muestras de trayectoria por nivel (cada muestra son unos 0.1 s; una escena con varias trayectorias cuenta varias veces):")
    lineas.append("")
    lineas.append("| conjunto | validas | controlado | incipiente | incontrolado | con tau por cambio | con tau 0 por empezar en incontrolado |")
    lineas.append("|---|---|---|---|---|---|---|")
    for c in CONJUNTOS:
        x = totales[c]
        lineas.append(f"| {c} | {x['validas']} | {x['nivel'][0]} | {x['nivel'][1]} | {x['nivel'][2]} | {x['tau'][0]} | {x['tau'][1]} |")
    lineas.append("")
    with open(os.path.join(a.resumen, "resumen.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    x = totales["entrenamiento"]
    print(f"entrenamiento, segundos de escena: controlado {x['escena'][0]:.0f} | incipiente {x['escena'][1]:.0f}"
          f" | incontrolado {x['escena'][2]:.0f}")
    print(f"entrenamiento, muestras de trayectoria: controlado {x['nivel'][0]} | incipiente {x['nivel'][1]}"
          f" | incontrolado {x['nivel'][2]}")
    print(f"secuencias en: {a.salida} | resumen en: {a.resumen}")


if __name__ == "__main__":
    main()
