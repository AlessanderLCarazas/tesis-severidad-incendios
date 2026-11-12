import os
import argparse
import numpy as np
import pandas as pd
from escena import leer_curva, tiempos
from escena_v2 import vectores_carpeta
from secuencias import leer_csv
from vector_escena import COLUMNAS, DESC, CAMARA

CLAVES = ["archivo", "tramo", "id", "k"]


def vector_anterior(df, n, curvas, nombre, paso=0.1, edad_min=1.0):
    x = np.zeros((n, len(COLUMNAS)), np.float64)
    x[:, 33] = 1.0
    k_con, t_con = np.zeros(0, np.int64), np.zeros(0)
    if len(df):
        seq = df["tramo"].astype(str) + "\t" + df["id"].astype(str)
        df = df.assign(edad=df["t"] - df.groupby(seq)["t"].transform("min"))
        df = df.assign(madura=(df["edad"] >= edad_min - 1e-6).astype(float))
        g = df.groupby("k", sort=True)
        tabla = pd.concat([g[DESC].max(), g[DESC].mean(), g["t"].size(), g["madura"].sum(), g["edad"].max(), g["t"].first()], axis=1)
        k_con = tabla.index.to_numpy(np.int64)
        x[k_con, :29] = tabla.to_numpy(np.float64)[:, :29]
        x[k_con, 33] = 0.0
        t_con = tabla.to_numpy(np.float64)[:, 29]
    t = tiempos(n, k_con, t_con, paso)
    curva = leer_curva(curvas, nombre)
    if curva is not None:
        pos = np.searchsorted(curva[:, 0], t + 0.05, side="right") - 1
        ok = pos >= 0
        x[ok, 29:33] = curva[pos[ok], 1:5]
    return x, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nueva", default=os.path.join("runs_fuego", "descriptores_v3_prueba"))
    p.add_argument("--anterior", default=os.path.join("runs_fuego", "descriptores_v2_noche", "descriptores.csv"))
    p.add_argument("--curvas", nargs="+", default=["curvas", os.path.join("runs_fuego", "curvas_sin_fuego")])
    p.add_argument("--salida", default=os.path.join("resultados", "vector_escena", "comparacion.md"))
    a = p.parse_args()

    videos = [r["archivo"] for r in leer_csv(os.path.join(a.nueva, "procesados.csv"))]
    nuevos = vectores_carpeta(a.nueva, videos)
    fot = pd.read_csv(os.path.join(a.nueva, "fotogramas.csv"), dtype={"archivo": str}, keep_default_na=False, na_values=[])
    des_n = pd.read_csv(os.path.join(a.nueva, "descriptores.csv"), dtype={"archivo": str}, keep_default_na=False, na_values=[])
    des_a = pd.read_csv(a.anterior, dtype={"archivo": str}, keep_default_na=False, na_values=[])
    des_a = des_a[des_a["archivo"].isin(set(videos))]

    lineas = ["# Vector de escena: datos del pipeline (v3) contra el vector anterior", ""]
    lineas.append(f"Generado por `python probar_vector_escena.py`. Videos: {len(videos)}. Vector anterior: descriptores de la corrida"
                  " de la noche y detecciones de `curvas/`. Vector nuevo: `descriptores_v3.py` y `vector_escena.VectorEscena`.")
    lineas.append("")
    lineas.append("| video | fotogramas | filas de descriptores (anterior / nueva) | filas distintas | fotogramas con trayectoria |"
                  " diferencia maxima de los 29 agregados | diferencia maxima de la marca de tiempo (s) |"
                  " fotogramas sin trayectoria con camara distinta de la identidad |")
    lineas.append("|---|---|---|---|---|---|---|---|")
    det = {c: [] for c in ("n_det_fuego", "n_det_humo", "area_det_fuego", "area_det_humo")}
    alta = {c: [] for c in det}
    todo_igual = True
    for v in videos:
        d = nuevos[v]
        n = len(d["k"])
        da, dn = des_a[des_a["archivo"] == v], des_n[des_n["archivo"] == v]
        columnas = [c for c in da.columns if c in dn.columns]
        ma = da.sort_values(CLAVES)[columnas].reset_index(drop=True)
        mn = dn.sort_values(CLAVES)[columnas].reset_index(drop=True)
        if len(ma) == len(mn):
            num = [c for c in columnas if c != "archivo"]
            distintas = int((np.abs(ma[num].to_numpy(float) - mn[num].to_numpy(float)) > 1e-6).any(axis=1).sum())
        else:
            distintas = abs(len(ma) - len(mn))
        x_ant, t_ant = vector_anterior(da, n, a.curvas, v)
        dif = float(np.abs(x_ant[:, :29] - d["x"][:, :29]).max()) if n else 0.0
        dif_t = float(np.abs(t_ant - d["t"]).max()) if n else 0.0
        sin = d["x"][:, 33] > 0
        cam = fot[fot["archivo"] == v].sort_values("k")[CAMARA].to_numpy(float)
        movida = int(((np.abs(cam[sin, 0] - 1.0) > 1e-4) | (np.abs(cam[sin, 1:4]) > 1e-4).any(axis=1)).sum()) if sin.any() else 0
        todo_igual = todo_igual and distintas == 0 and dif <= 1e-6 and bool((x_ant[:, 33] == d["x"][:, 33]).all())
        lineas.append(f"| {v} | {n} | {len(ma)} / {len(mn)} | {distintas} | {int((~sin).sum())} | {dif:.2e} | {dif_t:.4f} |"
                      f" {movida} de {int(sin.sum())} |")
        sub = fot[fot["archivo"] == v].sort_values("k")
        for j, c in enumerate(det):
            det[c].append(np.column_stack([x_ant[:, 29 + j], d["x"][:, 29 + j]]))
            alta[c].append(np.column_stack([x_ant[:, 29 + j], sub[c + "_alta"].to_numpy(float)]))
    lineas.append("")
    lineas.append(f"**Agregados de descriptores, numero de trayectorias, edad y bandera sin_trayectoria: "
                  f"{'coinciden en los ' + str(len(videos)) + ' videos' if todo_igual else 'hay diferencias'}** (tolerancia 1e-6).")
    lineas.append("")
    lineas.append("Detecciones por fotograma: anterior (`curvas/`, 5 fps, confianza 0.25, se mantiene la ultima medida) contra las"
                  " del propio seguimiento (10 fps, confianza 0.1) y contra las mismas filtradas a confianza 0.25.")
    lineas.append("")
    lineas.append("| valor | media anterior | media nueva (conf 0.1) | diferencia absoluta media | fotogramas distintos |"
                  " media nueva a conf 0.25 | diferencia absoluta media a conf 0.25 | fotogramas distintos a conf 0.25 |")
    lineas.append("|---|---|---|---|---|---|---|---|")
    for c in det:
        m, h = np.vstack(det[c]), np.vstack(alta[c])
        tol = 0.5 if c.startswith("n_") else 0.005
        lineas.append(f"| {c} | {m[:, 0].mean():.4f} | {m[:, 1].mean():.4f} | {np.abs(m[:, 0] - m[:, 1]).mean():.4f} |"
                      f" {100.0 * (np.abs(m[:, 0] - m[:, 1]) > tol).mean():.1f} % | {h[:, 1].mean():.4f} |"
                      f" {np.abs(h[:, 0] - h[:, 1]).mean():.4f} | {100.0 * (np.abs(h[:, 0] - h[:, 1]) > tol).mean():.1f} % |")
    lineas.append("")
    lineas.append("Un fotograma cuenta como distinto si el numero de cajas cambia o si el area difiere en mas de 0.005 del fotograma.")
    lineas.append("")
    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    with open(a.salida, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
