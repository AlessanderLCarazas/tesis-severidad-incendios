import os
import csv
import argparse
from datetime import datetime
import numpy as np

NIVELES = ["controlado", "incipiente", "incontrolado"]


def leer(ruta, anotador=None):
    with open(ruta, encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f) if anotador is None or r["anotador"] == anotador]
    return {r["archivo"]: r for r in filas}


def num(v):
    return None if v in ("", None) else float(v)


def nivel_max(r):
    if num(r["t_incontrolado"]) is not None or r["nivel_inicial"] == "incontrolado":
        return "incontrolado"
    if num(r["t_incipiente"]) is not None or r["nivel_inicial"] == "incipiente":
        return "incipiente"
    return "controlado"


def kappa(a, b, k, pesos):
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[x, y] += 1
    n = m.sum()
    if n == 0:
        return float("nan"), m
    i, j = np.indices((k, k))
    if pesos == "lineal":
        w = np.abs(i - j) / (k - 1)
    elif pesos == "cuadratico":
        w = ((i - j) / (k - 1)) ** 2
    else:
        w = (i != j).astype(float)
    esperado = np.outer(m.sum(1), m.sum(0)) / n
    den = (w * esperado).sum()
    return (1.0 - (w * m).sum() / den) if den > 0 else float("nan"), m


def calidad(hum, lineas):
    lineas.append("## calidad de las etiquetas humanas")
    fechas = sorted(datetime.strptime(r["fecha"], "%Y-%m-%d %H:%M:%S") for r in hum.values())
    dif = np.array([(b - a).total_seconds() for a, b in zip(fechas[:-1], fechas[1:])])
    lineas.append(f"videos: {len(hum)}")
    if len(dif):
        lineas.append(f"segundos entre guardados ({len(dif)} intervalos): minimo {dif.min():.0f}, mediana {np.median(dif):.0f},"
                      f" media {dif.mean():.0f}, maximo {dif.max():.0f}")
        lineas.append(f"intervalos menores a 10 s: {int((dif < 10).sum())}, menores a 20 s: {int((dif < 20).sum())}")
        lineas.append(f"tiempo total de la sesion: {(fechas[-1] - fechas[0]).total_seconds() / 60:.1f} min")
    sin_nivel = [n for n, r in hum.items() if r["nivel_inicial"] == "" and r["descartar"] != "1"]
    descartados = [n for n, r in hum.items() if r["descartar"] == "1"]
    sin_util = [n for n, r in hum.items() if r["t_inicio_util"] == "" or r["t_fin_util"] == ""]
    lineas.append(f"sin nivel inicial (no descartados): {len(sin_nivel)}")
    lineas.append(f"descartados: {len(descartados)}")
    lineas.append(f"sin inicio o fin util marcado (se toma el video completo): {len(sin_util)}")
    con_inc = [n for n, r in hum.items() if num(r["t_incipiente"]) is not None]
    con_unc = [n for n, r in hum.items() if num(r["t_incontrolado"]) is not None]
    lineas.append(f"cambios marcados: t_incipiente {len(con_inc)}, t_incontrolado {len(con_unc)}")


def comparar(hum, ia, lineas):
    comunes = [n for n in hum if n in ia]
    validos = [n for n in comunes if hum[n]["descartar"] != "1" and ia[n]["descartar"] != "1"
               and hum[n]["nivel_inicial"] in NIVELES and ia[n]["nivel_inicial"] in NIVELES]
    lineas.append("")
    lineas.append("## comparacion humano contra ia")
    lineas.append(f"videos en comun: {len(comunes)}, comparables (ambos con nivel y sin descartar): {len(validos)}")
    for titulo, f in [("nivel inicial", lambda r: r["nivel_inicial"]), ("nivel maximo del video", nivel_max)]:
        h = [NIVELES.index(f(hum[n])) for n in validos]
        a = [NIVELES.index(f(ia[n])) for n in validos]
        acuerdo = float(np.mean([x == y for x, y in zip(h, a)])) if validos else float("nan")
        kl, m = kappa(h, a, 3, "lineal")
        kc, _ = kappa(h, a, 3, "cuadratico")
        ks, _ = kappa(h, a, 3, "sin")
        lineas.append("")
        lineas.append(f"### {titulo}")
        lineas.append(f"acuerdo: {acuerdo * 100:.1f} % ({sum(x == y for x, y in zip(h, a))} de {len(validos)})")
        lineas.append(f"kappa ponderado lineal: {kl:.3f} | kappa ponderado cuadratico: {kc:.3f} | kappa sin ponderar: {ks:.3f}")
        lineas.append("matriz de confusion (filas: humano, columnas: ia)")
        lineas.append("| humano \\ ia | " + " | ".join(NIVELES) + " |")
        lineas.append("|---|" + "---|" * 3)
        for i, nv in enumerate(NIVELES):
            lineas.append(f"| {nv} | " + " | ".join(str(int(v)) for v in m[i]) + " |")
    lineas.append("")
    lineas.append("### cambios de nivel")
    detalle = []
    for campo in ["t_incipiente", "t_incontrolado"]:
        ambos, solo_h, solo_ia, difs = 0, 0, 0, []
        for n in validos:
            th, ta = num(hum[n][campo]), num(ia[n][campo])
            if th is not None and ta is not None:
                ambos += 1
                difs.append(abs(th - ta))
                detalle.append((n, campo, th, ta))
            elif th is not None:
                solo_h += 1
                detalle.append((n, campo, th, None))
            elif ta is not None:
                solo_ia += 1
                detalle.append((n, campo, None, ta))
        txt = f"{campo}: en ambos {ambos}, solo humano {solo_h}, solo ia {solo_ia}"
        if difs:
            txt += f" | diferencia absoluta en s: media {np.mean(difs):.2f}, mediana {np.median(difs):.2f}, maxima {np.max(difs):.2f}"
        lineas.append(txt)
    if detalle:
        lineas.append("")
        lineas.append("| video | cambio | humano (s) | ia (s) |")
        lineas.append("|---|---|---|---|")
        for n, c, th, ta in detalle:
            lineas.append(f"| {n} | {c} | {'-' if th is None else th} | {'-' if ta is None else ta} |")
    desacuerdos = [n for n in validos if hum[n]["nivel_inicial"] != ia[n]["nivel_inicial"]]
    if desacuerdos:
        lineas.append("")
        lineas.append("### videos con distinto nivel inicial")
        lineas.append("| video | humano | ia | motivo de la ia |")
        lineas.append("|---|---|---|---|")
        for n in desacuerdos:
            lineas.append(f"| {n} | {hum[n]['nivel_inicial']} | {ia[n]['nivel_inicial']} | {ia[n].get('motivo', '')} |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--humano", default="etiquetas.csv")
    p.add_argument("--anotador", default="aleCa")
    p.add_argument("--ia", default=os.path.join("resultados", "etiquetas_ia.csv"))
    p.add_argument("--lista", default=os.path.join("resultados", "prueba_humana.csv"))
    p.add_argument("--salida", default=os.path.join("resultados", "comparacion_humano_ia.md"))
    a = p.parse_args()
    with open(a.lista, encoding="utf-8") as f:
        lista = {r["archivo"] for r in csv.DictReader(f)}
    hum = {n: r for n, r in leer(a.humano, a.anotador).items() if n in lista}
    ia = leer(a.ia)
    lineas = []
    calidad(hum, lineas)
    comparar(hum, ia, lineas)
    with open(a.salida, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    print("\n".join(lineas))
    print(f"\nguardado en: {a.salida}")


if __name__ == "__main__":
    main()
