import os
import csv
import argparse


def nombre_base(nombre):
    base = os.path.splitext(nombre)[0].strip()
    for sufijo in [" - Copy", " - copia", " - Copia", " (1)", " (2)"]:
        if base.endswith(sufijo):
            base = base[:-len(sufijo)].strip()
    return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadatos", default="metadatos_videos.csv")
    p.add_argument("--curvas", default="curvas")
    p.add_argument("--clase", default="fire")
    p.add_argument("--duracion_min", type=float, default=20.0)
    p.add_argument("--horas", type=float, default=8.0)
    p.add_argument("--ritmo", type=float, default=0.14)
    p.add_argument("--salida", default="lista_sin_curva.csv")
    a = p.parse_args()

    with open(a.metadatos, encoding="utf-8") as f:
        meta = [r for r in csv.DictReader(f) if r["clase"] == a.clase]
    con_curva = {os.path.splitext(n)[0] for n in os.listdir(a.curvas)} if os.path.isdir(a.curvas) else set()
    vistos = set()
    candidatos = []
    copias = 0
    for r in sorted(meta, key=lambda x: -float(x["duracion_s"] or 0)):
        dur = float(r["duracion_s"] or 0)
        if dur <= a.duracion_min:
            continue
        base = nombre_base(r["archivo"])
        if base in vistos:
            copias += 1
            continue
        vistos.add(base)
        if os.path.splitext(r["archivo"])[0] in con_curva or base in con_curva:
            continue
        candidatos.append(r)
    primeros, resto, origenes = [], [], set()
    for r in candidatos:
        o = r["origen"].strip()
        (resto if o in origenes else primeros).append(r)
        origenes.add(o)
    orden = primeros + resto
    presupuesto = a.horas * 3600.0
    acumulado = 0.0
    entran = 0
    for r in orden:
        acumulado += float(r["duracion_s"]) * a.ritmo
        if acumulado <= presupuesto:
            entran += 1
    with open(a.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "origen", "duracion_s", "fps"])
        for r in orden:
            w.writerow([r["archivo"], r["origen"].strip(), r["duracion_s"], r["fps"]])
    total = sum(float(r["duracion_s"]) for r in orden)
    print(f"videos de {a.clase} de mas de {a.duracion_min:.0f} s sin curva: {len(orden)} ({total / 3600:.2f} h de video), copias omitidas: {copias}")
    print(f"primero un video por origen: {len(primeros)}, despues los demas clips: {len(resto)}")
    print(f"con un ritmo de {a.ritmo} s por segundo de video, en {a.horas:.0f} h entran unos {entran} videos")
    print(f"lista en: {a.salida}")


if __name__ == "__main__":
    main()
