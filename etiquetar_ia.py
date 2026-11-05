import os
import csv
import json
import random
import argparse
from datetime import datetime

NIVELES = ["controlado", "incipiente", "incontrolado"]
CAMPOS = ["archivo", "anotador", "nivel_inicial", "t_incipiente", "t_incontrolado", "t_inicio_util", "t_fin_util",
          "descartar", "motivo", "foto_incipiente", "foto_incontrolado", "foto_inicio_util", "foto_fin_util", "fecha"]


def cargar_tiempos(ruta):
    tiempos = {}
    with open(ruta, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tiempos.setdefault(r["archivo"], {})[int(r["foto"])] = max(0.0, float(r["t_s"]))
    return tiempos


def lista_videos(a, tiempos):
    with open(a.seleccion, encoding="utf-8") as f:
        lista = list(csv.DictReader(f))[:a.limite]
    return [r for r in lista if r["archivo"] in tiempos]


def preparar(a):
    tiempos = cargar_tiempos(os.path.join(a.tiras, "tiempos.csv"))
    lista = lista_videos(a, tiempos)
    os.makedirs(a.lotes, exist_ok=True)
    n_lotes = 0
    for i in range(0, len(lista), a.tam_lote):
        n_lotes += 1
        ruta = os.path.join(a.lotes, f"lote_{n_lotes:02d}.csv")
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["archivo", "tira", "fotos", "duracion_s"])
            for r in lista[i:i + a.tam_lote]:
                base = os.path.splitext(r["archivo"])[0]
                w.writerow([r["archivo"], f"{a.tiras}/{base}.jpg", len(tiempos[r["archivo"]]), r["duracion_s"]])
    print(f"videos: {len(lista)}, lotes de {a.tam_lote}: {n_lotes}, en: {a.lotes}")


def validar(d, fotos):
    if d.get("descartar") is True:
        return None
    if d.get("nivel_inicial") not in NIVELES:
        return "nivel_inicial invalido"
    for k in ["foto_incipiente", "foto_incontrolado", "foto_inicio_util", "foto_fin_util"]:
        v = d.get(k)
        if v is not None and (not isinstance(v, int) or v not in fotos):
            return f"{k} fuera de rango"
    ini = NIVELES.index(d["nivel_inicial"])
    if ini >= 1 and d.get("foto_incipiente") is not None:
        return "foto_incipiente con nivel inicial incipiente o mayor"
    if ini == 2 and d.get("foto_incontrolado") is not None:
        return "foto_incontrolado con nivel inicial incontrolado"
    fi, fu = d.get("foto_incipiente"), d.get("foto_incontrolado")
    if fi is not None and fu is not None and fi > fu:
        return "incipiente despues de incontrolado"
    a0, a1 = d.get("foto_inicio_util"), d.get("foto_fin_util")
    if a0 is not None and a1 is not None and a0 > a1:
        return "inicio util despues del fin util"
    return None


def juntar(a):
    tiempos = cargar_tiempos(os.path.join(a.tiras, "tiempos.csv"))
    lista = lista_videos(a, tiempos)
    filas, faltan, invalidos = [], [], []
    for r in lista:
        nombre = r["archivo"]
        ruta = os.path.join(a.crudo, os.path.splitext(nombre)[0] + ".json")
        if not os.path.exists(ruta):
            faltan.append(nombre)
            continue
        try:
            with open(ruta, encoding="utf-8") as f:
                d = json.loads(f.read().strip())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            invalidos.append((nombre, f"json invalido: {e}"))
            continue
        t = tiempos[nombre]
        error = validar(d, t)
        if error:
            invalidos.append((nombre, error))
            continue
        if d.get("foto_incontrolado") is not None and d.get("nivel_inicial") == "controlado" and d.get("foto_incipiente") is None:
            d["foto_incipiente"] = d["foto_incontrolado"]

        def tt(k):
            v = d.get(k)
            return "" if v is None else round(t[v], 2)

        fila = {"archivo": nombre, "anotador": a.anotador, "descartar": int(d.get("descartar") is True),
                "motivo": d.get("motivo", ""), "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        if fila["descartar"]:
            fila.update(nivel_inicial="", t_incipiente="", t_incontrolado="", t_inicio_util="", t_fin_util="",
                        foto_incipiente="", foto_incontrolado="", foto_inicio_util="", foto_fin_util="")
        else:
            fila.update(nivel_inicial=d["nivel_inicial"], t_incipiente=tt("foto_incipiente"), t_incontrolado=tt("foto_incontrolado"),
                        t_inicio_util=tt("foto_inicio_util"), t_fin_util=tt("foto_fin_util"))
            for k in ["foto_incipiente", "foto_incontrolado", "foto_inicio_util", "foto_fin_util"]:
                fila[k] = "" if d.get(k) is None else d[k]
        filas.append(fila)
    os.makedirs(os.path.dirname(a.salida) or ".", exist_ok=True)
    with open(a.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)
    print(f"videos: {len(lista)}, validos: {len(filas)}, faltan: {len(faltan)}, invalidos: {len(invalidos)}")
    for n in faltan:
        print(f"  falta: {n}")
    for n, e in invalidos:
        print(f"  invalido: {n} ({e})")
    print(f"guardado en: {a.salida}")


def nivel_max(r):
    if r["t_incontrolado"] != "" or r["nivel_inicial"] == "incontrolado":
        return "incontrolado"
    if r["t_incipiente"] != "" or r["nivel_inicial"] == "incipiente":
        return "incipiente"
    return "controlado"


def nombre_base(nombre):
    base = os.path.splitext(nombre)[0].strip()
    for sufijo in [" - Copy", " - copia", " - Copia", " (1)", " (2)"]:
        if base.endswith(sufijo):
            base = base[:-len(sufijo)].strip()
    return base


def prueba(a):
    with open(a.salida, encoding="utf-8") as f:
        todas = list(csv.DictReader(f))
    vistos = set()
    copias = []
    for r in todas:
        b = nombre_base(r["archivo"])
        if b in vistos:
            copias.append(r["archivo"])
        vistos.add(b)
    filas = [r for r in todas if r["descartar"] == "0" and r["archivo"] not in copias]
    for c in copias:
        print(f"copia excluida: {c}")
    estratos = {}
    for r in filas:
        estratos.setdefault(nivel_max(r), []).append(r["archivo"])
    rng = random.Random(a.semilla)
    total = len(filas)
    cuota = {k: a.n_prueba * len(v) / total for k, v in estratos.items()}
    asignado = {k: max(a.minimo_estrato, int(c)) for k, c in cuota.items()}
    asignado = {k: min(n, len(estratos[k])) for k, n in asignado.items()}
    resto = sorted(estratos, key=lambda k: -(cuota[k] - int(cuota[k])))
    i = 0
    while sum(asignado.values()) < a.n_prueba and i < 10 * len(resto):
        k = resto[i % len(resto)]
        if asignado[k] < len(estratos[k]):
            asignado[k] += 1
        i += 1
    while sum(asignado.values()) > a.n_prueba:
        reducibles = [x for x in asignado if asignado[x] > min(a.minimo_estrato, len(estratos[x]))]
        if not reducibles:
            break
        k = max(reducibles, key=lambda x: asignado[x] - cuota[x])
        asignado[k] -= 1
    elegidos = []
    for k in sorted(estratos):
        elegidos += rng.sample(sorted(estratos[k]), asignado[k])
    rng.shuffle(elegidos)
    with open(a.prueba, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo"])
        for n in elegidos:
            w.writerow([n])
    print(f"videos no descartados: {total}")
    for k in NIVELES:
        if k in estratos:
            print(f"  {k}: {len(estratos[k])} en total, {asignado[k]} elegidos")
    print(f"lista de prueba ({len(elegidos)} videos) en: {a.prueba}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("modo", choices=["preparar", "juntar", "prueba"])
    p.add_argument("--seleccion", default="seleccion_etiquetado.csv")
    p.add_argument("--limite", type=int, default=200)
    p.add_argument("--tiras", default="tiras")
    p.add_argument("--lotes", default=os.path.join("prompts", "lotes"))
    p.add_argument("--tam_lote", type=int, default=20)
    p.add_argument("--crudo", default=os.path.join("resultados", "ia_crudo"))
    p.add_argument("--salida", default=os.path.join("resultados", "etiquetas_ia.csv"))
    p.add_argument("--anotador", default="ia_sonnet")
    p.add_argument("--prueba", default=os.path.join("resultados", "prueba_humana.csv"))
    p.add_argument("--n_prueba", type=int, default=40)
    p.add_argument("--minimo_estrato", type=int, default=5)
    p.add_argument("--semilla", type=int, default=42)
    a = p.parse_args()
    {"preparar": preparar, "juntar": juntar, "prueba": prueba}[a.modo](a)


if __name__ == "__main__":
    main()
