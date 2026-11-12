import os
import re
import csv
import random
import argparse
from collections import Counter, defaultdict
from comparar_etiquetas import NIVELES, leer, num
from duplicados import cargar_excluidos

CONJUNTOS = ["entrenamiento", "validacion", "calibracion", "prueba"]
TIPOS = ["sin_fuego", "controlado", "incipiente", "incontrolado", "controlado_a_incipiente",
         "controlado_a_incontrolado", "incipiente_a_incontrolado", "descartado"]
CAMPOS_ORIGEN = ["archivo", "clase", "origen_metadatos", "grupo", "union_manual"]
CAMPOS_PARTICION = ["archivo", "clase", "grupo", "conjunto", "lista_prueba", "tipo", "nivel_max", "version_ia",
                    "calibra_umbral", "duracion_s", "motivo"]
CAMPOS_METAS = ["archivo", "conjunto", "clase", "fuente", "version", "valida", "nivel_inicial", "t_incipiente",
                "t_incontrolado", "t_inicio_util", "t_fin_util", "descartar", "duracion_s"]


def normalizar(origen):
    return re.sub(r"\s+", " ", origen.strip().lower())


def leer_csv(ruta):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tipo_de(r):
    ini = r["nivel_inicial"]
    if num(r["t_incontrolado"]) is not None and ini != "incontrolado":
        return f"{ini}_a_incontrolado"
    if num(r["t_incipiente"]) is not None and ini == "controlado":
        return "controlado_a_incipiente"
    return ini


def nivel_max_de(tipo):
    return "sin_fuego" if tipo == "sin_fuego" else tipo.split("_a_")[-1]


def repartir(grupos, tamanos, fracciones, previos, rng):
    grupos = sorted(grupos)
    rng.shuffle(grupos)
    grupos.sort(key=lambda g: -tamanos[g])
    total = sum(tamanos[g] for g in grupos) + sum(previos)
    metas = [f * total for f in fracciones]
    cuentas = list(previos)
    salida = [[] for _ in fracciones]
    for g in grupos:
        i = max(range(len(metas)), key=lambda j: (metas[j] - cuentas[j], -j))
        salida[i].append(g)
        cuentas[i] += tamanos[g]
    return salida


def fila_meta(archivo, conjunto, clase, fuente, version, valida, r, dur):
    ini, fin = num(r["t_inicio_util"]), num(r["t_fin_util"])
    return {"archivo": archivo, "conjunto": conjunto, "clase": clase, "fuente": fuente, "version": version, "valida": valida,
            "nivel_inicial": r["nivel_inicial"], "t_incipiente": r["t_incipiente"], "t_incontrolado": r["t_incontrolado"],
            "t_inicio_util": 0.0 if ini is None else ini, "t_fin_util": dur if fin is None else min(fin, dur),
            "descartar": r["descartar"], "duracion_s": dur}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadatos", default="metadatos_videos.csv")
    p.add_argument("--lista", default=os.path.join("resultados", "cortes", "lista_noche.csv"))
    p.add_argument("--ia_v1", default=os.path.join("resultados", "etiquetas_ia.csv"))
    p.add_argument("--ia_v2", default=os.path.join("resultados", "etiquetas_ia_v2.csv"))
    p.add_argument("--humano", default="etiquetas.csv")
    p.add_argument("--anotador_inicial", default="aleCa")
    p.add_argument("--anotador_cambios", default="aleCa_cambios")
    p.add_argument("--prueba_humana", default=os.path.join("resultados", "prueba_humana.csv"))
    p.add_argument("--prueba_cambios", default=os.path.join("resultados", "prueba_cambios.csv"))
    p.add_argument("--uniones", default=os.path.join("resultados", "particion", "uniones.csv"))
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--frac_val", type=float, default=0.15)
    p.add_argument("--frac_cal", type=float, default=0.15)
    p.add_argument("--frac_prueba_sin_fuego", type=float, default=0.15)
    p.add_argument("--salida", default=os.path.join("resultados", "particion"))
    a = p.parse_args()

    meta = {r["archivo"]: r for r in leer_csv(a.metadatos)}
    lista = leer_csv(a.lista)
    clase = {r["archivo"]: ("fuego" if r["clase"] == "fire" else "sin_fuego") for r in lista}
    v1, v2 = leer(a.ia_v1), leer(a.ia_v2)
    hum_ini, hum_cam = leer(a.humano, a.anotador_inicial), leer(a.humano, a.anotador_cambios)
    p_hum = [r["archivo"] for r in leer_csv(a.prueba_humana)]
    p_cam = [r["archivo"] for r in leer_csv(a.prueba_cambios)]
    if set(p_hum) & set(p_cam):
        raise SystemExit("las dos listas de prueba comparten videos")
    prueba_fuego = {v: "prueba_humana" for v in p_hum}
    prueba_fuego.update({v: "prueba_cambios" for v in p_cam})
    excluidos = cargar_excluidos()
    uniones = {r["archivo"]: r["grupo"] for r in leer_csv(a.uniones)} if os.path.exists(a.uniones) else {}
    sobran = [v for v in uniones if v not in clase]
    if sobran:
        raise SystemExit(f"uniones con {len(sobran)} videos que no estan en la lista")
    falta = [v for v in prueba_fuego if v not in clase]
    if falta:
        raise SystemExit(f"videos de prueba fuera de la lista: {len(falta)}")

    grupo = {v: uniones.get(v) or f"{clase[v]}: {normalizar(meta[v]['origen'])}" for v in clase}
    ia, version = {}, {}
    for v in clase:
        if clase[v] == "fuego":
            if v in v2:
                ia[v], version[v] = v2[v], "ia_v2"
            elif v in v1:
                ia[v], version[v] = v1[v], "ia_v1"
    sin_etiqueta = [v for v in clase if clase[v] == "fuego" and v not in ia]
    if sin_etiqueta:
        raise SystemExit(f"videos de fuego sin etiqueta de la ia: {len(sin_etiqueta)}")
    tipo = {v: ("sin_fuego" if clase[v] == "sin_fuego" else "descartado" if ia[v]["descartar"] == "1" else tipo_de(ia[v]))
            for v in clase}

    conjunto, motivo = {}, {}
    grupos_prueba = {grupo[v] for v in prueba_fuego}
    for v in clase:
        if v in excluidos:
            conjunto[v], motivo[v] = "fuera", "duplicado exacto (excluidos.csv)"
        elif v in prueba_fuego:
            conjunto[v], motivo[v] = "prueba", prueba_fuego[v]
        elif grupo[v] in grupos_prueba:
            conjunto[v], motivo[v] = "fuera", "mismo origen que un video de prueba"
        elif clase[v] == "fuego" and ia[v]["descartar"] == "1":
            conjunto[v], motivo[v] = "fuera", "descartado por la ia"

    rng = random.Random(a.semilla)
    libres = [v for v in clase if v not in conjunto]
    conteo_tipo = Counter(tipo[v] for v in libres)
    por_grupo = defaultdict(list)
    for v in libres:
        por_grupo[grupo[v]].append(v)
    tipo_grupo = {g: min((tipo[v] for v in vs), key=lambda t: (conteo_tipo[t], t)) for g, vs in por_grupo.items()}
    for t in sorted(conteo_tipo, key=lambda t: (conteo_tipo[t], t)):
        gs = [g for g in por_grupo if tipo_grupo[g] == t]
        tamanos = {g: sum(tipo[v] == t for v in por_grupo[g]) for g in gs}
        previos = [sum(1 for v in libres if tipo[v] == t and conjunto.get(v) == c) for c in CONJUNTOS]
        if t == "sin_fuego":
            fp = a.frac_prueba_sin_fuego
            resto = 1.0 - fp
            fracciones = [resto * (1.0 - a.frac_val - a.frac_cal), resto * a.frac_val, resto * a.frac_cal, fp]
        else:
            fracciones = [1.0 - a.frac_val - a.frac_cal, a.frac_val, a.frac_cal, 0.0]
        partes = repartir(gs, tamanos, fracciones, previos, rng)
        for nombre, parte in zip(CONJUNTOS, partes):
            for g in parte:
                for v in por_grupo[g]:
                    conjunto[v], motivo[v] = nombre, "sorteo por grupo de origen"

    dentro = defaultdict(set)
    for v in clase:
        if conjunto[v] != "fuera":
            dentro[grupo[v]].add(conjunto[v])
    cruzados = [g for g, c in dentro.items() if len(c) > 1]
    if cruzados:
        raise SystemExit(f"hay {len(cruzados)} grupos de origen en mas de un conjunto")
    fuga = [v for v in clase if conjunto[v] not in ("prueba", "fuera") and grupo[v] in grupos_prueba]
    if fuga:
        raise SystemExit("hay videos del origen de la prueba fuera de la prueba")

    os.makedirs(a.salida, exist_ok=True)
    orden = sorted(clase, key=lambda v: (CONJUNTOS.index(conjunto[v]) if conjunto[v] in CONJUNTOS else 9, clase[v], v))
    with open(os.path.join(a.salida, "origenes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_ORIGEN)
        w.writeheader()
        for v in sorted(clase, key=lambda v: (grupo[v], v)):
            w.writerow({"archivo": v, "clase": clase[v], "origen_metadatos": meta[v]["origen"], "grupo": grupo[v],
                        "union_manual": "si" if v in uniones else "no"})
    with open(os.path.join(a.salida, "particion.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_PARTICION)
        w.writeheader()
        for v in orden:
            umbral = conjunto[v] == "calibracion" and nivel_max_de(tipo[v]) in ("sin_fuego", "controlado")
            w.writerow({"archivo": v, "clase": clase[v], "grupo": grupo[v], "conjunto": conjunto[v],
                        "lista_prueba": prueba_fuego.get(v, ""), "tipo": tipo[v], "nivel_max": nivel_max_de(tipo[v]),
                        "version_ia": version.get(v, ""), "calibra_umbral": "si" if umbral else "no",
                        "duracion_s": meta[v]["duracion_s"], "motivo": motivo[v]})
    metas = []
    for v in orden:
        if conjunto[v] == "fuera":
            continue
        dur = float(meta[v]["duracion_s"])
        if clase[v] == "sin_fuego":
            vacio = {"nivel_inicial": "sin_fuego", "t_incipiente": "", "t_incontrolado": "", "t_inicio_util": "",
                     "t_fin_util": "", "descartar": "0"}
            metas.append(fila_meta(v, conjunto[v], clase[v], "dataset", "detectiumfire", "clase", vacio, dur))
            continue
        metas.append(fila_meta(v, conjunto[v], clase[v], "ia", version[v], "cambios", ia[v], dur))
        if v in hum_cam and prueba_fuego.get(v) == "prueba_cambios":
            metas.append(fila_meta(v, conjunto[v], clase[v], "humano", a.anotador_cambios, "cambios", hum_cam[v], dur))
        elif v in hum_ini and prueba_fuego.get(v) == "prueba_humana":
            metas.append(fila_meta(v, conjunto[v], clase[v], "humano", a.anotador_inicial, "nivel_inicial", hum_ini[v], dur))
    with open(os.path.join(a.salida, "metas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_METAS)
        w.writeheader()
        w.writerows(metas)

    tabla = {c: Counter(tipo[v] for v in clase if conjunto[v] == c) for c in CONJUNTOS + ["fuera"]}
    lineas = ["# Particion por origen (paso 10)", ""]
    lineas.append(f"- Generado por `python particion.py` (semilla {a.semilla}). Videos de la lista: {len(clase)}"
                  f" ({sum(c == 'fuego' for c in clase.values())} de fuego y {sum(c == 'sin_fuego' for c in clase.values())} sin fuego).")
    lineas.append(f"- Grupos de origen: {len(set(grupo.values()))}. Videos con union manual (`uniones.csv`): {len(uniones)}.")
    lineas.append(f"- Etiquetas de la IA: `ia_v2` en {sum(x == 'ia_v2' for x in version.values())} videos y `ia_v1` en"
                  f" {sum(x == 'ia_v1' for x in version.values())} (los que no pasaron por la segunda fase).")
    lineas.append(f"- Fracciones: validacion {a.frac_val:g}, calibracion {a.frac_cal:g}, prueba de los videos sin fuego"
                  f" {a.frac_prueba_sin_fuego:g}; el sorteo es por grupo de origen dentro de cada tipo.")
    lineas.append("- Comprobado: ningun grupo de origen esta en dos conjuntos y ningun video del origen de la prueba queda"
                  " en entrenamiento, validacion o calibracion.")
    lineas.append("")
    lineas.append("| tipo (segun la IA) | " + " | ".join(CONJUNTOS) + " | fuera | total |")
    lineas.append("|---|" + "---|" * (len(CONJUNTOS) + 2))
    for t in TIPOS:
        fila = [tabla[c][t] for c in CONJUNTOS + ["fuera"]]
        if sum(fila):
            lineas.append(f"| {t} | " + " | ".join(str(x) for x in fila) + f" | {sum(fila)} |")
    tot = [sum(tabla[c].values()) for c in CONJUNTOS + ["fuera"]]
    lineas.append("| total | " + " | ".join(str(x) for x in tot) + f" | {sum(tot)} |")
    lineas.append("")
    lineas.append("| conjunto | videos | de fuego | sin fuego | grupos de origen | horas | cuentan para el umbral conformal |")
    lineas.append("|---|---|---|---|---|---|---|")
    for c in CONJUNTOS:
        vs = [v for v in clase if conjunto[v] == c]
        horas = sum(float(meta[v]["duracion_s"]) for v in vs) / 3600.0
        umbral = sum(1 for v in vs if c == "calibracion" and nivel_max_de(tipo[v]) in ("sin_fuego", "controlado"))
        lineas.append(f"| {c} | {len(vs)} | {sum(clase[v] == 'fuego' for v in vs)} | {sum(clase[v] == 'sin_fuego' for v in vs)} |"
                      f" {len({grupo[v] for v in vs})} | {horas:.2f} | {umbral if c == 'calibracion' else '-'} |")
    lineas.append("")
    lineas.append("Videos fuera de todos los conjuntos:")
    lineas.append("")
    lineas.append("| video | tipo | motivo |")
    lineas.append("|---|---|---|")
    for v in orden:
        if conjunto[v] == "fuera":
            lineas.append(f"| {v} | {tipo[v]} | {motivo[v]} |")
    lineas.append("")
    with open(os.path.join(a.salida, "resumen.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))

    for c in CONJUNTOS + ["fuera"]:
        vs = [v for v in clase if conjunto[v] == c]
        detalle = ", ".join(f"{t} {tabla[c][t]}" for t in TIPOS if tabla[c][t])
        print(f"{c}: {len(vs)} videos | {detalle}")
    print(f"motivos de los que quedan fuera: {dict(Counter(motivo[v] for v in clase if conjunto[v] == 'fuera'))}")
    print(f"filas de metas: {len(metas)} | fuentes: {dict(Counter(m['fuente'] for m in metas))}")
    print(f"salida en: {a.salida}")


if __name__ == "__main__":
    main()
