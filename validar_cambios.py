import os
import csv
import argparse
import numpy as np
from comparar_etiquetas import NIVELES, leer, num, kappa

CAMBIOS = [("t_incipiente", "origen_t_incipiente", "a incipiente"),
           ("t_incontrolado", "origen_t_incontrolado", "a incontrolado")]
ORIGENES = [("pasada2", "pasada 2 (tira densa)"),
            ("pasada1", "pasada 1 (fotos separadas 1 s o menos)"),
            ("pasada1; pasada2 no ve el cambio", "pasada 1 (la pasada 2 no ve el cambio)")]
TOLERANCIAS = [1.0, 2.0, 5.0]
CLAVES = ["ambos", "solo_ia", "solo_h", "ninguno"]


def leer_lista(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [r["archivo"] for r in csv.DictReader(f)]


def leer_duraciones(ruta):
    with open(ruta, encoding="utf-8") as f:
        return {r["archivo"]: float(r["duracion_s"]) for r in csv.DictReader(f)}


def leer_tiempos(ruta, anotador):
    por_video = {}
    if not os.path.exists(ruta):
        return por_video
    with open(ruta, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["anotador"] != anotador:
                continue
            d = por_video.setdefault(r["archivo"], {"segundos": 0.0, "reproducciones": 0, "fecha": r["fecha"]})
            d["segundos"] += float(r["segundos_abierto"])
            d["reproducciones"] += int(r["reproducciones"])
            d["fecha"] = r["fecha"]
    return por_video


def hay_cambio(r):
    return any(num(r[campo]) is not None for campo, _, _ in CAMBIOS)


def texto_cambios(r):
    partes = [f"{nombre} en {num(r[campo]):.2f} s" for campo, _, nombre in CAMBIOS if num(r[campo]) is not None]
    return ", ".join(partes) if partes else "sin cambio"


def clave_par(h, i):
    if h and i:
        return "ambos"
    if i:
        return "solo_ia"
    if h:
        return "solo_h"
    return "ninguno"


def valor(v, formato="+.2f"):
    return "-" if v is None else format(v, formato)


def fila_error(nombre, errores, seps=None):
    if not errores:
        return f"| {nombre} | 0 | - | - | - | - | - | - | - |"
    e = np.array(errores)
    ab = np.abs(e)
    dentro = " | ".join(str(int((ab <= t).sum())) for t in TOLERANCIAS)
    if seps is None:
        ultimo = int((e < 0).sum())
    else:
        ultimo = int((ab <= np.array(seps) + 1e-6).sum())
    return f"| {nombre} | {len(e)} | {np.median(ab):.2f} | {ab.mean():.2f} | {np.median(e):+.2f} | {dentro} | {ultimo} |"


def linea_error(nombre, errores):
    if not errores:
        return f"{nombre}: sin instantes"
    e = np.array(errores)
    ab = np.abs(e)
    return (f"{nombre}: n {len(e)} | mediana abs {np.median(ab):.2f} s | media abs {ab.mean():.2f} s"
            f" | mediana con signo {np.median(e):+.2f} s | dentro de 1 s {int((ab <= 1).sum())},"
            f" 2 s {int((ab <= 2).sum())}, 5 s {int((ab <= 5).sum())} | negativos {int((e < 0).sum())}")


def seccion_nivel(hum, ia, videos, titulo, lineas):
    h = [NIVELES.index(hum[v]["nivel_inicial"]) for v in videos]
    a = [NIVELES.index(ia[v]["nivel_inicial"]) for v in videos]
    iguales = sum(x == y for x, y in zip(h, a))
    kl, m = kappa(h, a, 3, "lineal")
    lineas.append(f"**{titulo}:** acuerdo {iguales} de {len(videos)} ({100.0 * iguales / len(videos):.1f} %),"
                  f" kappa ponderado lineal {kl:.3f}.")
    lineas.append("")
    lineas.append("| humano / IA | " + " | ".join(NIVELES) + " | total |")
    lineas.append("|---|" + "---|" * 4)
    for i, nv in enumerate(NIVELES):
        lineas.append(f"| {nv} | " + " | ".join(str(int(x)) for x in m[i]) + f" | {int(m[i].sum())} |")
    lineas.append("| total | " + " | ".join(str(int(x)) for x in m.sum(0)) + f" | {int(m.sum())} |")
    lineas.append("")
    return iguales, kl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--etiquetas", default="etiquetas.csv")
    p.add_argument("--anotador", default="aleCa_cambios")
    p.add_argument("--ia", default=os.path.join("resultados", "etiquetas_ia_v2.csv"))
    p.add_argument("--lista", default=os.path.join("resultados", "prueba_cambios.csv"))
    p.add_argument("--tiempos", default=os.path.join("resultados", "tiempos_etiquetado.csv"))
    p.add_argument("--metadatos", default="metadatos_videos.csv")
    p.add_argument("--pausa_s", type=float, default=1000.0)
    p.add_argument("--salida", default=os.path.join("resultados", "validacion_cambios.md"))
    a = p.parse_args()

    lista = leer_lista(a.lista)
    hum = leer(a.etiquetas, a.anotador)
    ia = leer(a.ia)
    duracion = leer_duraciones(a.metadatos)
    tiempos = leer_tiempos(a.tiempos, a.anotador)
    faltan = [v for v in lista if v not in hum or v not in ia]
    if faltan:
        print(f"faltan etiquetas de {len(faltan)} videos de la lista; no se compara")
        return
    sin_nivel = [v for v in lista if hum[v]["nivel_inicial"] not in NIVELES or ia[v]["nivel_inicial"] not in NIVELES]
    desc_h = [v for v in lista if hum[v]["descartar"] == "1"]
    desc_ia = [v for v in lista if ia[v]["descartar"] == "1"]
    con_nivel = [v for v in lista if v not in sin_nivel]
    validos = [v for v in con_nivel if v not in desc_h and v not in desc_ia]

    lineas = ["# Validacion humana de los cambios de nivel (paso 8)", ""]
    lineas.append(f"- Lista: `{a.lista.replace(os.sep, '/')}` ({len(lista)} videos). Anotador humano: `{a.anotador}`,"
                  f" a ciegas con `etiquetar_v3.py --ciego`.")
    lineas.append(f"- Etiquetas de la IA: `{a.ia.replace(os.sep, '/')}`.")
    lineas.append(f"- Descartados por el humano: {len(desc_h)}. Descartados por la IA: {len(desc_ia)}."
                  f" Comparables en cambios, instantes y tramo util: {len(validos)}.")
    lineas.append("- Este archivo lo genera `python validar_cambios.py`. No modifica ninguna etiqueta.")
    lineas.append("- Error de un instante = instante de la IA menos instante del humano, en segundos."
                  " Negativo: la IA marca el cambio antes que el humano.")
    lineas.append("")

    lineas.append("## 1. Nivel inicial")
    lineas.append("")
    ig_todos, k_todos = seccion_nivel(hum, ia, con_nivel, f"Los {len(con_nivel)} videos", lineas)
    ig_val, k_val = ig_todos, k_todos
    if len(validos) != len(con_nivel):
        ig_val, k_val = seccion_nivel(hum, ia, validos, f"Sin los descartados ({len(validos)} videos)", lineas)
    distintos = [v for v in con_nivel if hum[v]["nivel_inicial"] != ia[v]["nivel_inicial"]]
    if distintos:
        lineas.append("Videos con distinto nivel inicial:")
        lineas.append("")
        lineas.append("| video | humano | IA | motivo de la IA |")
        lineas.append("|---|---|---|---|")
        for v in distintos:
            lineas.append(f"| {v} | {hum[v]['nivel_inicial']} | {ia[v]['nivel_inicial']} | {ia[v].get('motivo', '')} |")
        lineas.append("")

    lineas.append("## 2. Deteccion de cambios")
    lineas.append("")
    lineas.append(f"Sobre los {len(validos)} videos no descartados.")
    lineas.append("")
    lineas.append("| cambio | ambos | solo la IA | solo el humano | ninguno |")
    lineas.append("|---|---|---|---|---|")
    grupos = {}
    for campo, _, nombre in CAMBIOS:
        g = {k: [] for k in CLAVES}
        for v in validos:
            g[clave_par(num(hum[v][campo]) is not None, num(ia[v][campo]) is not None)].append(v)
        grupos[campo] = g
        lineas.append(f"| {nombre} | " + " | ".join(str(len(g[k])) for k in CLAVES) + " |")
    alg = {k: [] for k in CLAVES}
    for v in validos:
        alg[clave_par(hay_cambio(hum[v]), hay_cambio(ia[v]))].append(v)
    lineas.append("| algun cambio en el video | " + " | ".join(str(len(alg[k])) for k in CLAVES) + " |")
    lineas.append("")
    g_inc = grupos["t_incipiente"]
    ya_inc = [v for v in g_inc["solo_ia"] if hum[v]["nivel_inicial"] != "controlado"]
    lineas.append(f"- De los {len(g_inc['solo_ia'])} cambios a incipiente que marca solo la IA, en {len(ya_inc)} el humano"
                  f" puso un nivel inicial de incipiente o mas (para el humano el video ya empieza en ese nivel) y en"
                  f" {len(g_inc['solo_ia']) - len(ya_inc)} el humano dejo el video en controlado.")
    con_ia = [v for v in validos if hay_cambio(ia[v])]
    lineas.append(f"- De los {len(con_ia)} videos no descartados con cambio segun la IA, el humano marca algun cambio en"
                  f" {len(alg['ambos'])} y ninguno en {len(alg['solo_ia'])}.")
    sin_ia = [v for v in lista if not hay_cambio(ia[v])]
    sin_ia_h = [v for v in sin_ia if hay_cambio(hum[v]) and v not in desc_h]
    sin_ia_niv = [v for v in sin_ia if hum[v]["nivel_inicial"] == ia[v]["nivel_inicial"]]
    lineas.append(f"- **Videos sin cambio segun la IA ({len(sin_ia)}):** el humano marca cambio en {len(sin_ia_h)}"
                  f" y coincide en el nivel inicial en {len(sin_ia_niv)}.")
    lineas.append("")
    lineas.append("| video sin cambio segun la IA | nivel IA | nivel humano | cambios del humano |")
    lineas.append("|---|---|---|---|")
    for v in sin_ia:
        extra = " (descartado)" if v in desc_h else ""
        lineas.append(f"| {v} | {ia[v]['nivel_inicial']} | {hum[v]['nivel_inicial']}{extra} | {texto_cambios(hum[v])} |")
    lineas.append("")
    for campo, _, nombre in CAMBIOS:
        for clave, quien in [("solo_ia", "solo la IA"), ("solo_h", "solo el humano")]:
            if grupos[campo][clave]:
                lineas.append(f"Cambio {nombre} marcado por {quien}:")
                lineas.append("")
                lineas.append("| video | humano | IA |")
                lineas.append("|---|---|---|")
                for v in grupos[campo][clave]:
                    lineas.append(f"| {v} | {hum[v]['nivel_inicial']}; {texto_cambios(hum[v])} |"
                                  f" {ia[v]['nivel_inicial']}; {texto_cambios(ia[v])} |")
                lineas.append("")
    if desc_h:
        lineas.append("Descartados por el humano (fuera de las cuentas de cambios, instantes y tramo util):")
        lineas.append("")
        lineas.append("| video | IA |")
        lineas.append("|---|---|")
        for v in desc_h:
            lineas.append(f"| {v} | {ia[v]['nivel_inicial']}; {texto_cambios(ia[v])} |")
        lineas.append("")

    lineas.append("## 3. Error en segundos cuando ambos marcan el mismo cambio")
    lineas.append("")
    instantes = []
    for campo, col_origen, nombre in CAMBIOS:
        for v in grupos[campo]["ambos"]:
            th, ta = num(hum[v][campo]), num(ia[v][campo])
            instantes.append({"video": v, "cambio": nombre, "h": th, "ia": ta, "e": ta - th,
                              "origen": ia[v][col_origen], "sep": num(ia[v]["sep_p1_s"]), "dur": duracion[v]})
    grupos_e = [("todos", [x["e"] for x in instantes])]
    grupos_e += [(nombre, [x["e"] for x in instantes if x["cambio"] == nombre]) for _, _, nombre in CAMBIOS]
    grupos_e += [(f"origen: {nombre}", [x["e"] for x in instantes if x["origen"] == clave]) for clave, nombre in ORIGENES]
    otros = [x["e"] for x in instantes if x["origen"] not in [c for c, _ in ORIGENES]]
    if otros:
        grupos_e.append(("origen: otro", otros))
    lineas.append("| grupo | instantes | mediana del error absoluto (s) | media del error absoluto (s) |"
                  " mediana del error con signo (s) | dentro de 1 s | dentro de 2 s | dentro de 5 s | la IA marca antes |")
    lineas.append("|---|---|---|---|---|---|---|---|---|")
    for nombre, e in grupos_e:
        lineas.append(fila_error(nombre, e))
    lineas.append("")
    for clave, nombre in ORIGENES:
        d = [x["dur"] for x in instantes if x["origen"] == clave]
        if d:
            lineas.append(f"- Duracion de los videos con origen {nombre}: mediana {np.median(d):.0f} s"
                          f" (de {min(d):.0f} a {max(d):.0f} s).")
    lineas.append("")
    lineas.append("| video | cambio | humano (s) | IA (s) | error (s) | origen del instante de la IA |"
                  " separacion de fotos en la pasada 1 (s) | duracion (s) |")
    lineas.append("|---|---|---|---|---|---|---|---|")
    for x in sorted(instantes, key=lambda x: abs(x["e"])):
        lineas.append(f"| {x['video']} | {x['cambio']} | {x['h']:.2f} | {x['ia']:.2f} | {x['e']:+.2f} | {x['origen']} |"
                      f" {x['sep']:.2f} | {x['dur']:.1f} |")
    lineas.append("")

    lineas.append("## 4. Tramo util")
    lineas.append("")
    lineas.append("Diferencia = instante del humano menos instante de la IA, en los videos no descartados en que ambos"
                  " lo marcaron. La IA marca el tramo util con las fotos de la pasada 1, asi que su resolucion es la"
                  " separacion entre fotos.")
    lineas.append("")
    lineas.append("| extremo | videos | mediana de la diferencia absoluta (s) | media de la diferencia absoluta (s) |"
                  " mediana con signo (s) | dentro de 1 s | dentro de 2 s | dentro de 5 s | dentro de una foto de la pasada 1 |")
    lineas.append("|---|---|---|---|---|---|---|---|---|")
    util = {}
    for campo, nombre in [("t_inicio_util", "inicio"), ("t_fin_util", "fin")]:
        pares = [(v, num(hum[v][campo]), num(ia[v][campo])) for v in validos]
        pares = [(v, h, i) for v, h, i in pares if h is not None and i is not None]
        util[campo] = {v: h - i for v, h, i in pares}
        lineas.append(fila_error(nombre, [h - i for _, h, i in pares], [num(ia[v]["sep_p1_s"]) for v, _, _ in pares]))
    lineas.append("")
    vac_ini = [v for v in validos if num(hum[v]["t_inicio_util"]) is None]
    vac_fin = [v for v in validos if num(hum[v]["t_fin_util"]) is None]
    lineas.append(f"- El humano dejo vacio el inicio util en {len(vac_ini)} videos y el fin util en {len(vac_fin)}"
                  f" (de {len(validos)}); esos no entran en la tabla.")
    lineas.append("")
    lineas.append("| video | inicio humano | inicio IA | diferencia | fin humano | fin IA | diferencia | duracion (s) |")
    lineas.append("|---|---|---|---|---|---|---|---|")
    for v in validos:
        lineas.append(f"| {v} | {hum[v]['t_inicio_util'] or '-'} | {ia[v]['t_inicio_util'] or '-'} |"
                      f" {valor(util['t_inicio_util'].get(v))} | {hum[v]['t_fin_util'] or '-'} | {ia[v]['t_fin_util'] or '-'} |"
                      f" {valor(util['t_fin_util'].get(v))} | {duracion[v]:.1f} |")
    lineas.append("")

    lineas.append("## 5. Tiempo de etiquetado")
    lineas.append("")
    con_t = [v for v in lista if v in tiempos]
    resumen_t = None
    if con_t:
        seg = {v: tiempos[v]["segundos"] for v in con_t}
        pausas = [v for v in con_t if seg[v] > a.pausa_s]
        normales = [v for v in con_t if v not in pausas]
        total = sum(seg.values())
        s_norm = np.array([seg[v] for v in normales])
        d_norm = np.array([duracion[v] for v in normales])
        menos = int((s_norm < d_norm).sum())
        una = sum(1 for v in con_t if tiempos[v]["reproducciones"] <= 1)
        fechas = sorted(tiempos[v]["fecha"] for v in con_t)
        lineas.append(f"- Videos con registro: {len(con_t)} de {len(lista)}.")
        lineas.append(f"- Tiempo total con la ventana abierta: {total / 60:.1f} min ({total:.0f} s).")
        lineas.append(f"- {len(pausas)} videos pasan de {a.pausa_s:.0f} s abiertos"
                      f" ({', '.join(f'{seg[v]:.0f} s' for v in pausas)}); se toman como pausas con la ventana abierta.")
        lineas.append(f"- **Sin esos {len(pausas)}: {s_norm.sum() / 60:.1f} min en {len(normales)} videos**"
                      f" (mediana {np.median(s_norm):.1f} s, media {s_norm.mean():.1f} s, minimo {s_norm.min():.1f} s,"
                      f" maximo {s_norm.max():.1f} s por video).")
        lineas.append(f"- Duracion de esos {len(normales)} videos: {d_norm.sum() / 60:.1f} min. Tiempo abierto entre"
                      f" duracion del video: mediana {np.median(s_norm / d_norm):.2f}.")
        lineas.append(f"- **En {menos} de {len(normales)} videos la ventana estuvo abierta menos tiempo que la duracion del"
                      f" video**, es decir, no se vio el video completo a velocidad normal (se avanzo con saltos o con la"
                      f" linea de tiempo).")
        lineas.append(f"- Videos con una sola reproduccion o ninguna: {una} de {len(con_t)}.")
        lineas.append(f"- Primer guardado: {fechas[0]}. Ultimo guardado: {fechas[-1]}.")
        lineas.append("")
        lineas.append("| video | duracion (s) | abierto (s) | abierto / duracion | reproducciones | cambios del humano |")
        lineas.append("|---|---|---|---|---|---|")
        for v in sorted(con_t, key=lambda v: tiempos[v]["fecha"]):
            marca = " (pausa)" if v in pausas else ""
            lineas.append(f"| {v} | {duracion[v]:.1f} | {seg[v]:.1f}{marca} | {seg[v] / duracion[v]:.2f} |"
                          f" {tiempos[v]['reproducciones']} | {texto_cambios(hum[v])} |")
        lineas.append("")
        resumen_t = (total, float(s_norm.sum()), float(np.median(s_norm)), menos, len(normales), len(pausas))
    else:
        lineas.append("- No hay registro de tiempos para este anotador.")
        lineas.append("")

    with open(a.salida, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))

    print(f"videos {len(lista)} | descartados por el humano {len(desc_h)} | comparables {len(validos)}")
    print(f"nivel inicial: acuerdo {ig_todos} de {len(con_nivel)}, kappa lineal {k_todos:.3f}"
          f" | sin descartados {ig_val} de {len(validos)}, kappa lineal {k_val:.3f}")
    for campo, _, nombre in CAMBIOS:
        g = grupos[campo]
        print(f"{nombre}: ambos {len(g['ambos'])} | solo ia {len(g['solo_ia'])} | solo humano {len(g['solo_h'])}"
              f" | ninguno {len(g['ninguno'])}")
    print(f"algun cambio: ambos {len(alg['ambos'])} | solo ia {len(alg['solo_ia'])} | solo humano {len(alg['solo_h'])}"
          f" | ninguno {len(alg['ninguno'])}")
    print(f"sin cambio segun la ia: {len(sin_ia)} | con cambio del humano {len(sin_ia_h)}")
    for nombre, e in grupos_e:
        print(linea_error(f"error {nombre}", e))
    for campo, nombre in [("t_inicio_util", "inicio util"), ("t_fin_util", "fin util")]:
        print(linea_error(nombre, list(util[campo].values())))
    if resumen_t:
        print(f"tiempo: total {resumen_t[0] / 60:.1f} min | sin {resumen_t[5]} pausas {resumen_t[1] / 60:.1f} min"
              f" | mediana {resumen_t[2]:.1f} s | abierto menos que la duracion en {resumen_t[3]} de {resumen_t[4]}")
    print(f"reporte en: {a.salida}")


if __name__ == "__main__":
    main()
