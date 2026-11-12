import os
import re
import csv
import argparse
import cv2
import numpy as np
from comparar_etiquetas import NIVELES, leer, num
from etiquetar import estado_desde, nivel_en
from etiquetar_v3 import Video, COLOR_H, COLOR_IA, resumen, sin_tildes

CAMBIOS = [("t_incipiente", "origen_t_incipiente", "a incipiente"),
           ("t_incontrolado", "origen_t_incontrolado", "a incontrolado")]
NEUTRO = (200, 200, 200)
SEP, ALTO_PIE, ALTO_TITULO, ALTO_CABECERA = 8, 30, 34, 124
FUENTE = cv2.FONT_HERSHEY_SIMPLEX
CAMPOS_LISTA = ["archivo", "orden", "inicial_humano", "inicial_ia", "t_incipiente_humano", "t_incipiente_ia",
                "t_incontrolado_humano", "t_incontrolado_ia", "mayor_diferencia_s", "desacuerdos", "hoja"]


def comparar(h, i, umbral):
    d = {"nivel": h["nivel_inicial"] != i["nivel_inicial"], "cambios": [], "mayor": None}
    for campo, col_origen, nombre in CAMBIOS:
        th, ti = num(h[campo]), num(i[campo])
        c = {"campo": campo, "nombre": nombre, "h": th, "ia": ti, "dif": None, "tipo": "ninguno",
             "origen": i.get(col_origen, "")}
        if th is not None and ti is not None:
            c["dif"] = ti - th
            c["tipo"] = "instante" if abs(c["dif"]) > umbral else "coincide"
            d["mayor"] = abs(c["dif"]) if d["mayor"] is None else max(d["mayor"], abs(c["dif"]))
        elif ti is not None:
            c["tipo"] = "solo_ia"
        elif th is not None:
            c["tipo"] = "solo_h"
        d["cambios"].append(c)
    tipos = [c["tipo"] for c in d["cambios"]]
    d["solos"] = sum(t in ("solo_ia", "solo_h") for t in tipos)
    d["desacuerdo"] = d["nivel"] or d["solos"] > 0 or "instante" in tipos
    if "instante" in tipos:
        d["bloque"] = 0
    elif d["solos"] > 0:
        d["bloque"] = 1
    elif d["nivel"]:
        d["bloque"] = 2
    else:
        d["bloque"] = 3
    return d


def texto_desacuerdos(d):
    partes = ["nivel inicial"] if d["nivel"] else []
    for c in d["cambios"]:
        if c["tipo"] == "instante":
            partes.append(f"{c['nombre']}: {c['dif']:+.2f} s")
        elif c["tipo"] == "solo_ia":
            partes.append(f"{c['nombre']}: solo la IA")
        elif c["tipo"] == "solo_h":
            partes.append(f"{c['nombre']}: solo el humano")
    return "; ".join(partes) if partes else "ninguno"


def nombre_nivel(n):
    if n is None:
        return "fuera de su tramo util"
    return "sin nivel" if n == -1 else NIVELES[n]


def filas_hoja(d, h, i, dur):
    est_h, est_ia = estado_desde(h), estado_desde(i)
    inicios = [x for x in (est_h["t_inicio_util"], est_ia["t_inicio_util"]) if x is not None]
    fines = [x for x in (est_h["t_fin_util"], est_ia["t_fin_util"]) if x is not None]
    t_min = min(inicios) if inicios else 0.0
    t_comun = max(inicios) if inicios else 0.0
    t_fin = min(max(fines) if fines else dur, dur)
    filas = []
    if d["nivel"]:
        marcas = [x for c in d["cambios"] for x in (c["h"], c["ia"]) if x is not None and x > t_comun + 0.5]
        limite = min(marcas) if marcas else t_fin
        hasta = "antes del primer cambio" if marcas else "sin cambios marcados"
        puntos = [(t_comun + f * (limite - t_comun), "inicio del tramo util" if f == 0 else hasta, NEUTRO)
                  for f in (0.0, 0.3, 0.6, 0.9)]
        filas.append((f"nivel inicial: humano {h['nivel_inicial']} | ia {i['nivel_inicial']}"
                      f" | fotogramas desde {t_comun:.2f} s hasta {limite:.2f} s", puntos))
    for c in d["cambios"]:
        if c["tipo"] == "instante":
            pares = sorted([(c["h"], f"humano: {c['nombre']}", COLOR_H), (c["ia"], f"ia: {c['nombre']}", COLOR_IA)])
            a, b = pares[0][0], pares[1][0]
            puntos = [pares[0], (a + (b - a) / 3.0, "intermedio", NEUTRO), (a + 2.0 * (b - a) / 3.0, "intermedio", NEUTRO), pares[1]]
            filas.append((f"cambio {c['nombre']}: humano {c['h']:.2f} s | ia {c['ia']:.2f} s | ia - humano = {c['dif']:+.2f} s"
                          f" | origen del instante de la ia: {c['origen']}", puntos))
        elif c["tipo"] in ("solo_ia", "solo_h"):
            de_ia = c["tipo"] == "solo_ia"
            t = c["ia"] if de_ia else c["h"]
            quien, otro = ("ia", "humano") if de_ia else ("humano", "ia")
            color = COLOR_IA if de_ia else COLOR_H
            nivel_otro = nombre_nivel(nivel_en(t, est_h if de_ia else est_ia))
            puntos = [((t_min + t) / 2.0, "antes", NEUTRO), (t, f"{quien}: {c['nombre']}", color),
                      ((t + t_fin) / 2.0, "despues", NEUTRO), (max(t, t_fin - 0.2), "final del tramo util", NEUTRO)]
            origen = f" | origen del instante de la ia: {c['origen']}" if de_ia else ""
            filas.append((f"cambio {c['nombre']}: solo {quien} en {t:.2f} s | en ese instante {otro}: {nivel_otro}{origen}", puntos))
    return filas


def celda(video, t, etiqueta, color, ancho, alto):
    video.ir(t)
    img = np.zeros((alto, ancho, 3), np.uint8)
    if video.frame is not None:
        h, w = video.frame.shape[:2]
        esc = min(ancho / w, alto / h)
        f = cv2.resize(video.frame, (max(1, int(w * esc)), max(1, int(h * esc))), interpolation=cv2.INTER_AREA)
        y0, x0 = (alto - f.shape[0]) // 2, (ancho - f.shape[1]) // 2
        img[y0:y0 + f.shape[0], x0:x0 + f.shape[1]] = f
    cv2.rectangle(img, (1, 1), (ancho - 2, alto - 2), color, 3)
    pie = np.full((ALTO_PIE, ancho, 3), 30, np.uint8)
    cv2.putText(pie, f"t = {video.t:.2f} s | {etiqueta}", (6, 21), FUENTE, 0.6, color, 1, cv2.LINE_AA)
    return np.vstack([pie, img])


def hoja(ruta_video, nombre, h, i, filas, ancho):
    video = Video(ruta_video)
    alto_f, ancho_f = video.frame.shape[:2]
    alto = int(min(ancho * alto_f / ancho_f, ancho))
    total = 4 * ancho + 5 * SEP
    cab = np.full((ALTO_CABECERA, total, 3), 30, np.uint8)
    cv2.putText(cab, f"{sin_tildes(nombre)} | duracion {video.dur:.1f} s", (SEP, 26), FUENTE, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(cab, resumen("humano", estado_desde(h)), (SEP, 56), FUENTE, 0.65, COLOR_H, 1, cv2.LINE_AA)
    cv2.putText(cab, resumen("ia", estado_desde(i)), (SEP, 84), FUENTE, 0.65, COLOR_IA, 1, cv2.LINE_AA)
    motivo = f"motivo de la ia: {sin_tildes(i.get('motivo', ''))}"
    cv2.putText(cab, motivo[:total // 10], (SEP, 112), FUENTE, 0.55, NEUTRO, 1, cv2.LINE_AA)
    partes = [cab]
    for titulo, puntos in filas:
        barra = np.full((ALTO_TITULO, total, 3), 60, np.uint8)
        cv2.putText(barra, sin_tildes(titulo)[:total // 11], (SEP, 23), FUENTE, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        fila = np.full((ALTO_PIE + alto + SEP, total, 3), 15, np.uint8)
        for n, (t, etiqueta, color) in enumerate(puntos):
            x0 = SEP + n * (ancho + SEP)
            fila[:ALTO_PIE + alto, x0:x0 + ancho] = celda(video, t, etiqueta, color, ancho, alto)
        partes += [barra, fila]
    video.cerrar()
    return np.vstack(partes)


def nombre_hoja(orden, nombre):
    base = re.sub(r"[^A-Za-z0-9]+", "_", sin_tildes(os.path.splitext(nombre)[0])).strip("_")
    return f"{orden:02d}_{base[:80]}.jpg"


def t_txt(v):
    return "-" if v is None else f"{v:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--etiquetas", default="etiquetas.csv")
    p.add_argument("--anotador", default="aleCa_cambios")
    p.add_argument("--ia", default=os.path.join("resultados", "etiquetas_ia_v2.csv"))
    p.add_argument("--lista", default=os.path.join("resultados", "prueba_cambios.csv"))
    p.add_argument("--videos", default=os.path.join("real_video", "real_video", "fire"))
    p.add_argument("--umbral_s", type=float, default=5.0)
    p.add_argument("--ancho_celda", type=int, default=480)
    p.add_argument("--hojas", default=os.path.join("runs_fuego", "revision_cambios"))
    p.add_argument("--tabla", default=os.path.join("resultados", "revision_cambios.md"))
    p.add_argument("--lista_salida", default=os.path.join("resultados", "revision_cambios.csv"))
    p.add_argument("--sin_hojas", action="store_true")
    a = p.parse_args()

    with open(a.lista, encoding="utf-8") as f:
        lista = [r["archivo"] for r in csv.DictReader(f)]
    hum = leer(a.etiquetas, a.anotador)
    ia = leer(a.ia)
    faltan = [v for v in lista if v not in hum or v not in ia]
    if faltan:
        raise SystemExit(f"faltan etiquetas de {len(faltan)} videos de la lista; no se compara")
    descartados = [v for v in lista if hum[v]["descartar"] == "1" or ia[v]["descartar"] == "1"]
    validos = [v for v in lista if v not in descartados]
    comp = {v: comparar(hum[v], ia[v], a.umbral_s) for v in validos}
    orden = sorted(validos, key=lambda v: (comp[v]["bloque"], -(comp[v]["mayor"] or 0.0), -comp[v]["solos"], lista.index(v)))
    con_dif = [v for v in orden if comp[v]["desacuerdo"]]

    hojas = {}
    if not a.sin_hojas:
        rutas = {}
        for raiz, _, archivos in os.walk(a.videos):
            for n in archivos:
                rutas.setdefault(n, os.path.join(raiz, n))
        sin_video = [v for v in con_dif if v not in rutas]
        if sin_video:
            raise SystemExit(f"videos no encontrados en {a.videos}: {len(sin_video)}")
        os.makedirs(a.hojas, exist_ok=True)
        for n, v in enumerate(con_dif, 1):
            video = Video(rutas[v])
            dur = video.dur
            video.cerrar()
            img = hoja(rutas[v], v, hum[v], ia[v], filas_hoja(comp[v], hum[v], ia[v], dur), a.ancho_celda)
            hojas[v] = nombre_hoja(n, v)
            ok, datos = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise SystemExit(f"no se pudo codificar la hoja {n}")
            datos.tofile(os.path.join(a.hojas, hojas[v]))
            print(f"hoja {n} de {len(con_dif)} lista", flush=True)

    filas_csv = []
    lineas = ["# Revision de las diferencias entre el humano y la IA (prueba de cambios)", ""]
    lineas.append(f"- Lista: `{a.lista.replace(os.sep, '/')}`. Anotador humano: `{a.anotador}`."
                  f" Etiquetas de la IA: `{a.ia.replace(os.sep, '/')}`.")
    lineas.append(f"- Videos comparados: {len(validos)} (descartados: {len(descartados)}). Con alguna diferencia: {len(con_dif)}.")
    lineas.append(f"- Diferencia: nivel inicial distinto, un cambio que marca solo uno, o un instante con mas de"
                  f" {a.umbral_s:g} s de diferencia.")
    lineas.append("- Orden: primero los videos con un instante que difiere mas del umbral, de la mayor diferencia a la menor;"
                  " luego los que tienen un cambio marcado por uno solo; luego los que solo difieren en el nivel inicial;"
                  " al final los que coinciden.")
    lineas.append("- Diferencia de un instante = instante de la IA menos instante del humano, en segundos.")
    lineas.append(f"- Hojas de comparacion (no versionadas): `{a.hojas.replace(os.sep, '/')}/`."
                  " Este archivo lo genera `python revisar_cambios.py`. No modifica ninguna etiqueta.")
    lineas.append("")
    lineas.append("| n | video | inicial humano | inicial IA | a incipiente humano (s) | a incipiente IA (s) |"
                  " a incontrolado humano (s) | a incontrolado IA (s) | mayor diferencia (s) | diferencias |")
    lineas.append("|---|---|---|---|---|---|---|---|---|---|")
    for n, v in enumerate(orden, 1):
        d = comp[v]
        inc, unc = d["cambios"]
        mayor = "-" if d["mayor"] is None else f"{d['mayor']:.2f}"
        lineas.append(f"| {n} | {v} | {hum[v]['nivel_inicial']} | {ia[v]['nivel_inicial']} | {t_txt(inc['h'])} | {t_txt(inc['ia'])} |"
                      f" {t_txt(unc['h'])} | {t_txt(unc['ia'])} | {mayor} | {texto_desacuerdos(d)} |")
        if d["desacuerdo"]:
            filas_csv.append({"archivo": v, "orden": n, "inicial_humano": hum[v]["nivel_inicial"],
                              "inicial_ia": ia[v]["nivel_inicial"], "t_incipiente_humano": hum[v]["t_incipiente"],
                              "t_incipiente_ia": ia[v]["t_incipiente"], "t_incontrolado_humano": hum[v]["t_incontrolado"],
                              "t_incontrolado_ia": ia[v]["t_incontrolado"], "mayor_diferencia_s": "" if d["mayor"] is None else round(d["mayor"], 2),
                              "desacuerdos": texto_desacuerdos(d), "hoja": hojas.get(v, nombre_hoja(n, v))})
    lineas.append("")
    if descartados:
        lineas.append("Descartados (fuera de la comparacion): " + ", ".join(descartados) + ".")
        lineas.append("")
    os.makedirs(os.path.dirname(a.tabla) or ".", exist_ok=True)
    with open(a.tabla, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    with open(a.lista_salida, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_LISTA)
        w.writeheader()
        w.writerows(filas_csv)

    bloques = [sum(comp[v]["bloque"] == b for v in validos) for b in range(4)]
    print(f"videos comparados: {len(validos)} | descartados: {len(descartados)} | con alguna diferencia: {len(con_dif)}")
    print(f"con un instante que difiere mas de {a.umbral_s:g} s: {bloques[0]} | solo cambios marcados por uno: {bloques[1]}"
          f" | solo nivel inicial: {bloques[2]} | coinciden: {bloques[3]}")
    print(f"nivel inicial distinto: {sum(comp[v]['nivel'] for v in validos)}"
          f" | cambios marcados por uno solo: {sum(comp[v]['solos'] for v in validos)}"
          f" | instantes con mas de {a.umbral_s:g} s: {sum(c['tipo'] == 'instante' for v in validos for c in comp[v]['cambios'])}")
    if not a.sin_hojas:
        print(f"hojas: {len(hojas)} en {a.hojas}")
    print(f"tabla en {a.tabla} | lista para --revisar en {a.lista_salida}")


if __name__ == "__main__":
    main()
