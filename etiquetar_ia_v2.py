import os
import csv
import random
import argparse
import cv2
import numpy as np
from duplicados import cargar_excluidos

COLUMNAS = 6
POR_IMAGEN = 24


def leer_archivos(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [r["archivo"] for r in csv.DictReader(f)]


def candidatos(rutas_propuestas):
    c = []
    for ruta in rutas_propuestas:
        with open(ruta, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("t_incipiente") or r.get("t_incontrolado"):
                    c.append(r["archivo"])
    return sorted(set(c))


def lista_v2(a):
    prueba = set(leer_archivos(a.prueba))
    excluidos = cargar_excluidos(a.excluidos)
    todos = candidatos(a.propuestas)
    lista = [v for v in todos if v not in prueba and v not in excluidos]
    return todos, lista


def buscar_rutas(carpeta):
    rutas = {}
    for raiz, _, archivos in os.walk(carpeta):
        for n in archivos:
            rutas.setdefault(n, os.path.join(raiz, n))
    return rutas


def guardar_jpg(ruta, img, calidad=85):
    ok, datos = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, calidad])
    if not ok:
        raise IOError(f"no se pudo codificar: {ruta}")
    with open(ruta, "wb") as f:
        f.write(datos.tobytes())


def poner_texto(img, texto, y):
    cv2.putText(img, texto, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, texto, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def capturar(ruta, objetivos):
    cap = cv2.VideoCapture(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pendientes = sorted(set(objetivos))
    capturas, idx, j = [], 0, 0
    while j < len(pendientes):
        if not cap.grab():
            break
        if idx == pendientes[j]:
            ok, frame = cap.retrieve()
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = t_ms / 1000.0 if t_ms > 0 or idx == 0 else idx / fps
            if ok:
                capturas.append((idx, max(0.0, t), frame))
            j += 1
        idx += 1
    while cap.grab():
        idx += 1
    cap.release()
    return capturas, idx


def armar_imagen(capturas, primera, ancho_foto):
    alto_foto = int(round(capturas[0][2].shape[0] * ancho_foto / capturas[0][2].shape[1]))
    fotos = []
    for k, (_, t, frame) in enumerate(capturas, primera):
        img = cv2.resize(frame, (ancho_foto, alto_foto), interpolation=cv2.INTER_AREA)
        poner_texto(img, f"{k}", 22)
        poner_texto(img, f"{t:.1f} s", alto_foto - 8)
        fotos.append(img)
    while len(fotos) % COLUMNAS:
        fotos.append(np.zeros_like(fotos[0]))
    return np.vstack([np.hstack(fotos[r:r + COLUMNAS]) for r in range(0, len(fotos), COLUMNAS)])


def tiras(a):
    todos, lista = lista_v2(a)
    rutas = buscar_rutas(a.videos)
    os.makedirs(a.tiras, exist_ok=True)
    ruta_t = os.path.join(a.tiras, "tiempos.csv")
    hechos = set(leer_archivos(ruta_t)) if os.path.exists(ruta_t) else set()
    nuevo = not os.path.exists(ruta_t)
    print(f"candidatos {len(todos)} | a etiquetar {len(lista)} | ya con tira {len(hechos & set(lista))}")
    with open(ruta_t, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["archivo", "foto", "frame", "t_s"])
        for i, nombre in enumerate(lista, 1):
            if nombre in hechos:
                continue
            if nombre not in rutas:
                print(f"no encontrado: {nombre}")
                continue
            cap = cv2.VideoCapture(rutas[nombre])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            capturas, leidos = capturar(rutas[nombre], np.linspace(0, max(total - 1, 0), a.fotos).astype(int).tolist())
            if len(capturas) < a.fotos and leidos > 0:
                capturas, leidos = capturar(rutas[nombre], np.linspace(0, leidos - 1, a.fotos).astype(int).tolist())
                print(f"   recontado: {nombre[:50]} | metadatos {total} fotogramas, reales {leidos}")
            if len(capturas) < a.fotos:
                print(f"menos de {a.fotos} fotos ({len(capturas)}): {nombre}")
                if not capturas:
                    continue
            base = os.path.splitext(nombre)[0]
            for parte, inicio in zip("ab", range(0, len(capturas), POR_IMAGEN)):
                img = armar_imagen(capturas[inicio:inicio + POR_IMAGEN], inicio + 1, a.ancho_foto)
                guardar_jpg(os.path.join(a.tiras, f"{base}_{parte}.jpg"), img)
            for k, (fr, t, _) in enumerate(capturas, 1):
                w.writerow([nombre, k, fr, round(t, 3)])
            f.flush()
            if i % 10 == 0:
                print(f"tiras: {i} de {len(lista)}")
    print(f"tiras en: {a.tiras}")


def preparar(a):
    _, lista = lista_v2(a)
    ruta_t = os.path.join(a.tiras, "tiempos.csv")
    fotos = {}
    with open(ruta_t, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fotos[r["archivo"]] = fotos.get(r["archivo"], 0) + 1
    duracion = {}
    with open(a.metadatos, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            duracion[r["archivo"]] = r["duracion_s"]
    faltan = [v for v in lista if v not in fotos]
    if faltan:
        raise SystemExit(f"faltan tiras de {len(faltan)} videos; corre primero el modo tiras")
    prueba = set(leer_archivos(a.prueba))
    excluidos = cargar_excluidos(a.excluidos)
    if set(lista) & (prueba | excluidos):
        raise SystemExit("hay videos de prueba o excluidos en la lista")
    orden = list(lista)
    random.Random(a.semilla).shuffle(orden)
    os.makedirs(a.lotes, exist_ok=True)
    n = 0
    for i in range(0, len(orden), a.tam_lote):
        n += 1
        with open(os.path.join(a.lotes, f"p1_lote_{n:02d}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["archivo", "tira_a", "tira_b", "fotos", "duracion_s"])
            for v in orden[i:i + a.tam_lote]:
                base = os.path.splitext(v)[0]
                w.writerow([v, f"{a.tiras}/{base}_a.jpg", f"{a.tiras}/{base}_b.jpg", fotos[v], duracion[v]])
    print(f"videos {len(orden)} | lotes de {a.tam_lote}: {n} | en: {a.lotes}")


CLAVES_P1 = {"archivo", "modelo", "nivel_inicial", "foto_incipiente", "foto_incontrolado", "foto_inicio_util", "foto_fin_util",
             "descartar", "motivo", "t_foto_24", "t_foto_48"}
NIVELES = ["controlado", "incipiente", "incontrolado"]


def cargar_tiempos(ruta):
    t = {}
    with open(ruta, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t.setdefault(r["archivo"], {})[int(r["foto"])] = float(r["t_s"])
    return t


def leer_lote(ruta):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def leer_json(ruta):
    import json
    with open(ruta, encoding="utf-8") as f:
        return json.loads(f.read().strip())


def revisar_json(d, nombre, tiempos, a, completo=True):
    from etiquetar_ia import validar
    if set(d) != (CLAVES_P1 if completo else CLAVES_P1 - {"t_foto_24", "t_foto_48"}):
        return f"claves distintas: {sorted(set(d) ^ CLAVES_P1)}"
    if d.get("archivo") != nombre:
        return f"archivo distinto: {d.get('archivo')}"
    error = validar(d, set(tiempos))
    if error:
        return error
    if completo:
        if d.get("modelo") != a.modelo:
            return f"modelo {d.get('modelo')}"
        for k, foto in (("t_foto_24", 24), ("t_foto_48", 48)):
            v = d.get(k)
            if not isinstance(v, (int, float)) or abs(float(v) - tiempos[foto]) > a.tolerancia:
                return f"{k} = {v}, real {tiempos[foto]:.2f}"
    return None


def tipo_cambio(d):
    if d.get("descartar") is True:
        return "descartado"
    fi, fu = d.get("foto_incipiente"), d.get("foto_incontrolado")
    if fi is not None and fu is not None:
        return "directo a incontrolado" if fi == fu else "ambos"
    if fi is not None:
        return "solo a incipiente"
    if fu is not None:
        return "solo a incontrolado"
    return "ninguno"


def nivel_maximo(d):
    if d.get("descartar") is True:
        return None
    if d.get("foto_incontrolado") is not None or d.get("nivel_inicial") == "incontrolado":
        return "incontrolado"
    if d.get("foto_incipiente") is not None or d.get("nivel_inicial") == "incipiente":
        return "incipiente"
    return "controlado"


def revisar(a):
    from collections import Counter
    tiempos = cargar_tiempos(os.path.join(a.tiras, "tiempos.csv"))
    rutas_lote = [os.path.join(a.lotes, n) for n in a.archivos_lote]
    fallas, validos, tipos, modelos, iniciales, descartes = [], [], Counter(), Counter(), Counter(), []
    for ruta in rutas_lote:
        for r in leer_lote(ruta):
            nombre = r["archivo"]
            ruta_j = os.path.join(a.crudo, os.path.splitext(nombre)[0] + ".json")
            if not os.path.exists(ruta_j):
                fallas.append((r, os.path.basename(ruta), "falta el json"))
                continue
            try:
                d = leer_json(ruta_j)
            except Exception as e:
                fallas.append((r, os.path.basename(ruta), f"json invalido: {e}"))
                continue
            modelos[d.get("modelo")] += 1
            error = revisar_json(d, nombre, tiempos[nombre], a)
            if error:
                fallas.append((r, os.path.basename(ruta), error))
                continue
            validos.append(nombre)
            tipos[tipo_cambio(d)] += 1
            if d.get("descartar") is not True:
                iniciales[d["nivel_inicial"]] += 1
            else:
                descartes.append((nombre, d.get("motivo", "")))
    print(f"lotes: {', '.join(a.archivos_lote)} | videos validos {len(validos)} | fallas {len(fallas)}")
    print(f"modelo: {dict(modelos)}")
    print(f"nivel inicial: {dict(iniciales)}")
    print(f"cambios: {dict(tipos)}")
    for n, m in descartes:
        print(f"   descartado: {n} | {m}")
    for r, lote, e in fallas:
        print(f"   falla: {lote} | {r['archivo']} | {e}")
    if fallas and a.repetir:
        ruta = os.path.join(a.lotes, f"p1_repetir_{a.repetir}.csv")
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fallas[0][0].keys()))
            w.writeheader()
            w.writerows(r for r, _, _ in fallas)
        print(f"lote de repeticion: {ruta}")


def comparar_ensayo(a):
    from collections import Counter
    tiempos = cargar_tiempos(os.path.join(a.tiras, "tiempos.csv"))
    pares = []
    for n in sorted(os.listdir(a.ensayo)):
        if not n.endswith(".json"):
            continue
        r1, r2 = os.path.join(a.ensayo, n), os.path.join(a.crudo, n)
        if not os.path.exists(r2):
            continue
        d1, d2 = leer_json(r1), leer_json(r2)
        if revisar_json(d1, d1["archivo"], tiempos[d1["archivo"]], a, completo=False) or revisar_json(d2, d2["archivo"], tiempos[d2["archivo"]], a):
            continue
        pares.append((d1, d2))
    ambos = [(x, y) for x, y in pares if x.get("descartar") is not True and y.get("descartar") is not True]
    print(f"pares validos {len(pares)} | descartados en el ensayo {sum(x.get('descartar') is True for x, _ in pares)}"
          f" | en la repeticion {sum(y.get('descartar') is True for _, y in pares)} | sin descartar en ambos {len(ambos)}")
    for nombre, f in [("nivel inicial", lambda d: d["nivel_inicial"]), ("nivel maximo", nivel_maximo),
                      ("hay cambio", lambda d: tipo_cambio(d) != "ninguno")]:
        acuerdo = sum(f(x) == f(y) for x, y in ambos)
        print(f"{nombre}: acuerdo {acuerdo} de {len(ambos)} ({100 * acuerdo / max(1, len(ambos)):.1f} %)")
        conf = Counter((str(f(x)), str(f(y))) for x, y in ambos if f(x) != f(y))
        for (u, v), c in sorted(conf.items()):
            print(f"   ensayo {u} -> repeticion {v}: {c}")
    for campo in ["foto_incipiente", "foto_incontrolado"]:
        difs = [y[campo] - x[campo] for x, y in ambos if x.get(campo) is not None and y.get(campo) is not None]
        solo = sum((x.get(campo) is None) != (y.get(campo) is None) for x, y in ambos)
        print(f"{campo}: ambos lo marcan {len(difs)} | diferencia en fotos (repeticion - ensayo) {difs}"
              f" | lo marca solo uno {solo}")


CLAVES_P2 = {"archivo", "modelo", "cambio", "se_ve_cambio", "foto", "motivo", "t_foto_primera", "t_foto_ultima"}
CAMPOS_V2 = ["archivo", "anotador", "modelo", "alias", "agente", "nivel_inicial", "t_incipiente", "t_incontrolado", "t_inicio_util",
             "t_fin_util", "descartar", "motivo", "foto_incipiente", "foto_incontrolado", "foto_inicio_util", "foto_fin_util",
             "sep_p1_s", "sep_p2_incipiente_s", "sep_p2_incontrolado_s", "origen_t_incipiente", "origen_t_incontrolado", "fecha"]


def lotes_p1(a):
    import glob
    return sorted(glob.glob(os.path.join(a.lotes, "p1_lote_*.csv")))


def respuestas_p1(a, tiempos):
    res = {}
    for ruta in lotes_p1(a):
        for r in leer_lote(ruta):
            nombre = r["archivo"]
            ruta_j = os.path.join(a.crudo, os.path.splitext(nombre)[0] + ".json")
            if not os.path.exists(ruta_j):
                continue
            d = leer_json(ruta_j)
            if revisar_json(d, nombre, tiempos[nombre], a) is None:
                res[nombre] = d
    return res


def separacion(t):
    fotos = sorted(t)
    return (t[fotos[-1]] - t[fotos[0]]) / max(1, len(fotos) - 1)


def cambios_p1(d):
    if d.get("descartar") is True:
        return []
    fi, fu, ni = d.get("foto_incipiente"), d.get("foto_incontrolado"), d.get("nivel_inicial")
    if fu is not None and ni == "controlado" and (fi is None or fi == fu):
        return [("incontrolado", "controlado", fu)]
    out = []
    if fi is not None:
        out.append(("incipiente", "controlado", fi))
    if fu is not None:
        out.append(("incontrolado", "incipiente", fu))
    return out


def capturar_tiempos(ruta, objetivos, t_desde):
    cap = cv2.VideoCapture(ruta)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_desde - 1.0) * 1000.0)
    mejores = [None] * len(objetivos)
    primero = None
    while True:
        if not cap.grab():
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if primero is None:
            primero = t
            if t > objetivos[0] + 0.5:
                cap.release()
                cap = cv2.VideoCapture(ruta)
                primero = -1.0
                continue
        if t > objetivos[-1] + 0.5:
            break
        k = int(np.argmin([abs(t - o) for o in objetivos]))
        if mejores[k] is None or abs(t - objetivos[k]) < abs(mejores[k][1] - objetivos[k]):
            ok, frame = cap.retrieve()
            if ok:
                mejores[k] = (0, max(0.0, t), frame)
    cap.release()
    return [m for m in mejores if m is not None]


def precision(a):
    tiempos = cargar_tiempos(os.path.join(a.tiras, "tiempos.csv"))
    res = respuestas_p1(a, tiempos)
    rutas = buscar_rutas(a.videos)
    carpeta = os.path.join(a.tiras, "precision")
    os.makedirs(carpeta, exist_ok=True)
    filas, total_cambios, cortos = [], 0, 0
    ruta_t = os.path.join(carpeta, "tiempos.csv")
    with open(ruta_t, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "cambio", "foto", "t_s"])
        for nombre in sorted(res):
            t = tiempos[nombre]
            sep = separacion(t)
            for cambio, antes, j in cambios_p1(res[nombre]):
                total_cambios += 1
                if sep <= a.sep_min:
                    cortos += 1
                    continue
                t_desde = t[max(1, j - 2)]
                t_hasta = min(t[min(max(t), j + 1)] + 2.0, t[max(t)])
                n = min(a.max_fotos, int(round((t_hasta - t_desde) / a.paso)) + 1)
                objetivos = np.linspace(t_desde, t_hasta, n).tolist()
                capturas = capturar_tiempos(rutas[nombre], objetivos, t_desde)
                if len(capturas) < 2:
                    print(f"sin fotogramas: {nombre} {cambio}")
                    continue
                base = os.path.splitext(nombre)[0]
                guardar_jpg(os.path.join(carpeta, f"{base}_{cambio}.jpg"), armar_imagen(capturas, 1, a.ancho_foto))
                for k, (_, tk, _) in enumerate(capturas, 1):
                    w.writerow([nombre, cambio, k, round(tk, 3)])
                filas.append({"archivo": nombre, "cambio": cambio, "nivel_antes": antes,
                              "tira": f"{a.tiras}/precision/{base}_{cambio}.jpg", "fotos": len(capturas)})
                f.flush()
    random.Random(a.semilla).shuffle(filas)
    n_lotes = 0
    for i in range(0, len(filas), a.tam_lote_p2):
        n_lotes += 1
        with open(os.path.join(a.lotes, f"p2_lote_{n_lotes:02d}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["archivo", "cambio", "nivel_antes", "tira", "fotos"])
            w.writeheader()
            w.writerows(filas[i:i + a.tam_lote_p2])
    print(f"videos validos {len(res)} | cambios marcados {total_cambios} | con separacion <= {a.sep_min} s {cortos}"
          f" | tiras densas {len(filas)} | lotes de {a.tam_lote_p2}: {n_lotes}")


def cargar_tiempos_p2(ruta):
    t = {}
    with open(ruta, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t.setdefault((r["archivo"], r["cambio"]), {})[int(r["foto"])] = float(r["t_s"])
    return t


def revisar_json_p2(d, fila, t, a):
    if set(d) != CLAVES_P2:
        return f"claves distintas: {sorted(set(d) ^ CLAVES_P2)}"
    if d.get("archivo") != fila["archivo"] or d.get("cambio") != fila["cambio"]:
        return "archivo o cambio distinto"
    if d.get("modelo") != a.modelo:
        return f"modelo {d.get('modelo')}"
    if d.get("se_ve_cambio") is True:
        if not isinstance(d.get("foto"), int) or d["foto"] not in t:
            return f"foto fuera de rango: {d.get('foto')}"
    elif d.get("se_ve_cambio") is False:
        if d.get("foto") is not None:
            return "foto con se_ve_cambio false"
    else:
        return "se_ve_cambio invalido"
    for k, foto in (("t_foto_primera", min(t)), ("t_foto_ultima", max(t))):
        v = d.get(k)
        if not isinstance(v, (int, float)) or abs(float(v) - t[foto]) > a.tolerancia:
            return f"{k} = {v}, real {t[foto]:.2f}"
    return None


def respuestas_p2(a):
    import glob
    tp = cargar_tiempos_p2(os.path.join(a.tiras, "precision", "tiempos.csv"))
    res, fallas = {}, []
    for ruta in sorted(glob.glob(os.path.join(a.lotes, "p2_lote_*.csv"))) + sorted(glob.glob(os.path.join(a.lotes, "p2_repetir_*.csv"))):
        for r in leer_lote(ruta):
            ruta_j = os.path.join(a.crudo, "precision", f"{os.path.splitext(r['archivo'])[0]}_{r['cambio']}.json")
            clave = (r["archivo"], r["cambio"])
            if clave in res:
                continue
            if not os.path.exists(ruta_j):
                fallas.append((r, os.path.basename(ruta), "falta el json"))
                continue
            try:
                d = leer_json(ruta_j)
            except Exception as e:
                fallas.append((r, os.path.basename(ruta), f"json invalido: {e}"))
                continue
            e = revisar_json_p2(d, r, tp[clave], a)
            if e:
                fallas.append((r, os.path.basename(ruta), e))
            else:
                res[clave] = d
                fallas = [x for x in fallas if (x[0]["archivo"], x[0]["cambio"]) != clave]
    return res, fallas, tp


def revisar_p2(a):
    res, fallas, _ = respuestas_p2(a)
    ve = sum(d["se_ve_cambio"] is True for d in res.values())
    print(f"pasada 2: validos {len(res)} | se ve el cambio {ve} | no se ve el cambio {len(res) - ve} | fallas {len(fallas)}")
    print(f"modelo: {sorted({d['modelo'] for d in res.values()})}")
    for r, lote, e in fallas:
        print(f"   falla: {lote} | {r['archivo']} {r['cambio']} | {e}")
    if fallas and a.repetir:
        ruta = os.path.join(a.lotes, f"p2_repetir_{a.repetir}.csv")
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fallas[0][0].keys()))
            w.writeheader()
            w.writerows(r for r, _, _ in fallas)
        print(f"lote de repeticion: {ruta}")


def juntar(a):
    from datetime import datetime
    tiempos = cargar_tiempos(os.path.join(a.tiras, "tiempos.csv"))
    res = respuestas_p1(a, tiempos)
    p2, fallas, tp = respuestas_p2(a)
    if fallas:
        print(f"aviso: {len(fallas)} respuestas de la pasada 2 no validas; se usa el instante de la pasada 1")
    filas, cuenta = [], {}
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for nombre in sorted(res):
        d, t = res[nombre], tiempos[nombre]
        sep = separacion(t)
        fila = {"archivo": nombre, "anotador": "ia_sonnet_v2", "modelo": d["modelo"], "alias": a.alias, "agente": a.agente,
                "descartar": int(d["descartar"] is True), "motivo": d["motivo"], "sep_p1_s": round(sep, 3), "fecha": fecha,
                "sep_p2_incipiente_s": "", "sep_p2_incontrolado_s": "", "origen_t_incipiente": "", "origen_t_incontrolado": ""}
        for k in ["nivel_inicial", "t_incipiente", "t_incontrolado", "t_inicio_util", "t_fin_util", "foto_incipiente",
                  "foto_incontrolado", "foto_inicio_util", "foto_fin_util"]:
            fila[k] = ""
        if not fila["descartar"]:
            fila["nivel_inicial"] = d["nivel_inicial"]
            for k in ["foto_incipiente", "foto_incontrolado", "foto_inicio_util", "foto_fin_util"]:
                fila[k] = "" if d.get(k) is None else d[k]
            fila["t_inicio_util"] = "" if d.get("foto_inicio_util") is None else round(t[d["foto_inicio_util"]], 2)
            fila["t_fin_util"] = "" if d.get("foto_fin_util") is None else round(t[d["foto_fin_util"]], 2)
            instantes = {}
            for cambio, _, j in cambios_p1(d):
                valor, origen, sep2 = t[j], "pasada1", ""
                if sep > a.sep_min:
                    clave = (nombre, cambio)
                    if clave in p2:
                        tt = tp[clave]
                        sep2 = round((tt[max(tt)] - tt[min(tt)]) / max(1, len(tt) - 1), 3)
                        if p2[clave]["se_ve_cambio"] is True:
                            valor, origen = tt[p2[clave]["foto"]], "pasada2"
                        else:
                            origen = "pasada1; pasada2 no ve el cambio"
                    else:
                        origen = "pasada1; pasada2 sin respuesta valida"
                instantes[cambio] = (valor, origen, sep2)
            if "incontrolado" in instantes and d["nivel_inicial"] == "controlado" and len(instantes) == 1:
                instantes["incipiente"] = instantes["incontrolado"]
            if "incipiente" in instantes and "incontrolado" in instantes and instantes["incipiente"][0] > instantes["incontrolado"][0]:
                for cambio in instantes:
                    j = d[f"foto_{cambio}"]
                    instantes[cambio] = (t[j], "pasada1; pasada2 contradice el orden", instantes[cambio][2])
            for cambio, (valor, origen, sep2) in instantes.items():
                fila[f"t_{cambio}"] = round(valor, 2)
                fila[f"origen_t_{cambio}"] = origen
                fila[f"sep_p2_{cambio}_s"] = sep2
                cuenta[origen] = cuenta.get(origen, 0) + 1
        filas.append(fila)
    with open(a.salida_v2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_V2)
        w.writeheader()
        w.writerows(filas)
    print(f"videos {len(filas)} | descartados {sum(r['descartar'] for r in filas)} | origen de los instantes: {cuenta}")
    print(f"etiquetas en: {a.salida_v2}")


def nivel_max_fila(r):
    if r["t_incontrolado"] != "" or r["nivel_inicial"] == "incontrolado":
        return "incontrolado"
    if r["t_incipiente"] != "" or r["nivel_inicial"] == "incipiente":
        return "incipiente"
    return "controlado"


def consistencia(a):
    from comparar_etiquetas import kappa
    with open(a.etiquetas_v1, encoding="utf-8") as f:
        v1 = {r["archivo"]: r for r in csv.DictReader(f)}
    with open(a.salida_v2, encoding="utf-8") as f:
        v2 = {r["archivo"]: r for r in csv.DictReader(f)}
    comunes = sorted(set(v1) & set(v2))
    ambos = [v for v in comunes if v1[v]["descartar"] != "1" and v2[v]["descartar"] != "1"]
    lineas = ["# Consistencia entre el protocolo 1 (fase 1) y el protocolo 2 del etiquetado con IA", "",
              f"Videos con las dos etiquetas: {len(comunes)}. Descartados por v1: {sum(v1[v]['descartar'] == '1' for v in comunes)};"
              f" por v2: {sum(v2[v]['descartar'] == '1' for v in comunes)}. Sin descartar en ambos: {len(ambos)}.", ""]
    for nombre, f in [("nivel inicial", lambda r: r["nivel_inicial"]), ("nivel maximo", nivel_max_fila)]:
        x = [NIVELES.index(f(v1[v])) for v in ambos]
        y = [NIVELES.index(f(v2[v])) for v in ambos]
        acuerdo = sum(p == q for p, q in zip(x, y))
        kl, m = kappa(x, y, 3, "lineal")
        lineas += [f"## {nombre}", "", f"- acuerdo: {acuerdo} de {len(ambos)} ({100 * acuerdo / max(1, len(ambos)):.1f} %)",
                   f"- kappa ponderado lineal: {kl:.3f}", "", "| v1 \\ v2 | controlado | incipiente | incontrolado |", "|---|---|---|---|"]
        for i, n in enumerate(NIVELES):
            lineas.append(f"| {n} | " + " | ".join(str(int(m[i, j])) for j in range(3)) + " |")
        lineas.append("")
    con1 = sum(v1[v]["t_incipiente"] != "" or v1[v]["t_incontrolado"] != "" for v in ambos)
    con2 = sum(v2[v]["t_incipiente"] != "" or v2[v]["t_incontrolado"] != "" for v in ambos)
    lineas += [f"- videos con algun cambio: v1 {con1}, v2 {con2} (de {len(ambos)})", ""]
    with open(a.salida_consistencia, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    print("\n".join(lineas))


def tipo_fila(r):
    if r["descartar"] == "1":
        return "descartado"
    if r["t_incontrolado"] != "" and r["nivel_inicial"] == "controlado":
        return "controlado_a_incontrolado"
    if r["t_incontrolado"] != "":
        return "incipiente_a_incontrolado"
    if r["t_incipiente"] != "":
        return "controlado_a_incipiente"
    return "sin_cambio"


def prueba_cambios(a):
    with open(a.salida_v2, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    duracion = {}
    with open(a.metadatos, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            duracion[r["archivo"]] = float(r["duracion_s"])
    por_tipo = {}
    for r in filas:
        por_tipo.setdefault(tipo_fila(r), []).append(r["archivo"])
    disponibles = {k: len(v) for k, v in por_tipo.items()}
    if a.tope_s > 0:
        por_tipo = {k: v if k in a.sin_tope else [x for x in v if duracion[x] <= a.tope_s] for k, v in por_tipo.items()}
    tipos = [k for k in ["controlado_a_incipiente", "incipiente_a_incontrolado", "controlado_a_incontrolado"] if por_tipo.get(k)]
    n = {k: min(a.min_estrato, len(por_tipo[k])) for k in tipos}
    resto = a.n_cambio - sum(n.values())
    total = sum(len(por_tipo[k]) - n[k] for k in tipos)
    cuotas = {k: resto * (len(por_tipo[k]) - n[k]) / total if total else 0 for k in tipos}
    for k in tipos:
        n[k] += int(cuotas[k])
    faltan = a.n_cambio - sum(n.values())
    for k in sorted(tipos, key=lambda k: -(cuotas[k] - int(cuotas[k]))):
        if faltan <= 0:
            break
        if n[k] < len(por_tipo[k]):
            n[k] += 1
            faltan -= 1
    rng = random.Random(a.semilla)
    elegidos = []
    tipo_de = {}
    for k in tipos:
        muestra = rng.sample(sorted(por_tipo[k]), n[k])
        tipo_de.update({v: k for v in muestra})
        elegidos += muestra
    muestra = rng.sample(sorted(por_tipo["sin_cambio"]), a.n_sin_cambio)
    tipo_de.update({v: "sin_cambio" for v in muestra})
    elegidos += muestra
    rng.shuffle(elegidos)
    with open(a.salida_prueba, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo"])
        for v in elegidos:
            w.writerow([v])
    dur = [duracion[v] for v in elegidos]
    print(f"disponibles por tipo: {disponibles}")
    print(f"tope por video: {a.tope_s:.0f} s | tipos sin tope: {a.sin_tope}")
    print(f"disponibles con el tope: { {k: len(v) for k, v in por_tipo.items()} }")
    print(f"elegidos por tipo: {n} + sin cambio {a.n_sin_cambio} = {len(elegidos)}")
    for k in tipos + ["sin_cambio"]:
        d = [duracion[v] for v in elegidos if tipo_de[v] == k]
        sobre = sum(1 for x in d if a.tope_s > 0 and x > a.tope_s)
        print(f"  {k}: {len(d)} videos | {sum(d) / 60:.1f} min | maximo {max(d):.1f} s | sobre el tope {sobre}")
    print(f"duracion total {sum(dur) / 60:.1f} min | mediana {np.median(dur):.1f} s | maximo {max(dur):.1f} s")
    print(f"lista en: {a.salida_prueba}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("modo", choices=["tiras", "preparar", "revisar", "comparar_ensayo", "precision", "revisar_p2", "juntar",
                                    "consistencia", "prueba_cambios"])
    p.add_argument("--sep_min", type=float, default=1.0)
    p.add_argument("--paso", type=float, default=0.5)
    p.add_argument("--max_fotos", type=int, default=24)
    p.add_argument("--tam_lote_p2", type=int, default=20)
    p.add_argument("--alias", default="sonnet")
    p.add_argument("--agente", default="anotador_severidad")
    p.add_argument("--salida_v2", default=os.path.join("resultados", "etiquetas_ia_v2.csv"))
    p.add_argument("--etiquetas_v1", default=os.path.join("resultados", "etiquetas_ia.csv"))
    p.add_argument("--salida_consistencia", default=os.path.join("resultados", "consistencia_ia_v1_v2.md"))
    p.add_argument("--salida_prueba", default=os.path.join("resultados", "prueba_cambios.csv"))
    p.add_argument("--n_cambio", type=int, default=25)
    p.add_argument("--n_sin_cambio", type=int, default=5)
    p.add_argument("--min_estrato", type=int, default=5)
    p.add_argument("--tope_s", type=float, default=300.0)
    p.add_argument("--sin_tope", nargs="*", default=["controlado_a_incipiente"])
    p.add_argument("--archivos_lote", nargs="+", default=[f"p1_lote_{i:02d}.csv" for i in range(1, 6)])
    p.add_argument("--crudo", default=os.path.join("resultados", "ia_crudo_v2"))
    p.add_argument("--ensayo", default=os.path.join("resultados", "ia_crudo_v2_ensayo"))
    p.add_argument("--modelo", default="claude-sonnet-5-5")
    p.add_argument("--tolerancia", type=float, default=0.1)
    p.add_argument("--repetir", default="")
    p.add_argument("--propuestas", nargs="+", default=["propuestas.csv", "propuestas_extra.csv"])
    p.add_argument("--prueba", default=os.path.join("resultados", "prueba_humana.csv"))
    p.add_argument("--excluidos", default=os.path.join("resultados", "excluidos.csv"))
    p.add_argument("--metadatos", default="metadatos_videos.csv")
    p.add_argument("--videos", default=os.path.join("real_video", "real_video", "fire"))
    p.add_argument("--tiras", default="tiras_v2")
    p.add_argument("--fotos", type=int, default=48)
    p.add_argument("--ancho_foto", type=int, default=320)
    p.add_argument("--lotes", default=os.path.join("prompts", "lotes_v2"))
    p.add_argument("--tam_lote", type=int, default=10)
    p.add_argument("--semilla", type=int, default=42)
    a = p.parse_args()
    {"tiras": tiras, "preparar": preparar, "revisar": revisar, "comparar_ensayo": comparar_ensayo, "precision": precision,
     "revisar_p2": revisar_p2, "juntar": juntar, "consistencia": consistencia, "prueba_cambios": prueba_cambios}[a.modo](a)


if __name__ == "__main__":
    main()
