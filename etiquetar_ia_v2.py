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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("modo", choices=["tiras", "preparar", "revisar", "comparar_ensayo"])
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
    {"tiras": tiras, "preparar": preparar, "revisar": revisar, "comparar_ensayo": comparar_ensayo}[a.modo](a)


if __name__ == "__main__":
    main()
