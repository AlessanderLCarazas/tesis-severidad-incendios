import os
import csv
import time
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO

NIVELES = ["controlado", "incipiente", "incontrolado"]
CAMPOS = ["archivo", "nivel_inicial", "t_incipiente", "t_incontrolado", "t_inicio_util", "t_fin_util",
          "descartar", "a0", "h0", "area_max", "humo_max", "conf_media", "presencia", "cv_area", "cruces",
          "claridad", "motivos", "duracion_s", "muestras", "seg_proceso"]
CAMPOS_CURVA = ["t", "fuego", "humo", "fuego_suave", "humo_suave", "conf_fuego", "conf_humo", "n_fuego", "n_humo"]


def area_union(cajas, gx, gy):
    if len(cajas) == 0:
        return 0.0
    grilla = np.zeros((gy, gx), bool)
    for x1, y1, x2, y2 in cajas:
        c0, c1 = int(np.floor(max(0.0, x1) * gx)), int(np.ceil(min(1.0, x2) * gx))
        f0, f1 = int(np.floor(max(0.0, y1) * gy)), int(np.ceil(min(1.0, y2) * gy))
        grilla[f0:max(f1, f0 + 1), c0:max(c1, c0 + 1)] = True
    return float(grilla.mean())


def mediana(x, n):
    if n <= 1 or len(x) == 0:
        return x.copy()
    r = n // 2
    p = np.pad(x, r, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, 2 * r + 1), axis=1)


def ventana(segundos, fps):
    n = max(1, int(round(segundos * fps)))
    return n if n % 2 else n + 1


def tramos(cond):
    res = []
    i = 0
    n = len(cond)
    while i < n:
        if cond[i]:
            j = i
            while j + 1 < n and cond[j + 1]:
                j += 1
            res.append((i, j))
            i = j + 1
        else:
            i += 1
    return res


def primero_sostenido(t, cond, dur):
    for i, j in tramos(cond):
        if t[j] - t[i] >= dur - 1e-6:
            return i
    return None


def cruces_sostenidos(t, cond, dur, paso):
    estados = []
    i = 0
    n = len(cond)
    while i < n:
        j = i
        while j + 1 < n and cond[j + 1] == cond[i]:
            j += 1
        if t[j] - t[i] + paso >= dur - 1e-6 and (not estados or estados[-1] != cond[i]):
            estados.append(bool(cond[i]))
        i = j + 1
    return max(0, len(estados) - 1)


def medir(modelo, ruta, a, dispositivo):
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0
    paso = 1.0 / a.fps
    siguiente = 0.0
    idx = -1
    t_prev = -1.0
    filas = []
    while True:
        if not cap.grab():
            break
        idx += 1
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = t_ms / 1000.0 if t_ms > 0 else idx / fps
        t = max(t, 0.0)
        if t <= t_prev:
            t = t_prev + 1e-3
        t_prev = t
        if t + 1e-6 < siguiente:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        siguiente += paso * max(1, int((t - siguiente) / paso) + 1)
        r = modelo.predict(frame, imgsz=a.img, conf=a.conf, device=dispositivo, verbose=False)[0]
        cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(0, int)
        cf = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        xy = r.boxes.xyxyn.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        mf, mh = cls == 0, cls == 1
        filas.append([t, area_union(xy[mf], a.gx, a.gy), area_union(xy[mh], a.gx, a.gy),
                      float(cf[mf].max()) if mf.any() else 0.0, float(cf[mh].max()) if mh.any() else 0.0,
                      int(mf.sum()), int(mh.sum())])
    cap.release()
    return np.array(filas, float) if filas else None


def suavizar(crudo, a):
    n = ventana(a.ventana_mediana, a.fps)
    fs, hs = mediana(crudo[:, 1], n), mediana(crudo[:, 2], n)
    return np.column_stack([crudo[:, 0], crudo[:, 1], crudo[:, 2], fs, hs, crudo[:, 3], crudo[:, 4], crudo[:, 5], crudo[:, 6]])


def guardar_curva(ruta, curva):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CAMPOS_CURVA)
        for c in curva:
            w.writerow([round(c[0], 3)] + [round(v, 4) for v in c[1:7]] + [int(c[7]), int(c[8])])


def leer_crudo(ruta):
    with open(ruta, encoding="utf-8") as f:
        filas = [[float(r[k]) for k in ("t", "fuego", "humo", "conf_fuego", "conf_humo", "n_fuego", "n_humo")]
                 for r in csv.DictReader(f)]
    return np.array(filas, float) if filas else None


def duracion_video(fila, curva, a):
    fin = float(curva[-1, 0]) + 1.0 / a.fps
    try:
        meta = float(fila.get("duracion_s", "") or 0)
    except ValueError:
        meta = 0.0
    return round(min(meta, fin) if meta > 0 else fin, 2)


def proponer(curva, a, duracion):
    t, fr, hr, fs, hs, cf = curva[:, 0], curva[:, 1], curva[:, 2], curva[:, 3], curva[:, 4], curva[:, 5]
    motivos = []
    p = {"nivel_inicial": "", "t_incipiente": "", "t_incontrolado": "", "t_inicio_util": "", "t_fin_util": "",
         "descartar": 0, "a0": "", "h0": "", "area_max": round(float(fs.max()), 4), "humo_max": round(float(hs.max()), 4),
         "conf_media": "", "presencia": "", "cv_area": "", "cruces": ""}
    presente = mediana((fr > 0).astype(float), ventana(a.ventana_mediana, a.fps)) >= 0.5
    validos = [(i, j) for i, j in tramos(presente) if t[j] - t[i] >= a.presencia_min - 1e-6]
    if not validos:
        p["descartar"] = 1
        p["presencia"] = round(float((fr > 0).mean()), 3)
        return p, ["sin_fuego"]
    i0, i1 = validos[0][0], validos[-1][1]
    t0 = float(t[i0])
    t1 = max(duracion, float(t[i1])) if i1 == len(t) - 1 else float(t[i1])
    p["t_inicio_util"], p["t_fin_util"] = round(t0, 2), round(t1, 2)
    m = (t >= t0) & (t <= t1)
    tu, fu, hu = t[m], fs[m], hs[m]
    n_est = ventana(a.ventana_estabilidad, a.fps)
    fe, he = mediana(fr, n_est)[m], mediana(hr, n_est)[m]
    base = tu <= t0 + a.linea_base
    a0, h0 = float(np.median(fu[base])), float(np.median(hu[base]))
    conf_media = float(cf[m][fr[m] > 0].mean()) if (fr[m] > 0).any() else 0.0
    presencia = float((fr[m] > 0).mean())
    cv_area = float(fe.std()) / max(float(fe.mean()), 1e-6)
    nivel = 2 if a0 >= a.area_incontrolado or h0 >= a.humo_incontrolado else 0
    p["nivel_inicial"] = NIVELES[nivel]
    u_inc = max(a.factor_inc * a0, a0 + a.suma_inc)
    u_unc = max(a.factor_unc * a0, a.area_incontrolado)
    t_inc = t_unc = None
    if nivel == 0:
        k = primero_sostenido(tu, fu >= u_inc, a.sostener)
        t_inc = float(tu[k]) if k is not None else None
        k = primero_sostenido(tu, (fu >= u_unc) | (hu >= a.humo_incontrolado), a.sostener)
        t_unc = float(tu[k]) if k is not None else None
        if t_unc is not None and (t_inc is None or t_inc > t_unc):
            t_inc = t_unc
    p["t_incipiente"] = "" if t_inc is None else round(t_inc, 2)
    p["t_incontrolado"] = "" if t_unc is None else round(t_unc, 2)
    n_cruces = 0
    if nivel == 0:
        paso = 1.0 / a.fps
        n_cruces += cruces_sostenidos(tu, fe >= u_inc, a.cruce_min, paso)
        n_cruces += cruces_sostenidos(tu, (fe >= u_unc) | (he >= a.humo_incontrolado), a.cruce_min, paso)
    p.update(a0=round(a0, 4), h0=round(h0, 4), conf_media=round(conf_media, 3), presencia=round(presencia, 3),
             cv_area=round(cv_area, 3), cruces=n_cruces)

    if conf_media < a.conf_duda:
        motivos.append("conf_baja")
    if presencia < a.presencia_duda:
        motivos.append("intermitente")
    if t1 - t0 < a.util_min:
        motivos.append("util_corto")
    if cv_area > a.cv_duda:
        motivos.append("inestable")
    borde = [(a.area_incontrolado, a0), (a.humo_incontrolado, h0)]
    if any(u / a.margen_borde <= v <= u * a.margen_borde for u, v in borde):
        motivos.append("nivel_borde")
    if n_cruces > a.cruces_max:
        motivos.append("oscila")
    for campo, u in [("t_incipiente", u_inc), ("t_incontrolado", u_unc)]:
        if p[campo] != "" and float(np.max(fu[tu >= p[campo]])) < u * a.margen_borde:
            motivos.append("cambio_debil")
            break
    return p, motivos


def completar(prop, motivos, nombre, curva, duracion, seg):
    prop.update(archivo=nombre, claridad="dudosa" if motivos else "clara", motivos=";".join(motivos),
                duracion_s=duracion, muestras=len(curva), seg_proceso=seg)
    return prop


def resumen(n, total, nombre, prop, seg, restante):
    print(f"[{n}/{total}] {nombre[:50]} | {prop['nivel_inicial'] or 'descartar'} inc={prop['t_incipiente'] or '-'}"
          f" unc={prop['t_incontrolado'] or '-'} | {prop['claridad']} {prop['motivos']} | {seg} s | faltan ~{restante / 60:.0f} min")


def solo_reglas(lista, a):
    previos = {}
    if os.path.exists(a.salida):
        with open(a.salida, encoding="utf-8") as f:
            previos = {r["archivo"]: r for r in csv.DictReader(f)}
    filas = []
    sin_curva = 0
    for fila in lista:
        nombre = fila["archivo"]
        ruta = os.path.join(a.curvas, os.path.splitext(nombre)[0] + ".csv")
        if not os.path.exists(ruta):
            sin_curva += 1
            continue
        crudo = leer_crudo(ruta)
        if crudo is None:
            sin_curva += 1
            continue
        curva = suavizar(crudo, a)
        guardar_curva(ruta, curva)
        duracion = duracion_video(fila, curva, a)
        prop, motivos = proponer(curva, a, duracion)
        seg = previos.get(nombre, {}).get("seg_proceso", "")
        filas.append(completar(prop, motivos, nombre, curva, duracion, seg))
    tmp = a.salida + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)
    os.replace(tmp, a.salida)
    claras = sum(1 for r in filas if r["claridad"] == "clara")
    print(f"propuestas recalculadas: {len(filas)}, claras: {claras}, dudosas: {len(filas) - claras}, sin curva: {sin_curva}")
    conteo = {}
    for r in filas:
        for m in [x for x in r["motivos"].split(";") if x]:
            conteo[m] = conteo.get(m, 0) + 1
    for m, c in sorted(conteo.items(), key=lambda x: -x[1]):
        print(f"  {m}: {c}")
    print(f"guardado en: {a.salida}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seleccion", default="seleccion_etiquetado.csv")
    p.add_argument("--videos", default=os.path.join("real_video", "real_video", "fire"))
    p.add_argument("--modelo", default=os.path.join("runs_fuego", "fasdd_v8s_full", "weights", "best.pt"))
    p.add_argument("--salida", default="propuestas.csv")
    p.add_argument("--curvas", default="curvas")
    p.add_argument("--solo_reglas", action="store_true")
    p.add_argument("--limite", type=int, default=0)
    p.add_argument("--archivos", default="")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--img", type=int, default=640)
    p.add_argument("--gx", type=int, default=64)
    p.add_argument("--gy", type=int, default=36)
    p.add_argument("--ventana_mediana", type=float, default=1.0)
    p.add_argument("--ventana_estabilidad", type=float, default=2.0)
    p.add_argument("--presencia_min", type=float, default=1.0)
    p.add_argument("--linea_base", type=float, default=2.0)
    p.add_argument("--sostener", type=float, default=2.0)
    p.add_argument("--area_incontrolado", type=float, default=0.25)
    p.add_argument("--humo_incontrolado", type=float, default=0.4)
    p.add_argument("--factor_inc", type=float, default=2.0)
    p.add_argument("--suma_inc", type=float, default=0.02)
    p.add_argument("--factor_unc", type=float, default=5.0)
    p.add_argument("--conf_duda", type=float, default=0.40)
    p.add_argument("--presencia_duda", type=float, default=0.7)
    p.add_argument("--util_min", type=float, default=5.0)
    p.add_argument("--margen_borde", type=float, default=1.1)
    p.add_argument("--cruce_min", type=float, default=2.0)
    p.add_argument("--cruces_max", type=int, default=3)
    p.add_argument("--cv_duda", type=float, default=0.7)
    a = p.parse_args()

    with open(a.seleccion, encoding="utf-8") as f:
        lista = list(csv.DictReader(f))
    if a.archivos:
        pedidos = [s.strip() for s in a.archivos.split(",") if s.strip()]
        lista = [r for r in lista if r["archivo"] in pedidos]
    if a.limite > 0:
        lista = lista[:a.limite]
    os.makedirs(a.curvas, exist_ok=True)

    if a.solo_reglas:
        solo_reglas(lista, a)
        return

    rutas = {}
    for raiz, _, archivos in os.walk(a.videos):
        for nombre in archivos:
            rutas[nombre] = os.path.join(raiz, nombre)
    hechos = set()
    if os.path.exists(a.salida):
        with open(a.salida, encoding="utf-8") as f:
            hechos = {r["archivo"] for r in csv.DictReader(f)}
    nuevo = not os.path.exists(a.salida)
    salida = open(a.salida, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(salida, fieldnames=CAMPOS)
    if nuevo:
        w.writeheader()

    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    modelo = YOLO(a.modelo)
    pendientes = [r for r in lista if r["archivo"] not in hechos]
    print(f"videos en la lista: {len(lista)}, ya hechos: {len(lista) - len(pendientes)}, pendientes: {len(pendientes)}")
    inicio_total = time.time()
    for n, fila in enumerate(pendientes, 1):
        nombre = fila["archivo"]
        if nombre not in rutas:
            print(f"no encontrado: {nombre}")
            continue
        inicio = time.time()
        crudo = medir(modelo, rutas[nombre], a, dispositivo)
        if crudo is None:
            print(f"no se pudo leer: {nombre}")
            continue
        curva = suavizar(crudo, a)
        guardar_curva(os.path.join(a.curvas, os.path.splitext(nombre)[0] + ".csv"), curva)
        duracion = duracion_video(fila, curva, a)
        prop, motivos = proponer(curva, a, duracion)
        seg = round(time.time() - inicio, 1)
        w.writerow(completar(prop, motivos, nombre, curva, duracion, seg))
        salida.flush()
        restante = (time.time() - inicio_total) / n * (len(pendientes) - n)
        resumen(n, len(pendientes), nombre, prop, seg, restante)
    salida.close()
    print(f"listo. propuestas en: {a.salida}, curvas en: {a.curvas}")


if __name__ == "__main__":
    main()
