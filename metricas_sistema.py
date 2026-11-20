import os
import json
import time
import argparse
from collections import defaultdict
import numpy as np
import torch
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta
from modelo_escena import ModeloEscena
from entrenar_escena import resumen_conf
from evaluar_prueba import por_segundo_video, md
from analisis_prueba import referencias
from secuencias import leer_csv
from salidas_congeladas import leer_config, cargar_salidas, decidir
from tablas_latex import tabular, num, pm, pct

MODELOS = ["cfc", "gru", "lstm"]
TITULO = {"cfc": "CfC", "gru": "GRU", "lstm": "LSTM"}
REF = {"ia": "IA", "humano": "humana"}
COLOR = {"cfc": "#2a78d6", "gru": "#eb6834", "lstm": "#1baf7a"}


def clopper(k, n, nivel=0.95):
    a = (1.0 - nivel) / 2.0
    return (0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k)))


def media_desv(valores):
    v = [x for x in valores if x is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (None, None)


def probs_por_segundo(d, seg):
    salida = []
    if d is None:
        return salida
    s_fot = np.floor(d["t"]).astype(int)
    for s in sorted(seg):
        m = s_fot == s
        if m.any():
            salida.append((d["p"][m].mean(axis=0), seg[s]))
    return salida


def confiabilidad(pares, bins=10):
    p = np.array([x for x, _ in pares])
    y = np.array([n for _, n in pares], int)
    conf = p.max(axis=1)
    acierto = (p.argmax(axis=1) == y).astype(float)
    bordes = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, bordes[1:-1], right=True), 0, bins - 1)
    ece, detalle = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(acierto[m].mean() - conf[m].mean())
        detalle.append({"n": int(m.sum()), "confianza": float(conf[m].mean()) if m.any() else None,
                        "exactitud": float(acierto[m].mean()) if m.any() else None})
    brier = float(((p - np.eye(3)[y]) ** 2).sum(axis=1).mean())
    return {"ece": float(ece), "brier": brier, "segundos": int(len(y)), "bins": detalle}


def controles(a, config, refs, particion):
    peor = 0.0
    corr = cargar_salidas("calibracion", config, a.salidas)
    cal = sorted(v for v, r in particion.items() if r["conjunto"] == "calibracion" and r["calibra_umbral"] == "si")
    for nombre, c in corr.items():
        puntajes = sorted(float(c["pred"][v]["riesgo"].max()) if v in c["pred"] else 0.0 for v in cal)
        for alfa, u in c["umbral"].items():
            k = int(np.ceil((len(puntajes) + 1) * (1.0 - float(alfa))))
            peor = max(peor, abs(puntajes[k - 1] - u))
    with open(a.metricas_prueba, encoding="utf-8") as f:
        guardado = json.load(f)["corridas"]
    corr = cargar_salidas("prueba", config, a.salidas)
    for nombre, c in corr.items():
        final, _ = decidir(c, config["decision"])
        for clave, ref in refs.items():
            conf = np.zeros((3, 3), int)
            vacio = {"t": np.zeros(0), "p": np.zeros((0, 3))}
            for v in ref["nivel"]:
                _, real, est, _ = por_segundo_video(c["pred"].get(v, vacio), final.get(v, np.zeros(0, int)), ref["verdad"][v])
                np.add.at(conf, (real, est), 1)
            peor = max(peor, abs(resumen_conf(conf)["f1"] - guardado[nombre]["referencias"][clave]["final"]["f1"]))
    print(f"control de las salidas guardadas (umbral conformal y f1 de la fase f): diferencia maxima {peor:.2e}", flush=True)
    if peor > 1e-6:
        raise SystemExit("las salidas guardadas no reproducen los resultados congelados")
    return peor


def b4_confiabilidad(a, config, refs, corr):
    res = defaultdict(dict)
    pares_gru = {k: [] for k in refs}
    for nombre, c in corr.items():
        for clave, ref in refs.items():
            pares = [x for v in ref["nivel"] for x in probs_por_segundo(c["pred"].get(v), ref["verdad"][v])]
            res[nombre][clave] = confiabilidad(pares)
            if c["modelo"] == "gru":
                pares_gru[clave] += pares
    lineas = ["## B4. Confiabilidad de las probabilidades de nivel por segundo", ""]
    lineas.append("Probabilidad de cada nivel por segundo: media de los fotogramas del segundo de P(controlado) = 1 - p1,"
                  " P(incontrolado) = min(p1, p2) y P(incipiente) = p1 - P(incontrolado), las mismas de la fase F. Segundos dentro del tramo"
                  " util con al menos un fotograma. ECE con 10 intervalos iguales de la confianza (probabilidad del nivel mas probable) y Brier"
                  " multiclase (suma de los 3 niveles, de 0 a 2). Media +- desviacion de 3 semillas.")
    lineas.append("")
    lineas.append("| modelo | referencia | segundos | ECE | Brier |")
    lineas.append("|---|---|---|---|---|")
    filas_tex = []
    for m in MODELOS:
        nombres = [n for n, c in corr.items() if c["modelo"] == m]
        for clave in refs:
            e = media_desv([res[n][clave]["ece"] for n in nombres])
            b = media_desv([res[n][clave]["brier"] for n in nombres])
            seg = res[nombres[0]][clave]["segundos"]
            lineas.append(f"| {TITULO[m]} | {REF[clave]} | {seg} | {e[0]:.3f} +- {e[1]:.3f} | {b[0]:.3f} +- {b[1]:.3f} |")
            filas_tex.append([TITULO[m], REF[clave].capitalize() if clave == "humano" else "IA", str(seg), pm(*e), pm(*b)])
    lineas.append("")
    tabular(os.path.join(a.tablas, "b4_confiabilidad.tex"), ["Modelo", "Referencia", "Segundos", "ECE", "Brier"], filas_tex, "llccc")
    fig, ejes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    for eje, clave in zip(ejes, refs):
        r = confiabilidad(pares_gru[clave])
        centros = np.arange(0.05, 1.0, 0.1)
        acc = [b["exactitud"] if b["exactitud"] is not None else 0.0 for b in r["bins"]]
        conf = [b["confianza"] for b in r["bins"]]
        eje.bar(centros, acc, width=0.09, color="#2a78d6", label="exactitud")
        eje.plot([0, 1], [0, 1], color="#52514e", linewidth=1, linestyle=(0, (4, 3)), label="calibración perfecta")
        for x, cb, b in zip(centros, conf, r["bins"]):
            if cb is not None:
                eje.plot([x - 0.045, x + 0.045], [cb, cb], color="#eb6834", linewidth=2)
            if b["n"]:
                eje.text(x, 1.02, f"{b['n']}", ha="center", fontsize=7, color="#52514e")
        eje.plot([], [], color="#eb6834", linewidth=2, label="confianza media")
        eje.set_xlim(0, 1)
        eje.set_ylim(0, 1.08)
        eje.set_xlabel("confianza (probabilidad del nivel más probable)")
        eje.set_title(f"Referencia {'de la IA' if clave == 'ia' else 'humana'}: ECE {r['ece']:.3f}".replace(".", ","), fontsize=10, loc="left")
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
        eje.grid(axis="y", color="#e4e3df", linewidth=0.8)
        eje.set_axisbelow(True)
    coma = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ","))
    for eje in ejes:
        eje.xaxis.set_major_formatter(coma)
        eje.yaxis.set_major_formatter(coma)
    ejes[0].set_ylabel("exactitud en el intervalo")
    ejes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "GRU, 3 semillas juntas. Número sobre cada barra: segundos en el intervalo.", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(a.figuras, exist_ok=True)
    ruta = os.path.join(a.figuras, "confiabilidad_gru.pdf")
    fig.savefig(ruta)
    plt.close(fig)
    return dict(res), lineas


def grupos_alerta(particion, refs, sin_fuego_prueba):
    cal = {v: r for v, r in particion.items() if r["conjunto"] == "calibracion" and r["calibra_umbral"] == "si"}
    val = {v: r for v, r in particion.items() if r["conjunto"] == "validacion"}
    from analisis_prueba import nivel_max
    g = {"calibracion": {"controlados": sorted(v for v, r in cal.items() if r["clase"] == "fuego"),
                         "sin fuego": sorted(v for v, r in cal.items() if r["clase"] == "sin_fuego")},
         "validacion": {"controlados": sorted(v for v, r in val.items() if r["clase"] == "fuego" and r["nivel_max"] == "controlado"),
                        "sin fuego": sorted(v for v, r in val.items() if r["clase"] == "sin_fuego")},
         "prueba (IA)": {"controlados": [v for v in refs["ia"]["nivel"] if nivel_max(refs["ia"]["metas"][v]) == 0], "sin fuego": sin_fuego_prueba},
         "prueba (humana)": {"controlados": [v for v in refs["humano"]["nivel"] if nivel_max(refs["humano"]["metas"][v]) == 0],
                             "sin fuego": sin_fuego_prueba}}
    for c in g.values():
        c["juntos"] = c["controlados"] + c["sin fuego"]
    return g


def b5_cobertura(a, config, particion, refs, sin_fuego):
    grupos = grupos_alerta(particion, refs, sin_fuego)
    alfa = f"{config['decision']['alfa']:g}"
    salidas = {c: cargar_salidas(c, config, a.salidas) for c in ("calibracion", "validacion", "prueba")}
    res = defaultdict(dict)
    for conjunto, g in grupos.items():
        corr = salidas[conjunto.split(" ")[0]]
        for nombre, c in corr.items():
            _, t_alerta = decidir(c, config["decision"])
            fila = {}
            for k, vs in g.items():
                n_al = int(sum(t_alerta.get(v) is not None for v in vs))
                fila[k] = {"alarmas": n_al, "videos": len(vs), "tasa": n_al / len(vs), "ic": clopper(n_al, len(vs))}
            res[conjunto][nombre] = fila
    lineas = [f"## B5. Cobertura de la alerta conformal (alfa {alfa})", ""]
    lineas.append("Misma unidad que la calibracion: el video. Un video tiene falsa alarma si la alerta se activa en algun momento (equivale a que"
                  " su riesgo maximo llegue al umbral). El umbral se calibro con los 52 videos de calibracion sin fuego o controlados, juntos;"
                  " la garantia conformal es para esa poblacion junta (tasa esperada <= 10 %), no para cada grupo por separado."
                  " IC 95 % exacto de Clopper-Pearson por corrida; se cuenta en cuantas de las 3 corridas el intervalo incluye el 10 %.")
    lineas.append("")
    lineas.append("| conjunto | modelo | controlados | sin fuego | juntos | corridas con el 10 % dentro del IC (juntos) |")
    lineas.append("|---|---|---|---|---|---|")
    filas_tex = []
    for conjunto in grupos:
        for m in MODELOS:
            nombres = [n for n, c in salidas["prueba"].items() if c["modelo"] == m]
            celdas, celdas_tex = [], []
            for k in ("controlados", "sin fuego", "juntos"):
                tasas = [res[conjunto][n][k]["tasa"] for n in nombres]
                n_v = res[conjunto][nombres[0]][k]["videos"]
                al = float(np.mean([res[conjunto][n][k]["alarmas"] for n in nombres]))
                celdas.append(f"{100 * np.mean(tasas):.1f} +- {100 * np.std(tasas):.1f} % ({al:.1f} de {n_v})")
                celdas_tex.append(f"{pct(float(np.mean(tasas)))} ({num(al, 1)} de {n_v})")
            dentro = sum(res[conjunto][n]["juntos"]["ic"][0] <= 0.10 <= res[conjunto][n]["juntos"]["ic"][1] for n in nombres)
            lineas.append(f"| {conjunto} | {TITULO[m]} | " + " | ".join(celdas) + f" | {dentro} de {len(nombres)} |")
            nombre_tex = {"calibracion": "Calibración", "validacion": "Validación", "prueba (IA)": "Prueba (IA)",
                          "prueba (humana)": "Prueba (humana)"}[conjunto]
            filas_tex.append([nombre_tex, TITULO[m]] + celdas_tex + [f"{dentro} de {len(nombres)}"])
    lineas.append("")
    tabular(os.path.join(a.tablas, "b5_cobertura_conformal.tex"), ["Conjunto", "Modelo", "Controlados", "Sin fuego", "Juntos",
                                                                    "Corridas con 10\\,\\% en el IC"], filas_tex, "llcccc")
    return dict(res), lineas


def b6_estabilidad(a, config, refs, corr):
    res = defaultdict(dict)
    ref_tasa = {}
    for clave, ref in refs.items():
        videos = ref["completa"]
        seg_tot = sum(len(ref["verdad"][v]) for v in videos)
        cambios_ref = sum(int((np.diff([ref["verdad"][v][s] for s in sorted(ref["verdad"][v])]) != 0).sum()) for v in videos)
        ref_tasa[clave] = {"cambios_min": 60.0 * cambios_ref / seg_tot, "videos": len(videos), "minutos": seg_tot / 60.0}
        for nombre, c in corr.items():
            final, _ = decidir(c, config["decision"])
            crudo = {v: d["nivel"] for v, d in c["pred"].items()}
            fila = {}
            for etapa, niveles in (("final", final), ("sin_histeresis", crudo)):
                vacio = {"t": np.zeros(0), "p": np.zeros((0, 3))}
                cambios = 0
                for v in videos:
                    _, _, est, _ = por_segundo_video(c["pred"].get(v, vacio), niveles.get(v, np.zeros(0, int)), ref["verdad"][v])
                    cambios += int((np.diff(est) != 0).sum())
                fila[etapa] = 60.0 * cambios / seg_tot
            res[nombre][clave] = fila
    lineas = ["## B6. Estabilidad: cambios de nivel por minuto", ""]
    lineas.append("Cambios del nivel por segundo dentro del tramo util, sobre el total de minutos. Referencia de la IA: los 70 videos de fuego;"
                  f" referencia humana: solo los {ref_tasa['humano']['videos']} videos con cambios marcados reproduciendo el video (los otros 40"
                  " tienen un solo nivel por construccion).")
    lineas.append("")
    lineas.append("| modelo | referencia | minutos | cambios por minuto del modelo (final) | sin histeresis | cambios por minuto de la referencia |")
    lineas.append("|---|---|---|---|---|---|")
    filas_tex = []
    for m in MODELOS:
        nombres = [n for n, c in corr.items() if c["modelo"] == m]
        for clave in refs:
            f = media_desv([res[n][clave]["final"] for n in nombres])
            s = media_desv([res[n][clave]["sin_histeresis"] for n in nombres])
            lineas.append(f"| {TITULO[m]} | {REF[clave]} | {ref_tasa[clave]['minutos']:.1f} | {f[0]:.2f} +- {f[1]:.2f} | {s[0]:.2f} +- {s[1]:.2f} |"
                          f" {ref_tasa[clave]['cambios_min']:.2f} |")
            filas_tex.append([TITULO[m], "IA" if clave == "ia" else "Humana", num(ref_tasa[clave]["minutos"], 1), pm(f[0], f[1], 2),
                              pm(s[0], s[1], 2), num(ref_tasa[clave]["cambios_min"], 2)])
    lineas.append("")
    tabular(os.path.join(a.tablas, "b6_estabilidad.tex"), ["Modelo", "Referencia", "Minutos", "Cambios por minuto (final)",
                                                            "Sin histéresis", "Referencia"], filas_tex, "llcccc")
    return {"corridas": dict(res), "referencia": ref_tasa}, lineas


def paso(modelo, estado, x, c, ts):
    u = x * torch.sigmoid(modelo.compuerta(c))
    if modelo.tipo == "cfc":
        _, h = modelo.celda(u, estado, ts)
        salida, nuevo = h, h
    else:
        y, nuevo = modelo.rnn(torch.cat([u, ts], -1).unsqueeze(1), estado)
        salida = y[:, 0]
    return modelo.coral(salida) + modelo.sesgos, modelo.riesgo(salida), modelo.tau(salida), nuevo


@torch.no_grad()
def medir(modelo, dispositivo, n_x, pasos, calentamiento):
    modelo = modelo.to(dispositivo).eval()
    x = torch.randn(1, n_x, device=dispositivo)
    c = torch.randn(1, 6, device=dispositivo)
    ts = torch.full((1, 1), 0.1, device=dispositivo)
    estado = torch.zeros(1, modelo.unidades, device=dispositivo) if modelo.tipo == "cfc" else None
    tiempos = []
    for i in range(calentamiento + pasos):
        if dispositivo != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        *_, estado = paso(modelo, estado, x, c, ts)
        if dispositivo != "cpu":
            torch.cuda.synchronize()
        if i >= calentamiento:
            tiempos.append(1000.0 * (time.perf_counter() - t0))
    t = np.array(tiempos)
    return {"mediana_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)), "media_ms": float(t.mean())}


def b7_costo(a, config):
    res = {}
    gpu = torch.cuda.is_available()
    for m in MODELOS:
        c = config["corridas"][f"{m}_s42"]
        estado = torch.load(c["archivo_pesos"], map_location="cpu", weights_only=False)
        modelo = ModeloEscena(estado["tipo"], n_x=estado["n_x"], unidades=estado["unidades"], dropout=estado["dropout"])
        modelo.load_state_dict(estado["modelo"])
        r = {"parametros": int(sum(p.numel() for p in modelo.parameters())), "unidades": estado["unidades"],
             "cpu": medir(modelo, "cpu", estado["n_x"], a.pasos, a.calentamiento)}
        if gpu:
            r["gpu"] = medir(modelo, "cuda", estado["n_x"], a.pasos, a.calentamiento)
        res[m] = r
        print(f"costo {m}: {r['parametros']} parametros | cpu {r['cpu']['mediana_ms']:.3f} ms" + (f" | gpu {r['gpu']['mediana_ms']:.3f} ms" if gpu else ""), flush=True)
    lineas = ["## B7. Costo de cada modelo temporal", ""]
    lineas.append(f"Un paso = un fotograma con lote 1 (compuerta, celda o paso recurrente y las tres cabezas), como en el sistema en vivo, sin la"
                  f" normalizacion. {a.calentamiento} pasos de calentamiento y {a.pasos} medidos; mediana y percentil 95. CPU con"
                  f" {torch.get_num_threads()} hilos de PyTorch; GPU {torch.cuda.get_device_name(0) if gpu else 'no disponible'} con"
                  " sincronizacion en cada paso. Los parametros son los mismos en las 3 semillas.")
    lineas.append("")
    lineas.append("| modelo | parametros | CPU, mediana (ms) | CPU, p95 (ms) | GPU, mediana (ms) | GPU, p95 (ms) |")
    lineas.append("|---|---|---|---|---|---|")
    filas_tex = []
    for m, r in res.items():
        g = r.get("gpu", {})
        lineas.append(f"| {TITULO[m]} | {r['parametros']} | {r['cpu']['mediana_ms']:.3f} | {r['cpu']['p95_ms']:.3f} |"
                      f" {g.get('mediana_ms', float('nan')):.3f} | {g.get('p95_ms', float('nan')):.3f} |")
        from tablas_latex import miles
        filas_tex.append([TITULO[m], miles(r["parametros"]), num(r["cpu"]["mediana_ms"]), num(r["cpu"]["p95_ms"]),
                          num(g.get("mediana_ms")), num(g.get("p95_ms"))])
    lineas.append("")
    tabular(os.path.join(a.tablas, "b7_costo.tex"), ["Modelo", "Parámetros", "CPU, mediana (ms)", "CPU, p95 (ms)", "GPU, mediana (ms)",
                                                      "GPU, p95 (ms)"], filas_tex)
    return res, lineas


def b8_tau(a, config, refs, corr):
    d = np.load(os.path.join(a.datos, "prueba.npz"))
    ent = np.load(os.path.join(a.datos, "entrenamiento.npz"))
    constante = float(np.median(ent["tau"][(ent["tau_mascara"] == 1) & (ent["tau"] > 0)]))
    archivo = np.repeat(d["seq_archivo"], d["seq_inicio"][1:] - d["seq_inicio"][:-1])
    tau_ref = {"ia": (d["tau"], d["tau_mascara"]), "humano": (d["tau_humano"], d["tau_mascara_humano"])}
    with open(a.metricas_prueba, encoding="utf-8") as f:
        guardado = json.load(f)["corridas"]
    res, peor = defaultdict(dict), 0.0
    sel_ref = {}
    for clave, ref in refs.items():
        tr, tm = tau_ref[clave]
        sel = np.isin(archivo, list(ref["completa"])) & (tm == 1) & (tr > 0)
        sel_ref[clave] = sel
        for nombre, c in corr.items():
            est = np.zeros(len(tr), np.float32)
            for v, p in c["pred"].items():
                est[p["idx"]] = p["tau"]
            err = np.abs(est[sel] - tr[sel])
            res[nombre][clave] = {"mediana_s": float(np.median(err)), "mae_s": float(err.mean()), "fotogramas": int(sel.sum())}
            peor = max(peor, abs(err.mean() - guardado[nombre]["referencias"][clave]["tau"]["mae_s"]))
        err_c = np.abs(constante - tr[sel])
        res["constante"][clave] = {"mediana_s": float(np.median(err_c)), "mae_s": float(err_c.mean()), "fotogramas": int(sel.sum()),
                                   "videos": int(len(set(archivo[sel])))}
    print(f"control del mae de tau con la fase f: diferencia maxima {peor:.2e}", flush=True)
    if peor > 1e-3:
        raise SystemExit("el mae de tau no reproduce la fase f")
    lineas = ["## B8. Error absoluto del tiempo al peligro", ""]
    lineas.append("Solo en los fotogramas anteriores al cambio a incontrolado de los videos donde la referencia lo define (tau_mascara = 1, tau > 0)."
                  f" El MAE reproduce el de la fase F (diferencia {peor:.1e}). Constante: la mediana de tau en entrenamiento ({constante:.1f} s).")
    lineas.append("")
    lineas.append("| modelo | referencia | videos | fotogramas | error absoluto mediano (s) | MAE (s) |")
    lineas.append("|---|---|---|---|---|---|")
    filas_tex = []
    for m in MODELOS + ["constante"]:
        nombres = [n for n, c in corr.items() if c["modelo"] == m] if m != "constante" else ["constante"]
        for clave in refs:
            me = media_desv([res[n][clave]["mediana_s"] for n in nombres])
            ma = media_desv([res[n][clave]["mae_s"] for n in nombres])
            videos = res["constante"][clave]["videos"]
            nombre_m = TITULO.get(m, "Constante (mediana de entrenamiento)")
            lineas.append(f"| {nombre_m} | {REF[clave]} | {videos} | {res['constante'][clave]['fotogramas']} | {me[0]:.1f} +- {me[1]:.1f} |"
                          f" {ma[0]:.1f} +- {ma[1]:.1f} |")
            filas_tex.append([nombre_m, "IA" if clave == "ia" else "Humana", str(videos), pm(me[0], me[1] if m != "constante" else None, 1),
                              pm(ma[0], ma[1] if m != "constante" else None, 1)])
    lineas.append("")
    tabular(os.path.join(a.tablas, "b8_tiempo_peligro.tex"), ["Modelo", "Referencia", "Videos", "Error absoluto mediano (s)", "MAE (s)"],
            filas_tex, "llccc")
    return {"corridas": dict(res), "constante_s": constante, "control": peor}, lineas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partes", nargs="+", default=["b4", "b5", "b6", "b7", "b8"], choices=["b4", "b5", "b6", "b7", "b8"])
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--salidas", default=os.path.join("runs_fuego", "salidas_congeladas"))
    p.add_argument("--metricas_prueba", default=os.path.join("resultados", "prueba", "metricas.json"))
    p.add_argument("--pasos", type=int, default=2000)
    p.add_argument("--calentamiento", type=int, default=200)
    p.add_argument("--tablas", default=os.path.join("resultados", "tablas_tesis"))
    p.add_argument("--figuras", default=os.path.join("resultados", "figuras_tesis"))
    p.add_argument("--salida", default=os.path.join("resultados", "cap5"))
    a = p.parse_args()

    config = leer_config(a.config)
    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    refs, sin_fuego = referencias("prueba", particion, a.metas)
    control = controles(a, config, refs, particion)
    corr = cargar_salidas("prueba", config, a.salidas)
    resultados = {"control": control}
    lineas = ["# Metricas nuevas del sistema completo en la prueba (bloque del asesor, despues de la fase F)", "",
              "Generado por `python metricas_sistema.py` sobre las salidas por fotograma de los modelos congelados"
              f" (`runs_fuego/salidas_congeladas/`; reproducen el umbral conformal y el F1 de la fase F con diferencia {control:.1e})."
              " Nada de esto se uso para elegir el sistema. Modelos: CfC, GRU y LSTM con 3 semillas (no hay modelos Mamba entrenados).", ""]
    tareas = {"b4": lambda: b4_confiabilidad(a, config, refs, corr), "b5": lambda: b5_cobertura(a, config, particion, refs, sin_fuego),
              "b6": lambda: b6_estabilidad(a, config, refs, corr), "b7": lambda: b7_costo(a, config), "b8": lambda: b8_tau(a, config, refs, corr)}
    for parte, fn in tareas.items():
        if parte in a.partes:
            r, l = fn()
            resultados[parte] = r
            lineas += l
            print(f"{parte} listo", flush=True)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "sistema.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas) + "\n")
    with open(os.path.join(a.salida, "sistema.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=1)
    print(f"resultados en {a.salida}, tablas en {a.tablas} y figura en {a.figuras}")


if __name__ == "__main__":
    main()
