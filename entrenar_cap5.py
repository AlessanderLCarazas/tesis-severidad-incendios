import os
import csv
import json
import time
import copy
import random
import argparse
import numpy as np
import torch
from modelo_escena import perdidas
from modelos_cap5 import TIPOS, crear_modelo
from variantes_cap5 import VARIANTES, preparar, columnas_x, columnas_c
from entrenar_temporal import lote, verdad_escena
from entrenar_escena import cargar, evaluar, frecuencias, NIVELES, CAMPOS_HISTORIAL
from evaluar_prueba import inferir
from fase_e import histeresis, por_segundo
from normalizacion import Normalizador
from secuencias import leer_csv

CONJUNTOS = ["calibracion", "validacion", "prueba"]


def salidas_y_umbrales(modelo, a, norm_c, carpeta_pesos):
    modelo = modelo.to("cpu").eval()
    particion = {r["archivo"]: r for r in leer_csv(a.particion)}
    cal = sorted(v for v, r in particion.items() if r["conjunto"] == "calibracion" and r["calibra_umbral"] == "si")
    resumen = {}
    for conjunto in CONJUNTOS:
        datos, info = cargar(os.path.join(a.datos, f"{conjunto}.npz"), "cpu")
        datos = preparar(datos, a.variante, norm_c)
        p1, p2, riesgo, tau = inferir(modelo, datos, info, 1.0, 4 if getattr(modelo, "tipo", "") == "mamba" else 64)
        largos = info["inicio"][1:] - info["inicio"][:-1]
        archivo = np.repeat(info["archivo"], largos)
        np.savez_compressed(os.path.join(carpeta_pesos, f"salidas_{conjunto}.npz"), t=info["t"], archivo=archivo, p1=p1, p2=p2,
                            riesgo=riesgo, tau=tau)
        if conjunto == "calibracion":
            maximo = {}
            for v, r in zip(archivo, riesgo):
                maximo[v] = max(maximo.get(v, 0.0), float(r))
            puntajes = sorted(maximo.get(v, 0.0) for v in cal)
            n = len(puntajes)
            resumen["umbral_alerta"] = {f"{alfa:g}": (puntajes[int(np.ceil((n + 1) * (1.0 - alfa))) - 1]
                                                      if int(np.ceil((n + 1) * (1.0 - alfa))) <= n else float("inf")) for alfa in (0.10, 0.05)}
            resumen["videos_calibracion"] = n
        if conjunto == "validacion":
            metas = {m["archivo"]: m for m in leer_csv(a.metas) if m["fuente"] == "ia" and m["conjunto"] == "validacion"}
            verdad = verdad_escena(metas, sorted(metas))
            niveles, tiempos = {}, {}
            for v in np.unique(archivo):
                sel = np.flatnonzero(archivo == v)
                sel = sel[np.argsort(info["t"][sel], kind="stable")]
                arriba = p1[sel] > 0.5
                crudo = arriba.astype(np.int64) + (arriba & (p2[sel] > 0.5)).astype(np.int64)
                tiempos[str(v)] = info["t"][sel].astype(np.float64)
                niveles[str(v)] = histeresis(crudo, tiempos[str(v)], a.permanencia)
            r = por_segundo(niveles, tiempos, verdad)
            resumen["validacion_final"] = {"f1": r["f1"], "f1_clase": r["f1_clase"], "kappa": r["kappa"], "cambios": r["cambios"]}
    with open(os.path.join(carpeta_pesos, "umbral.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=1)
    return resumen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", choices=TIPOS, default="gru")
    p.add_argument("--variante", choices=VARIANTES, default="completo")
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--particion", default=os.path.join("resultados", "particion", "particion.csv"))
    p.add_argument("--normalizacion_contexto", default=os.path.join("resultados", "escena_v2", "normalizacion_contexto.json"))
    p.add_argument("--unidades", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--decaimiento", type=float, default=1e-3)
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--recorte", type=float, default=1.0)
    p.add_argument("--epocas", type=int, default=60)
    p.add_argument("--tope_peso", type=float, default=5.0)
    p.add_argument("--factor_incipiente", type=float, default=1.0)
    p.add_argument("--permanencia", type=float, default=2.0)
    p.add_argument("--sin_salidas", action="store_true")
    p.add_argument("--dispositivo", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pesos", default=os.path.join("runs_fuego", "cap5", "pruebas"))
    p.add_argument("--salida", default=os.path.join("resultados", "cap5", "entrenamientos", "pruebas"))
    a = p.parse_args()

    random.seed(a.semilla)
    np.random.seed(a.semilla)
    torch.manual_seed(a.semilla)
    torch.cuda.manual_seed_all(a.semilla)
    rng = np.random.default_rng(a.semilla)
    norm_c = Normalizador.cargar(a.normalizacion_contexto)
    ent, info_ent = cargar(os.path.join(a.datos, "entrenamiento.npz"), a.dispositivo)
    val, info_val = cargar(os.path.join(a.datos, "validacion.npz"), a.dispositivo)
    ent, val = preparar(ent, a.variante, norm_c), preparar(val, a.variante, norm_c)
    metas = {m["archivo"]: m for m in leer_csv(a.metas) if m["fuente"] == "ia" and m["conjunto"] == "validacion"}
    verdad = verdad_escena(metas, sorted(metas))

    por_video, frec = frecuencias(info_ent)
    videos = sorted(por_video)
    pesos_np = np.minimum(1.0 / (3.0 * np.maximum(frec, 1e-9)), a.tope_peso)
    pesos_np[1] *= a.factor_incipiente
    pesos = torch.tensor(pesos_np, dtype=torch.float32, device=a.dispositivo)
    lotes_epoca = int(np.ceil(len(info_ent["ventanas"]) / a.lote))
    nombre = f"{a.modelo}_s{a.semilla}"
    print(f"modelo {nombre} | variante {a.variante} | dispositivo {a.dispositivo} | tasa {a.lr:g} | factor incipiente {a.factor_incipiente:g}"
          f" | entradas {ent['x'].shape[1]} | contexto {ent['c'].shape[1]} | hilos {torch.get_num_threads()}", flush=True)

    modelo = crear_modelo(a.modelo, ent["x"].shape[1], ent["c"].shape[1], a.unidades, a.dropout, a.variante == "sin_compuerta").to(a.dispositivo)
    parametros = sum(x.numel() for x in modelo.parameters())
    opt = torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=a.decaimiento)
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epocas * lotes_epoca)
    carpeta_pesos = os.path.join(a.pesos, nombre)
    carpeta = os.path.join(a.salida, nombre)
    os.makedirs(carpeta_pesos, exist_ok=True)
    os.makedirs(carpeta, exist_ok=True)
    historial, estados, resultados = [], [], []
    inicio_total = time.time()
    for epoca in range(1, a.epocas + 1):
        modelo.train()
        t0 = time.time()
        acum, n_tau = np.zeros(4), 0
        for _ in range(lotes_epoca):
            tramos = []
            for v in rng.integers(0, len(videos), a.lote):
                vs = por_video[videos[v]]
                a0, a1, _ = vs[rng.integers(0, len(vs))]
                tramos.append((a0, a1))
            b = lote(ent, tramos)
            salida = modelo(b["x"], b["c"], b["ts"])
            total, ln, lr, lt, hay = perdidas(modelo, salida, b["nivel"], b["riesgo"], b["mascara"], b["tau"], b["tau_valida"], pesos)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), a.recorte)
            opt.step()
            plan.step()
            acum += [float(total), float(ln), float(lr), float(lt) if hay else 0.0]
            n_tau += int(bool(hay))
        s_ent = time.time() - t0
        t0 = time.time()
        r = evaluar(modelo, val, info_val, verdad, pesos, 8 if a.modelo == "mamba" else 64)
        s_val = time.time() - t0
        historial.append({"epoca": epoca, "tasa": plan.get_last_lr()[0], "perdida_ent": acum[0] / lotes_epoca,
                          "nivel_ent": acum[1] / lotes_epoca, "riesgo_ent": acum[2] / lotes_epoca, "tau_ent": acum[3] / max(1, n_tau),
                          "perdida_val": r["perdida"], "nivel_val": r["nivel"], "riesgo_val": r["riesgo"], "tau_val": r["tau"],
                          "f1_escena": r["f1_escena"], "f1_suavizado": float("nan"), "exactitud_escena": r["todo"]["exactitud"],
                          "kappa": r["todo"]["kappa"], "mae_tau_s": r["mae_tau_s"], "s_ent": s_ent, "s_val": s_val})
        estados.append(copy.deepcopy({k: v.detach().cpu() for k, v in modelo.state_dict().items()}))
        resultados.append(r)
        print(f"epoca {epoca:2d} | perdida ent {acum[0] / lotes_epoca:.4f} val {r['perdida']:.4f} | f1 escena {r['f1_escena']:.4f}"
              f" | kappa {r['todo']['kappa']:.3f} | {s_ent:.0f} s + {s_val:.0f} s", flush=True)
    f1s = [h["f1_escena"] for h in historial]
    for e, h in enumerate(historial):
        h["f1_suavizado"] = float(np.mean(f1s[max(0, e - 1):e + 2]))
    mejor = int(np.argmax([h["f1_suavizado"] for h in historial]))
    torch.save({"modelo": estados[mejor], "tipo": a.modelo, "variante": a.variante, "unidades": a.unidades, "dropout": a.dropout,
                "epoca": mejor + 1, "n_x": int(ent["x"].shape[1]), "n_c": int(ent["c"].shape[1]), "columnas_x": columnas_x(a.variante),
                "columnas_c": columnas_c(a.variante), "pesos_nivel": pesos_np.tolist(), "riesgo": "escalon", "lr": a.lr, "semilla": a.semilla,
                "factor_incipiente": a.factor_incipiente}, os.path.join(carpeta_pesos, "mejor.pt"))
    with open(os.path.join(carpeta, "historial.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_HISTORIAL)
        w.writeheader()
        for h in historial:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in h.items()})
    s_entrenamiento = time.time() - inicio_total
    resumen = {"modelo": a.modelo, "variante": a.variante, "semilla": a.semilla, "parametros": parametros, "unidades": a.unidades,
               "dropout": a.dropout, "epocas_corridas": len(historial), "mejor_epoca": mejor + 1, "s_entrenamiento": s_entrenamiento,
               "frecuencia_niveles": frec.tolist(), "pesos_nivel": pesos_np.tolist(), "riesgo": "escalon",
               "validacion": dict(resultados[mejor], epoca=mejor + 1, f1_suavizado=historial[mejor]["f1_suavizado"]),
               "hiperparametros": {"lr": a.lr, "decaimiento": a.decaimiento, "lote": a.lote, "recorte": a.recorte, "epocas": a.epocas,
                                   "tope_peso": a.tope_peso, "factor_incipiente": a.factor_incipiente}}
    if not a.sin_salidas:
        del ent, val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        modelo.load_state_dict(estados[mejor])
        t0 = time.time()
        resumen["fase_e"] = salidas_y_umbrales(modelo, a, norm_c, carpeta_pesos)
        resumen["s_salidas"] = time.time() - t0
    resumen["s_total"] = time.time() - inicio_total
    with open(os.path.join(carpeta, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=1)
    extra = f" | f1 final en validacion {resumen['fase_e']['validacion_final']['f1']:.4f}" if "fase_e" in resumen else ""
    print(f"mejor epoca {mejor + 1} | f1 escena {f1s[mejor]:.4f} (suavizado {historial[mejor]['f1_suavizado']:.4f}){extra}"
          f" | entrenamiento {s_entrenamiento:.0f} s | total {resumen['s_total']:.0f} s | metricas en {carpeta}")


if __name__ == "__main__":
    main()
