import os
import csv
import json
import time
import copy
import random
import argparse
import numpy as np
import torch
from modelo_escena import ModeloEscena, TIPOS, perdidas
from entrenar_temporal import lote, verdad_escena
from entrenar_escena import cargar, evaluar, frecuencias, NIVELES, CAMPOS_HISTORIAL
from secuencias import leer_csv

RIESGOS = ["escalon", "anticipado"]


def meta_riesgo(datos, tipo, horizonte):
    if tipo == "anticipado":
        rampa = (1.0 - datos["tau"] / horizonte).clamp(0.0, 1.0)
        datos["riesgo"] = torch.where(datos["tau_valida"], torch.maximum(datos["riesgo"], rampa), datos["riesgo"])
    return datos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", choices=TIPOS, default="gru")
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--fuente", default="ia")
    p.add_argument("--unidades", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--decaimiento", type=float, default=1e-3)
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--recorte", type=float, default=1.0)
    p.add_argument("--epocas", type=int, default=60)
    p.add_argument("--tope_peso", type=float, default=5.0)
    p.add_argument("--factor_incipiente", type=float, default=2.0)
    p.add_argument("--riesgo", choices=RIESGOS, default="escalon")
    p.add_argument("--horizonte", type=float, default=60.0)
    p.add_argument("--dispositivo", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pesos", default=os.path.join("runs_fuego", "modelos_final", "final"))
    p.add_argument("--salida", default=os.path.join("resultados", "modelos_final", "final"))
    a = p.parse_args()

    random.seed(a.semilla)
    np.random.seed(a.semilla)
    torch.manual_seed(a.semilla)
    torch.cuda.manual_seed_all(a.semilla)
    rng = np.random.default_rng(a.semilla)
    ent, info_ent = cargar(os.path.join(a.datos, "entrenamiento.npz"), a.dispositivo)
    val, info_val = cargar(os.path.join(a.datos, "validacion.npz"), a.dispositivo)
    meta_riesgo(ent, a.riesgo, a.horizonte)
    meta_riesgo(val, a.riesgo, a.horizonte)
    metas = {m["archivo"]: m for m in leer_csv(a.metas) if m["fuente"] == a.fuente and m["conjunto"] == "validacion"}
    verdad = verdad_escena(metas, sorted(metas))

    por_video, frec = frecuencias(info_ent)
    videos = sorted(por_video)
    pesos_np = np.minimum(1.0 / (3.0 * np.maximum(frec, 1e-9)), a.tope_peso)
    pesos_np[1] *= a.factor_incipiente
    pesos = torch.tensor(pesos_np, dtype=torch.float32, device=a.dispositivo)
    lotes_epoca = int(np.ceil(len(info_ent["ventanas"]) / a.lote))
    nombre = f"{a.modelo}_s{a.semilla}"
    print(f"modelo {nombre} | dispositivo {a.dispositivo} | tasa {a.lr:g} | riesgo {a.riesgo} | lotes por epoca: {lotes_epoca}")
    print("pesos de la perdida de nivel: " + " | ".join(f"{n} {w:.3f}" for n, w in zip(NIVELES, pesos_np)), flush=True)

    modelo = ModeloEscena(a.modelo, n_x=ent["x"].shape[1], n_c=ent["c"].shape[1], unidades=a.unidades, dropout=a.dropout).to(a.dispositivo)
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
        r = evaluar(modelo, val, info_val, verdad, pesos)
        s_val = time.time() - t0
        historial.append({"epoca": epoca, "tasa": plan.get_last_lr()[0], "perdida_ent": acum[0] / lotes_epoca,
                          "nivel_ent": acum[1] / lotes_epoca, "riesgo_ent": acum[2] / lotes_epoca, "tau_ent": acum[3] / max(1, n_tau),
                          "perdida_val": r["perdida"], "nivel_val": r["nivel"], "riesgo_val": r["riesgo"], "tau_val": r["tau"],
                          "f1_escena": r["f1_escena"], "f1_suavizado": float("nan"), "exactitud_escena": r["todo"]["exactitud"],
                          "kappa": r["todo"]["kappa"], "mae_tau_s": r["mae_tau_s"], "s_ent": s_ent, "s_val": s_val})
        estados.append(copy.deepcopy({k: v.detach().cpu() for k, v in modelo.state_dict().items()}))
        resultados.append(r)
        print(f"epoca {epoca:2d} | perdida ent {acum[0] / lotes_epoca:.4f} val {r['perdida']:.4f} | f1 escena {r['f1_escena']:.4f}"
              f" | exactitud {r['todo']['exactitud']:.4f} | kappa {r['todo']['kappa']:.3f} | {s_ent:.0f} s + {s_val:.0f} s", flush=True)
    f1s = [h["f1_escena"] for h in historial]
    for e, h in enumerate(historial):
        h["f1_suavizado"] = float(np.mean(f1s[max(0, e - 1):e + 2]))
    mejor = int(np.argmax([h["f1_suavizado"] for h in historial]))
    torch.save({"modelo": estados[mejor], "tipo": a.modelo, "unidades": a.unidades, "dropout": a.dropout, "epoca": mejor + 1,
                "n_x": int(ent["x"].shape[1]), "pesos_nivel": pesos_np.tolist(), "riesgo": a.riesgo, "horizonte": a.horizonte,
                "lr": a.lr, "semilla": a.semilla}, os.path.join(carpeta_pesos, "mejor.pt"))
    with open(os.path.join(carpeta, "historial.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_HISTORIAL)
        w.writeheader()
        for h in historial:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in h.items()})
    validacion = dict(resultados[mejor], epoca=mejor + 1, f1_suavizado=historial[mejor]["f1_suavizado"])
    resumen = {"modelo": a.modelo, "semilla": a.semilla, "parametros": parametros, "unidades": a.unidades, "dropout": a.dropout,
               "epocas_corridas": len(historial), "mejor_epoca": mejor + 1, "s_total": time.time() - inicio_total,
               "frecuencia_niveles": frec.tolist(), "pesos_nivel": pesos_np.tolist(), "riesgo": a.riesgo, "horizonte": a.horizonte,
               "log_var": modelo.log_var.detach().cpu().tolist(), "videos_entrenamiento": len(videos),
               "lotes_por_epoca": lotes_epoca, "validacion": validacion,
               "hiperparametros": {"lr": a.lr, "decaimiento": a.decaimiento, "lote": a.lote, "recorte": a.recorte, "epocas": a.epocas,
                                   "tope_peso": a.tope_peso, "factor_incipiente": a.factor_incipiente}}
    with open(os.path.join(carpeta, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=1)
    print(f"mejor epoca {mejor + 1} | f1 escena {validacion['f1_escena']:.4f} (suavizado {validacion['f1_suavizado']:.4f})"
          f" | total {resumen['s_total']:.0f} s | metricas en {carpeta}")


if __name__ == "__main__":
    main()
