import os
import csv
import json
import time
import copy
import random
import argparse
from collections import defaultdict
import numpy as np
import torch
from modelo_escena import ModeloEscena, TIPOS, nivel_de, perdidas
from entrenar_temporal import lote, f1_macro, confusion, verdad_escena
from secuencias import leer_csv

NIVELES = ["controlado", "incipiente", "incontrolado"]
CAMPOS_HISTORIAL = ["epoca", "tasa", "perdida_ent", "nivel_ent", "riesgo_ent", "tau_ent", "perdida_val", "nivel_val", "riesgo_val",
                    "tau_val", "f1_escena", "f1_suavizado", "exactitud_escena", "kappa", "mae_tau_s", "s_ent", "s_val"]


def kappa_lineal(m):
    m = np.asarray(m, float)
    n = m.sum()
    if n == 0:
        return float("nan")
    i, j = np.indices(m.shape)
    w = np.abs(i - j) / (m.shape[0] - 1)
    den = (w * np.outer(m.sum(1), m.sum(0)) / n).sum()
    return float(1.0 - (w * m).sum() / den) if den > 0 else float("nan")


def resumen_conf(m):
    m = np.asarray(m)
    f1, clase = f1_macro(m)
    return {"f1": f1, "f1_clase": clase, "exactitud": float(np.trace(m) / max(1, m.sum())), "kappa": kappa_lineal(m),
            "conf": m.tolist(), "segundos": int(m.sum())}


def cargar(ruta, dispositivo):
    d = np.load(ruta)
    datos = {k: torch.from_numpy(d[k]).to(dispositivo) for k in ["x", "c", "riesgo", "tau"]}
    datos["ts"] = torch.from_numpy(d["dt"].astype(np.float32)).to(dispositivo)
    datos["nivel"] = torch.from_numpy(d["nivel"].astype(np.int64)).to(dispositivo)
    datos["mascara"] = torch.from_numpy(d["mascara"]).to(dispositivo)
    datos["tau_valida"] = torch.from_numpy(d["tau_mascara"] == 1).to(dispositivo)
    info = {k: d[k] for k in ["t", "nivel", "mascara", "ventanas", "con_larga", "tau", "tau_mascara"]}
    info["inicio"], info["archivo"] = d["seq_inicio"], d["seq_archivo"]
    return datos, info


def por_segundo(info, pred, verdad):
    largos = info["inicio"][1:] - info["inicio"][:-1]
    archivo = np.repeat(info["archivo"], largos)
    seg = np.floor(info["t"]).astype(np.int64)
    votos = defaultdict(lambda: np.zeros(3, int))
    larga = defaultdict(bool)
    for v, s, p, l in zip(archivo, seg, pred, info["con_larga"]):
        votos[(v, s)][p] += 1
        if l:
            larga[(v, s)] = True
    real, est, con, cambios_p, cambios_r = [], [], [], [], []
    for v, segundos in verdad.items():
        rv, ev = [], []
        for s in sorted(segundos):
            c = votos.get((v, s))
            e = 0 if c is None else int(np.flatnonzero(c == c.max())[-1])
            rv.append(segundos[s])
            ev.append(e)
            con.append(larga[(v, s)])
        real += rv
        est += ev
        cambios_p.append(int((np.diff(ev) != 0).sum()) if len(ev) > 1 else 0)
        cambios_r.append(int((np.diff(rv) != 0).sum()) if len(rv) > 1 else 0)
    return np.array(real, np.int64), np.array(est, np.int64), np.array(con, bool), cambios_p, cambios_r


@torch.no_grad()
def evaluar(modelo, datos, info, verdad, pesos, tam_lote=64):
    modelo.eval()
    ini = info["inicio"]
    orden = np.argsort(ini[1:] - ini[:-1])
    pred = np.zeros(len(info["t"]), np.int64)
    tau_pred = np.zeros(len(info["t"]), np.float32)
    acum, pesos_acum = np.zeros(3), np.zeros(3)
    for i in range(0, len(orden), tam_lote):
        tramos = [(int(ini[j]), int(ini[j + 1])) for j in orden[i:i + tam_lote]]
        b = lote(datos, tramos)
        salida = modelo(b["x"], b["c"], b["ts"])
        _, ln, lr, lt, hay = perdidas(modelo, salida, b["nivel"], b["riesgo"], b["mascara"], b["tau"], b["tau_valida"], pesos)
        nm, nt = float(b["mascara"].sum()), float(b["tau_valida"].sum())
        acum += [float(ln) * nm, float(lr) * nm, float(lt) * nt]
        pesos_acum += [nm, nm, nt]
        niveles = nivel_de(salida[0]).cpu().numpy()
        taus = torch.expm1(salida[2]).clamp(min=0).cpu().numpy()
        for fila, (a0, a1) in enumerate(tramos):
            pred[a0:a1] = niveles[fila, :a1 - a0]
            tau_pred[a0:a1] = taus[fila, :a1 - a0]
    l_nivel, l_riesgo, l_tau = (acum / np.maximum(pesos_acum, 1)).tolist()
    s = modelo.log_var.detach().cpu().numpy()
    total = float(np.exp(-s[0]) * l_nivel + s[0] + np.exp(-s[1]) * l_riesgo + s[1] + np.exp(-s[2]) * l_tau + s[2])
    real, est, con, cambios_p, cambios_r = por_segundo(info, pred, verdad)
    tv = (info["tau_mascara"] == 1) & (info["tau"] > 0)
    mae = float(np.abs(tau_pred[tv] - info["tau"][tv]).mean()) if tv.any() else float("nan")
    r = {"perdida": total, "nivel": l_nivel, "riesgo": l_riesgo, "tau": l_tau, "mae_tau_s": mae, "muestras_tau": int(tv.sum()),
         "todo": resumen_conf(confusion(real, est)), "con_trayectoria": resumen_conf(confusion(real[con], est[con])),
         "sin_trayectoria": resumen_conf(confusion(real[~con], est[~con])),
         "cambios_modelo": float(np.mean(cambios_p)), "cambios_etiqueta": float(np.mean(cambios_r)), "videos": len(cambios_p)}
    r["f1_escena"] = r["todo"]["f1"]
    return r


def frecuencias(info):
    ini = info["inicio"]
    por_video = defaultdict(list)
    for s, a, b in info["ventanas"]:
        a0, a1 = int(ini[s] + a), int(ini[s] + b)
        m = info["mascara"][a0:a1]
        por_video[info["archivo"][s]].append((a0, a1, np.bincount(info["nivel"][a0:a1][m], minlength=3)[:3]))
    esperado = np.mean([np.mean([c for _, _, c in vs], axis=0) for vs in por_video.values()], axis=0)
    return por_video, esperado / esperado.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", choices=TIPOS, default="cfc")
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--fuente", default="ia")
    p.add_argument("--unidades", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--decaimiento", type=float, default=1e-3)
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--recorte", type=float, default=1.0)
    p.add_argument("--epocas", type=int, default=60)
    p.add_argument("--paciencia", type=int, default=10)
    p.add_argument("--tope_peso", type=float, default=5.0)
    p.add_argument("--dispositivo", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pesos", default=os.path.join("runs_fuego", "modelos_escena"))
    p.add_argument("--salida", default=os.path.join("resultados", "modelos_escena"))
    a = p.parse_args()

    random.seed(a.semilla)
    np.random.seed(a.semilla)
    torch.manual_seed(a.semilla)
    torch.cuda.manual_seed_all(a.semilla)
    rng = np.random.default_rng(a.semilla)
    ent, info_ent = cargar(os.path.join(a.datos, "entrenamiento.npz"), a.dispositivo)
    val, info_val = cargar(os.path.join(a.datos, "validacion.npz"), a.dispositivo)
    metas = {m["archivo"]: m for m in leer_csv(a.metas) if m["fuente"] == a.fuente and m["conjunto"] == "validacion"}
    verdad = verdad_escena(metas, sorted(metas))

    por_video, frec = frecuencias(info_ent)
    videos = sorted(por_video)
    pesos_np = np.minimum(1.0 / (3.0 * np.maximum(frec, 1e-9)), a.tope_peso)
    pesos = torch.tensor(pesos_np, dtype=torch.float32, device=a.dispositivo)
    lotes_epoca = int(np.ceil(len(info_ent["ventanas"]) / a.lote))
    nombre = f"{a.modelo}_s{a.semilla}"
    print(f"modelo {nombre} | dispositivo {a.dispositivo} | videos de entrenamiento: {len(videos)} | lotes por epoca: {lotes_epoca}")
    print("frecuencia de niveles con el muestreo por video: " + " | ".join(f"{n} {f:.3f}" for n, f in zip(NIVELES, frec)))
    print("pesos de la perdida de nivel: " + " | ".join(f"{n} {w:.3f}" for n, w in zip(NIVELES, pesos_np)), flush=True)

    modelo = ModeloEscena(a.modelo, n_x=ent["x"].shape[1], n_c=ent["c"].shape[1], unidades=a.unidades, dropout=a.dropout).to(a.dispositivo)
    parametros = sum(x.numel() for x in modelo.parameters())
    opt = torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=a.decaimiento)
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epocas * lotes_epoca)
    carpeta_pesos = os.path.join(a.pesos, nombre)
    carpeta = os.path.join(a.salida, nombre)
    os.makedirs(carpeta_pesos, exist_ok=True)
    os.makedirs(carpeta, exist_ok=True)
    historial, estados, resultados = [], {}, {}
    mejor, sin_mejora, inicio_total = None, 0, time.time()

    def revisar(e):
        nonlocal mejor, sin_mejora
        vecinos = [h["f1_escena"] for h in historial if abs(h["epoca"] - e) <= 1]
        suave = float(np.mean(vecinos))
        historial[e - 1]["f1_suavizado"] = suave
        if mejor is None or suave > mejor["f1_suavizado"] + 1e-9:
            mejor = dict(resultados[e], epoca=e, f1_suavizado=suave)
            sin_mejora = 0
            torch.save({"modelo": estados[e], "tipo": a.modelo, "unidades": a.unidades, "dropout": a.dropout, "epoca": e,
                        "n_x": int(ent["x"].shape[1]), "pesos_nivel": pesos_np.tolist()}, os.path.join(carpeta_pesos, "mejor.pt"))
        else:
            sin_mejora += 1
        estados.pop(e - 1, None)
        resultados.pop(e - 1, None)

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
        estados[epoca] = copy.deepcopy({k: v.detach().cpu() for k, v in modelo.state_dict().items()})
        resultados[epoca] = r
        if epoca >= 2:
            revisar(epoca - 1)
        print(f"epoca {epoca:2d} | perdida ent {acum[0] / lotes_epoca:.4f} val {r['perdida']:.4f} | f1 escena {r['f1_escena']:.4f}"
              f" | exactitud {r['todo']['exactitud']:.4f} | kappa {r['todo']['kappa']:.3f} | mae tau {r['mae_tau_s']:.1f} s"
              f" | {s_ent:.0f} s + {s_val:.0f} s", flush=True)
        if sin_mejora >= a.paciencia:
            print(f"parada temprana en la epoca {epoca}")
            break
    else:
        revisar(len(historial))
    with open(os.path.join(carpeta, "historial.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_HISTORIAL)
        w.writeheader()
        for h in historial:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in h.items()})
    resumen = {"modelo": a.modelo, "semilla": a.semilla, "parametros": parametros, "unidades": a.unidades, "dropout": a.dropout,
               "epocas_corridas": len(historial), "mejor_epoca": mejor["epoca"],
               "s_por_epoca_ent": float(np.mean([h["s_ent"] for h in historial])),
               "s_por_epoca_val": float(np.mean([h["s_val"] for h in historial])), "s_total": time.time() - inicio_total,
               "frecuencia_niveles": frec.tolist(), "pesos_nivel": pesos_np.tolist(),
               "log_var": modelo.log_var.detach().cpu().tolist(), "videos_entrenamiento": len(videos),
               "lotes_por_epoca": lotes_epoca, "validacion": mejor,
               "hiperparametros": {"lr": a.lr, "decaimiento": a.decaimiento, "lote": a.lote, "recorte": a.recorte,
                                   "epocas": a.epocas, "paciencia": a.paciencia, "tope_peso": a.tope_peso}}
    with open(os.path.join(carpeta, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=1)
    print(f"mejor epoca {mejor['epoca']} | f1 escena {mejor['f1_escena']:.4f} (suavizado {mejor['f1_suavizado']:.4f})"
          f" | parametros {parametros} | total {resumen['s_total']:.0f} s | metricas en {carpeta}")


if __name__ == "__main__":
    main()
