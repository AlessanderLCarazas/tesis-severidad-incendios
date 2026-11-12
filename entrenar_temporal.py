import os
import csv
import json
import time
import random
import argparse
from collections import defaultdict
import numpy as np
import torch
from modelo_temporal import ModeloTemporal, TIPOS, nivel_de, perdidas
from secuencias import instantes, leer_csv

NIVELES = ["controlado", "incipiente", "incontrolado"]
CAMPOS_HISTORIAL = ["epoca", "perdida_ent", "nivel_ent", "riesgo_ent", "tau_ent", "perdida_val", "nivel_val", "riesgo_val",
                    "tau_val", "f1_escena", "exactitud_escena", "f1_muestra", "exactitud_muestra", "mae_tau_s", "s_ent", "s_val"]


def cargar(ruta, dispositivo):
    d = np.load(ruta)
    t = d["t"].astype(np.float64)
    ini = d["seq_inicio"]
    ts = d["dt"].astype(np.float32).copy()
    dif = np.diff(t).astype(np.float32)
    interior = np.ones(len(t), bool)
    interior[ini[:-1]] = False
    ts[interior] = np.maximum(dif[interior[1:]], 1e-3)
    datos = {k: torch.from_numpy(d[k]).to(dispositivo) for k in ["x", "c", "riesgo", "tau"]}
    datos["ts"] = torch.from_numpy(ts).to(dispositivo)
    datos["nivel"] = torch.from_numpy(d["nivel"].astype(np.int64)).to(dispositivo)
    datos["mascara"] = torch.from_numpy(d["mascara"]).to(dispositivo)
    datos["tau_valida"] = torch.from_numpy(d["tau_mascara"] == 1).to(dispositivo)
    info = {"t": t, "inicio": ini, "archivo": d["seq_archivo"], "ventanas": d["ventanas"], "nivel": d["nivel"], "mascara": d["mascara"]}
    return datos, info


def lote(datos, tramos):
    largo = max(b - a for a, b in tramos)
    salida = {}
    for k, v in datos.items():
        forma = (len(tramos), largo) + tuple(v.shape[1:])
        x = torch.zeros(forma, dtype=v.dtype, device=v.device)
        if k == "ts":
            x += 0.1
        for i, (a, b) in enumerate(tramos):
            x[i, :b - a] = v[a:b]
        salida[k] = x
    return salida


def f1_macro(matriz):
    f = []
    for k in range(matriz.shape[0]):
        tp = matriz[k, k]
        den = matriz[k].sum() + matriz[:, k].sum()
        f.append(2.0 * tp / den if den > 0 else 0.0)
    return float(np.mean(f)), [float(x) for x in f]


def confusion(real, pred, k=3):
    m = np.zeros((k, k), np.int64)
    np.add.at(m, (real, pred), 1)
    return m


def verdad_escena(metas, videos):
    verdad = {}
    for v in videos:
        m = metas[v]
        if m["nivel_inicial"] not in NIVELES or m["descartar"] == "1":
            continue
        ini, ti, tu, t0, t1 = instantes(m)
        seg = {}
        for b in range(int(np.floor(t0)), int(np.ceil(t1))):
            c = b + 0.5
            if c < t0 or c > t1:
                continue
            n = ini
            if ti is not None and c >= ti:
                n = max(n, 1)
            if tu is not None and c >= tu:
                n = 2
            seg[b] = n
        verdad[v] = seg
    return verdad


@torch.no_grad()
def evaluar(modelo, datos, info, verdad, pesos, tam_lote=64):
    modelo.eval()
    ini = info["inicio"]
    orden = np.argsort(ini[1:] - ini[:-1])
    pred = np.zeros(len(info["t"]), np.int64)
    tau_pred = np.zeros(len(info["t"]), np.float32)
    acum = np.zeros(3)
    pesos_acum = np.zeros(3)
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
    m = info["mascara"]
    conf_muestra = confusion(info["nivel"][m].astype(np.int64), pred[m])
    por_video = defaultdict(dict)
    seg_idx = np.floor(info["t"]).astype(np.int64)
    largos = ini[1:] - ini[:-1]
    archivo_muestra = np.repeat(info["archivo"], largos)
    for v, b, p in zip(archivo_muestra, seg_idx, pred):
        d = por_video[v]
        if p > d.get(b, 0):
            d[b] = int(p)
    real, estimado = [], []
    for v, seg in verdad.items():
        d = por_video.get(v, {})
        for b, n in seg.items():
            real.append(n)
            estimado.append(d.get(b, 0))
    conf_escena = confusion(np.array(real, np.int64), np.array(estimado, np.int64))
    tv = datos["tau_valida"].cpu().numpy() & (datos["tau"].cpu().numpy() > 0)
    mae = float(np.abs(tau_pred[tv] - datos["tau"].cpu().numpy()[tv]).mean()) if tv.any() else float("nan")
    f1_e, f1_e_clase = f1_macro(conf_escena)
    f1_m, f1_m_clase = f1_macro(conf_muestra)
    return {"perdida": total, "nivel": l_nivel, "riesgo": l_riesgo, "tau": l_tau, "f1_escena": f1_e,
            "f1_escena_clase": f1_e_clase, "exactitud_escena": float(np.trace(conf_escena) / max(1, conf_escena.sum())),
            "f1_muestra": f1_m, "f1_muestra_clase": f1_m_clase,
            "exactitud_muestra": float(np.trace(conf_muestra) / max(1, conf_muestra.sum())),
            "conf_escena": conf_escena.tolist(), "conf_muestra": conf_muestra.tolist(), "mae_tau_s": mae,
            "muestras_tau": int(tv.sum()), "segundos_escena": int(conf_escena.sum())}


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
    p.add_argument("--datos", default=os.path.join("runs_fuego", "secuencias"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--fuente", default="ia")
    p.add_argument("--unidades", type=int, default=64)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--decaimiento", type=float, default=1e-4)
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--recorte", type=float, default=1.0)
    p.add_argument("--epocas", type=int, default=60)
    p.add_argument("--paciencia", type=int, default=10)
    p.add_argument("--tope_peso", type=float, default=5.0)
    p.add_argument("--dispositivo", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pesos", default=os.path.join("runs_fuego", "modelos"))
    p.add_argument("--salida", default=os.path.join("resultados", "modelos"))
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
    print(f"modelo {a.modelo} | dispositivo {a.dispositivo} | videos de entrenamiento con ventanas: {len(videos)}"
          f" | lotes por epoca: {lotes_epoca}")
    print("frecuencia de niveles con el muestreo por video: " + " | ".join(f"{n} {f:.3f}" for n, f in zip(NIVELES, frec)))
    print("pesos de la perdida de nivel: " + " | ".join(f"{n} {w:.3f}" for n, w in zip(NIVELES, pesos_np)))
    print(f"validacion: {len(verdad)} videos de fuego, {sum(len(s) for s in verdad.values())} segundos de escena", flush=True)

    modelo = ModeloTemporal(a.modelo, unidades=a.unidades).to(a.dispositivo)
    parametros = sum(x.numel() for x in modelo.parameters())
    opt = torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=a.decaimiento)
    carpeta_pesos = os.path.join(a.pesos, a.modelo)
    carpeta = os.path.join(a.salida, a.modelo)
    os.makedirs(carpeta_pesos, exist_ok=True)
    os.makedirs(carpeta, exist_ok=True)
    historial, mejor, sin_mejora = [], None, 0
    for epoca in range(1, a.epocas + 1):
        modelo.train()
        t0 = time.time()
        acum = np.zeros(4)
        n_tau = 0
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
            acum += [float(total), float(ln), float(lr), float(lt) if hay else 0.0]
            n_tau += int(bool(hay))
        s_ent = time.time() - t0
        t0 = time.time()
        r = evaluar(modelo, val, info_val, verdad, pesos)
        s_val = time.time() - t0
        fila = {"epoca": epoca, "perdida_ent": acum[0] / lotes_epoca, "nivel_ent": acum[1] / lotes_epoca,
                "riesgo_ent": acum[2] / lotes_epoca, "tau_ent": acum[3] / max(1, n_tau), "perdida_val": r["perdida"],
                "nivel_val": r["nivel"], "riesgo_val": r["riesgo"], "tau_val": r["tau"], "f1_escena": r["f1_escena"],
                "exactitud_escena": r["exactitud_escena"], "f1_muestra": r["f1_muestra"],
                "exactitud_muestra": r["exactitud_muestra"], "mae_tau_s": r["mae_tau_s"], "s_ent": s_ent, "s_val": s_val}
        historial.append(fila)
        with open(os.path.join(carpeta, "historial.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS_HISTORIAL)
            w.writeheader()
            for h in historial:
                w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in h.items()})
        marca = ""
        if mejor is None or r["f1_escena"] > mejor["f1_escena"] + 1e-9:
            mejor = dict(r, epoca=epoca)
            sin_mejora = 0
            torch.save({"modelo": modelo.state_dict(), "tipo": a.modelo, "unidades": a.unidades, "epoca": epoca,
                        "pesos_nivel": pesos_np.tolist()}, os.path.join(carpeta_pesos, "mejor.pt"))
            marca = " *"
        else:
            sin_mejora += 1
        print(f"epoca {epoca:2d} | perdida ent {fila['perdida_ent']:.4f} val {r['perdida']:.4f} | f1 escena {r['f1_escena']:.4f}"
              f" | exactitud escena {r['exactitud_escena']:.4f} | f1 muestra {r['f1_muestra']:.4f} | mae tau {r['mae_tau_s']:.1f} s"
              f" | {s_ent:.0f} s + {s_val:.0f} s{marca}", flush=True)
        if sin_mejora >= a.paciencia:
            print(f"parada temprana en la epoca {epoca}")
            break

    resumen = {"modelo": a.modelo, "parametros": parametros, "unidades": a.unidades, "semilla": a.semilla,
               "epocas_corridas": len(historial), "mejor_epoca": mejor["epoca"],
               "s_por_epoca_ent": float(np.mean([h["s_ent"] for h in historial])),
               "s_por_epoca_val": float(np.mean([h["s_val"] for h in historial])),
               "frecuencia_niveles": frec.tolist(), "pesos_nivel": pesos_np.tolist(), "log_var": modelo.log_var.detach().cpu().tolist(),
               "videos_entrenamiento": len(videos), "lotes_por_epoca": lotes_epoca, "validacion": mejor,
               "hiperparametros": {"lr": a.lr, "decaimiento": a.decaimiento, "lote": a.lote, "recorte": a.recorte,
                                   "epocas": a.epocas, "paciencia": a.paciencia, "tope_peso": a.tope_peso}}
    with open(os.path.join(carpeta, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=1)
    print(f"mejor epoca {mejor['epoca']} | f1 escena {mejor['f1_escena']:.4f} | parametros {parametros}"
          f" | pesos en {carpeta_pesos} | metricas en {carpeta}")


if __name__ == "__main__":
    main()
