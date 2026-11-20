import os
import csv
import json
import time
import argparse
from collections import defaultdict
import numpy as np
import cv2
import torch
from figlib_piloto import Muestra, elegir, descargar, leer_secuencia

MODELOS = ["cfc", "gru", "lstm", "mamba"]
NOMBRE = {"cfc": "CfC", "gru": "GRU", "lstm": "LSTM", "mamba": "Mamba", "regla_humo": "regla de humo en 2 imagenes seguidas"}


def corridas_vivo(a, config):
    lista = []
    for n, c in config["corridas"].items():
        lista.append({"nombre": n, "modelo": c["modelo"], "pesos": c["archivo_pesos"], "umbral": c["umbral_alerta"]["0.1"], "congelada": True})
    for m in ("cfc", "gru", "lstm"):
        for s in (45, 46):
            carpeta = os.path.join(a.pesos, "semillas", f"{m}_s{s}")
            u = json.load(open(os.path.join(carpeta, "umbral.json"), encoding="utf-8"))
            lista.append({"nombre": f"{m}_s{s}", "modelo": m, "pesos": os.path.join(carpeta, "mejor.pt"), "umbral": u["umbral_alerta"]["0.1"],
                          "congelada": False})
    eleccion = json.load(open(os.path.join(a.entrenamientos, "mamba_eleccion.json"), encoding="utf-8"))
    for s in (42, 43, 44, 45, 46):
        carpeta = os.path.join(a.pesos, eleccion["grupo"], f"mamba_s{s}")
        u = json.load(open(os.path.join(carpeta, "umbral.json"), encoding="utf-8"))
        lista.append({"nombre": f"mamba_s{s}", "modelo": "mamba", "pesos": os.path.join(carpeta, "mejor.pt"), "umbral": u["umbral_alerta"]["0.1"],
                      "congelada": False})
    return lista


def modelos_vivo(lista, config, norm_x, norm_c):
    from sistema import ModeloEnVivo
    from modelo_escena import ModeloEscena
    from modelos_cap5 import cargar_modelo

    class ModeloVivo(ModeloEnVivo):
        def __init__(self, c):
            self.nombre, self.modelo_tipo = c["nombre"], c["modelo"]
            if c["congelada"]:
                e = torch.load(c["pesos"], map_location="cpu", weights_only=False)
                self.red = ModeloEscena(e["tipo"], n_x=e["n_x"], unidades=e["unidades"], dropout=e["dropout"])
                self.red.load_state_dict(e["modelo"])
                self.red.eval()
            else:
                self.red, e = cargar_modelo(c["pesos"])
            self.tipo = e["tipo"]
            self.unidad = config["por_modelo"].get(c["modelo"], {}).get("unidad_dt_s") or 1.0
            self.norm_x, self.norm_c = norm_x, norm_c
            dec = config["decision"]
            self.u1, self.u2 = dec["umbral_coral"]
            self.permanencia, self.factor = dec["histeresis_s"], dec["factor_desactivacion"]
            self.umbral = c["umbral"]
            self.reiniciar_video()

        def reiniciar_tramo(self):
            self.h = None

        @torch.no_grad()
        def inferir(self, x, c, dt):
            if self.tipo != "mamba":
                return super().inferir(x, c, dt)
            red = self.red
            xx = torch.from_numpy(self.norm_x.transformar(x[None]).astype(np.float32))
            cc = torch.from_numpy(self.norm_c.transformar(c[None]).astype(np.float32))
            ts = torch.tensor([[dt / self.unidad]], dtype=torch.float32)
            if self.h is None:
                self.h = red.cache_inicial(1, "cpu")
            logits, r, lt, self.h = red.paso(xx, cc, ts, self.h)
            p = torch.sigmoid(logits)[0]
            return float(p[0]), float(p[1]), float(torch.sigmoid(r)[0, 0]), float(torch.expm1(lt).clamp(min=0)[0, 0]), xx[0].numpy(), cc[0].numpy()

    return [ModeloVivo(c) for c in lista]


def evaluar(a):
    from sistema import Sistema
    from salidas_congeladas import leer_config
    from normalizacion import Normalizador
    config = leer_config(a.config)
    sistema = Sistema(config, [])
    norm_x = Normalizador.cargar(config["datos"]["normalizacion"][0])
    norm_c = Normalizador.cargar(config["datos"]["normalizacion"][1])
    lista = corridas_vivo(a, config)
    sistema.modelos = modelos_vivo(lista, config, norm_x, norm_c)
    secuencias = [r["secuencia"] for r in csv.DictReader(open(a.lista, encoding="utf-8")) if r["estado"] == "aceptada"]
    hechas = set()
    ruta_img = os.path.join(a.salida, "imagenes.csv")
    ruta_seq = os.path.join(a.salida, "secuencias.csv")
    if os.path.exists(ruta_seq):
        hechas = {r["secuencia"] for r in csv.DictReader(open(ruta_seq, encoding="utf-8"))}
    campos_img = ["secuencia", "imagen", "k", "desfase_s", "humo_visible", "humo_conf_640", "fuego_conf_640", "humo_conf_1280", "fuego_conf_1280",
                  "tray_humo", "tray_fuego"] + [f"alerta_{c['nombre']}" for c in lista]
    os.makedirs(a.salida, exist_ok=True)
    f_img = open(ruta_img, "a", newline="", encoding="utf-8")
    w_img = csv.DictWriter(f_img, fieldnames=campos_img)
    if not hechas:
        w_img.writeheader()
    f_seq = open(ruta_seq, "a", newline="", encoding="utf-8")
    w_seq = csv.DictWriter(f_seq, fieldnames=["secuencia", "imagenes", "paso_s", "segundos", "cortes", "trayectorias_humo", "trayectorias_fuego"])
    if not hechas:
        w_seq.writeheader()
    inicio = time.time()
    for n_seq, seq in enumerate(secuencias, 1):
        if seq in hechas:
            continue
        carpeta = os.path.join(a.datos, seq)
        nombres, marcas, desfases, paso = leer_secuencia(carpeta)
        validos = [i for i, n in enumerate(nombres) if os.path.getsize(os.path.join(carpeta, n)) > 0]
        if len(validos) < len(nombres):
            print(f"{seq}: {len(nombres) - len(validos)} imagenes vacias omitidas", flush=True)
            nombres, marcas, desfases = ([x[i] for i in validos] for x in (nombres, marcas, desfases))
        sistema.iniciar()
        t0 = time.time()
        tray = {"humo": set(), "fuego": set()}
        cortes = 0
        for k, (nombre, marca, desfase) in enumerate(zip(nombres, marcas, desfases)):
            img = cv2.imread(os.path.join(carpeta, nombre))
            m = Muestra(k, k, float(marca - marcas[0]), paso if k == 0 else float(marca - marcas[k - 1]), img)
            conf = {}
            for tam in (640, 1280):
                r = sistema.yolo.predict(img, imgsz=tam, conf=0.1, device=sistema.dispositivo, verbose=False)[0]
                det = r.boxes.data.cpu().numpy() if r.boxes is not None else np.zeros((0, 6), np.float32)
                for clase, etiqueta in ((1, "humo"), (0, "fuego")):
                    sel = det[det[:, 5] == clase] if len(det) else det
                    conf[f"{etiqueta}_conf_{tam}"] = round(float(sel[:, 4].max()), 4) if len(sel) else 0.0
            s = sistema.paso(m)
            cortes += int(bool(s["corte"]))
            ids = {}
            for clase in ("humo", "fuego"):
                ids[clase] = [(s["tramo"], int(fila[4])) for fila in s[clase]]
                tray[clase] |= set(ids[clase])
            fila = {"secuencia": seq, "imagen": nombre, "k": k, "desfase_s": desfase, "humo_visible": int(desfase >= 0),
                    "tray_humo": len(ids["humo"]), "tray_fuego": len(ids["fuego"]), **conf}
            for c in lista:
                fila[f"alerta_{c['nombre']}"] = int(s["resultados"][c["nombre"]]["alerta"])
            w_img.writerow(fila)
        seg = time.time() - t0
        w_seq.writerow({"secuencia": seq, "imagenes": len(nombres), "paso_s": paso, "segundos": round(seg, 1), "cortes": cortes,
                        "trayectorias_humo": len(tray["humo"]), "trayectorias_fuego": len(tray["fuego"])})
        f_img.flush()
        f_seq.flush()
        print(f"[{n_seq}/{len(secuencias)}] {seq} | {len(nombres)} imagenes | {seg:.0f} s | llevo {(time.time() - inicio) / 60:.0f} min", flush=True)
    f_img.close()
    f_seq.close()
    resumen(a, [c["nombre"] for c in lista], {c["nombre"]: c["modelo"] for c in lista})


def ic(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def resumen(a, nombres, modelo_de):
    unicas = {}
    for r in csv.DictReader(open(os.path.join(a.salida, "imagenes.csv"), encoding="utf-8")):
        unicas[(r["secuencia"], r["imagen"])] = r
    filas = list(unicas.values())
    seqs = list(csv.DictReader(open(os.path.join(a.salida, "secuencias.csv"), encoding="utf-8")))
    por_seq = defaultdict(list)
    for r in filas:
        por_seq[r["secuencia"]].append(r)
    orden = [s["secuencia"] for s in seqs]
    rng = np.random.default_rng(a.semilla)
    muestras = rng.integers(0, len(orden), (a.remuestreos, len(orden)))

    def conteos(col, umbral):
        c = np.zeros((len(orden), 3))
        for i, s in enumerate(orden):
            y = np.array([int(r["humo_visible"]) for r in por_seq[s]])
            p = np.array([float(r[col]) >= umbral for r in por_seq[s]])
            c[i] = [(p & (y == 1)).sum(), (p & (y == 0)).sum(), (~p & (y == 1)).sum()]
        return c

    def met(c):
        tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            return tp / (tp + fn), np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)

    detector = {}
    for col in ("humo_conf_640", "fuego_conf_640", "humo_conf_1280", "fuego_conf_1280"):
        for u in (0.25, 0.1):
            c = conteos(col, u)
            rec, pre = met(c.sum(0))
            rb, pb = met(np.stack([c[m].sum(0) for m in muestras]))
            detector[f"{col}|{u:g}"] = {"recall": float(rec), "recall_ic": ic(rb), "precision": float(pre), "precision_ic": ic(pb[~np.isnan(pb)]),
                                         "tp": int(c[:, 0].sum()), "fp": int(c[:, 1].sum()), "fn": int(c[:, 2].sum())}
    trayectorias = {"humo": float(np.mean([int(s["trayectorias_humo"]) > 0 for s in seqs])),
                    "fuego": float(np.mean([int(s["trayectorias_fuego"]) > 0 for s in seqs]))}
    alerta = {}
    for n in nombres + ["regla_humo"]:
        antes, despues, minutos = [], [], []
        for s in orden:
            rs = sorted(por_seq[s], key=lambda r: int(r["k"]))
            if n == "regla_humo":
                vis = [float(r["humo_conf_640"]) >= 0.25 for r in rs]
                act = [i > 0 and vis[i] and vis[i - 1] for i in range(len(vis))]
            else:
                act = [r[f"alerta_{n}"] == "1" for r in rs]
            des = [int(r["desfase_s"]) for r in rs]
            antes.append(any(x for x, d in zip(act, des) if d < 0))
            prim = next((d for x, d in zip(act, des) if x and d >= 0), None)
            despues.append(prim is not None)
            if prim is not None:
                minutos.append(prim / 60.0)
        alerta[n] = {"fa_antes": float(np.mean(antes)), "alerta_despues": float(np.mean(despues)),
                     "minutos_mediana": float(np.median(minutos)) if minutos else None, "minutos": minutos}
    por_modelo = {}
    for m in MODELOS + ["regla_humo"]:
        ns = [n for n in alerta if (modelo_de.get(n) == m) or (n == m)]
        por_modelo[m] = {k: [alerta[n][k] for n in ns] for k in ("fa_antes", "alerta_despues", "minutos_mediana")}
    salida = {"secuencias": len(orden), "imagenes": len(filas), "imagenes_con_humo": int(sum(int(r["humo_visible"]) for r in filas)),
              "detector": detector, "trayectorias": trayectorias, "alerta": alerta, "por_modelo": por_modelo,
              "segundos_por_secuencia": float(np.mean([float(s["segundos"]) for s in seqs]))}
    with open(os.path.join(a.salida, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=1)
    print(f"resumen en {os.path.join(a.salida, 'resumen.json')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("accion", choices=["elegir", "descargar", "evaluar", "resumen"])
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--umbral_color", type=float, default=2.0)
    p.add_argument("--umbral_brillo", type=float, default=50.0)
    p.add_argument("--remuestreos", type=int, default=1000)
    p.add_argument("--lista", default=os.path.join("resultados", "figlib", "muestra60_lista.csv"))
    p.add_argument("--datos", default=os.path.join("..", "FIgLib_piloto"))
    p.add_argument("--config", default=os.path.join("resultados", "configuracion_final.json"))
    p.add_argument("--pesos", default=os.path.join("runs_fuego", "cap5"))
    p.add_argument("--entrenamientos", default=os.path.join("resultados", "cap5", "entrenamientos"))
    p.add_argument("--salida", default=os.path.join("resultados", "figlib", "muestra60"))
    a = p.parse_args()
    if a.accion == "elegir":
        elegir(a)
    elif a.accion == "descargar":
        descargar(a)
    elif a.accion == "evaluar":
        evaluar(a)
    else:
        from salidas_congeladas import leer_config
        lista = corridas_vivo(a, leer_config(a.config))
        resumen(a, [c["nombre"] for c in lista], {c["nombre"]: c["modelo"] for c in lista})


if __name__ == "__main__":
    main()
