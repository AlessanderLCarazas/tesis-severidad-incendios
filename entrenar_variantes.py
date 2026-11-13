import os
import json
import copy
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from entrenar_temporal import lote, verdad_escena
from entrenar_escena import cargar, frecuencias
from fase_e import histeresis, por_segundo
from secuencias import leer_csv

VARIANTES = ["base", "softmax", "peso_incipiente", "solo_nivel"]


class GRUVariante(nn.Module):
    def __init__(self, n_x, n_c=6, unidades=64, dropout=0.2, softmax=False):
        super().__init__()
        self.softmax = softmax
        self.compuerta = nn.Linear(n_c, n_x)
        self.dropout_entrada = nn.Dropout(dropout)
        self.dropout_salida = nn.Dropout(dropout)
        self.rnn = nn.GRU(n_x + 1, unidades, batch_first=True)
        if softmax:
            self.clases = nn.Linear(unidades, 3)
        else:
            self.coral = nn.Linear(unidades, 1, bias=False)
            self.sesgos = nn.Parameter(torch.zeros(2))
        self.riesgo = nn.Linear(unidades, 1)
        self.tau = nn.Linear(unidades, 1)
        self.log_var = nn.Parameter(torch.zeros(3))

    def forward(self, x, c, ts):
        u = self.dropout_entrada(x * torch.sigmoid(self.compuerta(c)))
        h, _ = self.rnn(torch.cat([u, ts.unsqueeze(-1)], -1))
        h = self.dropout_salida(h)
        nivel = self.clases(h) if self.softmax else self.coral(h) + self.sesgos
        return nivel, self.riesgo(h).squeeze(-1), self.tau(h).squeeze(-1)

    def nivel(self, logits):
        return logits.argmax(-1) if self.softmax else (logits > 0).sum(-1)


def perdida(modelo, salida, b, pesos, solo_nivel):
    logits, r, lt = salida
    m = b["mascara"].float()
    n = m.sum().clamp(min=1.0)
    if modelo.softmax:
        por_muestra = F.cross_entropy(logits.reshape(-1, 3), b["nivel"].reshape(-1), reduction="none").reshape(b["nivel"].shape)
    else:
        metas = torch.stack([(b["nivel"] > 0).float(), (b["nivel"] > 1).float()], -1)
        por_muestra = F.binary_cross_entropy_with_logits(logits, metas, reduction="none").sum(-1)
    l_nivel = (por_muestra * pesos[b["nivel"]] * m).sum() / n
    if solo_nivel:
        return l_nivel
    l_riesgo = (F.binary_cross_entropy_with_logits(r, b["riesgo"], reduction="none") * m).sum() / n
    mt = b["tau_valida"].float()
    l_tau = (F.huber_loss(lt, torch.log1p(b["tau"]), reduction="none", delta=1.0) * mt).sum() / mt.sum().clamp(min=1.0)
    s = modelo.log_var
    total = torch.exp(-s[0]) * l_nivel + s[0] + torch.exp(-s[1]) * l_riesgo + s[1]
    if mt.sum() > 0:
        total = total + torch.exp(-s[2]) * l_tau + s[2]
    return total


@torch.no_grad()
def niveles_video(modelo, datos, info, tam_lote=64):
    modelo.eval()
    ini = info["inicio"]
    orden = np.argsort(ini[1:] - ini[:-1])
    pred = np.zeros(len(info["t"]), np.int64)
    for i in range(0, len(orden), tam_lote):
        tramos = [(int(ini[j]), int(ini[j + 1])) for j in orden[i:i + tam_lote]]
        b = lote(datos, tramos)
        niveles = modelo.nivel(modelo(b["x"], b["c"], b["ts"])[0]).cpu().numpy()
        for fila, (a0, a1) in enumerate(tramos):
            pred[a0:a1] = niveles[fila, :a1 - a0]
    archivo = np.repeat(info["archivo"], ini[1:] - ini[:-1])
    salida, tiempos = {}, {}
    for v in np.unique(archivo):
        sel = np.flatnonzero(archivo == v)
        sel = sel[np.argsort(info["t"][sel], kind="stable")]
        salida[str(v)], tiempos[str(v)] = pred[sel], info["t"][sel].astype(np.float64)
    return salida, tiempos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variante", choices=VARIANTES, default="base")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tasas", default=os.path.join("resultados", "modelos_escena_v2", "tasas.json"))
    p.add_argument("--datos", default=os.path.join("runs_fuego", "escena_v2"))
    p.add_argument("--metas", default=os.path.join("resultados", "particion", "metas.csv"))
    p.add_argument("--unidades", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--decaimiento", type=float, default=1e-3)
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--recorte", type=float, default=1.0)
    p.add_argument("--epocas", type=int, default=60)
    p.add_argument("--tope_peso", type=float, default=5.0)
    p.add_argument("--factor_incipiente", type=float, default=2.0)
    p.add_argument("--permanencia", type=float, default=2.0)
    p.add_argument("--dispositivo", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--salida", default=os.path.join("resultados", "variantes_gru"))
    a = p.parse_args()

    if a.lr is None:
        with open(a.tasas, encoding="utf-8") as f:
            a.lr = float(json.load(f)["gru"]["elegida"])
    random.seed(a.semilla)
    np.random.seed(a.semilla)
    torch.manual_seed(a.semilla)
    torch.cuda.manual_seed_all(a.semilla)
    rng = np.random.default_rng(a.semilla)
    ent, info_ent = cargar(os.path.join(a.datos, "entrenamiento.npz"), a.dispositivo)
    val, info_val = cargar(os.path.join(a.datos, "validacion.npz"), a.dispositivo)
    metas = {m["archivo"]: m for m in leer_csv(a.metas) if m["fuente"] == "ia" and m["conjunto"] == "validacion"}
    verdad = verdad_escena(metas, sorted(metas))
    por_video, frec = frecuencias(info_ent)
    videos = sorted(por_video)
    pesos_np = np.minimum(1.0 / (3.0 * np.maximum(frec, 1e-9)), a.tope_peso)
    if a.variante == "peso_incipiente":
        pesos_np[1] *= a.factor_incipiente
    pesos = torch.tensor(pesos_np, dtype=torch.float32, device=a.dispositivo)
    lotes_epoca = int(np.ceil(len(info_ent["ventanas"]) / a.lote))
    modelo = GRUVariante(ent["x"].shape[1], ent["c"].shape[1], a.unidades, a.dropout, softmax=a.variante == "softmax").to(a.dispositivo)
    opt = torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=a.decaimiento)
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epocas * lotes_epoca)
    inicio = time.time()
    f1s, estados = [], []
    for epoca in range(1, a.epocas + 1):
        modelo.train()
        for _ in range(lotes_epoca):
            tramos = []
            for v in rng.integers(0, len(videos), a.lote):
                vs = por_video[videos[v]]
                a0, a1, _ = vs[rng.integers(0, len(vs))]
                tramos.append((a0, a1))
            b = lote(ent, tramos)
            total = perdida(modelo, modelo(b["x"], b["c"], b["ts"]), b, pesos, a.variante == "solo_nivel")
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), a.recorte)
            opt.step()
            plan.step()
        niveles, tiempos = niveles_video(modelo, val, info_val)
        f1s.append(por_segundo(niveles, tiempos, verdad)["f1"])
        estados.append(copy.deepcopy({k: v.detach().cpu() for k, v in modelo.state_dict().items()}))
    suave = [float(np.mean(f1s[max(0, e - 1):e + 2])) for e in range(len(f1s))]
    mejor = int(np.argmax(suave))
    modelo.load_state_dict(estados[mejor])
    niveles, tiempos = niveles_video(modelo, val, info_val)
    estable = {v: histeresis(niveles[v], tiempos[v], a.permanencia) for v in niveles}
    res = {"variante": a.variante, "semilla": a.semilla, "lr": a.lr, "mejor_epoca": mejor + 1, "f1_suavizado": suave[mejor],
           "pesos_nivel": pesos_np.tolist(), "segundos": time.time() - inicio,
           "antes": por_segundo(niveles, tiempos, verdad), "histeresis": por_segundo(estable, tiempos, verdad)}
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, f"{a.variante}_s{a.semilla}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"{a.variante} s{a.semilla} | mejor epoca {mejor + 1} | f1 {res['antes']['f1']:.3f} -> {res['histeresis']['f1']:.3f}"
          f" | incipiente {res['antes']['f1_clase'][1]:.3f} -> {res['histeresis']['f1_clase'][1]:.3f}"
          f" | kappa {res['antes']['kappa']:.3f} -> {res['histeresis']['kappa']:.3f} | {res['segundos']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
