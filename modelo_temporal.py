import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfCCell

TIPOS = ["cfc", "gru", "lstm"]


class ModeloTemporal(nn.Module):
    def __init__(self, tipo="cfc", n_x=13, n_c=6, unidades=64):
        super().__init__()
        if tipo not in TIPOS:
            raise ValueError(f"tipo de modelo desconocido: {tipo}")
        self.tipo, self.unidades = tipo, unidades
        self.compuerta = nn.Linear(n_c, n_x)
        if tipo == "cfc":
            self.celda = CfCCell(n_x, unidades)
        elif tipo == "gru":
            self.rnn = nn.GRU(n_x + 1, unidades, batch_first=True)
        else:
            self.rnn = nn.LSTM(n_x + 1, unidades, batch_first=True)
        self.coral = nn.Linear(unidades, 1, bias=False)
        self.sesgos = nn.Parameter(torch.zeros(2))
        self.riesgo = nn.Linear(unidades, 1)
        self.tau = nn.Linear(unidades, 1)
        self.log_var = nn.Parameter(torch.zeros(3))

    def forward(self, x, c, ts):
        u = x * torch.sigmoid(self.compuerta(c))
        if self.tipo == "cfc":
            h = torch.zeros(x.shape[0], self.unidades, device=x.device, dtype=x.dtype)
            salidas = []
            for k in range(x.shape[1]):
                _, h = self.celda(u[:, k], h, ts[:, k:k + 1])
                salidas.append(h)
            h = torch.stack(salidas, 1)
        else:
            h, _ = self.rnn(torch.cat([u, ts.unsqueeze(-1)], -1))
        return self.coral(h) + self.sesgos, self.riesgo(h).squeeze(-1), self.tau(h).squeeze(-1)


def nivel_de(logits):
    return (logits > 0).sum(-1)


def perdidas(modelo, salida, nivel, riesgo, mascara, tau, tau_valida, pesos_nivel):
    logits, r, lt = salida
    m = mascara.float()
    n = m.sum().clamp(min=1.0)
    metas = torch.stack([(nivel > 0).float(), (nivel > 1).float()], -1)
    bce = F.binary_cross_entropy_with_logits(logits, metas, reduction="none").sum(-1)
    l_nivel = (bce * pesos_nivel[nivel] * m).sum() / n
    l_riesgo = (F.binary_cross_entropy_with_logits(r, riesgo, reduction="none") * m).sum() / n
    mt = tau_valida.float()
    hay_tau = mt.sum() > 0
    l_tau = (F.huber_loss(lt, torch.log1p(tau), reduction="none", delta=1.0) * mt).sum() / mt.sum().clamp(min=1.0)
    s = modelo.log_var
    total = torch.exp(-s[0]) * l_nivel + s[0] + torch.exp(-s[1]) * l_riesgo + s[1]
    if hay_tau:
        total = total + torch.exp(-s[2]) * l_tau + s[2]
    return total, l_nivel, l_riesgo, l_tau, hay_tau
