import torch
import torch.nn as nn
from ncps.torch import CfCCell
from modelo_temporal import TIPOS, nivel_de, perdidas


class ModeloEscena(nn.Module):
    def __init__(self, tipo="cfc", n_x=34, n_c=6, unidades=64, dropout=0.2):
        super().__init__()
        if tipo not in TIPOS:
            raise ValueError(f"tipo de modelo desconocido: {tipo}")
        self.tipo, self.unidades = tipo, unidades
        self.compuerta = nn.Linear(n_c, n_x)
        self.dropout_entrada = nn.Dropout(dropout)
        self.dropout_salida = nn.Dropout(dropout)
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
        u = self.dropout_entrada(x * torch.sigmoid(self.compuerta(c)))
        if self.tipo == "cfc":
            h = torch.zeros(x.shape[0], self.unidades, device=x.device, dtype=x.dtype)
            salidas = []
            for k in range(x.shape[1]):
                _, h = self.celda(u[:, k], h, ts[:, k:k + 1])
                salidas.append(h)
            h = torch.stack(salidas, 1)
        else:
            h, _ = self.rnn(torch.cat([u, ts.unsqueeze(-1)], -1))
        h = self.dropout_salida(h)
        return self.coral(h) + self.sesgos, self.riesgo(h).squeeze(-1), self.tau(h).squeeze(-1)


__all__ = ["ModeloEscena", "TIPOS", "nivel_de", "perdidas"]
