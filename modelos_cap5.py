import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig
from modelo_escena import ModeloEscena

TIPOS = ["cfc", "gru", "lstm", "mamba"]


class ModeloAblacion(ModeloEscena):
    def __init__(self, tipo="cfc", n_x=34, n_c=6, unidades=64, dropout=0.2, sin_compuerta=False):
        super().__init__(tipo, n_x=n_x, n_c=n_c, unidades=unidades, dropout=dropout)
        self.sin_compuerta = sin_compuerta
        if sin_compuerta:
            del self.compuerta

    def forward(self, x, c, ts):
        if not self.sin_compuerta:
            return super().forward(x, c, ts)
        u = self.dropout_entrada(x)
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


class ModeloMamba(nn.Module):
    def __init__(self, n_x=34, n_c=6, unidades=64, dropout=0.2, sin_compuerta=False, capas=1, d_state=16, expansion=2, d_conv=4):
        super().__init__()
        self.tipo, self.unidades, self.sin_compuerta = "mamba", unidades, sin_compuerta
        if not sin_compuerta:
            self.compuerta = nn.Linear(n_c, n_x)
        self.dropout_entrada = nn.Dropout(dropout)
        self.dropout_salida = nn.Dropout(dropout)
        self.entrada = nn.Linear(n_x + 1, unidades)
        self.mamba = Mamba(MambaConfig(d_model=unidades, n_layers=capas, d_state=d_state, expand_factor=expansion, d_conv=d_conv))
        self.coral = nn.Linear(unidades, 1, bias=False)
        self.sesgos = nn.Parameter(torch.zeros(2))
        self.riesgo = nn.Linear(unidades, 1)
        self.tau = nn.Linear(unidades, 1)
        self.log_var = nn.Parameter(torch.zeros(3))
        self.config_mamba = {"capas": capas, "d_state": d_state, "expansion": expansion, "d_conv": d_conv}

    def entrada_compuerta(self, x, c):
        return self.dropout_entrada(x if self.sin_compuerta else x * torch.sigmoid(self.compuerta(c)))

    def forward(self, x, c, ts):
        u = self.entrada_compuerta(x, c)
        h = self.mamba(self.entrada(torch.cat([u, ts.unsqueeze(-1)], -1)))
        h = self.dropout_salida(h)
        return self.coral(h) + self.sesgos, self.riesgo(h).squeeze(-1), self.tau(h).squeeze(-1)

    def cache_inicial(self, lote, dispositivo):
        cfg = self.mamba.config
        ed = cfg.expand_factor * cfg.d_model
        return [(torch.zeros(lote, ed, cfg.d_state, device=dispositivo), torch.zeros(lote, ed, cfg.d_conv - 1, device=dispositivo))
                for _ in range(cfg.n_layers)]

    def paso(self, x, c, ts, cache):
        u = self.entrada_compuerta(x, c)
        h, cache = self.mamba.step(self.entrada(torch.cat([u, ts], -1)), cache)
        return self.coral(h) + self.sesgos, self.riesgo(h), self.tau(h), cache


def crear_modelo(tipo, n_x, n_c, unidades=64, dropout=0.2, sin_compuerta=False):
    if tipo == "mamba":
        return ModeloMamba(n_x=n_x, n_c=n_c, unidades=unidades, dropout=dropout, sin_compuerta=sin_compuerta)
    return ModeloAblacion(tipo, n_x=n_x, n_c=n_c, unidades=unidades, dropout=dropout, sin_compuerta=sin_compuerta)


def cargar_modelo(ruta, dispositivo="cpu"):
    estado = torch.load(ruta, map_location=dispositivo, weights_only=False)
    modelo = crear_modelo(estado["tipo"], estado["n_x"], estado.get("n_c", 6), estado["unidades"], estado["dropout"],
                          estado.get("variante") == "sin_compuerta").to(dispositivo)
    modelo.load_state_dict(estado["modelo"])
    modelo.eval()
    return modelo, estado
