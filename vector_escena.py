import numpy as np

DESC = [f"d{i}" for i in range(1, 14)]
CAMARA = ["s_k", "rot", "tx", "ty", "inliers"]
CONTEXTO = CAMARA + ["dt"]
DETECCIONES = ["n_det_fuego", "n_det_humo", "area_det_fuego", "area_det_humo"]
COLUMNAS = ([f"{d}_max" for d in DESC] + [f"{d}_media" for d in DESC]
            + ["n_tray", "n_tray_1s", "edad_max"] + DETECCIONES + ["sin_trayectoria"])


class VectorEscena:
    def __init__(self, edad_min=1.0):
        self.edad_min = edad_min
        self.nacimiento = {}

    def reiniciar(self):
        self.nacimiento = {}

    def paso(self, t, trayectorias, detecciones):
        t = round(float(t), 4)
        v = np.zeros(len(COLUMNAS), np.float64)
        if len(trayectorias):
            d = np.round(np.array([x for _, x in trayectorias], np.float64).reshape(len(trayectorias), 13), 6)
            edades = np.array([t - self.nacimiento.setdefault(i, t) for i, _ in trayectorias])
            v[:13] = d.max(axis=0)
            v[13:26] = d.mean(axis=0)
            v[26] = len(trayectorias)
            v[27] = float((edades >= self.edad_min - 1e-6).sum())
            v[28] = edades.max()
        else:
            v[33] = 1.0
        v[29:33] = [float(detecciones[c]) for c in DETECCIONES]
        return v


def contexto(mov, dt):
    return np.array([float(mov[c]) for c in CAMARA] + [float(dt)], np.float64)


def vectores_video(fotogramas, filas_por_k, edad_min=1.0):
    calculo = VectorEscena(edad_min)
    x = np.zeros((len(fotogramas), len(COLUMNAS)), np.float64)
    c = np.zeros((len(fotogramas), len(CONTEXTO)), np.float64)
    tramo_previo = None
    for n, f in enumerate(fotogramas):
        if f["tramo"] != tramo_previo:
            calculo.reiniciar()
            tramo_previo = f["tramo"]
        x[n] = calculo.paso(f["t"], filas_por_k.get(f["k"], []), f)
        c[n] = contexto(f, f["dt"])
    return x, c
