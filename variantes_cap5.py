import numpy as np
import torch
from vector_escena import COLUMNAS, CONTEXTO, DETECCIONES

GRUPOS = {"sin_deteccion": [1], "sin_tamano": [2, 3, 4], "sin_posicion": [5, 6, 7], "sin_tiempo": [8, 9], "sin_humo": [10, 11, 12],
          "sin_escena": [13]}
VARIANTES = ["completo", "sin_camara", "dt_constante", "sin_compuerta"] + list(GRUPOS) + ["sin_detecciones_fotograma"]
NOMBRE_VARIANTE = {"completo": "sistema completo", "sin_camara": "a. sin contexto de cámara en la compuerta", "dt_constante": "b. dt constante de 0,1 s",
                   "sin_compuerta": "c. sin compuerta", "sin_deteccion": "e. sin detección (d1)", "sin_tamano": "e. sin tamaño y crecimiento (d2 a d4)",
                   "sin_posicion": "e. sin posición y movimiento (d5 a d7)", "sin_tiempo": "e. sin comportamiento en el tiempo (d8 y d9)",
                   "sin_humo": "e. sin humo (d10 a d12)", "sin_escena": "e. sin escena (d13)",
                   "sin_detecciones_fotograma": "e. sin estadísticas de detección del fotograma",
                   "sin_compensacion": "f. sin compensación de cámara"}
ABLACIONES_EXTRA = ["sin_compensacion"]
DT_FIJO = 0.1


def columnas_x(variante):
    quitar = set()
    if variante in GRUPOS:
        for i in GRUPOS[variante]:
            quitar |= {f"d{i}_max", f"d{i}_media"}
    if variante == "sin_detecciones_fotograma":
        quitar |= set(DETECCIONES)
    return [j for j, c in enumerate(COLUMNAS) if c not in quitar]


def columnas_c(variante):
    return [CONTEXTO.index("dt")] if variante == "sin_camara" else list(range(len(CONTEXTO)))


def preparar(datos, variante, norm_c):
    salida = dict(datos)
    ix = torch.tensor(columnas_x(variante), device=datos["x"].device)
    ic = torch.tensor(columnas_c(variante), device=datos["c"].device)
    salida["x"] = datos["x"].index_select(1, ix)
    salida["c"] = datos["c"].index_select(1, ic).clone()
    if variante == "dt_constante":
        j = CONTEXTO.index("dt")
        valor = float((np.clip(DT_FIJO, norm_c.bajo[j], norm_c.alto[j]) - norm_c.media[j]) / norm_c.desv[j])
        salida["ts"] = torch.full_like(datos["ts"], DT_FIJO)
        salida["c"][:, columnas_c(variante).index(j)] = valor
    return salida
