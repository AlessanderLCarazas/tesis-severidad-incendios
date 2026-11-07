import argparse
import csv
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=".")
ap.add_argument("--salida", default="docs/img/cortes_regla.png")
args = ap.parse_args()
raiz, salida = args.raiz, args.salida
ind = collections.defaultdict(list)
for r in csv.DictReader(open(f"{raiz}/resultados/cortes/indicadores/indicadores.csv", encoding="utf-8")):
    ind[r["archivo"]].append(r)
pares = []
for r in csv.DictReader(open(f"{raiz}/resultados/cortes/confirmados.csv", encoding="utf-8")):
    m = min(ind[r["archivo"]], key=lambda x: abs(float(x["t"]) - float(r["t_corte"])))
    pares.append((r["tipo"] == "corte", float(m["h"]), m["valido"] == "1"))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False})
azul, naranja, tinta, gris = "#1f5a99", "#d9822b", "#333333", "#888888"
fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
filas = {True: 1.0, False: 0.0}
umbral = {True: 0.85, False: 0.65}
for valida, y in filas.items():
    ax.add_patch(Rectangle((umbral[valida], y - 0.38), 1.02 - umbral[valida], 0.76, color="#f2e6da", zorder=0, lw=0))
    ax.plot([umbral[valida]] * 2, [y - 0.38, y + 0.38], color=gris, lw=1.2, ls="--", zorder=1)
    ax.text(umbral[valida] + 0.008, y + 0.31, f"umbral {umbral[valida]:.2f}".replace(".", ","), fontsize=8.5, color=tinta, va="center")
rng = np.random.default_rng(7)
for corte, color, marca, nombre in [(False, azul, "o", "Sin corte (46)"), (True, naranja, "^", "Corte real (37)")]:
    xs = [h for c, h, v in pares if c == corte]
    ys = [filas[v] + rng.uniform(-0.22, 0.22) for c, h, v in pares if c == corte]
    ax.scatter(xs, ys, s=46, marker=marca, color=color, edgecolor="white", linewidth=1.2, label=nombre, zorder=3)
det = sum(1 for c, h, v in pares if c and h >= umbral[v])
fal = sum(1 for c, h, v in pares if not c and h >= umbral[v])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Cámara no válida", "Cámara válida"], color=tinta)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0.2, 1.02)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel("Distancia de Bhattacharyya entre histogramas consecutivos (h)", color=tinta)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:.1f}".replace(".", ",")))
ax.grid(axis="x", color="#e5e5e5", lw=0.8)
ax.set_axisbelow(True)
ax.set_title(f"Regla de cortes en los 83 pares revisados: {det} de 37 cortes detectados, {fal} falsos", loc="left", fontsize=11, color=tinta)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=3, frameon=False, fontsize=9,
          handles=ax.get_legend_handles_labels()[0] + [Rectangle((0, 0), 1, 1, color="#f2e6da")],
          labels=ax.get_legend_handles_labels()[1] + ["Zona donde se marca corte"])
fig.tight_layout()
fig.savefig(salida, facecolor="white")
print(f"pares {len(pares)} | cortes detectados {det} | falsos {fal} | figura {salida}")
