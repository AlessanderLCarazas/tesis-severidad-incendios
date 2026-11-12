import os
import csv
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NIVELES = ["controlado", "incipiente", "incontrolado"]
COLORES = {"cfc": "#c0392b", "gru": "#2471a3", "lstm": "#1e8449"}


def leer_historial(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelos", nargs="+", default=["cfc", "gru", "lstm"])
    p.add_argument("--carpeta", default=os.path.join("resultados", "modelos"))
    a = p.parse_args()

    datos = {}
    for m in a.modelos:
        ruta = os.path.join(a.carpeta, m, "metricas.json")
        if os.path.exists(ruta):
            with open(ruta, encoding="utf-8") as f:
                datos[m] = (json.load(f), leer_historial(os.path.join(a.carpeta, m, "historial.csv")))
    if not datos:
        raise SystemExit("no hay modelos entrenados en la carpeta")

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.2))
    for m, (r, h) in datos.items():
        x = [f["epoca"] for f in h]
        ejes[0].plot(x, [f["perdida_ent"] for f in h], color=COLORES.get(m), label=m.upper())
        ejes[1].plot(x, [f["perdida_val"] for f in h], color=COLORES.get(m), label=m.upper())
        ejes[2].plot(x, [f["f1_escena"] for f in h], color=COLORES.get(m), label=m.upper())
        ejes[2].plot(x, [f["f1_muestra"] for f in h], color=COLORES.get(m), linestyle=":", linewidth=1)
        ejes[2].axvline(r["mejor_epoca"], color=COLORES.get(m), linestyle="--", linewidth=0.8)
    ejes[0].set_title("Perdida total en entrenamiento")
    ejes[1].set_title("Perdida total en validacion")
    ejes[2].set_title("F1 macro en validacion\n(linea: escena por segundo; puntos: por muestra)")
    for e in ejes:
        e.set_xlabel("epoca")
        e.grid(alpha=0.3)
        e.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.carpeta, "curvas.png"), dpi=130)

    lineas = ["# Modelos temporales en validacion (fase D)", ""]
    lineas.append("Generado por `python figura_modelos.py`. Metricas de la mejor epoca de cada modelo en el conjunto de"
                  " validacion; la prueba no se usa.")
    lineas.append("")
    lineas.append("| modelo | parametros | s por epoca (entrenar) | s por epoca (validar) | epocas corridas | mejor epoca |"
                  " F1 macro escena | exactitud escena | F1 macro muestra | exactitud muestra | MAE tau (s) |")
    lineas.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m, (r, h) in datos.items():
        v = r["validacion"]
        lineas.append(f"| {m.upper()} | {r['parametros']} | {r['s_por_epoca_ent']:.1f} | {r['s_por_epoca_val']:.1f} |"
                      f" {r['epocas_corridas']} | {r['mejor_epoca']} | {v['f1_escena']:.3f} | {v['exactitud_escena']:.3f} |"
                      f" {v['f1_muestra']:.3f} | {v['exactitud_muestra']:.3f} | {v['mae_tau_s']:.1f} |")
    lineas.append("")
    lineas.append("| modelo | F1 escena controlado | F1 escena incipiente | F1 escena incontrolado | F1 muestra controlado |"
                  " F1 muestra incipiente | F1 muestra incontrolado |")
    lineas.append("|---|---|---|---|---|---|---|")
    for m, (r, h) in datos.items():
        v = r["validacion"]
        lineas.append(f"| {m.upper()} | " + " | ".join(f"{x:.3f}" for x in v["f1_escena_clase"] + v["f1_muestra_clase"]) + " |")
    lineas.append("")
    for m, (r, h) in datos.items():
        v = r["validacion"]
        lineas.append(f"Matriz de confusion por segundo de escena, {m.upper()} (filas: etiqueta de la IA; columnas: modelo;"
                      f" {v['segundos_escena']} s):")
        lineas.append("")
        lineas.append("| real / modelo | " + " | ".join(NIVELES) + " | total |")
        lineas.append("|---|---|---|---|---|")
        for i, n in enumerate(NIVELES):
            fila = v["conf_escena"][i]
            lineas.append(f"| {n} | " + " | ".join(str(x) for x in fila) + f" | {sum(fila)} |")
        lineas.append("")
    r0 = next(iter(datos.values()))[0]
    lineas.append("Frecuencia de niveles con el muestreo por video: "
                  + ", ".join(f"{n} {f:.3f}" for n, f in zip(NIVELES, r0["frecuencia_niveles"]))
                  + ". Pesos de la perdida de nivel: " + ", ".join(f"{n} {w:.3f}" for n, w in zip(NIVELES, r0["pesos_nivel"])) + ".")
    lineas.append("")
    with open(os.path.join(a.carpeta, "comparacion.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
    print("\n".join(lineas))
    print(f"figura en: {os.path.join(a.carpeta, 'curvas.png')}")


if __name__ == "__main__":
    main()
