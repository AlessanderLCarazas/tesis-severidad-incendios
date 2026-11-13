import os
import sys
import json
import argparse
import torch
import entrenar_final


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--unidad_dt", type=float, default=0.1)
    p.add_argument("--modelo", default="cfc")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--factor_incipiente", type=float, default=1.0)
    p.add_argument("--pesos", default=os.path.join("runs_fuego", "modelos_final", "cfc_dt"))
    p.add_argument("--salida", default=os.path.join("resultados", "modelos_final", "cfc_dt"))
    a, resto = p.parse_known_args()

    original = entrenar_final.cargar

    def cargar(ruta, dispositivo):
        datos, info = original(ruta, dispositivo)
        datos["ts"] = datos["ts"] / a.unidad_dt
        return datos, info

    entrenar_final.cargar = cargar
    sys.argv = [sys.argv[0], "--modelo", a.modelo, "--semilla", str(a.semilla), "--lr", str(a.lr),
                "--factor_incipiente", str(a.factor_incipiente), "--pesos", a.pesos, "--salida", a.salida] + resto
    print(f"intervalo del modelo en unidades de {a.unidad_dt:g} s", flush=True)
    entrenar_final.main()
    nombre = f"{a.modelo}_s{a.semilla}"
    ruta = os.path.join(a.pesos, nombre, "mejor.pt")
    estado = torch.load(ruta, map_location="cpu", weights_only=False)
    estado["unidad_dt"] = a.unidad_dt
    torch.save(estado, ruta)
    ruta = os.path.join(a.salida, nombre, "metricas.json")
    with open(ruta, encoding="utf-8") as f:
        r = json.load(f)
    r["unidad_dt"] = a.unidad_dt
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    print(f"unidad de tiempo {a.unidad_dt:g} s guardada en {ruta}")


if __name__ == "__main__":
    main()
