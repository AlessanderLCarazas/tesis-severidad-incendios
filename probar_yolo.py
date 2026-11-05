import os
import time
import argparse
import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default=os.path.join("runs_fuego", "fasdd_v8s_full", "weights", "best.pt"))
    p.add_argument("--fuente", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--img", type=int, default=640)
    p.add_argument("--mostrar", action="store_true")
    p.add_argument("--nombre", default="prueba")
    a = p.parse_args()

    fuente = int(a.fuente) if a.fuente.isdigit() else a.fuente
    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    modelo = YOLO(a.modelo)
    print(f"modelo: {a.modelo}")
    print(f"clases: {modelo.names}")

    cuadros = 0
    con_fuego = 0
    con_humo = 0
    inicio = time.time()
    resultados = modelo.predict(
        source=fuente,
        conf=a.conf,
        imgsz=a.img,
        device=dispositivo,
        stream=True,
        save=True,
        show=a.mostrar,
        project=os.path.abspath(os.path.join("runs_fuego", "pruebas")),
        name=a.nombre,
        exist_ok=True,
        verbose=False,
    )
    for r in resultados:
        cuadros += 1
        nombres = [modelo.names[int(c)] for c in r.boxes.cls.tolist()] if r.boxes is not None else []
        con_fuego += "fuego" in nombres
        con_humo += "humo" in nombres
    duracion = time.time() - inicio

    print(f"\ncuadros procesados: {cuadros}")
    print(f"cuadros con fuego: {con_fuego}")
    print(f"cuadros con humo: {con_humo}")
    if duracion > 0:
        print(f"velocidad: {cuadros / duracion:.1f} cuadros por segundo")
    print(f"resultado guardado en: {os.path.join('runs_fuego', 'pruebas', a.nombre)}")


if __name__ == "__main__":
    main()