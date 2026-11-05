import os
import random
import shutil
import argparse
import torch
from ultralytics import YOLO


def enlazar(origen, destino):
    if os.path.exists(destino):
        return
    try:
        os.link(origen, destino)
    except OSError:
        shutil.copy2(origen, destino)


def crear_val_mini(ruta_data, n):
    base = os.path.dirname(os.path.abspath(ruta_data))
    dir_img = os.path.join(base, "images", "val")
    dir_lbl = os.path.join(base, "labels", "val")
    dir_img_mini = os.path.join(base, "images", f"val_mini{n}")
    dir_lbl_mini = os.path.join(base, "labels", f"val_mini{n}")
    os.makedirs(dir_img_mini, exist_ok=True)
    os.makedirs(dir_lbl_mini, exist_ok=True)
    if len(os.listdir(dir_img_mini)) < n:
        nombres = sorted(os.listdir(dir_img))
        random.Random(42).shuffle(nombres)
        for nombre in nombres[:n]:
            enlazar(os.path.join(dir_img, nombre), os.path.join(dir_img_mini, nombre))
            lbl = os.path.splitext(nombre)[0] + ".txt"
            if os.path.exists(os.path.join(dir_lbl, lbl)):
                enlazar(os.path.join(dir_lbl, lbl), os.path.join(dir_lbl_mini, lbl))
    with open(ruta_data, encoding="utf-8") as f:
        contenido = f.read()
    ruta_nueva = os.path.join(base, f"data_entreno{n}.yaml")
    with open(ruta_nueva, "w", encoding="utf-8") as f:
        f.write(contenido.replace("val: images/val", f"val: images/val_mini{n}"))
    print(f"validacion durante el entrenamiento con {len(os.listdir(dir_img_mini))} imagenes de val")
    return ruta_nueva


def buscar_corrida(nombre):
    for ruta in [os.path.join("runs_fuego", nombre), os.path.join("runs", "detect", "runs_fuego", nombre)]:
        if os.path.exists(os.path.join(ruta, "weights", "last.pt")):
            return ruta
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="FASDD_yolo/data.yaml")
    p.add_argument("--modelo", default="yolov8s.pt")
    p.add_argument("--epocas", type=int, default=50)
    p.add_argument("--img", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--fraccion", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val_mini", type=int, default=5000)
    p.add_argument("--nombre", default="fasdd_v8s")
    p.add_argument("--cache", default="no", choices=["no", "disk", "ram"])
    p.add_argument("--reanudar", action="store_true")
    p.add_argument("--evaluar", default="")
    a = p.parse_args()

    print(f"torch {torch.__version__}, cuda disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    else:
        print("aviso: se entrenara en cpu, sera muy lento")
    dispositivo = 0 if torch.cuda.is_available() else "cpu"

    modelo = None
    if a.evaluar:
        pass
    elif a.reanudar:
        carpeta = buscar_corrida(a.nombre)
        if carpeta is None:
            print(f"no se encontro la corrida {a.nombre} para reanudar")
            return
        modelo = YOLO(os.path.join(carpeta, "weights", "last.pt"))
        modelo.train(resume=True)
    else:
        data_entreno = crear_val_mini(a.data, a.val_mini) if a.val_mini > 0 else a.data
        print("pesos iniciales: " + ("ninguno, entrenamiento desde cero" if a.modelo.endswith(".yaml") else a.modelo))
        modelo = YOLO(a.modelo)
        modelo.train(
            data=data_entreno,
            epochs=a.epocas,
            imgsz=a.img,
            batch=a.batch,
            fraction=a.fraccion,
            device=dispositivo,
            workers=a.workers,
            patience=15,
            seed=42,
            amp=True,
            project=os.path.abspath("runs_fuego"),
            name=a.nombre,
            exist_ok=True,
            plots=True,
            cache=a.cache if a.cache != "no" else False,
            pretrained=not a.modelo.endswith(".yaml"),
        )

    if a.evaluar:
        ruta_mejor = a.evaluar
    else:
        carpeta = str(modelo.trainer.save_dir) if getattr(modelo, "trainer", None) else buscar_corrida(a.nombre)
        ruta_mejor = os.path.join(carpeta, "weights", "best.pt")
    print(f"\nmejor modelo: {ruta_mejor}")
    mejor = YOLO(ruta_mejor)
    for split in ["val", "test"]:
        m = mejor.val(data=a.data, split=split, imgsz=a.img, batch=a.batch, device=dispositivo,
                      project=os.path.abspath("runs_fuego"), name=f"{a.nombre}_eval_{split}", exist_ok=True)
        print(f"\n=== resultados en {split} completo ===")
        print(f"map50: {m.box.map50:.4f}, map50-95: {m.box.map:.4f}")
        indices = list(m.box.ap_class_index)
        for i, nombre in m.names.items():
            ap50 = m.box.ap50[indices.index(i)] if i in indices else float("nan")
            print(f"clase {nombre}: ap50 {ap50:.4f}")


if __name__ == "__main__":
    main()