import os


def escapar(texto):
    return str(texto).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def miles(n):
    s = f"{int(round(n)):d}"
    negativo = s.startswith("-")
    s = s.lstrip("-")
    grupos = []
    while len(s) > 3:
        grupos.insert(0, s[-3:])
        s = s[:-3]
    grupos.insert(0, s)
    return ("$-$" if negativo else "") + "\\,".join(grupos)


def num(x, d=3, signo=False):
    if x is None:
        return "--"
    texto = f"{abs(x):.{d}f}".replace(".", "{,}")
    if x < 0 and float(f"{abs(x):.{d}f}") != 0:
        return "$-$" + texto
    return ("+" if signo and x > 0 else "") + texto


def pm(media, desv, d=3, signo=False):
    if media is None:
        return "--"
    if desv is None:
        return num(media, d, signo)
    return f"{num(media, d, signo)} $\\pm$ {num(desv, d)}"


def pct(x, d=1):
    return "--" if x is None else f"{num(100.0 * x, d)}\\,\\%"


def tabular(ruta, encabezados, filas, alineacion=None):
    n = len(encabezados)
    al = alineacion or ("l" + "c" * (n - 1))
    lineas = ["\\begin{tabular}{|" + "|".join(al) + "|}", "\\hline"]
    lineas.append(" & ".join(f"\\textbf{{{h}}}" for h in encabezados) + " \\\\ \\hline")
    for fila in filas:
        if len(fila) != n:
            raise ValueError(f"fila con {len(fila)} celdas en una tabla de {n} columnas")
        lineas.append(" & ".join(fila) + " \\\\ \\hline")
    lineas.append("\\end{tabular}")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas) + "\n")
    return ruta
