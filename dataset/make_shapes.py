#!/usr/bin/env python3
"""
Genera mallas .obj (triángulos) para la Tarea B. Unidades: milímetros. Cada malla está centrada en (0,0)
en el plano x-y y se extiende en z de -H/2 a +H/2 (los bigotes baten aproximadamente en el plano z≈0).

Formas:
  cylinder_r{R}.obj   cilindro vertical de radio R (como el poste, pero con radio variable)
  prism_{A}x{B}.obj   prisma rectangular de lados A (en x) y B (en y); al rotarlo con ENV_YAW cambia la cara que tocan
  wave_a{A}_l{L}.obj  pared ondulada: superficie x = A·sin(2π y / L), extruida en z (contorno no convexo)

Por qué triángulos explícitos: WHISKiT construye la malla de colisión a partir de la lista de vértices de 3 en 3,
así que escribimos cada triángulo con sus 3 vértices (sin compartir índices) para que la colisión sea exacta.
"""
import argparse, math, os
import numpy as np

H = 60.0  # altura (mm) suficiente para cubrir el plano de batido


def write_obj(path, tris):
    """tris: lista de triángulos, cada uno 3 puntos (x,y,z). Se escriben vértices sin compartir."""
    with open(path, "w") as f:
        f.write("# generado por make_shapes.py (mm)\n")
        for t in tris:
            for p in t:
                f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        for i in range(len(tris)):
            b = 3 * i + 1
            f.write(f"f {b} {b+1} {b+2}\n")


def extrude_polyline(pts, closed, h=H):
    """Convierte una polilínea 2D en una banda vertical de triángulos (dos por segmento)."""
    tris = []
    n = len(pts)
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        a, b = pts[i], pts[(i + 1) % n]
        a0, a1 = (a[0], a[1], -h / 2), (a[0], a[1], h / 2)
        b0, b1 = (b[0], b[1], -h / 2), (b[0], b[1], h / 2)
        tris.append((a0, b0, b1)); tris.append((a0, b1, a1))
    return tris


def cylinder(r, n=48):
    pts = [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    return extrude_polyline(pts, closed=True)


def prism(a, b):
    pts = [(-a / 2, -b / 2), (a / 2, -b / 2), (a / 2, b / 2), (-a / 2, b / 2)]
    return extrude_polyline(pts, closed=True)


def wave(amp, lam, length=30.0, n=60, thickness=5.0):
    """Pared cuyo frente sigue x = amp*sin(2π y/λ); cerrada por detrás para que sea un sólido."""
    ys = np.linspace(-length / 2, length / 2, n)
    front = [(amp * math.sin(2 * math.pi * y / lam), y) for y in ys]
    back = [(x - thickness - amp, y) for x, y in front][::-1]
    return extrude_polyline(front + back, closed=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    specs = {}
    for r in [1.0, 2.5, 5.0]:
        specs[f"cylinder_r{r:g}.obj"] = cylinder(r)
    for A, B in [(4, 12), (8, 8), (6, 20)]:
        specs[f"prism_{A}x{B}.obj"] = prism(A, B)
    for amp, lam in [(2, 15), (4, 25), (3, 10)]:
        specs[f"wave_a{amp}_l{lam}.obj"] = wave(amp, lam)
    for name, tris in specs.items():
        write_obj(os.path.join(a.out, name), tris)
        print(f"{name:22s} {len(tris):4d} triángulos")


if __name__ == "__main__":
    main()
