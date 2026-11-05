#!/usr/bin/env python3
"""
Tarea C: RECONSTRUCCIÓN DEL CONTORNO. Cada simulación tiene su propio objeto: una pared cuyo frente (la cara
que mira a la cabeza) sigue un perfil aleatorio, y la etiqueta es ese perfil muestreado en el marco de la cabeza.

Parametrización del perfil (en el marco del objeto, y a lo largo del frente):
    p(y) = sum_{k=1..K} a_k * cos(2*pi*k*y/L + phi_k),   K = 3, L = 30 mm (longitud del frente)
El frente queda en x_front(y) = x0 - p(y)  (hacia la cabeza, que está en x < x0), y la pared se cierra por detrás.

Etiqueta (lo que el modelo debe predecir): el frente en el MARCO DE LA CABEZA, muestreado en una rejilla fija
de M = 16 puntos y_k en [y0 - L/2, y0 + L/2]:   y_label = x_front(y_k) - 35   (centrado para que ~0 sea "distancia típica").
Con eso, la etiqueta codifica a la vez la forma (coeficientes) y la posición (x0, y0), como pide el título del paper.
Se guardan además los coeficientes y la pose por si se quiere evaluar por separado.

Uso:  python3 run_dataset_C.py --n 12 --seed 0 --out pilot_C --time 1.0 --jobs 2
"""
import argparse, csv, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_shapes import write_obj, extrude_polyline          # reutilizamos el escritor de mallas
from run_dataset_B import inside                               # test punto-en-polígono

CODE = os.path.abspath(os.path.join(HERE, "..", "whiskitphysics", "code"))
BIN = os.path.join(CODE, "build", "whiskit"); SCRIPTS = os.path.join(CODE, "scripts")
GEO = np.load(os.path.join(HERE, "whisker_geometry.npz"))

K, L, M, THICK = 3, 24.0, 16, 6.0
A_MAX = [4.0, 2.5, 1.5]                     # amplitud máxima por armónico (mm): perfiles suaves pero no triviales
X0_RANGE, Y0_RANGE = (49.0, 56.0), (14.0, 22.0)
YGRID = np.linspace(-L / 2, L / 2, M)       # rejilla relativa al centro del objeto


def profile(coef, ys):
    a, ph = coef[:K], coef[K:]
    return sum(a[k] * np.cos(2 * np.pi * (k + 1) * ys / L + ph[k]) for k in range(K))


def footprint(coef, x0, y0, n=60):
    ys = np.linspace(-L / 2, L / 2, n)
    front = [(x0 - p, y0 + y) for p, y in zip(profile(coef, ys), ys)]
    xb = x0 + THICK + max(0.0, -profile(coef, ys).min())      # cara trasera plana, detrás del punto más atrás
    back = [(xb, y0 + y) for y in ys][::-1]
    return np.array(front + back, dtype=np.float32)


def pose_ok(poly, margin=1.5):
    ctr = poly.mean(0); infl = ctr + (poly - ctr) * (1 + margin / max(np.abs(poly - ctr).max(), 1e-6))
    if inside(infl, GEO["rest"]).any():
        return False
    return inside(poly, GEO["sweep"]).sum() >= 20             # alcanzable en una porción apreciable del frente


def sample(n, seed):
    rng = np.random.default_rng(seed); out, tries = [], 0
    while len(out) < n:
        tries += 1
        coef = np.concatenate([rng.uniform(0, A_MAX), rng.uniform(0, 2 * np.pi, K)])
        x0, y0 = rng.uniform(*X0_RANGE), rng.uniform(*Y0_RANGE)
        poly = footprint(coef, x0, y0)
        if pose_ok(poly):
            label = (x0 - profile(coef, YGRID)) - 50.0          # frente en marco de la cabeza, centrado
            out.append(dict(coef=coef, x0=x0, y0=y0, label=label, poly=poly))
    print(f"[sample] {n} contornos válidos de {tries} intentos", flush=True)
    return out


def run_one(i, s, out_dir, t_stop):
    d = os.path.join(out_dir, f"sim_{i:04d}"); flag = os.path.join(d, "DONE")
    if os.path.exists(flag):
        return i, "skip", 0.0
    os.makedirs(d, exist_ok=True)
    mesh = os.path.join(d, "contour.obj")
    # La malla ya está en el marco de la cabeza (x0, y0 incluidos) -> ENV_LOC = 0 0 0, sin rotación.
    write_obj(mesh, extrude_polyline([tuple(p) for p in s["poly"]], closed=True))
    cmd = [BIN, "--MODEL_TYPE", "0", "--DEBUG", "0", "--PRINT", "0", "--OBJECT", "3",
           "--file_env", os.path.abspath(mesh), "--ENV_SCALE", "0", "--ENV_LOC", "0", "0", "0", "--ENV_YAW", "0",
           "--TIME_STOP", f"{t_stop}", "--ACTIVE", "1", "--WHISKER_NAMES", "R",
           "--SAVE", "1", "--SAVE_VIDEO", "0", "--dir_out", d]
    t0 = time.time()
    with open(os.path.join(d, "whiskit.log"), "w") as log:
        rc = subprocess.call(cmd, cwd=SCRIPTS, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if rc == 0 and os.path.exists(os.path.join(d, "dynamics", "Mz.csv")):
        open(flag, "w").write(f"{dt:.1f}\n"); return i, "ok", dt
    return i, f"fail(rc={rc})", dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pilot_C"); ap.add_argument("--time", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=2); a = ap.parse_args()
    out_dir = os.path.join(HERE, a.out); os.makedirs(out_dir, exist_ok=True)
    samples = sample(a.n, a.seed)
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_id", "x0_mm", "y0_mm"] + [f"a{k+1}" for k in range(K)] + [f"phi{k+1}" for k in range(K)]
                   + [f"label_{m}" for m in range(M)] + ["t_stop_s", "seed"])
        for i, s in enumerate(samples):
            w.writerow([i, f"{s['x0']:.3f}", f"{s['y0']:.3f}"] + [f"{c:.4f}" for c in s["coef"]] + [f"{v:.3f}" for v in s["label"]] + [a.time, a.seed])
    np.save(os.path.join(out_dir, "ygrid_rel.npy"), YGRID)
    print(f"[dataset C] {a.n} simulaciones -> {out_dir} (jobs={a.jobs}, t={a.time}s)", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, i, s, out_dir, a.time) for i, s in enumerate(samples)]
        for k, fu in enumerate(futs, 1):
            i, st, dt = fu.result()
            print(f"[{k:3d}/{a.n}] sim_{i:04d} {st:10s} {dt:6.1f}s  (transcurrido {(time.time()-t0)/60:5.1f} min)", flush=True)
    print("[dataset C] listo", flush=True)


if __name__ == "__main__":
    main()
