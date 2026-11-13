#!/usr/bin/env python3
"""
Conjunto S (barrido): RECONSTRUCCIÓN DE PERFIL. Una pared lateral con relieve aleatorio queda a la derecha de la cabeza
y la cabeza avanza en +x, de modo que los bigotes RECORREN la superficie (como una rata que sigue una pared).

Geometría (marco de la cabeza inicial, mm): la pared ocupa x en [X0, X0+L] y su superficie (la cara que mira a la cabeza)
está en  y_sup(s) = y0 - p(s),  s = x - X0,  p(s) = sum_k a_k cos(2 pi k s / L + phi_k),  K = 3.
Dos rampas de entrada y salida alejan la superficie en los extremos para que los bigotes entren y salgan sin chocar de frente.
Etiqueta: y_sup en M = 16 puntos equiespaciados de s (menos 25 mm para centrar). También se guardan coeficientes e y0.

Uso:  python3 run_dataset_S.py --n 16 --seed 0 --out pilot_S --time 0.7 --vel 120 --jobs 2 [--scratch /ruta/local]
"""
import argparse, csv, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from make_shapes import write_obj, extrude_polyline
CODE = os.path.abspath(os.path.join(HERE, "..", "whiskitphysics", "code"))
BIN = os.path.join(CODE, "build", "whiskit"); SCRIPTS = os.path.join(CODE, "scripts")

K, L, M = 3, 50.0, 16
A_MIN = np.array([1.5, 0.0, 0.0]); A_MAX = np.array([5.0, 3.0, 1.5])   # amplitud mínima y máxima por armónico (mm)
X0, Y0_RANGE = 55.0, (30.0, 34.0)          # la pared empieza más allá del alcance de los bigotes en reposo (x ~ 48)
RAMP, BACK, HEIGHT = 12.0, 14.0, 160.0
S_GRID = np.linspace(0, L, M)


def profile(coef, s):
    return sum(coef[k] * np.cos(2 * np.pi * (k + 1) * s / L + coef[K + k]) for k in range(K))


def polygon(coef, y0, n=80):
    s = np.linspace(0, L, n); front = [(X0 + si, y0 - pi) for si, pi in zip(s, profile(coef, s))]
    far = y0 + BACK
    return [(X0 - RAMP, far)] + front + [(X0 + L + RAMP, far), (X0 + L + RAMP, far + 6.0), (X0 - RAMP, far + 6.0)]


def sample(n, seed, y0r=Y0_RANGE):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        coef = np.concatenate([rng.uniform(A_MIN, A_MAX), rng.uniform(0, 2 * np.pi, K)]); y0 = float(rng.uniform(*y0r))
        out.append(dict(coef=coef, y0=y0, label=y0 - profile(coef, S_GRID) - 25.0))
    return out


def run_one(i, s, out_dir, t_stop, traj, scratch=None):
    d = os.path.join(out_dir, f"sim_{i:04d}"); flag = os.path.join(d, "DONE")
    if os.path.exists(flag): return i, "skip", 0.0
    w = os.path.join(scratch, f"sim_{i:04d}") if scratch else d
    if scratch and os.path.exists(w): shutil.rmtree(w)
    os.makedirs(w, exist_ok=True); mesh = os.path.join(w, "wall.obj")
    write_obj(mesh, extrude_polyline(polygon(s["coef"], s["y0"]), closed=True, h=HEIGHT))
    cmd = [BIN, "--MODEL_TYPE", "0", "--DEBUG", "0", "--PRINT", "0", "--OBJECT", "3", "--file_env", os.path.abspath(mesh),
           "--ENV_SCALE", "0", "--ENV_LOC", "0", "0", "0", "--ENV_YAW", "0", "--EXPLORING", "1", "--file_head_traj", traj,
           "--TIME_STOP", f"{t_stop}", "--ACTIVE", "1", "--WHISKER_NAMES", "R", "--SAVE", "1", "--SAVE_VIDEO", "0", "--dir_out", w]
    t0 = time.time()
    with open(os.path.join(w, "whiskit.log"), "w") as log: rc = subprocess.call(cmd, cwd=SCRIPTS, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if not (rc == 0 and os.path.exists(os.path.join(w, "dynamics", "Mz.csv"))): return i, f"fail(rc={rc})", dt
    if scratch:
        os.makedirs(d, exist_ok=True); shutil.copytree(os.path.join(w, "dynamics"), os.path.join(d, "dynamics"), dirs_exist_ok=True)
        for f in ("whisker_ID.csv", "whiskit.log"): shutil.copy(os.path.join(w, f), os.path.join(d, f))
        shutil.rmtree(w)
    open(flag, "w").write(f"{dt:.1f}\n"); return i, "ok", dt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pilot_S"); ap.add_argument("--time", type=float, default=0.7); ap.add_argument("--vel", type=float, default=120.0)
    ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--scratch", default=None)
    ap.add_argument("--y0", type=float, nargs=2, default=list(Y0_RANGE), help="rango de la distancia lateral media de la pared (mm)"); a = ap.parse_args()
    out_dir = os.path.join(HERE, a.out); os.makedirs(out_dir, exist_ok=True)
    if a.scratch: os.makedirs(a.scratch, exist_ok=True)
    traj = os.path.abspath(os.path.join(a.scratch or out_dir, "head_traj.csv")); T = np.zeros((int(a.time / 0.001) + 300, 9)); T[:, 3] = a.vel
    if not os.path.exists(traj): np.savetxt(traj, T, delimiter=",", fmt="%g")
    man = os.path.join(out_dir, "manifest.csv")
    if os.path.exists(man):
        R = list(csv.DictReader(open(man))); S = [dict(coef=np.array([float(r[f"a{k+1}"]) for k in range(K)] + [float(r[f"phi{k+1}"]) for k in range(K)]), y0=float(r["y0_mm"])) for r in R]
    else:
        S = sample(a.n, a.seed, tuple(a.y0))
        with open(man, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["sim_id", "y0_mm"] + [f"a{k+1}" for k in range(K)] + [f"phi{k+1}" for k in range(K)] + [f"label_{m}" for m in range(M)] + ["t_stop_s", "head_vel_mm_s", "seed"])
            for i, s in enumerate(S): w.writerow([i, f"{s['y0']:.6f}"] + [f"{c:.6f}" for c in s["coef"]] + [f"{v:.4f}" for v in s["label"]] + [a.time, a.vel, a.seed])
    print(f"[conjunto S] {len(S)} simulaciones -> {out_dir} (jobs={a.jobs}, t={a.time}s, vel={a.vel})", flush=True); t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for k, fu in enumerate([ex.submit(run_one, i, s, out_dir, a.time, traj, a.scratch) for i, s in enumerate(S)], 1):
            i, st, dt = fu.result(); print(f"[{k:3d}/{len(S)}] sim_{i:04d} {st:10s} {dt:6.1f}s  (transcurrido {(time.time()-t0)/60:5.1f} min)", flush=True)
    print("[conjunto S] listo", flush=True)

if __name__ == "__main__": main()
