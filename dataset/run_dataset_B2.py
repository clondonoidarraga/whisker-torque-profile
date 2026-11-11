#!/usr/bin/env python3
"""
Dataset B v2: forma + pose SIN atajo. Cambios respecto a v1:
  1. Las tres familias tienen tamaño comparable (envolvente de ~4-7 mm de radio), con 3 tamaños cada una.
  2. La pose (x, y, yaw) se muestrea PRIMERO y solo se acepta si es válida para LAS 9 mallas a la vez
     (ninguna atraviesa bigotes en reposo y todas son alcanzadas por el batido). Después se asigna la malla
     al azar, de forma balanceada. Así la forma es independiente de la pose por construcción.
  3. Antes de simular se puede comprobar el control "solo pose" sobre el manifiesto (debe dar ~azar).
Uso:  python3 run_dataset_B2.py --n 306 --seed 2 --out full_B2 --time 0.5 --jobs 2   [--dry para solo el manifiesto]
"""
import argparse, csv, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from make_shapes import cylinder, prism, wave, write_obj
from run_dataset_B import inside

CODE = os.path.abspath(os.path.join(HERE, "..", "whiskitphysics", "code"))
BIN = os.path.join(CODE, "build", "whiskit"); SCRIPTS = os.path.join(CODE, "scripts"); SHAPES = os.path.join(CODE, "data", "shapes2")
GEO = np.load(os.path.join(HERE, "whisker_geometry.npz"))

TRIS = {  # nombre -> (familia, parámetro, triángulos)
    "cyl_r4": ("cylinder", "r=4", cylinder(4.0)), "cyl_r5": ("cylinder", "r=5", cylinder(5.0)), "cyl_r6": ("cylinder", "r=6", cylinder(6.0)),
    "prism_8x8": ("prism", "8x8", prism(8, 8)), "prism_6x12": ("prism", "6x12", prism(6, 12)), "prism_10x10": ("prism", "10x10", prism(10, 10)),
    "wave_a1.5_l6": ("wave", "a1.5_l6", wave(1.5, 6, length=12, n=48, thickness=4)),
    "wave_a2_l8": ("wave", "a2_l8", wave(2, 8, length=12, n=48, thickness=4)),
    "wave_a2.5_l12": ("wave", "a2.5_l12", wave(2.5, 12, length=12, n=48, thickness=4)),
}
FOOT = {}
for k, (_, _, tris) in TRIS.items():
    p = np.array([[t[0][0], t[0][1]] for t in tris[::2]], np.float32)
    FOOT[k] = p - (p.max(0) + p.min(0)) / 2          # centramos la huella: la "posición" es el centro de la envolvente
X_RANGE, Y_RANGE, YAW_RANGE = (14.0, 56.0), (0.0, 36.0), (0.0, 180.0)


def world(mesh, x, y, yaw):
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw)); R = np.array([[c, -s], [s, c]], np.float32)
    return FOOT[mesh] @ R.T + np.array([x, y], np.float32)


def valid_for(mesh, x, y, yaw, margin=1.0, min_sweep=40):
    w = world(mesh, x, y, yaw); ctr = w.mean(0); infl = ctr + (w - ctr) * (1 + margin / max(np.abs(w - ctr).max(), 1e-6))
    return (not inside(infl, GEO["rest"]).any()) and inside(w, GEO["sweep"]).sum() >= min_sweep


def sample(n, seed):
    rng = np.random.default_rng(seed); names = sorted(TRIS); poses, tries = [], 0
    while len(poses) < n:
        tries += 1; x, y, yaw = rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE), rng.uniform(*YAW_RANGE)
        if all(valid_for(m, x, y, yaw) for m in names):
            poses.append((float(x), float(y), float(yaw)))
    meshes = [names[i % len(names)] for i in range(n)]; rng.shuffle(meshes)      # asignación independiente de la pose
    print(f"[sample] {n} poses válidas para las 9 mallas de {tries} intentos", flush=True)
    return [dict(mesh=m, x=p[0], y=p[1], yaw=p[2]) for m, p in zip(meshes, poses)]


def run_one(i, s, out_dir, t_stop):
    d = os.path.join(out_dir, f"sim_{i:04d}"); flag = os.path.join(d, "DONE")
    if os.path.exists(flag): return i, "skip", 0.0
    os.makedirs(d, exist_ok=True)
    cmd = [BIN, "--MODEL_TYPE", "0", "--DEBUG", "0", "--PRINT", "0", "--OBJECT", "3", "--file_env", os.path.join(SHAPES, s["mesh"] + ".obj"),
           "--ENV_SCALE", "0", "--ENV_LOC", f"{s['x']:.3f}", f"{s['y']:.3f}", "0", "--ENV_YAW", f"{s['yaw']:.2f}",
           "--TIME_STOP", f"{t_stop}", "--ACTIVE", "1", "--WHISKER_NAMES", "R", "--SAVE", "1", "--SAVE_VIDEO", "0", "--dir_out", d]
    t0 = time.time()
    with open(os.path.join(d, "whiskit.log"), "w") as log: rc = subprocess.call(cmd, cwd=SCRIPTS, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if rc == 0 and os.path.exists(os.path.join(d, "dynamics", "Mz.csv")):
        open(flag, "w").write(f"{dt:.1f}\n"); return i, "ok", dt
    return i, f"fail(rc={rc})", dt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=306); ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default="full_B2"); ap.add_argument("--time", type=float, default=0.5); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    os.makedirs(SHAPES, exist_ok=True)
    for k, (_, _, tris) in TRIS.items():                 # mallas centradas igual que la huella
        P = np.array([[p for p in t] for t in tris], np.float32); c = (FOOT[k] * 0).sum()  # noqa
        raw = np.array([[t[0][0], t[0][1]] for t in tris[::2]], np.float32); off = (raw.max(0) + raw.min(0)) / 2
        P[..., 0] -= off[0]; P[..., 1] -= off[1]; P[..., 2] *= 160.0 / 60.0   # altura 160 mm (como el poste de A): cubre todas las filas de bigotes
        write_obj(os.path.join(SHAPES, k + ".obj"), [tuple(map(tuple, t)) for t in P])
    out_dir = os.path.join(HERE, a.out); os.makedirs(out_dir, exist_ok=True); S = sample(a.n, a.seed)
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sim_id", "shape_family", "size_param", "mesh", "x_mm", "y_mm", "yaw_deg", "t_stop_s", "seed"])
        for i, s in enumerate(S): w.writerow([i, TRIS[s["mesh"]][0], TRIS[s["mesh"]][1], s["mesh"] + ".obj", f"{s['x']:.3f}", f"{s['y']:.3f}", f"{s['yaw']:.2f}", a.time, a.seed])
    if a.dry: return
    print(f"[dataset B2] {a.n} simulaciones -> {out_dir} (jobs={a.jobs}, t={a.time}s)", flush=True); t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for k, fu in enumerate([ex.submit(run_one, i, s, out_dir, a.time) for i, s in enumerate(S)], 1):
            i, st, dt = fu.result(); print(f"[{k:3d}/{a.n}] sim_{i:04d} {st:10s} {dt:6.1f}s  (transcurrido {(time.time()-t0)/60:5.1f} min)", flush=True)
    print("[dataset B2] listo", flush=True)


if __name__ == "__main__":
    main()
