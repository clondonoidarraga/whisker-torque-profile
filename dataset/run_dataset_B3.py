#!/usr/bin/env python3
"""
Dataset B v3: igual que B2 (forma independiente de la pose, 9 mallas) pero la CABEZA AVANZA hacia el objeto
a velocidad constante (modo EXPLORING del simulador). Así los bigotes pasan de rozar con la punta a recorrer el objeto.
Cambios respecto a B2:
  - la cabeza avanza en +x a --vel mm/s (desplazamiento total D ~ 0.95 * vel * t, medido en una prueba: 100 mm/s -> 45.9 mm en 0.48 s)
  - rango de x ampliado: el objeto debe quedar delante de la cabeza en todo momento (xmin >= D + 10) y ser alcanzado por
    el barrido de bigotes en alguna posición de la cabeza (huella de barrido desplazada de 0 a D)
  - la pose (x, y, yaw) se da en el marco de la cabeza INICIAL
Uso:  python3 run_dataset_B3.py --n 306 --seed 3 --out full_B3 --time 0.5 --vel 60 --jobs 2   [--dry]
"""
import argparse, csv, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_dataset_B2 as B2
from run_dataset_B import inside
GEO, TRIS, world = B2.GEO, B2.TRIS, B2.world
BIN, SCRIPTS, SHAPES = B2.BIN, B2.SCRIPTS, B2.SHAPES


def disp_of(vel, t): return 0.95 * vel * t


def valid_for(mesh, x, y, yaw, D, margin=1.0, min_sweep=40):
    w = world(mesh, x, y, yaw)
    if w[:, 0].min() < D + 10: return False                                  # no chocar con la cabeza al avanzar
    ctr = w.mean(0); infl = ctr + (w - ctr) * (1 + margin / max(np.abs(w - ctr).max(), 1e-6))
    if inside(infl, GEO["rest"]).any(): return False                         # sin penetración inicial
    sw = GEO["sweep"].astype(np.float32)
    return max(inside(w, sw + np.array([d, 0], np.float32)).sum() for d in np.linspace(0, D, 5)) >= min_sweep   # alcanzado en alguna posición de la cabeza


def sample(n, seed, D):
    rng = np.random.default_rng(seed); names = sorted(TRIS); poses, tries = [], 0
    xr = (D + 12.0, 57.0 + D - 14.0)
    while len(poses) < n:
        tries += 1; x, y, yaw = rng.uniform(*xr), rng.uniform(0, 36), rng.uniform(0, 180)
        if all(valid_for(m, x, y, yaw, D) for m in names): poses.append((float(x), float(y), float(yaw)))
    meshes = [names[i % len(names)] for i in range(n)]; rng.shuffle(meshes)
    print(f"[sample] {n} poses válidas para las 9 mallas de {tries} intentos (x en {xr[0]:.0f}-{xr[1]:.0f}, D={D:.1f} mm)", flush=True)
    return [dict(mesh=m, x=p[0], y=p[1], yaw=p[2]) for m, p in zip(meshes, poses)]


def run_one(i, s, out_dir, t_stop, traj, scratch=None):
    """Corre una simulación. Con scratch, el simulador escribe en disco local (rápido) y al terminar solo se copian
    a out_dir (Drive) los archivos que usa el empaquetado: dynamics/*.csv, whisker_ID.csv y el log."""
    d = os.path.join(out_dir, f"sim_{i:04d}"); flag = os.path.join(d, "DONE")
    if os.path.exists(flag): return i, "skip", 0.0
    w = os.path.join(scratch, f"sim_{i:04d}") if scratch else d
    if scratch and os.path.exists(w): shutil.rmtree(w)
    os.makedirs(w, exist_ok=True)
    cmd = [BIN, "--MODEL_TYPE", "0", "--DEBUG", "0", "--PRINT", "0", "--OBJECT", "3", "--file_env", os.path.join(SHAPES, s["mesh"] + ".obj"),
           "--ENV_SCALE", "0", "--ENV_LOC", f"{s['x']:.3f}", f"{s['y']:.3f}", "0", "--ENV_YAW", f"{s['yaw']:.2f}",
           "--EXPLORING", "1", "--file_head_traj", traj,
           "--TIME_STOP", f"{t_stop}", "--ACTIVE", "1", "--WHISKER_NAMES", "R", "--SAVE", "1", "--SAVE_VIDEO", "0", "--dir_out", w]
    t0 = time.time()
    with open(os.path.join(w, "whiskit.log"), "w") as log: rc = subprocess.call(cmd, cwd=SCRIPTS, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if not (rc == 0 and os.path.exists(os.path.join(w, "dynamics", "Mz.csv"))): return i, f"fail(rc={rc})", dt
    if scratch:
        os.makedirs(d, exist_ok=True)
        shutil.copytree(os.path.join(w, "dynamics"), os.path.join(d, "dynamics"), dirs_exist_ok=True)
        for f in ("whisker_ID.csv", "whiskit.log"): shutil.copy(os.path.join(w, f), os.path.join(d, f))
        shutil.rmtree(w)
    open(flag, "w").write(f"{dt:.1f}\n"); return i, "ok", dt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=306); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="full_B3"); ap.add_argument("--time", type=float, default=0.5); ap.add_argument("--vel", type=float, default=60.0)
    ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--from_idx", type=int, default=0, help="solo simulaciones con índice >= este valor (para repartir el trabajo entre dos sesiones)")
    ap.add_argument("--scratch", default=None, help="carpeta LOCAL de trabajo; a --out solo se copia lo necesario al terminar cada simulación"); a = ap.parse_args()
    if a.scratch: os.makedirs(a.scratch, exist_ok=True)
    os.makedirs(SHAPES, exist_ok=True)
    from make_shapes import write_obj
    for k, (_, _, tris) in TRIS.items():                      # mismas mallas que B2 (centradas, altura 160 mm)
        P = np.array([[p for p in t] for t in tris], np.float32); raw = np.array([[t[0][0], t[0][1]] for t in tris[::2]], np.float32)
        off = (raw.max(0) + raw.min(0)) / 2; P[..., 0] -= off[0]; P[..., 1] -= off[1]; P[..., 2] *= 160.0 / 60.0
        write_obj(os.path.join(SHAPES, k + ".obj"), [tuple(map(tuple, t)) for t in P])
    out_dir = os.path.join(HERE, a.out); os.makedirs(out_dir, exist_ok=True)
    traj = os.path.abspath(os.path.join(a.scratch or out_dir, "head_traj.csv")); T = np.zeros((int(a.time / 0.001) + 300, 9)); T[:, 3] = a.vel
    if not os.path.exists(traj): np.savetxt(traj, T, delimiter=",", fmt="%g")                # 9 columnas: x y z vx vy vz wx wy wz (el simulador usa 3..8)
    D = disp_of(a.vel, a.time); man = os.path.join(out_dir, "manifest.csv")
    if os.path.exists(man):      # otra sesión ya fijó las poses: se leen del manifiesto para que ambas sesiones simulen exactamente lo mismo
        S = [dict(mesh=r["mesh"][:-4], x=float(r["x_mm"]), y=float(r["y_mm"]), yaw=float(r["yaw_deg"])) for r in csv.DictReader(open(man))]
        assert len(S) == a.n, "el manifiesto existente no tiene --n filas"; print(f"[sample] {len(S)} poses leídas del manifiesto existente", flush=True)
    else: S = sample(a.n, a.seed, D)
    with open(man if not os.path.exists(man) else os.devnull, "w", newline="") as f:   # no reescribir si otra sesión ya lo creó (es idéntico: misma semilla)
        w = csv.writer(f); w.writerow(["sim_id", "shape_family", "size_param", "mesh", "x_mm", "y_mm", "yaw_deg", "t_stop_s", "seed", "head_vel_mm_s", "head_disp_mm"])
        for i, s in enumerate(S): w.writerow([i, TRIS[s["mesh"]][0], TRIS[s["mesh"]][1], s["mesh"] + ".obj", f"{s['x']:.3f}", f"{s['y']:.3f}", f"{s['yaw']:.2f}", a.time, a.seed, a.vel, f"{D:.1f}"])
    if a.dry: return
    print(f"[dataset B3] {a.n} simulaciones -> {out_dir} (jobs={a.jobs}, t={a.time}s, vel={a.vel})", flush=True); t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for k, fu in enumerate([ex.submit(run_one, i, s, out_dir, a.time, traj, a.scratch) for i, s in enumerate(S) if i >= a.from_idx], 1):
            i, st, dt = fu.result(); print(f"[{k:3d}/{a.n}] sim_{i:04d} {st:10s} {dt:6.1f}s  (transcurrido {(time.time()-t0)/60:5.1f} min)", flush=True)
    print("[dataset B3] listo", flush=True)

if __name__ == "__main__": main()
