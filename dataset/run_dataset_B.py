#!/usr/bin/env python3
"""
Generador de dataset para la Tarea B: clasificar la FORMA del objeto y estimar su POSE a partir del torque.

Cada simulación: se elige al azar una malla (3 familias × 3 tamaños), una posición (x, y) y una rotación (yaw).
Etiquetas en el manifiesto:  shape_family (cylinder / prism / wave), mesh (archivo), x, y, yaw, y el parámetro
de tamaño de la familia (radio, lados, amplitud/longitud de onda).

Uso:
   python3 run_dataset_B.py --n 18 --seed 0 --out pilot_B --time 1.0 --jobs 2
Requiere las mallas en whiskitphysics/code/data/shapes (python3 make_shapes.py --out .../data/shapes)
"""
import argparse, csv, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "whiskitphysics", "code"))
BIN = os.path.join(CODE, "build", "whiskit")
SCRIPTS = os.path.join(CODE, "scripts")

MESHES = {  # archivo -> (familia, parámetro de tamaño)
    "cylinder_r1.obj": ("cylinder", "r=1"), "cylinder_r2.5.obj": ("cylinder", "r=2.5"), "cylinder_r5.obj": ("cylinder", "r=5"),
    "prism_4x12.obj": ("prism", "4x12"), "prism_8x8.obj": ("prism", "8x8"), "prism_6x20.obj": ("prism", "6x20"),
    "wave_a2_l15.obj": ("wave", "a2_l15"), "wave_a4_l25.obj": ("wave", "a4_l25"), "wave_a3_l10.obj": ("wave", "a3_l10"),
}
# Pose: el objeto se coloca delante/al lado derecho de la cabeza. x más lejos que en A para reducir la
# penetración inicial de objetos grandes (los bigotes en reposo llegan a x≈36).
X_RANGE, Y_RANGE, YAW_RANGE = (18.0, 46.0), (6.0, 32.0), (0.0, 180.0)


# ---- Comprobación geométrica de la pose -------------------------------------------------------
# Rechazamos poses en las que el objeto (a) atraviesa algún bigote en reposo (t=0) -> artefacto de
# penetración con torques enormes, o (b) no es alcanzado por ningún bigote durante un ciclo de batido
# -> muestra sin información. Usa la huella 2D del objeto y las coordenadas de los bigotes guardadas
# en whisker_geometry.npz (extraídas de una simulación de batido libre).
import sys; sys.path.insert(0, HERE)
from make_shapes import cylinder, prism, wave
_GEO = np.load(os.path.join(HERE, "whisker_geometry.npz"))
_FOOT = {  # polígono (x,y) en el marco del objeto, obtenido de la misma función que genera la malla
    "cylinder_r1.obj": cylinder(1.0), "cylinder_r2.5.obj": cylinder(2.5), "cylinder_r5.obj": cylinder(5.0),
    "prism_4x12.obj": prism(4, 12), "prism_8x8.obj": prism(8, 8), "prism_6x20.obj": prism(6, 20),
    "wave_a2_l15.obj": wave(2, 15), "wave_a4_l25.obj": wave(4, 25), "wave_a3_l10.obj": wave(3, 10),
}

def footprint(mesh):
    tris = _FOOT[mesh]                      # banda vertical: el primer vértice de cada triángulo par recorre el contorno
    return np.array([[t[0][0], t[0][1]] for t in tris[::2]], dtype=np.float32)

def inside(poly, pts):
    """Test punto-en-polígono (ray casting) vectorizado. poly: (M,2), pts: (K,2) -> (K,) bool"""
    x, y = pts[:, 0], pts[:, 1]; res = np.zeros(len(pts), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]; xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        res ^= cond; j = i
    return res

def pose_ok(mesh, x, y, yaw, margin=1.5):
    poly = footprint(mesh); c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    R = np.array([[c, -s_], [s_, c]], dtype=np.float32)
    world = poly @ R.T + np.array([x, y], dtype=np.float32)
    # (a) sin penetración en reposo: inflamos el polígono con un margen alrededor de su centro
    ctr = world.mean(0); infl = ctr + (world - ctr) * (1 + margin / max(np.abs(world - ctr).max(), 1e-6))
    if inside(infl, _GEO["rest"]).any():
        return False
    # (b) alcanzable: algún punto del barrido cae dentro del objeto (habrá contacto)
    return inside(world, _GEO["sweep"]).any()


def sample(n, seed):
    rng = np.random.default_rng(seed); names = sorted(MESHES)
    out, tries = [], 0
    while len(out) < n:
        mesh = names[len(out) % len(names)]                 # balance entre mallas
        x, y, yaw = float(rng.uniform(*X_RANGE)), float(rng.uniform(*Y_RANGE)), float(rng.uniform(*YAW_RANGE))
        tries += 1
        if pose_ok(mesh, x, y, yaw):
            out.append(dict(mesh=mesh, x=x, y=y, yaw=yaw))
    print(f"[sample] {n} poses válidas de {tries} intentos", flush=True)
    rng.shuffle(out)
    return out


def run_one(i, s, out_dir, t_stop):
    d = os.path.join(out_dir, f"sim_{i:04d}"); flag = os.path.join(d, "DONE")
    if os.path.exists(flag):
        return i, "skip", 0.0
    os.makedirs(d, exist_ok=True)
    cmd = [BIN, "--MODEL_TYPE", "0", "--DEBUG", "0", "--PRINT", "0", "--OBJECT", "3",
           "--file_env", f"../data/shapes/{s['mesh']}", "--ENV_SCALE", "0",
           "--ENV_LOC", f"{s['x']:.3f}", f"{s['y']:.3f}", "0", "--ENV_YAW", f"{s['yaw']:.2f}",
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
    ap.add_argument("--n", type=int, default=18); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pilot_B"); ap.add_argument("--time", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=2); a = ap.parse_args()
    out_dir = os.path.join(HERE, a.out); os.makedirs(out_dir, exist_ok=True)
    samples = sample(a.n, a.seed)
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sim_id", "shape_family", "size_param", "mesh", "x_mm", "y_mm", "yaw_deg", "t_stop_s", "seed"])
        for i, s in enumerate(samples):
            fam, size = MESHES[s["mesh"]]
            w.writerow([i, fam, size, s["mesh"], f"{s['x']:.3f}", f"{s['y']:.3f}", f"{s['yaw']:.2f}", a.time, a.seed])
    print(f"[dataset B] {a.n} simulaciones -> {out_dir} (jobs={a.jobs}, t={a.time}s)", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, i, s, out_dir, a.time) for i, s in enumerate(samples)]
        for k, fu in enumerate(futs, 1):
            i, st, dt = fu.result()
            print(f"[{k:3d}/{a.n}] sim_{i:04d} {st:10s} {dt:6.1f}s  (transcurrido {(time.time()-t0)/60:5.1f} min)", flush=True)
    print("[dataset B] listo", flush=True)


if __name__ == "__main__":
    main()
