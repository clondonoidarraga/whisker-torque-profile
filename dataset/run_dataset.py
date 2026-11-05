#!/usr/bin/env python3
"""
Generador de dataset para la Tarea A (localización de un poste a partir del torque).

Qué hace, paso a paso:
 1. Muestrea N posiciones (x, y) del poste dentro del alcance de la matriz derecha de bigotes.
 2. Lanza WHISKiT Physics para cada posición (poste fijo, batido activo, 27 bigotes derechos).
 3. Guarda cada simulación en su carpeta y escribe un manifiesto CSV con la etiqueta exacta.
 4. Corre varias simulaciones en paralelo (una por núcleo) y se puede reanudar si se interrumpe.

Uso:
   python3 run_dataset.py --n 40 --seed 0 --out pilot_A --time 1.0 --jobs 2
"""
import argparse, csv, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "whiskitphysics", "code"))
BIN = os.path.join(CODE, "build", "whiskit")
SCRIPTS = os.path.join(CODE, "scripts")          # whiskit resuelve rutas relativas desde aquí

# Rango de posiciones del poste (mm, marco de la cabeza: +x adelante, +y derecha, z altura)
X_RANGE = (12.0, 42.0)
Y_RANGE = (9.0, 31.0)
Z_FIXED = 0.0


def sample_positions(n, seed):
    """Muestreo uniforme en el rectángulo de alcance. Semilla fija => reproducible."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        x, y = rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)
        # Zonas sin contacto observadas en el piloto (el poste queda fuera del alcance de la matriz derecha):
        if x > 33 and y > 27:      # esquina lejana
            continue
        if x < 14 and y < 11:      # rincón junto a la cabeza
            continue
        out.append((float(x), float(y), Z_FIXED))
    return out


def run_one(sim_id, pos, out_dir, t_stop):
    """Ejecuta una simulación. Devuelve (sim_id, estado, segundos)."""
    d = os.path.join(out_dir, f"sim_{sim_id:04d}")
    done_flag = os.path.join(d, "DONE")
    if os.path.exists(done_flag):
        return sim_id, "skip", 0.0
    os.makedirs(d, exist_ok=True)
    x, y, z = pos
    cmd = [BIN, "--MODEL_TYPE", "0", "--DEBUG", "0", "--PRINT", "0",
           "--OBJECT", "1", "--SPEED", "0", "--PEG_LOC", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}",
           "--TIME_STOP", f"{t_stop}", "--ACTIVE", "1", "--WHISKER_NAMES", "R",
           "--SAVE", "1", "--SAVE_VIDEO", "0", "--dir_out", d]
    t0 = time.time()
    with open(os.path.join(d, "whiskit.log"), "w") as log:
        rc = subprocess.call(cmd, cwd=SCRIPTS, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if rc == 0 and os.path.exists(os.path.join(d, "dynamics", "Mz.csv")):
        open(done_flag, "w").write(f"{dt:.1f}\n")
        return sim_id, "ok", dt
    return sim_id, f"fail(rc={rc})", dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pilot_A")
    ap.add_argument("--time", type=float, default=1.0, help="segundos simulados por corrida")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()

    out_dir = os.path.join(HERE, a.out)
    os.makedirs(out_dir, exist_ok=True)
    positions = sample_positions(a.n, a.seed)

    # Manifiesto: una fila por simulación con su etiqueta. Se escribe ANTES de correr,
    # así la etiqueta nunca depende de que la simulación termine.
    man_path = os.path.join(out_dir, "manifest.csv")
    with open(man_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_id", "peg_x_mm", "peg_y_mm", "peg_z_mm", "t_stop_s", "object", "whiskers", "seed"])
        for i, (x, y, z) in enumerate(positions):
            w.writerow([i, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", a.time, "peg_static", "R", a.seed])

    print(f"[dataset] {a.n} simulaciones -> {out_dir}  (jobs={a.jobs}, t={a.time}s)", flush=True)
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, i, p, out_dir, a.time) for i, p in enumerate(positions)]
        for k, fut in enumerate(futs, 1):
            sim_id, st, dt = fut.result()
            el = (time.time() - t_start) / 60
            print(f"[{k:3d}/{a.n}] sim_{sim_id:04d} {st:10s} {dt:6.1f}s  (transcurrido {el:5.1f} min)", flush=True)
    print("[dataset] listo", flush=True)


if __name__ == "__main__":
    main()
