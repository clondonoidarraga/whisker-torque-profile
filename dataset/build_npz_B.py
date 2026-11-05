#!/usr/bin/env python3
"""
Empaqueta un dataset de la Tarea B. Igual que build_npz.py, pero con etiquetas de forma y pose:
   X        (N, T, 27, 6)
   y_class  (N,)   0=cylinder, 1=prism, 2=wave
   y_pose   (N, 3) x, y (mm), yaw (grados)
   mesh     (N,)   nombre de la malla (para análisis por tamaño)
"""
import argparse, csv, json, os
import numpy as np, pandas as pd

CHANNELS = ["Mx", "My", "Mz", "Fx", "Fy", "Fz"]; CLASSES = ["cylinder", "prism", "wave"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--drop", type=int, default=5); a = ap.parse_args()
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.dir)
    man = list(csv.DictReader(open(os.path.join(base, "manifest.csv"))))
    X, yc, yp, mesh, ids, dropped, names, T = [], [], [], [], [], [], None, None
    for r in man:
        sid = int(r["sim_id"]); d = os.path.join(base, f"sim_{sid:04d}")
        if not os.path.exists(os.path.join(d, "DONE")):
            dropped.append((sid, "no terminada")); continue
        if names is None:
            names = [s.strip() for s in open(os.path.join(d, "whisker_ID.csv")).read().split(",") if s.strip()]
        arr = np.stack([pd.read_csv(os.path.join(d, "dynamics", f"{c}.csv"), header=None).values[:, :len(names)] for c in CHANNELS], -1)[a.drop:].astype(np.float32)
        if np.isnan(arr).any() or np.abs(arr).max() > 1e6:
            dropped.append((sid, "NaN/explosión")); continue
        T = arr.shape[0] if T is None else min(T, arr.shape[0])
        X.append(arr); yc.append(CLASSES.index(r["shape_family"])); yp.append([float(r["x_mm"]), float(r["y_mm"]), float(r["yaw_deg"])])
        mesh.append(r["mesh"]); ids.append(sid)
    X = np.stack([x[:T] for x in X]).astype(np.float32)
    np.savez_compressed(os.path.join(os.path.dirname(base), a.out), X=X, y_class=np.array(yc), y_pose=np.array(yp, np.float32),
                        mesh=np.array(mesh), sim_id=np.array(ids), whisker_names=np.array(names), channel_names=np.array(CHANNELS), classes=np.array(CLASSES))
    print(f"X {X.shape}  clases {np.bincount(yc, minlength=3)}  guardado en {a.out}")
    if dropped: print("excluidas:", dropped)


if __name__ == "__main__":
    main()
