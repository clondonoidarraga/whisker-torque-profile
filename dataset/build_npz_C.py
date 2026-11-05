#!/usr/bin/env python3
"""
Empaqueta un dataset de la Tarea C (contorno):
   X        (N, T, 27, 6)
   y        (N, 16)  perfil del frente en el marco de la cabeza: x_front(y_k) - 50 (mm), y_k = y0 + ygrid_rel
   ygrid    (N, 16)  coordenadas y absolutas de cada punto del perfil (marco de la cabeza)
   coef     (N, 6)   a1..a3, phi1..phi3   |  pose (N, 2) x0, y0
"""
import argparse, csv, os
import numpy as np, pandas as pd

CHANNELS = ["Mx", "My", "Mz", "Fx", "Fy", "Fz"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--drop", type=int, default=5); a = ap.parse_args()
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.dir)
    man = list(csv.DictReader(open(os.path.join(base, "manifest.csv")))); ygrid_rel = np.load(os.path.join(base, "ygrid_rel.npy"))
    M = len(ygrid_rel)
    X, Y, YG, CO, PO, ids, dropped, names, T = [], [], [], [], [], [], [], None, None
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
        X.append(arr); Y.append([float(r[f"label_{m}"]) for m in range(M)]); YG.append(float(r["y0_mm"]) + ygrid_rel)
        CO.append([float(r[k]) for k in ["a1", "a2", "a3", "phi1", "phi2", "phi3"]]); PO.append([float(r["x0_mm"]), float(r["y0_mm"])]); ids.append(sid)
    X = np.stack([x[:T] for x in X]).astype(np.float32)
    np.savez_compressed(os.path.join(os.path.dirname(base), a.out), X=X, y=np.array(Y, np.float32), ygrid=np.array(YG, np.float32),
                        coef=np.array(CO, np.float32), pose=np.array(PO, np.float32), sim_id=np.array(ids),
                        whisker_names=np.array(names), channel_names=np.array(CHANNELS))
    print(f"X {X.shape}  y {np.array(Y).shape}  guardado en {a.out}")
    if dropped: print("excluidas:", dropped)


if __name__ == "__main__":
    main()
