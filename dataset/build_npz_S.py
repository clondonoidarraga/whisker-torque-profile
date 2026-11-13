#!/usr/bin/env python3
"""
Empaqueta el conjunto S (barrido de una pared con relieve):
   X      (N, T, 27, 6)  momentos y fuerzas en la base de los bigotes
   y      (N, 16)        superficie de la pared en 16 puntos (mm, menos 25)
   relief (N, 16)        relieve: la superficie menos su nivel medio y0 (lo que hay que reconstruir)
   y0     (N,)           distancia lateral media de la pared  |  coef (N, 6) a1..a3, phi1..phi3
"""
import argparse, csv, os
import numpy as np, pandas as pd
CHANNELS = ["Mx", "My", "Mz", "Fx", "Fy", "Fz"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--drop", type=int, default=5)
    ap.add_argument("--stride", type=int, default=1, help="submuestreo temporal (2 = 500 Hz)"); a = ap.parse_args()
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.dir); man = list(csv.DictReader(open(os.path.join(base, "manifest.csv"))))
    M = sum(1 for k in man[0] if k.startswith("label_")); X, Y, Y0, CO, ids, dropped, names, T = [], [], [], [], [], [], None, None
    for r in man:
        sid = int(r["sim_id"]); d = os.path.join(base, f"sim_{sid:04d}")
        if not os.path.exists(os.path.join(d, "DONE")): dropped.append((sid, "no terminada")); continue
        if names is None: names = [s.strip() for s in open(os.path.join(d, "whisker_ID.csv")).read().split(",") if s.strip()]
        arr = np.stack([pd.read_csv(os.path.join(d, "dynamics", f"{c}.csv"), header=None).values[:, :len(names)] for c in CHANNELS], -1)[a.drop:].astype(np.float32)
        if np.isnan(arr).any() or np.abs(arr).max() > 1e6: dropped.append((sid, "NaN/explosión")); continue
        T = arr.shape[0] if T is None else min(T, arr.shape[0])
        X.append(arr); Y.append([float(r[f"label_{m}"]) for m in range(M)]); Y0.append(float(r["y0_mm"]))
        CO.append([float(r[k]) for k in ["a1", "a2", "a3", "phi1", "phi2", "phi3"]]); ids.append(sid)
    X = np.stack([x[:T:a.stride] for x in X]).astype(np.float32); Y = np.array(Y, np.float32); Y0 = np.array(Y0, np.float32)
    np.savez_compressed(os.path.join(os.path.dirname(base), a.out), X=X, y=Y, relief=Y + 25.0 - Y0[:, None], y0=Y0, coef=np.array(CO, np.float32),
                        sim_id=np.array(ids), stride=np.array(a.stride), whisker_names=np.array(names), channel_names=np.array(CHANNELS))
    print(f"X {X.shape}  y {Y.shape}  guardado en {a.out}")
    if dropped: print("excluidas:", dropped)

if __name__ == "__main__": main()
