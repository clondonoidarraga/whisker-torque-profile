#!/usr/bin/env python3
"""
Empaqueta las simulaciones de WHISKiT en un único archivo .npz listo para entrenar.

Forma de salida (Tarea A, localización):
   X : (N, T, W, C)  float32   N simulaciones, T pasos de 1 ms, W=27 bigotes, C=6 canales [Mx,My,Mz,Fx,Fy,Fz]
   y : (N, 2)        float32   posición del poste (x, y) en mm, marco de la cabeza
   meta: sim_id, whisker_names, channel_names, t_stop, dropped_initial_steps

Decisiones de diseño (y por qué):
 - El eje de secuencia es el TIEMPO, no el índice de bigote. Esto corrige la inversión de ejes del TFM.
 - Se descartan los primeros `drop` pasos: el simulador produce un transitorio numérico (~1e16) al inicializar.
 - NO se normaliza aquí. La normalización se ajusta SOLO con el conjunto de entrenamiento, en el script de
   entrenamiento, para evitar fuga de información del test.
 - Si una simulación tiene NaN (inestabilidad numérica) se registra y se excluye; nunca se rellena con ceros
   en silencio.

Uso:
   python3 build_npz.py --dir pilot_A --out pilot_A.npz
"""
import argparse, csv, os, json
import numpy as np
import pandas as pd

CHANNELS = ["Mx", "My", "Mz", "Fx", "Fy", "Fz"]


def read_matrix(path, n_whiskers):
    # Los CSV de WHISKiT terminan cada fila en coma -> columna vacía final; nos quedamos con las W primeras.
    return pd.read_csv(path, header=None).values[:, :n_whiskers].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop", type=int, default=5, help="pasos iniciales a descartar (transitorio)")
    a = ap.parse_args()

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.dir)
    man = list(csv.DictReader(open(os.path.join(base, "manifest.csv"))))

    X, y, ids, dropped = [], [], [], []
    whisker_names = None
    T_ref = None
    for row in man:
        sid = int(row["sim_id"])
        d = os.path.join(base, f"sim_{sid:04d}")
        if not os.path.exists(os.path.join(d, "DONE")):
            dropped.append((sid, "no terminada"))
            continue
        if whisker_names is None:
            whisker_names = [s.strip() for s in open(os.path.join(d, "whisker_ID.csv")).read().split(",") if s.strip()]
        W = len(whisker_names)
        mats = [read_matrix(os.path.join(d, "dynamics", f"{c}.csv"), W) for c in CHANNELS]
        arr = np.stack(mats, axis=-1)[a.drop:]            # (T, W, C)
        if np.isnan(arr).any() or np.abs(arr).max() > 1e6:
            dropped.append((sid, "NaN o explosión numérica"))
            continue
        if T_ref is None:
            T_ref = arr.shape[0]
        if arr.shape[0] != T_ref:                           # longitudes distintas -> recortar al mínimo común
            T_ref = min(T_ref, arr.shape[0])
        X.append(arr)
        y.append([float(row["peg_x_mm"]), float(row["peg_y_mm"])])
        ids.append(sid)

    X = np.stack([x[:T_ref] for x in X]).astype(np.float32)
    y = np.array(y, dtype=np.float32)
    np.savez_compressed(os.path.join(os.path.dirname(base), a.out), X=X, y=y, sim_id=np.array(ids),
                        whisker_names=np.array(whisker_names), channel_names=np.array(CHANNELS),
                        dropped_initial_steps=a.drop)
    print(f"X {X.shape}  y {y.shape}  guardado en {a.out}")
    print(f"rango |M| canales 0-2: {np.abs(X[..., :3]).max():.3g}   rango |F| canales 3-5: {np.abs(X[..., 3:]).max():.3g}")
    if dropped:
        print("excluidas:", dropped)
    json.dump({"n": len(ids), "T": int(T_ref), "W": int(X.shape[2]), "C": int(X.shape[3]),
               "dropped": dropped}, open(os.path.join(base, "build_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
