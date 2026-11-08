#!/usr/bin/env python3
"""
Robustez (Tarea A): entrena una vez por fold (CNN-LSTM y RF) con el protocolo limpio y evalúa el MISMO modelo
sobre el test perturbado. Perturbaciones aplicadas a la señal cruda (antes de escalar):
  - ruido gaussiano con σ = k · desviación de cada feature en train (k = 0.05, 0.10, 0.20)
  - amplitud × 0.8 y × 1.2
  - desplazamiento temporal de ±5 pasos (con relleno por borde)
Reporta error de posición medio (mm) por condición, media ± desv. entre folds.
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_A import load_data, ChannelScaler, train_net, predict, metrics, contact_map
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

def perturb(X, kind, sd):
    Xp = X.copy()
    if kind.startswith("noise"):
        k = float(kind.split("_")[1]); Xp += np.random.default_rng(0).normal(0, 1, Xp.shape).astype(np.float32) * (k * sd)
    elif kind.startswith("amp"):
        Xp *= float(kind.split("_")[1])
    elif kind.startswith("shift"):
        s = int(kind.split("_")[1]); Xp = np.roll(Xp, s, axis=1)
        if s > 0: Xp[:, :s] = Xp[:, s:s+1]
        else: Xp[:, s:] = Xp[:, s-1:s]
    return Xp

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--out", default="robust_A"); a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    X, y, info = load_data(a.data, a.stride)
    keep = np.abs(np.load(a.data)["X"][..., :3]).max(axis=(1, 2, 3)) >= 0.6; X, y = X[keep], y[keep]
    conds = ["clean", "noise_0.05", "noise_0.10", "noise_0.20", "amp_0.8", "amp_1.2", "shift_5", "shift_-5"]
    res = {m: {c: [] for c in conds} for m in ["cnn_lstm", "rf_contact_map"]}
    y_scale = np.array([50.0, 50.0], np.float32)
    for fold, (trval, te) in enumerate(KFold(a.folds, shuffle=True, random_state=0).split(X)):
        rng = np.random.default_rng(fold); perm = rng.permutation(trval); nv = int(0.2 * len(perm)); va, tr = perm[:nv], perm[nv:]
        sc = ChannelScaler().fit(X[tr]); sd = X[tr].reshape(-1, X.shape[-1]).std(0)
        net, _ = train_net(sc.transform(X[tr]), y[tr] / y_scale, sc.transform(X[va]), y[va] / y_scale, seed=fold, epochs=a.epochs)
        rf = RandomForestRegressor(400, random_state=fold, n_jobs=-1).fit(contact_map(X[tr]), y[tr])
        for c in conds:
            Xte = perturb(X[te], c, sd)
            res["cnn_lstm"][c].append(metrics(y[te], predict(net, sc.transform(Xte)) * y_scale)["err_pos_mm"])
            res["rf_contact_map"][c].append(metrics(y[te], rf.predict(contact_map(Xte)))["err_pos_mm"])
        print(f"[fold {fold}] " + " | ".join(f"{c}: cnn {np.mean(res['cnn_lstm'][c][-1]):.2f} rf {np.mean(res['rf_contact_map'][c][-1]):.2f}" for c in conds), flush=True)
    os.makedirs(a.out, exist_ok=True)
    summary = {m: {c: (float(np.mean(v)), float(np.std(v))) for c, v in d.items()} for m, d in res.items()}
    json.dump(summary, open(os.path.join(a.out, "robustness.json"), "w"), indent=1)
    print("\n=== ROBUSTEZ: error de posición (mm), media ± desv. entre folds ===")
    print(f"{'condición':12s} {'CNN-LSTM':>16s} {'RF mapa contacto':>18s}")
    for c in conds:
        m1, m2 = summary["cnn_lstm"][c], summary["rf_contact_map"][c]
        print(f"{c:12s} {m1[0]:7.2f} ± {m1[1]:4.2f}   {m2[0]:8.2f} ± {m2[1]:4.2f}")

if __name__ == "__main__":
    main()
