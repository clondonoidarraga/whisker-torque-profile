#!/usr/bin/env python3
"""
Tarea B: a partir del torque, (1) clasificar la FORMA del objeto (cilindro / prisma / pared ondulada) y
(2) estimar su POSICIÓN (x, y). Mismo protocolo que la Tarea A:
  - partición por simulación, K-fold ESTRATIFICADO por clase (cada pliegue conserva la proporción de formas)
  - validación interna para early stopping, escalador ajustado solo con train, varias semillas
Competidores:
  majority        predecir siempre la clase más frecuente / la posición media
  pose_only       Random Forest que ve SOLO la pose real (x, y, yaw), no el torque. No es un modelo: es un CONTROL.
                  Mide cuánto de la forma se puede adivinar por dónde está el objeto (atajo del diseño de datos).
  rf_contact_map  Random Forest sobre el máximo |señal| por bigote y canal (resumen estático)
  cnn_lstm, tcn   redes con dos cabezas: clasificación (3 logits) + regresión (x, y)
El yaw no se predice: cilindro y prisma son simétricos, así que el ángulo no está definido de forma única.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_A import ChannelScaler, contact_map
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

YS = np.array([50.0, 50.0], np.float32)


class Backbone(nn.Module):
    def __init__(s, f, kind, c=64, h=64, d=0.2):
        super().__init__(); s.kind = kind
        if kind == "cnn_lstm":
            s.conv = nn.Sequential(nn.Conv1d(f, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.Conv1d(c, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2))
            s.lstm = nn.LSTM(c, h, batch_first=True, bidirectional=True); out = 2 * h
        else:  # tcn
            layers, inp = [], f
            for i in range(5):
                layers += [nn.Conv1d(inp, c, 5, padding=2 * 2 ** i, dilation=2 ** i), nn.BatchNorm1d(c), nn.ReLU(), nn.Dropout(d)]; inp = c
            s.net = nn.Sequential(*layers); out = c
        s.drop = nn.Dropout(d)
        s.cls = nn.Sequential(nn.Linear(out, 64), nn.ReLU(), nn.Linear(64, 3))    # cabeza de clasificación
        s.reg = nn.Sequential(nn.Linear(out, 64), nn.ReLU(), nn.Linear(64, 2))    # cabeza de regresión

    def forward(s, x):
        if s.kind == "cnn_lstm":
            o, _ = s.lstm(s.conv(x.transpose(1, 2)).transpose(1, 2)); z = o.mean(1)
        else:
            z = s.net(x.transpose(1, 2)).mean(2)
        z = s.drop(z); return s.cls(z), s.reg(z)


def fit(kind, Xtr, ctr, ptr, Xva, cva, pva, seed, epochs=200, patience=25, bs=8):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Backbone(Xtr.shape[-1], kind); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    T = lambda a, t=torch.float32: torch.tensor(a, dtype=t)
    Xtr, ptr, Xva, pva = T(Xtr), T(ptr), T(Xva), T(pva); ctr, cva = T(ctr, torch.long), T(cva, torch.long)
    best, bad, state, n = np.inf, 0, None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]; lc, lr_ = m(Xtr[idx]); loss = ce(lc, ctr[idx]) + 10.0 * mse(lr_, ptr[idx])   # 10×: equilibra las dos tareas
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): lc, lr_ = m(Xva); va = (ce(lc, cva) + 10.0 * mse(lr_, pva)).item()
        sch.step(va); n = ep + 1
        if va < best - 1e-5: best, bad, state = va, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state); return m, n


def evaluate(c_true, c_pred, p_true, p_pred):
    return dict(acc=float(accuracy_score(c_true, c_pred)), f1=float(f1_score(c_true, c_pred, average="macro")),
                err_pos_mm=float(np.sqrt(((p_pred - p_true) ** 2).sum(1)).mean()), cm=confusion_matrix(c_true, c_pred, labels=[0, 1, 2]).tolist())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--out", default="results_B"); a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    d = np.load(a.data); Xr = d["X"]; keep = np.abs(Xr[..., :3]).max(axis=(1, 2, 3)) >= 0.6
    X = Xr[keep][:, ::a.stride]; N, T, W, C = X.shape; X = X.reshape(N, T, W * C).astype(np.float32)
    yc = d["y_class"][keep]; pose = d["y_pose"][keep].astype(np.float32); yp = pose[:, :2]
    print(f"[datos] {N} simulaciones con contacto de {len(keep)}; T={T}; clases {np.bincount(yc)}", flush=True)
    names = ["majority", "pose_only", "rf_contact_map", "cnn_lstm", "tcn"]; res = {k: [] for k in names}; os.makedirs(a.out, exist_ok=True); t0 = time.time()
    for fold, (trval, te) in enumerate(StratifiedKFold(a.folds, shuffle=True, random_state=0).split(X, yc)):
        rng = np.random.default_rng(fold); perm = rng.permutation(trval); nv = int(0.2 * len(perm)); va, tr = perm[:nv], perm[nv:]
        trv = np.concatenate([tr, va])   # los modelos sin early stopping usan train+validación (misma información total)
        res["majority"].append(evaluate(yc[te], np.full(len(te), np.bincount(yc[trv]).argmax()), yp[te], np.repeat(yp[trv].mean(0, keepdims=True), len(te), 0)))
        po = RandomForestClassifier(400, random_state=fold, n_jobs=-1).fit(pose[trv], yc[trv])
        res["pose_only"].append(evaluate(yc[te], po.predict(pose[te]), yp[te], yp[te]))            # error de pose = 0 por construcción (ve la pose)
        F = contact_map
        rc = RandomForestClassifier(400, random_state=fold, n_jobs=-1).fit(F(X[trv]), yc[trv]); rr = RandomForestRegressor(400, random_state=fold, n_jobs=-1).fit(F(X[trv]), yp[trv])
        res["rf_contact_map"].append(evaluate(yc[te], rc.predict(F(X[te])), yp[te], rr.predict(F(X[te]))))
        sc = ChannelScaler().fit(X[tr]); Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
        for kind in ["cnn_lstm", "tcn"]:
            for s in range(a.seeds):
                m, n = fit(kind, Xtr, yc[tr], yp[tr] / YS, Xva, yc[va], yp[va] / YS, seed=100 * fold + s)
                with torch.no_grad(): lc, lr_ = m(torch.tensor(Xte))
                r = evaluate(yc[te], lc.argmax(1).numpy(), yp[te], lr_.numpy() * YS); r.update(fold=fold, seed=s, epochs=n); res[kind].append(r)
        print(f"[fold {fold}] " + " | ".join(f"{k}: acc {np.mean([r['acc'] for r in res[k][-(a.seeds if k in ('cnn_lstm','tcn') else 1):]]):.2f} pos {np.mean([r['err_pos_mm'] for r in res[k][-(a.seeds if k in ('cnn_lstm','tcn') else 1):]]):.2f}" for k in names) + f"   ({(time.time()-t0)/60:.1f} min)", flush=True)
        json.dump(res, open(os.path.join(a.out, "raw.json"), "w"))
    summ = {k: {m: (float(np.mean([r[m] for r in v])), float(np.std([r[m] for r in v]))) for m in ["acc", "f1", "err_pos_mm"]} for k, v in res.items()}
    for k, v in res.items(): summ[k]["cm"] = np.sum([r["cm"] for r in v], 0).tolist()
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("\n=== RESUMEN TAREA B (media ± desv.) ===")
    for k in names: print(f"{k:15s} exactitud {summ[k]['acc'][0]:.3f} ± {summ[k]['acc'][1]:.3f} | F1 macro {summ[k]['f1'][0]:.3f} | error posición {summ[k]['err_pos_mm'][0]:.2f} ± {summ[k]['err_pos_mm'][1]:.2f} mm")
    for k in ["rf_contact_map", "cnn_lstm", "tcn"]: print(k, "matriz de confusión (filas real: cil/prisma/onda):", summ[k]["cm"])

if __name__ == "__main__":
    main()
