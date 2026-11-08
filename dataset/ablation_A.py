#!/usr/bin/env python3
"""
Ablation (Tarea A): qué parte de la CNN-LSTM aporta, y qué pérdida conviene. Mismo protocolo que train_A.py
(5 folds por simulación, validación interna, escalador en train, 3 semillas), a 500 Hz (stride 2).

Variantes de arquitectura:
  cnn_lstm      Conv1D×2 + MaxPool + BiLSTM(64)            (modelo propuesto)
  cnn_only      Conv1D×2 + MaxPool + promedio temporal      (sin memoria recurrente)
  lstm_only     BiLSTM(64) directo sobre la serie           (sin extracción local)
  tfm_3bilstm   Conv1D + BiLSTM(512,256,128) como en el TFM (misma familia, ~25× más parámetros)
  tcn           Red convolucional temporal dilatada (competidor moderno sin recurrencia)
Variantes de pérdida (sobre cnn_lstm): smoothl1 (propuesta), mse, mse_ssim (0,3·MSE + 0,7·(1−SSIM) como en el TFM;
  con 2 salidas el "SSIM 1D" degenera, se incluye solo para documentar por qué no aplica a esta tarea).
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_A import load_data, ChannelScaler, metrics, predict
from sklearn.model_selection import KFold


class CNNOnly(nn.Module):
    def __init__(s, f, c=64, d=0.2):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(f, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.Conv1d(c, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2))
        s.head = nn.Sequential(nn.Dropout(d), nn.Linear(c, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(s, x): return s.head(s.conv(x.transpose(1, 2)).mean(2))


class LSTMOnly(nn.Module):
    def __init__(s, f, h=64, d=0.2):
        super().__init__(); s.lstm = nn.LSTM(f, h, batch_first=True, bidirectional=True)
        s.head = nn.Sequential(nn.Dropout(d), nn.Linear(2 * h, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(s, x): o, _ = s.lstm(x); return s.head(o.mean(1))


class CNNLSTM(nn.Module):
    def __init__(s, f, c=64, h=64, d=0.2):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(f, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.Conv1d(c, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2))
        s.lstm = nn.LSTM(c, h, batch_first=True, bidirectional=True)
        s.head = nn.Sequential(nn.Dropout(d), nn.Linear(2 * h, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(s, x): o, _ = s.lstm(s.conv(x.transpose(1, 2)).transpose(1, 2)); return s.head(o.mean(1))


class TFM3BiLSTM(nn.Module):
    """Réplica de la arquitectura del TFM: Conv1D(128,k5)+BN, BiLSTM 512→256→128 (dropout 0,3), Dense 1500, salida."""
    def __init__(s, f):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(f, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU())
        s.l1 = nn.LSTM(128, 512, batch_first=True, bidirectional=True); s.l2 = nn.LSTM(1024, 256, batch_first=True, bidirectional=True); s.l3 = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        s.d = nn.Dropout(0.3); s.head = nn.Sequential(nn.Linear(256, 1500), nn.ReLU(), nn.Linear(1500, 2))
    def forward(s, x):
        h = s.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = s.l1(h); h = s.d(h); h, _ = s.l2(h); h = s.d(h); h, _ = s.l3(h); h = s.d(h[:, -1])
        return s.head(h)


class TCN(nn.Module):
    def __init__(s, f, c=64, levels=5, k=5, d=0.2):
        super().__init__(); layers = []; inp = f
        for i in range(levels):
            dil = 2 ** i; layers += [nn.Conv1d(inp, c, k, padding=dil * (k - 1) // 2, dilation=dil), nn.BatchNorm1d(c), nn.ReLU(), nn.Dropout(d)]; inp = c
        s.net = nn.Sequential(*layers); s.head = nn.Sequential(nn.Linear(c, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(s, x): return s.head(s.net(x.transpose(1, 2)).mean(2))


ARCH = {"cnn_lstm": CNNLSTM, "cnn_only": CNNOnly, "lstm_only": LSTMOnly, "tfm_3bilstm": TFM3BiLSTM, "tcn": TCN}


def ssim1d_loss(yt, yp):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mx, my = yt.mean(1, keepdim=True), yp.mean(1, keepdim=True)
    vx, vy = yt.var(1, unbiased=False, keepdim=True), yp.var(1, unbiased=False, keepdim=True)
    cov = ((yt - mx) * (yp - my)).mean(1, keepdim=True)
    ssim = ((2 * mx * my + C1) * (2 * cov + C2)) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))
    return 0.3 * nn.functional.mse_loss(yp, yt) + 0.7 * (1 - ssim.mean())


LOSS = {"smoothl1": nn.SmoothL1Loss(), "mse": nn.MSELoss(), "mse_ssim": ssim1d_loss}


def fit(model, Xtr, ytr, Xva, yva, loss_fn, seed, epochs=200, patience=20, bs=8, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4); sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    Xtr_t, ytr_t, Xva_t, yva_t = map(torch.tensor, (Xtr, ytr, Xva, yva)); best, bad, state, n_ep = np.inf, 0, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]; opt.zero_grad(); l = loss_fn(model(Xtr_t[idx]), ytr_t[idx]); l.backward(); opt.step()
        model.eval()
        with torch.no_grad(): va = nn.functional.smooth_l1_loss(model(Xva_t), yva_t).item()   # criterio de parada común a todas las variantes
        sched.step(va); n_ep = ep + 1
        if va < best - 1e-5: best, bad, state = va, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); return model, n_ep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--out", default="ablation_A")
    ap.add_argument("--only_tfm", action="store_true"); ap.add_argument("--max_folds", type=int, default=99); a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    X, y, info = load_data(a.data, a.stride); keep = np.abs(np.load(a.data)["X"][..., :3]).max(axis=(1, 2, 3)) >= 0.6; X, y = X[keep], y[keep]
    ysc = np.array([50.0, 50.0], np.float32)
    variants = [("cnn_lstm", "smoothl1"), ("cnn_only", "smoothl1"), ("lstm_only", "smoothl1"), ("tcn", "smoothl1"), ("cnn_lstm", "mse"), ("cnn_lstm", "mse_ssim")]
    if a.only_tfm:
        variants = [("tfm_3bilstm", "smoothl1")]
    res = {f"{ar}|{lo}": [] for ar, lo in variants}; os.makedirs(a.out, exist_ok=True); t0 = time.time()
    for fold, (trval, te) in enumerate(KFold(a.folds, shuffle=True, random_state=0).split(X)):
        if fold >= a.max_folds: break
        rng = np.random.default_rng(fold); perm = rng.permutation(trval); nv = int(0.2 * len(perm)); va, tr = perm[:nv], perm[nv:]
        sc = ChannelScaler().fit(X[tr]); Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
        for ar, lo in variants:
            for s in range(1 if ar == 'tfm_3bilstm' else a.seeds):
                m = ARCH[ar](X.shape[-1]); nparam = sum(p.numel() for p in m.parameters()); t1 = time.time()
                m, n_ep = fit(m, Xtr, y[tr] / ysc, Xva, y[va] / ysc, LOSS[lo], seed=100 * fold + s)
                r = metrics(y[te], predict(m, Xte) * ysc); r.update(fold=fold, seed=s, epochs=n_ep, params=nparam, train_s=time.time() - t1); res[f"{ar}|{lo}"].append(r)
                print(f"[fold {fold} seed {s}] {ar:12s} {lo:9s} err={r['err_pos_mm']:.2f} mm  ({n_ep} ep, {nparam/1e3:.0f}k par, {time.time()-t1:.0f}s)  total {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump(res, open(os.path.join(a.out, "ablation_raw.json"), "w"), indent=1)
    summ = {k: {"err_pos_mm": (float(np.mean([r["err_pos_mm"] for r in v])), float(np.std([r["err_pos_mm"] for r in v]))),
                "params": int(v[0]["params"]), "train_s": float(np.mean([r["train_s"] for r in v])), "epochs": float(np.mean([r["epochs"] for r in v]))} for k, v in res.items()}
    json.dump(summ, open(os.path.join(a.out, "ablation_summary.json"), "w"), indent=1)
    print("\n=== ABLATION (error de posición, mm) ===")
    for k, v in summ.items(): print(f"{k:24s} {v['err_pos_mm'][0]:5.2f} ± {v['err_pos_mm'][1]:4.2f}   {v['params']/1e3:7.0f}k parámetros   {v['train_s']:5.0f} s/entrenamiento   {v['epochs']:.0f} épocas")

if __name__ == "__main__":
    main()
