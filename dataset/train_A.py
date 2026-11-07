#!/usr/bin/env python3
"""
Pipeline de entrenamiento y evaluación para la Tarea A: localizar el poste (x, y) a partir del torque.

Diseño metodológico (cada punto corrige un error del TFM original):
  1. La unidad de muestreo es la SIMULACIÓN, nunca la ventana. Train/val/test se separan por simulación,
     así ninguna señal del test se parece "demasiado" a una de train por solapamiento.
  2. Tres particiones: train (ajusta pesos), validación (decide early stopping), test (se evalúa UNA vez).
  3. El escalador se ajusta SOLO con train y se aplica a val y test.
  4. Validación cruzada K-fold por simulación + varias semillas => media ± desviación, no un número suelto.
  5. Un baseline trivial (predecir la media) y uno clásico (Ridge sobre estadísticas) antes de la red:
     si la red no les gana con claridad, no hay resultado que publicar.
  6. Las métricas se reportan en unidades físicas (mm), que es lo que un lector puede juzgar.

Uso:
   python3 train_A.py --data pilot_A.npz --folds 5 --seeds 3 --epochs 150 --stride 5
"""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# Utilidades de datos
# --------------------------------------------------------------------------------------
def load_data(path, stride):
    d = np.load(path)
    X, y = d["X"], d["y"]                      # X: (N, T, W, C), y: (N, 2)
    X = X[:, ::stride]                         # submuestreo temporal (1 kHz -> 1000/stride Hz)
    N, T, W, C = X.shape
    X = X.reshape(N, T, W * C)                 # cada paso de tiempo = vector de W*C features
    return X.astype(np.float32), y.astype(np.float32), dict(N=N, T=T, W=W, C=C,
                                                              whiskers=list(d["whisker_names"]),
                                                              channels=list(d["channel_names"]))


class ChannelScaler:
    """Estandariza cada feature (bigote×canal) con media y desviación calculadas SOLO en train."""
    def fit(self, X):                          # X: (N, T, F)
        flat = X.reshape(-1, X.shape[-1])
        self.mu = flat.mean(0); self.sd = flat.std(0) + 1e-8
        return self
    def transform(self, X):
        return (X - self.mu) / self.sd


def summary_features(X):
    """Estadísticas por feature a lo largo del tiempo: entrada del baseline Ridge."""
    return np.concatenate([X.mean(1), X.std(1), np.abs(X).max(1)], axis=1)


def contact_map(X):
    """Máximo |M| por bigote a lo largo del tiempo (canales 0-2): el resumen 'estático' del contacto."""
    F = X.shape[-1] // 6
    M = X[..., :3 * F].reshape(X.shape[0], X.shape[1], F, 3) if False else X
    return np.abs(X).max(1)  # (N, F) sobre todas las features; suficiente como baseline fuerte


def metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = np.abs(err).mean(0)                                  # por coordenada (x, y)
    rmse = np.sqrt((err ** 2).mean(0))
    eucl = np.sqrt((err ** 2).sum(1)).mean()                   # error de posición medio (mm)
    ss_res = (err ** 2).sum(0); ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    return dict(mae_x=float(mae[0]), mae_y=float(mae[1]), rmse_x=float(rmse[0]), rmse_y=float(rmse[1]),
                err_pos_mm=float(eucl), r2_x=float(r2[0]), r2_y=float(r2[1]))


# --------------------------------------------------------------------------------------
# Modelo CNN-LSTM (misma familia que el TFM, con los ejes correctos)
# --------------------------------------------------------------------------------------
class CNNLSTM(nn.Module):
    def __init__(self, n_feat, conv=64, hidden=64, layers=1, dropout=0.2):
        super().__init__()
        # Conv1D sobre el eje TIEMPO: aprende patrones locales (picos de contacto) en cada paso.
        self.conv = nn.Sequential(nn.Conv1d(n_feat, conv, kernel_size=5, padding=2), nn.BatchNorm1d(conv), nn.ReLU(),
                                  nn.Conv1d(conv, conv, kernel_size=5, padding=2), nn.BatchNorm1d(conv), nn.ReLU(),
                                  nn.MaxPool1d(2))
        # BiLSTM: integra la secuencia completa de contactos a lo largo del batido.
        self.lstm = nn.LSTM(conv, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):                      # x: (B, T, F)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)   # (B, T/2, conv)
        out, _ = self.lstm(h)                               # (B, T/2, 2*hidden)
        return self.head(out.mean(1))                       # promedio temporal -> (B, 2)


def train_net(Xtr, ytr, Xva, yva, seed, epochs, lr=1e-3, patience=20, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CNNLSTM(Xtr.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    loss_fn = nn.SmoothL1Loss()                # robusta a outliers; las etiquetas están en mm/escala
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva)
    best, best_state, bad, hist = np.inf, None, 0, []
    bs = 8
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = loss_fn(model(Xva_t), yva_t).item()      # <- la VALIDACIÓN decide; el test no se mira
        sched.step(va); hist.append(va)
        if va < best - 1e-5:
            best, bad, best_state = va, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


# --------------------------------------------------------------------------------------
# Protocolo experimental
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="results_A")
    ap.add_argument("--contact_only", action="store_true", help="excluir simulaciones sin contacto (max|M|<0.6)")
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 1))

    X, y, info = load_data(a.data, a.stride)
    if a.contact_only:
        raw = np.load(a.data)["X"]; keep = np.abs(raw[..., :3]).max(axis=(1, 2, 3)) >= 0.6
        X, y = X[keep], y[keep]; info["N"] = int(keep.sum()); print(f"[datos] contact_only: {keep.sum()} de {len(keep)}")
    print(f"[datos] N={info['N']} simulaciones, T={info['T']} pasos (stride {a.stride}), F={info['W']}x{info['C']}={X.shape[-1]} features")
    os.makedirs(a.out, exist_ok=True)

    # Escala de las etiquetas (mm) -> [0,1] aprox. para que la red entrene estable; se deshace al evaluar.
    y_scale = np.array([50.0, 50.0], dtype=np.float32)

    results = {"mean_baseline": [], "ridge": [], "rf_contact_map": [], "cnn_lstm": []}
    preds_all = {"ridge": [], "cnn_lstm": [], "true": []}
    kf = KFold(n_splits=a.folds, shuffle=True, random_state=0)
    t0 = time.time()
    for fold, (trval_idx, te_idx) in enumerate(kf.split(X)):
        # Dentro del bloque de entrenamiento separamos un 20 % para VALIDACIÓN (early stopping)
        rng = np.random.default_rng(fold); perm = rng.permutation(trval_idx)
        n_val = max(2, int(0.2 * len(perm))); va_idx, tr_idx = perm[:n_val], perm[n_val:]

        sc = ChannelScaler().fit(X[tr_idx])                 # <- SOLO train
        Xtr, Xva, Xte = sc.transform(X[tr_idx]), sc.transform(X[va_idx]), sc.transform(X[te_idx])
        ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]

        # Baseline 0: predecir la media de train. Cualquier modelo útil debe ganarle por mucho.
        results["mean_baseline"].append(metrics(yte, np.repeat(ytr.mean(0, keepdims=True), len(yte), 0)))

        # Baseline 1: Ridge sobre estadísticas temporales (sin aprendizaje profundo).
        fs = StandardScaler().fit(summary_features(Xtr))
        ridge = Ridge(alpha=10.0).fit(fs.transform(summary_features(Xtr)), ytr)
        p_ridge = ridge.predict(fs.transform(summary_features(Xte)))
        results["ridge"].append(metrics(yte, p_ridge))

        # Baseline 2: Random Forest sobre el mapa de contacto (máx |·| por feature). Fuerte y sin dinámica temporal.
        rf = RandomForestRegressor(400, random_state=fold, n_jobs=-1).fit(contact_map(X[tr_idx]), ytr)
        results["rf_contact_map"].append(metrics(yte, rf.predict(contact_map(X[te_idx]))))

        # Modelo propuesto: CNN-LSTM, varias semillas (la varianza entre semillas es parte del resultado).
        fold_preds = []
        for s in range(a.seeds):
            model, hist = train_net(Xtr, ytr / y_scale, Xva, yva / y_scale, seed=100 * fold + s, epochs=a.epochs)
            p = predict(model, Xte) * y_scale
            m = metrics(yte, p); m.update(fold=fold, seed=s, epochs_run=len(hist)); results["cnn_lstm"].append(m)
            fold_preds.append(p)
            print(f"  fold {fold} seed {s}: err_pos={m['err_pos_mm']:.2f} mm  MAE x={m['mae_x']:.2f} y={m['mae_y']:.2f}  R2 x={m['r2_x']:.2f} y={m['r2_y']:.2f}  ({len(hist)} ep)", flush=True)
        preds_all["cnn_lstm"].append(np.mean(fold_preds, 0)); preds_all["ridge"].append(p_ridge); preds_all["true"].append(yte)
        print(f"[fold {fold}] media: {results['mean_baseline'][-1]['err_pos_mm']:.2f} mm | ridge: {results['ridge'][-1]['err_pos_mm']:.2f} mm | rf: {results['rf_contact_map'][-1]['err_pos_mm']:.2f} mm | "
              f"cnn-lstm: {np.mean([r['err_pos_mm'] for r in results['cnn_lstm'][-a.seeds:]]):.2f} mm   ({(time.time()-t0)/60:.1f} min)", flush=True)

    # Resumen: media ± desviación entre folds (y semillas)
    summary = {}
    for k, rows in results.items():
        summary[k] = {m: (float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows])))
                      for m in ["err_pos_mm", "mae_x", "mae_y", "r2_x", "r2_y"]}
    print("\n=== RESUMEN (media ± desv. entre folds/semillas) ===")
    for k, v in summary.items():
        print(f"{k:14s} err_pos = {v['err_pos_mm'][0]:5.2f} ± {v['err_pos_mm'][1]:4.2f} mm | MAE x {v['mae_x'][0]:.2f} y {v['mae_y'][0]:.2f} | R² x {v['r2_x'][0]:.2f} y {v['r2_y'][0]:.2f}")
    json.dump({"summary": summary, "per_run": results, "info": info, "args": vars(a)}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    np.savez(os.path.join(a.out, "preds.npz"), true=np.concatenate(preds_all["true"]), ridge=np.concatenate(preds_all["ridge"]),
             cnn_lstm=np.concatenate(preds_all["cnn_lstm"]))
    print(f"[guardado] {a.out}/results.json y preds.npz")


if __name__ == "__main__":
    main()
