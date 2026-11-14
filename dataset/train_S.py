#!/usr/bin/env python3
"""
Conjunto S (barrido): reconstrucción del RELIEVE de una pared (16 puntos) a partir de momentos y fuerzas en la base de los bigotes.
Protocolo: partición por simulación, K pliegues, 20 % de validación interna para parada temprana, escalado ajustado solo con
entrenamiento, errores en mm. Entrada de las redes: sign(x)·log(1+|x|) por canal (las señales tienen cola pesada) y estandarización.
Competidores: media de entrenamiento; control que solo ve y0; ExtraTrees sobre tramos de tiempo; CNN-LSTM y TCN (salida de 16 puntos).
Pérdidas de las redes: MSE, o 0.3·MSE + 0.7·(1 − SSIM 1D) sobre el perfil.
"""
import argparse, json, os, time, warnings
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
warnings.filterwarnings("ignore")
YS = 3.0   # escala de la etiqueta (mm)


def load(paths, stride):
    X, R, Y0 = [], [], []
    for p in paths:
        d = np.load(p); s = int(d["stride"]) if "stride" in d.files else 1; x = d["X"][:, ::max(1, 2 // s)]   # todo a 500 Hz
        X.append(x); R.append(d["relief"]); Y0.append(d["y0"])
    T = min(x.shape[1] for x in X); X = np.concatenate([x[:, :T] for x in X]); return X[:, ::stride], np.concatenate(R).astype(np.float32), np.concatenate(Y0).astype(np.float32)


def bins(A, nb):
    T = A.shape[1]; e = np.linspace(0, T, nb + 1).astype(int); return np.stack([A[:, e[k]:e[k + 1]].mean(1) for k in range(nb)], -1).reshape(len(A), -1)


def tree_feats(X):
    M = np.sqrt(X[..., 1] ** 2 + X[..., 2] ** 2); F = np.sqrt(X[..., 4] ** 2 + X[..., 5] ** 2)
    return np.log1p(np.concatenate([bins(M, 20), bins(np.abs(X[..., 0]), 20), bins(F, 20)], 1))


class Net(nn.Module):
    def __init__(s, f, kind, c=64, h=64, d=0.2, seg=8, out=16):
        super().__init__(); s.kind = kind
        if kind == "cnn_lstm":
            s.conv = nn.Sequential(nn.Conv1d(f, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.Conv1d(c, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2))
            s.lstm = nn.LSTM(c, h, batch_first=True, bidirectional=True); feat = 2 * h
        else:
            L, inp = [], f
            for i in range(5): L += [nn.Conv1d(inp, c, 5, padding=2 * 2 ** i, dilation=2 ** i), nn.BatchNorm1d(c), nn.ReLU(), nn.Dropout(d)]; inp = c
            s.net = nn.Sequential(*L); feat = c
        s.pool = nn.AdaptiveAvgPool1d(seg)      # conserva el ORDEN temporal: 8 tramos, no un único promedio
        s.head = nn.Sequential(nn.Dropout(d), nn.Linear(feat * seg, 128), nn.ReLU(), nn.Linear(128, out))

    def forward(s, x):
        if s.kind == "cnn_lstm":
            o, _ = s.lstm(s.conv(x.transpose(1, 2)).transpose(1, 2)); z = o.transpose(1, 2)
        else: z = s.net(x.transpose(1, 2))
        return s.head(s.pool(z).flatten(1))


def ssim1d(a, b, c1=1e-4, c2=9e-4):
    ma, mb = a.mean(1), b.mean(1); va, vb = a.var(1), b.var(1); cov = ((a - ma[:, None]) * (b - mb[:, None])).mean(1)
    return ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))


def fit(kind, loss, Xtr, ytr, Xva, yva, seed, epochs=150, patience=20, bs=32, aug=None):
    """aug = (Xraw_tr, sd_raw, prep): entrena con ruido aleatorio (nivel uniforme entre 0 y 20 % de la desviación de cada canal) añadido a la señal cruda en cada lote."""
    torch.manual_seed(seed); np.random.seed(seed); m = Net(Xtr.shape[-1], kind); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8); mse = nn.MSELoss()
    L = (lambda p, y: mse(p, y)) if loss == "mse" else (lambda p, y: 0.3 * mse(p, y) + 0.7 * (1 - ssim1d(p, y)).mean())
    Xtr, ytr, Xva, yva = map(torch.tensor, (Xtr, ytr, Xva, yva)); best, bad, state, n = np.inf, 0, None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if aug is None: xb = Xtr[idx]
            else:
                raw = aug[0][idx]; lv = torch.rand(len(idx), 1, 1, 1) * 0.2; xb = aug[2](raw + torch.randn_like(raw) * lv * aug[1])
            l = L(m(xb), ytr[idx]); opt.zero_grad(); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): va = mse(m(Xva), yva).item()      # criterio común de parada: MSE en validación
        sch.step(va); n = ep + 1
        if va < best - 1e-5: best, bad, state = va, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state); return m, n


def score(P, R):
    rm = float(np.sqrt(((P - R) ** 2).mean())); cs = [np.corrcoef(P[i], R[i])[0, 1] if P[i].std() > 1e-9 else 0.0 for i in range(len(R))]
    return dict(rmse=rm, corr=float(np.median(cs)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--out", default="results_S")
    ap.add_argument("--nets", nargs="*", default=["cnn_lstm|mse", "tcn|mse", "cnn_lstm|mse_ssim"]); ap.add_argument("--only_trees", action="store_true")
    ap.add_argument("--clip", type=float, default=0.0, help="recorta la entrada estandarizada de las redes a ±clip (0 = sin recorte)")
    ap.add_argument("--aug", action="store_true", help="entrenar con ruido añadido (redes: en cada lote; árboles: copias ruidosas del entrenamiento)")
    ap.add_argument("--noise", type=float, nargs="*", default=[], help="niveles de ruido en la prueba, como fracción de la desviación de cada canal"); a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count())); X, R, y0 = load(a.data, a.stride); N, T, W, C = X.shape
    Ft = tree_feats(X); slog = lambda A: (np.sign(A) * np.log1p(np.abs(A))).reshape(len(A), T, W * C).astype(np.float32); Xn = slog(X)
    NZ = {}; addn = lambda k, lv, v: NZ.setdefault(k, {}).setdefault(str(lv), []).append(v)
    print(f"[datos] N={N} T={T} | relieve RMS {np.sqrt((R**2).mean()):.2f} mm", flush=True)
    res = {}; add = lambda k, r: res.setdefault(k, []).append(r); os.makedirs(a.out, exist_ok=True); t0 = time.time(); P_all = {}
    for fold, (trv, te) in enumerate(KFold(a.folds, shuffle=True, random_state=0).split(X)):
        rng = np.random.default_rng(fold); perm = rng.permutation(trv); nv = int(0.2 * len(perm)); va, tr = perm[:nv], perm[nv:]
        add("media", score(np.repeat(R[trv].mean(0, keepdims=True), len(te), 0) + 1e-9 * rng.normal(size=(len(te), 16)), R[te]))
        add("solo_y0", score(RandomForestRegressor(200, min_samples_leaf=3, random_state=fold, n_jobs=-1).fit(y0[trv, None], R[trv]).predict(y0[te, None]), R[te]))
        sd_raw = X[tr].reshape(-1, W, C).std(0)
        if a.aug:
            ra = np.random.default_rng(50 + fold); Fa = np.concatenate([Ft[trv]] + [tree_feats(X[trv] + ra.normal(0, 1, X[trv].shape).astype(np.float32) * (lv * sd_raw)) for lv in (0.05, 0.1, 0.2)]); et = ExtraTreesRegressor(200, min_samples_leaf=2, random_state=fold, n_jobs=-1).fit(Fa, np.tile(R[trv], (4, 1)))
        else: et = ExtraTreesRegressor(400, min_samples_leaf=2, random_state=fold, n_jobs=-1).fit(Ft[trv], R[trv])
        p = et.predict(Ft[te]); add("extratrees", score(p, R[te])); noisy = {lv: X[te] + np.random.default_rng(7 + fold).normal(0, 1, X[te].shape).astype(np.float32) * (lv * sd_raw) for lv in a.noise}
        for lv, Xq in noisy.items(): addn("extratrees", lv, score(et.predict(tree_feats(Xq)), R[te])["rmse"]); P_all.setdefault("extratrees", np.zeros_like(R))[te] = p
        sh = np.random.default_rng(100 + fold).permutation(len(trv)); add("barajado", score(ExtraTreesRegressor(200, min_samples_leaf=2, random_state=fold, n_jobs=-1).fit(Ft[trv], R[trv][sh]).predict(Ft[te]), R[te]))
        if not a.only_trees:
            flat = Xn[tr].reshape(-1, W * C); mu, sd = flat.mean(0), flat.std(0) + 1e-8
            zc = (lambda A: np.clip((A - mu) / sd, -a.clip, a.clip)) if a.clip > 0 else (lambda A: (A - mu) / sd); Z = lambda I: zc(Xn[I])
            for spec in a.nets:
                kind, loss = spec.split("|")
                for s in range(a.seeds):
                    if a.aug:
                        tmu, tsd, tsr = torch.tensor(mu), torch.tensor(sd), torch.tensor(sd_raw.astype(np.float32))
                        def prep(raw):
                            z = (torch.sign(raw) * torch.log1p(raw.abs())).reshape(raw.shape[0], raw.shape[1], -1); z = (z - tmu) / tsd
                            return z.clamp(-a.clip, a.clip) if a.clip > 0 else z
                        rv = np.random.default_rng(900 + fold); Xva_n = zc(slog(X[va] + rv.normal(0, 1, X[va].shape).astype(np.float32) * (0.1 * sd_raw)))
                        m, n = fit(kind, loss, Z(tr), R[tr] / YS, np.concatenate([Z(va), Xva_n]), np.concatenate([R[va], R[va]]) / YS, seed=100 * fold + s, aug=(torch.tensor(X[tr]), tsr, prep))
                    else: m, n = fit(kind, loss, Z(tr), R[tr] / YS, Z(va), R[va] / YS, seed=100 * fold + s)
                    with torch.no_grad(): p = m(torch.tensor(Z(te))).numpy() * YS
                    r = score(p, R[te]); r.update(epochs=n); add(spec, r)
                    for lv, Xq in noisy.items():
                        with torch.no_grad(): addn(spec, lv, score(m(torch.tensor(zc(slog(Xq)))).numpy() * YS, R[te])["rmse"])
                    if s == 0: P_all.setdefault(spec, np.zeros_like(R))[te] = p
        print(f"[pliegue {fold}] " + " | ".join(f"{k}: {v[-1]['rmse']:.2f}" for k, v in res.items()) + f"   ({(time.time()-t0)/60:.1f} min)", flush=True)
        json.dump(res, open(os.path.join(a.out, "raw.json"), "w")); json.dump(NZ, open(os.path.join(a.out, "noise.json"), "w"))
    np.savez(os.path.join(a.out, "preds.npz"), R=R, y0=y0, **{k.replace("|", "_"): v for k, v in P_all.items()})
    summ = {k: {m: (float(np.mean([r[m] for r in v])), float(np.std([r[m] for r in v]))) for m in ("rmse", "corr")} for k, v in res.items()}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1); base = summ["media"]["rmse"][0]
    print("\n=== RESUMEN (relieve, 16 puntos) ===")
    if NZ:
        print("RUIDO en la prueba (RMSE mm): " + " | ".join(f"{k}: " + ", ".join(f"{int(float(lv)*100)}% {np.mean(v):.2f}" for lv, v in d.items()) for k, d in NZ.items()))
    for k, v in summ.items(): print(f"{k:20s} RMSE {v['rmse'][0]:.2f} ± {v['rmse'][1]:.2f} mm | mejora {100*(1-v['rmse'][0]/base):5.1f} % | corr. de forma (mediana) {v['corr'][0]:.2f}")

if __name__ == "__main__": main()
