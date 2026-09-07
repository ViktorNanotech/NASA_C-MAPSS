#!/usr/bin/env python3
"""
crosscheck.py - independent sanity check for cmapss_rul.py

Deliberately shares NO code with the neural-network script.  It:
  1. re-reads the raw files with its own parser,
  2. builds hand-crafted features (mean / slope / last value of each sensor
     over the final N cycles, plus operating-condition regime),
  3. fits a HistGradientBoosting regressor (a completely different model
     family),
  4. re-implements RMSE and the PHM08 score from eq. 11 of the paper,
  5. compares its own metrics on the NN's predictions.csv against the numbers
     the NN script printed - if the two metric implementations disagree, one
     of them is wrong.

Usage:  python crosscheck.py [--data data] [--out output]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.cluster import KMeans

CAP, N = 125, 40
S = [f"s{i}" for i in range(1, 22)]

def read(p):
    a = np.loadtxt(p)[:, :26]
    return pd.DataFrame(a, columns=["unit", "cycle", "o1", "o2", "o3"] + S)

def score_phm08(pred, true):
    d = pred - true
    return sum(np.exp(-x / 13) - 1 if x < 0 else np.exp(x / 10) - 1 for x in d)

def rmse(p, t): return np.sqrt(((p - t) ** 2).mean())

def features(df, km, stats):
    """One feature row per (unit, cycle) using the trailing N cycles."""
    reg = km.predict(df[["o1", "o2", "o3"]].values)
    x = df[S].values.copy()
    for r in range(len(stats)):
        m = reg == r
        x[m] = (x[m] - stats[r][0]) / stats[r][1]
    df = df.assign(reg=reg, **{s: x[:, i] for i, s in enumerate(S)})
    rows, ys, units = [], [], []
    t = np.arange(N) - N / 2
    for u, g in df.groupby("unit"):
        v = g[S].values; last = g.cycle.max()
        for end in range(1, len(g) + 1):
            w = v[max(0, end - N):end]
            if len(w) < N: w = np.vstack([np.repeat(w[:1], N - len(w), 0), w])
            slope = (t @ (w - w.mean(0))) / (t @ t)
            rows.append(np.concatenate([w.mean(0), slope, w[-1], w.std(0), [g.reg.values[end - 1], end]]))
            ys.append(min(last - g.cycle.values[end - 1], CAP)); units.append(u)
    return np.array(rows), np.array(ys), np.array(units)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="output")
    ap.add_argument("--datasets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    a = ap.parse_args(); D, O = Path(a.data), Path(a.out)
    for name in a.datasets:
        tr, te = read(D / f"train_{name}.txt"), read(D / f"test_{name}.txt")
        true = np.loadtxt(D / f"RUL_{name}.txt"); true_c = np.minimum(true, CAP)
        ops = tr[["o1", "o2", "o3"]].values
        k = 1 if ops.std(0).max() < 1e-2 else 6
        km = KMeans(k, n_init=10, random_state=1).fit(ops)
        reg = km.predict(ops)
        stats = [(tr.loc[reg == r, S].values.mean(0), np.where(tr.loc[reg == r, S].values.std(0) < 1e-6, 1, tr.loc[reg == r, S].values.std(0))) for r in range(k)]
        Xtr, ytr, _ = features(tr, km, stats)
        Xte, _, ute = features(te, km, stats)
        # last row of each test engine
        last_idx = [np.where(ute == u)[0][-1] for u in np.unique(ute)]
        gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
        p = np.clip(gb.predict(Xte[last_idx]), 0, CAP)
        print(f"{name}: GradientBoosting baseline  RMSE {rmse(p, true_c):.2f}  score {score_phm08(p, true_c):.0f}")
        f = O / name / "predictions.csv"
        if f.exists():
            nn = pd.read_csv(f)
            assert np.allclose(nn.true_rul.values, true), "predictions.csv true_rul does not match RUL file!"
            pn = nn.pred_rul.values
            print(f"{name}: NN predictions re-scored     RMSE {rmse(pn, true_c):.2f}  score {score_phm08(pn, true_c):.0f}"
                  f"   | corr(NN, GB) = {np.corrcoef(pn, p)[0,1]:.3f}")
        else:
            print(f"{name}: no NN predictions found at {f}")

if __name__ == "__main__":
    main()
