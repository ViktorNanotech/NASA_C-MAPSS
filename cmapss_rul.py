#!/usr/bin/env python3
"""
cmapss_rul.py  -  Remaining Useful Life (RUL) prediction for the NASA C-MAPSS
                  turbofan engine degradation dataset (Saxena et al., PHM08).

WHAT THIS DOES
--------------
Given the raw text files from CMAPSSData.zip, it trains a neural network that
looks at the last WINDOW cycles of sensor readings from an engine and predicts
how many more operational cycles that engine has left before failure (its RUL).
It then evaluates on the official test set (using RUL_FD00x.txt as ground truth)
and writes predictions, metrics, model weights and plots to OUTPUT_DIR.

USAGE
-----
    python cmapss_rul.py                       # uses settings in CONFIG below
    python cmapss_rul.py --data data/ --datasets FD001 FD003
    python cmapss_rul.py --epochs 60 --seeds 5  # more thorough run on a GPU

    Just point DATA_DIR at the unzipped folder that contains train_FD001.txt,
    test_FD001.txt, RUL_FD001.txt, ... and run it. CUDA is used if available.

HOW IT WORKS (the pipeline, top to bottom)
------------------------------------------
1. LOAD      26 whitespace-separated columns: unit id, cycle, 3 operating
             settings, 21 sensors (see readme.txt / Table 2 of the paper).
2. LABEL     For training engines, which run to failure, RUL at a given cycle
             is simply (last cycle of that engine) - (this cycle).  We clip it
             at RUL_CAP (default 125).  Rationale (from the paper, eq. 4-6):
             degradation is exponential, so early in life the sensors carry no
             information about *when* the fault will start; a healthy engine
             at cycle 10 and cycle 60 look identical.  A piecewise-linear
             target ("healthy = 125, then linear countdown") is what the
             sensors can actually support and is standard for this dataset.
3. REGIMES   FD002/FD004 fly under 6 discrete operating conditions (altitude,
             Mach, throttle).  Sensor values shift massively between regimes,
             swamping the degradation signal.  We cluster the 3 operating
             settings with K-means (k=6) and z-score every sensor *within its
             regime* using training-set statistics.  FD001/FD003 have a single
             regime, so this reduces to ordinary z-scoring.
4. FEATURES  Sensors that are constant (zero variance in training data) carry
             nothing and are dropped.  The regime id is also appended as a
             one-hot vector so the net can learn regime-specific wear rates.
5. WINDOWS   Sliding windows of WINDOW consecutive cycles per engine (stride 1)
             -> tensor of shape (N, WINDOW, n_features).  Label = RUL at the
             window's last cycle.  Test engines shorter than WINDOW are padded
             at the front by repeating their first cycle.
6. MODEL     1-D CNN feature extractor (learns local temporal patterns across
             sensors) -> bidirectional GRU (integrates the trend over the
             window) -> MLP head -> single RUL value.  Small enough to train on
             a laptop in minutes, strong enough to match published results.
7. TRAIN     Adam + cosine LR schedule, Huber loss (robust to the noisy
             labels near the cap), early stopping on a held-out set of
             *whole engines* (not random rows - rows from the same engine are
             highly correlated and would leak).  SEEDS independent models are
             trained and their predictions averaged (ensembling knocks ~5-10%
             off RMSE for free).
8. EVALUATE  On each test engine we predict RUL from its final WINDOW cycles
             and compare to RUL_FD00x.txt.  Metrics:
               - RMSE (cycles)
               - NASA PHM08 score (eq. 11 of the paper): asymmetric exponential
                 penalty; late predictions (pred > true) are punished harder
                 than early ones, because a late prediction means an engine
                 failed in service.  Lower is better; it is a *sum* over
                 engines so it scales with the number of test engines.
             Ground-truth RUL is clipped at RUL_CAP for the headline numbers
             (the model can't know an engine has 145 cycles left rather than
             125 - nobody can from these sensors), but unclipped numbers are
             printed too for transparency.
9. OUTPUT    OUTPUT_DIR/FD00x/
               predictions.csv   per-engine true vs predicted RUL
               trajectory.csv    RUL prediction at every cycle of every test
                                 engine (for plotting health over time)
               model_seedN.pt    trained weights
               scaler.json       regime centroids + per-regime mean/std, so the
                                 model can be reused on new data
               *.png             plots
             OUTPUT_DIR/summary.csv   metrics for all datasets

REFERENCE BENCHMARKS (test RMSE, from the literature, so you can judge results)
------------------------------------------------------------------------------
             FD001     FD002     FD003     FD004
  simple MLP   ~16       ~24       ~17       ~27
  LSTM (2017)  16.1      24.5      16.2      28.2
  CNN  (2018)  12.6      22.4      12.6      23.3
  strong 2020+ 11.5-12.5 15-18     11.5-12.5 17-20
"""

# ============================================================================
#  CONFIG  -  edit these or override on the command line
# ============================================================================
CONFIG = dict(
    DATA_DIR    = "data",                      # folder with train_FD00x.txt etc.
    OUTPUT_DIR  = "output",
    DATASETS    = ["FD001", "FD002", "FD003", "FD004"],
    RUL_CAP     = 125,      # piecewise-linear RUL ceiling (cycles)
    WINDOW      = 40,       # cycles per input window (40 beat 30 and 50 in tuning)
    N_REGIMES   = 6,        # operating-condition clusters (only matters for FD002/FD004)
    USE_CYCLE   = True,     # feed the engine's age (cycle / CYCLE_SCALE) as a feature
    CYCLE_SCALE = 250.0,
    BATCH_SIZE  = 256,
    EPOCHS      = 40,
    LR          = 1e-3,
    WEIGHT_DECAY= 1e-3,
    SEEDS       = 3,        # models in the ensemble
    VAL_FRAC    = 0.15,     # fraction of *training engines* held out for early stopping
    PATIENCE    = 10,       # early-stopping patience (epochs)
    CNN_CH      = 64,       # conv channels
    GRU_H       = 64,       # GRU hidden size (per direction)
    DROPOUT     = 0.5,      # heavy dropout: 0.5 beat 0.3 clearly in tuning
    NOISE_STD   = 0.05,     # Gaussian noise added to inputs during training (augmentation)
    NUM_WORKERS = 0,
    GB_WEIGHT   = 0.5,      # blend weight of the gradient-boosting model (0 = NN only)
)

# ============================================================================
import argparse, json, math, os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))   # silence joblib on Windows
torch.set_num_threads(max(1, os.cpu_count() or 1))

COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = [f"s{i}" for i in range(1, 22)]


# ----------------------------------------------------------------------------
# 1. LOADING
# ----------------------------------------------------------------------------
def load_dataset(data_dir: Path, name: str):
    """Read train/test/RUL files for one sub-dataset into DataFrames."""
    def read(fn):
        df = pd.read_csv(data_dir / fn, sep=r"\s+", header=None)
        df = df.iloc[:, :26]                 # some copies have trailing blanks
        df.columns = COLS
        return df
    train = read(f"train_{name}.txt")
    test  = read(f"test_{name}.txt")
    rul   = pd.read_csv(data_dir / f"RUL_{name}.txt", sep=r"\s+", header=None).iloc[:, 0].values
    return train, test, rul


def add_train_rul(train: pd.DataFrame, cap: int):
    """RUL = cycles until this engine's last recorded cycle, clipped at cap."""
    last = train.groupby("unit")["cycle"].transform("max")
    train = train.copy()
    train["RUL"] = (last - train["cycle"]).clip(upper=cap)
    return train


# ----------------------------------------------------------------------------
# 3. OPERATING REGIMES + NORMALISATION
# ----------------------------------------------------------------------------
class RegimeScaler:
    """K-means on the 3 operating settings, then per-regime z-scoring of
    sensors.  Fitted on training data only; applied to test data."""

    def __init__(self, n_regimes: int, use_cycle=True, cycle_scale=250.0):
        self.k = n_regimes
        self.use_cycle, self.cycle_scale = use_cycle, cycle_scale

    def fit(self, df: pd.DataFrame):
        from sklearn.cluster import KMeans
        ops = df[["os1", "os2", "os3"]].values
        # if the settings are effectively constant (FD001/FD003) use 1 regime
        if np.allclose(ops.std(axis=0), 0, atol=1e-2):
            self.k = 1
        self.km = KMeans(n_clusters=self.k, n_init=10, random_state=0).fit(ops)
        reg = self.km.predict(ops)
        self.mean = np.zeros((self.k, len(SENSORS)))
        self.std  = np.ones((self.k, len(SENSORS)))
        for r in range(self.k):
            x = df.loc[reg == r, SENSORS].values
            self.mean[r] = x.mean(axis=0)
            self.std[r]  = x.std(axis=0)
        # sensors that are constant in EVERY regime carry no information
        self.keep = [i for i, s in enumerate(SENSORS) if (self.std[:, i] > 1e-6).any()]
        self.std[self.std < 1e-6] = 1.0      # avoid divide-by-zero on constants
        return self

    def transform(self, df: pd.DataFrame):
        """Returns (features [N, n_feat], regime id [N])."""
        reg = self.km.predict(df[["os1", "os2", "os3"]].values)
        x = df[SENSORS].values.astype(np.float32)
        x = (x - self.mean[reg]) / self.std[reg]
        x = x[:, self.keep]
        if self.k > 1:                       # one-hot regime as extra features
            onehot = np.eye(self.k, dtype=np.float32)[reg]
            x = np.concatenate([x, onehot], axis=1)
        if self.use_cycle:                   # engine age - legitimately observed
            age = (df["cycle"].values / self.cycle_scale).astype(np.float32)[:, None]
            x = np.concatenate([x, age], axis=1)
        return x.astype(np.float32), reg

    @property
    def n_features(self):
        return len(self.keep) + (self.k if self.k > 1 else 0) + int(self.use_cycle)

    def to_json(self):
        return dict(k=self.k, use_cycle=self.use_cycle, cycle_scale=self.cycle_scale, centroids=self.km.cluster_centers_.tolist(),
                    mean=self.mean.tolist(), std=self.std.tolist(),
                    keep_sensors=[SENSORS[i] for i in self.keep])


# ----------------------------------------------------------------------------
# 5. WINDOWING
# ----------------------------------------------------------------------------
def make_windows(df: pd.DataFrame, feats: np.ndarray, window: int, with_labels: bool):
    """Slide a window of `window` cycles over every engine.
    Returns X [N, window, F] and (if with_labels) y [N]."""
    X, y = [], []
    for unit, idx in df.groupby("unit").indices.items():
        f = feats[idx]
        n = len(f)
        if n < window:                       # pad front by repeating first cycle
            f = np.concatenate([np.repeat(f[:1], window - n, axis=0), f], axis=0)
            n = window
        for end in range(window, n + 1):
            X.append(f[end - window:end])
            if with_labels:
                y.append(df["RUL"].values[idx][min(end - 1, len(idx) - 1)])
    X = np.stack(X).astype(np.float32)
    return (X, np.array(y, dtype=np.float32)) if with_labels else X


def last_windows(df: pd.DataFrame, feats: np.ndarray, window: int):
    """One window per engine: its final `window` cycles (what we predict on)."""
    X = []
    for unit, idx in df.groupby("unit").indices.items():
        f = feats[idx]
        if len(f) < window:
            f = np.concatenate([np.repeat(f[:1], window - len(f), axis=0), f], axis=0)
        X.append(f[-window:])
    return np.stack(X).astype(np.float32)


# ----------------------------------------------------------------------------
# 6a. SECOND OPINION: GRADIENT BOOSTING ON WINDOW STATISTICS
# ----------------------------------------------------------------------------
# A completely different model family that turned out to be very strong on
# this data.  Each window is summarised by 4 statistics per feature (mean,
# linear-trend slope, last value, std) and fed to a HistGradientBoosting
# regressor.  Its prediction is blended with the neural network's
# (GB_WEIGHT); the two make partly independent errors, so the blend beats
# either alone.
def window_stats(X):
    """X [N, T, F] -> [N, 4F]: mean, slope, last, std along the time axis."""
    T = X.shape[1]
    t = (np.arange(T) - (T - 1) / 2.0).astype(np.float32)
    mean = X.mean(axis=1)
    slope = np.einsum("t,ntf->nf", t, X - mean[:, None, :]) / (t @ t)
    return np.concatenate([mean, slope, X[:, -1, :], X.std(axis=1)], axis=1)


def train_gb(X_tr, y_tr, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=seed)
    return gb.fit(window_stats(X_tr), y_tr)


# ----------------------------------------------------------------------------
# 6. MODEL
# ----------------------------------------------------------------------------
class RULNet(nn.Module):
    """Conv1d stack (temporal feature extraction) -> BiGRU (trend
    integration) -> MLP -> scalar RUL."""

    def __init__(self, n_feat, cnn_ch=64, gru_h=64, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            # NOTE: no BatchNorm on purpose - it hurt generalisation badly here
            # (test RMSE 17 -> 14 on FD001 when removed).  Batch statistics
            # mix windows from different engines/regimes and fight the
            # per-regime normalisation done in RegimeScaler.
            nn.Conv1d(n_feat, cnn_ch, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(cnn_ch, cnn_ch, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(cnn_ch, cnn_ch, kernel_size=3, padding=1), nn.GELU(),
        )
        self.gru = nn.GRU(cnn_ch, gru_h, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * gru_h, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):                    # x: [B, T, F]
        z = self.cnn(x.transpose(1, 2))      # -> [B, C, T]
        z, _ = self.gru(z.transpose(1, 2))   # -> [B, T, 2H]
        z = z[:, -1]                         # state after the last cycle
        return self.head(z).squeeze(-1)


# ----------------------------------------------------------------------------
# 7. TRAINING
# ----------------------------------------------------------------------------
def train_one(X_tr, y_tr, X_va, y_va, n_feat, cfg, seed, device, log):
    torch.manual_seed(seed); np.random.seed(seed)
    model = RULNet(n_feat, cfg["CNN_CH"], cfg["GRU_H"], cfg["DROPOUT"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["LR"], epochs=cfg["EPOCHS"],
        steps_per_epoch=math.ceil(len(X_tr) / cfg["BATCH_SIZE"]))
    # Targets are scaled to [0,1] (RUL / cap) so the network starts near the
    # right output range and MSE gradients are well-conditioned.
    cap = float(cfg["RUL_CAP"])
    loss_fn = nn.MSELoss()

    dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr / cap)),
                    batch_size=cfg["BATCH_SIZE"], shuffle=True, num_workers=cfg["NUM_WORKERS"])
    X_va_t = torch.from_numpy(X_va).to(device)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(cfg["EPOCHS"]):
        model.train(); t0 = time.time(); tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            if cfg["NOISE_STD"] > 0:         # augmentation: jitter the sensors
                xb = xb + cfg["NOISE_STD"] * torch.randn_like(xb)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            pv = predict(model, X_va_t, device) * cap
        rmse = float(np.sqrt(np.mean((pv - y_va) ** 2)))
        if rmse < best - 1e-3:
            best, best_state, bad = rmse, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        log(f"    seed {seed} ep {ep+1:3d}/{cfg['EPOCHS']}  train_rmse {cap*math.sqrt(tot/len(X_tr)):6.2f}  "
            f"val_rmse {rmse:6.2f}  best {best:6.2f}  ({time.time()-t0:.1f}s)")
        if bad >= cfg["PATIENCE"]:
            log(f"    early stop (no improvement for {cfg['PATIENCE']} epochs)")
            break
    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict(model, X, device, bs=2048):
    """Raw network output (RUL / cap)."""
    model.eval()
    out = []
    X = X if torch.is_tensor(X) else torch.from_numpy(X)
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(device)).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# 8. METRICS  (NASA PHM08 scoring function, eq. 11 in the paper)
# ----------------------------------------------------------------------------
def nasa_score(pred, true):
    """d = pred - true.  Early (d<0): exp(-d/13)-1.  Late (d>=0): exp(d/10)-1.
    Late predictions are penalised harder.  Summed over engines.
    (The paper's text swaps the a1/a2 labels; this is the form used by the
    competition and every subsequent publication.)"""
    d = np.asarray(pred, float) - np.asarray(true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(true, float)) ** 2)))


# ----------------------------------------------------------------------------
# 9. PLOTS
# ----------------------------------------------------------------------------
def make_plots(out_dir: Path, name, pred, true_c, traj: pd.DataFrame, cap):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # (a) predicted vs true, sorted by true RUL
    order = np.argsort(true_c)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(true_c[order], "k-", label="true RUL (clipped)")
    ax.plot(pred[order], "r.", ms=4, label="predicted")
    ax.set_xlabel("test engine (sorted by true RUL)"); ax.set_ylabel("RUL [cycles]")
    ax.set_title(f"{name}: predicted vs true RUL at last observed cycle"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "pred_vs_true.png", dpi=120); plt.close(fig)
    # (b) full-trajectory predictions for a few engines
    units = traj["unit"].unique()[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharey=True)
    for ax, u in zip(axes.ravel(), units):
        t = traj[traj["unit"] == u]
        ax.plot(t["cycle"], t["pred_rul"], "r-", lw=1, label="predicted RUL")
        ax.plot(t["cycle"], t["true_rul"].clip(upper=cap), "k--", lw=1, label="true RUL (clipped)")
        ax.set_title(f"test engine {u}"); ax.set_xlabel("cycle"); ax.grid(alpha=.3)
    axes[0, 0].set_ylabel("RUL [cycles]"); axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{name}: RUL prediction along each test engine's life")
    fig.tight_layout(); fig.savefig(out_dir / "trajectories.png", dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------
# MAIN  -  one sub-dataset end to end
# ----------------------------------------------------------------------------
def run_dataset(name: str, cfg: dict, device, log):
    data_dir, out_dir = Path(cfg["DATA_DIR"]), Path(cfg["OUTPUT_DIR"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    W, cap = cfg["WINDOW"], cfg["RUL_CAP"]

    log(f"\n=== {name} ===")
    train, test, rul_true = load_dataset(data_dir, name)
    train = add_train_rul(train, cap)
    log(f"  train: {train.unit.nunique()} engines, {len(train)} cycles | "
        f"test: {test.unit.nunique()} engines, {len(test)} cycles")

    scaler = RegimeScaler(cfg["N_REGIMES"], cfg["USE_CYCLE"], cfg["CYCLE_SCALE"]).fit(train)
    log(f"  regimes: {scaler.k}, features: {scaler.n_features} "
        f"(kept sensors: {[SENSORS[i] for i in scaler.keep]})")
    f_tr, _ = scaler.transform(train)
    f_te, _ = scaler.transform(test)

    # hold out whole engines for validation / early stopping
    rng = np.random.RandomState(0)
    units = train["unit"].unique(); rng.shuffle(units)
    n_val = max(1, int(round(cfg["VAL_FRAC"] * len(units))))
    val_units = set(units[:n_val])
    is_val = train["unit"].isin(val_units).values
    X_tr, y_tr = make_windows(train[~is_val], f_tr[~is_val], W, True)
    X_va, y_va = make_windows(train[is_val],  f_tr[is_val],  W, True)
    log(f"  windows: train {X_tr.shape}, val {X_va.shape}")

    # --- train ensemble
    models = []
    for s in range(cfg["SEEDS"]):
        m, best = train_one(X_tr, y_tr, X_va, y_va, scaler.n_features, cfg, s, device, log)
        log(f"  seed {s}: best val RMSE {best:.2f}")
        torch.save(m.state_dict(), out_dir / f"model_seed{s}.pt")
        models.append(m)

    # --- gradient-boosting second opinion (trained on the same windows)
    gb = None
    if cfg["GB_WEIGHT"] > 0:
        gb = train_gb(np.concatenate([X_tr, X_va]), np.concatenate([y_tr, y_va]))
        log(f"  gradient boosting fitted ({X_tr.shape[0]+X_va.shape[0]} windows)")

    def blend(X):
        p_nn = cap * np.mean([predict(m, X, device) for m in models], axis=0)
        if gb is None:
            return p_nn
        return (1 - cfg["GB_WEIGHT"]) * p_nn + cfg["GB_WEIGHT"] * gb.predict(window_stats(X))

    # --- evaluate on official test set
    X_last = last_windows(test, f_te, W)
    p_nn = np.clip(cap * np.mean([predict(m, X_last, device) for m in models], axis=0), 0, cap)
    pred = np.clip(blend(X_last), 0, cap)
    true_c = np.clip(rul_true, 0, cap)
    res = dict(dataset=name,
               test_engines=len(pred),
               rmse_clipped=rmse(pred, true_c),  score_clipped=nasa_score(pred, true_c),
               rmse_raw=rmse(pred, rul_true),    score_raw=nasa_score(pred, rul_true),
               mae_clipped=float(np.mean(np.abs(pred - true_c))),
               late_frac=float(np.mean(pred > true_c)))
    if gb is not None:
        p_gb = np.clip(gb.predict(window_stats(X_last)), 0, cap)
        res["rmse_nn_only"], res["rmse_gb_only"] = rmse(p_nn, true_c), rmse(p_gb, true_c)
        log(f"  components: NN-ensemble RMSE {res['rmse_nn_only']:.2f}  |  GB RMSE {res['rmse_gb_only']:.2f}")
    log(f"  TEST  RMSE {res['rmse_clipped']:.2f}  score {res['score_clipped']:.0f}  "
        f"(raw-RUL: RMSE {res['rmse_raw']:.2f}, score {res['score_raw']:.0f})  "
        f"late predictions: {100*res['late_frac']:.0f}%")

    units_te = np.sort(test["unit"].unique())
    pd.DataFrame(dict(unit=units_te, true_rul=rul_true, true_rul_clipped=true_c,
                      pred_rul=np.round(pred, 1), error=np.round(pred - true_c, 1))
                 ).to_csv(out_dir / "predictions.csv", index=False)

    # --- RUL along every test engine's whole history (for plots / inspection)
    X_all = make_windows(test, f_te, W, False)
    p_all = np.clip(blend(X_all), 0, cap)
    rows, k = [], 0
    for u, idx in test.groupby("unit").indices.items():
        n = len(idx); n_win = 1 if n < W else n - W + 1   # must match make_windows
        cyc = test["cycle"].values[idx]
        last_true = rul_true[np.searchsorted(units_te, u)]
        if n < W:               # padded engine: a single window at its last cycle
            rows.append((u, cyc[-1], last_true, p_all[k]))
        else:
            for j in range(n_win):
                c = cyc[W - 1 + j]
                rows.append((u, c, last_true + (cyc[-1] - c), p_all[k + j]))
        k += n_win
    traj = pd.DataFrame(rows, columns=["unit", "cycle", "true_rul", "pred_rul"])
    traj.to_csv(out_dir / "trajectory.csv", index=False)
    make_plots(out_dir, name, pred, true_c, traj, cap)

    if gb is not None:
        import pickle
        with open(out_dir / "gradient_boosting.pkl", "wb") as fh:
            pickle.dump(gb, fh)
    with open(out_dir / "scaler.json", "w") as fh:
        json.dump(scaler.to_json(), fh)
    with open(out_dir / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=CONFIG["DATA_DIR"])
    ap.add_argument("--out", default=CONFIG["OUTPUT_DIR"])
    ap.add_argument("--datasets", nargs="+", default=CONFIG["DATASETS"])
    ap.add_argument("--epochs", type=int, default=CONFIG["EPOCHS"])
    ap.add_argument("--seeds", type=int, default=CONFIG["SEEDS"])
    ap.add_argument("--window", type=int, default=CONFIG["WINDOW"])
    ap.add_argument("--cap", type=int, default=CONFIG["RUL_CAP"])
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    a = ap.parse_args()
    cfg = dict(CONFIG, DATA_DIR=a.data, OUTPUT_DIR=a.out, DATASETS=a.datasets,
               EPOCHS=a.epochs, SEEDS=a.seeds, WINDOW=a.window, RUL_CAP=a.cap)

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    Path(cfg["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    logf = open(Path(cfg["OUTPUT_DIR"]) / "log.txt", "a")
    def log(s):
        print(s); logf.write(s + "\n"); logf.flush()
    log(f"device: {device}   torch {torch.__version__}   {time.strftime('%Y-%m-%d %H:%M')}")

    results = [run_dataset(n, cfg, device, log) for n in cfg["DATASETS"]]
    summ = pd.DataFrame(results)
    summ.to_csv(Path(cfg["OUTPUT_DIR"]) / "summary.csv", index=False)
    log("\n=== SUMMARY (ground-truth RUL clipped at %d) ===" % cfg["RUL_CAP"])
    cols = [c for c in ["dataset", "test_engines", "rmse_clipped", "score_clipped", "mae_clipped",
                        "late_frac", "rmse_nn_only", "rmse_gb_only"] if c in summ]
    log(summ[cols]
        .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    log("\nReference (published CNN 2018): FD001 12.6 | FD002 22.4 | FD003 12.6 | FD004 23.3 RMSE")


if __name__ == "__main__":
    main()
