"""Which frequencies does the residual-spectrum detector use, and do generators leave evenly spaced peaks?
Usage: importance.py [feature_file] [tag]   (default: data/feat_reenc_ltas_resid.npy reenc128)"""
import sys
from pathlib import Path

import cupy as cp
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

SEED = 20261011
SR = 32000
NFFT = 4096
FEAT_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/feat_reenc_ltas_resid.npy")
TAG = sys.argv[2] if len(sys.argv) > 2 else "reenc128"
DATA = Path("data")
FIG = Path("figures")
RESULTS = Path("results")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(DATA / "segments.parquet").reset_index(drop=True)
    y = seg["label"].to_numpy()
    gens = seg["generator"].to_numpy()
    X = np.load(FEAT_FILE).astype(np.float32)
    print(f"Features: {FEAT_FILE} {X.shape}; tag {TAG}")
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)[:X.shape[1]]
    bin_hz = SR / NFFT

    model = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                          tree_method="hist", device="cuda", eval_metric="logloss", random_state=SEED, n_jobs=-1)
    w = compute_sample_weight("balanced", y).astype(np.float32)
    model.fit(cp.asarray(X), cp.asarray(y, dtype=cp.float32), sample_weight=cp.asarray(w))
    gain = model.get_booster().get_score(importance_type="gain")
    imp = np.zeros(X.shape[1])
    for k, v in gain.items():
        imp[int(k[1:])] = v
    imp /= imp.sum()
    top = np.argsort(imp)[::-1][:20]
    print("Top 20 frequencies by importance (Hz, share of total gain):")
    for i in top:
        print(f"  {freqs[i]:8.1f} Hz  {imp[i]:.4f}")
    bands = [(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10000), (10000, 12000), (12000, 15001)]
    print("\nShare of importance by frequency band:")
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        print(f"  {lo:>5}-{min(hi, 15000):>5} Hz: {imp[m].sum():.3f}")

    real_mean = X[y == 0].mean(axis=0)
    fake_gens = sorted(g for g in np.unique(gens) if g != "real")
    diffs = {}
    rows = []
    for g in tqdm(fake_gens, desc="generator spectra", unit="gen"):
        d = X[gens == g].mean(axis=0) - real_mean
        diffs[g] = d
        dc = d - d.mean()
        ac = np.correlate(dc, dc, mode="full")[len(dc) - 1:]
        ac = ac / ac[0]
        lo, hi = 4, 400
        lag = lo + int(np.argmax(ac[lo:hi]))
        rows.append({"generator": g, "peak_spacing_hz": round(lag * bin_hz, 1), "autocorr_at_spacing": round(float(ac[lag]), 3),
                     "mean_abs_diff": round(float(np.abs(d).mean()), 4)})
    per = pd.DataFrame(rows).sort_values("autocorr_at_spacing", ascending=False)
    per.to_csv(RESULTS / f"spectral_periodicity_{TAG}.csv", index=False)
    print("\nRegular spacing in each generator's residual spectrum (vs real). High autocorrelation = evenly spaced peaks:")
    print(per.to_string(index=False))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(freqs, imp, color="#333333", lw=0.8)
    ax.scatter(freqs[top], imp[top], color="#D55E00", s=14, zorder=3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Share of model gain")
    ax.set_title(f"Which frequencies the residual-spectrum detector relies on ({TAG})", loc="left")
    fig.savefig(FIG / f"residual_importance_{TAG}.png", dpi=200)

    fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True, sharey=True, constrained_layout=True)
    for ax, g in zip(axes.ravel(), fake_gens):
        ax.plot(freqs, diffs[g], color="#0072B2", lw=0.6)
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_title(g, loc="left", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Residual vs real (log10 power)")
    fig.suptitle(f"Fine spectral structure: each generator's mean residual spectrum minus real music's ({TAG})", x=0.01, ha="left")
    fig.savefig(FIG / f"generator_residual_spectra_{TAG}.png", dpi=200)
    print(f"\nWrote {FIG / f'residual_importance_{TAG}.png'}, {FIG / f'generator_residual_spectra_{TAG}.png'}, "
          f"{RESULTS / f'spectral_periodicity_{TAG}.csv'}")


if __name__ == "__main__":
    main()