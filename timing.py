"""How simple is it? Single-CPU-thread time per 10 s segment for decoding, features and prediction, plus model size,
for the re-encoded (128 kbps) detector."""
import time
from pathlib import Path

import cupy as cp
import librosa
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.ndimage import median_filter
from scipy.signal import sosfiltfilt
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

from prep_hifi import KEEP, LTAS_HOP, LTAS_NFFT, RESID_WIDTH, SEG_S, SOS, SR, TARGET_RMS, segment_features

SEED = 20261011
N_TIMING = 50
N_LATENCY = 200
DATA = Path("data")
RESULTS = Path("results")
FEATS = [DATA / "feat_reenc_ltas.npy", DATA / "feat_reenc_ltas_resid.npy",
         DATA / "feat_reenc_mfcc20_stats.npy", DATA / "feat_reenc_band_sig.npy"]
MODEL_PATH = RESULTS / "detector_reenc128.ubj"


def spectral_only(seg):
    S = np.abs(librosa.stft(seg, n_fft=LTAS_NFFT, hop_length=LTAS_HOP)) ** 2
    ltas = np.log10(S.mean(axis=1) + 1e-12)[KEEP]
    return ltas, ltas - median_filter(ltas, size=RESID_WIDTH, mode="nearest")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(DATA / "segments_reenc.parquet").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    pick = seg.iloc[rng.choice(len(seg), N_TIMING, replace=False)]

    t_decode, t_all, t_spec = [], [], []
    for _, r in tqdm(list(pick.iterrows()), desc="timing features (1 CPU thread)", unit="segment"):
        t0 = time.perf_counter()
        y, _ = librosa.load(r["path"], sr=SR, mono=True, offset=float(r["seg_index"] * SEG_S), duration=float(SEG_S))
        y = sosfiltfilt(SOS, y).astype(np.float32)
        rms = float(np.sqrt(np.mean(y ** 2)))
        y = y * (TARGET_RMS / max(rms, 1e-12))
        t1 = time.perf_counter()
        segment_features(y)
        t2 = time.perf_counter()
        spectral_only(y)
        t3 = time.perf_counter()
        t_decode.append(t1 - t0)
        t_all.append(t2 - t1)
        t_spec.append(t3 - t2)

    print("Training the detector on all re-encoded segments (GPU) to measure size and CPU prediction speed...")
    X = np.hstack([np.load(f).astype(np.float32) for f in FEATS])
    y = pd.read_parquet(DATA / "segments.parquet")["label"].to_numpy()
    model = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                          tree_method="hist", device="cuda", eval_metric="logloss", random_state=SEED, n_jobs=-1)
    w = compute_sample_weight("balanced", y).astype(np.float32)
    model.fit(cp.asarray(X), cp.asarray(y, dtype=cp.float32), sample_weight=cp.asarray(w))
    booster = model.get_booster()
    booster.save_model(MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    trees = booster.trees_to_dataframe()
    n_trees = trees["Tree"].nunique()
    n_nodes = len(trees)
    n_leaves = int((trees["Feature"] == "Leaf").sum())

    booster.set_param({"device": "cpu", "nthread": 1})
    Xs = X[:1000]
    booster.predict(xgb.DMatrix(Xs[:1]))
    t0 = time.perf_counter()
    for i in tqdm(range(N_LATENCY), desc="timing single-segment prediction", unit="call"):
        booster.predict(xgb.DMatrix(Xs[i:i + 1]))
    t_single = (time.perf_counter() - t0) / N_LATENCY
    t0 = time.perf_counter()
    booster.predict(xgb.DMatrix(Xs))
    t_batch = (time.perf_counter() - t0) / len(Xs)

    med = lambda v: 1000 * float(np.median(v))
    total = med(t_decode) + med(t_all) + 1000 * t_single
    print("\n=== How simple is it? (single CPU thread, per 10-second segment) ===")
    print(f"Decode + low-pass + normalize:        {med(t_decode):8.1f} ms (median)")
    print(f"All features (9,142 numbers):          {med(t_all):8.1f} ms (median)")
    print(f"Spectral features only (LTAS + resid): {med(t_spec):8.1f} ms (median)")
    print(f"XGBoost prediction, one segment:       {1000 * t_single:8.2f} ms")
    print(f"XGBoost prediction, batched:           {1000 * t_batch:8.3f} ms per segment")
    print(f"End to end, one segment:               {total:8.1f} ms  -> real-time factor {total / 1000 / SEG_S:.3f} "
          f"({SEG_S * 1000 / total:.0f}x faster than real time)")
    print(f"\nModel file: {size_mb:.2f} MB ({MODEL_PATH}); {n_trees} trees, {n_nodes:,} nodes ({n_leaves:,} leaves)")
    print("Reference: Wav2Vec2 XLS-R 2B, about 2 billion parameters (about 8 GB in float32), "
          "plus a logistic-regression head; a GPU is needed for practical speed.")


if __name__ == "__main__":
    main()