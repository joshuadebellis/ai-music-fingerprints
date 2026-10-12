"""Chroma path features for the same segments as data/segments.parquet (rows align):
depth-4 log-signature of (time + 12 chroma) with basepoint, raw (~313 frames) and pooled x4 (~78 frames),
plus a baseline of per-chroma mean and std."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import esig
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

SR = 16000
SEG_S = 10
MAX_SEGS = 18
N_FFT = 4096
HOP = 512
POOL = 4
DEPTH = 4
TARGET_RMS = 0.1
N_CH = 13
DATA = Path("data")
SEGMENTS = DATA / "segments.parquet"
OUT_RAW = DATA / "feat_chroma_logsig.npy"
OUT_POOL = DATA / "feat_chroma_logsig_pool4.npy"
OUT_BASE = DATA / "feat_chroma_baseline.npy"


def to_path(c):
    t = np.linspace(0.0, 1.0, len(c))[:, None]
    p = np.hstack([t, c])
    return np.vstack([np.zeros((1, p.shape[1])), p]).astype(np.float64)


def pool(c, k):
    n = (len(c) // k) * k
    return c[:n].reshape(-1, k, c.shape[1]).mean(axis=1)


def featurize_track(args):
    path, seg_indices, idxs = args
    y, _ = librosa.load(path, sr=SR, mono=True, duration=SEG_S * MAX_SEGS)
    seg_len = SEG_S * SR
    raw, pooled, base = [], [], []
    for i in seg_indices:
        seg = y[i * seg_len:(i + 1) * seg_len]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        seg = seg * (TARGET_RMS / max(rms, 1e-12))
        c = librosa.feature.chroma_stft(y=seg, sr=SR, n_fft=N_FFT, hop_length=HOP).T
        raw.append(esig.stream2logsig(to_path(c), DEPTH))
        pooled.append(esig.stream2logsig(to_path(pool(c, POOL)), DEPTH))
        base.append(np.r_[c.mean(axis=0), c.std(axis=0)])
    return idxs, np.asarray(raw, np.float32), np.asarray(pooled, np.float32), np.asarray(base, np.float32)


def main():
    seg = pd.read_parquet(SEGMENTS).reset_index(drop=True)
    n = len(seg)
    dim = esig.logsigdim(N_CH, DEPTH)
    print(f"{n} segments; path channels {N_CH}, depth {DEPTH}, log-signature length {dim}")
    tasks = [(g["path"].iloc[0], g["seg_index"].to_numpy(), g.index.to_numpy()) for _, g in seg.groupby("track_key")]

    idxs, r, p, b = featurize_track(tasks[0])
    if r.shape[1] != dim or p.shape[1] != dim:
        raise SystemExit(f"Unexpected log-signature length {r.shape[1]} / {p.shape[1]} (expected {dim})")
    print(f"Probe OK on first track: {len(idxs)} segments, raw {r.shape}, pooled {p.shape}, baseline {b.shape}")

    raw_out = np.lib.format.open_memmap(OUT_RAW, mode="w+", dtype=np.float32, shape=(n, dim))
    pool_out = np.lib.format.open_memmap(OUT_POOL, mode="w+", dtype=np.float32, shape=(n, dim))
    base_out = np.lib.format.open_memmap(OUT_BASE, mode="w+", dtype=np.float32, shape=(n, 24))
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(featurize_track, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"chroma + log-signatures ({workers} workers)", unit="track"):
            idxs, r, p, b = fut.result()
            raw_out[idxs] = r
            pool_out[idxs] = p
            base_out[idxs] = b
    for a in (raw_out, pool_out, base_out):
        a.flush()
    print(f"\nWrote {OUT_RAW} {raw_out.shape}, {OUT_POOL} {pool_out.shape}, {OUT_BASE} {base_out.shape}")
    check = np.asarray(raw_out[:2000])
    print(f"Non-finite values in first 2000 raw log-signature rows: {int((~np.isfinite(check)).sum())}")


if __name__ == "__main__":
    main()