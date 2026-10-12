"""Per-segment features: depth-4 log-signature of the (time + 13 standardized MFCC) path with basepoint,
plus a baseline of per-MFCC mean and std. Rows align with data/segments.parquet."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import esig
import numpy as np
import pandas as pd
from tqdm import tqdm

DEPTH = 4
SEED = 20261011
SCALER_SAMPLE = 2000
DATA = Path("data")
SEGMENTS = DATA / "segments.parquet"
OUT_SIG = DATA / "feat_logsig.npy"
OUT_BASE = DATA / "feat_baseline.npy"


def segment_path(m, mu, sd):
    z = (m - mu) / sd
    t = np.linspace(0.0, 1.0, len(z))[:, None]
    p = np.hstack([t, z])
    return np.vstack([np.zeros((1, p.shape[1])), p]).astype(np.float64)


def featurize_file(args):
    path, rows, idxs, mu, sd = args
    X = np.load(path)
    sig = []
    base = []
    for r in rows:
        m = X[r]
        sig.append(esig.stream2logsig(segment_path(m, mu, sd), DEPTH))
        base.append(np.r_[m.mean(axis=0), m.std(axis=0)])
    return idxs, np.asarray(sig, dtype=np.float32), np.asarray(base, dtype=np.float32)


def main():
    seg = pd.read_parquet(SEGMENTS)
    n = len(seg)
    print(f"{n} segments")

    rng = np.random.default_rng(SEED)
    sample = seg.iloc[rng.choice(n, size=min(SCALER_SAMPLE, n), replace=False)]
    frames = []
    for f, g in tqdm(sample.groupby("mfcc_file"), desc="scaler sample", unit="file"):
        X = np.load(f)
        frames.append(X[g["row_in_file"].to_numpy()].reshape(-1, X.shape[2]))
    frames = np.vstack(frames)
    mu = frames.mean(axis=0)
    sd = frames.std(axis=0) + 1e-8
    print(f"Global MFCC scaler from {len(sample)} segments: mean {np.round(mu[:3], 2)}..., std {np.round(sd[:3], 2)}...")

    first = seg.iloc[0]
    probe = esig.stream2logsig(segment_path(np.load(first["mfcc_file"])[first["row_in_file"]], mu, sd), DEPTH)
    n_ch = 1 + np.load(first["mfcc_file"]).shape[2]
    print(f"Path channels: {n_ch}, depth {DEPTH}, log-signature length {len(probe)}"
          f" (esig.logsigdim says {esig.logsigdim(n_ch, DEPTH)})")
    dim = len(probe)

    sig_out = np.lib.format.open_memmap(OUT_SIG, mode="w+", dtype=np.float32, shape=(n, dim))
    base_out = np.lib.format.open_memmap(OUT_BASE, mode="w+", dtype=np.float32, shape=(n, 26))
    seg = seg.reset_index(drop=True)
    tasks = [(f, g["row_in_file"].to_numpy(), g.index.to_numpy(), mu, sd) for f, g in seg.groupby("mfcc_file")]
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(featurize_file, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"log-signatures ({workers} workers)", unit="track"):
            idxs, s, b = fut.result()
            sig_out[idxs] = s
            base_out[idxs] = b
    sig_out.flush()
    base_out.flush()
    print(f"\nWrote {OUT_SIG} {sig_out.shape} and {OUT_BASE} {base_out.shape}")
    print(f"Non-finite values in log-signatures: {int((~np.isfinite(np.asarray(sig_out[:2000]))).sum())} (first 2000 rows checked)")


if __name__ == "__main__":
    main()