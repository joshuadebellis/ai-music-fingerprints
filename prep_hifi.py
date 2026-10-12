"""High-band traditional features for the same 10 s segments as the segments table (rows align).
32 kHz, zero-phase low-pass at 15 kHz, RMS-normalized segments. Families:
  ltas        : long-term average log power spectrum, n_fft 4096, bins 0-15 kHz
  ltas_resid  : ltas minus a median-smoothed copy (highlights narrow spectral peaks)
  ltas_12_15k : ltas restricted to 12-15 kHz (shortcut check)
  mfcc20_stats: mean/std of 20 MFCCs and their deltas (n_fft 2048)
  band_sig    : full depth-3 signature of time + lead-lag(8 log-mel band energies), frames pooled x4
Environment: SEGMENTS_FILE (default data/segments.parquet), FEAT_PREFIX (default feat_hifi)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import esig
import librosa
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

SR = 32000
SEG_S = 10
MAX_TOTAL_S = 180
CUTOFF_HZ = 15000
TARGET_RMS = 0.1
LTAS_NFFT = 4096
LTAS_HOP = 1024
RESID_WIDTH = 31
MFCC_N = 20
MEL_NFFT = 2048
MEL_HOP = 512
N_MELS = 64
N_BANDS = 8
POOL = 4
DEPTH = 3
DATA = Path("data")
SEGMENTS = Path(os.environ.get("SEGMENTS_FILE", str(DATA / "segments.parquet")))
PREFIX = os.environ.get("FEAT_PREFIX", "feat_hifi")
OUT = {
    "ltas": DATA / f"{PREFIX}_ltas.npy",
    "ltas_resid": DATA / f"{PREFIX}_ltas_resid.npy",
    "ltas_12_15k": DATA / f"{PREFIX}_ltas_12_15k.npy",
    "mfcc20_stats": DATA / f"{PREFIX}_mfcc20_stats.npy",
    "band_sig": DATA / f"{PREFIX}_band_sig.npy",
}

FREQS = librosa.fft_frequencies(sr=SR, n_fft=LTAS_NFFT)
KEEP = FREQS <= CUTOFF_HZ
HI = (FREQS[KEEP] >= 12000)
SOS = butter(8, CUTOFF_HZ, btype="low", fs=SR, output="sos")


def lead_lag(x):
    r = np.repeat(x, 2, axis=0)
    return np.hstack([r[1:], r[:-1]])


def segment_features(seg):
    S = np.abs(librosa.stft(seg, n_fft=LTAS_NFFT, hop_length=LTAS_HOP)) ** 2
    ltas = np.log10(S.mean(axis=1) + 1e-12)[KEEP]
    resid = ltas - median_filter(ltas, size=RESID_WIDTH, mode="nearest")

    mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=MFCC_N, n_fft=MEL_NFFT, hop_length=MEL_HOP, fmax=CUTOFF_HZ)
    d = librosa.feature.delta(mfcc)
    mstats = np.r_[mfcc.mean(axis=1), mfcc.std(axis=1), d.mean(axis=1), d.std(axis=1)]

    mel = librosa.feature.melspectrogram(y=seg, sr=SR, n_fft=MEL_NFFT, hop_length=MEL_HOP, n_mels=N_MELS, fmax=CUTOFF_HZ)
    logmel = librosa.power_to_db(mel) / 10.0
    bands = logmel.reshape(N_BANDS, N_MELS // N_BANDS, -1).mean(axis=1).T
    n = (len(bands) // POOL) * POOL
    bands = bands[:n].reshape(-1, POOL, N_BANDS).mean(axis=1)
    ll = lead_lag(bands)
    t = np.linspace(0.0, 1.0, len(ll))[:, None]
    sig = esig.stream2sig(np.hstack([t, ll]).astype(np.float64), DEPTH)
    return ltas, resid, ltas[HI], mstats, sig


def featurize_track(args):
    path, seg_indices, idxs = args
    y, _ = librosa.load(path, sr=SR, mono=True, duration=MAX_TOTAL_S)
    y = sosfiltfilt(SOS, y).astype(np.float32)
    seg_len = SEG_S * SR
    cols = {k: [] for k in OUT}
    for i in seg_indices:
        seg = y[i * seg_len:(i + 1) * seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        rms = float(np.sqrt(np.mean(seg ** 2)))
        seg = seg * (TARGET_RMS / max(rms, 1e-12))
        ltas, resid, hi, mstats, sig = segment_features(seg)
        cols["ltas"].append(ltas)
        cols["ltas_resid"].append(resid)
        cols["ltas_12_15k"].append(hi)
        cols["mfcc20_stats"].append(mstats)
        cols["band_sig"].append(sig)
    return idxs, {k: np.asarray(v, dtype=np.float32) for k, v in cols.items()}


def main():
    print(f"Segments table: {SEGMENTS}; output prefix: {PREFIX}")
    seg = pd.read_parquet(SEGMENTS).reset_index(drop=True)
    n = len(seg)
    tasks = [(g["path"].iloc[0], g["seg_index"].to_numpy(), g.index.to_numpy()) for _, g in seg.groupby("track_key")]
    idxs, probe = featurize_track(tasks[0])
    dims = {k: v.shape[1] for k, v in probe.items()}
    print(f"{n} segments, {len(tasks)} tracks. Feature dims: {dims}")

    outs = {k: np.lib.format.open_memmap(OUT[k], mode="w+", dtype=np.float32, shape=(n, dims[k])) for k in OUT}
    for k in OUT:
        outs[k][idxs] = probe[k]
    workers = max(1, (os.cpu_count() or 2) - 1)
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(featurize_track, t): t[0] for t in tasks[1:]}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"hi-fi features ({workers} workers)", unit="track"):
            try:
                idxs, cols = fut.result()
            except Exception as e:
                failed.append(f"{futures[fut]}: {type(e).__name__}: {e}")
                continue
            for k in OUT:
                outs[k][idxs] = cols[k]
    for a in outs.values():
        a.flush()
    print(f"\nWrote: " + ", ".join(f"{OUT[k].name} {outs[k].shape}" for k in OUT))
    for k in OUT:
        sample = np.asarray(outs[k][:3000])
        print(f"  {k}: non-finite values in first 3000 rows = {int((~np.isfinite(sample)).sum())}")
    if failed:
        print(f"\n{len(failed)} tracks failed (their rows are zeros), first few:")
        for f in failed[:5]:
            print(f"  {f}")


if __name__ == "__main__":
    main()