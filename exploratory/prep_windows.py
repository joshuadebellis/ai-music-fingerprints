"""Longer windows: re-cut tracks into W-second segments (first 180 s), 13 MFCCs, RMS-normalized per segment.
Features: depth-3 log-signature of the (time + standardized MFCC) path with basepoint, plus per-MFCC mean and std.
Usage: prep_windows.py <window_seconds>"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import esig
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

SR = 16000
MAX_TOTAL_S = 180
N_MFCC = 13
N_FFT = 512
HOP = 256
TARGET_RMS = 0.1
SILENCE_RMS = 1e-4
DEPTH = 3
SEED = 20261011
SCALER_SAMPLE = 2000
W = int(sys.argv[1]) if len(sys.argv) > 1 else 30
DATA = Path("data")
SRC = DATA / "segments.parquet"
OUT_SEG = DATA / f"segments_w{W}.parquet"
OUT_SIG = DATA / f"feat_w{W}_logsig_d3.npy"
OUT_BASE = DATA / f"feat_w{W}_baseline.npy"
META = ["track_key", "path", "label", "generator", "gen_type", "genre", "reference"]


def mfcc_scaler(seg):
    rng = np.random.default_rng(SEED)
    sample = seg.iloc[rng.choice(len(seg), size=min(SCALER_SAMPLE, len(seg)), replace=False)]
    frames = []
    for f, g in tqdm(sample.groupby("mfcc_file"), desc="scaler sample", unit="file"):
        X = np.load(f)
        frames.append(X[g["row_in_file"].to_numpy()].reshape(-1, X.shape[2]))
    frames = np.vstack(frames)
    return frames.mean(axis=0), frames.std(axis=0) + 1e-8


def process(args):
    row, mu, sd = args
    try:
        y, _ = librosa.load(row["path"], sr=SR, mono=True, duration=MAX_TOTAL_S)
    except Exception as e:
        return [], None, None, f"{row['track_key']}: {type(e).__name__}: {e}"
    seg_len = W * SR
    recs, sigs, bases = [], [], []
    for i in range(len(y) // seg_len):
        seg = y[i * seg_len:(i + 1) * seg_len]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < SILENCE_RMS:
            continue
        seg = seg * (TARGET_RMS / rms)
        m = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP).T
        z = (m - mu) / sd
        t = np.linspace(0.0, 1.0, len(z))[:, None]
        path = np.vstack([np.zeros((1, N_MFCC + 1)), np.hstack([t, z])]).astype(np.float64)
        sigs.append(esig.stream2logsig(path, DEPTH))
        bases.append(np.r_[m.mean(axis=0), m.std(axis=0)])
        rec = {k: row[k] for k in META}
        rec.update({"seg_index": i, "rms_raw": rms})
        recs.append(rec)
    if not recs:
        return [], None, None, None
    return recs, np.asarray(sigs, np.float32), np.asarray(bases, np.float32), None


def main():
    seg10 = pd.read_parquet(SRC)
    mu, sd = mfcc_scaler(seg10)
    tracks = seg10.drop_duplicates("track_key")[META].to_dict("records")
    print(f"Window {W} s: {len(tracks)} tracks; depth-{DEPTH} log-signature length {esig.logsigdim(N_MFCC + 1, DEPTH)}")

    recs, sigs, bases, errors = [], [], [], []
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process, (r, mu, sd)) for r in tracks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{W}s windows ({workers} workers)", unit="track"):
            r, s, b, err = fut.result()
            if err:
                errors.append(err)
            if r:
                recs.extend(r)
                sigs.append(s)
                bases.append(b)

    out = pd.DataFrame(recs)
    out.to_parquet(OUT_SEG, index=False)
    np.save(OUT_SIG, np.vstack(sigs))
    np.save(OUT_BASE, np.vstack(bases))
    pd.set_option("display.width", 200)
    print(f"\nSegments: {len(out)} from {out['track_key'].nunique()} tracks; reference songs: {out['reference'].nunique()}")
    print("\nBy generator (tracks with at least one full window):")
    before = seg10.drop_duplicates("track_key").groupby("generator").size().rename("tracks_before")
    after = out.groupby("generator").agg(segments=("track_key", "size"), tracks=("track_key", "nunique"))
    print(after.join(before).sort_values("segments", ascending=False).to_string())
    if errors:
        print(f"\n{len(errors)} files failed, first few:")
        for e in errors[:5]:
            print(f"  {e}")
    print(f"\nWrote {OUT_SEG}, {OUT_SIG}, {OUT_BASE}")


if __name__ == "__main__":
    main()