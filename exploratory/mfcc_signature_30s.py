"""MFCC path signature, one 30 s window per track reproduced on Echoes + FMA: first 30 s at 16 kHz (zero-padded if shorter, or dropped with 'nopad'),
13 librosa MFCCs (default frames), path = time + raw MFCC, full signature depth 3. One sample per track.
Usage: mfcc_signature_30s.py <pad|nopad|pad_norm|nopad_norm>   ('_norm' = RMS-normalize the audio before MFCC)"""
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
DURATION = 30
N_MFCC = 13
DEPTH = 3
TARGET_RMS = 0.1
ARM = sys.argv[1] if len(sys.argv) > 1 else "pad"
NORM = ARM.endswith("_norm")
PAD_MODE = ARM.replace("_norm", "")
DATA = Path("data")
SRC = DATA / "segments.parquet"
OUT_TABLE = DATA / f"mfcc30_{ARM}.parquet"
OUT_FEAT = DATA / f"feat_mfcc30_{ARM}.npy"
META = ["track_key", "path", "label", "generator", "gen_type", "genre", "reference"]


def process(row):
    try:
        data, _ = librosa.load(row["path"], sr=SR, mono=True, duration=DURATION)
    except Exception as e:
        return None, None, f"{row['track_key']}: {type(e).__name__}: {e}"
    target = SR * DURATION
    raw_len = len(data)
    rms = float(np.sqrt(np.mean(data ** 2))) if raw_len else 0.0
    if raw_len < target:
        if PAD_MODE == "nopad":
            return None, None, None
        data = np.pad(data, (0, target - raw_len))
    if NORM and rms > 0:
        data = data * (TARGET_RMS / rms)
    mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=N_MFCC).T
    t = np.linspace(0.0, 1.0, len(mfcc))[:, None]
    path = np.hstack([t, mfcc]).astype(np.float64)
    sig = esig.stream2sig(path, DEPTH)
    rec = {k: row[k] for k in META}
    rec.update({"raw_seconds": raw_len / SR, "padded_seconds": max(0, target - raw_len) / SR, "rms_raw": rms})
    return rec, np.asarray(sig, np.float32), None


def main():
    tracks = pd.read_parquet(SRC).drop_duplicates("track_key")[META].to_dict("records")
    print(f"Arm '{ARM}' (pad mode {PAD_MODE}, loudness-normalized: {NORM}): {len(tracks)} tracks; "
          f"signature length {esig.sigdim(N_MFCC + 1, DEPTH)}")
    recs, feats, errors = [], [], []
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process, r) for r in tracks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"MFCC path signature, one 30 s window per track ({ARM}, {workers} workers)", unit="track"):
            rec, f, err = fut.result()
            if err:
                errors.append(err)
            if rec is not None:
                recs.append(rec)
                feats.append(f)

    table = pd.DataFrame(recs)
    table.to_parquet(OUT_TABLE, index=False)
    np.save(OUT_FEAT, np.vstack(feats))
    pd.set_option("display.width", 200)
    print(f"\nKept {len(table)} tracks; reference songs: {table['reference'].nunique()}")
    print("\nPadding and loudness by generator:")
    print(table.groupby("generator").agg(tracks=("track_key", "size"),
                                         padded_tracks=("padded_seconds", lambda s: int((s > 0).sum())),
                                         mean_padded_s=("padded_seconds", "mean"),
                                         mean_rms=("rms_raw", "mean")).round(3).to_string())
    if errors:
        print(f"\n{len(errors)} files failed, first few:")
        for e in errors[:5]:
            print(f"  {e}")
    print(f"\nWrote {OUT_TABLE} and {OUT_FEAT}")


if __name__ == "__main__":
    main()
