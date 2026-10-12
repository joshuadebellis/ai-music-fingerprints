"""Prepare Echoes (AI) and matched FMA (real) audio: 16 kHz mono, first 18 x 10 s segments per track,
RMS-normalized, 13 MFCCs per segment. Saves one .npy of MFCCs per track plus a segment table."""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

SR = 16000
SEG_S = 10
MAX_SEGS = 18
N_MFCC = 13
N_FFT = 512
HOP = 256
TARGET_RMS = 0.1
SILENCE_RMS = 1e-4

DATA = Path("data")
ECHOES_DIR = DATA / "echoes" / "Echoes"
MANIFEST = ECHOES_DIR / "dataset_manifest.csv"
MATCHES = DATA / "fma_matches.csv"
REAL_DIR = DATA / "real_fma"
MFCC_DIR = DATA / "mfcc"
OUT = DATA / "segments.parquet"
META = ["track_key", "path", "label", "generator", "gen_type", "genre", "reference"]


def track_list():
    fake = pd.read_csv(MANIFEST)
    fake = pd.DataFrame({
        "track_key": "fake_" + fake.index.astype(str),
        "path": [str(ECHOES_DIR / p) for p in fake["path_in_dataset"]],
        "label": 1,
        "generator": fake["generator"],
        "gen_type": fake["type"],
        "genre": fake["genre"],
        "reference": fake["original_audio"],
    })
    real = pd.read_csv(MATCHES)
    real = pd.DataFrame({
        "track_key": "real_" + real["track_id"].astype(int).astype(str),
        "path": [str(REAL_DIR / f"{int(t):06d}.mp3") for t in real["track_id"]],
        "label": 0,
        "generator": "real",
        "gen_type": "real",
        "genre": real["genre"],
        "reference": real["original_audio"],
    })
    return pd.concat([real, fake], ignore_index=True)


def process(row):
    key = row["track_key"]
    try:
        y, _ = librosa.load(row["path"], sr=SR, mono=True, duration=SEG_S * MAX_SEGS)
    except Exception as e:
        return [], f"{key} ({row['path']}): {type(e).__name__}: {e}"
    seg_len = SEG_S * SR
    n = min(len(y) // seg_len, MAX_SEGS)
    mfccs = []
    recs = []
    for i in range(n):
        seg = y[i * seg_len:(i + 1) * seg_len]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < SILENCE_RMS:
            continue
        seg = seg * (TARGET_RMS / rms)
        m = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
        rec = {k: row[k] for k in META}
        rec.update({"seg_index": i, "row_in_file": len(mfccs), "rms_raw": rms,
                    "mfcc_file": str(MFCC_DIR / f"{key}.npy")})
        recs.append(rec)
        mfccs.append(m.T.astype(np.float32))
    if mfccs:
        np.save(MFCC_DIR / f"{key}.npy", np.stack(mfccs))
    return recs, None


def main():
    MFCC_DIR.mkdir(parents=True, exist_ok=True)
    tracks = track_list()
    exists = tracks["path"].map(lambda p: Path(p).exists())
    if (~exists).any():
        print(f"{int((~exists).sum())} listed files not found on disk, for example:")
        print(tracks.loc[~exists, "path"].head(5).to_string(index=False))
    tracks = tracks[exists]
    print(f"Tracks to process: {len(tracks)} ({int((tracks['label'] == 0).sum())} real, "
          f"{int((tracks['label'] == 1).sum())} AI)")

    rows = tracks.to_dict("records")
    workers = max(1, (os.cpu_count() or 2) - 1)
    recs = []
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process, r) for r in rows]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"tracks ({workers} workers)", unit="track"):
            r, err = f.result()
            recs.extend(r)
            if err:
                errors.append(err)

    seg = pd.DataFrame(recs)
    seg.to_parquet(OUT, index=False)
    pd.set_option("display.width", 200)
    print(f"\nSegments: {len(seg)} from {seg['track_key'].nunique()} tracks")
    print("\nSegments and tracks by class:")
    print(seg.groupby("label").agg(segments=("track_key", "size"), tracks=("track_key", "nunique")).to_string())
    print("\nSegments and tracks by generator:")
    print(seg.groupby("generator").agg(segments=("track_key", "size"), tracks=("track_key", "nunique"))
          .sort_values("segments", ascending=False).to_string())
    print(f"\nReference songs represented: {seg['reference'].nunique()}")
    no_segs = len(tracks) - seg["track_key"].nunique()
    print(f"Tracks with no usable segment (shorter than {SEG_S} s, silent, or failed): {no_segs}")
    print(f"\nRaw RMS before normalization, by class (shortcut check later):")
    print(seg.groupby("label")["rms_raw"].describe().round(4).to_string())
    if errors:
        print(f"\n{len(errors)} files failed to load, first few:")
        for e in errors[:10]:
            print(f"  {e}")
    print(f"\nWrote {OUT} and {seg['track_key'].nunique()} MFCC files to {MFCC_DIR}")


if __name__ == "__main__":
    main()