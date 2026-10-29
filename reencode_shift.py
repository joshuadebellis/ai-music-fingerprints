"""Encoding-history guard: like reencode.py at 128 kbps, but each track is first shifted by a random sub-frame
number of samples, so the new MP3 frame/granule grid cannot line up with any earlier encoding's grid.
Shift N (at 44.1 kHz) has N mod 576 in [144, 432], i.e. a quarter to three quarters of an MP3 granule off.
Fixed per track (seeded by track_key), so reruns are reproducible.
-> data/reenc_shift/ and data/segments_reenc_shift.parquet; shifts logged to data/reenc_shift_offsets.csv."""
import os
import subprocess
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd
from tqdm import tqdm

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
KBPS = 128
DATA = Path("data")
OUT_DIR = DATA / "reenc_shift"
SEGMENTS = DATA / "segments.parquet"
OUT_SEG = DATA / "segments_reenc_shift.parquet"
OFFSETS = DATA / "reenc_shift_offsets.csv"


def shift_for(key):
    rng = np.random.default_rng(zlib.crc32(key.encode("utf-8")))
    return int(rng.integers(144, 433) + 576 * rng.integers(0, 2))


def encode(args):
    src, dst, n = args
    if Path(dst).exists():
        return None
    tmp = str(dst) + ".part.mp3"
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", src,
           "-af", f"aresample=44100,atrim=start_sample={n},asetpts=PTS-STARTPTS",
           "-t", "180", "-ac", "2", "-ar", "44100", "-codec:a", "libmp3lame", "-b:a", f"{KBPS}k", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return f"{src}: {r.stderr.strip()[:200]}"
    Path(tmp).replace(dst)
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(SEGMENTS)
    tracks = seg.drop_duplicates("track_key")[["track_key", "path"]].copy()
    tracks["shift"] = [shift_for(k) for k in tracks["track_key"]]
    tracks[["track_key", "shift"]].to_csv(OFFSETS, index=False, encoding="utf-8")
    jobs = [(p, str(OUT_DIR / f"{k}.mp3"), n) for k, p, n in zip(tracks["track_key"], tracks["path"], tracks["shift"])]
    print(f"ffmpeg: {FFMPEG}\nShift + re-encode {len(jobs)} tracks to {KBPS} kbps CBR, 44.1 kHz stereo -> {OUT_DIR}")
    print(f"Shift samples: min {tracks['shift'].min()}, max {tracks['shift'].max()}, mean {tracks['shift'].mean():.0f}")
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        errors = [e for e in tqdm(ex.map(encode, jobs), total=len(jobs), desc=f"shift+re-encode ({workers} workers)", unit="track") if e]
    seg["path"] = [str(OUT_DIR / f"{k}.mp3") for k in seg["track_key"]]
    seg.to_parquet(OUT_SEG, index=False)
    print(f"\nWrote {OUT_SEG} and {OFFSETS}; {len(errors)} failures")
    for e in errors[:10]:
        print(f"  {e}")


if __name__ == "__main__":
    main()
