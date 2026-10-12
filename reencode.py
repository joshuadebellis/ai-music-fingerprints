"""Control 2: re-encode every track (real and AI) through the same MP3 encoder at the same settings:
first 180 s, 44.1 kHz stereo, LAME CBR at the given bitrate (default 128).
Usage: reencode.py [kbps]   -> data/reenc<kbps>/ and data/segments_reenc<kbps>.parquet (128 keeps the original names)."""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import pandas as pd
from tqdm import tqdm

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
KBPS = int(sys.argv[1]) if len(sys.argv) > 1 else 128
DATA = Path("data")
SUFFIX = "" if KBPS == 128 else str(KBPS)
OUT_DIR = DATA / f"reenc{SUFFIX}"
SEGMENTS = DATA / "segments.parquet"
OUT_SEG = DATA / f"segments_reenc{SUFFIX}.parquet"


def encode(args):
    src, dst = args
    if Path(dst).exists():
        return None
    tmp = str(dst) + ".part.mp3"
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", src, "-t", "180", "-ac", "2", "-ar", "44100",
           "-codec:a", "libmp3lame", "-b:a", f"{KBPS}k", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return f"{src}: {r.stderr.strip()[:200]}"
    Path(tmp).replace(dst)
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(SEGMENTS)
    tracks = seg.drop_duplicates("track_key")[["track_key", "path"]]
    jobs = [(p, str(OUT_DIR / f"{k}.mp3")) for k, p in zip(tracks["track_key"], tracks["path"])]
    print(f"ffmpeg: {FFMPEG}\nRe-encoding {len(jobs)} tracks to {KBPS} kbps CBR MP3, 44.1 kHz stereo -> {OUT_DIR}")
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        errors = [e for e in tqdm(ex.map(encode, jobs), total=len(jobs), desc=f"re-encoding ({workers} workers)", unit="track") if e]
    seg["path"] = [str(OUT_DIR / f"{k}.mp3") for k in seg["track_key"]]
    seg.to_parquet(OUT_SEG, index=False)
    print(f"\nWrote {OUT_SEG}; {len(errors)} failures")
    for e in errors[:10]:
        print(f"  {e}")


if __name__ == "__main__":
    main()