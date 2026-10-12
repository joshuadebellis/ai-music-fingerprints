"""Fetch only the matched FMA tracks out of fma_full.zip (879 GB) using HTTP range requests, without the full download."""
import shutil
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip
from tqdm import tqdm

URL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
MATCHES = Path("data/fma_matches.csv")
OUT_DIR = Path("data/real_fma")


def main():
    m = pd.read_csv(MATCHES)
    m = m[m["match_type"] != "unmatched"]
    dup = m[m.duplicated("track_id", keep=False)].sort_values("track_id")
    if len(dup):
        print(f"{dup['track_id'].nunique()} FMA track IDs are shared by more than one Echoes reference name:")
        print(dup[["track_id", "original_audio"]].to_string(index=False))
    tracks = m.drop_duplicates("track_id")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.part"):
        stale.unlink()
    todo = [(int(r.track_id), r.zip_path) for r in tracks.itertuples()
            if not (OUT_DIR / f"{int(r.track_id):06d}.mp3").exists()]
    print(f"\n{len(tracks)} unique matched tracks, {len(todo)} still to fetch")
    if not todo:
        return

    print("Reading the zip index over HTTP (only the index, not the archive)...")
    with RemoteZip(URL) as rz:
        names = set(rz.namelist())
        print(f"Zip index read: {len(names)} entries")
        missing = []
        for tid, zp in tqdm(todo, desc="fetching tracks", unit="track"):
            if zp not in names:
                missing.append(zp)
                continue
            dest = OUT_DIR / f"{tid:06d}.mp3"
            if dest.exists():
                continue
            tmp = dest.with_suffix(".part")
            with rz.open(zp) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            tmp.replace(dest)

    files = list(OUT_DIR.glob("*.mp3"))
    print(f"\nHave {len(files)} real tracks in {OUT_DIR}, {sum(f.stat().st_size for f in files) / 1e9:.2f} GB")
    if missing:
        print(f"{len(missing)} paths not found in the zip, for example: {missing[:5]}")
        print("Sample of actual zip entries:", sorted(names)[:5])


if __name__ == "__main__":
    main()