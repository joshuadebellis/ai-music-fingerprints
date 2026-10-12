"""Download Echoes.zip from Hugging Face, extract it, and summarize what's inside (real vs AI, providers, formats)."""
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO = "Octavian97/Echoes"
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/echoes")
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
REAL_HINTS = ["bona", "real", "fma", "original", "reference", "genuine"]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = Path(hf_hub_download(REPO, "Echoes.zip", repo_type="dataset", local_dir=RAW_DIR))
    print(f"\nZip at {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        names = [m.filename for m in members if not m.is_dir()]
        print(f"{len(names)} files in zip")
        print("\nFolders inside the zip (first three levels, file counts):")
        for name, n in Counter("/".join(PurePosixPath(x).parts[:3]) for x in names
                               if len(PurePosixPath(x).parts) > 1).most_common(60):
            print(f"  {name}: {n}")
        print("\nFile types in zip:")
        for ext, n in Counter(PurePosixPath(x).suffix.lower() or "(none)" for x in names).most_common():
            print(f"  {ext}: {n}")

        marker = OUT_DIR / ".extracted"
        if marker.exists():
            print(f"\nAlready extracted to {OUT_DIR}, skipping extraction.")
        else:
            for m in tqdm(members, desc="extracting", unit="file"):
                zf.extract(m, OUT_DIR)
            marker.write_text("ok", encoding="utf-8")

    manifests = sorted(OUT_DIR.rglob("*.csv"))
    print(f"\nCSV files found: {len(manifests)}")
    for m in manifests:
        df = pd.read_csv(m)
        print(f"\n=== {m} : {df.shape[0]} rows, columns {list(df.columns)}")
        for col in ["generator", "type", "genre"]:
            if col in df.columns:
                print(f"\n{col}:")
                print(df[col].value_counts().to_string())
        if "original_audio" in df.columns:
            print(f"\nDistinct original_audio (real reference songs): {df['original_audio'].nunique()}")
        if "duration" in df.columns:
            print(f"\nDuration (s): {df['duration'].describe().round(1).to_dict()}")
        print("\nFirst 3 rows:")
        print(df.head(3).to_string())

    audio = [p for p in tqdm(list(OUT_DIR.rglob("*")), desc="scanning extracted files", unit="file")
             if p.suffix.lower() in AUDIO_EXT]
    print(f"\nAudio files extracted: {len(audio)}")
    print("\nAudio by folder (first three levels):")
    for name, n in Counter("/".join(p.relative_to(OUT_DIR).parts[:3]) for p in audio).most_common(60):
        print(f"  {name}: {n}")
    real_like = [p for p in audio if any(h in str(p).lower() for h in REAL_HINTS)]
    print(f"\nAudio files whose path suggests real/bona fide tracks: {len(real_like)}")
    for p in real_like[:10]:
        print(f"  {p}")


if __name__ == "__main__":
    main()