"""Match Echoes' bona fide reference songs to FMA track IDs using FMA's tracks.csv metadata."""
import difflib
import re
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

META_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
DATA = Path("data")
META_ZIP = DATA / "raw" / "fma_metadata.zip"
TRACKS_CSV = DATA / "fma_metadata" / "tracks.csv"
MANIFEST = DATA / "echoes" / "Echoes" / "dataset_manifest.csv"
OUT = DATA / "fma_matches.csv"
ARTIST_SIM = 0.80
TITLE_SIM = 0.85


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"{dest} already downloaded.")
        return
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f, tqdm(
            total=int(r.headers.get("Content-Length", 0)), unit="B", unit_scale=True, desc=dest.name) as bar:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))
    tmp.rename(dest)


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold().replace("&", " and ")
    s = re.sub(r"[\W_]+", " ", s)
    return " ".join(s.split())


def sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def fma_zip_path(track_id):
    tid = f"{int(track_id):06d}"
    return f"fma_full/{tid[:3]}/{tid}.mp3"


def main():
    download(META_URL, META_ZIP)
    if not TRACKS_CSV.exists():
        with zipfile.ZipFile(META_ZIP) as zf:
            zf.extract("fma_metadata/tracks.csv", DATA)
    tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1], low_memory=False)
    title = tracks[("track", "title")].fillna("").map(norm)
    artist = tracks[("artist", "name")].fillna("").map(norm)
    print(f"FMA tracks: {len(tracks)}")

    by_pair = defaultdict(list)
    by_title = defaultdict(list)
    by_artist = defaultdict(list)
    for tid, t, a in zip(tracks.index, title, artist):
        by_pair[(t, a)].append(tid)
        by_title[t].append((tid, a))
        by_artist[a].append((tid, t))

    manifest = pd.read_csv(MANIFEST)
    originals = manifest.groupby("original_audio")["genre"].first()
    rows = []
    for orig, genre in tqdm(originals.items(), total=len(originals), desc="matching", unit="song"):
        left, right = (orig.rsplit(" - ", 1) + [""])[:2] if " - " in orig else (orig, "")
        t_norm, a_norm = norm(left), norm(right)
        ids, how = [], "unmatched"
        if (t_norm, a_norm) in by_pair:
            ids, how = by_pair[(t_norm, a_norm)], "exact"
        elif (a_norm, t_norm) in by_pair:
            ids, how = by_pair[(a_norm, t_norm)], "exact_swapped"
        else:
            cands = [tid for tid, a in by_title.get(t_norm, []) if sim(a, a_norm) >= ARTIST_SIM]
            if cands:
                ids, how = cands, "title_exact_artist_close"
            else:
                pool = by_artist.get(a_norm, [])
                close = difflib.get_close_matches(t_norm, [t for _, t in pool], n=1, cutoff=TITLE_SIM)
                if close:
                    ids, how = [tid for tid, t in pool if t == close[0]], "artist_exact_title_close"
        rec = {"original_audio": orig, "genre": genre, "match_type": how, "n_candidates": len(ids)}
        if ids:
            tid = int(ids[0])
            rec.update({
                "track_id": tid,
                "fma_title": tracks.at[tid, ("track", "title")],
                "fma_artist": tracks.at[tid, ("artist", "name")],
                "license": tracks.at[tid, ("track", "license")],
                "duration_s": tracks.at[tid, ("track", "duration")],
                "zip_path": fma_zip_path(tid),
                "all_candidate_ids": ";".join(str(int(i)) for i in ids),
            })
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(f"\nMatch types for {len(out)} reference songs:")
    print(out["match_type"].value_counts().to_string())
    print(f"\nSongs with more than one candidate track ID: {int((out['n_candidates'] > 1).sum())}")
    matched = out[out["match_type"] != "unmatched"]
    if len(matched):
        print("\nLicenses of matched tracks:")
        print(matched["license"].value_counts().to_string())
        print(f"\nDuration of matched tracks (s): {matched['duration_s'].describe().round(1).to_dict()}")
        print("\nNon-exact matches (check these look right):")
        print(matched[matched["match_type"] != "exact"][["original_audio", "fma_title", "fma_artist", "match_type"]]
              .to_string(index=False))
    print("\nUnmatched:")
    print(out[out["match_type"] == "unmatched"][["original_audio", "genre"]].to_string(index=False)
          if (out["match_type"] == "unmatched").any() else "  none")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()