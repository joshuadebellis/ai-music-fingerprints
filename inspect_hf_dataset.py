"""List what's in a Hugging Face dataset repo and print its README, without downloading audio."""
import sys
from collections import Counter
from pathlib import PurePosixPath

from huggingface_hub import HfApi, hf_hub_download

REPO = sys.argv[1] if len(sys.argv) > 1 else "Octavian97/Echoes"

api = HfApi()
info = api.dataset_info(REPO, files_metadata=True)
files = info.siblings
total_gb = sum((f.size or 0) for f in files) / 1e9
print(f"{REPO}: {len(files)} files, about {total_gb:.2f} GB in total")

print("\nTop-level entries:")
for name, n in Counter(f.rfilename.split("/")[0] for f in files).most_common():
    print(f"  {name}: {n} files")

print("\nSecond-level folders:")
second = Counter("/".join(f.rfilename.split("/")[:2]) for f in files if f.rfilename.count("/") >= 1)
for name, n in second.most_common(40):
    print(f"  {name}: {n} files")

print("\nFile types:")
for ext, n in Counter(PurePosixPath(f.rfilename).suffix.lower() or "(none)" for f in files).most_common():
    print(f"  {ext}: {n}")

print("\nFirst 30 files:")
for f in files[:30]:
    print(f"  {f.rfilename}  ({(f.size or 0) / 1e6:.2f} MB)")

try:
    readme = hf_hub_download(REPO, "README.md", repo_type="dataset")
    with open(readme, encoding="utf-8") as fh:
        text = fh.read()
    print("\n===== README (first 4000 characters) =====")
    print(text[:4000])
except Exception as e:
    print(f"\nCould not fetch README: {e}")