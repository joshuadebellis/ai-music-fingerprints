"""In-domain AI-vs-real detection on any feature files: 5 random 80:20 splits grouped by reference song.
Usage: evaluate_features.py <cuda|cpu> <tag> <xgb|lr> name=spec [name=spec ...]
spec: path[:ncols][+path[:ncols]...]   ('+' concatenates column-wise; ':N' keeps the first N columns)."""
import re
import sys
from pathlib import Path
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

N_SPLITS = 5
TEST_SIZE = 0.2
SEED = 20261011
DATA = Path("data")
RESULTS = Path("results")

DEVICE = sys.argv[1]
TAG = sys.argv[2]
MODEL = sys.argv[3]
SPECS = dict(a.split("=", 1) for a in sys.argv[4:])
OVERALL_CSV = RESULTS / f"{TAG}_overall.csv"
PER_GEN_CSV = RESULTS / f"{TAG}_per_generator.csv"

if DEVICE == "cuda" and MODEL == "xgb":
    import cupy as cp


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def parse_spec(spec):
    parts = []
    for piece in spec.split("+"):
        m = re.fullmatch(r"(.+?):(\d+)", piece)
        path, ncols = (m.group(1), int(m.group(2))) if m else (piece, None)
        parts.append((np.load(path, mmap_mode="r"), ncols))
    return parts


def take(parts, idx):
    cols = []
    for arr, ncols in parts:
        block = arr[idx] if ncols is None else arr[idx, :ncols]
        cols.append(np.asarray(block, dtype=np.float32))
    return np.hstack(cols)


def fit_predict(Xtr, ytr, Xte):
    if MODEL == "rf":
        model = RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=-1,
                                       random_state=SEED, class_weight="balanced_subsample")
        model.fit(Xtr, ytr)
        return model.predict_proba(Xte)[:, 1]
    if MODEL.startswith("lr"):
        c = float(MODEL[2:]) if len(MODEL) > 2 else 1.0
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000, class_weight="balanced"))
        model.fit(Xtr, ytr)
        return model.predict_proba(Xte)[:, 1]
    model = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                          tree_method="hist", device=DEVICE, eval_metric="logloss", random_state=SEED, n_jobs=-1)
    w = compute_sample_weight("balanced", ytr).astype(np.float32)
    if DEVICE == "cuda":
        Xtr_d = cp.asarray(Xtr, dtype=cp.float32)
        model.fit(Xtr_d, cp.asarray(ytr, dtype=cp.float32), sample_weight=cp.asarray(w))
        del Xtr_d
        Xte_d = cp.asarray(Xte, dtype=cp.float32)
        p = model.predict_proba(Xte_d)[:, 1]
        p = cp.asnumpy(p) if isinstance(p, cp.ndarray) else np.asarray(p)
        del Xte_d, model
        cp.get_default_memory_pool().free_all_blocks()
        return p
    model.fit(Xtr, ytr, sample_weight=w)
    return model.predict_proba(Xte)[:, 1]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(os.environ.get("SEGMENTS_FILE", str(DATA / "segments.parquet"))).reset_index(drop=True)
    y = seg["label"].to_numpy()
    groups = seg["reference"].to_numpy()
    gens = seg["generator"].to_numpy()
    track = seg["track_key"].to_numpy()
    fake_gens = sorted(g for g in seg["generator"].unique() if g != "real")
    splits = list(GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED).split(seg, y, groups))

    if OVERALL_CSV.exists():
        overall = pd.read_csv(OVERALL_CSV).to_dict("records")
        per_gen = pd.read_csv(PER_GEN_CSV).to_dict("records") if PER_GEN_CSV.exists() else []
    else:
        overall, per_gen = [], []
    done = {(r["features"], r["split"]) for r in overall}
    jobs = [(name, k) for name in SPECS for k in range(N_SPLITS) if (name, k) not in done]
    print(f"Model {MODEL}; feature sets {list(SPECS)}; {len(jobs)} fits to run")

    current, parts = None, None
    for name, k in tqdm(jobs, desc="feature set x split", unit="fit"):
        if name != current:
            parts = parse_spec(SPECS[name])
            current = name
        tr, te = splits[k]
        Xtr, Xte = take(parts, tr), take(parts, te)
        p = fit_predict(Xtr, y[tr], Xte)
        yt = y[te]
        tdf = pd.DataFrame({"track": track[te], "y": yt, "p": p}).groupby("track").agg(y=("y", "first"), p=("p", "mean"))
        overall.append({"features": name, "model": MODEL, "split": k, "n_features": Xtr.shape[1],
                        "eer_segment": eer(yt, p), "auc_segment": float(roc_auc_score(yt, p)),
                        "eer_track": eer(tdf["y"].to_numpy(), tdf["p"].to_numpy())})
        for g in fake_gens:
            m = (gens[te] == "real") | (gens[te] == g)
            if (yt[m] == 1).sum() and (yt[m] == 0).sum():
                per_gen.append({"features": name, "model": MODEL, "split": k, "generator": g,
                                "eer_segment": eer(yt[m], p[m])})
        pd.DataFrame(overall).to_csv(OVERALL_CSV, index=False)
        pd.DataFrame(per_gen).to_csv(PER_GEN_CSV, index=False)

    ov = pd.DataFrame(overall)
    pg = pd.DataFrame(per_gen)
    pd.set_option("display.width", 250)
    print(f"\n=== In-domain {MODEL}, 5 splits by reference song (EER as a fraction; paper's Wav2Vec2: 0.153) ===")
    print(ov.groupby("features").agg(n_features=("n_features", "first"),
                                     eer_segment_mean=("eer_segment", "mean"), eer_segment_std=("eer_segment", "std"),
                                     auc_segment_mean=("auc_segment", "mean"),
                                     eer_track_mean=("eer_track", "mean"), eer_track_std=("eer_track", "std"))
          .sort_values("eer_segment_mean").round(4).to_string())
    print("\n=== Per-generator segment EER, mean over splits ===")
    print(pg.pivot_table(index="generator", columns="features", values="eer_segment", aggfunc="mean").round(3).to_string())
    print(f"\nWrote {OVERALL_CSV} and {PER_GEN_CSV}")


if __name__ == "__main__":
    main()