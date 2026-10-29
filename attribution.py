"""Generator attribution: which of the 12 generators made an AI segment (real segments excluded).
Same 5 splits by reference song as evaluate_features.py (built on the full table, then restricted to AI rows).
XGBoost multiclass, same hyperparameters as the detector. Compares with Echoes Table (frozen Wav2Vec2 XLS-R + LR,
track-level split): accuracy 78.6%, macro-F1 0.82.
Usage: attribution.py <cuda|cpu> <feat_prefix> <tag> [baseline_tag]
  feat_prefix: e.g. feat_reenc (uses data/<prefix>_ltas.npy etc.)
  tag: results/<tag>_preds.csv (checkpointed; reruns skip finished fits)
  baseline_tag: optional; prints the pre-specified shift-guard check against that run.
Pre-specified rule (primary = 'all' features): passes only if accuracy > 78.6% and macro-F1 > 0.82,
and under the time-shift re-encode both drop by less than 2 points (0.02 macro-F1)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

N_SPLITS = 5
TEST_SIZE = 0.2
SEED = 20261011
DATA = Path("data")
RESULTS = Path("results")
PAPER_ACC, PAPER_F1 = 0.786, 0.82
MAX_DROP = 0.02

DEVICE = sys.argv[1]
PREFIX = sys.argv[2]
TAG = sys.argv[3]
BASELINE = sys.argv[4] if len(sys.argv) > 4 else None
BLOCKS = ["ltas", "ltas_resid", "mfcc20_stats", "band_sig"]
SPECS = {"ltas_resid": ["ltas_resid"], "all": BLOCKS}
PREDS_CSV = RESULTS / f"{TAG}_preds.csv"

if DEVICE == "cuda":
    import cupy as cp


def fit_predict(Xtr, ytr, Xte, n_classes):
    model = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                          tree_method="hist", device=DEVICE, objective="multi:softprob", num_class=n_classes,
                          eval_metric="mlogloss", random_state=SEED, n_jobs=-1)
    w = compute_sample_weight("balanced", ytr).astype(np.float32)
    if DEVICE == "cuda":
        Xtr_d = cp.asarray(Xtr, dtype=cp.float32)
        model.fit(Xtr_d, cp.asarray(ytr, dtype=cp.int32), sample_weight=cp.asarray(w))
        del Xtr_d
        Xte_d = cp.asarray(Xte, dtype=cp.float32)
        p = model.predict_proba(Xte_d)
        p = cp.asnumpy(p) if isinstance(p, cp.ndarray) else np.asarray(p)
        del Xte_d, model
        cp.get_default_memory_pool().free_all_blocks()
        return p
    model.fit(Xtr, ytr, sample_weight=w)
    return model.predict_proba(Xte)


def summarize(preds, classes):
    rows, track_rows = [], []
    for (name, k), d in preds.groupby(["features", "split"]):
        rows.append({"features": name, "split": k, "acc": accuracy_score(d["y"], d["pred"]),
                     "macro_f1": f1_score(d["y"], d["pred"], average="macro")})
        pcols = [f"p{i}" for i in range(len(classes))]
        t = d.groupby("track")[["y"] + pcols].mean()
        tp = t[pcols].to_numpy().argmax(axis=1)
        track_rows.append({"features": name, "split": k, "acc_track": accuracy_score(t["y"].astype(int), tp),
                           "macro_f1_track": f1_score(t["y"].astype(int), tp, average="macro")})
    ov = pd.DataFrame(rows).merge(pd.DataFrame(track_rows), on=["features", "split"])
    return ov.groupby("features").agg(acc=("acc", "mean"), acc_std=("acc", "std"), macro_f1=("macro_f1", "mean"),
                                      macro_f1_std=("macro_f1", "std"), acc_track=("acc_track", "mean"),
                                      macro_f1_track=("macro_f1_track", "mean"))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(DATA / "segments.parquet").reset_index(drop=True)
    splits = list(GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)
                  .split(seg, seg["label"].to_numpy(), seg["reference"].to_numpy()))
    gens = seg["generator"].to_numpy()
    classes = sorted(g for g in np.unique(gens) if g != "real")
    cidx = {g: i for i, g in enumerate(classes)}
    ai = gens != "real"
    y = np.array([cidx.get(g, -1) for g in gens])
    track = seg["track_key"].to_numpy()
    majority = pd.Series(gens[ai]).value_counts(normalize=True).iloc[0]
    print(f"{ai.sum()} AI segments, {len(classes)} generators; majority-class chance {majority:.3f}")

    preds = pd.read_csv(PREDS_CSV) if PREDS_CSV.exists() else pd.DataFrame()
    done = set(zip(preds["features"], preds["split"])) if len(preds) else set()
    jobs = [(name, k) for name in SPECS for k in range(N_SPLITS) if (name, k) not in done]
    print(f"Features {PREFIX}; {len(jobs)} fits to run -> {PREDS_CSV}")

    current, X = None, None
    for name, k in tqdm(jobs, desc="feature set x split", unit="fit"):
        if name != current:
            X = None
            X = np.hstack([np.load(DATA / f"{PREFIX}_{b}.npy").astype(np.float32) for b in SPECS[name]])
            current = name
        tr, te = splits[k]
        tr, te = tr[ai[tr]], te[ai[te]]
        p = fit_predict(X[tr], y[tr], X[te], len(classes))
        d = pd.DataFrame({"features": name, "split": k, "row": te, "track": track[te], "y": y[te],
                          "pred": p.argmax(axis=1)})
        d = pd.concat([d, pd.DataFrame(p.astype(np.float32), columns=[f"p{i}" for i in range(len(classes))])], axis=1)
        preds = pd.concat([preds, d], ignore_index=True)
        preds.to_csv(PREDS_CSV, index=False, encoding="utf-8")

    pd.set_option("display.width", 250)
    summ = summarize(preds, classes)
    print(f"\n=== Attribution, 12 generators, AI segments only, 5 splits by reference song ({PREFIX}) ===")
    print(f"Echoes paper (Wav2Vec2 XLS-R 2B + LR, track-level split): acc {PAPER_ACC:.3f}, macro-F1 {PAPER_F1:.2f}")
    print(summ.round(4).to_string())

    recall = {}
    for name, d in preds.groupby("features"):
        cm = confusion_matrix(d["y"], d["pred"], labels=range(len(classes)))
        recall[name] = cm.diagonal() / cm.sum(axis=1)
        cmn = pd.DataFrame(cm / cm.sum(axis=1, keepdims=True), index=classes, columns=classes)
        cmn.to_csv(RESULTS / f"{TAG}_confusion_{name}.csv", encoding="utf-8")
        print(f"\n=== Confusion matrix ({name}), rows = true generator, row-normalized, pooled over splits ===")
        print((cmn * 100).round(1).to_string())
    rec = pd.DataFrame(recall, index=classes)
    rec.to_csv(RESULTS / f"{TAG}_recall.csv", encoding="utf-8")
    summ.to_csv(RESULTS / f"{TAG}_overall.csv", encoding="utf-8")
    print("\n=== Per-generator recall (segment) ===")
    print(rec.round(3).sort_values("all" if "all" in rec else rec.columns[0]).to_string())

    if "all" in summ.index:
        a = summ.loc["all"]
        beats = a["acc"] > PAPER_ACC and a["macro_f1"] > PAPER_F1
        print(f"\nRule part 1 (all features beat Echoes on acc AND macro-F1): {'PASS' if beats else 'FAIL'}")
        if BASELINE:
            base = summarize(pd.read_csv(RESULTS / f"{BASELINE}_preds.csv"), classes).loc["all"]
            da, df = base["acc"] - a["acc"], base["macro_f1"] - a["macro_f1"]
            ok = da < MAX_DROP and df < MAX_DROP
            print(f"Rule part 2 (vs {BASELINE}): acc drop {da:+.4f}, macro-F1 drop {df:+.4f} -> {'PASS' if ok else 'FAIL'}")
    print(f"\nWrote {PREDS_CSV}, {TAG}_overall.csv, {TAG}_recall.csv, {TAG}_confusion_*.csv in {RESULTS}")


if __name__ == "__main__":
    main()
