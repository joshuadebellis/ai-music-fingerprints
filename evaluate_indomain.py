"""In-domain AI-vs-real detection on Echoes (AI) + matched FMA (real): 5 random 80:20 splits grouped by
reference song; XGBoost (GPU) and RF (CPU) on baseline MFCC stats and on depth-4 log-signatures.
Checkpoints every fit, so a rerun resumes where it stopped."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

N_SPLITS = 5
TEST_SIZE = 0.2
SEED = 20261011
XGB_DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
DATA = Path("data")
RESULTS = Path("results")
FEATURES = {"baseline": DATA / "feat_baseline.npy", "logsig": DATA / "feat_logsig.npy"}
OVERALL_CSV = RESULTS / "indomain_overall.csv"
PER_GEN_CSV = RESULTS / "indomain_per_generator.csv"

if XGB_DEVICE == "cuda":
    import cupy as cp


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def make_model(name):
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, n_jobs=-1,
                                      random_state=SEED, class_weight="balanced_subsample")
    return XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                         tree_method="hist", device=XGB_DEVICE, eval_metric="logloss", random_state=SEED, n_jobs=-1)


def fit_predict(model_name, Xtr, ytr, Xte):
    model = make_model(model_name)
    if model_name == "rf":
        model.fit(Xtr, ytr)
        return model.predict_proba(Xte)[:, 1]
    w = compute_sample_weight("balanced", ytr).astype(np.float32)
    if XGB_DEVICE == "cuda":
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


def load_done():
    if OVERALL_CSV.exists():
        ov = pd.read_csv(OVERALL_CSV)
        pg = pd.read_csv(PER_GEN_CSV) if PER_GEN_CSV.exists() else pd.DataFrame()
        done = set(zip(ov["features"], ov["model"], ov["split"]))
        return ov.to_dict("records"), pg.to_dict("records"), done
    return [], [], set()


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(DATA / "segments.parquet").reset_index(drop=True)
    y = seg["label"].to_numpy()
    groups = seg["reference"].to_numpy()
    gens = seg["generator"].to_numpy()
    track = seg["track_key"].to_numpy()
    fake_gens = sorted(g for g in seg["generator"].unique() if g != "real")
    splits = list(GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED).split(seg, y, groups))

    overall, per_gen, done = load_done()
    jobs = [(f, m, k) for f in FEATURES for m in ["xgb", "rf"] for k in range(N_SPLITS)]
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs) - len(todo)} fits already done, {len(todo)} to run")
    loaded = {}
    for feat, model_name, k in tqdm(todo, desc="feature x model x split", unit="fit"):
        if feat not in loaded:
            loaded = {feat: np.load(FEATURES[feat], mmap_mode="r")}
        X = loaded[feat]
        tr, te = splits[k]
        p = fit_predict(model_name, np.asarray(X[tr], dtype=np.float32), y[tr], np.asarray(X[te], dtype=np.float32))
        yt = y[te]
        tdf = pd.DataFrame({"track": track[te], "y": yt, "p": p}).groupby("track").agg(y=("y", "first"), p=("p", "mean"))
        overall.append({
            "features": feat, "model": model_name, "split": k,
            "n_test_real_segs": int((yt == 0).sum()), "n_test_ai_segs": int((yt == 1).sum()),
            "eer_segment": eer(yt, p), "auc_segment": float(roc_auc_score(yt, p)),
            "eer_track": eer(tdf["y"].to_numpy(), tdf["p"].to_numpy()),
        })
        for g in fake_gens:
            m = (gens[te] == "real") | (gens[te] == g)
            if (yt[m] == 1).sum() and (yt[m] == 0).sum():
                per_gen.append({"features": feat, "model": model_name, "split": k, "generator": g,
                                "eer_segment": eer(yt[m], p[m])})
        pd.DataFrame(overall).to_csv(OVERALL_CSV, index=False)
        pd.DataFrame(per_gen).to_csv(PER_GEN_CSV, index=False)

    ov = pd.DataFrame(overall)
    pg = pd.DataFrame(per_gen)
    pd.set_option("display.width", 250)
    summary = ov.groupby(["features", "model"]).agg(
        eer_segment_mean=("eer_segment", "mean"), eer_segment_std=("eer_segment", "std"),
        auc_segment_mean=("auc_segment", "mean"),
        eer_track_mean=("eer_track", "mean"), eer_track_std=("eer_track", "std"),
        n_splits=("split", "count"),
    )
    print("\n=== In-domain, 5 splits by reference song (EER as a fraction; paper's Wav2Vec2 baseline: 0.153) ===")
    print(summary.round(4).to_string())
    print("\n=== Per-generator segment EER, mean over splits (each generator vs. real) ===")
    pg["config"] = pg["features"] + "_" + pg["model"]
    print(pg.pivot_table(index="generator", columns="config", values="eer_segment", aggfunc="mean")
          .round(3).sort_values(pg["config"].iloc[-1]).to_string())
    print(f"\nWrote {OVERALL_CSV} and {PER_GEN_CSV}")


if __name__ == "__main__":
    main()