"""Leave-one-generator-out: train without generator g, test real vs g (unseen). Same 5 splits by reference song.
Checkpointed; compares with the Echoes paper's unseen-provider EERs."""
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

SEED = 20261011
N_SPLITS = 5
DATA = Path("data")
RESULTS = Path("results")
OUT = RESULTS / "lopo_reenc.csv"
SPECS = {
    "all": [DATA / "feat_reenc_ltas.npy", DATA / "feat_reenc_ltas_resid.npy",
            DATA / "feat_reenc_mfcc20_stats.npy", DATA / "feat_reenc_band_sig.npy"],
    "ltas_resid": [DATA / "feat_reenc_ltas_resid.npy"],
}
PAPER_UNSEEN = {"diffrhythm": 0.1138, "musicgen": 0.2281, "songgen": 0.2063, "acestep": 0.1944, "suno": 0.1645,
                "producer": 0.1857, "brev": 0.1705, "udio": 0.2488, "elevenlabs": 0.2508, "audioldm": 0.2748,
                "stableaudio": 0.3109, "mubert": 0.3316}


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def xgb_fit_predict(Xtr, ytr, Xte):
    model = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                          tree_method="hist", device="cuda", eval_metric="logloss", random_state=SEED, n_jobs=-1)
    w = compute_sample_weight("balanced", ytr).astype(np.float32)
    Xtr_d = cp.asarray(Xtr, dtype=cp.float32)
    model.fit(Xtr_d, cp.asarray(ytr, dtype=cp.float32), sample_weight=cp.asarray(w))
    del Xtr_d
    Xte_d = cp.asarray(Xte, dtype=cp.float32)
    p = model.predict_proba(Xte_d)[:, 1]
    p = cp.asnumpy(p) if isinstance(p, cp.ndarray) else np.asarray(p)
    del Xte_d, model
    cp.get_default_memory_pool().free_all_blocks()
    return p


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(DATA / "segments.parquet").reset_index(drop=True)
    y = seg["label"].to_numpy()
    gens = seg["generator"].to_numpy()
    groups = seg["reference"].to_numpy()
    fake_gens = sorted(g for g in np.unique(gens) if g != "real")
    splits = list(GroupShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=SEED).split(seg, y, groups))

    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["features"], r["generator"], r["split"]) for r in rows}
    for name, files in SPECS.items():
        todo = [(g, k) for g in fake_gens for k in range(N_SPLITS) if (name, g, k) not in done]
        if not todo:
            continue
        X = np.hstack([np.load(f).astype(np.float32) for f in files])
        for g, k in tqdm(todo, desc=f"leave-one-generator-out ({name})", unit="fit"):
            tr, te = splits[k]
            trn = tr[gens[tr] != g]
            tst = te[(gens[te] == "real") | (gens[te] == g)]
            p = xgb_fit_predict(X[trn], y[trn], X[tst])
            rows.append({"features": name, "generator": g, "split": k, "eer_unseen": eer(y[tst], p)})
            pd.DataFrame(rows).to_csv(OUT, index=False)
        del X

    res = pd.DataFrame(rows)
    table = res.pivot_table(index="generator", columns="features", values="eer_unseen", aggfunc="mean")
    table.columns = [f"unseen_{c}" for c in table.columns]
    seen_path = RESULTS / "reenc_per_generator.csv"
    if seen_path.exists():
        seen = pd.read_csv(seen_path)
        seen = seen[seen["features"].isin(["all", "ltas_resid"])].pivot_table(
            index="generator", columns="features", values="eer_segment", aggfunc="mean")
        seen.columns = [f"seen_{c}" for c in seen.columns]
        table = table.join(seen)
    table["paper_unseen"] = pd.Series(PAPER_UNSEEN)
    table = table.sort_values("unseen_all")
    pd.set_option("display.width", 200)
    print("\n=== Leave-one-generator-out segment EER (lower is better) ===")
    print(table.round(3).to_string())
    print("\nMeans:")
    print(table.mean().round(4).to_string())
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()