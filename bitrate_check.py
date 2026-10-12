"""Control 1: is the detector partly reading MP3 bitrate?
Out-of-fold AI scores (all hi-fi features, XGBoost) vs FMA-listed bitrate within real tracks; bitrate-only separability."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cupy as cp
import librosa
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

SEED = 20261011
DATA = Path("data")
RESULTS = Path("results")
FEATS = [DATA / "feat_hifi_ltas.npy", DATA / "feat_hifi_ltas_resid.npy",
         DATA / "feat_hifi_mfcc20_stats.npy", DATA / "feat_hifi_band_sig.npy"]


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


def file_kbps(path):
    try:
        dur = librosa.get_duration(path=path)
        return Path(path).stat().st_size * 8 / 1000 / dur if dur > 0 else np.nan
    except Exception:
        return np.nan


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(DATA / "segments.parquet").reset_index(drop=True)
    y = seg["label"].to_numpy()
    groups = seg["reference"].to_numpy()
    X = np.hstack([np.load(f).astype(np.float32) for f in tqdm(FEATS, desc="loading features", unit="file")])
    print(f"Features: {X.shape}")

    oof = np.zeros(len(seg))
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups))
    for tr, te in tqdm(folds, desc="out-of-fold scores", unit="fold"):
        oof[te] = xgb_fit_predict(X[tr], y[tr], X[te])
    print(f"Out-of-fold segment EER (sanity, should be near 0.07): {eer(y, oof):.4f}")

    seg["score"] = oof
    trk = seg.groupby("track_key").agg(label=("label", "first"), generator=("generator", "first"),
                                       path=("path", "first"), score=("score", "mean")).reset_index()
    with ThreadPoolExecutor(max_workers=16) as ex:
        trk["file_kbps"] = list(tqdm(ex.map(file_kbps, trk["path"]), total=len(trk), desc="estimating file bitrates", unit="file"))

    meta = pd.read_csv(DATA / "fma_metadata" / "tracks.csv", index_col=0, header=[0, 1], low_memory=False)
    listed = (meta[("track", "bit_rate")] / 1000.0).rename("listed_kbps")
    real = trk[trk["label"] == 0].copy()
    real["track_id"] = real["track_key"].str.split("_").str[1].astype(int)
    real = real.join(listed, on="track_id")
    trk.to_csv(RESULTS / "bitrate_check_tracks.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Real tracks: FMA-listed bitrate (kbps) ===")
    print(real["listed_kbps"].round(0).value_counts().sort_index().to_string())
    rho, pval = spearmanr(real["listed_kbps"], real["score"], nan_policy="omit")
    print(f"\nWithin REAL tracks, Spearman(listed bitrate, AI score) = {rho:.3f} (p = {pval:.3g}, n = {real['listed_kbps'].notna().sum()})")
    real["bucket"] = pd.cut(real["listed_kbps"], [0, 160, 200, 260, 400], labels=["<=160", "161-200", "201-260", ">260"])
    print("\nMean AI score of real tracks by listed bitrate (a strong trend here = encoding leak):")
    print(real.groupby("bucket", observed=False)["score"].agg(["count", "mean", "median"]).round(3).to_string())

    print("\n=== Estimated file bitrate (kbps) by generator ===")
    print(trk.groupby("generator")["file_kbps"].describe()[["count", "25%", "50%", "75%"]].round(0).to_string())
    ok = trk["file_kbps"].notna()
    auc = roc_auc_score(trk.loc[ok, "label"], trk.loc[ok, "file_kbps"])
    print(f"\nBitrate ALONE separating real vs AI (track level): AUC = {max(auc, 1 - auc):.3f}  (0.5 = no information)")
    fake = trk[(trk["label"] == 1) & ok]
    rho_f, p_f = spearmanr(fake["file_kbps"], fake["score"])
    print(f"Within AI tracks, Spearman(file bitrate, AI score) = {rho_f:.3f} (p = {p_f:.3g})")
    print(f"\nWrote {RESULTS / 'bitrate_check_tracks.csv'}")


if __name__ == "__main__":
    main()