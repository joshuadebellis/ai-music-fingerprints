"""Figures for the attribution section, from the CSVs that attribution.py writes.
Usage: fig_attribution.py [results_dir]   -> figures/attr_*.png"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
FIG = Path("figures")
PAPER_ACC, PAPER_F1 = 0.786, 0.82
NAMES = {"acestep": "AceStep", "audioldm": "AudioLDM", "brev": "Brev", "diffrhythm": "DiffRhythm",
         "elevenlabs": "ElevenLabs", "mubert": "Mubert", "musicgen": "MusicGen", "producer": "Producer",
         "songgen": "SongGen", "stableaudio": "Stable Audio", "suno": "Suno", "udio": "Udio"}
ORDER = ["musicgen", "audioldm", "songgen", "acestep", "diffrhythm", "udio", "producer", "elevenlabs",
         "stableaudio", "mubert", "brev", "suno"]
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#999999"


def confusion():
    cm = pd.read_csv(RESULTS / "attr_reenc_confusion_all.csv", index_col=0).loc[ORDER, ORDER] * 100
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    im = ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=100)
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            v = cm.iat[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 50 else "#222222")
    labels = [NAMES[g] for g in ORDER]
    ax.set_xticks(range(len(ORDER)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(ORDER)), labels)
    ax.set_xlabel("Predicted generator")
    ax.set_ylabel("True generator")
    ax.set_title("Which generator made this clip? Row-normalized %, 5 splits pooled", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.75, label="% of the true generator's segments")
    fig.savefig(FIG / "attr_confusion.png", dpi=200)


def headline():
    ov = pd.read_csv(RESULTS / "attr_reenc_overall.csv", index_col=0)
    sh = pd.read_csv(RESULTS / "attr_reenc_shift_overall.csv", index_col=0)
    groups = ["Accuracy", "Macro-F1"]
    bars = [("Echoes: Wav2Vec2 XLS-R 2B + LR", [PAPER_ACC, PAPER_F1], GREY),
            ("Ours: residual only", [ov.at["ltas_resid", "acc"], ov.at["ltas_resid", "macro_f1"]], "#56B4E9"),
            ("Ours: all features", [ov.at["all", "acc"], ov.at["all", "macro_f1"]], BLUE),
            ("Ours: all, time-shift control", [sh.at["all", "acc"], sh.at["all", "macro_f1"]], ORANGE)]
    x = np.arange(len(groups))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    for k, (label, vals, color) in enumerate(bars):
        pos = x + (k - 1.5) * w
        ax.bar(pos, vals, w * 0.92, color=color, label=label)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, groups)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Score (12 generators)")
    ax.set_title("Generator attribution on Echoes, 12 classes", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.savefig(FIG / "attr_headline.png", dpi=200)


def recall():
    r = pd.read_csv(RESULTS / "attr_reenc_recall.csv", index_col=0).loc[ORDER[::-1]] * 100
    y = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.hlines(y, r["ltas_resid"], r["all"], color="#cccccc", lw=2, zorder=1)
    ax.scatter(r["ltas_resid"], y, color="#56B4E9", s=36, zorder=2, label="Residual only")
    ax.scatter(r["all"], y, color=BLUE, s=36, zorder=3, label="All features")
    ax.set_yticks(y, [NAMES[g] for g in r.index])
    ax.set_xlabel("Recall (% of the generator's segments named correctly)")
    ax.set_xlim(50, 101)
    ax.set_title("Per-generator recall", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG / "attr_recall.png", dpi=200)


def shift():
    a = pd.read_csv(RESULTS / "attr_reenc_recall.csv", index_col=0)["all"] * 100
    b = pd.read_csv(RESULTS / "attr_reenc_shift_recall.csv", index_col=0)["all"] * 100
    g = ORDER[::-1]
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.hlines(y, a[g], b[g], color="#cccccc", lw=2, zorder=1)
    ax.scatter(a[g], y, color=BLUE, s=36, zorder=2, label="Re-encoded 128 kbps")
    ax.scatter(b[g], y, color=ORANGE, s=18, zorder=3, label="+ random time shift")
    for yi, gi in zip(y, g):
        ax.text(101.5, yi, f"{b[gi] - a[gi]:+.1f}", va="center", fontsize=8, color="#444444")
    ax.text(101.5, len(g) - 0.3, "change", fontsize=8, color="#444444")
    ax.set_yticks(y, [NAMES[x] for x in g])
    ax.set_xlim(65, 101)
    ax.set_xlabel("Recall, all features (%)")
    ax.set_title("Encoding-history control: recall barely moves", loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(FIG / "attr_shift.png", dpi=200)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for f in (headline, confusion, recall, shift):
        f()
    print(f"Wrote {', '.join(str(FIG / n) for n in ['attr_headline.png', 'attr_confusion.png', 'attr_recall.png', 'attr_shift.png'])}")


if __name__ == "__main__":
    main()
