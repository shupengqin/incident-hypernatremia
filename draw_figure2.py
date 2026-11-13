"""Figure 2: risk groups, same-variable comparison, sodium ablation, lead time."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")
NAVY = "#1f4e79"
AMBER = "#b86e2a"
GRAY = "#5c6370"
TEAL = "#2a6f6f"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
})


def letter(ax, s):
    ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.5))
    fig.subplots_adjust(hspace=0.48, wspace=0.72, bottom=0.10, right=0.78)

    ax = axes[0, 0]
    labels = ["Low", "Intermediate", "High"]
    xgb = [1.2, 3.2, 15.2]
    sodium = [1.4, 3.9, 14.5]
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, xgb, w, color=NAVY, label="XGBoost")
    ax.bar(x + w / 2, sodium, w, color=GRAY, label="Two-sodium logistic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Observed risk of sodium ≥146 mmol/L, %")
    ax.set_ylim(0, 20)
    ax.legend(loc="upper left")
    letter(ax, "A")

    ax = axes[0, 1]
    names = ["Harmonized\nXGBoost", "Harmonized\nlogistic", "Two-sodium\nlogistic"]
    # External AUROC and patient-level 95% intervals. Filled after figure_panel_cis.py.
    a146 = np.array([0.817, 0.810, 0.790])
    a150 = np.array([0.829, 0.835, 0.799])
    lo146 = np.array([0.006, 0.006, 0.006])
    hi146 = np.array([0.006, 0.006, 0.006])
    lo150 = np.array([0.015, 0.014, 0.014])
    hi150 = np.array([0.012, 0.012, 0.013])
    y = np.arange(len(names))
    ax.errorbar(a146, y + 0.12, xerr=[lo146, hi146], fmt="o", color=NAVY, ms=4.5, lw=0.8, capsize=2, label="Sodium ≥146", zorder=3)
    ax.errorbar(a150, y - 0.12, xerr=[lo150, hi150], fmt="o", color=AMBER, ms=4.5, lw=0.8, capsize=2, label="Sodium ≥150", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("External AUROC")
    ax.set_xlim(0.77, 0.86)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    letter(ax, "B")

    ax = axes[1, 0]
    labs = ["24 h", "Other labs\nupdated", "Sodium\nupdated", "All\nupdated"]
    vals = [0.846, 0.845, 0.874, 0.873]
    colors = [GRAY, GRAY, TEAL, TEAL]
    ax.bar(np.arange(4), vals, color=colors, width=0.72)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labs)
    ax.set_ylabel("External AUROC")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    letter(ax, "C")

    ax = axes[1, 1]
    bins = ["<2 h", "2–6 h", "6–12 h"]
    shares = [25.7, 39.0, 35.3]
    ax.bar(np.arange(3), shares, color=NAVY, width=0.7)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(bins)
    ax.set_ylabel("Share of detected events, %")
    ax.set_ylim(0, 55)
    ax.set_xlabel("Hours after hour 36")
    letter(ax, "D")

    fig.savefig(OUT / "figure2_abcd.svg", bbox_inches="tight")
    fig.savefig(OUT / "figure2_abcd.pdf", bbox_inches="tight")
    png = OUT / "figure2_abcd.png"
    tiff = OUT / "figure2_abcd.tiff"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    tmp = OUT / "figure2_abcd_rgb.tiff"
    im.save(tmp, format="TIFF", compression="tiff_lzw", dpi=(600, 600))
    tmp.replace(tiff)
    print("wrote", tiff, tiff.stat().st_size)


if __name__ == "__main__":
    main()
