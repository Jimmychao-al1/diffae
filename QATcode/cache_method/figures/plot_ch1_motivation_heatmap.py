#!/usr/bin/env python3
"""
Ch1 Motivation Figure: Block x Timestep Feature Change Heatmap

Color: bright (yellow) = low change = cacheable
       dark  (black)  = high change = must recompute

Placement: QATcode/cache_method/figures/plot_ch1_motivation_heatmap.py
Run:  cd QATcode/cache_method/figures && python plot_ch1_motivation_heatmap.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

L1_DIR = Path("../a_L1_L2_cosine/T_100/v2_latest/L1")
OUTPUT_DIR = Path(".")

BLOCK_ORDER = (
    [f"model_input_blocks_{i}" for i in range(15)]
    + ["model_middle_block"]
    + [f"model_output_blocks_{i}" for i in range(15)]
)


def make_label(slug):
    if "input_blocks" in slug:
        return f"E{slug.split('_')[-1]}"
    elif "middle" in slug:
        return "M"
    elif "output_blocks" in slug:
        return f"D{slug.split('_')[-1]}"
    return slug


def load_l1rel_matrix(l1_dir):
    rows, labels = [], []
    for slug in BLOCK_ORDER:
        csv_path = l1_dir / slug / f"{slug}_l1rel.csv"
        if not csv_path.exists():
            print(f"WARNING: not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path, index_col=0)
        adjacent = np.array([df.values[i, i + 1] for i in range(df.shape[0] - 1)])
        rows.append(adjacent)
        labels.append(make_label(slug))
    return np.stack(rows, axis=0), labels


def plot_heatmap(matrix, labels, output_dir):
    n_intervals = matrix.shape[1]
    t_display = (n_intervals - 1) - np.arange(n_intervals)

    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=200)

    norm = mcolors.LogNorm(vmin=5e-4, vmax=0.7)
    im = ax.imshow(matrix, aspect="auto", cmap="inferno_r", norm=norm,
                   interpolation="nearest", origin="upper")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.015, aspect=25)
    cbar.ax.tick_params(labelsize=7, pad=2)
    cbar.set_label("")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.5, fontfamily="monospace")
    ax.set_ylabel("UNet Block", fontsize=10)

    tick_positions = list(range(0, n_intervals, 10))
    tick_labels_x = [str(int(t_display[i])) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_x, fontsize=7.5)
    ax.set_xlabel(r"Denoising Step $t$  (noise $\rightarrow$ clean)", fontsize=9.5)

    if "M" in labels:
        mid_idx = labels.index("M")
        ax.axhline(y=mid_idx - 0.5, color="white", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(y=mid_idx + 0.5, color="white", lw=0.8, ls="--", alpha=0.7)

    plt.tight_layout()

    # Labels above/below colorbar (right-aligned, close to bar edges)
    # top = vmax = dark = recompute; bottom = vmin = bright = cacheable
    X_LABEL, Y_TOP, Y_BOT = 0.78, 1.02, -0.02
    cbar.ax.text(
        X_LABEL, Y_TOP, "recomp.",
        fontsize=6.5, color="#555555", style="italic",
        va="bottom", ha="left", transform=cbar.ax.transAxes,
    )
    cbar.ax.text(
        X_LABEL, Y_BOT, "cache",
        fontsize=6.5, color="#555555", style="italic",
        va="top", ha="left", transform=cbar.ax.transAxes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = output_dir / f"ch1_motivation_heatmap.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    if not L1_DIR.exists():
        print(f"ERROR: L1 directory not found: {L1_DIR.resolve()}")
        return
    print("Loading L1rel data...")
    matrix, labels = load_l1rel_matrix(L1_DIR)
    print(f"Loaded: {matrix.shape} (blocks x intervals)")
    print(f"L1rel range: [{matrix.min():.6f}, {matrix.max():.4f}]")
    print(f"Max/min ratio: {matrix.max() / matrix.min():.0f}x")
    plot_heatmap(matrix, labels, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()