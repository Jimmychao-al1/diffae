#!/usr/bin/env python3
"""
Ch1 Motivation Figure: Block x Timestep Feature Change Heatmap
==============================================================
From S3-Cache Stage 0 L1rel CSV, extract adjacent-step L1 relative change
and plot a (31 blocks x 99 intervals) heatmap.

Color logic: bright (yellow) = low change = cacheable
             dark  (black)  = high change = must recompute

Placement: QATcode/cache_method/figures/plot_ch1_motivation_heatmap.py
Run:
  cd QATcode/cache_method/figures
  python plot_ch1_motivation_heatmap.py

Output:
  ch1_motivation_heatmap.pdf  (for LaTeX)
  ch1_motivation_heatmap.png  (preview, 300 dpi)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ============================================================
# Path config -- adjust for your environment
# ============================================================

# Stage 0 L1 results (each block has a subdirectory with *_l1rel.csv)
L1_DIR = Path("../a_L1_L2_cosine/T_100/v2_latest/L1")

# Output directory
OUTPUT_DIR = Path(".")

# ============================================================
# Block ordering (UNet architecture order)
# ============================================================

BLOCK_ORDER = (
    [f"model_input_blocks_{i}" for i in range(15)]
    + ["model_middle_block"]
    + [f"model_output_blocks_{i}" for i in range(15)]
)


def make_label(slug: str) -> str:
    if "input_blocks" in slug:
        return f"E{slug.split('_')[-1]}"
    elif "middle" in slug:
        return "M"
    elif "output_blocks" in slug:
        return f"D{slug.split('_')[-1]}"
    return slug


# ============================================================
# Data loading
# ============================================================

def load_l1rel_matrix(l1_dir: Path):
    """
    From each block's L1rel CSV, extract the super-diagonal
    (adjacent-step L1 relative change) and stack into (31, 99).
    """
    rows = []
    labels = []

    for slug in BLOCK_ORDER:
        csv_path = l1_dir / slug / f"{slug}_l1rel.csv"
        if not csv_path.exists():
            print(f"WARNING: not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path, index_col=0)
        adjacent = np.array([df.values[i, i + 1] for i in range(df.shape[0] - 1)])
        rows.append(adjacent)
        labels.append(make_label(slug))

    matrix = np.stack(rows, axis=0)
    return matrix, labels


# ============================================================
# Plotting
# ============================================================

def plot_heatmap(matrix: np.ndarray, labels: list, output_dir: Path):
    n_intervals = matrix.shape[1]
    t_display = (n_intervals - 1) - np.arange(n_intervals)

    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=200)

    norm = mcolors.LogNorm(vmin=5e-4, vmax=0.7)
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="inferno_r",
        norm=norm,
        interpolation="nearest",
        origin="upper",
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.015, aspect=25)
    cbar.set_label("Relative Feature Change (log scale)", fontsize=9, labelpad=6)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.text(
        3.0, 0.01, "cacheable", fontsize=6, color="#666666",
        transform=cbar.ax.transAxes, va="bottom", ha="left", style="italic",
    )
    cbar.ax.text(
        3.0, 0.96, "recompute", fontsize=6, color="#666666",
        transform=cbar.ax.transAxes, va="top", ha="left", style="italic",
    )

    # Y-axis
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.5, fontfamily="monospace")
    ax.set_ylabel("UNet Block", fontsize=10)

    # X-axis
    tick_positions = list(range(0, n_intervals, 10))
    tick_labels_x = [str(int(t_display[i])) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_x, fontsize=7.5)
    ax.set_xlabel(r"Denoising Step $t$  (noise $\rightarrow$ clean)", fontsize=9.5)

    # Encoder / Middle / Decoder dividers
    if "M" in labels:
        mid_idx = labels.index("M")
        ax.axhline(y=mid_idx - 0.5, color="white", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(y=mid_idx + 0.5, color="white", lw=0.8, ls="--", alpha=0.7)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "ch1_motivation_heatmap.pdf"
    png_path = output_dir / "ch1_motivation_heatmap.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    if not L1_DIR.exists():
        print(f"ERROR: L1 directory not found: {L1_DIR.resolve()}")
        print("Please adjust L1_DIR at the top of this script.")
        return

    print("Loading L1rel data...")
    matrix, labels = load_l1rel_matrix(L1_DIR)
    print(f"Loaded: {matrix.shape} (blocks x intervals)")
    print(f"L1rel range: [{matrix.min():.6f}, {matrix.max():.4f}]")
    print(f"Mean: {matrix.mean():.4f}, Median: {np.median(matrix):.4f}")

    print(f"\n--- Caption stats ---")
    print(f"Max/min ratio: {matrix.max() / matrix.min():.0f}x")
    print(f"Cells with L1rel < 0.01: {(matrix < 0.01).sum() / matrix.size * 100:.1f}%")
    print(f"Cells with L1rel > 0.1:  {(matrix > 0.1).sum() / matrix.size * 100:.1f}%")

    plot_heatmap(matrix, labels, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()