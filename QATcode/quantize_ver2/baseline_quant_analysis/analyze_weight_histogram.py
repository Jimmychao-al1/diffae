"""
Effective-weight quantization analysis for all QuantModule_DiffAE_LoRA layers.

Research question: how does the final effective weight (w + LoRA delta) change
after quantization?  The focus is purely on w_lora vs Q(w_lora).

Per-layer output – 3-panel figure + JSON stats:
  Panel A  (top)    : full-range overlay   – w_lora vs w_lora_q
  Panel B  (middle) : zoomed overlay       – w_lora vs w_lora_q
                      x-axis = q{zoom_q*100:.0f} … q{(1-zoom_q)*100:.0f} of w_lora
                      shows subtle centre-region differences hidden in Panel A
  Panel C  (bottom) : residual histogram   – (w_lora_q − w_lora)
                      directly shows what quantization adds / subtracts

Stats text boxes embedded in each panel:
  Panel A: w_lora distribution stats + w_lora_q distribution stats
  Panel C: residual error stats

Output paths:
  Per-layer PNG  → weight_hist/per_layer/<idx>_<name>.png
  Per-layer JSON → weight_hist/stats/<idx>_<name>.json
  All-layers JSON → weight_hist/stats/all_layers.json

Usage:
    python -m QATcode.quantize_ver2.baseline_quant_analysis.analyze_weight_histogram
    python -m QATcode.quantize_ver2.baseline_quant_analysis.analyze_weight_histogram \\
        --ckpt  QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth \\
        --zoom-quantile 0.01
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(".")
sys.path.append("./model")

from QATcode.quantize_ver2.quant_layer_v2 import normalized_fake_quant
from QATcode.quantize_ver2.baseline_quant_analysis.pred_xstart_quantile_analysis import (
    CONFIG,
    _load_quant_and_ema_from_ckpt,
    create_float_quantized_model,
    load_calibration_data,
    load_diffae_model,
)
from QATcode.quantize_ver2.quant_model_lora_v2 import QuantModule_DiffAE_LoRA

LOGGER = logging.getLogger("analyze_weight_histogram")

OUT_ROOT = Path("QATcode/quantize_ver2/baseline_quant_analysis/weight_hist")


# ---------------------------------------------------------------------------
# Quantization helpers  (unchanged from previous version)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _compute_a_w(w: torch.Tensor) -> torch.Tensor:
    """Per-channel absmax scale — mirrors QuantModule_DiffAE_LoRA._compute_a_w()."""
    if len(w.shape) == 4:  # Conv [Cout, Cin, H, W]
        return w.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8
    elif len(w.shape) == 2:  # Linear [Cout, Cin]
        return w.abs().amax(dim=1, keepdim=True) + 1e-8
    return w.abs().max() + 1e-8  # fallback: per-tensor


@torch.no_grad()
def _dequant(w: torch.Tensor) -> torch.Tensor:
    """
    Fake-quantize then dequantize to original scale.

    Mirrors QuantModule_DiffAE_LoRA.forward() weight-only-quant path:
        a_w    = _compute_a_w(weight_eff)
        w_norm = normalized_fake_quant(weight_eff, a_w)   # ∈ [-1, 1]
        w_q    = w_norm * a_w                              # back to original scale
    """
    a_w = _compute_a_w(w)
    return normalized_fake_quant(w, a_w) * a_w


@torch.no_grad()
def _compute_lora_weight(mod: QuantModule_DiffAE_LoRA) -> torch.Tensor:
    """Reconstruct LoRA delta using the same identity-trick as forward()."""
    device = mod.org_weight.device
    if mod.fwd_func is F.linear:
        E = torch.eye(mod.org_weight.shape[1], device=device)
        lora_weight = mod.loraB(mod.loraA(E)).T  # [Cout, Cin]
    else:
        lora_weight = (
            mod.loraB.weight.squeeze(-1).squeeze(-1)  # [Cout, rank]
            @ mod.loraA.weight.permute(2, 3, 0, 1)  # [kH, kW, rank, Cin]
        ).permute(
            2, 3, 0, 1
        )  # [Cout, Cin, kH, kW]
    return lora_weight.to(device)


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_effective_weight_pair(
    mod: QuantModule_DiffAE_LoRA,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (w_lora, w_lora_q) as flat float32 numpy arrays in original weight scale.

      w_lora   = org_weight + lora_delta
      w_lora_q = dequant(w_lora)   — per-channel fake-quant → rescaled back
    """
    w = mod.org_weight.detach().float()
    lora = _compute_lora_weight(mod)
    w_lora = (w + lora).float()
    w_lora_q = _dequant(w_lora)
    return w_lora.cpu().flatten().numpy(), w_lora_q.cpu().flatten().numpy()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_distribution_stats(arr: np.ndarray) -> Dict:
    """Basic distribution statistics for a flat array."""
    return {
        "numel": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "abs_max": float(np.abs(arr).max()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q001": float(np.quantile(arr, 0.001)),
        "q01": float(np.quantile(arr, 0.01)),
        "q99": float(np.quantile(arr, 0.99)),
        "q999": float(np.quantile(arr, 0.999)),
    }


def compute_residual_stats(w_lora: np.ndarray, w_lora_q: np.ndarray) -> Dict:
    """Error statistics for residual = w_lora_q − w_lora."""
    res = w_lora_q - w_lora
    norm_wl = float(np.linalg.norm(w_lora))
    norm_res = float(np.linalg.norm(res))
    return {
        "mean": float(np.mean(res)),
        "std": float(np.std(res)),
        "abs_max": float(np.abs(res).max()),
        "MAE": float(np.mean(np.abs(res))),
        "RMSE": float(np.sqrt(np.mean(res**2))),
        "rel_L2": float(norm_res / norm_wl) if norm_wl > 0 else float("nan"),
        "zero_ratio": float(np.mean(res == 0.0)),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

# Keys shown in the Panel A comparison table (pre vs post)
_PANEL_A_KEYS = ["mean", "std", "q01", "q99", "abs_max"]

# Keys shown in the Panel C residual table
_PANEL_C_KEYS = ["mean", "std", "abs_max", "MAE", "RMSE", "rel_L2", "zero_ratio"]

# Shared text-box visual style
_TEXT_BOX_STYLE = dict(
    boxstyle="round,pad=0.4",
    facecolor="lightyellow",
    alpha=0.88,
    edgecolor="#aaaaaa",
    linewidth=0.5,
)


def _fmt_num(v, key: str) -> str:
    """
    Right-justify a stat value in 10 characters for monospace table alignment.

    numel          → integer with thousands separator
    float in       → fixed-point  (+X.XXXX), 4 decimal places
      [−9999, 9999]
    float outside  → scientific   (+X.XXXe+YY), 3 sig figs
    nan            → 'nan'
    """
    if key == "numel":
        return f"{int(v):>10,}"
    if isinstance(v, float) and np.isnan(v):
        return f"{'nan':>10s}"
    if v == 0.0 or (1e-4 <= abs(v) < 1e4):
        return f"{v:>+10.4f}"
    return f"{v:>+10.3e}"


def _make_comparison_table(dist_pre: Dict, dist_post: Dict) -> str:
    """
    Monospace two-column comparison table for Panel A.

    Format:
        <9-char metric>  <10-char pre-quant>  <10-char post-quant>
    """
    header = f"{'':9s}  {'pre-quant':>10s}  {'post-quant':>10s}"
    sep = f"{'':9s}  {'----------':>10s}  {'----------':>10s}"
    lines = [header, sep]
    for k in _PANEL_A_KEYS:
        lines.append(f"{k:<9s}  {_fmt_num(dist_pre[k], k)}  {_fmt_num(dist_post[k], k)}")
    return "\n".join(lines)


def _make_stats_table(stats: Dict, keys: List[str]) -> str:
    """
    Monospace single-column stats table for Panel C.

    Format:
        <10-char metric>: <10-char value>
    """
    lines = []
    for k in keys:
        lines.append(f"{k:<10s}: {_fmt_num(stats[k], k)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------


def _subsample_pair(
    a: np.ndarray, b: np.ndarray, n_max: int = 100_000, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (a_s, b_s, effective_n) with paired subsampling for large arrays."""
    n = a.size
    if n <= n_max:
        return a, b, n
    rng = np.random.default_rng(seed)
    sel = rng.choice(n, n_max, replace=False)
    return a[sel], b[sel], n_max


def _compute_ks_distance(a_sorted: np.ndarray, b_sorted: np.ndarray) -> float:
    """Kolmogorov–Smirnov distance between two empirical CDFs."""
    n_a = a_sorted.size
    n_b = b_sorted.size
    common = np.sort(np.unique(np.concatenate([a_sorted, b_sorted])))
    ecdf_a = np.searchsorted(a_sorted, common, side="right") / n_a
    ecdf_b = np.searchsorted(b_sorted, common, side="right") / n_b
    return float(np.max(np.abs(ecdf_a - ecdf_b)))


def plot_effective_weight_analysis(
    layer_name: str,
    w_lora: np.ndarray,
    w_lora_q: np.ndarray,
    w_lora_norm: np.ndarray,
    w_lora_q_norm: np.ndarray,
    out_path: Path,
    zoom_quantile: float = 0.01,
    bins: int = 301,
) -> Dict:
    """
    Four-panel figure for one layer — designed to avoid histogram binning
    artifacts that show up when a continuous pre-quant distribution is
    overlaid with a discrete post-quant distribution.

    Panel A — raw overlay using matplotlib.stairs (no vertical connector
              artifact at bin edges).  Odd `bins` puts 0 inside a bin, not
              on an edge.

    Panel B — normalized view (x / a_w).  Pre-quant drawn as a filled
              histogram; post-quant drawn as a *stem plot* at the exact
              quantization levels k/127, k ∈ [-127, 127], which is the
              true support after `round_ste` in normalized_fake_quant().
              Heights shown as PMF × 127 so area ≈ pre-quant density.

    Panel C — ECDF overlay (no binning at all).  Annotated KS distance
              gives a single-number summary of the distribution gap.

    Panel D — residual in LSB units  (LSB = a_w / 127).  Expected to be
              approximately uniform on [-0.5, 0.5] under rounding
              quantization; the theoretical U(-0.5, 0.5) is overlaid for
              direct correctness inspection.

    Returns the per-layer KS distance (also embedded in Panel C title).
    """
    dist_wl = compute_distribution_stats(w_lora)
    dist_wlq = compute_distribution_stats(w_lora_q)
    res_stats = compute_residual_stats(w_lora, w_lora_q)

    # ── Panel A: raw histogram edges (symmetric, clipped at q99.9%) ─────────
    all_sym = np.concatenate([w_lora, w_lora_q])
    q_lo_a = float(np.quantile(all_sym, 0.001))
    q_hi_a = float(np.quantile(all_sym, 0.999))
    L_a = max(abs(q_lo_a), abs(q_hi_a))
    edges_a = np.linspace(-L_a, L_a, bins + 1)
    hist_pre_a, _ = np.histogram(w_lora, bins=edges_a, density=True)
    hist_post_a, _ = np.histogram(w_lora_q, bins=edges_a, density=True)

    # ── Panel B: normalized pre-quant histogram + post-quant PMF stem ──────
    bins_b_norm = 200
    edges_b = np.linspace(-1.0, 1.0, bins_b_norm + 1)
    hist_pre_b, _ = np.histogram(w_lora_norm, bins=edges_b, density=True)
    k_values = np.clip(np.round(w_lora_q_norm * 127.0).astype(np.int64), -127, 127)
    level_counts = np.bincount(k_values + 127, minlength=255)
    pmf = level_counts / max(1, level_counts.sum())
    stem_density = pmf * 127.0  # convert PMF → density using level spacing 1/127
    levels_x = np.arange(-127, 128) / 127.0

    # ── Panel C: ECDF (subsample to cap plotting cost) ─────────────────────
    wl_s, wlq_s, n_eff = _subsample_pair(w_lora, w_lora_q, n_max=100_000)
    wl_sorted = np.sort(wl_s)
    wlq_sorted = np.sort(wlq_s)
    ecdf_y = np.arange(1, n_eff + 1, dtype=float) / n_eff
    ks_dist = _compute_ks_distance(wl_sorted, wlq_sorted)

    # ── Panel D: residual in LSB units  (post_norm − pre_norm) × 127 ───────
    residual_lsb = (w_lora_q_norm - w_lora_norm) * 127.0
    edges_d = np.linspace(-0.6, 0.6, 121)
    hist_r, _ = np.histogram(residual_lsb, bins=edges_d, density=True)

    # ── Figure layout ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(10, 14.5), gridspec_kw={"hspace": 0.55})
    fig.suptitle(layer_name, fontsize=9, y=0.985)

    # ── Panel A ────────────────────────────────────────────────────────────
    ax_a = axes[0]
    ax_a.stairs(
        hist_pre_a,
        edges_a,
        color="steelblue",
        linewidth=1.8,
        fill=False,
        label="w+LoRA  (pre-quant)",
    )
    ax_a.stairs(
        hist_post_a,
        edges_a,
        color="crimson",
        linewidth=1.2,
        linestyle="--",
        fill=False,
        label="Q(w+LoRA)  (post-quant)",
    )
    ax_a.axvline(0.0, color="gray", linewidth=0.7, linestyle=":")
    ax_a.set_xlim(-L_a, L_a)
    ax_a.set_ylabel("density")
    ax_a.set_title(
        f"Panel A — raw overlay  (x = ±{L_a:.4g}, clipped at q0.1%/q99.9%, bins={bins}, stairs)",
        fontsize=8,
    )
    ax_a.legend(fontsize=7, loc="upper left")
    ax_a.text(
        0.985,
        0.97,
        _make_comparison_table(dist_wl, dist_wlq),
        transform=ax_a.transAxes,
        fontsize=6.5,
        fontfamily="monospace",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=_TEXT_BOX_STYLE,
    )

    # ── Panel B ────────────────────────────────────────────────────────────
    ax_b = axes[1]
    ax_b.stairs(
        hist_pre_b,
        edges_b,
        color="steelblue",
        linewidth=1.3,
        fill=True,
        alpha=0.30,
        label="w+LoRA / a_w  (pre-quant, filled hist)",
    )
    ax_b.stairs(hist_pre_b, edges_b, color="steelblue", linewidth=1.3, fill=False)
    ax_b.vlines(
        levels_x,
        0,
        stem_density,
        colors="crimson",
        linewidth=0.7,
        alpha=0.85,
        label="Q(w+LoRA) / a_w  (PMF × 127 at k/127)",
    )
    ax_b.scatter(levels_x, stem_density, c="crimson", s=3, alpha=0.9)
    ax_b.axvline(0.0, color="gray", linewidth=0.5, linestyle=":")
    ax_b.set_xlim(-1.0, 1.0)
    ax_b.set_ylabel("density")
    ax_b.set_title(
        "Panel B — normalized view  (post-quant drawn as PMF stem at exact k/127 levels)",
        fontsize=8,
    )
    ax_b.legend(fontsize=7, loc="upper right")

    # ── Panel C ────────────────────────────────────────────────────────────
    ax_c = axes[2]
    ax_c.step(
        wl_sorted,
        ecdf_y,
        where="post",
        color="steelblue",
        linewidth=1.5,
        label="ECDF  pre-quant",
    )
    ax_c.step(
        wlq_sorted,
        ecdf_y,
        where="post",
        color="crimson",
        linewidth=1.2,
        linestyle="--",
        label="ECDF  post-quant",
    )
    ax_c.set_xlim(-L_a, L_a)
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_ylabel("cumulative prob.")
    sample_str = f"N={n_eff:,}" + (f" (sub-sampled from {w_lora.size:,})" if n_eff < w_lora.size else "")
    ax_c.set_title(
        f"Panel C — ECDF overlay  (no binning artifact;  KS distance = {ks_dist:.4f};  {sample_str})",
        fontsize=8,
    )
    ax_c.legend(fontsize=7, loc="upper left")
    ax_c.grid(alpha=0.25)

    # ── Panel D ────────────────────────────────────────────────────────────
    ax_d = axes[3]
    ax_d.stairs(
        hist_r,
        edges_d,
        fill=True,
        color="mediumpurple",
        alpha=0.70,
        label="residual in LSB",
    )
    ax_d.hlines(
        1.0,
        -0.5,
        0.5,
        colors="black",
        linestyles="--",
        linewidth=1.0,
        label="theoretical U(-0.5, 0.5)",
    )
    ax_d.axvline(-0.5, color="gray", linewidth=0.5, linestyle=":")
    ax_d.axvline(0.5, color="gray", linewidth=0.5, linestyle=":")
    ax_d.axvline(0.0, color="black", linewidth=0.7, linestyle=":")
    ax_d.set_xlabel("residual in LSB units  (LSB = a_w / 127;  expected: uniform on [-0.5, 0.5])")
    ax_d.set_ylabel("density")
    ax_d.set_title(
        "Panel D — quantization residual in LSB units  (rounding quantization → approx. uniform)",
        fontsize=8,
    )
    ax_d.legend(fontsize=7, loc="upper left")
    ax_d.text(
        0.985,
        0.97,
        _make_stats_table(res_stats, _PANEL_C_KEYS),
        transform=ax_d.transAxes,
        fontsize=6.5,
        fontfamily="monospace",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=_TEXT_BOX_STYLE,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    LOGGER.debug("Saved → %s", out_path)
    return {"ks_distance": ks_dist, "level_counts": level_counts}


# ---------------------------------------------------------------------------
# Extended extraction — raw + normalized in one pass
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_effective_weight_and_scale(
    mod: QuantModule_DiffAE_LoRA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (w_lora_flat, w_lora_q_flat, w_lora_norm_flat, w_lora_q_norm_flat).

    All four are flat float32 numpy arrays.

    Raw arrays are in original weight scale.
    Normalized arrays are divided by the per-channel absmax scale a_w
    (same _compute_a_w used inside _dequant), which removes the scale
    difference across layers/channels — making cross-layer comparison
    meaningful.

    Division is done BEFORE flatten to respect per-channel shape:
        Conv  [Cout, Cin, H, W]  →  a_w shape [Cout, 1, 1, 1]
        Linear [Cout, Cin]       →  a_w shape [Cout, 1]
    """
    w = mod.org_weight.detach().float()
    lora = _compute_lora_weight(mod)
    w_lora = (w + lora).float()
    a_w = _compute_a_w(w_lora)  # same scale as inside _dequant
    w_lora_q = normalized_fake_quant(w_lora, a_w) * a_w

    w_lora_norm = w_lora / a_w  # per-channel normalize, then flatten
    w_lora_q_norm = w_lora_q / a_w

    return (
        w_lora.cpu().flatten().numpy(),
        w_lora_q.cpu().flatten().numpy(),
        w_lora_norm.cpu().flatten().numpy(),
        w_lora_q_norm.cpu().flatten().numpy(),
    )


# ---------------------------------------------------------------------------
# Per-layer metrics
# ---------------------------------------------------------------------------


def compute_layerwise_metrics(
    name: str, idx: int, w_lora: np.ndarray, w_lora_q: np.ndarray
) -> Dict:
    """Compute the full set of per-layer summary metrics."""
    res = w_lora_q - w_lora
    abs_res = np.abs(res)
    norm_wl = float(np.linalg.norm(w_lora))
    norm_res = float(np.linalg.norm(res))
    return {
        "idx": idx,
        "layer_name": name,
        "numel": int(w_lora.size),
        "residual_mean": float(np.mean(res)),
        "residual_std": float(np.std(res)),
        "residual_abs_max": float(abs_res.max()),
        "residual_MAE": float(np.mean(abs_res)),
        "residual_RMSE": float(np.sqrt(np.mean(res**2))),
        "residual_rel_L2": float(norm_res / norm_wl) if norm_wl > 0 else float("nan"),
        "residual_q99_abs": float(np.quantile(abs_res, 0.99)),
        "residual_q999_abs": float(np.quantile(abs_res, 0.999)),
        "w_lora_mean": float(np.mean(w_lora)),
        "w_lora_std": float(np.std(w_lora)),
        "w_lora_q_mean": float(np.mean(w_lora_q)),
        "w_lora_q_std": float(np.std(w_lora_q)),
    }


# ---------------------------------------------------------------------------
# Overall raw distribution
# ---------------------------------------------------------------------------


def plot_overall_raw_distribution(
    all_wl: np.ndarray,
    all_wlq: np.ndarray,
    out_dir: Path,
    bins: int = 400,
) -> None:
    """
    Plot overall raw distribution by concatenating all layer weights.

    NOTE – exploratory only.
    Different layers / channels have very different dynamic ranges, so the
    raw concatenation conflates multiple scales.  Use the normalized view
    for cross-layer comparisons.
    """
    # ── figure 1: w_lora vs w_lora_q overlay (stairs — no step artifact) ──
    all_cat = np.concatenate([all_wl, all_wlq])
    lo = float(np.quantile(all_cat, 0.001))
    hi = float(np.quantile(all_cat, 0.999))
    # Use odd bin count so 0 falls inside a bin rather than on a bin edge.
    n_bins = bins + 1 if bins % 2 == 0 else bins
    edges = np.linspace(lo, hi, n_bins + 1)
    hist_pre, _ = np.histogram(all_wl, bins=edges, density=True)
    hist_post, _ = np.histogram(all_wlq, bins=edges, density=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stairs(
        hist_pre, edges, color="steelblue", linewidth=1.6, fill=False,
        label="w+LoRA  (pre-quant)",
    )
    ax.stairs(
        hist_post, edges, color="crimson", linewidth=1.1, linestyle="--", fill=False,
        label="Q(w+LoRA)  (post-quant)",
    )
    ax.axvline(0.0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlabel("weight value")
    ax.set_ylabel("density")
    ax.set_title(
        "Overall raw distribution — all layers concatenated\n"
        "(exploratory only: mixes different per-channel scales)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "overall_w_lora_vs_q.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── figure 2: raw residual ───────────────────────────────────────────
    residual = all_wlq - all_wl
    r_lo = float(np.quantile(residual, 0.001))
    r_hi = float(np.quantile(residual, 0.999))
    r_edges = np.linspace(r_lo, r_hi, n_bins + 1)
    hist_res, _ = np.histogram(residual, bins=r_edges, density=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stairs(
        hist_res, r_edges, fill=True, color="mediumpurple", alpha=0.75,
        label="Q(w+LoRA) − w+LoRA",
    )
    ax.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("residual value")
    ax.set_ylabel("density")
    ax.set_title(
        "Overall raw residual distribution — all layers concatenated\n"
        "(exploratory only: mixes different per-channel scales)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "overall_residual_raw.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── stats JSON ───────────────────────────────────────────────────────
    residual_full = all_wlq - all_wl
    norm_wl = float(np.linalg.norm(all_wl))
    norm_res = float(np.linalg.norm(residual_full))
    stats = {
        "_note": (
            "Raw overall stats — exploratory only.  "
            "Values are dominated by large-scale layers; "
            "use overall_normalized/ for cross-layer comparisons."
        ),
        "numel": int(all_wl.size),
        "w_lora": {
            "mean": float(np.mean(all_wl)),
            "std": float(np.std(all_wl)),
            "min": float(all_wl.min()),
            "max": float(all_wl.max()),
            "abs_max": float(np.abs(all_wl).max()),
            "q001": float(np.quantile(all_wl, 0.001)),
            "q01": float(np.quantile(all_wl, 0.01)),
            "q99": float(np.quantile(all_wl, 0.99)),
            "q999": float(np.quantile(all_wl, 0.999)),
        },
        "residual": {
            "mean": float(np.mean(residual_full)),
            "std": float(np.std(residual_full)),
            "abs_max": float(np.abs(residual_full).max()),
            "q001": float(np.quantile(residual_full, 0.001)),
            "q01": float(np.quantile(residual_full, 0.01)),
            "q99": float(np.quantile(residual_full, 0.99)),
            "q999": float(np.quantile(residual_full, 0.999)),
            "MAE": float(np.mean(np.abs(residual_full))),
            "RMSE": float(np.sqrt(np.mean(residual_full**2))),
            "rel_L2": float(norm_res / norm_wl) if norm_wl > 0 else float("nan"),
        },
    }
    (out_dir / "overall_raw_stats.json").write_text(json.dumps(stats, indent=2))
    LOGGER.info("Saved overall raw plots + stats → %s", out_dir)


# ---------------------------------------------------------------------------
# Overall normalized distribution
# ---------------------------------------------------------------------------


def plot_overall_normalized_distribution(
    all_nl: np.ndarray,
    all_nlq: np.ndarray,
    out_dir: Path,
    bins: int = 400,
) -> None:
    """
    Plot overall normalized distribution (w / a_w for each channel).

    Three figures are produced:
      1. overall_w_norm_vs_q_norm.png  —  pre-quant filled histogram
         overlaid with the post-quant PMF drawn as a stem plot at the
         *exact* quantization levels k/127.  This removes the step-
         histogram "downward spike" artifact at bin edges.
      2. overall_residual_norm.png     —  residual expressed in LSB units
         with theoretical U(-0.5, 0.5) overlaid for rounding-quantization
         correctness inspection.
      3. overall_ecdf_norm.png         —  ECDF overlay (no binning of any
         kind) with KS distance.

    Normalized overall view is more interpretable than raw because it
    removes the per-channel / per-layer scale differences, mapping every
    channel into [-1, 1].

    After normalization, the effective values should lie in [-1, 1] and
    the quantized values should cluster at multiples of 1/127 (~0.00787).
    """
    # ── figure 1: pre-quant filled histogram + post-quant PMF stem ──────
    n_bins = bins + 1 if bins % 2 == 0 else bins  # odd → 0 inside a bin
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    hist_pre, _ = np.histogram(all_nl, bins=edges, density=True)

    k_values = np.clip(np.round(all_nlq * 127.0).astype(np.int64), -127, 127)
    level_counts = np.bincount(k_values + 127, minlength=255)
    pmf = level_counts / max(1, level_counts.sum())
    stem_density = pmf * 127.0
    levels_x = np.arange(-127, 128) / 127.0

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stairs(
        hist_pre, edges, color="steelblue", linewidth=1.4, fill=True, alpha=0.30,
        label="w+LoRA / a_w  (pre-quant, filled hist)",
    )
    ax.stairs(hist_pre, edges, color="steelblue", linewidth=1.4, fill=False)
    ax.vlines(
        levels_x, 0, stem_density, colors="crimson", linewidth=0.7, alpha=0.85,
        label="Q(w+LoRA) / a_w  (PMF × 127 at k/127)",
    )
    ax.scatter(levels_x, stem_density, c="crimson", s=3, alpha=0.9)
    ax.axvline(0.0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlabel("normalized weight value  (w / absmax per channel)")
    ax.set_ylabel("density")
    ax.set_xlim(-1.0, 1.0)
    ax.set_title(
        "Overall normalized distribution — all layers  "
        "(post-quant as stem plot at the exact k/127 levels)",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "overall_w_norm_vs_q_norm.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── figure 2: residual in LSB units + theoretical uniform ──────────
    residual_lsb = (all_nlq - all_nl) * 127.0
    edges_r = np.linspace(-0.6, 0.6, 121)
    hist_r, _ = np.histogram(residual_lsb, bins=edges_r, density=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stairs(
        hist_r, edges_r, fill=True, color="mediumpurple", alpha=0.75,
        label="residual in LSB  ((Q − w) · 127 / a_w)",
    )
    ax.hlines(
        1.0, -0.5, 0.5, colors="black", linestyles="--", linewidth=1.0,
        label="theoretical U(-0.5, 0.5)",
    )
    ax.axvline(-0.5, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0.5, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0.0, color="black", linewidth=0.7, linestyle=":")
    ax.set_xlabel("residual in LSB units  (LSB = a_w / 127)")
    ax.set_ylabel("density")
    ax.set_title(
        "Overall normalized residual — all layers  "
        "(rounding quantization → approx. uniform on [-0.5, 0.5])",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "overall_residual_norm.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    res_norm = all_nlq - all_nl  # kept for the stats JSON below

    # ── figure 3: ECDF overlay (no binning artifact) ───────────────────
    n = all_nl.size
    n_max = 300_000
    if n > n_max:
        rng = np.random.default_rng(0)
        sel = rng.choice(n, n_max, replace=False)
        nl_s = all_nl[sel]
        nlq_s = all_nlq[sel]
        n_eff = n_max
    else:
        nl_s = all_nl
        nlq_s = all_nlq
        n_eff = n
    nl_sorted = np.sort(nl_s)
    nlq_sorted = np.sort(nlq_s)
    ecdf_y = np.arange(1, n_eff + 1, dtype=float) / n_eff
    ks_dist = _compute_ks_distance(nl_sorted, nlq_sorted)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step(
        nl_sorted, ecdf_y, where="post", color="steelblue", linewidth=1.6,
        label="ECDF  pre-quant  (w+LoRA / a_w)",
    )
    ax.step(
        nlq_sorted, ecdf_y, where="post", color="crimson", linewidth=1.2,
        linestyle="--", label="ECDF  post-quant  (Q(w+LoRA) / a_w)",
    )
    ax.axvline(0.0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("normalized weight value")
    ax.set_ylabel("cumulative prob.")
    sample_str = f"N={n_eff:,}" + (f" (sub-sampled from {n:,})" if n_eff < n else "")
    ax.set_title(
        f"Overall normalized ECDF — no binning artifact  "
        f"(KS distance = {ks_dist:.4f};  {sample_str})",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "overall_ecdf_norm.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── stats JSON ───────────────────────────────────────────────────────
    norm_nl = float(np.linalg.norm(all_nl))
    norm_res = float(np.linalg.norm(res_norm))
    stats = {
        "_note": (
            "Normalized overall stats.  "
            "Per-channel absmax (_compute_a_w) applied before flatten, "
            "so values are scale-invariant across layers/channels."
        ),
        "numel": int(all_nl.size),
        "w_lora_norm": {
            "mean": float(np.mean(all_nl)),
            "std": float(np.std(all_nl)),
            "abs_max": float(np.abs(all_nl).max()),
            "q001": float(np.quantile(all_nl, 0.001)),
            "q01": float(np.quantile(all_nl, 0.01)),
            "q99": float(np.quantile(all_nl, 0.99)),
            "q999": float(np.quantile(all_nl, 0.999)),
        },
        "residual_norm": {
            "mean": float(np.mean(res_norm)),
            "std": float(np.std(res_norm)),
            "abs_max": float(np.abs(res_norm).max()),
            "q001": float(np.quantile(res_norm, 0.001)),
            "q01": float(np.quantile(res_norm, 0.01)),
            "q99": float(np.quantile(res_norm, 0.99)),
            "q999": float(np.quantile(res_norm, 0.999)),
            "MAE": float(np.mean(np.abs(res_norm))),
            "RMSE": float(np.sqrt(np.mean(res_norm**2))),
            "rel_L2": float(norm_res / norm_nl) if norm_nl > 0 else float("nan"),
        },
    }
    (out_dir / "overall_normalized_stats.json").write_text(json.dumps(stats, indent=2))
    LOGGER.info("Saved overall normalized plots + stats → %s", out_dir)


# ---------------------------------------------------------------------------
# Per-layer metrics table + summary figures
# ---------------------------------------------------------------------------


def save_layerwise_metrics_table(metrics: List[Dict], out_dir: Path) -> None:
    """Save layerwise_metrics.json and layerwise_metrics.csv."""
    (out_dir / "layerwise_metrics.json").write_text(json.dumps(metrics, indent=2))

    if not metrics:
        return
    fieldnames = list(metrics[0].keys())
    with open(out_dir / "layerwise_metrics.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    LOGGER.info("Saved layerwise metrics table → %s", out_dir)


def _short_name(layer_name: str) -> str:
    """Last two dot-segments, e.g. 'emb_layers.1'."""
    parts = layer_name.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else layer_name


def plot_layerwise_summary_figures(metrics: List[Dict], out_dir: Path) -> None:
    """
    Five summary figures for the per-layer metrics overview.

    1. layerwise_rel_l2_hist.png        — histogram of residual_rel_L2
    2. layerwise_rel_l2_rank_top20.png  — top-20 layers by residual_rel_L2
    3. layerwise_mae_hist.png           — histogram of residual_MAE
    4. layerwise_q99_abs_rank_top20.png — top-20 layers by residual_q99_abs
    5. layerwise_numel_vs_rel_l2.png    — scatter: numel vs residual_rel_L2
    """
    names = [m["layer_name"] for m in metrics]
    short = [f"{m['idx']}:{_short_name(m['layer_name'])}" for m in metrics]
    rel_l2 = np.array([m["residual_rel_L2"] for m in metrics], dtype=float)
    mae = np.array([m["residual_MAE"] for m in metrics], dtype=float)
    q99_abs = np.array([m["residual_q99_abs"] for m in metrics], dtype=float)
    numel = np.array([m["numel"] for m in metrics], dtype=float)

    # ── 1. rel_L2 histogram ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    valid = rel_l2[~np.isnan(rel_l2)]
    ax.hist(valid, bins=30, color="steelblue", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("residual_rel_L2  (‖residual‖ / ‖w+LoRA‖)")
    ax.set_ylabel("layer count")
    ax.set_title("Per-layer residual rel-L2 distribution", fontsize=9)
    ax.axvline(
        float(np.nanmedian(rel_l2)),
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        label=f"median = {np.nanmedian(rel_l2):.4f}",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise_rel_l2_hist.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── 2. top-20 rel_L2 bar chart ────────────────────────────────────────
    top20_idx = np.argsort(rel_l2)[-20:][::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top20_idx))
    ax.bar(x, rel_l2[top20_idx], color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([short[i] for i in top20_idx], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("residual_rel_L2")
    ax.set_title("Top-20 layers by residual rel-L2", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise_rel_l2_rank_top20.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── 3. MAE histogram ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mae, bins=30, color="mediumpurple", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("residual_MAE")
    ax.set_ylabel("layer count")
    ax.set_title("Per-layer residual MAE distribution", fontsize=9)
    ax.axvline(
        float(np.median(mae)),
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        label=f"median = {np.median(mae):.6f}",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise_mae_hist.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── 4. top-20 q99_abs bar chart ───────────────────────────────────────
    top20q_idx = np.argsort(q99_abs)[-20:][::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top20q_idx))
    ax.bar(x, q99_abs[top20q_idx], color="mediumpurple", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([short[i] for i in top20q_idx], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("residual_q99_abs  (q99 of |residual|)")
    ax.set_title("Top-20 layers by residual q99_abs", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise_q99_abs_rank_top20.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── 5. scatter: numel vs rel_L2 ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = ~np.isnan(rel_l2)
    sc = ax.scatter(
        numel[mask], rel_l2[mask], c=rel_l2[mask], cmap="viridis", s=18, alpha=0.75, linewidths=0
    )
    plt.colorbar(sc, ax=ax, label="residual_rel_L2")
    ax.set_xlabel("numel  (number of weight parameters)")
    ax.set_ylabel("residual_rel_L2")
    ax.set_title("Layer size vs quantization rel-L2 error", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise_numel_vs_rel_l2.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    LOGGER.info("Saved 5 layerwise summary figures → %s", out_dir)


def plot_overall_level_occupancy(
    level_counts_per_layer: np.ndarray,
    layer_names: List[str],
    out_dir: Path,
) -> None:
    """
    Heat-map showing each layer's usage of the 255 quantization levels
    k ∈ [-127, 127].

    Rows = layers (ordered by idx),
    Columns = 255 integer levels,
    Color = fraction of weights landing on that level (log-scaled for
    dynamic range).

    Useful for spotting layers that:
      · waste levels (most weights compressed into a narrow central strip),
      · saturate at ±127 (clamp activated).
    """
    # Row-normalize → fraction per layer (so colour is comparable across rows)
    row_sums = level_counts_per_layer.sum(axis=1, keepdims=True).clip(min=1)
    frac = level_counts_per_layer / row_sums

    # log1p so near-empty levels are visible without being overshadowed by k=0
    frac_log = np.log1p(frac * 1000.0)

    n_layers = frac.shape[0]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.06 * n_layers + 2)))
    im = ax.imshow(
        frac_log,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        extent=[-127.5, 127.5, n_layers - 0.5, -0.5],
    )
    plt.colorbar(im, ax=ax, label="log1p(fraction × 1000)")
    ax.set_xlabel("quantization level  k  (∈ [-127, 127])")
    ax.set_ylabel("layer idx")
    ax.set_title(
        f"Per-layer level-occupancy heat-map  ({n_layers} layers × 255 levels)\n"
        "bright band near k=0 is expected (weights concentrate near zero)",
        fontsize=9,
    )
    ax.axvline(0.0, color="white", linewidth=0.4, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_dir / "overall_level_occupancy.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Companion line-plot: aggregate PMF over all layers
    total_counts = level_counts_per_layer.sum(axis=0)
    total_pmf = total_counts / max(1, total_counts.sum())
    k_grid = np.arange(-127, 128)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.vlines(k_grid, 0, total_pmf, colors="crimson", linewidth=0.7, alpha=0.85)
    ax.scatter(k_grid, total_pmf, c="crimson", s=4, alpha=0.9)
    ax.set_xlabel("quantization level  k")
    ax.set_ylabel("aggregate PMF")
    ax.set_title("Aggregate level PMF — all layers concatenated", fontsize=9)
    ax.axvline(0.0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlim(-128, 128)
    fig.tight_layout()
    fig.savefig(out_dir / "overall_level_pmf.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    LOGGER.info("Saved level-occupancy heat-map + aggregate PMF → %s", out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main(args: argparse.Namespace) -> None:
    """Public function main."""
    _setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    # ── 1. Build model ────────────────────────────────────────────────────────
    LOGGER.info("Loading base Diff-AE model …")
    base_model = load_diffae_model()
    diffusion_model = base_model.ema_model

    quant_model = create_float_quantized_model(
        diffusion_model,
        num_steps=CONFIG.NUM_DIFFUSION_STEPS,
        lora_rank=32,
        mode="train",
    )
    quant_model.to(device).eval()

    # ── 2. Calibrate (init TemporalActivationQuantizer scales) ───────────────
    LOGGER.info("Loading calibration data …")
    cali_images, cali_t, cali_y = load_calibration_data()

    quant_model.set_first_last_layer_to_8bit()
    quant_model.set_quant_state(True, True)

    if hasattr(quant_model, "set_runtime_mode"):
        quant_model.set_runtime_mode(mode="train", use_cached_aw=False, clear_cached_aw=True)

    LOGGER.info("Running calibration forward pass (32 samples) …")
    with torch.no_grad():
        _ = quant_model(
            x=cali_images[:32].to(device),
            t=cali_t[:32].to(device),
            cond=cali_y[:32].to(device),
        )

    # ── 3. Load checkpoint ────────────────────────────────────────────────────
    ckpt_path = args.ckpt or CONFIG.BEST_CKPT_PATH_100
    LOGGER.info("Loading QAT checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)
    LOGGER.info("Checkpoint loaded successfully")

    analysis_model = base_model.ema_model
    analysis_model.to(device).eval()

    if hasattr(analysis_model, "set_runtime_mode"):
        analysis_model.set_runtime_mode(mode="infer", use_cached_aw=True, clear_cached_aw=True)

    # ── 4. Collect layers ─────────────────────────────────────────────────────
    lora_modules = [
        (name, mod)
        for name, mod in analysis_model.named_modules()
        if isinstance(mod, QuantModule_DiffAE_LoRA)
    ]
    LOGGER.info("Found %d QuantModule_DiffAE_LoRA layers", len(lora_modules))

    # ── 5. Per-layer analysis + accumulate overall data ───────────────────────
    per_layer_dir = OUT_ROOT / "per_layer"
    stats_dir = OUT_ROOT / "stats"
    raw_dir = OUT_ROOT / "overall_raw"
    norm_dir = OUT_ROOT / "overall_normalized"
    summ_dir = OUT_ROOT / "overall_summary"
    for d in (per_layer_dir, stats_dir, raw_dir, norm_dir, summ_dir):
        d.mkdir(parents=True, exist_ok=True)

    all_layers_stats: List[Dict] = []
    # Containers for overall analysis (filled during per-layer loop)
    acc_raw_wl: List[np.ndarray] = []
    acc_raw_wlq: List[np.ndarray] = []
    acc_nl: List[np.ndarray] = []  # normalized w_lora
    acc_nlq: List[np.ndarray] = []  # normalized w_lora_q
    all_lm: List[Dict] = []  # per-layer metrics
    all_level_counts: List[np.ndarray] = []  # per-layer [255] k-level counts
    all_layer_names: List[str] = []

    for idx, (name, mod) in enumerate(lora_modules):
        LOGGER.info("[%d/%d] %s", idx + 1, len(lora_modules), name)

        # One call gives both raw and normalized arrays
        w_lora, w_lora_q, w_lora_norm, w_lora_q_norm = extract_effective_weight_and_scale(mod)

        # ── per-layer plot ────────────────────────────────────────────────
        safe = name.replace(".", "_")
        png_path = per_layer_dir / f"{idx:03d}_{safe}.png"
        plot_info = plot_effective_weight_analysis(
            layer_name=name,
            w_lora=w_lora,
            w_lora_q=w_lora_q,
            w_lora_norm=w_lora_norm,
            w_lora_q_norm=w_lora_q_norm,
            out_path=png_path,
            zoom_quantile=args.zoom_quantile,
        )

        # ── per-layer stats JSON ──────────────────────────────────────────
        layer_stats = {
            "idx": idx,
            "layer_name": name,
            "w_lora": compute_distribution_stats(w_lora),
            "w_lora_q": compute_distribution_stats(w_lora_q),
            "residual": compute_residual_stats(w_lora, w_lora_q),
            "ks_distance": plot_info["ks_distance"],
        }
        json_path = stats_dir / f"{idx:03d}_{safe}.json"
        json_path.write_text(json.dumps(layer_stats, indent=2))
        all_layers_stats.append(layer_stats)

        # ── accumulate for overall analysis ──────────────────────────────
        acc_raw_wl.append(w_lora)
        acc_raw_wlq.append(w_lora_q)
        acc_nl.append(w_lora_norm)
        acc_nlq.append(w_lora_q_norm)
        lm = compute_layerwise_metrics(name, idx, w_lora, w_lora_q)
        lm["ks_distance"] = plot_info["ks_distance"]
        all_lm.append(lm)
        all_level_counts.append(plot_info["level_counts"])
        all_layer_names.append(name)

    LOGGER.info("Saved %d per-layer PNGs → %s", len(lora_modules), per_layer_dir)

    # ── 6. Combined per-layer stats JSON ──────────────────────────────────────
    all_json = stats_dir / "all_layers.json"
    all_json.write_text(json.dumps(all_layers_stats, indent=2))
    LOGGER.info("Saved combined stats → %s", all_json)

    # ── 7. Overall raw distribution ───────────────────────────────────────────
    LOGGER.info("Generating overall raw distribution …")
    plot_overall_raw_distribution(
        np.concatenate(acc_raw_wl),
        np.concatenate(acc_raw_wlq),
        raw_dir,
    )

    # ── 8. Overall normalized distribution ───────────────────────────────────
    LOGGER.info("Generating overall normalized distribution …")
    plot_overall_normalized_distribution(
        np.concatenate(acc_nl),
        np.concatenate(acc_nlq),
        norm_dir,
    )

    # ── 8b. Overall level-occupancy heat-map + aggregate PMF ─────────────────
    LOGGER.info("Generating overall level-occupancy heat-map …")
    plot_overall_level_occupancy(
        np.stack(all_level_counts, axis=0),
        all_layer_names,
        norm_dir,
    )

    # ── 9. Per-layer metrics summary ──────────────────────────────────────────
    LOGGER.info("Generating per-layer metrics summary …")
    save_layerwise_metrics_table(all_lm, summ_dir)
    plot_layerwise_summary_figures(all_lm, summ_dir)

    LOGGER.info("Done.  Output root: %s", OUT_ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Effective-weight quantization analysis for all QuantModule_DiffAE_LoRA layers. "
            "Analyses how w+LoRA changes after quantization."
        )
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Path to QAT checkpoint .pth file.  "
            "Defaults to CONFIG.BEST_CKPT_PATH_100 "
            f"({CONFIG.BEST_CKPT_PATH_100!r})"
        ),
    )
    parser.add_argument(
        "--zoom-quantile",
        type=float,
        default=0.01,
        metavar="Q",
        help=(
            "Tail quantile for Panel B zoom.  "
            "0.01 → x-axis = q1%%…q99%% of w+LoRA.  "
            "0.05 → q5%%…q95%%.  (default: 0.01)"
        ),
    )
    args = parser.parse_args()
    main(args)
