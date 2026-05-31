#!/usr/bin/env python3
"""Layer 3-b Bootstrap: Statistical significance of PRDC differences."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from prdc import compute_prdc
from tqdm.auto import tqdm


REPO_ROOT = Path("/home/jimmy/diffae")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

DEFAULT_FEATURE_DIR = "analysis_L3b_results/features"
DEFAULT_OUTPUT = "analysis_L3b_results/l3b_bootstrap_results.json"
METRICS = ("precision", "recall", "density", "coverage")


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a path relative to the Diff-AE repository root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def load_feature_file(feature_dir: Path, candidates: Iterable[str]) -> np.ndarray:
    """Load the first existing feature cache among compatible filename variants."""
    for name in candidates:
        path = feature_dir / name
        if path.exists():
            arr = np.load(path)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D features in {path}, got shape {arr.shape}")
            print(f"[features] loaded {path} {arr.shape}")
            return arr.astype(np.float32, copy=False)

    candidate_text = ", ".join(str(feature_dir / name) for name in candidates)
    raise FileNotFoundError(f"No compatible feature cache found. Tried: {candidate_text}")


def load_features(feature_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached real, FP32, and QAT Inception features."""
    feature_dir = resolve_repo_path(feature_dir)
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")

    real = load_feature_file(feature_dir, ("real_inception_pool3.npy", "real_features.npy"))
    fp = load_feature_file(
        feature_dir,
        ("fp32_inception_pool3.npy", "fake_fp_features.npy", "fp_features.npy"),
    )
    qat = load_feature_file(
        feature_dir,
        ("qat_inception_pool3.npy", "fake_qat_features.npy", "qat_features.npy"),
    )
    return real, fp, qat


def bootstrap_prdc(
    real: np.ndarray,
    fp: np.ndarray,
    qat: np.ndarray,
    n_bootstrap: int,
    nearest_k: int,
    seed: int = 42,
) -> Dict[str, Dict[str, list[float]]]:
    """Run paired bootstrap using the same resampled real set for FP32 and QAT."""
    rng = np.random.RandomState(seed)
    results: Dict[str, Dict[str, list[float]]] = {
        metric: {"fp": [], "qat": [], "delta": []} for metric in METRICS
    }

    n_real = len(real)
    n_fp = len(fp)
    n_qat = len(qat)

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        idx_real = rng.choice(n_real, size=n_real, replace=True)
        idx_fp = rng.choice(n_fp, size=n_fp, replace=True)
        idx_qat = rng.choice(n_qat, size=n_qat, replace=True)

        real_b = real[idx_real]
        fp_b = fp[idx_fp]
        qat_b = qat[idx_qat]

        res_fp = compute_prdc(real_b, fp_b, nearest_k=nearest_k)
        res_qat = compute_prdc(real_b, qat_b, nearest_k=nearest_k)

        for metric in METRICS:
            fp_val = float(res_fp[metric])
            qat_val = float(res_qat[metric])
            results[metric]["fp"].append(fp_val)
            results[metric]["qat"].append(qat_val)
            results[metric]["delta"].append(qat_val - fp_val)

    return results


def two_tailed_sign_p_value(deltas: np.ndarray) -> float:
    """Compute the requested two-tailed bootstrap sign p-value."""
    p_val = 2.0 * min(float(np.mean(deltas >= 0)), float(np.mean(deltas <= 0)))
    return min(p_val, 1.0)


def compute_statistics(
    results: Dict[str, Dict[str, list[float]]]
) -> Dict[str, Dict[str, float | bool | list[float]]]:
    """Compute bootstrap mean, std, percentile CI, p-value, and significance."""
    stats: Dict[str, Dict[str, float | bool | list[float]]] = {}

    for metric, vals in results.items():
        deltas = np.asarray(vals["delta"], dtype=np.float64)
        fp_arr = np.asarray(vals["fp"], dtype=np.float64)
        qat_arr = np.asarray(vals["qat"], dtype=np.float64)

        p_val = two_tailed_sign_p_value(deltas)
        stats[metric] = {
            "fp32_mean": float(np.mean(fp_arr)),
            "fp32_std": float(np.std(fp_arr)),
            "qat_mean": float(np.mean(qat_arr)),
            "qat_std": float(np.std(qat_arr)),
            "delta_mean": float(np.mean(deltas)),
            "delta_std": float(np.std(deltas)),
            "delta_ci_95": [
                float(np.percentile(deltas, 2.5)),
                float(np.percentile(deltas, 97.5)),
            ],
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
        }

    return stats


def significance_marks(p_value: float) -> str:
    """Return conventional significance marks for a p-value."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def print_table(
    stats: Dict[str, Dict[str, float | bool | list[float]]],
    n_bootstrap: int,
) -> None:
    """Print a compact bootstrap significance table."""
    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "density": "Density",
        "coverage": "Coverage",
    }

    print("=" * 64)
    print(f"  Layer 3-b Bootstrap Significance Test (B={n_bootstrap})")
    print("=" * 64)
    print("  Metric      Delta(QAT-FP)  95% CI             p-value  Sig?")
    print("  ---------   -------------  -----------------  -------  ----")

    for metric in METRICS:
        row = stats[metric]
        ci = row["delta_ci_95"]
        assert isinstance(ci, list)
        p_value = float(row["p_value"])
        print(
            f"  {labels[metric]:<11}"
            f"{float(row['delta_mean']):+12.4f}   "
            f"[{ci[0]:+0.4f}, {ci[1]:+0.4f}]   "
            f"{p_value:0.4f}   "
            f"{significance_marks(p_value)}"
        )

    print("=" * 64)
    print("  * p < 0.05    ** p < 0.01    *** p < 0.001")
    print("=" * 64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap PRDC differences for Layer 3-b"
    )
    parser.add_argument("--feature-dir", type=str, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    real, fp, qat = load_features(args.feature_dir)
    if real.shape[1] != fp.shape[1] or real.shape[1] != qat.shape[1]:
        raise ValueError(
            "Feature dimensions do not match: "
            f"real={real.shape}, fp={fp.shape}, qat={qat.shape}"
        )

    print(
        "[config] "
        f"n_bootstrap={args.n_bootstrap}, nearest_k={args.nearest_k}, "
        f"seed={args.seed}, feature_dim={real.shape[1]}"
    )
    print(f"[config] real_n={len(real)}, fake_fp_n={len(fp)}, fake_qat_n={len(qat)}")

    raw_results = bootstrap_prdc(
        real=real,
        fp=fp,
        qat=qat,
        n_bootstrap=args.n_bootstrap,
        nearest_k=args.nearest_k,
        seed=args.seed,
    )
    stats = compute_statistics(raw_results)

    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_bootstrap": args.n_bootstrap,
        "nearest_k": args.nearest_k,
        "seed": args.seed,
        "real_n": int(real.shape[0]),
        "fake_n": int(fp.shape[0]),
        "fake_fp_n": int(fp.shape[0]),
        "fake_qat_n": int(qat.shape[0]),
        "feature_dim": int(real.shape[1]),
        "feature_dir": str(resolve_repo_path(args.feature_dir)),
        "metrics": stats,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print_table(stats, args.n_bootstrap)
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
