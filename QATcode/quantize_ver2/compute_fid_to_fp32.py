#!/usr/bin/env python3
"""Compute FID-to-FP32: Q-DiffAE generated images vs FP32 Diff-AE reference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from pytorch_fid import fid_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

DEFAULT_FP32_DIR = "mycache/gen_images/ffhq128_autoenc_latent_T100/"
DEFAULT_QAT_DIR = "mycache/gen_images/ffhq128_autoenc_latent_QAT_T100/"
DEFAULT_REAL_DIR = "mycache/eval_images/ffhqlmdb256_size128_5000_5000/"
DEFAULT_OUTPUT_JSON = "QATcode/quantize_ver2/fid_to_fp32_results.json"


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def count_png_images(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.png"))


def validate_directory(directory: Path, label: str) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    n_png = count_png_images(directory)
    if n_png == 0:
        raise FileNotFoundError(f"{label} directory has no PNG images: {directory}")
    print(f"[validate] {label}: {directory} ({n_png} PNG files)")


def validate_indexed_images(directory: Path, num_images: int, label: str) -> None:
    missing = [i for i in range(num_images) if not (directory / f"{i}.png").exists()]
    if missing:
        preview = ", ".join(str(i) for i in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{label} is missing {len(missing)} indexed PNG files "
            f"(expected 0.png .. {num_images - 1}.png). Examples: {preview}{suffix}"
        )


def create_symlink_subset(src_dir: Path, num_images: int) -> str:
    """Create a temp directory with symlinks to the first num_images PNG files."""
    tmp_dir = tempfile.mkdtemp(prefix="qat_fid_subset_")
    print(f"Creating QAT subset ({num_images} images) in {tmp_dir} ...")
    try:
        for i in range(num_images):
            src = src_dir / f"{i}.png"
            dst = Path(tmp_dir) / f"{i}.png"
            os.symlink(src.resolve(), dst)
    except OSError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    print(f"QAT subset ready: {num_images} symlinks -> {src_dir}")
    return tmp_dir


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warn] CUDA not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def compute_fid(
    ref_dir: Path,
    gen_dir: Path,
    batch_size: int,
    device: torch.device,
    dims: int = 2048,
) -> float:
    return float(
        fid_score.calculate_fid_given_paths(
            [str(ref_dir), str(gen_dir)],
            batch_size,
            device=device,
            dims=dims,
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute FID-to-FP32 and Real-reference FID baselines."
    )
    parser.add_argument(
        "--fp32-dir",
        type=str,
        default=DEFAULT_FP32_DIR,
        help="FP32 Diff-AE generated images directory.",
    )
    parser.add_argument(
        "--qat-dir",
        type=str,
        default=DEFAULT_QAT_DIR,
        help="Q-DiffAE generated images directory (subset taken for @N).",
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        default=DEFAULT_REAL_DIR,
        help="Real validation images directory (reference baseline).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5000,
        help="Number of indexed images (0.png .. N-1.png) for FP32/QAT comparison.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for Inception feature extraction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for FID computation (e.g. cuda, cpu).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=DEFAULT_OUTPUT_JSON,
        help="Path to write JSON results.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    fp32_dir = resolve_repo_path(args.fp32_dir)
    qat_dir = resolve_repo_path(args.qat_dir)
    real_dir = resolve_repo_path(args.real_dir)
    output_json = resolve_repo_path(args.output_json)
    num_images = args.num_images
    dims = 2048
    device = resolve_device(args.device)

    print("=== FID-to-FP32 setup ===")
    validate_directory(fp32_dir, "FP32")
    validate_directory(qat_dir, "QAT")
    validate_directory(real_dir, "Real")
    validate_indexed_images(fp32_dir, num_images, "FP32")
    validate_indexed_images(qat_dir, num_images, "QAT")

    qat_subset_dir: str | None = None
    try:
        qat_subset_dir = create_symlink_subset(qat_dir, num_images)
        qat_subset_path = Path(qat_subset_dir)

        print(f"Computing FID (A): FP32_gen vs QAT_gen (N={num_images}) ...")
        fid_to_fp32 = compute_fid(fp32_dir, qat_subset_path, args.batch_size, device, dims)

        print("Computing FID (B): Real vs FP32_gen ...")
        fid_real_vs_fp32 = compute_fid(real_dir, fp32_dir, args.batch_size, device, dims)

        print(f"Computing FID (C): Real vs QAT_gen (N={num_images}) ...")
        fid_real_vs_qat = compute_fid(real_dir, qat_subset_path, args.batch_size, device, dims)

        print()
        print(f"=== FID-to-FP32 Results (N={num_images}) ===")
        print(f"(A) FID-to-FP32  [FP32_gen vs QAT_gen]:  {fid_to_fp32:.4f}")
        print(f"(B) FID-to-Real  [Real vs FP32_gen]:     {fid_real_vs_fp32:.4f}")
        print(
            f"(C) FID-to-Real  [Real vs QAT_gen]:      {fid_real_vs_qat:.4f}  "
            "(verify: should ≈ 14.95)"
        )
        print()
        print("Interpretation:")
        print(
            f"  FID-to-FP32 = {fid_to_fp32:.2f} means Q-DiffAE output is "
            f"{fid_to_fp32:.2f} FID units from FP32 output."
        )
        print(
            f"  For reference, FP32 itself is {fid_real_vs_fp32:.2f} FID units from Real."
        )

        payload = {
            "experiment": "FID-to-FP32",
            "num_images": num_images,
            "dims": dims,
            "results": {
                "fid_to_fp32": {
                    "ref": "FP32_gen",
                    "gen": "QAT_gen",
                    "value": fid_to_fp32,
                },
                "fid_real_vs_fp32": {
                    "ref": "Real",
                    "gen": "FP32_gen",
                    "value": fid_real_vs_fp32,
                },
                "fid_real_vs_qat": {
                    "ref": "Real",
                    "gen": "QAT_gen",
                    "value": fid_real_vs_qat,
                },
            },
            "paths": {
                "fp32_dir": str(fp32_dir),
                "qat_dir": str(qat_dir),
                "qat_subset_dir": str(qat_subset_path),
                "real_dir": str(real_dir),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"\nWrote JSON results to {output_json}")

    finally:
        if qat_subset_dir is not None and os.path.isdir(qat_subset_dir):
            print(f"Cleaning up QAT subset directory: {qat_subset_dir}")
            shutil.rmtree(qat_subset_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
