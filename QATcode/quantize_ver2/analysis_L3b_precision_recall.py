#!/usr/bin/env python3
"""Layer 3-b: Precision / Recall / Density / Coverage decomposition.

Compare Diff-AE (FP32) vs Q-DiffAE (W8A8) generated samples against real
FFHQ 128x128 using Inception v3 pool3 features + PRDC.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from PIL import Image
from prdc import compute_prdc
from pytorch_fid.inception import InceptionV3
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


REPO_ROOT = Path("/home/jimmy/diffae")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

DIR_REAL = "mycache/eval_images/ffhqlmdb256_size128_5000_5000/"
DIR_FAKE_FP = "mycache/gen_images/ffhq128_autoenc_latent_T100/"
DIR_FAKE_QAT = "mycache/gen_images/ffhq128_autoenc_latent_QAT_T100/"
OUTPUT_JSON = "analysis_L3b_results/l3b_prdc_results.json"

FEATURE_DIM = 2048
QAT_MAX_N = 5000


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a path relative to the Diff-AE repository root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def numeric_png_paths(img_dir: str | Path, max_n: Optional[int] = None) -> list[Path]:
    """Return PNG paths sorted by integer filename stem, optionally truncated."""
    img_dir = resolve_repo_path(img_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {img_dir}")

    paths = list(img_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG files found in: {img_dir}")

    def stem_as_int(path: Path) -> int:
        try:
            return int(path.stem)
        except ValueError as exc:
            raise ValueError(f"Expected numeric PNG filename, got: {path.name}") from exc

    paths = sorted(paths, key=stem_as_int)
    if max_n is not None:
        paths = paths[:max_n]
    return paths


class ImageFolderFlat(Dataset):
    """Load PNG images from one flat folder using numeric filename ordering."""

    def __init__(
        self,
        img_dir: str | Path,
        max_n: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.img_dir = resolve_repo_path(img_dir)
        self.paths = numeric_png_paths(self.img_dir, max_n=max_n)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                return self.transform(img)
            return transforms.ToTensor()(img)


def make_transform() -> transforms.Compose:
    """Resize to Inception input size and map pixels to [0, 1]."""
    return transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]
    )


def extract_features(
    dataloader: DataLoader,
    model: InceptionV3,
    device: torch.device,
    desc: str,
) -> np.ndarray:
    """Batch extract Inception features and return an [N, 2048] float32 array."""
    features: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            batch = batch.to(device, non_blocking=True)
            pred = model(batch)[0]
            if pred.ndim == 4:
                pred = pred.squeeze(3).squeeze(2)
            features.append(pred.detach().cpu().numpy().astype(np.float32, copy=False))

    if not features:
        raise RuntimeError(f"No features extracted for {desc}")

    out = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    if out.ndim != 2 or out.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"Unexpected feature shape for {desc}: {out.shape}")
    return out


def load_or_extract_features(
    name: str,
    dataset: Dataset,
    model: InceptionV3,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cache_dir: Path,
    refresh_cache: bool,
) -> np.ndarray:
    """Reuse feature cache when present, otherwise extract and save it."""
    cache_path = cache_dir / f"{name}_inception_pool3.npy"
    expected_n = len(dataset)

    if cache_path.exists() and not refresh_cache:
        features = np.load(cache_path)
        if features.shape == (expected_n, FEATURE_DIM):
            print(f"[cache] {name}: loaded {cache_path} {features.shape}")
            return features.astype(np.float32, copy=False)
        print(
            f"[cache] {name}: ignoring stale cache {cache_path} "
            f"with shape {features.shape}, expected {(expected_n, FEATURE_DIM)}"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    features = extract_features(dataloader, model, device, desc=f"extract {name}")
    np.save(cache_path, features)
    print(f"[cache] {name}: saved {cache_path} {features.shape}")
    return features


def to_builtin_floats(metrics: Dict[str, float]) -> Dict[str, float]:
    """Convert numpy scalar values returned by PRDC to JSON-safe floats."""
    return {key: float(value) for key, value in metrics.items()}


def print_comparison_table(result_fp: Dict[str, float], result_qat: Dict[str, float]) -> None:
    """Print the Layer 3-b PRDC comparison table."""
    metrics = ["precision", "recall", "density", "coverage"]
    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "density": "Density",
        "coverage": "Coverage",
    }

    print("=" * 60)
    print("  Layer 3-b: Precision & Recall Decomposition")
    print("=" * 60)
    print("  Metric        FP32 (Diff-AE)    QAT (Q-DiffAE)    \u0394(QAT-FP)")
    print("  ----------    --------------    ---------------    ----------")
    for metric in metrics:
        fp_val = result_fp[metric]
        qat_val = result_qat[metric]
        delta = qat_val - fp_val
        print(
            f"  {labels[metric]:<12}"
            f"{fp_val:>10.4f}        "
            f"{qat_val:>10.4f}        "
            f"{delta:+10.4f}"
        )
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Layer 3-b PRDC analysis for Diff-AE FP32 vs Q-DiffAE W8A8"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--real-dir", type=str, default=DIR_REAL)
    parser.add_argument("--fake-fp-dir", type=str, default=DIR_FAKE_FP)
    parser.add_argument("--fake-qat-dir", type=str, default=DIR_FAKE_QAT)
    parser.add_argument("--output-json", type=str, default=OUTPUT_JSON)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    output_json = resolve_repo_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_json.parent / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)

    transform = make_transform()
    real_dataset = ImageFolderFlat(args.real_dir, transform=transform)
    fp_dataset = ImageFolderFlat(args.fake_fp_dir, transform=transform)
    qat_dataset = ImageFolderFlat(args.fake_qat_dir, max_n=QAT_MAX_N, transform=transform)

    print("[data] real:", resolve_repo_path(args.real_dir), len(real_dataset))
    print("[data] fp32:", resolve_repo_path(args.fake_fp_dir), len(fp_dataset))
    print("[data] qat:", resolve_repo_path(args.fake_qat_dir), len(qat_dataset))

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIM]
    model = InceptionV3([block_idx]).to(device).eval()

    real_features = load_or_extract_features(
        "real",
        real_dataset,
        model,
        device,
        args.batch_size,
        args.num_workers,
        cache_dir,
        args.refresh_cache,
    )
    fp_features = load_or_extract_features(
        "fp32",
        fp_dataset,
        model,
        device,
        args.batch_size,
        args.num_workers,
        cache_dir,
        args.refresh_cache,
    )
    qat_features = load_or_extract_features(
        "qat",
        qat_dataset,
        model,
        device,
        args.batch_size,
        args.num_workers,
        cache_dir,
        args.refresh_cache,
    )

    print(f"[prdc] computing FP32, nearest_k={args.nearest_k}")
    result_fp = to_builtin_floats(
        compute_prdc(real_features, fp_features, nearest_k=args.nearest_k)
    )
    print(f"[prdc] computing QAT, nearest_k={args.nearest_k}")
    result_qat = to_builtin_floats(
        compute_prdc(real_features, qat_features, nearest_k=args.nearest_k)
    )

    payload = {
        "fp32": result_fp,
        "qat": result_qat,
        "config": {
            "real_n": int(real_features.shape[0]),
            "fake_fp_n": int(fp_features.shape[0]),
            "fake_qat_n": int(qat_features.shape[0]),
            "nearest_k": args.nearest_k,
            "feature_dim": FEATURE_DIM,
            "feature_extractor": "InceptionV3_pool3",
            "real_dir": str(resolve_repo_path(args.real_dir)),
            "fake_fp_dir": str(resolve_repo_path(args.fake_fp_dir)),
            "fake_qat_dir": str(resolve_repo_path(args.fake_qat_dir)),
            "feature_cache_dir": str(cache_dir),
        },
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print_comparison_table(result_fp, result_qat)
    print(f"[done] wrote {output_json}")


if __name__ == "__main__":
    main()
