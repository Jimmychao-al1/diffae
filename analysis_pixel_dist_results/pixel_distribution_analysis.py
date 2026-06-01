from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats
from tqdm.auto import tqdm


FP32_DIR = Path("/home/jimmy/diffae/mycache/gen_images/ffhq128_autoenc_latent_T100")
QAT_DIR = Path("/home/jimmy/diffae/mycache/gen_images/ffhq128_autoenc_latent_QAT_T100")
REAL_DIR = Path("/home/jimmy/diffae/mycache/eval_images/ffhqlmdb256_size128_5000_5000")
OUTPUT_DIR = Path("/home/jimmy/diffae/analysis_pixel_dist_results")

SOURCES = {
    "fp32": FP32_DIR,
    "qat": QAT_DIR,
    "real": REAL_DIR,
}
CHANNELS = ("overall", "R", "G", "B")
CORE_STATS = ("mean", "std", "min", "max", "skewness", "kurtosis")
PAIRED_STATS = ("mean", "std", "skewness", "kurtosis")
TAIL_STATS = ("pixel_range", "tail_low_ratio", "tail_high_ratio", "extreme_ratio")
INDEX_START = 0
INDEX_END = 4991
EPS = 1e-10


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def round_float(value: Any, digits: int = 6) -> Any:
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {k: round_float(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_float(v, digits) for v in value]
    return value


def load_image(path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001 - keep the batch running on bad files.
        logging.warning("Failed to read %s: %s", path, exc)
        return None


def calc_stats(values: np.ndarray) -> Dict[str, float]:
    flat = values.reshape(-1).astype(np.float64)
    return {
        "mean": round(float(np.mean(flat)), 6),
        "std": round(float(np.std(flat)), 6),
        "min": round(float(np.min(flat)), 6),
        "max": round(float(np.max(flat)), 6),
        "skewness": round(float(stats.skew(flat)), 6),
        "kurtosis": round(float(stats.kurtosis(flat, fisher=True)), 6),
    }


def calc_tail(values: np.ndarray) -> Dict[str, float]:
    flat = values.reshape(-1)
    total = float(flat.size)
    return {
        "pixel_range": round(float(np.max(flat) - np.min(flat)), 6),
        "tail_low_ratio": round(float(np.count_nonzero(flat <= 10) / total), 6),
        "tail_high_ratio": round(float(np.count_nonzero(flat >= 245) / total), 6),
        "extreme_ratio": round(float(np.count_nonzero((flat == 0) | (flat == 255)) / total), 6),
    }


def update_histograms(histograms: Dict[str, np.ndarray], image: np.ndarray) -> None:
    histograms["overall"] += np.bincount(image.reshape(-1), minlength=256)
    histograms["R"] += np.bincount(image[:, :, 0].reshape(-1), minlength=256)
    histograms["G"] += np.bincount(image[:, :, 1].reshape(-1), minlength=256)
    histograms["B"] += np.bincount(image[:, :, 2].reshape(-1), minlength=256)


def analyze_source(name: str, folder: Path, indices: Iterable[int]) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    rows: List[Dict[str, Any]] = []
    histograms = {channel: np.zeros(256, dtype=np.float64) for channel in CHANNELS}

    for idx in tqdm(list(indices), desc=f"analyze {name}"):
        image = load_image(folder / f"{idx}.png")
        if image is None:
            continue

        row: Dict[str, Any] = {
            "index": idx,
            "overall": calc_stats(image),
            "R": calc_stats(image[:, :, 0]),
            "G": calc_stats(image[:, :, 1]),
            "B": calc_stats(image[:, :, 2]),
            "tail_behavior": calc_tail(image),
        }
        rows.append(row)
        update_histograms(histograms, image)

    return rows, histograms


def aggregate_section(rows: List[Dict[str, Any]], section: str, stat_names: Iterable[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for stat_name in stat_names:
        values = np.asarray([row[section][stat_name] for row in rows], dtype=np.float64)
        out[stat_name] = {
            "avg": round(float(np.mean(values)), 6),
            "std": round(float(np.std(values)), 6),
        }
    return out


def aggregate_stats(per_image: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for source, rows in per_image.items():
        out[source] = {channel: aggregate_section(rows, channel, CORE_STATS) for channel in CHANNELS}
        out[source]["tail_behavior"] = aggregate_section(rows, "tail_behavior", TAIL_STATS)
    return out


def normalize_histograms(histograms: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for source, sections in histograms.items():
        out[source] = {}
        for channel, counts in sections.items():
            total = float(np.sum(counts))
            density = counts / total if total > 0 else counts
            out[source][channel] = [round(float(x), 6) for x in density]
    return out


def histogram_distances(histograms: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    bins = np.arange(256, dtype=np.float64)
    pairs = {
        "fp32_to_real": ("fp32", "real"),
        "qat_to_real": ("qat", "real"),
        "fp32_to_qat": ("fp32", "qat"),
    }
    out: Dict[str, Any] = {}
    for channel in CHANNELS:
        out[channel] = {}
        for pair_name, (left, right) in pairs.items():
            hist_left = np.asarray(histograms[left][channel], dtype=np.float64)
            hist_right = np.asarray(histograms[right][channel], dtype=np.float64)
            out[channel][pair_name] = {
                "emd": round(float(stats.wasserstein_distance(bins, bins, hist_left, hist_right)), 6),
                "kl": round(float(stats.entropy(hist_left + EPS, hist_right + EPS)), 6),
            }
    return out


def rows_by_index(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(row["index"]): row for row in rows}


def safe_wilcoxon(left: np.ndarray, right: np.ndarray) -> Dict[str, float]:
    try:
        result = stats.wilcoxon(left, right, alternative="two-sided")
        return {
            "statistic": round(float(result.statistic), 6),
            "p_value": round(float(result.pvalue), 6),
        }
    except ValueError as exc:
        logging.warning("Wilcoxon fallback used: %s", exc)
        return {"statistic": 0.0, "p_value": 1.0}


def paired_stat_comparison(per_image: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    maps = {source: rows_by_index(rows) for source, rows in per_image.items()}
    common = sorted(set(maps["fp32"]) & set(maps["qat"]) & set(maps["real"]))
    out: Dict[str, Any] = {"num_pairs": len(common), "sections": {}}

    for channel in CHANNELS:
        out["sections"][channel] = {}
        for stat_name in PAIRED_STATS:
            delta_fp32 = np.asarray(
                [abs(maps["fp32"][idx][channel][stat_name] - maps["real"][idx][channel][stat_name]) for idx in common],
                dtype=np.float64,
            )
            delta_qat = np.asarray(
                [abs(maps["qat"][idx][channel][stat_name] - maps["real"][idx][channel][stat_name]) for idx in common],
                dtype=np.float64,
            )
            mean_fp32 = float(np.mean(delta_fp32))
            mean_qat = float(np.mean(delta_qat))
            test = safe_wilcoxon(delta_fp32, delta_qat)
            direction = "qat_closer" if mean_qat < mean_fp32 else "fp32_closer" if mean_fp32 < mean_qat else "tie"
            out["sections"][channel][stat_name] = {
                **test,
                "mean_delta_fp32": round(mean_fp32, 6),
                "mean_delta_qat": round(mean_qat, 6),
                "direction": direction,
            }
    return out


def paired_tail_comparison(per_image: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    maps = {source: rows_by_index(rows) for source, rows in per_image.items()}
    common = sorted(set(maps["fp32"]) & set(maps["qat"]) & set(maps["real"]))
    pairs = {
        "fp32_vs_real": ("fp32", "real"),
        "qat_vs_real": ("qat", "real"),
        "fp32_vs_qat": ("fp32", "qat"),
    }
    out: Dict[str, Any] = {"num_pairs": len(common), "metrics": {}}

    for metric in TAIL_STATS:
        out["metrics"][metric] = {}
        for pair_name, (left, right) in pairs.items():
            left_values = np.asarray([maps[left][idx]["tail_behavior"][metric] for idx in common], dtype=np.float64)
            right_values = np.asarray([maps[right][idx]["tail_behavior"][metric] for idx in common], dtype=np.float64)
            test = safe_wilcoxon(left_values, right_values)
            mean_left = float(np.mean(left_values))
            mean_right = float(np.mean(right_values))
            out["metrics"][metric][pair_name] = {
                **test,
                f"mean_{left}": round(mean_left, 6),
                f"mean_{right}": round(mean_right, 6),
                "direction": f"{left}_lower" if mean_left < mean_right else f"{right}_lower" if mean_right < mean_left else "tie",
            }
    return out


def plot_histograms(histograms: Dict[str, Dict[str, List[float]]], output_path: Path) -> None:
    bins = np.arange(256)
    colors = {"real": "black", "fp32": "blue", "qat": "red"}
    labels = {"real": "Real", "fp32": "FP32", "qat": "QAT"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, channel in zip(axes.reshape(-1), CHANNELS):
        for source in ("real", "fp32", "qat"):
            ax.step(
                bins,
                histograms[source][channel],
                where="mid",
                label=labels[source],
                color=colors[source],
                alpha=0.7,
            )
        ax.set_title(channel)
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_json(path: Path, data: Any, indent: Optional[int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    indices = range(INDEX_START, INDEX_END + 1)
    for name, folder in SOURCES.items():
        count = len(list(folder.glob("*.png")))
        logging.info("%s folder: %s (%d PNG files)", name, folder, count)

    per_image: Dict[str, List[Dict[str, Any]]] = {}
    raw_histograms: Dict[str, Dict[str, np.ndarray]] = {}
    for name, folder in SOURCES.items():
        per_image[name], raw_histograms[name] = analyze_source(name, folder, indices)
        logging.info("%s analyzed images: %d", name, len(per_image[name]))

    aggregate = aggregate_stats(per_image)
    histograms = normalize_histograms(raw_histograms)
    distances = histogram_distances(histograms)
    paired = paired_stat_comparison(per_image)
    tail = paired_tail_comparison(per_image)

    outputs = {
        "per_image_stats": OUTPUT_DIR / "per_image_stats.json",
        "aggregate_stats": OUTPUT_DIR / "aggregate_stats.json",
        "aggregate_histograms": OUTPUT_DIR / "aggregate_histograms.json",
        "histogram_overlay": OUTPUT_DIR / "histogram_overlay.png",
        "distribution_distances": OUTPUT_DIR / "distribution_distances.json",
        "paired_comparison": OUTPUT_DIR / "paired_comparison.json",
        "tail_comparison": OUTPUT_DIR / "tail_comparison.json",
    }

    write_json(outputs["per_image_stats"], per_image, indent=None)
    write_json(outputs["aggregate_stats"], aggregate, indent=2)
    write_json(outputs["aggregate_histograms"], histograms, indent=2)
    plot_histograms(histograms, outputs["histogram_overlay"])
    write_json(outputs["distribution_distances"], distances, indent=2)
    write_json(outputs["paired_comparison"], paired, indent=2)
    write_json(outputs["tail_comparison"], tail, indent=2)

    logging.info("Output files:")
    for path in outputs.values():
        logging.info("  %s", path)


if __name__ == "__main__":
    main()
