"""
掃描 Stage 1 sweep 輸出目錄，讀取每個 scheduler_config.json，
彙整為 stage1_sweep_summary CSV。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_sweep_dir_name(name: str) -> Dict[str, Optional[str]]:
    """Parse sweep_K16_sw2_lam0.25_kmax4 style directory names."""
    out: Dict[str, Optional[str]] = {
        "K": None,
        "smooth_window": None,
        "lambda": None,
        "k_max": None,
    }
    if not name.startswith("sweep_"):
        return out
    rest = name[len("sweep_") :]
    for part in rest.split("_"):
        if part.startswith("K") and part[1:].isdigit():
            out["K"] = part[1:]
        elif part.startswith("sw") and part[2:].isdigit():
            out["smooth_window"] = part[2:]
        elif part.startswith("lam"):
            out["lambda"] = part[3:]
        elif part.startswith("kmax") and part[4:].isdigit():
            out["k_max"] = part[4:]
    return out


def _load_scheduler_row(
    sweep_dir: Path,
    *,
    run_id_prefix: str,
) -> Dict[str, Any]:
    cfg_path = sweep_dir / "scheduler_config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    params = cfg.get("stage1_baseline_params", {})
    parsed = _parse_sweep_dir_name(sweep_dir.name)

    K = int(params.get("K_change_points", parsed["K"] or 0))
    sw = int(params.get("smooth_window", parsed["smooth_window"] or 0))
    lam = float(params.get("lambda", parsed["lambda"] or 0.0))
    k_max = int(params.get("k_max", parsed["k_max"] or 0))

    zones = cfg.get("shared_zones", [])
    num_zones = len(zones)
    avg_zone_length = (
        float(np.mean([int(z["length"]) for z in zones])) if zones else 0.0
    )

    masks = np.array([b["expanded_mask"] for b in cfg["blocks"]], dtype=bool)
    total_recompute_ratio = float(masks.mean() * 100.0)

    lam_str = f"{lam:.2f}"
    run_id = f"{run_id_prefix}K{K}_sw{sw}_lam{lam_str}"

    return {
        "run_id": run_id,
        "K": K,
        "smooth_window": sw,
        "lambda": lam,
        "k_max": k_max,
        "num_zones": num_zones,
        "total_recompute_ratio": f"{total_recompute_ratio:.1f}%",
        "avg_zone_length": f"{avg_zone_length:.1f}",
        "schedule_json_path": str(cfg_path.resolve()),
    }


def summarize_sweep(
    base_out: Path,
    *,
    output_csv: Path,
    run_id_prefix: str = "",
    pattern: str = "sweep_*",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sweep_dirs = sorted(
        d for d in base_out.glob(pattern) if (d / "scheduler_config.json").is_file()
    )
    if not sweep_dirs:
        raise FileNotFoundError(
            f"No sweep directories with scheduler_config.json under {base_out}"
        )

    for sweep_dir in sweep_dirs:
        rows.append(
            _load_scheduler_row(sweep_dir, run_id_prefix=run_id_prefix)
        )

    rows.sort(key=lambda r: (r["K"], r["smooth_window"], r["lambda"]))

    fieldnames = [
        "run_id",
        "K",
        "smooth_window",
        "lambda",
        "k_max",
        "num_zones",
        "total_recompute_ratio",
        "avg_zone_length",
        "schedule_json_path",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage-1 sweep into CSV")
    parser.add_argument(
        "--base_out",
        type=str,
        default="QATcode/cache_method/Stage1/stage1_output_fp",
        help="Stage1 sweep output root",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <base_out>/stage1_sweep_summary_fp.csv)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="fp_",
        help="run_id prefix (e.g. fp_)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="sweep_*",
        help="Glob pattern for sweep subdirectories",
    )
    args = parser.parse_args()

    base_out = Path(args.base_out)
    if not base_out.is_absolute():
        base_out = _REPO_ROOT / base_out

    output_csv = (
        Path(args.output)
        if args.output
        else base_out / "stage1_sweep_summary_fp.csv"
    )
    if not output_csv.is_absolute():
        output_csv = _REPO_ROOT / output_csv

    rows = summarize_sweep(
        base_out,
        output_csv=output_csv,
        run_id_prefix=str(args.prefix),
        pattern=str(args.pattern),
    )
    print(f"Wrote {len(rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
