"""
由 stage2_runtime_diagnostics.json 的 per-block 誤差分布，以**純 quantile** 產生每 block 的
`zone_l1_threshold` / `peak_l1_threshold`。

說明（與 a_L1_L2_cosine similarity 圖的關係）：
- Similarity 圖可用來觀察各 UNet block 之間 error 尺度差異很大。
- 本工具產生的 Stage2 正式 threshold 數值僅來自本診斷檔中 cache-vs-full 的
  `per_block_zone_error` / `per_block_step_error`（baseline vs cache 特徵 L1），
  **不**把 similarity 圖上的數值直接當 threshold。

不使用固定 zone_min / zone_max / peak_min / peak_max；僅
`quantile(分布, q)` + 弱相對約束 `peak >= peak_over_zone_ratio_min * zone`。
若 quantile 結果為 NaN / inf / <=0，直接報錯（不以固定上下界蓋掉）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from QATcode.cache_method.Stage2.stage2_scheduler_adapter import (
    EXPECTED_NUM_BLOCKS,
    RUNTIME_LAYER_NAMES,
    runtime_block_to_stage1_name,
)
from QATcode.cache_method.Stage2.verify_stage2 import verify_blockwise_threshold_config_dict

METHOD_NAME = "blockwise_quantile_v1"


def _finite_values_zone(per_block_zone: Dict[str, Any], rt: str) -> List[float]:
    out: List[float] = []
    zm = per_block_zone.get(rt)
    if not isinstance(zm, dict):
        raise KeyError(f"per_block_zone_error missing or invalid entry for {rt!r}")
    for _zid, st in zm.items():
        if not isinstance(st, dict):
            continue
        v = float(st.get("mean_l1", float("nan")))
        if math.isfinite(v):
            out.append(v)
    return out


def _finite_values_step(per_block_step: Dict[str, Any], rt: str) -> List[float]:
    out: List[float] = []
    sm = per_block_step.get(rt)
    if not isinstance(sm, dict):
        raise KeyError(f"per_block_step_error missing or invalid entry for {rt!r}")
    for _ti, st in sm.items():
        if not isinstance(st, dict):
            continue
        v = float(st.get("l1", float("nan")))
        if math.isfinite(v):
            out.append(v)
    return out


def _quantile_or_raise(vals: List[float], q: float, *, label: str) -> float:
    if not vals:
        raise ValueError(f"{label}: no finite samples to compute quantile")
    if not (0.0 <= q <= 1.0) or math.isnan(q) or math.isinf(q):
        raise ValueError(f"{label}: invalid quantile q={q!r}")
    x = float(np.quantile(np.asarray(vals, dtype=np.float64), q))
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"{label}: quantile({q}) produced non-finite value {x!r}")
    if x <= 0.0:
        raise ValueError(f"{label}: quantile({q}) must be > 0, got {x!r}")
    return x


def build_blockwise_thresholds(
    *,
    diagnostics_path: Path,
    output_path: Path,
    q_zone: float,
    q_peak: float,
    peak_over_zone_ratio_min: float,
) -> Dict[str, Any]:
    if not (peak_over_zone_ratio_min > 0.0) or math.isnan(peak_over_zone_ratio_min):
        raise ValueError(f"peak_over_zone_ratio_min must be > 0, got {peak_over_zone_ratio_min!r}")

    with open(diagnostics_path, "r", encoding="utf-8") as f:
        diag = json.load(f)
    per_block_step = diag.get("per_block_step_error")
    per_block_zone = diag.get("per_block_zone_error")
    if not isinstance(per_block_step, dict) or not isinstance(per_block_zone, dict):
        raise ValueError("diagnostics must contain per_block_step_error and per_block_zone_error objects")

    per_block: List[Dict[str, Any]] = []
    for block_id, rt in enumerate(RUNTIME_LAYER_NAMES):
        zvals = _finite_values_zone(per_block_zone, rt)
        pvals = _finite_values_step(per_block_step, rt)
        zone_thr = _quantile_or_raise(zvals, q_zone, label=f"block {block_id} ({rt}) zone")
        peak_thr = _quantile_or_raise(pvals, q_peak, label=f"block {block_id} ({rt}) peak")
        peak_thr = max(peak_thr, peak_over_zone_ratio_min * zone_thr)
        if math.isnan(peak_thr) or math.isinf(peak_thr) or peak_thr <= 0.0:
            raise ValueError(f"block {block_id} ({rt}): invalid peak_l1_threshold after constraint: {peak_thr!r}")

        canonical = runtime_block_to_stage1_name(rt)
        per_block.append(
            {
                "block_id": block_id,
                "canonical_runtime_block_id": block_id,
                "canonical_name": canonical,
                "runtime_name": rt,
                "num_zone_samples": len(zvals),
                "num_step_samples": len(pvals),
                "zone_l1_threshold": zone_thr,
                "peak_l1_threshold": peak_thr,
            }
        )

    if len(per_block) != EXPECTED_NUM_BLOCKS:
        raise RuntimeError("internal: per_block length mismatch")

    out: Dict[str, Any] = {
        "method": METHOD_NAME,
        "block_identity_semantics": {
            "block_id": "canonical runtime block index",
            "canonical_runtime_block_id": "same as block_id (explicit alias for readability)",
        },
        "source_diagnostics_path": str(diagnostics_path.resolve()),
        "q_zone": float(q_zone),
        "q_peak": float(q_peak),
        "peak_over_zone_ratio_min": float(peak_over_zone_ratio_min),
        "per_block": per_block,
    }
    verify_blockwise_threshold_config_dict(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-block quantile thresholds from Stage2 diagnostics JSON")
    ap.add_argument("--diagnostics", type=str, required=True, help="Path to stage2_runtime_diagnostics.json")
    ap.add_argument("--output", type=str, required=True, help="Output path, e.g. stage2_thresholds_blockwise.json")
    ap.add_argument("--q_zone", type=float, default=0.75, help="Quantile over zone mean_l1 values per block (default 0.75)")
    ap.add_argument("--q_peak", type=float, default=0.95, help="Quantile over per-step l1 values per block (default 0.95)")
    ap.add_argument(
        "--peak_over_zone_ratio_min",
        type=float,
        default=1.5,
        help="Enforce peak_l1 >= this * zone_l1 after quantiles (default 1.5)",
    )
    args = ap.parse_args()
    dp = Path(args.diagnostics)
    op = Path(args.output)
    build_blockwise_thresholds(
        diagnostics_path=dp,
        output_path=op,
        q_zone=float(args.q_zone),
        q_peak=float(args.q_peak),
        peak_over_zone_ratio_min=float(args.peak_over_zone_ratio_min),
    )
    print(f"Wrote {op.resolve()}")


if __name__ == "__main__":
    main()
