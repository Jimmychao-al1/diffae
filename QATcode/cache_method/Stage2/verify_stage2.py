"""
驗證 Stage2 輸出的 refined scheduler_config.json（或任何同結構的 Stage1/Stage2 JSON）。

檢查：
- time_order 為 ddim_99_to_0
- blocks 數量為 31、id 為 0..30 各一
- T 與 shared_zones、expanded_mask 長度一致
- shared_zones 為 DDIM timestep 0..T-1 的分割（與 Stage1 相同驗證）
- 每個 block：len(k_per_zone) == len(shared_zones)
- expanded_mask 與由 k_per_zone 重建的 mask 一致，或為其超集（peak repair 只會多開 True）
- expanded_mask[0] == True（step_idx=0 ↔ DDIM i=T-1）
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

from QATcode.cache_method.Stage1.stage1_scheduler import validate_shared_zones_ddim
from QATcode.cache_method.Stage2.stage2_scheduler_adapter import (
    EXPECTED_NUM_BLOCKS,
    TIME_ORDER_EXPECTED,
    rebuild_expanded_mask_from_shared_zones_and_k_per_zone,
    RUNTIME_LAYER_NAMES,
    stage1_block_to_runtime_block,
)


def verify_refined_scheduler_config(cfg: Dict[str, Any], *, require_full_coverage: bool = True) -> None:
    if cfg.get("time_order") != TIME_ORDER_EXPECTED:
        raise ValueError(
            f"time_order must be {TIME_ORDER_EXPECTED!r}, got {cfg.get('time_order')!r}"
        )

    T = int(cfg["T"])
    if T < 2:
        raise ValueError(f"T must be >= 2, got {T}")

    shared = cfg.get("shared_zones")
    if not isinstance(shared, list):
        raise TypeError("shared_zones must be a list")
    validate_shared_zones_ddim(shared, T)
    nz = len(shared)

    blocks = cfg.get("blocks")
    if not isinstance(blocks, list):
        raise TypeError("blocks must be a list")
    if require_full_coverage and len(blocks) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"blocks must have length {EXPECTED_NUM_BLOCKS}, got {len(blocks)}"
        )
    if not require_full_coverage and len(blocks) < 1:
        raise ValueError("blocks must be non-empty when require_full_coverage=False")

    ids = [int(b["id"]) for b in blocks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"block ids must be unique, got {ids}")
    if require_full_coverage:
        ids_sorted = sorted(ids)
        if ids_sorted != list(range(EXPECTED_NUM_BLOCKS)):
            raise ValueError(
                f"block ids must be 0..{EXPECTED_NUM_BLOCKS - 1} exactly once, got {ids_sorted}"
            )

    mapped_runtime_names: List[str] = []
    for b in blocks:
        bid = int(b["id"])
        name = str(b.get("name", ""))
        rt = stage1_block_to_runtime_block(name)
        mapped_runtime_names.append(rt)
        rt_declared = b.get("runtime_name", None)
        if rt_declared is not None and str(rt_declared) != rt:
            raise ValueError(
                f"block id={bid}: runtime_name {rt_declared!r} contradicts name {name!r} -> {rt!r}"
            )

    if len(set(mapped_runtime_names)) != len(mapped_runtime_names):
        raise ValueError(
            "mapped runtime block names from blocks[].name must be unique"
        )
    if require_full_coverage:
        if set(mapped_runtime_names) != set(RUNTIME_LAYER_NAMES):
            missing = sorted(set(RUNTIME_LAYER_NAMES) - set(mapped_runtime_names))
            extra = sorted(set(mapped_runtime_names) - set(RUNTIME_LAYER_NAMES))
            raise ValueError(
                "mapped runtime block set mismatch for full config: "
                f"missing={missing}, extra={extra}"
            )

    for b in blocks:
        bid = int(b["id"])
        em = b.get("expanded_mask")
        if not isinstance(em, list) or len(em) != T:
            raise ValueError(
                f"block id={bid}: expanded_mask must be length T={T}, "
                f"got {len(em) if isinstance(em, list) else type(em)}"
            )
        row = np.asarray(em, dtype=bool)
        if row.shape != (T,):
            raise ValueError(f"block id={bid}: bad expanded_mask shape")
        if not bool(row[0]):
            raise ValueError(
                f"block id={bid}: first step must be full compute: "
                "expanded_mask[0] must be True (step_idx=0 <-> DDIM i=T-1)"
            )

        kz = b.get("k_per_zone")
        if not isinstance(kz, list):
            raise TypeError(f"block id={bid}: k_per_zone must be a list")
        if len(kz) != nz:
            raise ValueError(
                f"block id={bid}: len(k_per_zone)={len(kz)} != len(shared_zones)={nz}"
            )
        for j, kv in enumerate(kz):
            if int(kv) < 1:
                raise ValueError(f"block id={bid}: k_per_zone[{j}] must be >= 1, got {kv}")

        rebuilt = np.asarray(
            rebuild_expanded_mask_from_shared_zones_and_k_per_zone(
                shared, [int(x) for x in kz], T, block_id=bid
            ),
            dtype=bool,
        )
        # Zone refine 後基底為 rebuild(k)；peak 只會把若干 False→True，不可刪掉 rebuild 要求的 True
        if not np.all(row >= rebuilt):
            bad = np.where(~row & rebuilt)[0].tolist()
            raise ValueError(
                f"block id={bid}: expanded_mask must be >= mask from k_per_zone (missing steps {bad[:24]}"
                + (" ..." if len(bad) > 24 else "")
                + ")"
            )


def verify_blockwise_threshold_config_dict(data: Dict[str, Any], *, eps: float = 1e-9) -> None:
    """
    驗證 `build_blockwise_thresholds.py` 產生的 stage2_thresholds_blockwise.json。

    檢查：31 blocks、id 0..30、zone/peak > 0、peak >= ratio * zone（允許浮點誤差）。
    """
    if not isinstance(data, dict):
        raise TypeError("threshold config root must be a dict")
    method = data.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError("threshold config must have non-empty string 'method'")
    ratio_min = float(data.get("peak_over_zone_ratio_min", 0.0))
    if not (ratio_min > 0.0) or math.isnan(ratio_min) or math.isinf(ratio_min):
        raise ValueError(f"peak_over_zone_ratio_min must be finite and > 0, got {ratio_min!r}")

    blocks = data.get("per_block")
    if not isinstance(blocks, list) or len(blocks) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"per_block must be a list of length {EXPECTED_NUM_BLOCKS}, got {len(blocks) if isinstance(blocks, list) else type(blocks)}"
        )

    seen_ids: set[int] = set()
    for entry in blocks:
        if not isinstance(entry, dict):
            raise TypeError("each per_block entry must be a dict")
        bid = int(entry["block_id"])
        if bid < 0 or bid >= EXPECTED_NUM_BLOCKS:
            raise ValueError(f"invalid block_id {bid}")
        if bid in seen_ids:
            raise ValueError(f"duplicate block_id {bid}")
        seen_ids.add(bid)

        rt = str(entry.get("runtime_name", ""))
        if rt != RUNTIME_LAYER_NAMES[bid]:
            raise ValueError(
                f"block_id {bid}: runtime_name must be {RUNTIME_LAYER_NAMES[bid]!r}, got {rt!r}"
            )
        cn = str(entry.get("canonical_name", ""))
        if not cn.startswith("model."):
            raise ValueError(f"block_id {bid}: bad canonical_name {cn!r}")

        zt = float(entry["zone_l1_threshold"])
        pt = float(entry["peak_l1_threshold"])
        if math.isnan(zt) or math.isinf(zt) or zt <= 0.0:
            raise ValueError(f"block_id {bid}: zone_l1_threshold must be finite and > 0, got {zt!r}")
        if math.isnan(pt) or math.isinf(pt) or pt <= 0.0:
            raise ValueError(f"block_id {bid}: peak_l1_threshold must be finite and > 0, got {pt!r}")
        if pt + eps < ratio_min * zt:
            raise ValueError(
                f"block_id {bid}: need peak_l1_threshold >= peak_over_zone_ratio_min * zone_l1_threshold "
                f"({pt} < {ratio_min} * {zt})"
            )

    if seen_ids != set(range(EXPECTED_NUM_BLOCKS)):
        raise ValueError(f"block_id set must be 0..{EXPECTED_NUM_BLOCKS - 1} exactly once, got {sorted(seen_ids)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "config_path",
        type=str,
        nargs="?",
        default=None,
        help="scheduler_config.json (refined or Stage1)",
    )
    ap.add_argument(
        "--threshold-config",
        type=str,
        default=None,
        help="若指定，改為驗證 stage2_thresholds_blockwise.json（blockwise quantile 輸出）",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="允許 partial/ablation scheduler（不強制 31 blocks 全覆蓋）",
    )
    args = ap.parse_args()
    if args.threshold_config:
        p = Path(args.threshold_config)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        verify_blockwise_threshold_config_dict(data)
        print(f"OK (threshold config): {p.resolve()}")
        return
    if not args.config_path:
        ap.error("需要 config_path，或使用 --threshold-config")
    p = Path(args.config_path)
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    verify_refined_scheduler_config(cfg, require_full_coverage=not bool(args.allow_partial))
    print(f"OK: {p.resolve()}")


if __name__ == "__main__":
    main()
