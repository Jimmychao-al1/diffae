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
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from QATcode.cache_method.Stage1.stage1_scheduler import validate_shared_zones_ddim
from QATcode.cache_method.Stage2.stage2_scheduler_adapter import (
    EXPECTED_NUM_BLOCKS,
    TIME_ORDER_EXPECTED,
    rebuild_expanded_mask_from_shared_zones_and_k_per_zone,
)


def verify_refined_scheduler_config(cfg: Dict[str, Any]) -> None:
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
    if len(blocks) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"blocks must have length {EXPECTED_NUM_BLOCKS}, got {len(blocks)}"
        )

    ids = sorted(int(b["id"]) for b in blocks)
    if ids != list(range(EXPECTED_NUM_BLOCKS)):
        raise ValueError(
            f"block ids must be 0..{EXPECTED_NUM_BLOCKS - 1} exactly once, got {ids}"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path", type=str, help="scheduler_config.json (refined or Stage1)")
    args = ap.parse_args()
    p = Path(args.config_path)
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    verify_refined_scheduler_config(cfg)
    print(f"OK: {p.resolve()}")


if __name__ == "__main__":
    main()
