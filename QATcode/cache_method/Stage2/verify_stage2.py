"""
驗證 Stage2 輸出的 refined scheduler_config.json（或任何同結構的 Stage1/Stage2 JSON）。

檢查：
- T 與 shared_zones、expanded_mask 長度一致
- shared_zones 為 DDIM timestep 0..T-1 的分割（與 Stage1 相同驗證）
- 每個 block 的 expanded_mask 形狀 [T]
- 第一步必須 full compute：expanded_mask[step_idx=0]==True（對應 DDIM i=T-1）
- 每個 k_per_zone 元素 >= 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from QATcode.cache_method.Stage1.stage1_scheduler import validate_shared_zones_ddim


def verify_refined_scheduler_config(cfg: Dict[str, Any]) -> None:
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

    for b in blocks:
        em = b.get("expanded_mask")
        if not isinstance(em, list) or len(em) != T:
            raise ValueError(
                f"block id={b.get('id')}: expanded_mask must be length T={T}, got {len(em) if isinstance(em, list) else type(em)}"
            )
        row = np.asarray(em, dtype=bool)
        if row.shape != (T,):
            raise ValueError(f"block id={b.get('id')}: bad expanded_mask shape")
        if not bool(row[0]):
            raise ValueError(
                f"block id={b.get('id')}: first step must be full compute: "
                "expanded_mask[0] must be True (step_idx=0 <-> DDIM i=T-1)"
            )
        kz = b.get("k_per_zone")
        if not isinstance(kz, list):
            raise TypeError(f"block id={b.get('id')}: k_per_zone must be a list")
        if len(kz) != nz:
            raise ValueError(
                f"block id={b.get('id')}: len(k_per_zone)={len(kz)} != len(shared_zones)={nz}"
            )
        for j, kv in enumerate(kz):
            if int(kv) < 1:
                raise ValueError(f"block id={b.get('id')}: k_per_zone[{j}] must be >= 1, got {kv}")


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
