"""
Stage2：將 Stage1 的 scheduler_config.json 轉成 diffusion runtime 可用的 cache_scheduler。

時間軸（與 Stage1 一致，務必與 diffusion/base.py 的 DDIM 迴圈變數 i 對齊）：
- DDIM 採樣迴圈：indices = range(T)[::-1]，故 i 由 T-1 遞減到 0。
- Stage1 expanded_mask[b, step_idx]：step_idx=0 對應「第一步」= DDIM i=T-1；
  step_idx=T-1 對應 DDIM i=0。
- 換算：對 DDIM timestep i（0..T-1），expanded_mask 索引為 (T-1)-i。
- cache_scheduler[runtime_layer] = 需要 **recompute（full compute）** 的 DDIM timestep i 的集合
  （與 diffusion/base.py：「i in cache_scheduler[key] → cached_scheduler=1」一致）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from QATcode.cache_method.Stage1.stage1_scheduler import (
    expand_zone_mask_ddim,
    or_expanded_with_zone_mask,
)

EXPECTED_NUM_BLOCKS = 31
TIME_ORDER_EXPECTED = "ddim_99_to_0"

# 與 diffusion/base.py layer_keys 順序一致：encoder 0..14 → middle → decoder 0..14
RUNTIME_LAYER_NAMES: Tuple[str, ...] = tuple(
    [f"encoder_layer_{i}" for i in range(15)] + ["middle_layer"] + [f"decoder_layer_{i}" for i in range(15)]
)
assert len(RUNTIME_LAYER_NAMES) == EXPECTED_NUM_BLOCKS


def load_stage1_scheduler_config(path: str | Path) -> Dict[str, Any]:
    """讀取 Stage1 scheduler_config.json 並做基本結構檢查。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"scheduler_config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise TypeError("scheduler_config root must be a JSON object")
    return cfg


def validate_stage1_scheduler_config(cfg: Dict[str, Any], *, require_full_coverage: bool = True) -> None:
    """Fail-fast：Stage1/Stage2 scheduler 結構與語意驗證。"""
    if cfg.get("time_order") != TIME_ORDER_EXPECTED:
        raise ValueError(
            f"time_order must be {TIME_ORDER_EXPECTED!r}, got {cfg.get('time_order')!r}"
        )
    T = int(cfg["T"])
    if T < 2:
        raise ValueError(f"T must be >= 2, got {T}")

    blocks = cfg.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("blocks must be a list")
    if require_full_coverage and len(blocks) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"blocks must be a list of length {EXPECTED_NUM_BLOCKS}, got {len(blocks)}"
        )
    if not require_full_coverage and len(blocks) < 1:
        raise ValueError("blocks must be non-empty when require_full_coverage=False")

    ids = [int(b["id"]) for b in blocks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"block ids must be unique, got {ids}")
    if require_full_coverage:
        ids_sorted = sorted(ids)
        if ids_sorted != list(range(EXPECTED_NUM_BLOCKS)):
            raise ValueError(f"block ids must be 0..{EXPECTED_NUM_BLOCKS-1} exactly once, got {ids_sorted}")

    for b in blocks:
        mask = b.get("expanded_mask")
        if not isinstance(mask, list) or len(mask) != T:
            raise ValueError(
                f"block id={b.get('id')}: expanded_mask must have length T={T}, "
                f"got {len(mask) if isinstance(mask, list) else type(mask)}"
            )
        row = np.asarray(mask, dtype=bool)
        if row.shape != (T,):
            raise ValueError(f"block id={b.get('id')}: expanded_mask shape must be ({T},), got {row.shape}")
        if not bool(row[0]):
            raise ValueError(
                f"block id={b.get('id')}: expanded_mask[0] must be True "
                "(step_idx=0 <-> DDIM i=T-1)"
            )

    shared = cfg.get("shared_zones")
    if not isinstance(shared, list) or len(shared) < 1:
        raise ValueError("shared_zones must be a non-empty list")
    nz = len(shared)

    # 與 Stage1 相同：分區不重疊、覆蓋 0..T-1
    from QATcode.cache_method.Stage1.stage1_scheduler import validate_shared_zones_ddim

    validate_shared_zones_ddim(shared, T)

    mapped_runtime_names: List[str] = []
    for b in blocks:
        bid = int(b.get("id"))
        name = str(b.get("name", ""))
        rt = stage1_block_to_runtime_block(name)
        mapped_runtime_names.append(rt)
        rt_declared = b.get("runtime_name", None)
        if rt_declared is not None and str(rt_declared) != rt:
            raise ValueError(
                f"block id={bid}: runtime_name {rt_declared!r} contradicts name {name!r} -> {rt!r}"
            )
        kz = b.get("k_per_zone")
        if not isinstance(kz, list) or len(kz) != nz:
            raise ValueError(
                f"block id={b.get('id')}: k_per_zone length must be len(shared_zones)={nz}, "
                f"got {len(kz) if isinstance(kz, list) else type(kz)}"
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
        row = np.asarray(b["expanded_mask"], dtype=bool)
        if not np.all(row >= rebuilt):
            bad = np.where(~row & rebuilt)[0].tolist()
            raise ValueError(
                f"block id={bid}: expanded_mask must be >= rebuild(shared_zones,k_per_zone), missing steps {bad[:24]}"
                + (" ..." if len(bad) > 24 else "")
            )

    if len(set(mapped_runtime_names)) != len(mapped_runtime_names):
        raise ValueError("mapped runtime block names from blocks[].name must be unique")
    if require_full_coverage:
        if set(mapped_runtime_names) != set(RUNTIME_LAYER_NAMES):
            missing = sorted(set(RUNTIME_LAYER_NAMES) - set(mapped_runtime_names))
            extra = sorted(set(mapped_runtime_names) - set(RUNTIME_LAYER_NAMES))
            raise ValueError(
                "mapped runtime block set mismatch for full config: "
                f"missing={missing}, extra={extra}"
            )


def runtime_block_to_stage1_name(runtime: str) -> str:
    """
    Runtime 名稱 → Stage1 `blocks[].name`（canonical JSON 字串）。

    與 `stage1_block_to_runtime_block` 互為反函數（在 31 層命名空間內）。
    """
    s = runtime.strip()
    m = re.match(r"^encoder_layer_(\d+)$", s)
    if m:
        return f"model.input_blocks.{int(m.group(1))}"
    if s == "middle_layer":
        return "model.middle_block"
    m = re.match(r"^decoder_layer_(\d+)$", s)
    if m:
        return f"model.output_blocks.{int(m.group(1))}"
    raise ValueError(f"unrecognized runtime block name: {runtime!r}")


def stage1_block_to_runtime_block(stage1_name: str) -> str:
    """
    Stage1 block name（JSON）→ diffusion / unet_autoenc 使用的 runtime 名稱。

    - model.input_blocks.i -> encoder_layer_i
    - model.middle_block -> middle_layer
    - model.output_blocks.i -> decoder_layer_i
    """
    s = stage1_name.strip()
    m = re.match(r"^model\.input_blocks\.(\d+)$", s)
    if m:
        return f"encoder_layer_{int(m.group(1))}"
    if s == "model.middle_block":
        return "middle_layer"
    m = re.match(r"^model\.output_blocks\.(\d+)$", s)
    if m:
        return f"decoder_layer_{int(m.group(1))}"
    raise ValueError(f"unrecognized Stage1 block name: {stage1_name!r}")


def expanded_mask_row_to_recompute_ddim_timesteps(expanded_mask_row: List[bool] | np.ndarray, T: int) -> Set[int]:
    """
    單一 block 的 expanded_mask 列（長度 T）→ recompute 的 DDIM timestep i 集合（0..T-1）。

    expanded_mask[step_idx]：step_idx=0 對應 DDIM i=T-1；step_idx=T-1 對應 DDIM i=0。
    True = 該步對該 block 做 full compute（recompute）。
    """
    row = np.asarray(expanded_mask_row, dtype=bool)
    if row.shape != (T,):
        raise ValueError(f"expanded_mask row must have shape ({T},), got {row.shape}")
    out: Set[int] = set()
    for i in range(T):
        step_idx = (T - 1) - i
        if bool(row[step_idx]):
            out.add(i)
    return out


def stage1_mask_to_runtime_cache_scheduler(cfg: Dict[str, Any]) -> Dict[str, Set[int]]:
    """
    由整份 Stage1 config 建出完整 31 層的 cache_scheduler。

    回傳 dict：key 為 encoder_layer_* / middle_layer / decoder_layer_*，
    value 為「該層要 recompute 的 DDIM timestep i」的集合。
    """
    validate_stage1_scheduler_config(cfg)
    T = int(cfg["T"])
    blocks = sorted(cfg["blocks"], key=lambda b: int(b["id"]))
    sched: Dict[str, Set[int]] = {}
    for b in blocks:
        name = str(b["name"])
        rt = stage1_block_to_runtime_block(name)
        if rt in sched:
            raise ValueError(f"duplicate runtime block {rt} from Stage1 name {name}")
        sched[rt] = expanded_mask_row_to_recompute_ddim_timesteps(b["expanded_mask"], T)
    if len(sched) != EXPECTED_NUM_BLOCKS:
        raise ValueError(f"internal error: expected {EXPECTED_NUM_BLOCKS} runtime keys, got {len(sched)}")
    return sched


def cache_scheduler_to_jsonable(sched: Dict[str, Set[int]]) -> Dict[str, List[int]]:
    """將 set 轉成可 json.dump 的 sorted list（便於診斷輸出）。"""
    return {k: sorted(v) for k, v in sorted(sched.items())}


def ddim_timestep_to_step_index(i: int, T: int) -> int:
    """DDIM timestep i（0..T-1）→ expanded_mask 列索引（0=第一步=T-1）。"""
    return (T - 1) - i


def step_index_to_ddim_timestep(step_idx: int, T: int) -> int:
    """expanded_mask 列索引 → DDIM timestep i。"""
    return (T - 1) - step_idx


def rebuild_expanded_mask_from_shared_zones_and_k_per_zone(
    shared_zones: List[Dict[str, Any]],
    k_per_zone: List[int],
    T: int,
    *,
    block_id: int = 0,
) -> List[bool]:
    """
    由 shared_zones + k_per_zone 重建單一 block 的 expanded_mask（長度 T），
    規則與 Stage1 `rebuild_expanded_mask_from_config` / `expand_zone_mask_ddim` 相同。

    step_idx=0 對應 DDIM i=T-1；強制 expanded_mask[0]=True。
    """
    if len(k_per_zone) != len(shared_zones):
        raise ValueError(f"k_per_zone len {len(k_per_zone)} != shared_zones len {len(shared_zones)}")
    row = np.zeros(T, dtype=bool)
    for z, k in zip(shared_zones, k_per_zone):
        ms, _, _ = expand_zone_mask_ddim(int(z["t_start"]), int(z["t_end"]), int(k), T)
        or_expanded_with_zone_mask(row, ms, block_id=block_id, zone_id=int(z["id"]))
    row[0] = True
    return row.tolist()


def config_block_expanded_mask_consistent_with_k(
    expanded_mask: List[bool],
    k_per_zone: List[int],
    shared_zones: List[Dict[str, Any]],
    T: int,
    block_id: int,
) -> bool:
    """驗證 expanded_mask 是否等於由 k_per_zone 重建的結果。"""
    rebuilt = rebuild_expanded_mask_from_shared_zones_and_k_per_zone(
        shared_zones, k_per_zone, T, block_id=block_id
    )
    return list(expanded_mask) == rebuilt
