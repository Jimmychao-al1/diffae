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
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from QATcode.cache_method.Stage1.stage1_scheduler import (
    expand_zone_mask_ddim,
    or_expanded_with_zone_mask,
)

EXPECTED_NUM_BLOCKS = 31
TIME_ORDER_EXPECTED = "ddim_99_to_0"

# 與 `stage1_block_to_runtime_block("model.input_blocks.0")` 一致：UNet 第一個可快取 encoder 子模組
# （接在輸入卷積之後的第一個 residual block stack；對齊 diffusion/base.py layer_keys 的 encoder_layer_0）
FIRST_INPUT_RUNTIME_BLOCK_NAME = "encoder_layer_0"

# 與 diffusion/base.py layer_keys 順序一致：encoder 0..14 → middle → decoder 0..14
RUNTIME_LAYER_NAMES: Tuple[str, ...] = tuple(
    [f"encoder_layer_{i}" for i in range(15)]
    + ["middle_layer"]
    + [f"decoder_layer_{i}" for i in range(15)]
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


def validate_stage1_scheduler_config(
    cfg: Dict[str, Any],
    *,
    require_full_coverage: bool = True,
    require_k_per_zone: bool = True,
) -> None:
    """Fail-fast：Stage1/Stage2 scheduler 結構與語意驗證。

    require_k_per_zone=False：僅供「只依 expanded_mask 做 runtime cache_scheduler」的載入路徑；
    不檢查 k_per_zone 長度與 rebuild 一致性（仍要求 expanded_mask 合法）。
    """
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
            raise ValueError(
                f"block ids must be 0..{EXPECTED_NUM_BLOCKS-1} exactly once, got {ids_sorted}"
            )

    for b in blocks:
        mask = b.get("expanded_mask")
        if not isinstance(mask, list) or len(mask) != T:
            raise ValueError(
                f"block id={b.get('id')}: expanded_mask must have length T={T}, "
                f"got {len(mask) if isinstance(mask, list) else type(mask)}"
            )
        row = np.asarray(mask, dtype=bool)
        if row.shape != (T,):
            raise ValueError(
                f"block id={b.get('id')}: expanded_mask shape must be ({T},), got {row.shape}"
            )
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
        runtime_bid = runtime_name_to_block_id(rt)
        mapped_runtime_names.append(rt)
        rt_declared = b.get("runtime_name", None)
        if rt_declared is not None and str(rt_declared) != rt:
            raise ValueError(
                f"block id={bid}: runtime_name {rt_declared!r} contradicts name {name!r} -> {rt!r}"
            )
        runtime_bid_declared = b.get("canonical_runtime_block_id", None)
        if runtime_bid_declared is not None and int(runtime_bid_declared) != runtime_bid:
            raise ValueError(
                f"block id={bid}: canonical_runtime_block_id {runtime_bid_declared!r} contradicts name {name!r} -> {runtime_bid}"
            )
        local_bid_declared = b.get("scheduler_local_block_id", None)
        if local_bid_declared is not None and int(local_bid_declared) != bid:
            raise ValueError(
                f"block id={bid}: scheduler_local_block_id {local_bid_declared!r} must equal id"
            )
        if require_k_per_zone:
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


def runtime_name_to_block_id(runtime: str) -> int:
    """Canonical runtime block index from runtime block name."""
    s = runtime.strip()
    try:
        return RUNTIME_LAYER_NAMES.index(s)
    except ValueError as e:
        raise ValueError(f"unrecognized runtime block name: {runtime!r}") from e


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


def expanded_mask_row_to_recompute_ddim_timesteps(
    expanded_mask_row: List[bool] | np.ndarray, T: int
) -> Set[int]:
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


def stage1_mask_to_runtime_cache_scheduler(
    cfg: Dict[str, Any],
    *,
    require_k_per_zone: bool = True,
) -> Dict[str, Set[int]]:
    """
    由整份 Stage1 config 建出完整 31 層的 cache_scheduler。

    回傳 dict：key 為 encoder_layer_* / middle_layer / decoder_layer_*，
    value 為「該層要 recompute 的 DDIM timestep i」的集合。
    """
    validate_stage1_scheduler_config(cfg, require_k_per_zone=require_k_per_zone)
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
        raise ValueError(
            f"internal error: expected {EXPECTED_NUM_BLOCKS} runtime keys, got {len(sched)}"
        )
    return sched


def cache_scheduler_to_jsonable(sched: Dict[str, Set[int]]) -> Dict[str, List[int]]:
    """將 set 轉成可 json.dump 的 sorted list（便於診斷輸出）。"""
    return {k: sorted(v) for k, v in sorted(sched.items())}


def prefix_ddim_timesteps_first_n(T: int, n: int) -> Set[int]:
    """
    採樣「前 N 個」DDIM timestep i（與 diffusion/base.py 迴圈 indices 順序一致：先 i=T-1）：
    i ∈ {T-1, T-2, ..., T-N}（共 N 個；N 超過 T 時只取前 T 個）。
    """
    if int(n) < 0:
        raise ValueError(f"prefix steps N must be >= 0, got {n!r}")
    if n <= 0:
        return set()
    T = int(T)
    n_eff = min(int(n), T)
    return set(range(T - n_eff, T))


def apply_cache_scheduler_runtime_overrides(
    sched: Dict[str, Set[int]],
    T: int,
    *,
    force_full_prefix_steps: int = 0,
    force_full_runtime_blocks: Optional[List[str]] = None,
) -> Tuple[Dict[str, Set[int]], Dict[str, Any]]:
    """
    在已由 Stage1 展開的 runtime cache_scheduler 上套用實驗用保守約束（union 更多 recompute timesteps）。

    cache_scheduler[key] = 該層在 DDIM timestep i 需要 recompute（full) 的集合（見 diffusion/base.py）。

    - force_full_prefix_steps: 所有 runtime 層在前 N 個 DDIM 步皆強制 full（union prefix timesteps）。
    - force_full_runtime_blocks: 列出的 runtime_name 在所有 timestep 強制 full（union 0..T-1）。
    """
    T = int(T)
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if int(force_full_prefix_steps) < 0:
        raise ValueError(f"force_full_prefix_steps must be >= 0, got {force_full_prefix_steps!r}")
    if set(sched.keys()) != set(RUNTIME_LAYER_NAMES):
        raise ValueError(
            "cache_scheduler keys must match full runtime inventory; "
            f"got {sorted(sched.keys())[:8]}... (len={len(sched)})"
        )

    force_full_runtime_blocks = list(force_full_runtime_blocks or [])
    for name in force_full_runtime_blocks:
        if name not in RUNTIME_LAYER_NAMES:
            raise ValueError(
                f"unknown runtime block in force_full_runtime_blocks: {name!r}; "
                f"expected one of known encoder/middle/decoder names"
            )

    out: Dict[str, Set[int]] = {k: set(v) for k, v in sched.items()}
    prefix_ts = prefix_ddim_timesteps_first_n(T, force_full_prefix_steps)
    all_ts = set(range(T))

    for k in out:
        out[k] |= prefix_ts
    for name in force_full_runtime_blocks:
        out[name] |= all_ts

    meta: Dict[str, Any] = {
        "force_full_prefix_steps": int(force_full_prefix_steps),
        "prefix_ddim_timesteps": sorted(prefix_ts),
        "force_full_runtime_blocks": list(force_full_runtime_blocks),
        "first_input_runtime_block_name": FIRST_INPUT_RUNTIME_BLOCK_NAME,
        "note": (
            "Overrides are unions on top of Stage1-expanded recompute sets; "
            "they do not replace the Stage1 scheduler JSON."
        ),
    }
    return out, meta


def cache_runtime_override_variant_label(
    *,
    force_full_prefix_steps: int,
    force_full_runtime_blocks: List[str],
    safety_first_input_block: bool,
) -> str:
    """Human-readable experiment bucket for logging / JSON (not used for logic)."""
    prefix = int(force_full_prefix_steps)
    blocks = list(force_full_runtime_blocks or [])
    if safety_first_input_block and FIRST_INPUT_RUNTIME_BLOCK_NAME not in blocks:
        blocks = blocks + [FIRST_INPUT_RUNTIME_BLOCK_NAME]
    has_p = prefix > 0
    has_first = FIRST_INPUT_RUNTIME_BLOCK_NAME in blocks
    extra = [b for b in blocks if b != FIRST_INPUT_RUNTIME_BLOCK_NAME]
    if not has_p and not blocks and not safety_first_input_block:
        return "baseline"
    if has_p and not extra and not has_first:
        return "prefix_only"
    if has_first and not extra and not has_p:
        return "first_input_only"
    if has_p and has_first and not extra:
        return "combined"
    return "custom"


def ddim_timestep_to_step_index(i: int, T: int) -> int:
    """DDIM timestep i（0..T-1）→ expanded_mask 列索引（0=第一步=T-1）。"""
    return (T - 1) - i


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
        raise ValueError(
            f"k_per_zone len {len(k_per_zone)} != shared_zones len {len(shared_zones)}"
        )
    row = np.zeros(T, dtype=bool)
    for z, k in zip(shared_zones, k_per_zone):
        ms, _, _ = expand_zone_mask_ddim(int(z["t_start"]), int(z["t_end"]), int(k), T)
        or_expanded_with_zone_mask(row, ms, block_id=block_id, zone_id=int(z["id"]))
    row[0] = True
    return row.tolist()
