"""
Sampling entrypoint with optional Stage2 refined cache scheduler.

Goal:
- Keep the same Q-DiffAE sampling pipeline as sample_lora_intmodel_v2.py
- Add explicit Stage2 scheduler loading/validation/mapping
- Baseline path is preserved when --use_cache_scheduler is not enabled
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics import evaluate_fid
from QATcode.cache_method.Stage2.stage2_scheduler_adapter import (
    EXPECTED_NUM_BLOCKS,
    FIRST_INPUT_RUNTIME_BLOCK_NAME,
    RUNTIME_LAYER_NAMES,
    apply_cache_scheduler_runtime_overrides,
    cache_runtime_override_variant_label,
    cache_scheduler_to_jsonable,
    load_stage1_scheduler_config,
    stage1_block_to_runtime_block,
    stage1_mask_to_runtime_cache_scheduler,
    validate_stage1_scheduler_config,
)
from QATcode.cache_method.a_L1_L2_cosine.similarity_calculation import (
    _load_quant_and_ema_from_ckpt,
)
from QATcode.quantize_ver2.sample_lora_intmodel_v2 import (
    CONFIG,
    LOGGER,
    _seed_all,
    create_float_quantized_model,
    load_calibration_data,
    load_diffae_model,
)


DEFAULT_STAGE2_SCHEDULER_JSON = (
    "QATcode/cache_method/Stage2/stage2_output/run_per_baseline/"
    "stage2_refined_scheduler_config.json"
)


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def _setup_environment() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG.GPU_ID
    log_dir = os.path.dirname(CONFIG.LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(CONFIG.LOG_FILE)],
        force=True,
    )


def _validate_required_stage2_fields(
    cfg: Dict[str, Any],
    *,
    cfg_path: Path,
    require_k_per_zone: bool = True,
) -> None:
    required_top = ("T", "shared_zones", "blocks")
    for key in required_top:
        if key not in cfg:
            raise ValueError(f"{cfg_path}: missing required top-level field '{key}'")

    blocks = cfg.get("blocks")
    if not isinstance(blocks, list):
        raise TypeError(f"{cfg_path}: 'blocks' must be a list")
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise TypeError(f"{cfg_path}: blocks[{idx}] must be an object")
        for k in ("name", "expanded_mask"):
            if k not in block:
                raise ValueError(f"{cfg_path}: blocks[{idx}] missing required field '{k}'")
        if require_k_per_zone and "k_per_zone" not in block:
            raise ValueError(f"{cfg_path}: blocks[{idx}] missing required field 'k_per_zone'")


def _parse_force_full_runtime_blocks(s: str) -> List[str]:
    if not s or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _nonnegative_int(s: str) -> int:
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


# ──────────────────────────────────────────────────────────────────────────────
# Run-artifact helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sanitize_name(s: str) -> str:
    """Make a string safe for use as a directory name component."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)


def _compact_run_index_id(start_dt: datetime, scheduler_name: str) -> str:
    """Short id for runs_index.jsonl (no year, MMDDHHMMSS + scheduler slug)."""
    return f"{start_dt.strftime('%m%d%H%M%S')}__{_sanitize_name(scheduler_name)}"


def _fid_index_key(num_images: int) -> str:
    if num_images == 5000:
        return "FID@5K"
    if num_images == 50000:
        return "FID@50K"
    return f"FID@{num_images}"


def _round_fid_index(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return round(float(x), 3)


def _repo_rel_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(p.resolve())


def _get_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def _append_runs_index(index_path: Path, entry: Dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _compute_schedule_stats(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Derive recompute/reuse counts purely from expanded_mask in scheduler JSON on disk.

    Does not include CLI runtime unions (e.g. ``--force-full-prefix-steps``). Prefer
    :func:`_compute_schedule_stats_from_effective_runtime_scheduler` for stats aligned with sampling.
    """
    T = int(cfg["T"])
    blocks: List[Dict[str, Any]] = cfg.get("blocks", [])
    shared_zones: List[Dict[str, Any]] = cfg.get("shared_zones", [])

    # zone_id -> list of DDIM timestep t-values (t=T-1 is first step)
    zone_ts: Dict[int, List[int]] = {}
    for z in shared_zones:
        zid = int(z["id"])
        t_s, t_e = int(z["t_start"]), int(z["t_end"])
        zone_ts[zid] = list(range(t_e, t_s + 1))

    total_full = 0
    total_reuse = 0
    per_block_recompute: Dict[str, int] = {}
    per_block_reuse: Dict[str, int] = {}
    full_compute_blocks_count = 0

    # per zone: aggregate counts across all blocks
    per_zone: Dict[str, Dict[str, Any]] = {
        str(zid): {"full_count": 0, "reuse_count": 0, "num_timesteps_in_zone": len(ts)}
        for zid, ts in zone_ts.items()
    }

    for b in blocks:
        rt = str(b.get("runtime_name") or b.get("name") or f"block_{b.get('id', '?')}")
        mask: List[bool] = b.get("expanded_mask", [])
        # mask index: step_idx = (T-1) - t  (step_idx 0 = first DDIM step = t=T-1)
        n_full = sum(1 for v in mask if v)
        n_reuse = T - n_full
        total_full += n_full
        total_reuse += n_reuse
        per_block_recompute[rt] = n_full
        per_block_reuse[rt] = n_reuse
        if n_full == T:
            full_compute_blocks_count += 1

        for zid, ts_list in zone_ts.items():
            zkey = str(zid)
            for t_val in ts_list:
                si = (T - 1) - t_val  # DDIM timestep t → step_idx
                if 0 <= si < len(mask):
                    if mask[si]:
                        per_zone[zkey]["full_count"] += 1
                    else:
                        per_zone[zkey]["reuse_count"] += 1

    num_cells = T * len(blocks) if blocks else 1
    full_compute_ratio = round(total_full / num_cells, 6)

    return {
        "T": T,
        "num_blocks": len(blocks),
        "total_full_compute_count": total_full,
        "total_cache_reuse_count": total_reuse,
        "full_compute_ratio": full_compute_ratio,
        "full_compute_blocks_count": full_compute_blocks_count,
        "per_block_recompute_count": per_block_recompute,
        "per_block_reuse_count": per_block_reuse,
        "per_block_recompute_ratio": {
            rt: round(cnt / T, 6) for rt, cnt in per_block_recompute.items()
        },
        "per_zone_recompute_stats": per_zone,
    }


def _compute_schedule_stats_from_effective_runtime_scheduler(
    cache_scheduler: Dict[str, Set[int]],
    *,
    T: int,
    shared_zones: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recompute/reuse counts from the **effective** runtime cache_scheduler used in sampling.

    ``cache_scheduler[layer]`` = DDIM timestep indices ``i`` where that layer does full compute
    (same convention as ``diffusion/base.py`` / ``stage1_mask_to_runtime_cache_scheduler`` +
    ``apply_cache_scheduler_runtime_overrides``).
    """
    T = int(T)
    keys = set(cache_scheduler.keys())
    if keys != set(RUNTIME_LAYER_NAMES):
        raise ValueError(
            "effective cache_scheduler keys must match RUNTIME_LAYER_NAMES; "
            f"got missing={sorted(set(RUNTIME_LAYER_NAMES) - keys)} extra={sorted(keys - set(RUNTIME_LAYER_NAMES))}"
        )

    zone_ts: Dict[int, List[int]] = {}
    for z in shared_zones:
        zid = int(z["id"])
        t_s, t_e = int(z["t_start"]), int(z["t_end"])
        zone_ts[zid] = list(range(t_e, t_s + 1))

    total_full = 0
    total_reuse = 0
    per_block_recompute: Dict[str, int] = {}
    per_block_reuse: Dict[str, int] = {}
    full_compute_blocks_count = 0

    per_zone: Dict[str, Dict[str, Any]] = {
        str(zid): {"full_count": 0, "reuse_count": 0, "num_timesteps_in_zone": len(ts)}
        for zid, ts in zone_ts.items()
    }

    for rt in RUNTIME_LAYER_NAMES:
        rec = cache_scheduler[rt]
        n_full = len(rec)
        n_reuse = T - n_full
        total_full += n_full
        total_reuse += n_reuse
        per_block_recompute[rt] = n_full
        per_block_reuse[rt] = n_reuse
        if n_full == T:
            full_compute_blocks_count += 1

        for zid, ts_list in zone_ts.items():
            zkey = str(zid)
            for t_val in ts_list:
                if t_val in rec:
                    per_zone[zkey]["full_count"] += 1
                else:
                    per_zone[zkey]["reuse_count"] += 1

    num_blocks = len(RUNTIME_LAYER_NAMES)
    num_cells = T * num_blocks
    full_compute_ratio = round(total_full / num_cells, 6)

    return {
        "T": T,
        "num_blocks": num_blocks,
        "total_full_compute_count": total_full,
        "total_cache_reuse_count": total_reuse,
        "full_compute_ratio": full_compute_ratio,
        "full_compute_blocks_count": full_compute_blocks_count,
        "per_block_recompute_count": per_block_recompute,
        "per_block_reuse_count": per_block_reuse,
        "per_block_recompute_ratio": {
            rt: round(per_block_recompute[rt] / T, 6) for rt in RUNTIME_LAYER_NAMES
        },
        "per_zone_recompute_stats": per_zone,
    }


def _format_effective_scheduler_fr_grid(
    cache_scheduler: Dict[str, Set[int]],
    *,
    T: int,
) -> Dict[str, Any]:
    """Human-readable F/R grid for the effective runtime scheduler (order A: DDIM i = T-1 .. 0)."""
    T = int(T)
    if set(cache_scheduler.keys()) != set(RUNTIME_LAYER_NAMES):
        raise ValueError(
            "effective cache_scheduler keys must match RUNTIME_LAYER_NAMES for FR grid"
        )

    ddim_order = list(range(T - 1, -1, -1))
    name_w = max(len(x) for x in RUNTIME_LAYER_NAMES)

    by_layer: Dict[str, str] = {}
    aligned_lines: List[str] = []

    for rt in RUNTIME_LAYER_NAMES:
        rec = cache_scheduler[rt]
        # Width-2 cells so F/R columns line up in monospace.
        cells = [f"{('F' if i in rec else 'R'):^2}" for i in ddim_order]
        row = " ".join(cells)
        by_layer[rt] = row
        aligned_lines.append(f"{rt.ljust(name_w)} : {row}")

    return {
        "meta": {
            "source": "runtime_conf_cache_scheduler_after_cli_overrides",
            "timestep_order": "A_ddim_execution",
            "columns_left_to_right": "DDIM timestep i from T-1 down to 0 (first denoising step → last)",
            "ddim_timestep_i_left_to_right": ddim_order,
            "T": T,
            "num_blocks": EXPECTED_NUM_BLOCKS,
            "cell_token": "F=full_compute R=cache_reuse",
        },
        "by_layer": by_layer,
        "aligned_lines": aligned_lines,
        "aligned_text_block": "\n".join(aligned_lines),
    }


def _cfg_snapshot_with_effective_expanded_masks(
    base_cfg: Dict[str, Any],
    effective: Dict[str, Set[int]],
    *,
    T: int,
) -> Dict[str, Any]:
    """Build a scheduler_config-shaped dict with expanded_mask rows from the effective runtime scheduler.

    Inverts the same convention as ``expanded_mask_row_to_recompute_ddim_timesteps`` /
    ``stage1_mask_to_runtime_cache_scheduler``: for each ``step_idx`` in ``0..T-1``,
    DDIM timestep ``i = (T-1) - step_idx`` is full compute iff ``i in effective[runtime_name]``.
    """
    T = int(T)
    if set(effective.keys()) != set(RUNTIME_LAYER_NAMES):
        raise ValueError(
            "effective cache_scheduler keys must match RUNTIME_LAYER_NAMES for snapshot synthesis"
        )

    out = copy.deepcopy(base_cfg)
    for b in out.get("blocks", []):
        name = str(b["name"])
        rt = stage1_block_to_runtime_block(name)
        rec = effective[rt]
        row: List[bool] = []
        for step_idx in range(T):
            i = (T - 1) - step_idx
            row.append(bool(i in rec))
        b["expanded_mask"] = row

    snap_meta: Dict[str, Any] = {
        "source": "effective_runtime_cache_scheduler_after_cli_overrides",
        "expanded_mask_semantics": (
            "step_idx=0 is first DDIM step (t=T-1); True = full compute for this block at that step"
        ),
        "note": (
            "k_per_zone copied from on-disk JSON and may not reconstruct this expanded_mask "
            "when CLI runtime overrides (e.g. --force-full-prefix-steps) apply."
        ),
    }
    prev = out.get("snapshot_meta")
    if isinstance(prev, dict):
        out["snapshot_meta"] = {**prev, **snap_meta}
    else:
        out["snapshot_meta"] = snap_meta
    return out


def _load_stage2_summary_stats(scheduler_json_path: Path) -> Dict[str, Any]:
    """Best-effort: load stage2_refinement_summary.json from same dir as scheduler config.

    Returns {} if file not found or parse error.
    """
    summary_p = scheduler_json_path.parent / "stage2_refinement_summary.json"
    if not summary_p.is_file():
        return {}
    try:
        with open(summary_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        zone_adj: List[Dict[str, Any]] = data.get("zone_k_adjustments", [])
        peak_adj: List[Dict[str, Any]] = data.get("peak_mask_adjustments", [])
        per_zone_adj: Dict[str, int] = {}
        for entry in zone_adj:
            zid = str(entry.get("zone_id", "?"))
            per_zone_adj[zid] = per_zone_adj.get(zid, 0) + 1
        return {
            "zone_adjustments_count": len(zone_adj),
            "peak_adjustments_count": len(peak_adj),
            "per_zone_adjustment_stats": per_zone_adj,
        }
    except Exception as e:
        LOGGER.warning("Could not load stage2_refinement_summary.json: %s", e)
        return {}


def _load_runtime_cache_scheduler(
    *,
    cache_scheduler_json: str,
    num_steps: int,
    force_full_prefix_steps: int = 0,
    force_full_runtime_blocks: Optional[List[str]] = None,
    safety_first_input_block: bool = False,
    allow_missing_k_per_zone: bool = False,
) -> Tuple[Dict[str, Set[int]], Dict[str, Any]]:
    cfg_path = _resolve_repo_path(cache_scheduler_json)
    cfg = load_stage1_scheduler_config(cfg_path)
    _validate_required_stage2_fields(
        cfg,
        cfg_path=cfg_path,
        require_k_per_zone=not allow_missing_k_per_zone,
    )
    validate_stage1_scheduler_config(cfg, require_k_per_zone=not allow_missing_k_per_zone)

    T_cfg = int(cfg["T"])
    if T_cfg != int(num_steps):
        raise ValueError(
            f"Scheduler T mismatch: scheduler T={T_cfg}, sampling num_steps={num_steps}. "
            "Please align --num_steps with JSON['T']."
        )

    blocks = cfg["blocks"]
    if len(blocks) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"Scheduler block count mismatch: expected {EXPECTED_NUM_BLOCKS}, got {len(blocks)}"
        )

    runtime_names_seen = []
    for i, block in enumerate(blocks):
        name = str(block["name"])
        expanded_mask = block["expanded_mask"]
        if not isinstance(expanded_mask, list):
            raise TypeError(f"blocks[{i}] '{name}': expanded_mask must be a list")
        if len(expanded_mask) != T_cfg:
            raise ValueError(
                f"blocks[{i}] '{name}': expanded_mask length={len(expanded_mask)} != T={T_cfg}"
            )
        runtime_name = stage1_block_to_runtime_block(name)
        runtime_names_seen.append(runtime_name)

    # Explicit mapping check: do not trust list order, align by block name mapping.
    if set(runtime_names_seen) != set(RUNTIME_LAYER_NAMES):
        missing = sorted(set(RUNTIME_LAYER_NAMES) - set(runtime_names_seen))
        extra = sorted(set(runtime_names_seen) - set(RUNTIME_LAYER_NAMES))
        raise ValueError(
            "Scheduler block-name mapping mismatch with runtime cacheable blocks. "
            f"missing={missing}, extra={extra}"
        )

    cache_scheduler = stage1_mask_to_runtime_cache_scheduler(
        cfg,
        require_k_per_zone=not allow_missing_k_per_zone,
    )
    if len(cache_scheduler) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"Runtime cache scheduler size mismatch: expected {EXPECTED_NUM_BLOCKS}, got {len(cache_scheduler)}"
        )
    blocks_eff = list(force_full_runtime_blocks or [])
    if safety_first_input_block and FIRST_INPUT_RUNTIME_BLOCK_NAME not in blocks_eff:
        blocks_eff.append(FIRST_INPUT_RUNTIME_BLOCK_NAME)
    cache_effective, override_meta = apply_cache_scheduler_runtime_overrides(
        cache_scheduler,
        T_cfg,
        force_full_prefix_steps=int(force_full_prefix_steps),
        force_full_runtime_blocks=blocks_eff,
    )
    override_meta["variant_label"] = cache_runtime_override_variant_label(
        force_full_prefix_steps=int(force_full_prefix_steps),
        force_full_runtime_blocks=list(force_full_runtime_blocks or []),
        safety_first_input_block=bool(safety_first_input_block),
    )
    override_meta["force_full_runtime_blocks_effective"] = list(blocks_eff)
    LOGGER.info("cache_scheduler runtime overrides: %s", override_meta)
    run_record: Dict[str, Any] = {
        "stage": "fid_sampling",
        "scheduler_config_path": str(cfg_path.resolve()),
        "T": T_cfg,
        "allow_missing_k_per_zone": bool(allow_missing_k_per_zone),
        **override_meta,
        "cache_scheduler_effective_for_sampling": cache_scheduler_to_jsonable(cache_effective),
    }
    return cache_effective, run_record


@torch.no_grad()
def main_sample_with_optional_stage2_scheduler(
    *,
    use_cache_scheduler: bool,
    cache_scheduler_json: str,
    force_full_prefix_steps: int = 0,
    force_full_runtime_blocks: Optional[List[str]] = None,
    safety_first_input_block: bool = False,
    allow_missing_k_per_zone: bool = False,
    # Run-artifact params (all optional; if None, no artifacts are written).
    run_output_dir: Optional[Path] = None,
    scheduler_name: str = "unknown",
    runs_index_path: Optional[Path] = None,
) -> None:
    import datetime as _dt
    import socket as _socket
    import traceback as _tb

    start_dt = _dt.datetime.now()
    start_time_str = start_dt.isoformat(timespec="seconds")
    run_id = f"{start_dt.strftime('%Y%m%dT%H%M%S')}__{_sanitize_name(scheduler_name)}"

    _manifest: Dict[str, Any] = {}
    if run_output_dir is not None:
        run_output_dir = Path(run_output_dir)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        _manifest = {
            "run_id": run_id,
            "status": "running",
            "start_time": start_time_str,
            "end_time": None,
            "duration_sec": None,
            "scheduler_name": scheduler_name,
            "scheduler_config_path": (
                str(_resolve_repo_path(cache_scheduler_json).resolve())
                if use_cache_scheduler
                else None
            ),
            "threshold_config_path": None,
            "num_images": CONFIG.EVAL_SAMPLES,
            "seed": CONFIG.SEED,
            "script_path": str(Path(__file__).resolve()),
            "command_argv": sys.argv[:],
            "git_commit": _get_git_commit(),
            "hostname": _socket.gethostname(),
            "output_dir": str(run_output_dir.resolve()),
        }
        _write_json(run_output_dir / "run_manifest.json", _manifest)

    _score: Optional[float] = None
    runtime_effective_cache_scheduler: Optional[Dict[str, Set[int]]] = None
    try:
        LOGGER.info("=" * 50)
        LOGGER.info("Diff-AE sampling with optional Stage2 scheduler")
        LOGGER.info("=" * 50)
        _seed_all(CONFIG.SEED)

        base_model = load_diffae_model()
        LOGGER.info("✅ Diff-AE 模型載入成功")
        quant_model = create_float_quantized_model(
            base_model.ema_model,
            num_steps=CONFIG.NUM_DIFFUSION_STEPS,
            lora_rank=CONFIG.LORA_RANK,
            mode=CONFIG.MODE,
        )
        quant_model.to(CONFIG.DEVICE)
        quant_model.eval()

        cali_images, cali_t, cali_y = load_calibration_data()
        quant_model.set_first_last_layer_to_8bit()
        quant_model.set_quant_state(CONFIG.QUANT_STATE_WEIGHT, CONFIG.QUANT_STATE_ACT)
        if hasattr(quant_model, "set_runtime_mode"):
            quant_model.set_runtime_mode(mode="train", use_cached_aw=False, clear_cached_aw=True)
        _ = quant_model(
            x=cali_images[:32].to(CONFIG.DEVICE),
            t=cali_t[:32].to(CONFIG.DEVICE),
            cond=cali_y[:32].to(CONFIG.DEVICE),
        )

        ckpt = torch.load(CONFIG.BEST_CKPT_PATH, map_location="cpu", weights_only=False)
        _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)
        if hasattr(base_model.ema_model, "set_runtime_mode"):
            base_model.ema_model.set_runtime_mode(mode="infer", use_cached_aw=True, clear_cached_aw=True)
        LOGGER.info("✅ 量化模型權重載入成功")

        base_model.to(CONFIG.DEVICE)
        base_model.eval()
        base_model.setup()
        base_model.train_dataloader()

        T = CONFIG.NUM_DIFFUSION_STEPS
        sampler = base_model.conf._make_diffusion_conf(T=T).make_sampler()
        latent_sampler = base_model.conf._make_latent_diffusion_conf(T=T).make_sampler()
        conf = base_model.conf.clone()
        conf.eval_num_images = CONFIG.EVAL_SAMPLES

        runtime_override_run_record: Optional[Dict[str, Any]] = None
        if use_cache_scheduler:
            runtime_cache_scheduler, runtime_override_run_record = _load_runtime_cache_scheduler(
                cache_scheduler_json=cache_scheduler_json,
                num_steps=T,
                force_full_prefix_steps=int(force_full_prefix_steps),
                force_full_runtime_blocks=force_full_runtime_blocks,
                safety_first_input_block=bool(safety_first_input_block),
                allow_missing_k_per_zone=bool(allow_missing_k_per_zone),
            )
            runtime_effective_cache_scheduler = runtime_cache_scheduler
            conf.cache_scheduler = runtime_cache_scheduler
            LOGGER.info("✅ Stage2 cache scheduler enabled: %s", _resolve_repo_path(cache_scheduler_json))
            LOGGER.info("scheduler blocks=%d, T=%d", len(runtime_cache_scheduler), T)
            output_dir = f"{conf.generate_dir}_QAT_T{T}_cache_stage2"
        else:
            conf.cache_scheduler = None
            LOGGER.info("🚫 use_cache_scheduler=False, run baseline Q-DiffAE sampling path")
            output_dir = f"{conf.generate_dir}_QAT_T{T}"

        score = evaluate_fid(
            sampler,
            base_model.ema_model,
            conf,
            device=CONFIG.DEVICE,
            train_data=base_model.train_data,
            val_data=base_model.val_data,
            latent_sampler=latent_sampler,
            conds_mean=base_model.conds_mean,
            conds_std=base_model.conds_std,
            remove_cache=False,
            clip_latent_noise=False,
            T=T,
            output_dir=output_dir,
        )
        _score = score
        LOGGER.info("FID@%d T=%d score: %s", CONFIG.EVAL_SAMPLES, T, score)
        LOGGER.info("Output dir: %s", output_dir)

        if use_cache_scheduler and runtime_override_run_record is not None:
            runtime_override_run_record["fid_score"] = score
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            sidecar = out_p / "cache_runtime_overrides_run.json"
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump(runtime_override_run_record, f, indent=2, ensure_ascii=False)
            LOGGER.info("Wrote override run record: %s", sidecar.resolve())

    except Exception as _run_exc:
        if run_output_dir is not None:
            _end_dt = _dt.datetime.now()
            _manifest.update({
                "status": "failed",
                "end_time": _end_dt.isoformat(timespec="seconds"),
                "duration_sec": round((_end_dt - start_dt).total_seconds(), 2),
                "error_message": str(_run_exc),
                "traceback": _tb.format_exc(),
            })
            _write_json(run_output_dir / "run_manifest.json", _manifest)
            if runs_index_path is not None:
                _n = int(CONFIG.EVAL_SAMPLES)
                _fk = _fid_index_key(_n)
                _append_runs_index(
                    Path(runs_index_path),
                    {
                        "rid": _compact_run_index_id(start_dt, scheduler_name),
                        _fk: _round_fid_index(None),
                        "d": start_dt.strftime("%Y%m%d"),
                        "sch": scheduler_name,
                        "r": None,
                        "seed": CONFIG.SEED,
                        "out": _repo_rel_path(Path(run_output_dir)),
                        "sum": None,
                        "st": "failed",
                    },
                )
        raise

    # ── Write run artifacts (success path) ─────────────────────────────────
    if run_output_dir is not None:
        end_dt = _dt.datetime.now()
        end_time_str = end_dt.isoformat(timespec="seconds")
        duration_sec = round((end_dt - start_dt).total_seconds(), 2)

        sched_stats: Dict[str, Any] = {}
        stage2_sidecar_stats: Dict[str, Any] = {}
        if use_cache_scheduler:
            try:
                _cfg_snap = load_stage1_scheduler_config(_resolve_repo_path(cache_scheduler_json))
                if runtime_effective_cache_scheduler is not None:
                    sched_stats = _compute_schedule_stats_from_effective_runtime_scheduler(
                        runtime_effective_cache_scheduler,
                        T=int(_cfg_snap["T"]),
                        shared_zones=list(_cfg_snap.get("shared_zones", [])),
                    )
                else:
                    sched_stats = _compute_schedule_stats(_cfg_snap)
                stage2_sidecar_stats = _load_stage2_summary_stats(
                    _resolve_repo_path(cache_scheduler_json)
                )
                _t_snap = int(_cfg_snap["T"])
                if runtime_effective_cache_scheduler is not None:
                    _snap_out = _cfg_snapshot_with_effective_expanded_masks(
                        _cfg_snap,
                        runtime_effective_cache_scheduler,
                        T=_t_snap,
                    )
                    _write_json(run_output_dir / "scheduler_config.snapshot.json", _snap_out)
                else:
                    _write_json(run_output_dir / "scheduler_config.snapshot.json", _cfg_snap)
            except Exception as _e:
                LOGGER.warning("Could not write schedule stats/snapshot: %s", _e)

        _manifest.update({
            "status": "success",
            "end_time": end_time_str,
            "duration_sec": duration_sec,
        })
        _write_json(run_output_dir / "run_manifest.json", _manifest)

        _fps = int(force_full_prefix_steps)
        summary_obj: Dict[str, Any] = {
            "run_id": run_id,
            "status": "success",
            "scheduler_name": scheduler_name,
            "scheduler_config_path": _manifest.get("scheduler_config_path"),
            "threshold_config_path": None,
            "force-prefix": "T" if _fps > 0 else "F",
            "force-full-prefix-steps": _fps,
            "num_images": CONFIG.EVAL_SAMPLES,
            "seed": CONFIG.SEED,
            "fid_5k": float(_score) if _score is not None else None,
            "full_compute_ratio": sched_stats.get("full_compute_ratio"),
            "total_full_compute_count": sched_stats.get("total_full_compute_count"),
            "total_cache_reuse_count": sched_stats.get("total_cache_reuse_count"),
            "full_compute_blocks_count": sched_stats.get("full_compute_blocks_count"),
            "zone_adjustments_count": stage2_sidecar_stats.get("zone_adjustments_count"),
            "peak_adjustments_count": stage2_sidecar_stats.get("peak_adjustments_count"),
            "start_time": start_time_str,
            "end_time": end_time_str,
            "duration_sec": duration_sec,
            "detail_stats_path": str((run_output_dir / "detail_stats.json").resolve()),
        }
        _write_json(run_output_dir / "summary.json", summary_obj)

        detail_stats_obj: Dict[str, Any] = {
            "run_id": run_id,
            "per_block_recompute_count": sched_stats.get("per_block_recompute_count", {}),
            "per_block_reuse_count": sched_stats.get("per_block_reuse_count", {}),
            "per_zone_recompute_stats": sched_stats.get("per_zone_recompute_stats", {}),
            "per_zone_adjustment_stats": stage2_sidecar_stats.get("per_zone_adjustment_stats", {}),
            "raw_estimation_stats": {
                "full_compute_ratio": sched_stats.get("full_compute_ratio"),
                "total_full_compute_count": sched_stats.get("total_full_compute_count"),
                "total_cache_reuse_count": sched_stats.get("total_cache_reuse_count"),
                "full_compute_blocks_count": sched_stats.get("full_compute_blocks_count"),
                "T": sched_stats.get("T"),
                "num_blocks": sched_stats.get("num_blocks"),
                "per_block_recompute_ratio": sched_stats.get("per_block_recompute_ratio", {}),
            },
        }
        if runtime_effective_cache_scheduler is not None:
            try:
                _t_fr = int(sched_stats.get("T") or CONFIG.NUM_DIFFUSION_STEPS)
                detail_stats_obj["effective_scheduler_fr_grid"] = _format_effective_scheduler_fr_grid(
                    runtime_effective_cache_scheduler,
                    T=_t_fr,
                )
            except Exception as _ge:
                LOGGER.warning("effective_scheduler_fr_grid failed: %s", _ge)
                detail_stats_obj["effective_scheduler_fr_grid"] = {"error": str(_ge)}
        _write_json(run_output_dir / "detail_stats.json", detail_stats_obj)

        if runs_index_path is not None:
            _n = int(CONFIG.EVAL_SAMPLES)
            _fk = _fid_index_key(_n)
            _append_runs_index(
                Path(runs_index_path),
                {
                    "rid": _compact_run_index_id(start_dt, scheduler_name),
                    _fk: _round_fid_index(float(_score) if _score is not None else None),
                    "d": start_dt.strftime("%Y%m%d"),
                    "sch": scheduler_name,
                    "r": sched_stats.get("full_compute_ratio"),
                    "seed": CONFIG.SEED,
                    "out": _repo_rel_path(run_output_dir),
                    "sum": _repo_rel_path(run_output_dir / "summary.json"),
                    "st": "success",
                },
            )
        LOGGER.info("Run artifacts written to: %s", run_output_dir)


def _set_quant_state_from_cli(quant_state: str) -> None:
    if quant_state == "tt":
        CONFIG.QUANT_STATE_WEIGHT = True
        CONFIG.QUANT_STATE_ACT = True
    elif quant_state == "tf":
        CONFIG.QUANT_STATE_WEIGHT = True
        CONFIG.QUANT_STATE_ACT = False
    elif quant_state == "ft":
        CONFIG.QUANT_STATE_WEIGHT = False
        CONFIG.QUANT_STATE_ACT = True
    else:
        CONFIG.QUANT_STATE_WEIGHT = False
        CONFIG.QUANT_STATE_ACT = False


def _set_ckpt_for_mode(mode: str, num_steps: int) -> None:
    if mode == "final":
        CONFIG.BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_final.pth"
        return
    if num_steps == 100:
        CONFIG.BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth"
    else:
        CONFIG.BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_20steps.pth"


def main() -> None:
    epilog = """
Examples (repo root):
  Baseline FID (do not pass --use_cache_scheduler):
    python QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py \\
      --num_steps 100 --eval_samples 5000

  Stage2 refined scheduler + artifacts + JSONL index:
    python QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py \\
      --use_cache_scheduler \\
      --cache_scheduler_json QATcode/cache_method/Stage2/stage2_output/.../stage2_refined_scheduler_config.json \\
      --scheduler-name prefix_15 \\
      --run-output-dir QATcode/cache_method/results/my_run \\
      --runs-index-path QATcode/cache_method/results/my_run/runs_index.jsonl

Pipeline context: Stage2 refine first → QATcode/cache_method/Stage2/stage2ExperimentsGuide.md
Batch FID driver: QATcode/cache_method/start_run/runFidWithStage2Scheduler.sh
Sampling how-to: QATcode/cache_method/start_run/sampleStage2FidGuide.md
"""
    parser = argparse.ArgumentParser(
        description=(
            "FFHQ latent sampling + FID. Optional Stage2 cache scheduler from "
            "stage2_refined_scheduler_config.json (--use_cache_scheduler)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    g_samp = parser.add_argument_group("Sampling / FID")
    g_samp.add_argument("--num_steps", "--n", type=int, default=100, help="DDIM steps T (also selects default QAT ckpt when mode=float)")
    g_samp.add_argument(
        "--eval_samples",
        "--es",
        type=int,
        default=50000,
        help="Number of images for FID (5K vs 50K affects metric key in runs_index)",
    )
    g_samp.add_argument("--seed", type=int, default=0)
    g_samp.add_argument(
        "--mode",
        "--m",
        type=str,
        default="float",
        help="best ckpt :diffae_step6_lora_best_final.pth",
    )

    g_q = parser.add_argument_group("Quantization (passed to CONFIG)")
    g_q.add_argument(
        "--quant-state",
        type=str,
        default="tt",
        choices=["tt", "ff", "tf", "ft"],
        help="Quant flags: tt=weight+act, ff=neither, tf=weight only, ft=act only",
    )

    g_st2 = parser.add_argument_group("Stage2 scheduler (cache)")
    g_st2.add_argument(
        "--use_cache_scheduler",
        action="store_true",
        help="Load Stage2 JSON and run cache path; omit for baseline (no scheduler file)",
    )
    g_st2.add_argument(
        "--cache_scheduler_json",
        type=str,
        default=DEFAULT_STAGE2_SCHEDULER_JSON,
        help="Path to stage2_refined_scheduler_config.json (ignored if --use_cache_scheduler not set)",
    )
    g_st2.add_argument(
        "--scheduler-name",
        type=str,
        default="unknown",
        help="Label for logs, summary.json, runs_index (e.g. baseline, prefix_15)",
    )
    g_st2.add_argument(
        "--allow-missing-k-per-zone",
        action="store_true",
        help="Allow blocks without k_per_zone (still need expanded_mask + shared_zones)",
    )

    g_safe = parser.add_argument_group("Safety / runtime overrides (union with Stage1 mask)")
    g_safe.add_argument(
        "--force-full-prefix-steps",
        type=_nonnegative_int,
        default=0,
        help="Full-compute all layers on first N DDIM timesteps (0=off)",
    )
    g_safe.add_argument(
        "--force-full-runtime-blocks",
        type=str,
        default="",
        help="Comma-separated runtime_name list, full at every timestep (e.g. encoder_layer_0)",
    )
    g_safe.add_argument(
        "--safety-first-input-block",
        action="store_true",
        help=f"Shortcut: add {FIRST_INPUT_RUNTIME_BLOCK_NAME} to --force-full-runtime-blocks",
    )

    g_out = parser.add_argument_group("Logging / run artifacts")
    g_out.add_argument(
        "--log_file",
        "--lf",
        type=str,
        default="QATcode/cache_method/start_run/log/sample_stage2_cache_scheduler.log",
        help="File log path (also see stream handler in _setup_environment)",
    )
    g_out.add_argument(
        "--run-output-dir",
        type=str,
        default=None,
        help="Write run_manifest.json, summary.json, detail_stats.json, scheduler_config.snapshot.json",
    )
    g_out.add_argument(
        "--runs-index-path",
        type=str,
        default=None,
        help=(
            "Append one JSONL line per run: rid,FID@5K|FID@50K|…,d,sch,r,seed,out,sum,st "
            "(paths relative to repo root)"
        ),
    )
    args = parser.parse_args()

    CONFIG.NUM_DIFFUSION_STEPS = int(args.num_steps)
    CONFIG.EVAL_SAMPLES = int(args.eval_samples)
    CONFIG.SEED = int(args.seed)
    CONFIG.LOG_FILE = str(args.log_file)
    _set_quant_state_from_cli(str(args.quant_state))
    _set_ckpt_for_mode(str(args.mode), CONFIG.NUM_DIFFUSION_STEPS)
    _setup_environment()

    LOGGER.info("Using device: %s", CONFIG.DEVICE)
    LOGGER.info("num_steps=%s eval_samples=%s seed=%s", CONFIG.NUM_DIFFUSION_STEPS, CONFIG.EVAL_SAMPLES, CONFIG.SEED)
    LOGGER.info(
        "quant-state=%s -> weight=%s act=%s",
        args.quant_state,
        CONFIG.QUANT_STATE_WEIGHT,
        CONFIG.QUANT_STATE_ACT,
    )
    LOGGER.info("best_ckpt=%s", CONFIG.BEST_CKPT_PATH)
    LOGGER.info("use_cache_scheduler=%s", bool(args.use_cache_scheduler))
    LOGGER.info("cache_scheduler_json=%s", _resolve_repo_path(args.cache_scheduler_json))
    if args.use_cache_scheduler:
        # Print quick schema info for easier debugging before runtime.
        cfg = load_stage1_scheduler_config(_resolve_repo_path(args.cache_scheduler_json))
        LOGGER.info(
            "scheduler meta: T=%s shared_zones=%s blocks=%s",
            cfg.get("T"),
            len(cfg.get("shared_zones", []) if isinstance(cfg.get("shared_zones"), list) else []),
            len(cfg.get("blocks", []) if isinstance(cfg.get("blocks"), list) else []),
        )
        # Friendly fail-fast for malformed JSON before heavy model loading.
        _validate_required_stage2_fields(
            cfg,
            cfg_path=_resolve_repo_path(args.cache_scheduler_json),
            require_k_per_zone=not bool(args.allow_missing_k_per_zone),
        )

    main_sample_with_optional_stage2_scheduler(
        use_cache_scheduler=bool(args.use_cache_scheduler),
        cache_scheduler_json=str(args.cache_scheduler_json),
        force_full_prefix_steps=args.force_full_prefix_steps,
        force_full_runtime_blocks=_parse_force_full_runtime_blocks(str(args.force_full_runtime_blocks)),
        safety_first_input_block=bool(args.safety_first_input_block),
        allow_missing_k_per_zone=bool(args.allow_missing_k_per_zone),
        run_output_dir=Path(args.run_output_dir) if args.run_output_dir else None,
        scheduler_name=str(args.scheduler_name),
        runs_index_path=Path(args.runs_index_path) if args.runs_index_path else None,
    )


if __name__ == "__main__":
    main()
