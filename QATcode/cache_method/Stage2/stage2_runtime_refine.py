"""
Stage2 主流程：載入 Stage1 scheduler_config.json → baseline 與 cache 各跑一次 DDIM →
特徵誤差診斷 → zone（更新 k 並重建 mask）→ peak 修 mask → 輸出 refined JSON。

時間軸（expanded_mask）：
- step_idx=0 對應第一步 DDIM i=T-1；step_idx=T-1 對應 i=0。
- peak repair：expanded_mask[(T-1)-i] 設為 True；was_reuse = 該步在修復前 mask 為 False。
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from renderer import render_uncondition

from QATcode.cache_method.a_L1_L2_cosine.similarity_calculation import _load_quant_and_ema_from_ckpt
from QATcode.cache_method.Stage2.stage2_error_collector import (
    Stage2ErrorCollector,
    aggregate_per_timestep,
)
from QATcode.cache_method.Stage2.stage2_scheduler_adapter import (
    EXPECTED_NUM_BLOCKS,
    FIRST_INPUT_RUNTIME_BLOCK_NAME,
    apply_cache_scheduler_runtime_overrides,
    cache_runtime_override_variant_label,
    cache_scheduler_to_jsonable,
    ddim_timestep_to_step_index,
    load_stage1_scheduler_config,
    rebuild_expanded_mask_from_shared_zones_and_k_per_zone,
    runtime_name_to_block_id,
    stage1_block_to_runtime_block,
    stage1_mask_to_runtime_cache_scheduler,
    validate_stage1_scheduler_config,
)
from QATcode.cache_method.Stage2.verify_stage2 import verify_blockwise_threshold_config_dict
from QATcode.quantize_ver2.sample_lora_intmodel_v2 import (
    CONFIG as QAT_CONFIG,
    create_float_quantized_model,
    load_calibration_data,
    load_diffae_model,
)
from experiment import LitModel

_STAGE2_DIR = Path(__file__).resolve().parent
_STAGE2_LOG_FILE = _STAGE2_DIR / "stage2_runtime_refine.log"
_LOG_FMT = logging.Formatter("%(asctime)s [%(levelname)s] [Stage2] %(message)s")


def _configure_stage2_logging() -> None:
    """Console + append to QATcode/cache_method/Stage2/stage2_runtime_refine.log"""
    lg = logging.getLogger("Stage2RuntimeRefine")
    if lg.handlers:
        return
    lg.setLevel(logging.INFO)
    h_err = logging.StreamHandler(sys.stderr)
    h_err.setFormatter(_LOG_FMT)
    lg.addHandler(h_err)
    h_file = logging.FileHandler(_STAGE2_LOG_FILE, mode="a", encoding="utf-8")
    h_file.setFormatter(_LOG_FMT)
    lg.addHandler(h_file)
    lg.propagate = False


LOGGER = logging.getLogger("Stage2RuntimeRefine")


def _seed_all(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_quant_model_for_sampling(
    *,
    repo_root: Path,
    model_path: str,
    best_ckpt_path: str,
    calib_path: str,
    device: torch.device,
) -> LitModel:
    """與正式 sampling 一致：對 base_model.ema_model 做 Quant 包裝。"""
    mp = model_path if os.path.isabs(model_path) else str(repo_root / model_path)
    bp = best_ckpt_path if os.path.isabs(best_ckpt_path) else str(repo_root / best_ckpt_path)
    cp = calib_path if os.path.isabs(calib_path) else str(repo_root / calib_path)
    QAT_CONFIG.CALIB_DATA_PATH = cp

    base_model: LitModel = load_diffae_model(mp)
    quant_model = create_float_quantized_model(
        base_model.ema_model,
        num_steps=QAT_CONFIG.NUM_DIFFUSION_STEPS,
    )
    quant_model.to(device)

    cali_images, cali_t, cali_y = load_calibration_data()
    quant_model.set_quant_state(True, True)
    if hasattr(quant_model, "set_runtime_mode"):
        quant_model.set_runtime_mode(mode="train", use_cached_aw=False, clear_cached_aw=True)

    with torch.no_grad():
        _ = quant_model(
            x=cali_images[:32].to(device),
            t=cali_t[:32].to(device),
            cond=cali_y[:32].to(device),
        )

    ckpt = torch.load(bp, map_location="cpu", weights_only=False)
    _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)

    if hasattr(base_model.ema_model, "set_runtime_mode"):
        base_model.ema_model.set_runtime_mode(mode="infer", use_cached_aw=True, clear_cached_aw=True)

    base_model.to(device)
    base_model.eval()
    base_model.setup()
    try:
        base_model.train_dataloader()
    except Exception as e:
        LOGGER.warning("train_dataloader() skipped: %s", e)
    return base_model


def _run_single_render(
    *,
    conf,
    model: torch.nn.Module,
    sampler,
    latent_sampler,
    x_T: torch.Tensor,
    conds_mean: torch.Tensor,
    conds_std: torch.Tensor,
    cache_scheduler: Optional[Dict[str, Set[int]]],
) -> None:
    c = conf.clone()
    if cache_scheduler is None:
        c.cache_scheduler = None
    else:
        c.cache_scheduler = cache_scheduler
    # deepcopy(conf) 會深拷貝 bound method 的 __self__，變成另一個 Stage2ErrorCollector 在收資料；
    # 外層持有的 collector 仍為空。必須沿用原 conf 上的同一個 callback。
    cb = getattr(conf, "cache_debug_collector", None)
    if cb is not None:
        c.cache_debug_collector = cb
    with torch.inference_mode():
        _ = render_uncondition(
            conf=c,
            model=model,
            x_T=x_T,
            sampler=sampler,
            latent_sampler=latent_sampler,
            conds_mean=conds_mean,
            conds_std=conds_std,
            clip_latent_noise=False,
        )


def _aggregate_step_metrics_inplace(
    agg: Dict[str, Dict[str, Dict[str, float]]],
    per_block_step_error: Dict[str, Dict[str, Dict[str, float]]],
    *,
    batch_size: int,
) -> None:
    w = float(batch_size)
    for rt, steps in per_block_step_error.items():
        rt_acc = agg.setdefault(rt, {})
        for ti_str, m in steps.items():
            slot = rt_acc.setdefault(
                ti_str,
                {"sum_l1": 0.0, "sum_l2_sq": 0.0, "sum_cos": 0.0, "weight": 0.0},
            )
            l1 = float(m["l1"])
            l2 = float(m["l2"])
            cos = float(m["cosine"])
            slot["sum_l1"] += l1 * w
            slot["sum_l2_sq"] += (l2 * l2) * w
            slot["sum_cos"] += cos * w
            slot["weight"] += w


def _finalize_per_block_step_error(
    agg: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rt, steps in agg.items():
        row: Dict[str, Dict[str, float]] = {}
        for ti_str, slot in steps.items():
            w = float(slot["weight"])
            if w <= 0.0:
                continue
            row[ti_str] = {
                "l1": float(slot["sum_l1"] / w),
                "l2": float(math.sqrt(max(slot["sum_l2_sq"] / w, 0.0))),
                "cosine": float(slot["sum_cos"] / w),
            }
        out[rt] = row
    return out


def _compute_per_block_zone_error(
    per_block_step_error: Dict[str, Dict[str, Dict[str, float]]],
    shared_zones: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    per_block_zone_error: Dict[str, Dict[str, Dict[str, Any]]] = {}
    zone_ts: Dict[int, List[int]] = {}
    for z in shared_zones:
        zid = int(z["id"])
        ts, te = int(z["t_start"]), int(z["t_end"])
        zone_ts[zid] = list(range(te, ts + 1))

    for rt, steps in per_block_step_error.items():
        per_block_zone_error[rt] = {}
        for zid, ts in zone_ts.items():
            zs_l1: List[float] = []
            zs_l2: List[float] = []
            zs_cos: List[float] = []
            for ddim_i in ts:
                st = steps.get(str(ddim_i))
                if st is None:
                    continue
                zs_l1.append(float(st["l1"]))
                zs_l2.append(float(st["l2"]))
                zs_cos.append(float(st["cosine"]))
            if not zs_l1:
                per_block_zone_error[rt][str(zid)] = {
                    "mean_l1": float("nan"),
                    "mean_l2": float("nan"),
                    "mean_cosine": float("nan"),
                    "num_steps": len(ts),
                    "num_compared_in_zone": 0,
                }
            else:
                per_block_zone_error[rt][str(zid)] = {
                    "mean_l1": float(np.mean(zs_l1)),
                    "mean_l2": float(np.mean(zs_l2)),
                    "mean_cosine": float(np.mean(zs_cos)),
                    "num_steps": len(ts),
                    "num_compared_in_zone": len(zs_l1),
                }
    return per_block_zone_error


def _build_diagnostics_from_aggregated_steps(
    *,
    per_block_step_error: Dict[str, Dict[str, Dict[str, float]]],
    shared_zones: List[Dict[str, Any]],
    T: int,
) -> Dict[str, Any]:
    per_block_zone_error = _compute_per_block_zone_error(per_block_step_error, shared_zones)
    all_l1: List[float] = []
    all_l2: List[float] = []
    all_cos: List[float] = []
    for steps in per_block_step_error.values():
        for m in steps.values():
            all_l1.append(float(m["l1"]))
            all_l2.append(float(m["l2"]))
            all_cos.append(float(m["cosine"]))
    global_summary = {
        "mean_l1": float(np.mean(all_l1)) if all_l1 else None,
        "mean_l2": float(np.mean(all_l2)) if all_l2 else None,
        "mean_cosine": float(np.mean(all_cos)) if all_cos else None,
        "num_entries": len(all_l1),
        "note": "含 reuse 步：cache 側為 cached_data 讀出的 tensor，與 baseline 全算比較。",
    }
    return {
        "per_block_step_error": per_block_step_error,
        "per_block_zone_error": per_block_zone_error,
        "global_summary": global_summary,
        "T": int(T),
        "time_axis_note": (
            "step_idx 0..T-1：0=第一步(DDIM i=T-1)，T-1=最後一步(DDIM i=0)；"
            "per_block_step_error 的 key 為字串化的 DDIM timestep i"
        ),
    }


def _load_blockwise_threshold_config(
    path: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """讀取 blockwise threshold JSON；回傳 (runtime_name -> entry, block_id -> entry, root doc)。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"threshold config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    verify_blockwise_threshold_config_dict(data)
    by_runtime: Dict[str, Dict[str, Any]] = {}
    by_id: Dict[int, Dict[str, Any]] = {}
    for entry in data["per_block"]:
        bid = int(entry["block_id"])
        rt = str(entry["runtime_name"])
        if rt in by_runtime:
            raise ValueError(f"duplicate runtime_name in threshold config: {rt}")
        by_runtime[rt] = entry
        by_id[bid] = entry
    if len(by_runtime) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"threshold config must contain exactly {EXPECTED_NUM_BLOCKS} runtime_name entries, got {len(by_runtime)}"
        )
    if set(by_id.keys()) != set(range(EXPECTED_NUM_BLOCKS)):
        raise ValueError(
            f"threshold config must contain exactly block_id 0..{EXPECTED_NUM_BLOCKS - 1}, got {sorted(by_id.keys())}"
        )
    return by_runtime, by_id, data


def _parse_force_full_runtime_blocks(s: Optional[str]) -> List[str]:
    if not s or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _nonnegative_int(s: str) -> int:
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


def _cache_runtime_override_contract(*, override_active: bool) -> Dict[str, str]:
    return {
        "must_reapply_same_runtime_overrides_for_sampling": (
            "true" if override_active else "false"
        ),
        "refined_json_expanded_mask_is_algorithm_only": "true",
        "diagnostics_cache_pass_scheduler_field": "cache_scheduler_effective_for_cache_pass",
        "interpretation": (
            "Safety overrides apply only to the cache diagnostic render pass (runtime union). "
            "stage2_refined_scheduler_config.json masks are from Stage2 refine rules unless you merge policy elsewhere."
        ),
    }


def _json_safe(obj: Any) -> Any:
    import math

    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(x) for x in obj)
    if isinstance(obj, (np.floating,)):
        return _json_safe(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def run_stage2_refine(
    *,
    scheduler_config_path: str,
    output_dir: str,
    seed: int = 0,
    zone_l1_threshold: float = 0.02,
    peak_l1_threshold: float = 0.08,
    model_path: str = "checkpoints/ffhq128_autoenc_latent/last.ckpt",
    best_ckpt_path: str = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth",
    calib_path: str = "QATcode/quantize_ver2/calibration_diffae.pth",
    device: Optional[torch.device] = None,
    threshold_config_path: Optional[str] = None,
    force_full_prefix_steps: int = 0,
    force_full_runtime_blocks: Optional[List[str]] = None,
    safety_first_input_block: bool = False,
    eval_num_images: int = 4,
    eval_chunk_size: Optional[int] = None,
) -> Dict[str, Any]:
    _configure_stage2_logging()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_all(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    conf = None
    try:
        cfg = load_stage1_scheduler_config(scheduler_config_path)
        validate_stage1_scheduler_config(cfg)
        T = int(cfg["T"])
        shared_zones: List[Dict[str, Any]] = cfg["shared_zones"]

        QAT_CONFIG.NUM_DIFFUSION_STEPS = T

        cache_sched_stage1 = stage1_mask_to_runtime_cache_scheduler(cfg)
        blocks_eff = list(force_full_runtime_blocks or [])
        if safety_first_input_block and FIRST_INPUT_RUNTIME_BLOCK_NAME not in blocks_eff:
            blocks_eff.append(FIRST_INPUT_RUNTIME_BLOCK_NAME)
        cache_sched_effective, override_meta = apply_cache_scheduler_runtime_overrides(
            cache_sched_stage1,
            T,
            force_full_prefix_steps=int(force_full_prefix_steps),
            force_full_runtime_blocks=blocks_eff,
        )
        override_meta["variant_label"] = cache_runtime_override_variant_label(
            force_full_prefix_steps=int(force_full_prefix_steps),
            force_full_runtime_blocks=list(force_full_runtime_blocks or []),
            safety_first_input_block=bool(safety_first_input_block),
        )
        override_meta["force_full_runtime_blocks_effective"] = list(blocks_eff)
        override_meta["safety_first_input_block"] = bool(safety_first_input_block)
        LOGGER.info(
            "cache runtime overrides | variant=%s | prefix_steps=%s | forced_blocks=%s",
            override_meta.get("variant_label"),
            override_meta.get("force_full_prefix_steps"),
            override_meta.get("force_full_runtime_blocks_effective"),
        )
        _ov_active = bool(int(force_full_prefix_steps) > 0 or blocks_eff)
        _contract = _cache_runtime_override_contract(override_active=_ov_active)

        base_model = _load_quant_model_for_sampling(
            repo_root=_REPO_ROOT,
            model_path=model_path,
            best_ckpt_path=best_ckpt_path,
            calib_path=calib_path,
            device=device,
        )
        conf = base_model.conf.clone()
        total_eval_images = int(eval_num_images)
        if total_eval_images < 1:
            raise ValueError(f"eval_num_images must be >= 1, got {total_eval_images}")
        chunk_size = int(eval_chunk_size) if eval_chunk_size is not None else min(4, total_eval_images)
        if chunk_size < 1:
            raise ValueError(f"eval_chunk_size must be >= 1, got {chunk_size}")
        chunk_size = min(chunk_size, total_eval_images)
        conf.eval_num_images = chunk_size
        sampler = base_model.conf._make_diffusion_conf(T=T).make_sampler()
        latent_sampler = base_model.conf._make_latent_diffusion_conf(T=T).make_sampler()

        g = torch.Generator(device=device)
        g.manual_seed(seed)
        conf.seed = seed

        agg_step_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        done = 0
        chunk_idx = 0
        while done < total_eval_images:
            bsz = min(chunk_size, total_eval_images - done)
            x_T = torch.randn((bsz, 3, conf.img_size, conf.img_size), generator=g, device=device)
            collector = Stage2ErrorCollector(T=T, device=device)
            cb = collector.make_cache_debug_callback()
            try:
                conf.cache_debug_collector = cb
                collector.set_run("baseline")
                _seed_all(seed + chunk_idx)
                _run_single_render(
                    conf=conf,
                    model=base_model.ema_model,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                    x_T=x_T,
                    conds_mean=base_model.conds_mean,
                    conds_std=base_model.conds_std,
                    cache_scheduler=None,
                )
                LOGGER.info("%s", collector.debug_snapshot_line(f"after_baseline_chunk_{chunk_idx}"))

                collector.set_run("cache")
                _seed_all(seed + chunk_idx)
                _run_single_render(
                    conf=conf,
                    model=base_model.ema_model,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                    x_T=x_T,
                    conds_mean=base_model.conds_mean,
                    conds_std=base_model.conds_std,
                    cache_scheduler=cache_sched_effective,
                )
                LOGGER.info("%s", collector.debug_snapshot_line(f"after_cache_chunk_{chunk_idx}"))
                chunk_diag = collector.compute_diagnostics(shared_zones)
                _aggregate_step_metrics_inplace(
                    agg_step_metrics,
                    chunk_diag["per_block_step_error"],
                    batch_size=bsz,
                )
            finally:
                conf.cache_debug_collector = None
                collector.clear_storage()
                del collector

            done += bsz
            chunk_idx += 1

        per_block_step_agg = _finalize_per_block_step_error(agg_step_metrics)
        diagnostics = _build_diagnostics_from_aggregated_steps(
            per_block_step_error=per_block_step_agg,
            shared_zones=shared_zones,
            T=T,
        )
        diagnostics["cache_scheduler_input"] = cache_scheduler_to_jsonable(cache_sched_stage1)
        diagnostics["cache_scheduler_effective_for_cache_pass"] = cache_scheduler_to_jsonable(
            cache_sched_effective
        )
        diagnostics["cache_scheduler_runtime_overrides"] = dict(override_meta)
        diagnostics["cache_runtime_override_contract"] = dict(_contract)
        diagnostics["scheduler_config_path"] = str(Path(scheduler_config_path).resolve())

        per_block_step = diagnostics["per_block_step_error"]
        per_block_zone = diagnostics["per_block_zone_error"]
        per_t = aggregate_per_timestep(per_block_step)

        blockwise_by_runtime: Optional[Dict[str, Dict[str, Any]]] = None
        blockwise_by_id: Optional[Dict[int, Dict[str, Any]]] = None
        threshold_mode = "global"
        threshold_meta_diag: Dict[str, Any] = {
            "threshold_mode": "global",
            "global_zone_l1": zone_l1_threshold,
            "global_peak_l1": peak_l1_threshold,
            "note": (
                "Single global zone/peak thresholds (CLI). "
                "Per-block quantile thresholds 請用 build_blockwise_thresholds.py 產生 JSON 後以 --threshold-config 指定。"
            ),
        }
        if threshold_config_path:
            blockwise_by_runtime, blockwise_by_id, tc_doc = _load_blockwise_threshold_config(threshold_config_path)
            threshold_mode = "blockwise_quantile"
            threshold_meta_diag = {
                "threshold_mode": threshold_mode,
                "threshold_config_path": str(Path(threshold_config_path).resolve()),
                "method": tc_doc.get("method"),
                "source_diagnostics_path": tc_doc.get("source_diagnostics_path"),
                "q_zone": tc_doc.get("q_zone"),
                "q_peak": tc_doc.get("q_peak"),
                "peak_over_zone_ratio_min": tc_doc.get("peak_over_zone_ratio_min"),
                "note": (
                    "各 block 的 zone/peak threshold 來自診斷內 per-block 分布的 quantile（見 threshold_config）。"
                    "Similarity 圖僅能說明 block 間尺度差異；正式數值以 cache-vs-full diagnostics 為準。"
                ),
            }
        diagnostics["stage2_threshold_meta"] = threshold_meta_diag
        diagnostics["block_identity_semantics"] = {
            "scheduler_local_block_id": "scheduler-local id from scheduler_config blocks[].id",
            "canonical_runtime_block_id": "canonical runtime index from runtime_name",
        }

        refined = copy.deepcopy(cfg)
        refined["version"] = "stage2_refined_v1"
        refined["stage2_meta"] = {
            "zone_l1_threshold": zone_l1_threshold,
            "peak_l1_threshold": peak_l1_threshold,
            "seed": seed,
            "threshold_mode": threshold_mode,
            "threshold_config_path": str(Path(threshold_config_path).resolve()) if threshold_config_path else None,
            "cache_runtime_overrides": {
                "variant_label": override_meta.get("variant_label"),
                "force_full_prefix_steps": int(force_full_prefix_steps),
                "force_full_runtime_blocks_effective": list(blocks_eff),
                "safety_first_input_block": bool(safety_first_input_block),
                "first_input_runtime_block_name": FIRST_INPUT_RUNTIME_BLOCK_NAME,
                **_contract,
            },
        }

        k_touch: List[Dict[str, Any]] = []
        blocks = sorted(refined["blocks"], key=lambda b: int(b["id"]))

        for b in blocks:
            rt = stage1_block_to_runtime_block(str(b["name"]))
            runtime_bid = runtime_name_to_block_id(rt)
            if blockwise_by_runtime is not None and rt not in blockwise_by_runtime:
                raise RuntimeError(f"threshold config missing runtime_name {rt} (block id={b['id']})")
            zone_thr_used = (
                float(blockwise_by_runtime[rt]["zone_l1_threshold"])
                if blockwise_by_runtime is not None
                else zone_l1_threshold
            )
            kz = [int(x) for x in b["k_per_zone"]]
            for z in shared_zones:
                zid = int(z["id"])
                st = per_block_zone.get(rt, {}).get(str(zid), {})
                ml1 = float(st.get("mean_l1", 0.0))
                if not math.isnan(ml1) and ml1 > zone_thr_used:
                    if zid < 0 or zid >= len(kz):
                        raise RuntimeError(f"block {b['id']}: bad zone id {zid}")
                    old = kz[zid]
                    kz[zid] = max(1, old - 1)
                    k_touch.append(
                        {
                            "block_id": b["id"],
                            "scheduler_local_block_id": int(b["id"]),
                            "runtime_name": rt,
                            "canonical_runtime_block_id": int(runtime_bid),
                            "zone_id": zid,
                            "k_before": old,
                            "k_after": kz[zid],
                            "threshold_mode": threshold_mode,
                            "zone_l1_threshold_used": zone_thr_used,
                        }
                    )
            b["k_per_zone"] = kz

        for b in blocks:
            bid = int(b["id"])
            b["expanded_mask"] = rebuild_expanded_mask_from_shared_zones_and_k_per_zone(
                shared_zones,
                [int(x) for x in b["k_per_zone"]],
                T,
                block_id=bid,
            )

        mask_touch: List[Dict[str, Any]] = []
        for b in blocks:
            rt = stage1_block_to_runtime_block(str(b["name"]))
            runtime_bid = runtime_name_to_block_id(rt)
            peak_thr_used = (
                float(blockwise_by_runtime[rt]["peak_l1_threshold"])
                if blockwise_by_runtime is not None
                else peak_l1_threshold
            )
            row = list(b["expanded_mask"])
            for ti_str, m in per_block_step.get(rt, {}).items():
                ddim_i = int(ti_str)
                if float(m["l1"]) <= peak_thr_used:
                    continue
                si = ddim_timestep_to_step_index(ddim_i, T)
                was_reuse = not bool(row[si])
                row[si] = True
                mask_touch.append(
                    {
                        "block_id": b["id"],
                        "scheduler_local_block_id": int(b["id"]),
                        "runtime_name": rt,
                        "canonical_runtime_block_id": int(runtime_bid),
                        "ddim_timestep": ddim_i,
                        "step_index": si,
                        "was_reuse_before_peak_repair": was_reuse,
                        "expanded_mask_after": True,
                        "threshold_mode": threshold_mode,
                        "peak_l1_threshold_used": peak_thr_used,
                    }
                )
            b["expanded_mask"] = row

        if T >= 1:
            for b in blocks:
                rt = stage1_block_to_runtime_block(str(b["name"]))
                runtime_bid = runtime_name_to_block_id(rt)
                em = b["expanded_mask"]
                if not bool(em[0]):
                    em[0] = True
                    peak_thr_used = (
                        float(blockwise_by_runtime[rt]["peak_l1_threshold"])
                        if blockwise_by_runtime is not None
                        else peak_l1_threshold
                    )
                    mask_touch.append(
                        {
                            "block_id": b["id"],
                            "scheduler_local_block_id": int(b["id"]),
                            "runtime_name": rt,
                            "canonical_runtime_block_id": int(runtime_bid),
                            "note": "enforce first step full compute (step_idx=0 -> DDIM i=T-1)",
                            "expanded_mask_after": True,
                            "threshold_mode": threshold_mode,
                            "peak_l1_threshold_used": peak_thr_used,
                        }
                    )

        refined_cache_sched = stage1_mask_to_runtime_cache_scheduler(refined)
        diagnostics["refined_cache_scheduler"] = cache_scheduler_to_jsonable(refined_cache_sched)

        per_block_thr_summary: Optional[List[Dict[str, Any]]] = None
        if blockwise_by_id is not None:
            per_block_thr_summary = [
                {
                    "block_id": int(blockwise_by_id[i]["block_id"]),
                    "canonical_runtime_block_id": int(blockwise_by_id[i].get("canonical_runtime_block_id", blockwise_by_id[i]["block_id"])),
                    "canonical_name": blockwise_by_id[i]["canonical_name"],
                    "runtime_name": blockwise_by_id[i]["runtime_name"],
                    "zone_l1_threshold": float(blockwise_by_id[i]["zone_l1_threshold"]),
                    "peak_l1_threshold": float(blockwise_by_id[i]["peak_l1_threshold"]),
                }
                for i in range(EXPECTED_NUM_BLOCKS)
            ]

        summary = {
            "cache_runtime_overrides": {
                "variant_label": override_meta.get("variant_label"),
                "force_full_prefix_steps": int(force_full_prefix_steps),
                "force_full_runtime_blocks_effective": list(blocks_eff),
                "safety_first_input_block": bool(safety_first_input_block),
                "first_input_runtime_block_name": FIRST_INPUT_RUNTIME_BLOCK_NAME,
                **_contract,
                "note": (
                    "Diagnostics / refinement used cache pass scheduler = cache_scheduler_effective_for_cache_pass "
                    "(Stage1 expanded + optional runtime unions). Refined JSON masks follow Stage2 algorithm only."
                ),
            },
            "zone_k_adjustments": k_touch,
            "peak_mask_adjustments": mask_touch,
            "block_identity_semantics": {
                "block_id": "backward-compatible alias of scheduler_local_block_id",
                "scheduler_local_block_id": "id in scheduler JSON; do not interpret as canonical runtime index",
                "canonical_runtime_block_id": "canonical runtime index matching runtime_name",
            },
            "threshold_mode": threshold_mode,
            "global_thresholds": {"zone_l1": zone_l1_threshold, "peak_l1": peak_l1_threshold},
            "per_block_thresholds": per_block_thr_summary,
            "thresholds": {"zone_l1": zone_l1_threshold, "peak_l1": peak_l1_threshold},
            "aggregate_per_ddim_timestep_l1": per_t,
        }

        with open(out / "stage2_runtime_diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(diagnostics), f, indent=2, ensure_ascii=False)
        with open(out / "stage2_refined_scheduler_config.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(refined), f, indent=2, ensure_ascii=False)
        with open(out / "stage2_refinement_summary.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)

        with open(out / "cache_runtime_overrides_run.json", "w", encoding="utf-8") as f:
            json.dump(
                _json_safe(
                    {
                        "stage": "stage2_runtime_refine",
                        "scheduler_config_path": str(Path(scheduler_config_path).resolve()),
                        "T": T,
                        "seed": seed,
                        **dict(override_meta),
                        "cache_runtime_override_contract": _contract,
                    }
                ),
                f,
                indent=2,
                ensure_ascii=False,
            )

        LOGGER.info("寫入 %s", out)

        return {
            "output_dir": str(out),
            "diagnostics": diagnostics,
            "summary": summary,
            "refined_config": refined,
        }
    finally:
        if conf is not None:
            conf.cache_debug_collector = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    epilog = """
Typical two-pass pipeline (from repo root):
  1) Pass 1 — omit --threshold-config; use global --zone_l1_threshold / --peak_l1_threshold
     to write stage2_runtime_diagnostics.json under --output_dir.
  2) build_blockwise_thresholds.py --diagnostics <that>/stage2_runtime_diagnostics.json ...
  3) Pass 2 — same --scheduler_config, new --output_dir, add --threshold-config <blockwise json>.

See QATcode/cache_method/Stage2/stage2ExperimentsGuide.md and README.md section 「如何執行」.
"""
    p = argparse.ArgumentParser(
        description="Stage2 runtime refine (single-pass). Run twice for global then blockwise thresholds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    g_in = p.add_argument_group("Stage1 input / output")
    g_in.add_argument("--scheduler_config", type=str, required=True, help="Stage1 scheduler_config.json")
    g_in.add_argument("--output_dir", type=str, required=True)
    g_in.add_argument("--seed", type=int, default=0)

    g_thr = p.add_argument_group("Threshold (global unless --threshold-config)")
    g_thr.add_argument("--zone_l1_threshold", type=float, default=0.02)
    g_thr.add_argument("--peak_l1_threshold", type=float, default=0.08)
    g_thr.add_argument(
        "--threshold-config",
        type=str,
        default=None,
        help="Optional: stage2_thresholds_blockwise.json（build_blockwise_thresholds.py）；若省略則用 --zone_l1_threshold / --peak_l1_threshold",
    )

    g_model = p.add_argument_group("Model / calibration (paths relative to repo root unless absolute)")
    g_model.add_argument("--model_path", type=str, default="checkpoints/ffhq128_autoenc_latent/last.ckpt")
    g_model.add_argument("--best_ckpt", type=str, default="QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth")
    g_model.add_argument("--calib", type=str, default="QATcode/quantize_ver2/calibration_diffae.pth")

    g_eval = p.add_argument_group("Diagnostics eval (Stage2 cache vs full)")
    g_eval.add_argument(
        "--eval-num-images",
        type=int,
        default=4,
        help="Number of images used for Stage2 diagnostics.",
    )
    g_eval.add_argument(
        "--eval-chunk-size",
        type=_nonnegative_int,
        default=0,
        help="Chunk size for eval images (0=auto=min(4, eval_num_images)); lower value reduces CPU RAM peak.",
    )

    g_safe = p.add_argument_group("Safety overrides (diagnostic cache pass only; see README)")
    g_safe.add_argument(
        "--force-full-prefix-steps",
        type=_nonnegative_int,
        default=0,
        help="Safety experiment: all runtime layers full-compute on first N DDIM timesteps (i=T-1..T-N); 0=off",
    )
    g_safe.add_argument(
        "--force-full-runtime-blocks",
        type=str,
        default="",
        help="Comma-separated runtime_name list (e.g. encoder_layer_0) forced full-compute at all timesteps; empty=off",
    )
    g_safe.add_argument(
        "--safety-first-input-block",
        action="store_true",
        help=f"Shortcut: add {FIRST_INPUT_RUNTIME_BLOCK_NAME} to forced full-compute blocks (canonical first input block)",
    )
    args = p.parse_args()
    _configure_stage2_logging()

    LOGGER.info(
        "----- Stage2 run start | log_file=%s | scheduler_config=%s | output_dir=%s | seed=%s -----",
        _STAGE2_LOG_FILE,
        args.scheduler_config,
        args.output_dir,
        args.seed,
    )
    run_stage2_refine(
        scheduler_config_path=args.scheduler_config,
        output_dir=args.output_dir,
        seed=args.seed,
        zone_l1_threshold=args.zone_l1_threshold,
        peak_l1_threshold=args.peak_l1_threshold,
        model_path=args.model_path,
        best_ckpt_path=args.best_ckpt,
        calib_path=args.calib,
        threshold_config_path=args.threshold_config,
        force_full_prefix_steps=args.force_full_prefix_steps,
        force_full_runtime_blocks=_parse_force_full_runtime_blocks(args.force_full_runtime_blocks),
        safety_first_input_block=bool(args.safety_first_input_block),
        eval_num_images=int(args.eval_num_images),
        eval_chunk_size=(None if int(args.eval_chunk_size) <= 0 else int(args.eval_chunk_size)),
    )


if __name__ == "__main__":
    main()
