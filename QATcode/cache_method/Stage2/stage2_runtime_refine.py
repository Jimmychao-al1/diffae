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
    cache_scheduler_to_jsonable,
    ddim_timestep_to_step_index,
    load_stage1_scheduler_config,
    rebuild_expanded_mask_from_shared_zones_and_k_per_zone,
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
    quant_model = create_float_quantized_model(base_model.ema_model)
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
) -> Dict[str, Any]:
    _configure_stage2_logging()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_all(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    conf = None
    collector: Optional[Stage2ErrorCollector] = None

    try:
        cfg = load_stage1_scheduler_config(scheduler_config_path)
        validate_stage1_scheduler_config(cfg)
        T = int(cfg["T"])
        shared_zones: List[Dict[str, Any]] = cfg["shared_zones"]

        QAT_CONFIG.NUM_DIFFUSION_STEPS = T

        cache_sched_input = stage1_mask_to_runtime_cache_scheduler(cfg)

        base_model = _load_quant_model_for_sampling(
            repo_root=_REPO_ROOT,
            model_path=model_path,
            best_ckpt_path=best_ckpt_path,
            calib_path=calib_path,
            device=device,
        )
        conf = base_model.conf.clone()
        conf.eval_num_images = 1
        sampler = base_model.conf._make_diffusion_conf(T=T).make_sampler()
        latent_sampler = base_model.conf._make_latent_diffusion_conf(T=T).make_sampler()

        g = torch.Generator(device=device)
        g.manual_seed(seed)
        x_T = torch.randn((1, 3, conf.img_size, conf.img_size), generator=g, device=device)
        conf.seed = seed

        collector = Stage2ErrorCollector(T=T, device=device)
        cb = collector.make_cache_debug_callback()

        try:
            conf.cache_debug_collector = cb
            collector.set_run("baseline")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
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
            LOGGER.info("%s", collector.debug_snapshot_line("after_baseline"))

            collector.set_run("cache")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            _run_single_render(
                conf=conf,
                model=base_model.ema_model,
                sampler=sampler,
                latent_sampler=latent_sampler,
                x_T=x_T,
                conds_mean=base_model.conds_mean,
                conds_std=base_model.conds_std,
                cache_scheduler=cache_sched_input,
            )
            LOGGER.info("%s", collector.debug_snapshot_line("after_cache"))
        finally:
            conf.cache_debug_collector = None

        LOGGER.info("%s", collector.debug_snapshot_line("before_compute_diagnostics"))
        diagnostics = collector.compute_diagnostics(shared_zones)
        diagnostics["cache_scheduler_input"] = cache_scheduler_to_jsonable(cache_sched_input)
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

        refined = copy.deepcopy(cfg)
        refined["version"] = "stage2_refined_v1"
        refined["stage2_meta"] = {
            "zone_l1_threshold": zone_l1_threshold,
            "peak_l1_threshold": peak_l1_threshold,
            "seed": seed,
            "threshold_mode": threshold_mode,
            "threshold_config_path": str(Path(threshold_config_path).resolve()) if threshold_config_path else None,
        }

        k_touch: List[Dict[str, Any]] = []
        blocks = sorted(refined["blocks"], key=lambda b: int(b["id"]))

        for b in blocks:
            rt = stage1_block_to_runtime_block(str(b["name"]))
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
                            "runtime": rt,
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
                        "runtime": rt,
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
                    "canonical_name": blockwise_by_id[i]["canonical_name"],
                    "runtime_name": blockwise_by_id[i]["runtime_name"],
                    "zone_l1_threshold": float(blockwise_by_id[i]["zone_l1_threshold"]),
                    "peak_l1_threshold": float(blockwise_by_id[i]["peak_l1_threshold"]),
                }
                for i in range(EXPECTED_NUM_BLOCKS)
            ]

        summary = {
            "zone_k_adjustments": k_touch,
            "peak_mask_adjustments": mask_touch,
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
        if collector is not None:
            collector.clear_storage()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description="Stage2 runtime refine (single-pass)")
    p.add_argument("--scheduler_config", type=str, required=True, help="Stage1 scheduler_config.json")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zone_l1_threshold", type=float, default=0.02)
    p.add_argument("--peak_l1_threshold", type=float, default=0.08)
    p.add_argument("--model_path", type=str, default="checkpoints/ffhq128_autoenc_latent/last.ckpt")
    p.add_argument("--best_ckpt", type=str, default="QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth")
    p.add_argument("--calib", type=str, default="QATcode/quantize_ver2/calibration_diffae.pth")
    p.add_argument(
        "--threshold-config",
        type=str,
        default=None,
        help="Optional: stage2_thresholds_blockwise.json（build_blockwise_thresholds.py）；若省略則用 --zone_l1_threshold / --peak_l1_threshold",
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
    )


if __name__ == "__main__":
    main()
