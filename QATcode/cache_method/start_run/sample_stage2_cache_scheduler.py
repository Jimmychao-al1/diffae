"""
Sampling entrypoint with optional Stage2 refined cache scheduler.

Goal:
- Keep the same Q-DiffAE sampling pipeline as sample_lora_intmodel_v2.py
- Add explicit Stage2 scheduler loading/validation/mapping
- Baseline path is preserved when --use_cache_scheduler is not enabled
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics import evaluate_fid
from QATcode.cache_method.Stage2.stage2_scheduler_adapter import (
    EXPECTED_NUM_BLOCKS,
    RUNTIME_LAYER_NAMES,
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


def _validate_required_stage2_fields(cfg: Dict[str, Any], *, cfg_path: Path) -> None:
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
        for k in ("name", "k_per_zone", "expanded_mask"):
            if k not in block:
                raise ValueError(f"{cfg_path}: blocks[{idx}] missing required field '{k}'")


def _load_runtime_cache_scheduler(
    *,
    cache_scheduler_json: str,
    num_steps: int,
) -> Dict[str, Set[int]]:
    cfg_path = _resolve_repo_path(cache_scheduler_json)
    cfg = load_stage1_scheduler_config(cfg_path)
    _validate_required_stage2_fields(cfg, cfg_path=cfg_path)
    validate_stage1_scheduler_config(cfg)

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

    cache_scheduler = stage1_mask_to_runtime_cache_scheduler(cfg)
    if len(cache_scheduler) != EXPECTED_NUM_BLOCKS:
        raise ValueError(
            f"Runtime cache scheduler size mismatch: expected {EXPECTED_NUM_BLOCKS}, got {len(cache_scheduler)}"
        )
    return cache_scheduler


@torch.no_grad()
def main_sample_with_optional_stage2_scheduler(
    *,
    use_cache_scheduler: bool,
    cache_scheduler_json: str,
) -> None:
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

    if use_cache_scheduler:
        runtime_cache_scheduler = _load_runtime_cache_scheduler(
            cache_scheduler_json=cache_scheduler_json,
            num_steps=T,
        )
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
    LOGGER.info("FID@%d T=%d score: %s", CONFIG.EVAL_SAMPLES, T, score)
    LOGGER.info("Output dir: %s", output_dir)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", "--n", type=int, default=100)
    parser.add_argument("--eval_samples", "--es", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", "--m", type=str, default="float", choices=["float", "final"])
    parser.add_argument(
        "--cache_scheduler_json",
        type=str,
        default=DEFAULT_STAGE2_SCHEDULER_JSON,
        help="Path to Stage2 refined scheduler JSON",
    )
    parser.add_argument(
        "--use_cache_scheduler",
        action="store_true",
        help="Enable Stage2 refined scheduler; when disabled runs baseline sampling path",
    )
    parser.add_argument(
        "--quant-state",
        type=str,
        default="tt",
        choices=["tt", "ff", "tf", "ft"],
        help="Quant state mapping: tt/ff/tf/ft -> set_quant_state(weight, act)",
    )
    parser.add_argument(
        "--log_file",
        "--lf",
        type=str,
        default="QATcode/cache_method/start_run/log/sample_stage2_cache_scheduler.log",
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
        _validate_required_stage2_fields(cfg, cfg_path=_resolve_repo_path(args.cache_scheduler_json))

    main_sample_with_optional_stage2_scheduler(
        use_cache_scheduler=bool(args.use_cache_scheduler),
        cache_scheduler_json=str(args.cache_scheduler_json),
    )


if __name__ == "__main__":
    main()
