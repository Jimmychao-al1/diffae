"""
Activation input distribution analysis (T=100 default).

Goal:
- Analyze module forward *input* activations for Conv2d/Linear during DDIM sampling.
- Per-mode stats under models/<MODE>/ (BASELINE = original Diff-AE ema, no QAT ckpt; FF/FT/TT = same w+lora quant graph).
- Pairwise deltas/ratios under comparisons/<a>_vs_<b>/ plus plots under plots/pairwise/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(".")
sys.path.append("./model")

from QATcode.quantize_ver2.baseline_quant_analysis.analysis_utils import (
    TimestepActivationAccumulator,
    calc_delta_and_ratio,
    match_any_pattern,
    parse_module_types,
    parse_target_patterns,
    quant_state_from_mode,
    safe_layer_name,
)
from QATcode.quantize_ver2.baseline_quant_analysis.pred_xstart_quantile_analysis import (
    CONFIG,
    _compute_latent_cond,
    _load_quant_and_ema_from_ckpt,
    _make_noise_banks,
    _seed_all,
    _setup_logging,
    create_float_quantized_model,
    load_calibration_data,
    load_diffae_model,
)
from QATcode.quantize_ver2.quant_layer_v2 import QuantModule
from QATcode.quantize_ver2.quant_model_lora_v2 import (
    INT_QuantModule_DiffAE_LoRA,
    QuantModule_DiffAE_LoRA,
)


LOGGER = logging.getLogger("activation_distribution_analysis")


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _load_baseline_lit_model(*, model_path: str, device: torch.device) -> Tuple[Any, nn.Module]:
    """Original Diff-AE LitModel + float ema (no QAT / no LoRA ckpt merge)."""
    base_model = load_diffae_model(model_path)
    base_model.to(device)
    base_model.eval()
    base_model.setup()
    base_model.train_dataloader()
    return base_model, base_model.ema_model


def _parse_mode_token(tok: str) -> str:
    t = tok.strip().lower()
    if t == "baseline":
        return "BASELINE"
    return t.upper()


def _parse_compare_pairs(s: str) -> List[Tuple[str, str, str]]:
    """
    Parse e.g. 'ff_vs_ft,baseline_vs_ft,baseline_vs_tt' ->
    list of (folder_name_lowercase, mode_a, mode_b).
    """
    out: List[Tuple[str, str, str]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        p = part.lower()
        if "_vs_" not in p:
            LOGGER.warning("skip invalid compare pair (expected a_vs_b): %s", part)
            continue
        a, b = p.split("_vs_", 1)
        ma, mb = _parse_mode_token(a), _parse_mode_token(b)
        out.append((p, ma, mb))
    return out


def _resolve_run_dirs(root: Path, T: int, seed: int, images_mode: str) -> Dict[str, Path]:
    """Align with pred_xstart_results: T_<T>/<images_mode>/seed<N>/{models,comparisons,plots}."""
    base = root / f"T_{T}" / images_mode / f"seed{seed}"
    return {
        "base": base,
        "models": base / "models",
        "comparisons": base / "comparisons",
        "plots_pairwise": base / "plots" / "pairwise",
        "plots_summary": base / "plots" / "summary",
        "LOG": root / "logs",
    }


def _ensure_activation_legacy_symlinks(out_root: Path, models_root: Path, comparisons_root: Path) -> None:
    """Backward compat: seed dir/FF -> models/FF, seed dir/FF_vs_FT -> comparisons/ff_vs_ft."""
    legacy_models = [("FF", "FF"), ("FT", "FT"), ("TT", "TT"), ("BASELINE", "BASELINE")]
    for legacy_name, sub in legacy_models:
        src = models_root / sub
        dst = out_root / legacy_name
        if not src.exists() or dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(f"models/{sub}", target_is_directory=True)
        except OSError:
            LOGGER.warning("Could not symlink %s -> models/%s", dst, sub)

    cmp_src = comparisons_root / "ff_vs_ft"
    cmp_dst = out_root / "FF_vs_FT"
    if cmp_src.exists() and not cmp_dst.exists() and not cmp_dst.is_symlink():
        try:
            cmp_dst.symlink_to("comparisons/ff_vs_ft", target_is_directory=True)
        except OSError:
            LOGGER.warning("Could not symlink FF_vs_FT -> comparisons/ff_vs_ft")


def _to_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    if isinstance(x, tuple):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(x, torch.Tensor):
        return _json_safe(x.detach().cpu().numpy())
    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    return x


def _mean_safe(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    arr = np.array(vals, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _effect_summary_from_metrics(
    *,
    mean_q999_ratio: float,
    mean_std_ratio: float,
    mean_abs_max_ratio: float,
    mean_zero_ratio_delta: float,
) -> str:
    # Lightweight heuristic for quick first-pass interpretation.
    if np.isfinite(mean_q999_ratio) and np.isfinite(mean_std_ratio):
        if (mean_q999_ratio < 0.95 and mean_std_ratio < 0.97) or (
            mean_abs_max_ratio < 0.95 and mean_std_ratio < 0.97
        ):
            return "tail_compression"
        if mean_std_ratio < 0.97 and abs(mean_q999_ratio - 1.0) < 0.08:
            return "variance_reduction"
        if (
            abs(mean_q999_ratio - 1.0) < 0.03
            and abs(mean_std_ratio - 1.0) < 0.03
            and abs(mean_abs_max_ratio - 1.0) < 0.05
            and abs(mean_zero_ratio_delta) < 0.01
        ):
            return "minimal_change"
    return "mixed_effect"


def _is_target_module_name(name: str) -> bool:
    # Keep only UNet-core modules for this analysis.
    # NOTE: time_embed is intentionally excluded (user requested).
    allow_prefixes = (
        "model.input_blocks.",
        "model.middle_block.",
        "model.output_blocks.",
        "model.out.",
    )
    return name.startswith(allow_prefixes)


def _is_internal_aux_submodule(name: str) -> bool:
    aux_suffixes = (
        ".loraA",
        ".loraB",
        ".weight_quantizer",
        ".act_quantizer",
        ".lora_dropout_layer",
    )
    return name.endswith(aux_suffixes)


def _is_quant_wrapper_module(module: nn.Module) -> bool:
    return isinstance(module, (QuantModule_DiffAE_LoRA, QuantModule, INT_QuantModule_DiffAE_LoRA))


def _infer_logical_module_type(module: nn.Module) -> str:
    fwd_func = getattr(module, "fwd_func", None)
    if fwd_func is F.conv2d:
        return "Conv2d"
    if fwd_func is F.linear:
        return "Linear"
    ori_shape = getattr(module, "ori_shape", None)
    if isinstance(ori_shape, (tuple, list)):
        if len(ori_shape) == 4:
            return "Conv2d"
        if len(ori_shape) == 2:
            return "Linear"
    cls_name = module.__class__.__name__
    return cls_name


def _infer_block_name(full_module_name: str) -> str:
    parts = full_module_name.split(".")
    if len(parts) >= 3 and parts[0] == "model":
        if parts[1] == "input_blocks" and len(parts) >= 3:
            return ".".join(parts[:3])
        if parts[1] == "output_blocks" and len(parts) >= 3:
            return ".".join(parts[:3])
        if parts[1] == "middle_block":
            return "model.middle_block"
        if parts[1] == "time_embed":
            return "model.time_embed"
        if parts[1] == "out":
            return "model.out"
    return ".".join(parts[:2]) if len(parts) >= 2 else full_module_name


def _pick_target_layers(
    model: nn.Module,
    target_module_types: Iterable[str],
    target_name_patterns: Iterable[str],
    max_layers: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    logical_type_set = set(target_module_types)
    selected: List[Dict[str, Any]] = []
    stat = {
        "total_named_modules": 0,
        "total_quant_wrappers_found": 0,
        "skipped_internal_aux_submodules": 0,
        "selected_conv2d_wrappers": 0,
        "selected_linear_wrappers": 0,
        "selected_other_wrappers": 0,
    }
    for name, module in model.named_modules():
        stat["total_named_modules"] += 1
        if _is_internal_aux_submodule(name):
            stat["skipped_internal_aux_submodules"] += 1
            continue
        if not _is_quant_wrapper_module(module):
            continue
        stat["total_quant_wrappers_found"] += 1
        if not _is_target_module_name(name):
            continue
        logical_module_type = _infer_logical_module_type(module)
        if logical_module_type not in logical_type_set:
            continue
        if not match_any_pattern(name, target_name_patterns):
            continue
        wrapper_module_type = module.__class__.__name__
        selected.append(
            {
                "layer_index": len(selected),
                "full_module_name": name,
                "wrapper_module_type": wrapper_module_type,
                "logical_module_type": logical_module_type,
                "block_name": _infer_block_name(name),
                "safe_name": safe_layer_name(name),
            }
        )
        if logical_module_type == "Conv2d":
            stat["selected_conv2d_wrappers"] += 1
        elif logical_module_type == "Linear":
            stat["selected_linear_wrappers"] += 1
        else:
            stat["selected_other_wrappers"] += 1
        if max_layers is not None and max_layers > 0 and len(selected) >= max_layers:
            break
    return selected, stat


def _build_quant_model(
    *,
    ckpt_path: str,
    model_path: str,
    num_steps: int,
    device: torch.device,
) -> Tuple[Any, nn.Module]:
    base_model = load_diffae_model(model_path)
    base_model.to(device)
    base_model.eval()
    base_model.setup()
    base_model.train_dataloader()

    quant_model = create_float_quantized_model(
        diffusion_model=base_model.ema_model,
        num_steps=num_steps,
        lora_rank=CONFIG.LORA_RANK,
        mode="train",
    )
    quant_model.to(device)
    quant_model.eval()
    quant_model.set_first_last_layer_to_8bit()

    # TemporalActivationQuantizer requires one warmup pass to initialize per-step scales.
    cali_images, cali_t, cali_y = load_calibration_data()
    with torch.no_grad():
        quant_model.set_quant_state(True, True)
        _ = quant_model(
            x=cali_images[:4].to(device),
            t=cali_t[:4].to(device),
            cond=cali_y[:4].to(device),
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)
    if hasattr(base_model.ema_model, "set_runtime_mode"):
        base_model.ema_model.set_runtime_mode(mode="infer", use_cached_aw=True, clear_cached_aw=True)
    return base_model, base_model.ema_model


def _resolve_hook_module_name_for_baseline(
    layer_name: str,
    name_to_module: Dict[str, nn.Module],
) -> Optional[str]:
    """
    Quantized graph uses paths like `model.input_blocks...` (QuantModel_DiffAE_LoRA.model = UNet).
    Original BeatGANsAutoencModel has the same UNet at top level: `input_blocks...` (no leading `model.`).
    Keep canonical layer_name (quant path) as dict keys for FF/FT compare; only hook target differs.
    """
    if layer_name in name_to_module:
        return layer_name
    if layer_name.startswith("model."):
        stripped = layer_name[len("model.") :]
        if stripped in name_to_module:
            return stripped
    return None


def _register_input_hooks(
    model: nn.Module,
    selected_layers: List[Dict[str, Any]],
    acc_map: Dict[str, Dict[int, TimestepActivationAccumulator]],
    current_t_ref: Dict[str, int],
    sample_cap_per_t: int,
    *,
    baseline_unet_path_mapping: bool = False,
) -> List[Any]:
    name_to_module = dict(model.named_modules())
    handles: List[Any] = []

    for li in selected_layers:
        layer_name = li["full_module_name"]
        resolved = (
            _resolve_hook_module_name_for_baseline(layer_name, name_to_module)
            if baseline_unet_path_mapping
            else (layer_name if layer_name in name_to_module else None)
        )
        if resolved is None:
            resolved = layer_name
        module = name_to_module.get(resolved)
        if module is None:
            LOGGER.warning(
                "layer missing at hook registration: %s (resolved=%s, baseline_mapping=%s)",
                layer_name,
                resolved,
                baseline_unet_path_mapping,
            )
            continue

        def _make_hook(name_key: str):
            def _hook(_module, inputs):
                if not inputs:
                    return
                x = inputs[0]
                if not torch.is_tensor(x):
                    return
                t_idx = int(current_t_ref.get("t", -1))
                if t_idx < 0:
                    return
                if name_key not in acc_map:
                    acc_map[name_key] = {}
                if t_idx not in acc_map[name_key]:
                    acc_map[name_key][t_idx] = TimestepActivationAccumulator(sample_cap=sample_cap_per_t)
                acc_map[name_key][t_idx].update(x)
            return _hook

        handles.append(module.register_forward_pre_hook(_make_hook(layer_name)))
    return handles


def _patch_timestep_tracking(model: nn.Module, current_t_ref: Dict[str, int]) -> Tuple[Any, Any]:
    original_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        t_tensor = kwargs.get("t", None)
        if t_tensor is None and len(args) >= 2:
            t_tensor = args[1]
        if torch.is_tensor(t_tensor) and t_tensor.numel() > 0:
            current_t_ref["t"] = int(t_tensor.reshape(-1)[0].item())
        return original_forward(*args, **kwargs)

    model.forward = wrapped_forward  # type: ignore[method-assign]
    return original_forward, wrapped_forward


@torch.no_grad()
def _collect_mode_stats(
    *,
    mode: str,
    model: nn.Module,
    sampler,
    latent_sampler,
    conf: Any,
    x_T_bank: torch.Tensor,
    latent_noise_bank: torch.Tensor,
    conds_mean: torch.Tensor,
    conds_std: torch.Tensor,
    chunk_batch: int,
    device: torch.device,
    selected_layers: List[Dict[str, Any]],
    max_batches: Optional[int],
    sample_cap_per_t: int,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    mode_u = mode.strip().upper()
    if mode_u == "BASELINE":
        LOGGER.info("[mode=BASELINE] original float ema (no quant_state)")
    else:
        wq, aq = quant_state_from_mode(mode)
        model.set_quant_state(wq, aq)
        LOGGER.info("[mode=%s] set_quant_state(weight=%s, act=%s)", mode, wq, aq)

    acc_map: Dict[str, Dict[int, TimestepActivationAccumulator]] = {}
    current_t_ref = {"t": -1}
    handles = _register_input_hooks(
        model=model,
        selected_layers=selected_layers,
        acc_map=acc_map,
        current_t_ref=current_t_ref,
        sample_cap_per_t=sample_cap_per_t,
        baseline_unet_path_mapping=(mode_u == "BASELINE"),
    )
    original_forward, _wrapped = _patch_timestep_tracking(model, current_t_ref)
    try:
        num_images = int(x_T_bank.shape[0])
        num_chunks = num_images // chunk_batch
        cache_scheduler = getattr(conf, "cache_scheduler", None)
        if max_batches is not None and max_batches > 0:
            num_chunks = min(num_chunks, max_batches)

        for ci in range(num_chunks):
            b0 = ci * chunk_batch
            b1 = (ci + 1) * chunk_batch
            x_T_chunk = x_T_bank[b0:b1].to(device)
            latent_noise_chunk = latent_noise_bank[b0:b1].to(device)
            if conf.train_mode.is_latent_diffusion():
                cond_chunk = _compute_latent_cond(
                    conf=conf,
                    latent_sampler=latent_sampler,
                    latent_net=model.latent_net,
                    latent_noise_chunk=latent_noise_chunk,
                    conds_mean=conds_mean,
                    conds_std=conds_std,
                    device=device,
                )
                model_kwargs = {"cond": cond_chunk}
            else:
                model_kwargs = None

            for _out in sampler.ddim_sample_loop_progressive(
                model=model,
                noise=x_T_chunk,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=device,
                progress=False,
                eta=0.0,
                cache_scheduler=cache_scheduler,
            ):
                pass

        layer_stats: Dict[str, Dict[int, Dict[str, float]]] = {}
        for li in selected_layers:
            name = li["full_module_name"]
            t_map = acc_map.get(name, {})
            if not t_map:
                LOGGER.warning("[mode=%s] no data collected for layer: %s", mode, name)
                continue
            layer_stats[name] = {int(t): acc.summary() for t, acc in sorted(t_map.items(), key=lambda kv: kv[0])}
        return layer_stats
    finally:
        model.forward = original_forward  # type: ignore[method-assign]
        for h in handles:
            try:
                h.remove()
            except Exception:
                LOGGER.warning("failed removing hook", exc_info=True)


def _save_mode_outputs(
    *,
    mode: str,
    layer_stats: Dict[str, Dict[int, Dict[str, float]]],
    selected_layers: List[Dict[str, Any]],
    out_dir: Path,
    save_per_timestep_npz: bool,
    T: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = {
        "meta": {"mode": mode, "T": T},
        "layers": layer_stats,
    }
    with open(out_dir / f"{mode}_activation_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary_json), f, indent=2, ensure_ascii=False)

    if save_per_timestep_npz:
        npz_dir = out_dir / "npz"
        npz_dir.mkdir(parents=True, exist_ok=True)
        for li in selected_layers:
            name = li["full_module_name"]
            safe = li["safe_name"]
            t_map = layer_stats.get(name, {})
            if not t_map:
                continue
            ts = np.array(sorted(t_map.keys()), dtype=np.int32)
            keys = [
                "numel", "mean", "std", "min", "max", "abs_mean", "abs_max",
                "q001", "q01", "q05", "q50", "q95", "q99", "q999",
                "zero_ratio", "pos_ratio", "neg_ratio", "kurtosis", "skewness",
            ]
            arrs = {k: np.array([t_map[int(t)].get(k, np.nan) for t in ts], dtype=np.float64) for k in keys}
            np.savez_compressed(npz_dir / f"layer_{safe}_{mode}.npz", t=ts, **arrs)

    # Intentionally no per-mode plots in FF/FT dirs for formal version.


def _build_compare_summary(
    *,
    ff_stats: Dict[str, Dict[int, Dict[str, float]]],
    ft_stats: Dict[str, Dict[int, Dict[str, float]]],
    selected_layers: List[Dict[str, Any]],
    T: int,
    seed: int,
    num_samples: int,
    mode_a: str = "FF",
    mode_b: str = "FT",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "meta": {
            "T": int(T),
            "seed": int(seed),
            "num_samples": int(num_samples),
            "modes": [mode_a, mode_b],
            "weight_effective": "w_plus_lora",
            "comparison_focus": "activation_quant_effect",
        },
        "layers": {},
    }
    for li in selected_layers:
        name = li["full_module_name"]
        ff_t = ff_stats.get(name, {})
        ft_t = ft_stats.get(name, {})
        t_keys = sorted(set(ff_t.keys()) & set(ft_t.keys()))
        if not t_keys:
            LOGGER.warning("no common timestep for compare layer=%s", name)
            continue
        layer_entry: Dict[str, Any] = {
            "wrapper_module_type": li["wrapper_module_type"],
            "logical_module_type": li["logical_module_type"],
            "block_name": li["block_name"],
            "timesteps": {},
        }
        std_ratios: List[float] = []
        abs_max_ratios: List[float] = []
        q999_ratios: List[float] = []
        q99_ratios: List[float] = []
        q95_ratios: List[float] = []
        zero_ratio_deltas: List[float] = []
        for t in t_keys:
            ff = ff_t[t]
            ft = ft_t[t]
            dr = calc_delta_and_ratio(ff, ft)
            delta_q99 = float(ft.get("q99", np.nan) - ff.get("q99", np.nan))
            delta_q95 = float(ft.get("q95", np.nan) - ff.get("q95", np.nan))
            delta_zero = float(ft.get("zero_ratio", np.nan) - ff.get("zero_ratio", np.nan))
            ratio_q99 = float(ft.get("q99", np.nan) / (ff.get("q99", np.nan) + 1e-12))
            ratio_q95 = float(ft.get("q95", np.nan) / (ff.get("q95", np.nan) + 1e-12))
            layer_entry["timesteps"][str(int(t))] = {
                mode_a: ff,
                mode_b: ft,
                "delta": {
                    **dr["delta"],
                    "delta_q99": delta_q99,
                    "delta_q95": delta_q95,
                    "delta_zero_ratio": delta_zero,
                },
                "ratio": {
                    **dr["ratio"],
                    "q99_ratio": ratio_q99,
                    "q95_ratio": ratio_q95,
                },
            }
            std_ratios.append(float(dr["ratio"]["std_ratio"]))
            abs_max_ratios.append(float(dr["ratio"]["abs_max_ratio"]))
            q999_ratios.append(float(dr["ratio"]["q999_ratio"]))
            q99_ratios.append(ratio_q99)
            q95_ratios.append(ratio_q95)
            zero_ratio_deltas.append(delta_zero)
        mean_std_ratio = _mean_safe(std_ratios)
        mean_absmax_ratio = _mean_safe(abs_max_ratios)
        mean_q999_ratio = _mean_safe(q999_ratios)
        mean_q99_ratio = _mean_safe(q99_ratios)
        mean_q95_ratio = _mean_safe(q95_ratios)
        mean_zero_ratio_delta = _mean_safe(zero_ratio_deltas)
        layer_entry["global_summary"] = {
            "mean_std_ratio": mean_std_ratio,
            "mean_abs_max_ratio": mean_absmax_ratio,
            "mean_q999_ratio": mean_q999_ratio,
            "mean_q99_ratio": mean_q99_ratio,
            "mean_q95_ratio": mean_q95_ratio,
            "mean_zero_ratio_delta": mean_zero_ratio_delta,
            "max_abs_std_ratio_deviation": float(np.nanmax(np.abs(np.array(std_ratios, dtype=np.float64) - 1.0))) if std_ratios else float("nan"),
            "max_abs_q999_ratio_deviation": float(np.nanmax(np.abs(np.array(q999_ratios, dtype=np.float64) - 1.0))) if q999_ratios else float("nan"),
            "effect_summary": _effect_summary_from_metrics(
                mean_q999_ratio=mean_q999_ratio,
                mean_std_ratio=mean_std_ratio,
                mean_abs_max_ratio=mean_absmax_ratio,
                mean_zero_ratio_delta=mean_zero_ratio_delta,
            ),
        }
        out["layers"][name] = layer_entry
    return out


def _build_block_summary(compare_summary: Dict[str, Any], topk: int = 3) -> Dict[str, Any]:
    grouped: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for lname, lentry in compare_summary.get("layers", {}).items():
        bname = lentry.get("block_name", "unknown")
        grouped.setdefault(bname, []).append((lname, lentry))

    out = {"meta": dict(compare_summary.get("meta", {})), "blocks": {}}
    for bname, items in grouped.items():
        num_layers = len(items)
        ltype_counts: Dict[str, int] = {}
        stds: List[float] = []
        absmaxs: List[float] = []
        q999s: List[float] = []
        q99s: List[float] = []
        zero_ds: List[float] = []
        changed: List[Tuple[float, str]] = []
        for lname, lentry in items:
            gs = lentry.get("global_summary", {})
            lt = lentry.get("logical_module_type", "unknown")
            ltype_counts[lt] = ltype_counts.get(lt, 0) + 1
            stds.append(float(gs.get("mean_std_ratio", np.nan)))
            absmaxs.append(float(gs.get("mean_abs_max_ratio", np.nan)))
            q999 = float(gs.get("mean_q999_ratio", np.nan))
            q99 = float(gs.get("mean_q99_ratio", np.nan))
            q999s.append(q999)
            q99s.append(q99)
            zero_ds.append(float(gs.get("mean_zero_ratio_delta", np.nan)))
            score = np.nanmax(
                [
                    abs(float(gs.get("mean_q999_ratio", np.nan)) - 1.0),
                    abs(float(gs.get("mean_std_ratio", np.nan)) - 1.0),
                ]
            )
            changed.append((float(score), lname))
        m_std = _mean_safe(stds)
        m_absmax = _mean_safe(absmaxs)
        m_q999 = _mean_safe(q999s)
        m_q99 = _mean_safe(q99s)
        m_zero_d = _mean_safe(zero_ds)
        out["blocks"][bname] = {
            "num_layers": num_layers,
            "logical_type_counts": ltype_counts,
            "mean_std_ratio": m_std,
            "mean_abs_max_ratio": m_absmax,
            "mean_q999_ratio": m_q999,
            "mean_q99_ratio": m_q99,
            "mean_zero_ratio_delta": m_zero_d,
            "top_changed_layers": [n for _s, n in sorted(changed, key=lambda x: x[0], reverse=True)[:topk]],
            "effect_summary": _effect_summary_from_metrics(
                mean_q999_ratio=m_q999,
                mean_std_ratio=m_std,
                mean_abs_max_ratio=m_absmax,
                mean_zero_ratio_delta=m_zero_d,
            ),
        }
    return out


def _make_rankings(compare_summary: Dict[str, Any], block_summary: Dict[str, Any], topk: int = 10) -> Dict[str, Any]:
    layers = compare_summary.get("layers", {})
    blocks = block_summary.get("blocks", {})

    def _layer_entry(name: str, e: Dict[str, Any]) -> Dict[str, Any]:
        gs = e.get("global_summary", {})
        return {
            "name": name,
            "block_name": e.get("block_name"),
            "logical_module_type": e.get("logical_module_type"),
            "mean_std_ratio": gs.get("mean_std_ratio"),
            "mean_abs_max_ratio": gs.get("mean_abs_max_ratio"),
            "mean_q999_ratio": gs.get("mean_q999_ratio"),
            "mean_zero_ratio_delta": gs.get("mean_zero_ratio_delta"),
            "effect_summary": gs.get("effect_summary"),
        }

    layer_items = [(n, _layer_entry(n, e)) for n, e in layers.items()]
    top_layers_by_q999 = sorted(
        layer_items,
        key=lambda x: abs(float(x[1].get("mean_q999_ratio", np.nan)) - 1.0),
        reverse=True,
    )[:topk]
    top_layers_by_std = sorted(
        layer_items,
        key=lambda x: abs(float(x[1].get("mean_std_ratio", np.nan)) - 1.0),
        reverse=True,
    )[:topk]
    top_layers_by_absmax = sorted(
        layer_items,
        key=lambda x: abs(float(x[1].get("mean_abs_max_ratio", np.nan)) - 1.0),
        reverse=True,
    )[:topk]

    def _block_entry(name: str, e: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": name,
            "mean_std_ratio": e.get("mean_std_ratio"),
            "mean_abs_max_ratio": e.get("mean_abs_max_ratio"),
            "mean_q999_ratio": e.get("mean_q999_ratio"),
            "mean_zero_ratio_delta": e.get("mean_zero_ratio_delta"),
            "effect_summary": e.get("effect_summary"),
        }

    block_items = [(n, _block_entry(n, e)) for n, e in blocks.items()]
    top_blocks_by_q999 = sorted(
        block_items,
        key=lambda x: abs(float(x[1].get("mean_q999_ratio", np.nan)) - 1.0),
        reverse=True,
    )[:topk]
    top_blocks_by_std = sorted(
        block_items,
        key=lambda x: abs(float(x[1].get("mean_std_ratio", np.nan)) - 1.0),
        reverse=True,
    )[:topk]
    return {
        "top_layers_by_q999_change": [x[1] for x in top_layers_by_q999],
        "top_layers_by_std_change": [x[1] for x in top_layers_by_std],
        "top_layers_by_absmax_change": [x[1] for x in top_layers_by_absmax],
        "top_blocks_by_q999_change": [x[1] for x in top_blocks_by_q999],
        "top_blocks_by_std_change": [x[1] for x in top_blocks_by_std],
    }


def _pick_representative_layers(rankings: Dict[str, Any], topk: int) -> List[str]:
    order_keys = [
        "top_layers_by_q999_change",
        "top_layers_by_std_change",
        "top_layers_by_absmax_change",
    ]
    out: List[str] = []
    seen = set()
    for k in order_keys:
        for e in rankings.get(k, [])[:topk]:
            n = e.get("name")
            if n and n not in seen:
                out.append(n)
                seen.add(n)
            if len(out) >= max(5, topk):
                return out
    return out[: max(5, topk)]


def _plot_block_bars(block_summary: Dict[str, Any], out_dir: Path, *, filename_prefix: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = block_summary.get("blocks", {})
    if not blocks:
        return
    names = list(blocks.keys())
    for metric, fn in [
        ("mean_q999_ratio", "block_mean_q999_ratio_bar.png"),
        ("mean_std_ratio", "block_mean_std_ratio_bar.png"),
        ("mean_abs_max_ratio", "block_mean_absmax_ratio_bar.png"),
    ]:
        vals = np.array([blocks[n].get(metric, np.nan) for n in names], dtype=np.float64)
        if vals.size == 0:
            continue
        order = np.argsort(np.abs(vals - 1.0))[::-1]
        show_n = min(20, len(order))
        idx = order[:show_n]
        plt.figure(figsize=(max(8, show_n * 0.45), 4.5))
        plt.bar(np.arange(show_n), vals[idx])
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        plt.xticks(np.arange(show_n), [names[i] for i in idx], rotation=60, ha="right")
        plt.ylabel(metric)
        plt.title(metric)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"{filename_prefix}{fn}", dpi=180)
        plt.close()


def _plot_compare_curves_for_layers(
    compare_summary: Dict[str, Any],
    selected_layers: List[Dict[str, Any]],
    target_layer_names: List[str],
    out_dir: Path,
    *,
    filename_prefix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_map = {x["full_module_name"]: x for x in selected_layers}
    modes = compare_summary.get("meta", {}).get("modes", ["FF", "FT"])
    mode_a = str(modes[0]) if len(modes) >= 1 else "FF"
    mode_b = str(modes[1]) if len(modes) >= 2 else "FT"
    for name in target_layer_names:
        layer = compare_summary.get("layers", {}).get(name)
        if layer is None:
            LOGGER.warning("representative layer missing in compare summary: %s", name)
            continue
        t_keys = sorted(int(x) for x in layer.get("timesteps", {}).keys())
        if not t_keys:
            LOGGER.warning("no timestep data for representative layer: %s", name)
            continue
        safe = meta_map.get(name, {}).get("safe_name", safe_layer_name(name))

        def _series(path: List[str]) -> np.ndarray:
            vals = []
            for t in t_keys:
                cur: Any = layer["timesteps"][str(t)]
                for p in path:
                    if p not in cur:
                        cur = np.nan
                        break
                    cur = cur[p]
                vals.append(cur)
            return np.array(vals, dtype=np.float64)

        for metric in ["std", "abs_max", "q999", "q99", "zero_ratio"]:
            ff = _series([mode_a, metric])
            ft = _series([mode_b, metric])
            plt.figure(figsize=(8, 4))
            plt.plot(t_keys, ff, label=mode_a, linewidth=1.8)
            plt.plot(t_keys, ft, label=mode_b, linewidth=1.8)
            plt.gca().invert_xaxis()
            plt.grid(True, alpha=0.25)
            plt.xlabel("DDIM timestep t")
            plt.ylabel(metric)
            plt.title(f"{name} | {metric} ({mode_a} vs {mode_b})")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(
                out_dir / f"{filename_prefix}layer_{safe}_{metric}_curve_{mode_a.lower()}_vs_{mode_b.lower()}.png",
                dpi=180,
            )
            plt.close()


def _generate_md_report(
    *,
    report_path: Path,
    selected_layers: List[Dict[str, Any]],
    output_base: Path,
    pair_sections: List[Dict[str, Any]],
) -> None:
    def _fmt(v: Any) -> str:
        try:
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv):
                return "nan"
            return f"{fv:.4f}"
        except Exception:
            return "nan"

    conv_n = sum(1 for x in selected_layers if x.get("logical_module_type") == "Conv2d")
    linear_n = sum(1 for x in selected_layers if x.get("logical_module_type") == "Linear")
    lines: List[str] = []
    lines.append("# Activation Distribution Analysis")
    lines.append("")
    lines.append("## Analysis goal")
    lines.append(
        "- Per-layer input activation statistics during DDIM sampling; pairwise ratio/delta under `comparisons/<a>_vs_<b>/`."
    )
    lines.append("- `BASELINE`: original Diff-AE `ema` (no QAT/LoRA ckpt). FF/FT/TT: quantized graph with shared `w+lora` from `--lora-ckpt`."
    )
    lines.append("")
    lines.append("## Layer selection")
    lines.append(f"- selected layers: {len(selected_layers)}")
    lines.append(f"- Conv2d layers: {conv_n}")
    lines.append(f"- Linear layers: {linear_n}")
    lines.append("")
    lines.append("## Output layout")
    lines.append(f"- run root: `{output_base}`")
    lines.append(f"- per-mode: `{output_base / 'models'}`")
    lines.append(f"- pairwise json: `{output_base / 'comparisons'}`")
    lines.append(f"- curves / block bars (prefixed): `{output_base / 'plots' / 'pairwise'}`")
    lines.append("")

    for sec in pair_sections:
        pair_name = sec.get("pair_name", "pair")
        compare_summary = sec.get("compare_summary", {})
        rankings = sec.get("rankings", {})
        meta = compare_summary.get("meta", {})
        top_layers = rankings.get("top_layers_by_q999_change", [])[:5]
        top_blocks = rankings.get("top_blocks_by_q999_change", [])[:5]
        effect_counts: Dict[str, int] = {}
        for _lname, e in compare_summary.get("layers", {}).items():
            ef = e.get("global_summary", {}).get("effect_summary", "mixed_effect")
            effect_counts[ef] = effect_counts.get(ef, 0) + 1
        dom_effect = sorted(effect_counts.items(), key=lambda x: x[1], reverse=True)[0][0] if effect_counts else "unknown"
        cmp_rel = sec.get("cmp_rel", "")
        plots_prefix = sec.get("plots_prefix", "")

        lines.append(f"## Pair: `{pair_name}`")
        lines.append(f"- modes: {meta.get('modes')}")
        lines.append(f"- T / seed / num_samples: {meta.get('T')} / {meta.get('seed')} / {meta.get('num_samples')}")
        lines.append(f"- dominant effect label (layer vote): `{dom_effect}`")
        lines.append("")
        lines.append("- top changed layers (by q999 ratio deviation):")
        for e in top_layers:
            lines.append(
                f"  - `{e.get('name')}` | block={e.get('block_name')} | q999_ratio={_fmt(e.get('mean_q999_ratio'))} | std_ratio={_fmt(e.get('mean_std_ratio'))}"
            )
        lines.append("")
        lines.append("- top changed blocks:")
        for e in top_blocks:
            lines.append(
                f"  - `{e.get('name')}` | q999_ratio={_fmt(e.get('mean_q999_ratio'))} | std_ratio={_fmt(e.get('mean_std_ratio'))}"
            )
        lines.append("")
        lines.append(f"- artifacts: `{output_base / cmp_rel}`" if cmp_rel else "- artifacts: (see comparisons directory)")
        if plots_prefix:
            lines.append(f"- plot filename prefix: `{plots_prefix}` under `plots/pairwise/`")
        lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    if int(args.T) != 100:
        LOGGER.warning("目前第一版預期 T=100；你給的是 T=%s，仍會繼續執行。", args.T)
    _seed_all(int(args.seed))
    device = _resolve_device(args.device)
    CONFIG.DEVICE = device
    CONFIG.NUM_DIFFUSION_STEPS = int(args.T)

    out_root = Path(args.output_root)
    images_mode = str(getattr(args, "images_mode", "official") or "official")
    dirs = _resolve_run_dirs(out_root, int(args.T), int(args.seed), images_mode)
    for key in ("base", "models", "comparisons", "plots_pairwise", "plots_summary", "LOG"):
        dirs[key].mkdir(parents=True, exist_ok=True)

    # Quantized w+lora graph (FF/FT/TT)
    base_model, eval_model = _build_quant_model(
        ckpt_path=args.lora_ckpt if args.lora_ckpt else args.ckpt,
        model_path=args.ckpt,
        num_steps=int(args.T),
        device=device,
    )

    conf = base_model.conf.clone()
    sampler = conf._make_diffusion_conf(T=int(args.T)).make_sampler()
    latent_sampler = conf._make_latent_diffusion_conf(T=int(args.T)).make_sampler()
    num_samples = int(args.num_samples)
    batch_size = int(args.batch_size)
    x_T_bank, latent_noise_bank = _make_noise_banks(
        num_images=num_samples,
        chunk_batch=batch_size,
        img_size=int(args.image_size),
        style_ch=conf.style_ch,
        seed=int(args.seed),
        device=device,
    )

    target_types = parse_module_types(args.target_module_types)
    target_patterns = parse_target_patterns(args.target_name_patterns)
    selected_layers, layer_pick_stat = _pick_target_layers(
        model=eval_model,
        target_module_types=target_types,
        target_name_patterns=target_patterns,
        max_layers=args.max_layers,
    )
    with open(dirs["base"] / "selected_layers.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(selected_layers), f, indent=2, ensure_ascii=False)
    LOGGER.info(
        "layer selection summary | total_named_modules=%d total_wrappers_found=%d "
        "selected_total=%d selected_conv2d=%d selected_linear=%d selected_other=%d "
        "skipped_internal_aux_submodules=%d",
        layer_pick_stat["total_named_modules"],
        layer_pick_stat["total_quant_wrappers_found"],
        len(selected_layers),
        layer_pick_stat["selected_conv2d_wrappers"],
        layer_pick_stat["selected_linear_wrappers"],
        layer_pick_stat["selected_other_wrappers"],
        layer_pick_stat["skipped_internal_aux_submodules"],
    )
    if len(selected_layers) == 0:
        raise RuntimeError("No target layers selected. 請檢查 --target-module-types / --target-name-patterns。")

    modes = [m.strip().upper() for m in args.collect_modes.split(",") if m.strip()]
    compare_pairs = _parse_compare_pairs(str(getattr(args, "compare_pairs", "") or "").strip())
    if not compare_pairs:
        compare_modes = [m.strip().upper() for m in args.compare_modes.split(",") if m.strip()]
        if len(compare_modes) >= 2:
            a, b = compare_modes[0], compare_modes[1]
            compare_pairs = [(f"{a.lower()}_vs_{b.lower()}", a, b)]

    need_baseline = "BASELINE" in modes or any(
        ma == "BASELINE" or mb == "BASELINE" for _f, ma, mb in compare_pairs
    )
    baseline_lit: Any = None
    baseline_model: Optional[nn.Module] = None
    if need_baseline:
        baseline_lit, baseline_model = _load_baseline_lit_model(model_path=str(args.ckpt), device=device)

    per_mode_stats: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    for mode in modes:
        if mode == "BASELINE":
            if baseline_model is None:
                baseline_lit, baseline_model = _load_baseline_lit_model(model_path=str(args.ckpt), device=device)
            layer_stats = _collect_mode_stats(
                mode="BASELINE",
                model=baseline_model,
                sampler=sampler,
                latent_sampler=latent_sampler,
                conf=conf,
                x_T_bank=x_T_bank,
                latent_noise_bank=latent_noise_bank,
                conds_mean=baseline_lit.conds_mean,
                conds_std=baseline_lit.conds_std,
                chunk_batch=batch_size,
                device=device,
                selected_layers=selected_layers,
                max_batches=args.max_batches,
                sample_cap_per_t=int(args.quantile_sample_cap),
            )
        else:
            layer_stats = _collect_mode_stats(
                mode=mode,
                model=eval_model,
                sampler=sampler,
                latent_sampler=latent_sampler,
                conf=conf,
                x_T_bank=x_T_bank,
                latent_noise_bank=latent_noise_bank,
                conds_mean=base_model.conds_mean,
                conds_std=base_model.conds_std,
                chunk_batch=batch_size,
                device=device,
                selected_layers=selected_layers,
                max_batches=args.max_batches,
                sample_cap_per_t=int(args.quantile_sample_cap),
            )
        per_mode_stats[mode] = layer_stats
        out_dir = dirs["models"] / mode
        _save_mode_outputs(
            mode=mode,
            layer_stats=layer_stats,
            selected_layers=selected_layers,
            out_dir=out_dir,
            save_per_timestep_npz=bool(args.save_per_timestep_npz),
            T=int(args.T),
        )

    if not compare_pairs:
        LOGGER.warning("No compare pairs (--compare-pairs or --compare-modes); pairwise outputs skipped.")
        _ensure_activation_legacy_symlinks(dirs["base"], dirs["models"], dirs["comparisons"])
        return {"selected_layers": selected_layers, "per_mode": list(per_mode_stats.keys()), "output_root": str(dirs["base"])}

    pair_sections: List[Dict[str, Any]] = []
    completed_pairs: List[str] = []

    for folder_name, mode_a, mode_b in compare_pairs:
        if mode_a not in per_mode_stats or mode_b not in per_mode_stats:
            LOGGER.warning(
                "skip pair %s: missing collected stats for %s or %s (have: %s)",
                folder_name,
                mode_a,
                mode_b,
                list(per_mode_stats.keys()),
            )
            continue
        cmp_dir = dirs["comparisons"] / folder_name
        cmp_dir.mkdir(parents=True, exist_ok=True)
        plots_prefix = f"{folder_name}_"

        compare = _build_compare_summary(
            ff_stats=per_mode_stats[mode_a],
            ft_stats=per_mode_stats[mode_b],
            selected_layers=selected_layers,
            T=int(args.T),
            seed=int(args.seed),
            num_samples=num_samples,
            mode_a=mode_a,
            mode_b=mode_b,
        )
        if "BASELINE" in (mode_a, mode_b):
            compare.setdefault("meta", {})["comparison_focus"] = "baseline_vs_quantized"
            compare["meta"]["note"] = (
                "BASELINE uses original float ema from --ckpt; other side uses QAT graph + --lora-ckpt. "
                "Module paths align with quant wrappers; interpret ratios as structural alignment, not identical weights."
            )
        else:
            compare.setdefault("meta", {})["comparison_focus"] = "activation_quant_effect_within_quant_graph"

        with open(cmp_dir / "activation_compare_summary.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(compare), f, indent=2, ensure_ascii=False)

        if _to_bool(args.generate_block_summary, True):
            block_summary = _build_block_summary(compare, topk=max(3, int(args.representative_topk)))
            with open(cmp_dir / "block_compare_summary.json", "w", encoding="utf-8") as f:
                json.dump(_json_safe(block_summary), f, indent=2, ensure_ascii=False)
        else:
            block_summary = {"meta": compare.get("meta", {}), "blocks": {}}

        if _to_bool(args.generate_rankings, True):
            rankings = _make_rankings(compare, block_summary, topk=max(10, int(args.representative_topk)))
            with open(cmp_dir / "rankings.json", "w", encoding="utf-8") as f:
                json.dump(_json_safe(rankings), f, indent=2, ensure_ascii=False)
        else:
            rankings = {}

        representative_layers = _pick_representative_layers(rankings, int(args.representative_topk)) if rankings else []
        with open(cmp_dir / "representative_layers.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe({"layers": representative_layers}), f, indent=2, ensure_ascii=False)

        if _to_bool(args.save_representative_plots, True):
            _plot_compare_curves_for_layers(
                compare_summary=compare,
                selected_layers=selected_layers,
                target_layer_names=representative_layers,
                out_dir=dirs["plots_pairwise"],
                filename_prefix=plots_prefix,
            )
        if _to_bool(args.save_all_layer_plots, False):
            all_names = [x["full_module_name"] for x in selected_layers]
            all_dir = cmp_dir / "all_layer_plots"
            _plot_compare_curves_for_layers(
                compare_summary=compare,
                selected_layers=selected_layers,
                target_layer_names=all_names,
                out_dir=all_dir,
                filename_prefix="",
            )
        if _to_bool(args.save_block_plots, True):
            _plot_block_bars(block_summary, dirs["plots_pairwise"], filename_prefix=plots_prefix)

        pair_sections.append(
            {
                "pair_name": folder_name,
                "compare_summary": compare,
                "rankings": rankings,
                "cmp_rel": Path("comparisons") / folder_name,
                "plots_prefix": plots_prefix,
            }
        )
        completed_pairs.append(folder_name)

    if _to_bool(args.generate_md_report, True) and pair_sections:
        _generate_md_report(
            report_path=dirs["base"] / "ACTIVATION_ANALYSIS.md",
            selected_layers=selected_layers,
            output_base=dirs["base"],
            pair_sections=pair_sections,
        )

    _ensure_activation_legacy_symlinks(dirs["base"], dirs["models"], dirs["comparisons"])

    return {
        "selected_layers": selected_layers,
        "layer_selection_summary": layer_pick_stat,
        "modes": modes,
        "compare_pairs": completed_pairs,
        "output_root": str(dirs["base"]),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Activation input distribution analysis (multi-mode / multi-pair)")
    p.add_argument("--run", action="store_true", help="required flag to execute")
    p.add_argument("--ckpt", type=str, default="checkpoints/ffhq128_autoenc_latent/last.ckpt")
    p.add_argument(
        "--lora-ckpt",
        type=str,
        default="QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth",
        help="QAT/LoRA checkpoint that provides w+lora (ema_model.model.*).",
    )
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--output-root",
        type=str,
        default="QATcode/quantize_ver2/baseline_quant_analysis/activation_results",
    )
    p.add_argument(
        "--images-mode",
        type=str,
        default="official",
        choices=["debug", "v1", "official"],
        help="Subfolder under T_<T>/ (align with pred_xstart_results).",
    )
    p.add_argument(
        "--collect-modes",
        type=str,
        default="FF,FT",
        help="Comma-separated modes: BASELINE (float ema from --ckpt only), FF, FT, TT, TF, ...",
    )
    p.add_argument(
        "--compare-modes",
        type=str,
        default="FF,FT",
        help="Legacy: if --compare-pairs is empty, first two modes become one pair (e.g. ff_vs_ft).",
    )
    p.add_argument(
        "--compare-pairs",
        type=str,
        default="ff_vs_ft",
        help=(
            "Comma-separated pairs as a_vs_b (lowercase). "
            "For baseline_vs_ft / baseline_vs_tt also pass e.g. "
            "--collect-modes BASELINE,FF,FT,TT --compare-pairs ff_vs_ft,baseline_vs_ft,baseline_vs_tt"
        ),
    )
    p.add_argument(
        "--target-module-types",
        type=str,
        default="Conv2d,Linear",
        help="Logical layer types represented by quant wrappers (e.g., Conv2d,Linear).",
    )
    p.add_argument(
        "--target-name-patterns",
        type=str,
        default=None,
        help="optional regex/substring list separated by comma",
    )
    p.add_argument("--save-per-timestep-npz", action="store_true")
    p.add_argument("--save-plots", action="store_true", help="(deprecated) kept for compatibility; compare plots are controlled by specific flags below.")
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--quantile-sample-cap", type=int, default=4096)
    p.add_argument("--generate-block-summary", type=str, default="true")
    p.add_argument("--generate-rankings", type=str, default="true")
    p.add_argument("--generate-md-report", type=str, default="true")
    p.add_argument("--representative-topk", type=int, default=5)
    p.add_argument("--block-aggregation-metric", type=str, default="mean", choices=["mean"])
    p.add_argument("--save-all-layer-plots", type=str, default="false")
    p.add_argument("--save-representative-plots", type=str, default="true")
    p.add_argument("--save-block-plots", type=str, default="true")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        raise SystemExit(0)
    dirs = _resolve_run_dirs(Path(args.output_root), int(args.T), int(args.seed), str(args.images_mode))
    log_file = dirs["LOG"] / f"activation_distribution_T{int(args.T)}_seed{int(args.seed)}.log"
    _setup_logging(str(log_file))
    LOGGER.info("Start activation distribution analysis")
    res = run_analysis(args)
    LOGGER.info("Done. outputs: %s", json.dumps(_json_safe(res), ensure_ascii=False))
