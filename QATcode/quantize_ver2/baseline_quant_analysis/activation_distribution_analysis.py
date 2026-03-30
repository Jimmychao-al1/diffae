"""
Activation input distribution analysis for FF vs FT (T=100 default).

Goal:
- Analyze module forward *input* activations for Conv2d/Linear during DDIM sampling.
- Compare FF (weight off, act off) vs FT (weight off, act on), with identical w+lora.
- Output per-mode stats and FF_vs_FT deltas/ratios + plots.
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


LOGGER = logging.getLogger("activation_distribution_analysis")


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


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


def _is_target_module_name(name: str) -> bool:
    # Keep only UNet-core modules as requested.
    allow_prefixes = (
        "model.input_blocks.",
        "model.middle_block.",
        "model.output_blocks.",
        "model.out.",
        "model.time_embed.",
    )
    return name.startswith(allow_prefixes)


def _pick_target_layers(
    model: nn.Module,
    target_module_types: Iterable[str],
    target_name_patterns: Iterable[str],
    max_layers: Optional[int],
) -> List[Dict[str, Any]]:
    type_set = set(target_module_types)
    selected: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        if module_type not in type_set:
            continue
        if not _is_target_module_name(name):
            continue
        if not match_any_pattern(name, target_name_patterns):
            continue
        selected.append(
            {
                "layer_index": len(selected),
                "full_module_name": name,
                "module_type": module_type,
                "safe_name": safe_layer_name(name),
            }
        )
        if max_layers is not None and max_layers > 0 and len(selected) >= max_layers:
            break
    return selected


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


def _register_input_hooks(
    model: nn.Module,
    selected_layers: List[Dict[str, Any]],
    acc_map: Dict[str, Dict[int, TimestepActivationAccumulator]],
    current_t_ref: Dict[str, int],
    sample_cap_per_t: int,
) -> List[Any]:
    name_to_module = dict(model.named_modules())
    handles: List[Any] = []

    for li in selected_layers:
        layer_name = li["full_module_name"]
        module = name_to_module.get(layer_name)
        if module is None:
            LOGGER.warning("layer missing at hook registration: %s", layer_name)
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
    save_plots: bool,
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

    if save_plots:
        plot_dir = out_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        metrics = ["std", "abs_max", "q999", "q99", "zero_ratio"]
        for li in selected_layers:
            name = li["full_module_name"]
            safe = li["safe_name"]
            t_map = layer_stats.get(name, {})
            if not t_map:
                continue
            ts = np.array(sorted(t_map.keys()), dtype=np.int32)
            for m in metrics:
                ys = np.array([t_map[int(t)].get(m, np.nan) for t in ts], dtype=np.float64)
                plt.figure(figsize=(8, 4))
                plt.plot(ts, ys, linewidth=1.8)
                plt.gca().invert_xaxis()
                plt.grid(True, alpha=0.25)
                plt.xlabel("DDIM timestep t")
                plt.ylabel(m)
                plt.title(f"{mode} | {name} | {m}")
                plt.tight_layout()
                plt.savefig(plot_dir / f"layer_{safe}_{m}_curve.png", dpi=180)
                plt.close()


def _build_compare_summary(
    *,
    ff_stats: Dict[str, Dict[int, Dict[str, float]]],
    ft_stats: Dict[str, Dict[int, Dict[str, float]]],
    selected_layers: List[Dict[str, Any]],
    T: int,
    seed: int,
    num_samples: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "meta": {
            "T": int(T),
            "seed": int(seed),
            "num_samples": int(num_samples),
            "modes": ["FF", "FT"],
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
            "module_type": li["module_type"],
            "timesteps": {},
        }
        std_ratios: List[float] = []
        abs_max_ratios: List[float] = []
        q999_ratios: List[float] = []
        for t in t_keys:
            ff = ff_t[t]
            ft = ft_t[t]
            dr = calc_delta_and_ratio(ff, ft)
            layer_entry["timesteps"][str(int(t))] = {
                "FF": ff,
                "FT": ft,
                "delta": dr["delta"],
                "ratio": dr["ratio"],
            }
            std_ratios.append(float(dr["ratio"]["std_ratio"]))
            abs_max_ratios.append(float(dr["ratio"]["abs_max_ratio"]))
            q999_ratios.append(float(dr["ratio"]["q999_ratio"]))
        layer_entry["global_summary"] = {
            "mean_std_ratio": float(np.nanmean(std_ratios)),
            "mean_abs_max_ratio": float(np.nanmean(abs_max_ratios)),
            "mean_q999_ratio": float(np.nanmean(q999_ratios)),
        }
        out["layers"][name] = layer_entry
    return out


def _save_compare_plots(compare_summary: Dict[str, Any], selected_layers: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for li in selected_layers:
        name = li["full_module_name"]
        safe = li["safe_name"]
        layer = compare_summary["layers"].get(name)
        if layer is None:
            continue
        t_keys = sorted(int(x) for x in layer["timesteps"].keys())
        if not t_keys:
            continue

        def _series(path: List[str]) -> np.ndarray:
            vals = []
            for t in t_keys:
                cur: Any = layer["timesteps"][str(t)]
                for p in path:
                    cur = cur[p]
                vals.append(cur)
            return np.array(vals, dtype=np.float64)

        # A: FF vs FT overlay curves
        for metric in ["std", "abs_max", "q999", "q99", "zero_ratio"]:
            ff = _series(["FF", metric])
            ft = _series(["FT", metric])
            plt.figure(figsize=(8, 4))
            plt.plot(t_keys, ff, label="FF", linewidth=1.8)
            plt.plot(t_keys, ft, label="FT", linewidth=1.8)
            plt.gca().invert_xaxis()
            plt.grid(True, alpha=0.25)
            plt.xlabel("DDIM timestep t")
            plt.ylabel(metric)
            plt.title(f"{name} | {metric} (FF vs FT)")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(out_dir / f"layer_{safe}_{metric}_curve_ff_vs_ft.png", dpi=180)
            plt.close()

        # B: ratio curves
        for metric in ["std_ratio", "abs_max_ratio", "q999_ratio"]:
            ys = _series(["ratio", metric])
            plt.figure(figsize=(8, 4))
            plt.plot(t_keys, ys, linewidth=1.8)
            plt.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
            plt.gca().invert_xaxis()
            plt.grid(True, alpha=0.25)
            plt.xlabel("DDIM timestep t")
            plt.ylabel(metric)
            plt.title(f"{name} | {metric}")
            plt.tight_layout()
            plt.savefig(out_dir / f"layer_{safe}_{metric}_curve.png", dpi=180)
            plt.close()

        # C: simple summary bar
        gs = layer["global_summary"]
        labels = ["mean_std_ratio", "mean_abs_max_ratio", "mean_q999_ratio"]
        vals = [gs[k] for k in labels]
        plt.figure(figsize=(7, 4))
        plt.bar(np.arange(len(labels)), vals)
        plt.xticks(np.arange(len(labels)), labels, rotation=20)
        plt.grid(axis="y", alpha=0.25)
        plt.title(f"{name} | global ratio summary")
        plt.tight_layout()
        plt.savefig(out_dir / f"layer_{safe}_global_ratio_summary.png", dpi=180)
        plt.close()


def _resolve_official_dirs(root: Path, T: int, seed: int) -> Dict[str, Path]:
    base = root / f"T_{T}" / "official" / f"seed{seed}"
    return {
        "base": base,
        "FF": base / "FF",
        "FT": base / "FT",
        "CMP": base / "FF_vs_FT",
        "LOG": root / "logs",
    }


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    if int(args.T) != 100:
        LOGGER.warning("目前第一版預期 T=100；你給的是 T=%s，仍會繼續執行。", args.T)
    _seed_all(int(args.seed))
    device = _resolve_device(args.device)
    CONFIG.DEVICE = device
    CONFIG.NUM_DIFFUSION_STEPS = int(args.T)

    out_root = Path(args.output_root)
    dirs = _resolve_official_dirs(out_root, int(args.T), int(args.seed))
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    # Build model once; FF/FT share exactly the same w+lora.
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
    selected_layers = _pick_target_layers(
        model=eval_model,
        target_module_types=target_types,
        target_name_patterns=target_patterns,
        max_layers=args.max_layers,
    )
    with open(dirs["base"] / "selected_layers.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(selected_layers), f, indent=2, ensure_ascii=False)
    LOGGER.info("selected layers: %d", len(selected_layers))
    if len(selected_layers) == 0:
        raise RuntimeError("No target layers selected. 請檢查 --target-module-types / --target-name-patterns。")

    modes = [m.strip().upper() for m in args.collect_modes.split(",") if m.strip()]
    per_mode_stats: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    for mode in modes:
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
        if mode in dirs:
            md = dirs[mode]
        else:
            md = dirs["base"] / mode
        _save_mode_outputs(
            mode=mode,
            layer_stats=layer_stats,
            selected_layers=selected_layers,
            out_dir=md,
            save_per_timestep_npz=bool(args.save_per_timestep_npz),
            save_plots=bool(args.save_plots),
            T=int(args.T),
        )

    if "FF" not in per_mode_stats or "FT" not in per_mode_stats:
        LOGGER.warning("collect-modes does not include both FF and FT; compare summary skipped.")
        return {"selected_layers": selected_layers, "per_mode": list(per_mode_stats.keys())}

    compare = _build_compare_summary(
        ff_stats=per_mode_stats["FF"],
        ft_stats=per_mode_stats["FT"],
        selected_layers=selected_layers,
        T=int(args.T),
        seed=int(args.seed),
        num_samples=num_samples,
    )
    with open(dirs["CMP"] / "activation_compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(compare), f, indent=2, ensure_ascii=False)
    if args.save_plots:
        _save_compare_plots(compare, selected_layers, dirs["CMP"])

    return {
        "selected_layers": selected_layers,
        "modes": modes,
        "output_root": str(dirs["base"]),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Activation input distribution analysis (FF vs FT)")
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
        "--collect-modes",
        type=str,
        default="FF,FT",
        help="Comma-separated quant modes (from {FF,FT,TF,TT}); first version compare focuses FF/FT.",
    )
    p.add_argument("--target-module-types", type=str, default="Conv2d,Linear")
    p.add_argument(
        "--target-name-patterns",
        type=str,
        default=None,
        help="optional regex/substring list separated by comma",
    )
    p.add_argument("--save-per-timestep-npz", action="store_true")
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--quantile-sample-cap", type=int, default=4096)
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        raise SystemExit(0)
    dirs = _resolve_official_dirs(Path(args.output_root), int(args.T), int(args.seed))
    log_file = dirs["LOG"] / f"activation_distribution_T{int(args.T)}_seed{int(args.seed)}.log"
    _setup_logging(str(log_file))
    LOGGER.info("Start activation distribution analysis")
    res = run_analysis(args)
    LOGGER.info("Done. outputs: %s", json.dumps(_json_safe(res), ensure_ascii=False))
