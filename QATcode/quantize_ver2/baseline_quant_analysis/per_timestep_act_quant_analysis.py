"""
per_timestep_act_quant_analysis.py

Per-Timestep Activation Distribution Analysis for TemporalActivationQuantizer.

Goal:
  Analyze the *input* activation of TemporalActivationQuantizer per DDIM timestep and
  compare Q-DiffAE (TT checkpoint) against the original Diff-AE (BASELINE).

Collected metrics (per layer, per timestep t):
  - min, max, mean, std
  - q01, q99  (1%/99% quantile via reservoir sampling)
  - scale      (s_x^t = act_quantizer.scale_list[current_step], TT only)
  - clipping_ratio  (fraction of |x| > scale, TT only)

Output:
  {output_root}/
    selected_layers.json
    stats_TT.json        -- {layer_key: {str(t): {...}}}
    stats_BASELINE.json
    plots/
      layer_<safe_name>.png   -- q01/mean/q99 per model, TT solid / BASELINE dashed

Per-step stats accumulation runs on the same device as model activations (use ``--device cuda``
for GPU end-to-end; reservoir + quantiles execute on that device).

Usage example:
  python -m QATcode.quantize_ver2.baseline_quant_analysis.per_timestep_act_quant_analysis \
      --ckpt  checkpoints/ffhq128_autoenc_latent/last.ckpt \
      --lora-ckpt QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth \
      --output-root QATcode/quantize_ver2/results/act_quant_per_timestep \
      --num-samples 64 --seed 42 --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(".")
sys.path.append("./model")

from QATcode.quantize_ver2.quant_layer_v2 import TemporalActivationQuantizer
from QATcode.quantize_ver2.quant_model_lora_v2 import QuantModel_DiffAE_LoRA
from QATcode.quantize_ver2.baseline_quant_analysis.pred_xstart_quantile_analysis import (
    CONFIG,
    _compute_latent_cond,
    _load_quant_and_ema_from_ckpt,
    _make_noise_banks,
    _seed_all,
    create_float_quantized_model,
    load_calibration_data,
    load_diffae_model,
)

LOGGER = logging.getLogger("per_timestep_act_quant_analysis")


# ============================================================
# Per-step statistics accumulator
# ============================================================

@dataclass
class _StepAccum:
    """Online accumulator for one (layer, timestep) cell.

    Tracks moments exactly (no approximation) and keeps a reservoir of
    random samples for q01/q99 estimation.

    All tensor work stays on ``x.device`` (typically CUDA during inference)
    until ``summary()``; only Python floats are returned for JSON.
    """
    # online moments
    n: int = 0
    sum_v: float = 0.0
    sum_sq: float = 0.0
    min_v: float = float("inf")
    max_v: float = float("-inf")
    # reservoir for quantile estimation
    samples: List = field(default_factory=list)
    sample_n: int = 0
    sample_cap: int = 2048
    # TT-only: per-step scale (fixed parameter, overwritten each call)
    scale: Optional[float] = None
    # TT-only: clipping ratio (averaged over batch elements across all calls)
    clip_sum: float = 0.0
    clip_count: int = 0

    def update(self, x: torch.Tensor, scale: Optional[float] = None) -> None:
        # Keep activations on the same device as the model (GPU in normal runs).
        flat = x.detach().reshape(-1).float()
        dev = flat.device
        n = int(flat.numel())
        if n == 0:
            return

        self.n += n
        self.sum_v += float(flat.sum().item())
        self.sum_sq += float((flat * flat).sum().item())
        self.min_v = min(self.min_v, float(flat.min().item()))
        self.max_v = max(self.max_v, float(flat.max().item()))

        # Reservoir sampling: keep at most sample_cap values
        take = min(n, max(64, self.sample_cap // 4))
        if n > take:
            idx = torch.randperm(n, device=dev)[:take]
            smp = flat[idx]
        else:
            smp = flat
        self.samples.append(smp)
        self.sample_n += int(smp.numel())
        # Compact reservoir when it grows too large
        if self.sample_n > self.sample_cap * 2:
            merged = torch.cat(self.samples)
            keep = min(self.sample_cap, int(merged.numel()))
            idx = torch.randperm(int(merged.numel()), device=merged.device)[:keep]
            self.samples = [merged[idx]]
            self.sample_n = keep

        # TT-only bookkeeping
        if scale is not None:
            self.scale = float(scale)
            cr = float((flat.abs() > scale).float().mean().item())
            self.clip_sum += cr
            self.clip_count += 1

    def summary(self) -> Dict[str, Any]:
        if self.n == 0:
            return {}
        mean = self.sum_v / self.n
        var = max(0.0, self.sum_sq / self.n - mean * mean)
        std = float(math.sqrt(var))

        if self.samples:
            smp = torch.cat(self.samples)
        else:
            smp = torch.empty(0, dtype=torch.float32, device="cpu")
        if int(smp.numel()) >= 2:
            q01 = float(torch.quantile(smp.float(), 0.01).item())
            q99 = float(torch.quantile(smp.float(), 0.99).item())
        else:
            q01 = q99 = float("nan")

        out: Dict[str, Any] = {
            "n": self.n,
            "min": self.min_v,
            "max": self.max_v,
            "mean": mean,
            "std": std,
            "q01": q01,
            "q99": q99,
        }
        if self.scale is not None:
            out["scale"] = self.scale
        if self.clip_count > 0:
            out["clipping_ratio"] = self.clip_sum / self.clip_count
        return out


# ============================================================
# JSON serialization helper
# ============================================================

def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(x, torch.Tensor):
        return _json_safe(x.detach().cpu().numpy())
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    return x


# ============================================================
# Representative layer selection
# ============================================================

def _get_block_key(module_path: str) -> str:
    """Map a quant-model module path to a UNet block identifier.

    Example:
      'model.input_blocks.3.0.in_layers.1' -> 'input_blocks.3'
      'model.middle_block.0.in_layers.1'   -> 'middle_block'
      'model.output_blocks.2.0.out_layers.3' -> 'output_blocks.2'
    """
    p = module_path[len("model."):] if module_path.startswith("model.") else module_path
    parts = p.split(".")
    if not parts:
        return "unknown"
    if parts[0] in ("input_blocks", "output_blocks") and len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    if parts[0] == "middle_block":
        return "middle_block"
    if parts[0] == "out":
        return "out"
    return parts[0]


def _select_representative_layers(
    quant_model: nn.Module,
    max_layers: int = 12,
) -> List[Dict[str, str]]:
    """Return one representative layer per UNet block, capped at max_layers.

    Each entry contains:
      act_quant_path  -- path of TemporalActivationQuantizer in quant_model
      parent_path     -- path of QuantModule_DiffAE_LoRA (parent of act_quantizer)
      baseline_path   -- same path in BeatGANsAutoencModel (strip leading 'model.')
      block_key       -- UNet block identifier, e.g. 'input_blocks.3'
      safe_name       -- filesystem-safe version of parent_path
    """
    block_to_layers: Dict[str, List[Dict]] = {}

    for name, module in quant_model.named_modules():
        if not isinstance(module, TemporalActivationQuantizer):
            continue
        if not name.endswith(".act_quantizer"):
            continue
        parent_path = name[: -len(".act_quantizer")]
        block_key = _get_block_key(parent_path)
        baseline_path = (
            parent_path[len("model."):] if parent_path.startswith("model.") else parent_path
        )
        entry: Dict[str, str] = {
            "act_quant_path": name,
            "parent_path": parent_path,
            "baseline_path": baseline_path,
            "block_key": block_key,
            "safe_name": re.sub(r"[^0-9A-Za-z._-]+", "_", parent_path).strip("_"),
        }
        block_to_layers.setdefault(block_key, []).append(entry)

    # Sort by block key and pick the first (shallowest) layer per block
    per_block: List[Dict] = []
    for bk in sorted(block_to_layers.keys()):
        per_block.append(block_to_layers[bk][0])

    LOGGER.info(
        "Found %d blocks with TemporalActivationQuantizer (max_layers=%d)",
        len(per_block), max_layers,
    )

    if len(per_block) <= max_layers:
        return per_block

    # Subsample: keep all middle_block layers, evenly sample encoder and decoder
    encoder = [x for x in per_block if x["block_key"].startswith("input_blocks")]
    mid = [x for x in per_block if x["block_key"] == "middle_block"]
    decoder = [x for x in per_block if x["block_key"].startswith("output_blocks")]
    other = [x for x in per_block if x["block_key"] not in
             {x2["block_key"] for x2 in encoder + mid + decoder}]

    budget = max(0, max_layers - len(mid) - len(other))
    n_enc = max(2, budget // 2)
    n_dec = max(2, budget - n_enc)

    def _even_sample(lst: List, n: int) -> List:
        if len(lst) <= n or n <= 0:
            return lst[:n] if n > 0 else []
        if n == 1:
            return [lst[0]]
        indices = [int(round(i * (len(lst) - 1) / (n - 1))) for i in range(n)]
        return [lst[i] for i in indices]

    result = _even_sample(encoder, n_enc) + mid + _even_sample(decoder, n_dec) + other
    return result[:max_layers]


# ============================================================
# Model loading
# ============================================================

def _build_tt_model(
    *,
    ckpt_path: str,
    model_path: str,
    num_steps: int,
    device: torch.device,
) -> Tuple[Any, nn.Module]:
    """Load Q-DiffAE (TT checkpoint): quant wrapper + calibration + checkpoint weights."""
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

    # One calibration pass to initialize TemporalActivationQuantizer scales
    cali_images, cali_t, cali_y = load_calibration_data()
    with torch.no_grad():
        quant_model.set_quant_state(True, True)
        _ = quant_model(
            x=cali_images[:4].to(device),
            t=cali_t[:4].to(device),
            cond=cali_y[:4].to(device),
        )

    # Load TT checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)

    eval_model = base_model.ema_model  # QuantModel_DiffAE_LoRA with TT weights
    if hasattr(eval_model, "set_runtime_mode"):
        eval_model.set_runtime_mode(mode="infer", use_cached_aw=True, clear_cached_aw=True)

    LOGGER.info("TT model loaded. ema_model type: %s", type(eval_model).__name__)
    return base_model, eval_model


def _build_baseline_model(
    *,
    model_path: str,
    device: torch.device,
) -> Tuple[Any, nn.Module]:
    """Load original Diff-AE (BASELINE): raw EMA model, no QAT."""
    base_model = load_diffae_model(model_path)
    base_model.to(device)
    base_model.eval()
    base_model.setup()
    base_model.train_dataloader()

    eval_model = base_model.ema_model  # BeatGANsAutoencModel
    LOGGER.info("BASELINE model loaded. ema_model type: %s", type(eval_model).__name__)
    return base_model, eval_model


# ============================================================
# Timestep tracking (wrap model.forward to capture 't')
# ============================================================

def _patch_t_tracking(model: nn.Module, t_ref: Dict) -> Any:
    """Wrap model.forward to intercept the 't' argument and store in t_ref."""
    orig_forward = model.forward

    def _wrapped(*args, **kwargs):
        t_val = kwargs.get("t", None)
        if t_val is None and len(args) >= 2:
            t_val = args[1]
        if torch.is_tensor(t_val) and t_val.numel() > 0:
            t_ref["t"] = int(t_val.reshape(-1)[0].item())
        return orig_forward(*args, **kwargs)

    model.forward = _wrapped  # type: ignore[method-assign]
    return orig_forward


# ============================================================
# Stats collection
# ============================================================

@torch.no_grad()
def _collect_stats(
    *,
    eval_model: nn.Module,
    is_tt: bool,
    selected_layers: List[Dict[str, str]],
    sampler: Any,
    latent_sampler: Any,
    conf: Any,
    x_T_bank: torch.Tensor,
    latent_noise_bank: torch.Tensor,
    conds_mean: torch.Tensor,
    conds_std: torch.Tensor,
    chunk_batch: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Run DDIM sampling and collect per-layer, per-timestep activation stats.

    For TT:   hooks on TemporalActivationQuantizer; reads scale_list[current_step].
    For BASELINE: hooks on the original Conv2d/Linear at the corresponding path.
    Both use the noise-level 't' (captured via _patch_t_tracking) as the timestep key.
    """
    label = "TT" if is_tt else "BASELINE"

    # --- quant state and current_step reset for TT ---
    if is_tt:
        eval_model.set_quant_state(True, True)
        n_reset = 0
        for m in eval_model.modules():
            if isinstance(m, TemporalActivationQuantizer):
                m.inited = True
                m.current_step = m.total_steps - 1
                n_reset += 1
        LOGGER.info("[%s] set_quant_state(True,True), reset current_step to T-1 for %d quantizers",
                    label, n_reset)
    else:
        LOGGER.info("[%s] using original float model (no quant_state change)", label)

    # --- build hook targets ---
    acc_map: Dict[str, Dict[int, _StepAccum]] = {}
    t_ref: Dict = {"t": -1}
    name_to_module: Dict[str, nn.Module] = dict(eval_model.named_modules())
    handles: List = []

    for layer in selected_layers:
        layer_key = layer["parent_path"]          # canonical key shared by both models
        hook_path = layer["act_quant_path"] if is_tt else layer["baseline_path"]
        target = name_to_module.get(hook_path)

        if target is None:
            LOGGER.warning("[%s] Module not found in named_modules: '%s' -- skipping",
                           label, hook_path)
            continue

        def _make_hook(key: str, _is_tt: bool) -> Any:
            def _hook(module: nn.Module, inputs: Tuple) -> None:
                if not inputs:
                    return
                x = inputs[0]
                if not torch.is_tensor(x):
                    return
                t = t_ref.get("t", -1)
                if t < 0:
                    return

                acc_map.setdefault(key, {})
                if t not in acc_map[key]:
                    acc_map[key][t] = _StepAccum()

                # TT: read scale from TemporalActivationQuantizer before it decrements
                scale: Optional[float] = None
                if _is_tt and isinstance(module, TemporalActivationQuantizer):
                    step_idx = max(0, min(module.current_step, module.total_steps - 1))
                    scale = float(module.scale_list[step_idx].item())

                acc_map[key][t].update(x, scale=scale)
            return _hook

        handles.append(target.register_forward_pre_hook(_make_hook(layer_key, is_tt)))

    LOGGER.info("[%s] Registered %d hooks", label, len(handles))

    orig_forward = _patch_t_tracking(eval_model, t_ref)
    try:
        num_images = int(x_T_bank.shape[0])
        num_chunks = num_images // chunk_batch
        if max_batches is not None and max_batches > 0:
            num_chunks = min(num_chunks, max_batches)

        for ci in range(num_chunks):
            LOGGER.info("[%s] Chunk %d / %d", label, ci + 1, num_chunks)
            b0, b1 = ci * chunk_batch, (ci + 1) * chunk_batch
            x_T_chunk = x_T_bank[b0:b1].to(device)
            latent_noise_chunk = latent_noise_bank[b0:b1].to(device)

            if conf.train_mode.is_latent_diffusion():
                cond_chunk = _compute_latent_cond(
                    conf=conf,
                    latent_sampler=latent_sampler,
                    latent_net=eval_model.latent_net,
                    latent_noise_chunk=latent_noise_chunk,
                    conds_mean=conds_mean,
                    conds_std=conds_std,
                    device=device,
                )
                model_kwargs: Optional[Dict] = {"cond": cond_chunk}
            else:
                model_kwargs = None

            # Reset current_step before each chunk for TT
            if is_tt:
                for m in eval_model.modules():
                    if isinstance(m, TemporalActivationQuantizer):
                        m.current_step = m.total_steps - 1

            cache_scheduler = getattr(conf, "cache_scheduler", None)
            for _out in sampler.ddim_sample_loop_progressive(
                model=eval_model,
                noise=x_T_chunk,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=device,
                progress=False,
                eta=0.0,
                cache_scheduler=cache_scheduler,
            ):
                pass

    finally:
        eval_model.forward = orig_forward  # type: ignore[method-assign]
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    # Compile final results
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for layer in selected_layers:
        key = layer["parent_path"]
        t_map = acc_map.get(key, {})
        if not t_map:
            LOGGER.warning("[%s] No data collected for layer: %s", label, key)
            continue
        results[key] = {
            int(t): acc.summary()
            for t, acc in sorted(t_map.items(), key=lambda kv: kv[0])
        }
        LOGGER.info("[%s] Layer '%s': collected %d timesteps", label, key, len(results[key]))

    return results


# ============================================================
# Output: JSON
# ============================================================

def _save_stats_json(
    stats: Dict[str, Dict[int, Dict[str, Any]]],
    out_path: Path,
    label: str,
    selected_layers: List[Dict[str, str]],
    T: int,
    num_samples: int,
    seed: int,
) -> None:
    payload = {
        "meta": {
            "label": label,
            "T": T,
            "num_samples": num_samples,
            "seed": seed,
            "num_layers": len(stats),
        },
        "layers": stats,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved %s stats -> %s", label, out_path)


# ============================================================
# Output: per-layer line plots (q01 / mean / q99)
# ============================================================

def _plot_layer_curves(
    tt_stats: Dict[str, Dict[int, Dict[str, Any]]],
    baseline_stats: Dict[str, Dict[int, Dict[str, Any]]],
    selected_layers: List[Dict[str, str]],
    out_dir: Path,
) -> None:
    """One PNG per layer: q01/mean/q99, TT solid vs BASELINE dashed."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Color scheme: consistent across lines
    COLOR = {"q01": "#2196F3", "mean": "#FF9800", "q99": "#F44336"}
    METRICS = ["q01", "mean", "q99"]
    METRIC_LABELS = {"q01": "q01", "mean": "mean", "q99": "q99"}

    for layer in selected_layers:
        key = layer["parent_path"]
        safe = layer["safe_name"]

        tt_t = tt_stats.get(key, {})
        bl_t = baseline_stats.get(key, {})

        if not tt_t and not bl_t:
            LOGGER.warning("No data for layer '%s', skipping plot", key)
            continue

        # Collect all timesteps from both models
        all_t = sorted(set(list(tt_t.keys()) + list(bl_t.keys())), reverse=True)
        t_arr = np.array(all_t, dtype=np.float64)

        fig, ax = plt.subplots(figsize=(10, 4))

        for metric in METRICS:
            color = COLOR[metric]
            label_base = METRIC_LABELS[metric]

            # TT: solid lines
            if tt_t:
                vals = np.array(
                    [tt_t[t].get(metric, np.nan) if t in tt_t else np.nan for t in all_t],
                    dtype=np.float64,
                )
                ax.plot(t_arr, vals, color=color, linestyle="-", linewidth=1.8,
                        label=f"TT {label_base}")

            # BASELINE: dashed lines
            if bl_t:
                vals = np.array(
                    [bl_t[t].get(metric, np.nan) if t in bl_t else np.nan for t in all_t],
                    dtype=np.float64,
                )
                ax.plot(t_arr, vals, color=color, linestyle="--", linewidth=1.5,
                        label=f"BL {label_base}")

        ax.set_xlabel("Diffusion timestep  t  (high noise → low noise)")
        ax.set_ylabel("Activation value")
        ax.set_title(f"{key}\nq01 / mean / q99  |  solid=TT  dashed=BASELINE", fontsize=9)
        ax.invert_xaxis()          # t goes high → low (left to right = denoising direction)
        ax.grid(True, alpha=0.25)

        # Build a tidy legend (deduplicate color-only entries)
        handles_leg, labels_leg = ax.get_legend_handles_labels()
        ax.legend(handles_leg, labels_leg, fontsize=7, ncol=3, loc="best")

        fig.tight_layout()
        out_path = out_dir / f"layer_{safe}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        LOGGER.info("Saved plot: %s", out_path)


# ============================================================
# Main analysis runner
# ============================================================

def run_analysis(args: argparse.Namespace) -> None:
    _seed_all(int(args.seed))
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    CONFIG.DEVICE = device
    CONFIG.NUM_DIFFUSION_STEPS = int(args.T)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)

    # ---- Load models ----
    LOGGER.info("=== Loading TT (Q-DiffAE) model ===")
    tt_base, tt_eval = _build_tt_model(
        ckpt_path=args.lora_ckpt,
        model_path=args.ckpt,
        num_steps=int(args.T),
        device=device,
    )

    LOGGER.info("=== Loading BASELINE (Diff-AE) model ===")
    bl_base, bl_eval = _build_baseline_model(
        model_path=args.ckpt,
        device=device,
    )

    # ---- Select representative layers (from TT model) ----
    LOGGER.info("=== Selecting representative layers ===")
    selected_layers = _select_representative_layers(tt_eval, max_layers=int(args.max_layers))
    LOGGER.info("Selected %d layers:", len(selected_layers))
    for i, lyr in enumerate(selected_layers):
        LOGGER.info("  [%02d] block=%-25s  path=%s", i, lyr["block_key"], lyr["parent_path"])

    with open(out_root / "selected_layers.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(selected_layers), f, indent=2, ensure_ascii=False)

    # ---- Build samplers (shared config) ----
    conf = tt_base.conf.clone()
    T = int(args.T)
    sampler = conf._make_diffusion_conf(T=T).make_sampler()
    latent_sampler = conf._make_latent_diffusion_conf(T=T).make_sampler()

    num_samples = int(args.num_samples)
    chunk_batch = int(args.batch_size)
    # Ensure divisibility
    num_samples = (num_samples // chunk_batch) * chunk_batch
    if num_samples == 0:
        raise ValueError(f"num_samples ({args.num_samples}) must be >= batch_size ({chunk_batch})")

    x_T_bank, latent_noise_bank = _make_noise_banks(
        num_images=num_samples,
        chunk_batch=chunk_batch,
        img_size=int(args.image_size),
        style_ch=conf.style_ch,
        seed=int(args.seed),
        device=device,
    )

    # ---- Collect TT stats ----
    LOGGER.info("=== Collecting TT stats (%d images, batch %d) ===", num_samples, chunk_batch)
    tt_stats = _collect_stats(
        eval_model=tt_eval,
        is_tt=True,
        selected_layers=selected_layers,
        sampler=sampler,
        latent_sampler=latent_sampler,
        conf=conf,
        x_T_bank=x_T_bank,
        latent_noise_bank=latent_noise_bank,
        conds_mean=tt_base.conds_mean,
        conds_std=tt_base.conds_std,
        chunk_batch=chunk_batch,
        device=device,
        max_batches=args.max_batches,
    )

    # ---- Collect BASELINE stats ----
    LOGGER.info("=== Collecting BASELINE stats ===")
    baseline_stats = _collect_stats(
        eval_model=bl_eval,
        is_tt=False,
        selected_layers=selected_layers,
        sampler=sampler,
        latent_sampler=latent_sampler,
        conf=conf,
        x_T_bank=x_T_bank,
        latent_noise_bank=latent_noise_bank,
        conds_mean=bl_base.conds_mean,
        conds_std=bl_base.conds_std,
        chunk_batch=chunk_batch,
        device=device,
        max_batches=args.max_batches,
    )

    # ---- Save JSON ----
    LOGGER.info("=== Saving JSON outputs ===")
    _save_stats_json(
        tt_stats, out_root / "stats_TT.json",
        label="TT", selected_layers=selected_layers,
        T=T, num_samples=num_samples, seed=int(args.seed),
    )
    _save_stats_json(
        baseline_stats, out_root / "stats_BASELINE.json",
        label="BASELINE", selected_layers=selected_layers,
        T=T, num_samples=num_samples, seed=int(args.seed),
    )

    # ---- Plots ----
    LOGGER.info("=== Generating per-layer plots ===")
    _plot_layer_curves(
        tt_stats=tt_stats,
        baseline_stats=baseline_stats,
        selected_layers=selected_layers,
        out_dir=out_root / "plots",
    )

    LOGGER.info("=== Done. Results saved to: %s ===", out_root)


# ============================================================
# CLI entry point
# ============================================================

def _setup_logging(log_file: Optional[str] = None) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-Timestep Act-Quantizer Activation Distribution Analysis"
    )
    parser.add_argument(
        "--ckpt", type=str,
        default="checkpoints/ffhq128_autoenc_latent/last.ckpt",
        help="Path to Diff-AE base checkpoint (used by both TT and BASELINE)",
    )
    parser.add_argument(
        "--lora-ckpt", type=str,
        default="QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth",
        help="Path to TT (Q-DiffAE) LoRA+quant checkpoint",
    )
    parser.add_argument(
        "--output-root", type=str,
        default="QATcode/quantize_ver2/results/act_quant_per_timestep",
        help="Root directory for all outputs",
    )
    parser.add_argument("--T", type=int, default=100, help="DDIM steps (default: 100)")
    parser.add_argument(
        "--num-samples", type=int, default=64,
        help="Number of images to generate for analysis (divisible by --batch-size)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Chunk batch size")
    parser.add_argument(
        "--max-layers", type=int, default=12,
        help="Maximum representative layers to analyze (8–15 recommended)",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Limit number of batches per model (for quick testing)",
    )
    parser.add_argument(
        "--image-size", type=int, default=128, help="Image spatial size"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', 'cpu', or 'cuda:N'",
    )
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path")
    args = parser.parse_args()

    _setup_logging(args.log_file)
    LOGGER.info("Starting per-timestep act-quantizer analysis")
    LOGGER.info("  ckpt        : %s", args.ckpt)
    LOGGER.info("  lora_ckpt   : %s", args.lora_ckpt)
    LOGGER.info("  output_root : %s", args.output_root)
    LOGGER.info("  T=%d  num_samples=%d  batch=%d  max_layers=%d  seed=%d",
                args.T, args.num_samples, args.batch_size, args.max_layers, args.seed)

    run_analysis(args)


if __name__ == "__main__":
    main()
