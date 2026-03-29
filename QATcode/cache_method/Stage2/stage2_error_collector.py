"""
Stage2：在 baseline（無 cache）與 cache 兩次前向中，對 31 個 TimestepEmbedSequential
收集輸出並計算 L1 / L2 / cosine（逐樣本 flatten 後再平均）。

時間軸（與 similarity_calculation / SVD collector 一致）：
- 模型每個 DDIM 子步呼叫一次 UNet forward；forward_pre_hook 將 _step_counter 在 0..T-1 循環遞增。
- 此處 _step_counter 即 expanded_mask 的「步序索引」：0 對應第一步（DDIM i=T-1），T-1 對應 i=0。
- 輸出聚合時同時給出 ddim_timestep = (T-1) - step_idx，與 diffusion/base.py 的 i 一致。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import TimestepEmbedSequential

from QATcode.cache_method.Stage2.stage2_scheduler_adapter import (
    EXPECTED_NUM_BLOCKS,
    step_index_to_ddim_timestep,
)

LOGGER = logging.getLogger("Stage2ErrorCollector")


def _flatten_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a,b: (B,C,H,W) -> 每個樣本 cosine，shape (B,)"""
    a2 = a.flatten(1)
    b2 = b.flatten(1)
    return F.cosine_similarity(a2, b2, dim=1, eps=1e-8)


def _l1_scalar(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().mean()


def _l2_rms(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).mean().sqrt()


def _zone_ddim_timesteps(shared_zones: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    """shared_zones 與 Stage1 相同：t_start >= t_end，覆蓋 DDIM i=0..T-1。"""
    out: Dict[int, List[int]] = {}
    for z in shared_zones:
        zid = int(z["id"])
        ts, te = int(z["t_start"]), int(z["t_end"])
        out[zid] = list(range(te, ts + 1))
    return out


class Stage2ErrorCollector:
    """
    兩階段收集：先 run_name='baseline'，再 'cache'；各存
    feats[runtime_name][step_idx] = float32 tensor on CPU (B,C,H,W)。

    注意：cache 模式下若某層在該 DDIM 步為 reuse，unet 會跳過該層 forward，
    因此不會有 hook 輸出。比較時只對「兩趟都有 tensor」的 step_idx 計算誤差。
    """

    def __init__(self, T: int, device: Optional[torch.device] = None):
        self.T = int(T)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hooks: List[Any] = []
        self._step_counter = -1
        self._run_name = "baseline"
        self._feats: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {
            "baseline": {},
            "cache": {},
        }
        self._runtime_order: List[str] = []
        self._module_by_runtime: Dict[str, nn.Module] = {}

    def set_run(self, name: str) -> None:
        if name not in ("baseline", "cache"):
            raise ValueError("run name must be 'baseline' or 'cache'")
        self._run_name = name
        self._step_counter = -1

    def reset_storage(self) -> None:
        self._feats = {"baseline": {}, "cache": {}}
        self._step_counter = -1

    def register_hooks(self, model: nn.Module) -> None:
        """掃描 31 個 TimestepEmbedSequential 並註冊 hook（fail-fast）。"""
        self.remove_hooks()
        self._module_by_runtime = self._discover_unet_blocks(model)
        self._runtime_order = list(self._module_by_runtime.keys())
        if len(self._runtime_order) != EXPECTED_NUM_BLOCKS:
            raise RuntimeError(
                f"expected {EXPECTED_NUM_BLOCKS} TimestepEmbedSequential UNet blocks, "
                f"found {len(self._runtime_order)}"
            )

        for rt, mod in self._module_by_runtime.items():
            self._feats["baseline"].setdefault(rt, {})
            self._feats["cache"].setdefault(rt, {})
            self.hooks.append(mod.register_forward_hook(self._make_block_hook(rt)))

        self.hooks.append(model.register_forward_pre_hook(self._make_step_pre_hook(), with_kwargs=True))
        LOGGER.info("[Stage2] registered %d block hooks + step pre-hook", len(self._module_by_runtime))

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def _discover_unet_blocks(self, model: nn.Module) -> Dict[str, nn.Module]:
        """
        以模組路徑 regex 對齊 UNet 的 TimestepEmbedSequential（Quant 包裝下仍應出現
        ...input_blocks.N / ...middle_block / ...output_blocks.N）。
        """
        found: Dict[str, nn.Module] = {}
        for name, mod in model.named_modules():
            if not isinstance(mod, TimestepEmbedSequential):
                continue
            if re.search(r"\.encoder\.", name) and "input_blocks" not in name:
                continue
            rt: Optional[str] = None
            m = re.search(r"input_blocks\.(\d+)(?:\.|$)", name)
            if m:
                # 影像 encoder 也有 input_blocks.*，必須與 UNet 的 model.input_blocks 區分
                if ".encoder.input_blocks" in name:
                    continue
                rt = f"encoder_layer_{int(m.group(1))}"
            elif re.search(r"middle_block(?:\.|$)", name):
                rt = "middle_layer"
            else:
                m = re.search(r"output_blocks\.(\d+)(?:\.|$)", name)
                if m:
                    rt = f"decoder_layer_{int(m.group(1))}"
            if rt is None:
                continue
            if rt in found:
                raise RuntimeError(f"duplicate runtime {rt}: already registered vs {name!r}")
            found[rt] = mod

        if len(found) != EXPECTED_NUM_BLOCKS:
            raise RuntimeError(
                f"TimestepEmbedSequential discovery: expected {EXPECTED_NUM_BLOCKS} blocks, "
                f"got {len(found)} keys={sorted(found.keys())}"
            )
        ordered: List[str] = [f"encoder_layer_{i}" for i in range(15)]
        ordered.append("middle_layer")
        ordered.extend(f"decoder_layer_{i}" for i in range(15))
        return {k: found[k] for k in ordered}

    def _make_step_pre_hook(self) -> Callable:
        def pre_hook(module, args, kwargs):
            self._step_counter = (self._step_counter + 1) % self.T

        return pre_hook

    def _make_block_hook(self, runtime_name: str) -> Callable:
        def hook(module, inp, out):
            if self._step_counter < 0 or self._step_counter >= self.T:
                return
            step_idx = int(self._step_counter)
            x = out.detach().float().cpu()
            bucket = self._feats[self._run_name].setdefault(runtime_name, {})
            # 若同一步重入（不預期），覆寫
            bucket[step_idx] = x

        return hook

    def compute_diagnostics(
        self,
        shared_zones: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """比對 baseline vs cache，產出 per_block_step / per_block_zone / global_summary。"""
        per_block_step_error: Dict[str, Dict[str, Dict[str, float]]] = {}
        per_block_zone_error: Dict[str, Dict[str, Dict[str, float]]] = {}
        zone_ts = _zone_ddim_timesteps(shared_zones)

        all_l1: List[float] = []
        all_l2: List[float] = []
        all_cos: List[float] = []

        for rt in self._runtime_order:
            per_block_step_error[rt] = {}
            per_block_zone_error[rt] = {}
            bmap = self._feats["baseline"].get(rt, {})
            cmap = self._feats["cache"].get(rt, {})
            for step_idx in range(self.T):
                ddim_i = step_index_to_ddim_timestep(step_idx, self.T)
                key = str(ddim_i)
                if step_idx not in bmap or step_idx not in cmap:
                    # cache reuse 時該層未 forward，無法與 baseline 對齊比較
                    continue
                a, b = bmap[step_idx], cmap[step_idx]
                if a.shape != b.shape:
                    raise RuntimeError(f"{rt} step {step_idx}: shape mismatch {a.shape} vs {b.shape}")
                l1 = float(_l1_scalar(a, b))
                l2 = float(_l2_rms(a, b))
                cos = float(_flatten_cosine(a, b).mean())
                per_block_step_error[rt][key] = {"l1": l1, "l2": l2, "cosine": cos}
                all_l1.append(l1)
                all_l2.append(l2)
                all_cos.append(cos)

            for zid, ts in zone_ts.items():
                zs_l1: List[float] = []
                zs_l2: List[float] = []
                zs_cos: List[float] = []
                for ddim_i in ts:
                    key = str(ddim_i)
                    if key not in per_block_step_error[rt]:
                        continue
                    st = per_block_step_error[rt][key]
                    zs_l1.append(st["l1"])
                    zs_l2.append(st["l2"])
                    zs_cos.append(st["cosine"])
                if not zs_l1:
                    per_block_zone_error[rt][str(zid)] = {
                        "mean_l1": float("nan"),
                        "mean_l2": float("nan"),
                        "mean_cosine": float("nan"),
                        "num_steps": 0,
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

        global_summary = {
            "mean_l1": float(np.mean(all_l1)) if all_l1 else float("nan"),
            "mean_l2": float(np.mean(all_l2)) if all_l2 else float("nan"),
            "mean_cosine": float(np.mean(all_cos)) if all_cos else float("nan"),
            "num_entries": len(all_l1),
            "note": (
                "僅統計 baseline 與 cache 兩邊皆有 forward 的 (block,step)；"
                "reuse 步該層無 hook 輸出故略過"
            ),
        }

        return {
            "per_block_step_error": per_block_step_error,
            "per_block_zone_error": per_block_zone_error,
            "global_summary": global_summary,
            "T": self.T,
            "time_axis_note": (
                "step_idx 0..T-1：0=第一步(DDIM i=T-1)，T-1=最後一步(DDIM i=0)；"
                "per_block_step_error 的 key 為字串化的 DDIM timestep i"
            ),
        }


def aggregate_per_timestep(
    per_block_step_error: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """跨 block 聚合：每個 DDIM i 取所有 block 的 mean/max。"""
    by_t: Dict[str, List[float]] = {}
    for rt, steps in per_block_step_error.items():
        for ti, m in steps.items():
            by_t.setdefault(ti, []).append(m["l1"])
    out: Dict[str, Dict[str, float]] = {}
    for ti, vals in by_t.items():
        out[ti] = {"mean_l1": float(np.mean(vals)), "max_l1": float(np.max(vals))}
    return out
