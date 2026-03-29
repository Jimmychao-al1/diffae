"""
Stage2：透過 model/unet_autoenc.py 的 cache_debug_collector，在 baseline 與 cache
兩趟採樣中收集每層、每 DDIM 步的 tensor（含 reuse 時從 cached_data 取出的值），
再計算與 baseline 的 L1 / L2 / cosine。

時間軸：
- DDIM timestep i 由 t 張量取得；expanded_mask 步序 step_idx = (T-1) - i。
- step_idx=0 對應第一步（i=T-1）；step_idx=T-1 對應 i=0。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from QATcode.cache_method.Stage2.stage2_scheduler_adapter import RUNTIME_LAYER_NAMES

LOGGER = logging.getLogger("Stage2ErrorCollector")


def _flatten_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a2 = a.flatten(1)
    b2 = b.flatten(1)
    return F.cosine_similarity(a2, b2, dim=1, eps=1e-8)


def _l1_scalar(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().mean()


def _l2_rms(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).mean().sqrt()


def _zone_ddim_timesteps(shared_zones: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for z in shared_zones:
        zid = int(z["id"])
        ts, te = int(z["t_start"]), int(z["t_end"])
        out[zid] = list(range(te, ts + 1))
    return out


class Stage2ErrorCollector:
    """
    使用 UNet forward 內的 cache_debug_collector（見 unet_autoenc.py），
    對 baseline / cache 各存一份 feats[run][layer_key][step_idx]。
    """

    def __init__(self, T: int, device: Optional[torch.device] = None):
        self.T = int(T)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._run_name = "baseline"
        self._feats: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {
            "baseline": {},
            "cache": {},
        }

    def set_run(self, name: str) -> None:
        if name not in ("baseline", "cache"):
            raise ValueError("run name must be 'baseline' or 'cache'")
        self._run_name = name

    def clear_storage(self) -> None:
        """釋放收集到的 tensor（run 結束後呼叫）。"""
        self._feats = {"baseline": {}, "cache": {}}

    def make_cache_debug_callback(self) -> Callable[..., None]:
        """傳給 conf.cache_debug_collector / BeatGANsAutoencModel.forward。"""
        return self.on_layer_output

    def on_layer_output(self, layer_key: str, h: torch.Tensor, *, recompute: bool, t: torch.Tensor) -> None:
        """與 unet_autoenc._cache_dbg 簽名一致；recompute 僅供除錯，不比對用。"""
        step_idx = (self.T - 1) - int(t[0].item())
        if not (0 <= step_idx < self.T):
            return
        x = h.detach().float().cpu()
        bucket = self._feats[self._run_name].setdefault(layer_key, {})
        bucket[step_idx] = x

    def compute_diagnostics(
        self,
        shared_zones: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        per_block_step_error: Dict[str, Dict[str, Dict[str, float]]] = {}
        per_block_zone_error: Dict[str, Dict[str, Dict[str, Any]]] = {}
        zone_ts = _zone_ddim_timesteps(shared_zones)

        all_l1: List[float] = []
        all_l2: List[float] = []
        all_cos: List[float] = []

        for rt in RUNTIME_LAYER_NAMES:
            per_block_step_error[rt] = {}
            per_block_zone_error[rt] = {}
            bmap = self._feats["baseline"].get(rt, {})
            cmap = self._feats["cache"].get(rt, {})
            for step_idx in range(self.T):
                ddim_i = (self.T - 1) - step_idx
                key = str(ddim_i)
                if step_idx not in bmap or step_idx not in cmap:
                    raise RuntimeError(
                        f"[Stage2] missing feature for {rt} step_idx={step_idx} (ddim_i={ddim_i}); "
                        f"baseline keys {sorted(bmap.keys())}, cache keys {sorted(cmap.keys())}"
                    )
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
                    st = per_block_step_error[rt].get(str(ddim_i))
                    if st is None:
                        continue
                    zs_l1.append(st["l1"])
                    zs_l2.append(st["l2"])
                    zs_cos.append(st["cosine"])
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
            "T": self.T,
            "time_axis_note": (
                "step_idx 0..T-1：0=第一步(DDIM i=T-1)，T-1=最後一步(DDIM i=0)；"
                "per_block_step_error 的 key 為字串化的 DDIM timestep i"
            ),
        }


def aggregate_per_timestep(
    per_block_step_error: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    by_t: Dict[str, List[float]] = {}
    for rt, steps in per_block_step_error.items():
        for ti, m in steps.items():
            by_t.setdefault(ti, []).append(m["l1"])
    out: Dict[str, Dict[str, float]] = {}
    for ti, vals in by_t.items():
        out[ti] = {"mean_l1": float(np.mean(vals)), "max_l1": float(np.max(vals))}
    return out
