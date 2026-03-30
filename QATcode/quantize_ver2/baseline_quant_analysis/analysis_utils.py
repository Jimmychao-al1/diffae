from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


def safe_layer_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")


def parse_target_patterns(patterns: Optional[str]) -> List[str]:
    if not patterns:
        return []
    return [x.strip() for x in patterns.split(",") if x.strip()]


def match_any_pattern(text: str, patterns: Iterable[str]) -> bool:
    ps = list(patterns)
    if not ps:
        return True
    for pat in ps:
        try:
            if re.search(pat, text):
                return True
        except re.error:
            if pat in text:
                return True
    return False


def parse_module_types(spec: str) -> List[str]:
    out = [x.strip() for x in spec.split(",") if x.strip()]
    return out if out else ["Conv2d", "Linear"]


def quant_state_from_mode(mode: str) -> tuple[bool, bool]:
    mode_u = mode.strip().upper()
    if mode_u not in {"FF", "FT", "TF", "TT"}:
        raise ValueError(f"Unsupported mode: {mode}")
    return mode_u[0] == "T", mode_u[1] == "T"


@dataclass
class TimestepActivationAccumulator:
    sample_cap: int = 4096
    device: Optional[torch.device] = None
    # running moments / counts
    numel: int = 0
    sum_v: float = 0.0
    sum_sq: float = 0.0
    min_v: float = math.inf
    max_v: float = -math.inf
    sum_abs: float = 0.0
    abs_max: float = 0.0
    zero_count: int = 0
    pos_count: int = 0
    neg_count: int = 0
    # compressed samples for quantiles/skew/kurtosis (keep on the same device as activations)
    samples: List[torch.Tensor] = field(default_factory=list)
    sample_numel: int = 0

    def update(self, x: torch.Tensor) -> None:
        if x is None:
            return
        flat = x.detach().reshape(-1)
        if flat.numel() == 0:
            return
        if self.device is None:
            self.device = flat.device
        flat = flat.to(device=self.device, dtype=torch.float32)
        n = int(flat.numel())
        self.numel += n
        self.sum_v += float(flat.sum().item())
        self.sum_sq += float((flat * flat).sum().item())
        self.min_v = min(self.min_v, float(flat.min().item()))
        self.max_v = max(self.max_v, float(flat.max().item()))
        abs_flat = flat.abs()
        self.sum_abs += float(abs_flat.sum().item())
        self.abs_max = max(self.abs_max, float(abs_flat.max().item()))
        self.zero_count += int((flat == 0).sum().item())
        self.pos_count += int((flat > 0).sum().item())
        self.neg_count += int((flat < 0).sum().item())
        self._update_sample_pool(flat)

    def _update_sample_pool(self, flat_dev: torch.Tensor) -> None:
        if self.sample_cap <= 0:
            return
        n = int(flat_dev.numel())
        # keep a random subset from this chunk, then compact globally
        take_n = min(n, max(64, self.sample_cap // 4))
        if take_n <= 0:
            return
        if n > take_n:
            idx = torch.randperm(n, device=flat_dev.device)[:take_n]
            sampled = flat_dev[idx]
        else:
            sampled = flat_dev
        sampled = sampled.to(dtype=torch.float32)
        self.samples.append(sampled)
        self.sample_numel += int(sampled.numel())
        if self.sample_numel <= self.sample_cap:
            return
        merged = torch.cat(self.samples, dim=0)
        keep = min(self.sample_cap, int(merged.numel()))
        if int(merged.numel()) > keep:
            idx = torch.randperm(int(merged.numel()), device=merged.device)[:keep]
            merged = merged[idx]
        self.samples = [merged]
        self.sample_numel = int(merged.numel())

    def _merged_samples(self) -> torch.Tensor:
        if not self.samples:
            dev = self.device if self.device is not None else torch.device("cpu")
            return torch.empty((0,), dtype=torch.float32, device=dev)
        if len(self.samples) == 1:
            return self.samples[0]
        merged = torch.cat(self.samples, dim=0)
        self.samples = [merged]
        self.sample_numel = int(merged.numel())
        return merged

    def summary(self, include_shape: bool = True) -> Dict[str, float]:
        if self.numel <= 0:
            return {}
        mean = self.sum_v / self.numel
        var = max(0.0, self.sum_sq / self.numel - mean * mean)
        std = math.sqrt(var)
        qvec = torch.tensor([0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999], dtype=torch.float32, device=self.device)
        smp = self._merged_samples().to(dtype=torch.float32)
        if int(smp.numel()) > 0:
            q = torch.quantile(smp, qvec)
            smp_mean = smp.mean()
            smp_std = smp.std(unbiased=False)
            smp_var = smp.var(unbiased=False)
            skew = float((((smp - smp_mean) ** 3).mean() / ((smp_std + 1e-12) ** 3)).item())
            kurt = float((((smp - smp_mean) ** 4).mean() / ((smp_var + 1e-12) ** 2) - 3.0).item())
        else:
            q = torch.full((7,), float("nan"), dtype=torch.float32, device=self.device)
            skew = float("nan")
            kurt = float("nan")
        out: Dict[str, float] = {
            "numel": int(self.numel),
            "mean": float(mean),
            "std": float(std),
            "min": float(self.min_v),
            "max": float(self.max_v),
            "abs_mean": float(self.sum_abs / self.numel),
            "abs_max": float(self.abs_max),
            "q001": float(q[0].item()),
            "q01": float(q[1].item()),
            "q05": float(q[2].item()),
            "q50": float(q[3].item()),
            "q95": float(q[4].item()),
            "q99": float(q[5].item()),
            "q999": float(q[6].item()),
            "zero_ratio": float(self.zero_count / self.numel),
            "pos_ratio": float(self.pos_count / self.numel),
            "neg_ratio": float(self.neg_count / self.numel),
            "kurtosis": float(kurt),
            "skewness": float(skew),
            "sample_numel_for_quantiles": int(smp.numel()),
        }
        if not include_shape:
            out.pop("sample_numel_for_quantiles", None)
        return out


def calc_delta_and_ratio(ff: Dict[str, float], ft: Dict[str, float], eps: float = 1e-12) -> Dict[str, Dict[str, float]]:
    delta = {
        "delta_std": float(ft["std"] - ff["std"]),
        "delta_abs_max": float(ft["abs_max"] - ff["abs_max"]),
        "delta_q999": float(ft["q999"] - ff["q999"]),
        "delta_q99": float(ft["q99"] - ff["q99"]),
        "delta_q95": float(ft["q95"] - ff["q95"]),
        "delta_zero_ratio": float(ft["zero_ratio"] - ff["zero_ratio"]),
    }
    ratio = {
        "std_ratio": float(ft["std"] / (ff["std"] + eps)),
        "abs_max_ratio": float(ft["abs_max"] / (ff["abs_max"] + eps)),
        "q999_ratio": float(ft["q999"] / (ff["q999"] + eps)),
    }
    return {"delta": delta, "ratio": ratio}
