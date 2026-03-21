from __future__ import annotations

import logging
from typing import Tuple

import torch


logger = logging.getLogger(__name__)


def get_default_device() -> str:
    """Return 'cuda' if available else 'cpu'. Does not change any external API."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def assert_finite_tensor(x: torch.Tensor, name: str) -> None:
    """Raise ValueError if tensor contains NaN/Inf."""
    if not torch.all(torch.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")


def assert_positive_tensor(x: torch.Tensor, name: str) -> None:
    """Raise ValueError if any element of tensor is <= 0."""
    if not (x > 0).all():
        raise ValueError(f"{name} must be > 0 everywhere")


def assert_uint8_range(x: torch.Tensor, name: str) -> None:
    """Check tensor codes within [0,255] for 8-bit."""
    if x.min().item() < 0 or x.max().item() > 255:
        raise ValueError(f"{name} out of 8-bit code range [0,255]")


def ensure_broadcastable_shapes(a: torch.Tensor, b: torch.Tensor, name_a: str, name_b: str) -> Tuple[torch.Size, torch.Size]:
    """
    Ensure two tensors are broadcastable. Returns their shapes; raises ValueError if not.
    Does not modify tensors.
    """
    try:
        # Attempt a broadcast via a dummy operation without allocating large tensors
        _ = (a.unsqueeze(0) + b.unsqueeze(0))  # type: ignore[operator]
    except Exception as e:
        raise ValueError(f"{name_a} and {name_b} are not broadcastable: {a.shape} vs {b.shape}: {e}") from e
    return a.shape, b.shape


def reconstruct_weight_from_int(w_int: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Compute W_hat = (W_int - z) * s with basic validation. Does not move devices."""
    assert_finite_tensor(scale, "scale")
    assert_finite_tensor(zero_point, "zero_point")
    ensure_broadcastable_shapes(w_int, scale, "w_int", "scale")
    ensure_broadcastable_shapes(w_int, zero_point, "w_int", "zero_point")
    return (w_int.float() - zero_point.float()) * scale.float()


