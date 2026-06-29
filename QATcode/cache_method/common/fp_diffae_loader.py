"""Load FP (non-quantized) Diff-AE for sampling and evidence collection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from experiment import LitModel

LOGGER = logging.getLogger("fp_diffae")


def load_fp_diffae_for_sampling(
    *,
    model_path: str = "checkpoints/ffhq128_autoenc_latent/last.ckpt",
    device: torch.device,
    num_steps: int = 100,
) -> "LitModel":
    """
    Load pretrained FP Diff-AE without QuantModel wrapper.

    Args:
        model_path: Path to last.ckpt (relative to repo root unless absolute).
        device: Target device.
        num_steps: DDIM steps T (kept for API symmetry with quant loaders).
    """
    del num_steps  # unused; caller sets diffusion T via conf/sampler
    from QATcode.quantize_ver2.common_utils import load_diffae_model

    base: LitModel = load_diffae_model(model_path, LOGGER)
    base.to(device)
    base.eval()
    base.setup()
    try:
        base.train_dataloader()
    except Exception as exc:
        LOGGER.warning("train_dataloader() skipped: %s", exc)
    return base
