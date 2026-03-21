"""
擴散模型訓練器模組
包含各種專門的訓練器，支持不同的訓練策略
"""

from .diffusion import SpacedDiffusionBeatGans
from .base import GaussianDiffusionBeatGans

# 重新導出以保持向後兼容
__all__ = ['SpacedDiffusionBeatGans', 'GaussianDiffusionBeatGans']

# 可以在這裡添加其他訓練器的導入
# 例如: from ..QATcode.diffae_trainer import SpacedDiffusionBeatGans_Trainer