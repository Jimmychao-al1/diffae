"""Selective quantization model wrappers used in Step4 tooling."""

import logging
import torch
import torch.nn as nn
from QATcode.quantize_ver2.quant_layer_v2 import QuantModule, StraightThrough
from model.unet_autoenc import BeatGANsAutoencModel

logger = logging.getLogger(__name__)


class SelectiveQuantModel(nn.Module):
    """
    選擇性量化模型 - 只量化 UNet 的核心架構

    Note: Public constructor signature is preserved. Internally, it refactors modules
    under target components into QuantModule instances.
    """

    def __init__(
        self,
        model: BeatGANsAutoencModel,
        weight_quant_params,
        act_quant_params,
        need_init=True,
        quantize_skip_connections=False,
        target_modules=None,
    ):
        super().__init__()
        self.model = model
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        self.quantize_skip_connections = quantize_skip_connections

        # 對指定的 UNet 模組進行量化
        self.quant_unet_modules(need_init)

    def quant_unet_modules(self, need_init: bool = True) -> "Any":
        """
        只對 UNet 的核心模組進行量化：
        - time_embed
        - input_blocks
        - middle_block
        - output_blocks
        - out
        """
        logger.info("=== 開始選擇性量化 UNet 模組 ===")

        # 要量化的模組列表
        target_modules = ["time_embed", "input_blocks", "middle_block", "output_blocks", "out"]

        for module_name in target_modules:
            if hasattr(self.model, module_name):
                module = getattr(self.model, module_name)
                logger.info("量化模組: %s", module_name)

                # 遞歸量化該模組中的 Conv2d 和 Linear 層
                self.quant_module_refactor(
                    module, self.weight_quant_params, self.act_quant_params, need_init=need_init
                )
            else:
                logger.warning("模組 %s 不存在", module_name)

    def quant_module_refactor(
        self,
        module: "Any",
        weight_quant_params: "Any",
        act_quant_params: "Any",
        need_init: bool = True,
        target_modules: "Any" = None,
    ) -> "Any":
        """
        遞歸替換 Conv2d 和 Linear 層為 QuantModule
        可選擇性跳過 skip connection 相關層
        """
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)):
                # 檢查是否需要跳過 skip connection
                should_skip = False
                if not self.quantize_skip_connections:
                    # 跳過包含 'skip' 或 'op' 的層名
                    if "skip" in name.lower():
                        should_skip = True
                        logger.info("  跳過量化 skip connection: %s", name)

                if not should_skip:
                    setattr(
                        module,
                        name,
                        QuantModule(
                            child_module, weight_quant_params, act_quant_params, need_init=need_init
                        ),
                    )
                    logger.info("  量化層: %s", name)
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                # 遞歸處理子模組
                self.quant_module_refactor(
                    child_module, weight_quant_params, act_quant_params, need_init=need_init
                )

    def set_quant_state(
        self, weight_quant: bool = False, act_quant: bool = False, target_modules: "Any" = None
    ) -> "Any":
        """設置量化狀態（傳遞到所有 QuantModule）。"""
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)

    def set_runtime_mode(
        self, mode: str = "train", use_cached_aw: bool = False, clear_cached_aw: bool = False
    ) -> "Any":
        """
        設置 runtime mode（傳遞到所有 QuantModule）。
        """
        for m in self.model.modules():
            if isinstance(m, QuantModule) and hasattr(m, "set_runtime_mode"):
                m.set_runtime_mode(
                    mode=mode, use_cached_aw=use_cached_aw, clear_cached_aw=clear_cached_aw
                )

    def forward(self, *args: "Any", **kwargs: "Any") -> "Any":
        """前向傳播，直接調用原始模型。"""
        return self.model(*args, **kwargs)


def count_quantized_modules(
    model: "Any", target_modules: "Any" = ["input_blocks", "middle_block", "output_blocks", "out"]
) -> "Any":
    """
    統計指定模組中的量化層數量
    """
    total_quant = 0
    total_conv_linear = 0

    logger.info("\n=== 量化統計 ===")

    for module_name in target_modules:
        if hasattr(model, module_name):
            module = getattr(model, module_name)

            quant_count = 0
            conv_linear_count = 0

            for name, child in module.named_modules():
                if isinstance(child, QuantModule):
                    quant_count += 1
                elif isinstance(child, (nn.Conv2d, nn.Linear)):
                    conv_linear_count += 1

            logger.info("%s:", module_name)
            logger.info("  量化層數: %d", quant_count)
            logger.info("  原始 Conv2d/Linear: %d", conv_linear_count)

            total_quant += quant_count
            total_conv_linear += conv_linear_count

    logger.info("\n總計:")
    logger.info("  總量化層數: %d", total_quant)
    logger.info("  總原始層數: %d", total_conv_linear)

    return total_quant, total_conv_linear


def analyze_unet_structure(model: "Any") -> "Any":
    """
    分析 UNet 結構中的 Conv2d 和 Linear 層
    """
    target_modules = ["input_blocks", "middle_block", "output_blocks", "out"]

    logger.info("\n=== UNet 結構分析 ===")

    for module_name in target_modules:
        if hasattr(model, module_name):
            module = getattr(model, module_name)

            conv_layers = []
            linear_layers = []

            for name, child in module.named_modules():
                if isinstance(child, nn.Conv2d):
                    conv_layers.append(name)
                elif isinstance(child, nn.Linear):
                    linear_layers.append(name)

            logger.info("\n%s:", module_name)
            logger.info("  Conv2d 層: %d", len(conv_layers))
            logger.info("  Linear 層: %d", len(linear_layers))

            # 顯示前幾個層的名稱
            if conv_layers:
                logger.info("  前 3 個 Conv2d: %s", conv_layers[:3])
            if linear_layers:
                logger.info("  前 3 個 Linear: %s", linear_layers[:3])

    # 分析非 UNet 部分
    non_unet_modules = ["time_embed", "encoder", "latent_net"]
    logger.info("\n=== 非 UNet 部分 (不量化) ===")

    for module_name in non_unet_modules:
        if hasattr(model, module_name):
            module = getattr(model, module_name)

            conv_count = sum(
                1 for _, child in module.named_modules() if isinstance(child, nn.Conv2d)
            )
            linear_count = sum(
                1 for _, child in module.named_modules() if isinstance(child, nn.Linear)
            )

            if conv_count > 0 or linear_count > 0:
                logger.info(
                    "%s: %d Conv2d, %d Linear (跳過量化)", module_name, conv_count, linear_count
                )
