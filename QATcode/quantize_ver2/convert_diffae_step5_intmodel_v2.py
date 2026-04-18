"""Convert Step4 quantized checkpoints into Step5 integer-model artifacts."""

import os
import sys
import time
import atexit
from typing import Any

sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from QATcode.quantize_ver2.quant_layer_v2 import QuantModule
from QATcode.quant_utils import assert_finite_tensor, assert_uint8_range, get_default_device
from QATcode.quantize_ver2.quant_model_selective_v2 import SelectiveQuantModel
from templates import *
from templates_latent import *


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: Any) -> Any:
        """Public function write."""
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> Any:
        """Public function flush."""
        for s in self.streams:
            s.flush()


def setup_log_file(log_name: str = "step5.log") -> str:
    """
    Redirect stdout/stderr to console + log file.
    """
    log_dir = os.path.join("QATcode", "quantize_ver2", "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    log_fp = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fp)
    sys.stderr = _Tee(sys.__stderr__, log_fp)
    atexit.register(log_fp.close)
    return log_path


def setup_first_last_layers(qnn: Any) -> Any:
    """設置首尾層為 8-bit"""
    print("=" * 100)
    print("5. Setting first and last layers to 8-bit...")

    quant_modules = []
    target_modules = ["time_embed", "input_blocks", "middle_block", "output_blocks", "out"]

    for module_name in target_modules:
        if hasattr(qnn.model, module_name):
            module = getattr(qnn.model, module_name)
            for name, child in module.named_modules():
                if isinstance(child, QuantModule):
                    quant_modules.append((f"{module_name}.{name}", child))

    # 設置前 3 層為 8-bit
    for i in range(min(3, len(quant_modules))):
        name, module = quant_modules[i]
        module.weight_quantizer.bitwidth_refactor(8)
        module.act_quantizer.bitwidth_refactor(8)
        module.ignore_reconstruction = True
        print(f"  Set layer {i} ({name}) to 8-bit")

    # 設置最後一層為 8-bit
    if len(quant_modules) > 3:
        name, module = quant_modules[-1]
        module.weight_quantizer.bitwidth_refactor(8)
        module.act_quantizer.bitwidth_refactor(8)
        module.ignore_reconstruction = True
        print(f"  Set last layer ({name}) to 8-bit")
    return qnn


def load_quantized_model() -> Any:
    """載入 Step 4 的量化模型和配置"""
    print("=" * 100)
    print("1. Loading quantized model from Step 4...")

    # 載入配置
    config_path = "QATcode/quantize_ver2/diffae_unet_quantw8a8_selective_config.pth"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Quantization config not found: {config_path}")

    config = torch.load(config_path, map_location="cpu")
    n_bits_w = config["n_bits_w"]
    n_bits_a = config["n_bits_a"]
    print(f"✅ 量化配置: Weight={n_bits_w}bit, Activation={n_bits_a}bit")

    # 載入原始模型
    print("Loading original Diff-AE model...")
    conf = ffhq128_autoenc_latent()
    model = LitModel(conf)
    ckpt = f"{conf.logdir}/last.ckpt"

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])

    diffusion_model = model.ema_model
    assert diffusion_model is not None, "ema_model is None"
    diffusion_model.cuda()
    diffusion_model.eval()

    # 重建量化模型
    print("Recreating quantized model...")
    quantize_skip_connections = config.get("quantize_skip_connections", False)
    print(f"Skip connection 量化設定: {'啟用' if quantize_skip_connections else '停用'}")

    wq_params = {"n_bits": n_bits_w, "channel_wise": True, "scale_method": "max"}
    aq_params = {
        "n_bits": n_bits_a,
        "channel_wise": False,
        "scale_method": "mse",
        "leaf_param": True,
    }
    qnn = SelectiveQuantModel(
        model=diffusion_model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        need_init=False,
        quantize_skip_connections=False,
    )
    qnn.cuda()
    qnn.eval()

    qnn = setup_first_last_layers(qnn)

    qnn.set_quant_state(weight_quant=True, act_quant=True)

    # 載入量化權重
    model_path = f"QATcode/quantize_ver2/diffae_unet_quantw8a8_selective.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Quantized model not found: {model_path}")

    print(f"Loading quantized weights from: {model_path}")
    qnn_state = torch.load(model_path, map_location="cpu")
    qnn.load_state_dict(qnn_state)
    qnn.cuda()
    qnn.eval()

    print("✅ Quantized model loaded successfully!")
    return qnn, config


def main() -> Any:
    """主函數 - Diff-AE Step 5: 轉換為整數模型"""
    log_path = setup_log_file("step5.log")
    print("=== Diff-AE Step 5: Convert Quantized Model to Integer Model ===")
    print(f"📝 Log file: {log_path}")

    try:
        start_time = time.time()

        # 1. 載入量化模型
        qnn, config = load_quantized_model()

        # 2. 轉換為整數權重
        n_bits_w = config["n_bits_w"]

        for name, param in qnn.named_parameters():
            param.requires_grad = False

        for name, module in qnn.named_modules():
            if isinstance(module, QuantModule) and module.ignore_reconstruction is False:
                x = module.weight

                # module.weight.data = intweight

        qnn_sd = qnn.state_dict()
        torch.save(qnn_sd, "QATcode/quantize_ver2/diffae_unet_quantw8a8_intmodel.pth")

    except Exception as e:
        print(f"❌ Error in Step 5: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
