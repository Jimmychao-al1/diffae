import os
import sys
import time

sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from QATcode.quant_layer import QuantModule
from QATcode.quant_utils import assert_finite_tensor, assert_uint8_range, get_default_device
from QATcode.quant_model_selective import SelectiveQuantModel
from templates import *
from templates_latent import *

def setup_first_last_layers(qnn):
    """設置首尾層為 8-bit"""
    print("="*100)
    print("5. Setting first and last layers to 8-bit...")
    
    quant_modules = []
    target_modules = ['time_embed', 'input_blocks', 'middle_block', 'output_blocks', 'out']
    
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

def load_quantized_model():
    """載入 Step 4 的量化模型和配置"""
    print("="*100)
    print("1. Loading quantized model from Step 4...")
    
    # 載入配置
    config_path = 'QATcode/diffae_unet_quant_config.pth'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Quantization config not found: {config_path}")
    
    config = torch.load(config_path, map_location='cpu')
    n_bits_w = config['n_bits_w']
    n_bits_a = config['n_bits_a']
    print(f"✅ 量化配置: Weight={n_bits_w}bit, Activation={n_bits_a}bit")
    
    # 載入原始模型
    print("Loading original Diff-AE model...")
    conf = ffhq128_autoenc_latent()
    model = LitModel(conf)
    ckpt = f'{conf.logdir}/last.ckpt'
    
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'])
    
    diffusion_model = model.ema_model
    assert diffusion_model is not None, "ema_model is None"
    diffusion_model.cuda()
    diffusion_model.eval()
    
    # 重建量化模型
    print("Recreating quantized model...")
    quantize_skip_connections = config.get('quantize_skip_connections', False)
    print(f"Skip connection 量化設定: {'啟用' if quantize_skip_connections else '停用'}")
    
    wq_params = {'n_bits': n_bits_w, 'channel_wise': True, 'scale_method': 'max'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = SelectiveQuantModel(
        model=diffusion_model, 
        weight_quant_params=wq_params, 
        act_quant_params=aq_params, 
        need_init=False,
        quantize_skip_connections=False
    )
    qnn.cuda()
    qnn.eval()

    qnn = setup_first_last_layers(qnn)


    qnn.set_quant_state(weight_quant=True, act_quant=True)
    
    # 載入量化權重
    model_path = f'QATcode/diffae_unet_quantw{n_bits_w}a{n_bits_a}_selective.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Quantized model not found: {model_path}")
    
    print(f"Loading quantized weights from: {model_path}")
    qnn_state = torch.load(model_path, map_location='cpu', weights_only=False)
    qnn.load_state_dict(qnn_state)
    qnn.cuda()
    qnn.eval()
    
    print("✅ Quantized model loaded successfully!")
    return qnn, config
    



def convert_to_integer_weights(qnn, n_bits_w):
    """將 fake quantized weights 轉換為真正的整數權重"""
    print("="*100)
    print("2. Converting fake quantized weights to integer weights...")
    
    # 凍結所有參數
    for param in qnn.parameters():
        param.requires_grad = False
    
    converted_modules = 0
    original_size = 0
    compressed_size = 0
    
    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule) and module.ignore_reconstruction is False:
            print(f"Processing module: {name}")
            
            # 取得原始權重
            x = module.weight
            original_size += x.numel() * 4  # float32 = 4 bytes
            
            # 轉換為整數：確保與推論的 SimpleDequantizer 對齊（無號、通道對齊）
            x_int = torch.round(x / module.weight_quantizer.delta) + module.weight_quantizer.zero_point
            x_quant = torch.clamp(x_int, 0, module.weight_quantizer.n_levels - 1)
            
            # 保存原始形狀
            ori_shape = x_quant.shape
            
            # 對於 Conv2d，需要flatten維度1以後的所有維度
            if module.fwd_func is F.conv2d:
                x_quant = x_quant.flatten(1)
            
            # 打包為整數（針對低位寬量化）
            intweight = x_quant.to(torch.uint8).cpu().numpy()
            
            if n_bits_w == 8:
                # 8-bit 不需要打包
                packed_weight = torch.tensor(intweight).cuda()
                packed_weight = packed_weight.reshape(ori_shape)
                compressed_size += packed_weight.numel() * 1  # uint8 = 1 byte
            else:
                # 對於 4-bit, 2-bit 等需要打包
                elements_per_byte = 8 // n_bits_w
                packed_rows = intweight.shape[0] // elements_per_byte * n_bits_w
                
                qweight = np.zeros((packed_rows, intweight.shape[1]), dtype=np.uint8)
                
                i = 0
                row = 0
                while row < qweight.shape[0]:
                    for j in range(i, i + elements_per_byte):
                        if j < intweight.shape[0]:
                            qweight[row] |= intweight[j] << (n_bits_w * (j - i))
                    i += elements_per_byte
                    row += 1
                
                packed_weight = torch.tensor(qweight).cuda()
                packed_weight = packed_weight.reshape([packed_weight.shape[0]] + list(ori_shape[1:]))
                compressed_size += packed_weight.numel() * 1  # uint8 = 1 byte
            
            # 替換權重（保持現有導出語意）
            module.weight.data = packed_weight
            converted_modules += 1
    
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    
    print(f"✅ 權重轉換完成:")
    print(f"   - 轉換模組數: {converted_modules}")
    print(f"   - 原始大小: {original_size / (1024*1024):.2f} MB")
    print(f"   - 壓縮大小: {compressed_size / (1024*1024):.2f} MB")
    print(f"   - 壓縮比: {compression_ratio:.2f}x")
    
    return converted_modules, compression_ratio



def save_integer_model(qnn, config, converted_modules, compression_ratio):
    """保存整數模型"""
    print("="*100)
    print("4. Saving integer model...")
    
    output_dir = "QATcode"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存整數模型
    n_bits_w = config['n_bits_w']
    n_bits_a = config['n_bits_a']
    model_path = f'{output_dir}/diffae_unet_quantw{n_bits_w}a{n_bits_a}_intmodel.pth'
    
    # 獲取模型大小信息
    model_state = qnn.state_dict()
    torch.save(model_state, model_path)
    
    file_size = os.path.getsize(model_path) / (1024*1024)  # MB
    print(f"✅ Integer model saved: {model_path}")
    print(f"   - 文件大小: {file_size:.2f} MB")
    
    # 更新配置
    config_path = f'{output_dir}/diffae_unet_intmodel_config.pth'
    int_config = config.copy()
    int_config.update({
        'model_type': 'packed_integer',
        'converted_modules': converted_modules,
        'compression_ratio': compression_ratio,
        'file_size_mb': file_size
    })
    torch.save(int_config, config_path)
    print(f"✅ Integer model config saved: {config_path}")
    
    return model_path, config_path, file_size

def main():
    """主函數 - Diff-AE Step 5: 轉換為整數模型"""
    print("=== Diff-AE Step 5: Convert Quantized Model to Integer Model ===")
    
    try:
        start_time = time.time()
        
        # 1. 載入量化模型
        qnn, config = load_quantized_model()

        # 2. 轉換為整數權重
        n_bits_w = config['n_bits_w']
        
        for name, param in qnn.named_parameters():
            param.requires_grad = False

        for name, module in qnn.named_modules():
            if isinstance(module, QuantModule) and module.ignore_reconstruction is False:
                x = module.weight
                x_int = torch.round(x / module.weight_quantizer.delta) + module.weight_quantizer.zero_point
                x_quant = torch.clamp(x_int, 0, module.weight_quantizer.n_levels - 1)
                
                ori_shape = x_quant.shape
                if module.fwd_func is F.conv2d:
                    x_quant = x_quant.flatten(1)
                i = 0
                row = 0
                intweight = x_quant.int().cpu().numpy().astype(np.uint8)
                #qweight = np.zeros(
                #    (intweight.shape[0] // 8 * n_bits_w, intweight.shape[1]), dtype=np.uint8
                #)
                #while row < qweight.shape[0]:
                #    if n_bits_w in [2, 4, 8]:
                #        for j in range(i, i + (8 // n_bits_w)):
                #            qweight[row] |= intweight[j] << (n_bits_w * (j - i))
                #        i += 8 // n_bits_w
                #        row += 1  
#
                #qweight = torch.tensor(qweight).cuda()
                #qweight = qweight.reshape([qweight.shape[0]]+list(ori_shape[1:]))

                #module.weight.data = intweight
        
        qnn_sd = qnn.state_dict()
        torch.save(qnn_sd, 'QATcode/diffae_unet_quantw8a8_intmodel.pth')
        
    except Exception as e:
        print(f"❌ Error in Step 5: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 