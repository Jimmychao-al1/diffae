import torch
import torch.nn as nn
import os

def count_layers_from_step5():
    """從 Step 5 的檢查點直接計算可量化層數"""
    print("="*80)
    print("🔍 從 Step 5 檢查點分析 Diff-AE UNet 層數")
    print("="*80)
    
    # Step 5 檢查點路徑
    step5_path = 'QATcode/diffae_unet_quantw8a8_intmodel.pth'
    
    if not os.path.exists(step5_path):
        print(f"❌ Step 5 檢查點不存在: {step5_path}")
        return None
    
    # 載入 Step 5 檢查點
    print(f"載入 Step 5 檢查點: {step5_path}")
    state_dict = torch.load(step5_path, map_location='cpu')
    
    print(f"✅ Step 5 檢查點載入成功")
    
    # 分析 UNet 模組中的量化層
    unet_modules = ['input_blocks', 'middle_block', 'output_blocks', 'out']
    quantizable_layers = []
    
    print(f"\n📊 各模組統計:")
    total_quantizable = 0
    
    for module_name in unet_modules:
        module_layers = []
        
        # 搜尋該模組下的所有層
        for key in state_dict.keys():
            if key.startswith(f'model.{module_name}'):
                # 只看 .weight 參數
                if '.weight' in key and 'delta' not in key and 'zero_point' not in key:
                    layer_name = key.replace('.weight', '')
                    
                    # 排除 skip connection 和其他不量化的層
                    if 'skip' not in layer_name.lower() and 'norm' not in layer_name.lower():
                        # 檢查權重形狀來判斷是 Conv2d 還是 Linear
                        weight = state_dict[key]
                        if len(weight.shape) == 4:  # Conv2d
                            module_layers.append((layer_name, 'Conv2d', weight.shape))
                        elif len(weight.shape) == 2:  # Linear  
                            module_layers.append((layer_name, 'Linear', weight.shape))
        
        quantizable_count = len(module_layers)
        total_quantizable += quantizable_count
        quantizable_layers.extend(module_layers)
        
        print(f"  {module_name:15s}: {quantizable_count:3d} 層")
        
        # 顯示前幾層的詳細信息
        if module_layers:
            print(f"    前3層: {[layer[1] for layer in module_layers[:3]]}")
    
    return quantizable_layers, total_quantizable

def analyze_quantization_parameters(state_dict):
    """分析量化參數"""
    print(f"\n🔧 量化參數分析:")
    
    delta_params = [key for key in state_dict.keys() if 'delta' in key and 'list' not in key]
    zero_point_params = [key for key in state_dict.keys() if 'zero_point' in key]
    
    print(f"  Delta 參數數量: {len(delta_params)}")
    print(f"  Zero_point 參數數量: {len(zero_point_params)}")
    
    if delta_params:
        print(f"  前3個 delta: {delta_params[:3]}")

def generate_step6_config(total_layers):
    """生成 Step 6 配置"""
    print(f"\n" + "="*80)
    print("💡 Step 6 EfficientDM 配置")
    print("="*80)
    
    # EfficientDM 混合量化策略
    special_list = [1, 2, 3, total_layers]
    
    print(f"🎯 建議配置:")
    print(f"```python")
    print(f"self.total_count = {total_layers}")
    print(f"self.special_module_count_list = {special_list}")
    print(f"```")
    print(f"")
    print(f"📋 量化策略:")
    print(f"  - 第 1-3 層: QuantModule (普通量化，保證精度)")
    print(f"  - 第 4-{total_layers-1} 層: QuantModule_intnlora (LoRA量化，可訓練)")
    print(f"  - 第 {total_layers} 層: QuantModule (普通量化，保證輸出)")
    print(f"  - LoRA 層數: {total_layers - 4}")
    
    return special_list

def compare_with_original_efficientdm():
    """與原始 EfficientDM 比較"""
    print(f"\n📈 與原始 EfficientDM 比較:")
    print(f"  LSUN-bedroom: 73 層")
    print(f"  LSUN-church:  109 層") 
    print(f"  ImageNet:     265 層")
    print(f"  Diff-AE:      待計算")

def main():
    """主函數"""
    print("=== 從 Step 5 檢查點分析 Diff-AE UNet 層數 ===\n")
    
    try:
        # 1. 從 Step 5 檢查點計算層數
        layers_info, total_layers = count_layers_from_step5()
        
        if layers_info is None:
            print("❌ 無法分析檢查點")
            return
        
        # 2. 分析量化參數
        step5_path = 'QATcode/diffae_unet_quantw8a8_intmodel.pth'
        state_dict = torch.load(step5_path, map_location='cpu')
        analyze_quantization_parameters(state_dict)
        
        # 3. 生成 Step 6 配置
        special_list = generate_step6_config(total_layers)
        
        # 4. 比較其他模型
        compare_with_original_efficientdm()
        
        # 5. 詳細層信息
        print(f"\n📋 層序列預覽:")
        print(f"總層數: {total_layers}")
        print(f"前5層:")
        for i, (layer_name, layer_type, shape) in enumerate(layers_info[:5], 1):
            simple_name = layer_name.split('.')[-1]
            print(f"  {i:2d}. {layer_type:6s} {simple_name} {shape}")
        
        if total_layers > 5:
            print(f"後5層:")
            for i, (layer_name, layer_type, shape) in enumerate(layers_info[-5:], total_layers-4):
                simple_name = layer_name.split('.')[-1]  
                print(f"  {i:2d}. {layer_type:6s} {simple_name} {shape}")
        
        # 6. 保存結果
        result = {
            'total_count': total_layers,
            'special_module_count_list': special_list,
            'layers_info': layers_info,
            'analysis_method': 'from_step5_checkpoint'
        }
        
        output_path = 'QATcode/diffae_layer_count_step5.pth'
        torch.save(result, output_path)
        
        print(f"\n🎉 分析完成!")
        print(f"✅ Diff-AE UNet 可量化層數: {total_layers}")
        print(f"✅ 建議配置已保存: {output_path}")
        
        return total_layers, special_list
        
    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == '__main__':
    main()