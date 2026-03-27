import json
import ast
from collections import defaultdict
from pathlib import Path

def parse_cache_scheduler(cache_scheduler_str):
    """解析 cache_scheduler 字串格式的集合"""
    if isinstance(cache_scheduler_str, str):
        return ast.literal_eval(cache_scheduler_str)
    return cache_scheduler_str

def infer_timestep_embed_sequential_transfer(data_transfer_quantized):
    """
    從量化層的資料推斷 TimestepEmbedSequential 的資料傳輸
    
    根據需求：Q-DiffAE float 中所有 TimestepEmbedSequential 的輸入/輸出總和。
    
    每個 TimestepEmbedSequential block 的資料傳輸應該是：
    - Block 的輸入：進入此 block 的 feature map（通常是第一個量化層的輸入）
    - Block 的輸出：離開此 block 的 feature map（通常是最後一個量化層的輸出，或 skip_connection 的輸出）
    
    策略：
    1. 對於每個 block，找到第一個量化層的輸入作為 block 的輸入
    2. 找到最後一個量化層的輸出作為 block 的輸出
    3. 如果有 skip_connection，使用它的輸出作為 block 的輸出（因為它代表最終輸出）
    """
    layer_breakdown = data_transfer_quantized.get('layer_breakdown', {})
    
    # 建立 TimestepEmbedSequential 到其內部量化層的對應映射
    sequential_layers = defaultdict(list)
    
    for layer_name, layer_info in layer_breakdown.items():
        # 擷取 TimestepEmbedSequential 名稱
        parts = layer_name.split('.')
        if 'input_blocks' in parts:
            idx = parts.index('input_blocks')
            try:
                block_idx = int(parts[idx+1])
                seq_name = f'input_blocks.{block_idx}'
            except (ValueError, IndexError):
                continue
        elif 'output_blocks' in parts:
            idx = parts.index('output_blocks')
            try:
                block_idx = int(parts[idx+1])
                seq_name = f'output_blocks.{block_idx}'
            except (ValueError, IndexError):
                continue
        elif 'middle_block' in parts:
            seq_name = 'middle_block'
        else:
            continue
        
        # 判斷層的順序：使用層名稱的深度和位置
        # 例如：input_blocks.1.0.in_layers.2 在 input_blocks.1.0.out_layers.3 之前
        is_skip = 'skip_connection' in layer_name
        is_in_layer = 'in_layers' in layer_name
        is_out_layer = 'out_layers' in layer_name
        
        # 計算層的順序：in_layers 在前，out_layers 在後，skip_connection 最後
        if is_skip:
            order = 2
        elif is_out_layer:
            order = 1
        elif is_in_layer:
            order = 0
        else:
            order = 0  # emb_layers, cond_emb_layers 等
        
        sequential_layers[seq_name].append({
            'name': layer_name,
            'order': order,
            'input_size_mb': float(layer_info['input_size_mb']),
            'output_size_mb': float(layer_info['output_size_mb']),
            'is_skip': is_skip,
            'layer_type': layer_info['layer_type']
        })
    
    sequential_transfers = {}
    
    for seq_name, layers in sequential_layers.items():
        if not layers:
            continue
        
        # 按順序排序
        layers_sorted = sorted(layers, key=lambda x: (x['order'], x['name']))
        
        # Block 的輸入：第一個層的輸入
        first_layer = layers_sorted[0]
        block_input = first_layer['input_size_mb']
        
        # Block 的輸出：最後一個層的輸出，或 skip_connection 的輸出（如果存在）
        skip_layers = [l for l in layers if l['is_skip']]
        if skip_layers:
            # 如果有 skip_connection，使用它的輸出
            skip_layer = skip_layers[0]
            block_output = skip_layer['output_size_mb']
        else:
            # 否則使用最後一個層的輸出
            last_layer = layers_sorted[-1]
            block_output = last_layer['output_size_mb']
        
        sequential_transfers[seq_name] = {
            'input_size_mb': block_input,
            'output_size_mb': block_output,
            'total_size_mb': block_input + block_output,
            'layer_type': 'TimestepEmbedSequential',
            'num_quantized_layers': len(layers)
        }
    
    total_input = sum(info['input_size_mb'] for info in sequential_transfers.values())
    total_output = sum(info['output_size_mb'] for info in sequential_transfers.values())
    total_transfer = total_input + total_output
    
    return {
        'total_input_size_mb': total_input,
        'total_output_size_mb': total_output,
        'total_transfer_mb': total_transfer,
        'layer_breakdown': sequential_transfers
    }

def calculate_cached_quantized_transfer(data_transfer_quantized, cache_scheduler, num_steps):
    """
    計算使用 cache 後的量化層資料傳輸
    根據 cache_scheduler 決定哪些層在哪些 timestep 需要重新計算
    """
    layer_breakdown = data_transfer_quantized.get('layer_breakdown', {})
    
    # 建立 layer 名稱到 cache_key 的對應映射
    # cache_key 格式: encoder_layer_0, decoder_layer_0, middle_layer
    layer_to_cache_key = {}
    
    for layer_name in layer_breakdown.keys():
        parts = layer_name.split('.')
        if 'input_blocks' in parts:
            idx = parts.index('input_blocks')
            try:
                block_idx = int(parts[idx+1])
                layer_to_cache_key[layer_name] = f'encoder_layer_{block_idx}'
            except (ValueError, IndexError):
                pass
        elif 'output_blocks' in parts:
            idx = parts.index('output_blocks')
            try:
                block_idx = int(parts[idx+1])
                layer_to_cache_key[layer_name] = f'decoder_layer_{block_idx}'
            except (ValueError, IndexError):
                pass
        elif 'middle_block' in parts:
            layer_to_cache_key[layer_name] = 'middle_layer'
    
    # 計算每個層在所有 timestep 的累積資料傳輸
    cached_layer_transfers = {}
    timestep_breakdown = defaultdict(lambda: {
        'layers_executed': 0,
        'input_size_mb': 0.0,
        'output_size_mb': 0.0
    })
    
    for layer_name, layer_info in layer_breakdown.items():
        cache_key = layer_to_cache_key.get(layer_name)
        if cache_key and cache_key in cache_scheduler:
            recompute_timesteps = parse_cache_scheduler(cache_scheduler[cache_key])
            num_recomputes = len(recompute_timesteps)
            
            # 單次 forward pass 的資料傳輸
            single_input = float(layer_info['input_size_mb'])
            single_output = float(layer_info['output_size_mb'])
            single_total = float(layer_info['total_size_mb'])
            
            # 累積所有需要重新計算的 timestep
            total_input = single_input * num_recomputes
            total_output = single_output * num_recomputes
            total_transfer = single_total * num_recomputes
            
            cached_layer_transfers[layer_name] = {
                'input_size_mb': total_input,
                'output_size_mb': total_output,
                'total_size_mb': total_transfer,
                'layer_type': layer_info['layer_type'],
                'execution_count': num_recomputes,
                'single_pass_input_mb': single_input,
                'single_pass_output_mb': single_output
            }
            
            # 記錄每個 timestep 的統計
            for t in recompute_timesteps:
                timestep_breakdown[t]['layers_executed'] += 1
                timestep_breakdown[t]['input_size_mb'] += single_input
                timestep_breakdown[t]['output_size_mb'] += single_output
        else:
            # 如果沒有對應的 cache_key，假設所有 timestep 都執行
            cached_layer_transfers[layer_name] = {
                'input_size_mb': float(layer_info['input_size_mb']) * num_steps,
                'output_size_mb': float(layer_info['output_size_mb']) * num_steps,
                'total_size_mb': float(layer_info['total_size_mb']) * num_steps,
                'layer_type': layer_info['layer_type'],
                'execution_count': num_steps,
                'single_pass_input_mb': float(layer_info['input_size_mb']),
                'single_pass_output_mb': float(layer_info['output_size_mb'])
            }
    
    total_input = sum(info['input_size_mb'] for info in cached_layer_transfers.values())
    total_output = sum(info['output_size_mb'] for info in cached_layer_transfers.values())
    total_transfer = total_input + total_output
    
    return {
        'total_input_size_mb': total_input,
        'total_output_size_mb': total_output,
        'total_transfer_mb': total_transfer,
        'layer_breakdown': cached_layer_transfers,
        'timestep_breakdown': {str(k): v for k, v in timestep_breakdown.items()}
    }

def derive_data_transfer_results(json_file_path):
    """從現有的 JSON 文件推導三個 data transfer 結果"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    config = data.get('config', {})
    data_transfer_quantized = data.get('data_transfer_quantized', {})
    cache_scheduler = config.get('cache_scheduler', {})
    num_steps = int(config.get('num_steps', 20))
    
    results = {}
    
    # 1. TimestepEmbedSequential 的資料傳輸
    results['data_transfer_timestep_embed_sequential'] = infer_timestep_embed_sequential_transfer(
        data_transfer_quantized
    )
    
    # 2. 所有量化層的資料傳輸（需要乘以 timestep 數量）
    # 從現有資料推導所有 timestep 的總和
    single_timestep_input = float(data_transfer_quantized['total_input_size_mb'])
    single_timestep_output = float(data_transfer_quantized['total_output_size_mb'])
    single_timestep_transfer = float(data_transfer_quantized['total_transfer_mb'])
    
    # 計算所有 timestep 的總和
    total_input_all_steps = single_timestep_input * num_steps
    total_output_all_steps = single_timestep_output * num_steps
    total_transfer_all_steps = single_timestep_transfer * num_steps
    
    # 更新 layer_breakdown 為所有 timestep 的總和
    layer_breakdown_all_steps = {}
    for layer_name, layer_info in data_transfer_quantized.get('layer_breakdown', {}).items():
        layer_breakdown_all_steps[layer_name] = {
            'input_size_mb': float(layer_info['input_size_mb']) * num_steps,
            'output_size_mb': float(layer_info['output_size_mb']) * num_steps,
            'total_size_mb': float(layer_info['total_size_mb']) * num_steps,
            'layer_type': layer_info['layer_type'],
            'single_timestep_input_mb': float(layer_info['input_size_mb']),
            'single_timestep_output_mb': float(layer_info['output_size_mb']),
            'single_timestep_total_mb': float(layer_info['total_size_mb'])
        }
    
    results['data_transfer_quantized_layers'] = {
        'total_input_size_mb': total_input_all_steps,
        'total_output_size_mb': total_output_all_steps,
        'total_transfer_mb': total_transfer_all_steps,
        'single_timestep_input_mb': single_timestep_input,
        'single_timestep_output_mb': single_timestep_output,
        'single_timestep_transfer_mb': single_timestep_transfer,
        'num_steps': num_steps,
        'layer_breakdown': layer_breakdown_all_steps
    }
    
    # 2.5. 所有量化層的資料傳輸（排除 skip_connection 和 LoRA，所有 timestep 總和）
    layer_breakdown_no_skip = {}
    for layer_name, layer_info in data_transfer_quantized.get('layer_breakdown', {}).items():
        # 排除 skip_connection 和 LoRA 層
        if 'skip_connection' not in layer_name:
            if 'lora' not in layer_name.lower() or ('loraA' not in layer_name and 'loraB' not in layer_name):
                layer_breakdown_no_skip[layer_name] = {
                    'input_size_mb': float(layer_info['input_size_mb']) * num_steps,
                    'output_size_mb': float(layer_info['output_size_mb']) * num_steps,
                    'total_size_mb': float(layer_info['total_size_mb']) * num_steps,
                    'layer_type': layer_info['layer_type'],
                    'single_timestep_input_mb': float(layer_info['input_size_mb']),
                    'single_timestep_output_mb': float(layer_info['output_size_mb']),
                    'single_timestep_total_mb': float(layer_info['total_size_mb'])
                }
    
    # 計算排除 skip_connection 和 LoRA 的總和
    single_timestep_input_no_skip = sum(float(info['input_size_mb']) for name, info in data_transfer_quantized.get('layer_breakdown', {}).items() 
                                        if 'skip_connection' not in name and ('lora' not in name.lower() or ('loraA' not in name and 'loraB' not in name)))
    single_timestep_output_no_skip = sum(float(info['output_size_mb']) for name, info in data_transfer_quantized.get('layer_breakdown', {}).items() 
                                         if 'skip_connection' not in name and ('lora' not in name.lower() or ('loraA' not in name and 'loraB' not in name)))
    single_timestep_transfer_no_skip = single_timestep_input_no_skip + single_timestep_output_no_skip
    
    results['data_transfer_quantized_layers_no_skip'] = {
        'total_input_size_mb': single_timestep_input_no_skip * num_steps,
        'total_output_size_mb': single_timestep_output_no_skip * num_steps,
        'total_transfer_mb': single_timestep_transfer_no_skip * num_steps,
        'single_timestep_input_mb': single_timestep_input_no_skip,
        'single_timestep_output_mb': single_timestep_output_no_skip,
        'single_timestep_transfer_mb': single_timestep_transfer_no_skip,
        'num_steps': num_steps,
        'layer_breakdown': layer_breakdown_no_skip
    }
        
        # 3. 使用 cache 後的量化層資料傳輸
    if cache_scheduler:
        results['data_transfer_cached_quantized_layers'] = calculate_cached_quantized_transfer(
            data_transfer_quantized, cache_scheduler, num_steps
        )
    else:
        results['data_transfer_cached_quantized_layers'] = None
    
    return results, data

# 使用範例
if __name__ == '__main__':
    import sys
    
    # 處理所有相關的 JSON 文件
    json_files = [
        'QATcode/quantitative_analysis_T20_Res_th0.1.json',
        'QATcode/quantitative_analysis_T20_Att_th0.1.json',
        'QATcode/quantitative_analysis_T100_Res_th0.1.json',
        'QATcode/quantitative_analysis_T100_Att_th0.1.json',
    ]
    
    if len(sys.argv) > 1:
        json_files = sys.argv[1:]
    
    for json_file in json_files:
        json_path = Path(json_file)
        if not json_path.exists():
            print(f"⚠️  文件不存在: {json_file}")
            continue
        
        print("=" * 80)
        print(f"處理文件: {json_file}")
        print("=" * 80)
        
        try:
            results, original_data = derive_data_transfer_results(json_file)
            
            print("\n1. TimestepEmbedSequential Data Transfer:")
            tes = results['data_transfer_timestep_embed_sequential']
            print(f"   Total input: {tes['total_input_size_mb']:.2f} MB")
            print(f"   Total output: {tes['total_output_size_mb']:.2f} MB")
            print(f"   Total transfer: {tes['total_transfer_mb']:.2f} MB")
            print(f"   Number of blocks: {len(tes['layer_breakdown'])}")
            
            print("\n2. Quantized Layers Data Transfer (All Timesteps):")
            ql = results['data_transfer_quantized_layers']
            num_steps = ql.get('num_steps', int(original_data.get('config', {}).get('num_steps', 20)))
            print(f"   Single timestep - Input: {ql['single_timestep_input_mb']:.2f} MB, Output: {ql['single_timestep_output_mb']:.2f} MB, Total: {ql['single_timestep_transfer_mb']:.2f} MB")
            print(f"   All {num_steps} timesteps - Input: {ql['total_input_size_mb']:.2f} MB, Output: {ql['total_output_size_mb']:.2f} MB, Total: {ql['total_transfer_mb']:.2f} MB")
            print(f"   Number of quantized layers: {len(ql.get('layer_breakdown', {}))}")
            
            print("\n2.5. Quantized Layers Data Transfer (No Skip Connection, All Timesteps):")
            ql_no_skip = results.get('data_transfer_quantized_layers_no_skip', {})
            if ql_no_skip:
                print(f"   Single timestep - Input: {ql_no_skip['single_timestep_input_mb']:.2f} MB, Output: {ql_no_skip['single_timestep_output_mb']:.2f} MB, Total: {ql_no_skip['single_timestep_transfer_mb']:.2f} MB")
                print(f"   All {num_steps} timesteps - Input: {ql_no_skip['total_input_size_mb']:.2f} MB, Output: {ql_no_skip['total_output_size_mb']:.2f} MB, Total: {ql_no_skip['total_transfer_mb']:.2f} MB")
                print(f"   Number of quantized layers (no skip): {len(ql_no_skip.get('layer_breakdown', {}))}")
            
            if results['data_transfer_cached_quantized_layers']:
                print("\n3. Cached Quantized Layers Data Transfer:")
                cql = results['data_transfer_cached_quantized_layers']
                print(f"   Total input: {cql['total_input_size_mb']:.2f} MB")
                print(f"   Total output: {cql['total_output_size_mb']:.2f} MB")
                print(f"   Total transfer: {cql['total_transfer_mb']:.2f} MB")
                
                # 計算 cache 減少的比例
                # original_total 應該是所有 timestep 的總和（已經在 ql 中計算了）
                original_total = ql['total_transfer_mb']  # 已經是所有 timestep 的總和
                cached_total = cql['total_transfer_mb']
                reduction_ratio = (1 - cached_total / original_total) * 100 if original_total > 0 else 0
                print(f"   Cache reduction: {reduction_ratio:.1f}%")
                print(f"   Original total ({num_steps} steps): {original_total:.2f} MB")
                print(f"   Cached total: {cached_total:.2f} MB")
            
            # 保存結果到新文件
            output_file = json_path.parent / f"{json_path.stem}_derived.json"
            original_data.update(results)
            with open(output_file, 'w') as f:
                json.dump(original_data, f, indent=2)
            print(f"\n✅ 結果已保存到: {output_file}")
            
        except Exception as e:
            print(f"❌ 處理失敗: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n")

