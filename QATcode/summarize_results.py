#!/usr/bin/env python3
"""
彙總定量分析結果
"""

import json
import glob
from pathlib import Path
import pandas as pd

def safe_float(value, default=0.0):
    """安全地將值轉換爲浮點數"""
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    return float(value)

def safe_bool(value, default=False):
    """安全地將值轉換爲布爾值"""
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    return bool(value)

def summarize_results(results_dir="QATcode"):
    """彙總所有分析結果"""
    
    # 查找所有 JSON 結果文件
    json_files = list(Path(results_dir).glob("quantitative_analysis_*.json"))
    
    if not json_files:
        print("❌ 未找到結果文件")
        return
    
    print(f"✅ 找到 {len(json_files)} 個結果文件\n")
    
    results = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                config = data.get('config', {})
                
                result = {
                    'steps': int(safe_float(config.get('num_steps'), 0)),
                    'method': config.get('cache_method'),
                    'threshold': safe_float(config.get('cache_threshold'), 0.0),
                    'enable_cache': safe_bool(config.get('enable_cache', False)),
                    
                    # 模型大小
                    'original_size_mb': round(safe_float(data.get('original_model', {}).get('model_size_mb', 0)), 2),
                    'quantized_size_mb': round(safe_float(data.get('quantized_model', {}).get('quantized_size_mb', 0)), 2),
                    'compression_ratio': round(safe_float(data.get('compression_ratio', 0)), 2),
                    
                    # MACs
                    'original_macs_g': round(safe_float(data.get('macs_original', {}).get('total_macs', 0)) / 1e9, 2),
                    'quantized_macs_g': round(safe_float(data.get('macs_quantized', {}).get('total_macs', 0)) / 1e9, 2),
                    'quantized_cache_macs_g': round(
                        safe_float(data.get('macs_quantized_cache', {}).get('total_macs_all_timesteps', 0)) / 1e9, 2
                    ) if data.get('macs_quantized_cache') else None,
                    'cache_reduction_ratio': round(
                        safe_float(data.get('macs_quantized_cache', {}).get('cache_reduction_ratio', 0)) * 100, 1
                    ) if data.get('macs_quantized_cache') else None,
                    
                    # 生成時間
                    'original_time_s': round(safe_float(data.get('time_original', {}).get('avg_time_per_image', 0)), 3),
                    'quantized_time_s': round(safe_float(data.get('time_quantized', {}).get('avg_time_per_image', 0)), 3),
                    'quantized_cache_time_s': round(
                        safe_float(data.get('time_quantized_cache', {}).get('avg_time_per_image', 0)), 3
                    ) if data.get('time_quantized_cache') else None,
                    
                    # Data Transfer
                    'original_transfer_mb': round(safe_float(data.get('data_transfer_original', {}).get('total_transfer_mb', 0)), 2),
                    'quantized_transfer_mb': round(safe_float(data.get('data_transfer_quantized', {}).get('total_transfer_mb', 0)), 2),
                }
                
                results.append(result)
        except Exception as e:
            print(f"⚠️ 讀取 {json_file} 失敗: {e}")
    
    if not results:
        print("❌ 沒有有效的結果")
        return
    
    # 轉換爲 DataFrame
    df = pd.DataFrame(results)
    
    # 排序
    df = df.sort_values(['steps', 'method', 'threshold'])
    
    # 保存爲 CSV
    csv_file = Path(results_dir) / "quantitative_results_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ 結果已保存到: {csv_file}\n")
    
    # 打印摘要表格
    print("=" * 120)
    print("定量分析結果彙總")
    print("=" * 120)
    
    # 按 steps 和 method 分組顯示
    for steps in sorted(df['steps'].unique()):
        print(f"\n📊 Steps = {steps}")
        print("-" * 120)
        
        for method in sorted(df[df['steps'] == steps]['method'].unique()):
            print(f"\n  Method = {method}")
            print("  " + "-" * 116)
            
            subset = df[(df['steps'] == steps) & (df['method'] == method)]
            
            # 打印表頭 - 顯示所有時間信息
            print(f"  {'Threshold':<10} {'Size(MB)':<15} {'Compress':<10} {'MACs(G)':<25} {'Time: Orig/Q/Q+C':<40} {'Transfer: Orig/Q':<20}")
            print("  " + "-" * 120)
            
            for _, row in subset.iterrows():
                size_info = f"{row['quantized_size_mb']:.1f}/{row['original_size_mb']:.1f}"
                macs_info = f"{row['quantized_macs_g']:.1f}"
                if row['quantized_cache_macs_g']:
                    macs_info += f" ({row['quantized_cache_macs_g']:.1f} cache)"
                
                # 顯示三個時間：Original / Quantized / Quantized+Cache
                time_info = f"{row['original_time_s']:.3f}/{row['quantized_time_s']:.3f}"
                if row['quantized_cache_time_s']:
                    time_info += f"/{row['quantized_cache_time_s']:.3f}"
                else:
                    time_info += "/-"
                
                # Data Transfer: Original / Quantized
                transfer_info = f"{row['original_transfer_mb']:.2f}/{row['quantized_transfer_mb']:.2f}"
                
                print(f"  {row['threshold']:<10.3f} {size_info:<15} {row['compression_ratio']:<10.2f}x "
                      f"{macs_info:<25} {time_info:<40} {transfer_info:<20}")
    
    print("\n" + "=" * 120)
    
    # 找出最佳配置
    print("\n🏆 最佳配置:")
    print("-" * 120)
    
    # 最小模型大小
    min_size_idx = df['quantized_size_mb'].idxmin()
    print(f"最小模型大小: {df.loc[min_size_idx, 'quantized_size_mb']:.2f} MB "
          f"(T={df.loc[min_size_idx, 'steps']}, {df.loc[min_size_idx, 'method']}, "
          f"th={df.loc[min_size_idx, 'threshold']})")
    
    # 最快生成時間（如果有 cache）
    cache_df = df[df['quantized_cache_time_s'].notna()]
    if len(cache_df) > 0:
        min_time_idx = cache_df['quantized_cache_time_s'].idxmin()
        print(f"最快生成時間: {cache_df.loc[min_time_idx, 'quantized_cache_time_s']:.3f}s "
              f"(T={cache_df.loc[min_time_idx, 'steps']}, {cache_df.loc[min_time_idx, 'method']}, "
              f"th={cache_df.loc[min_time_idx, 'threshold']})")
    
    # 最大 cache reduction
    cache_reduction_df = df[df['cache_reduction_ratio'].notna()]
    if len(cache_reduction_df) > 0:
        max_reduction_idx = cache_reduction_df['cache_reduction_ratio'].idxmax()
        print(f"最大 Cache 減少: {cache_reduction_df.loc[max_reduction_idx, 'cache_reduction_ratio']:.1f}% "
              f"(T={cache_reduction_df.loc[max_reduction_idx, 'steps']}, "
              f"{cache_reduction_df.loc[max_reduction_idx, 'method']}, "
              f"th={cache_reduction_df.loc[max_reduction_idx, 'threshold']})")
    
    print("=" * 120)

if __name__ == "__main__":
    summarize_results()

