#!/bin/bash
# run_quantitative_analysis.sh
# 批量运行定量分析脚本

# 配置
STEPS_LIST=(20 100)
METHODS=("Res" "Att")
THRESHOLDS=(0.01 0.03 0.04 0.05 0.075 0.1)
ANALYSIS_SAMPLES=10  # 生成时间测试的样本数
EVAL_SAMPLES=64      # FID评估样本数（用于快速测试）

echo "=========================================="
echo "Q-DiffAE Quantitative Analysis Batch Run"
echo "=========================================="
echo "Steps: ${STEPS_LIST[@]}"
echo "Methods: ${METHODS[@]}"
echo "Thresholds: ${THRESHOLDS[@]}"
echo "Analysis samples: ${ANALYSIS_SAMPLES}"
echo "=========================================="

# 创建结果目录
mkdir -p QATcode/quantitative_results

# 计数器
total_runs=0
successful_runs=0
failed_runs=0

for STEPS in "${STEPS_LIST[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for THRESHOLD in "${THRESHOLDS[@]}"; do
            total_runs=$((total_runs + 1))
            
            echo ""
            echo "=========================================="
            echo "Run $total_runs: T=${STEPS}, Method=${METHOD}, Threshold=${THRESHOLD}"
            echo "=========================================="
            
            # 运行分析
            python QATcode/sample_lora_intmodel.py \
                --mode int \
                --num_steps ${STEPS} \
                --eval_samples ${EVAL_SAMPLES} \
                --enable_cache \
                --cache_method ${METHOD} \
                --cache_threshold ${THRESHOLD} \
                --enable_quantitative_analysis \
                --analysis_num_samples ${ANALYSIS_SAMPLES} \
                2>&1 | tee QATcode/quantitative_results/run_T${STEPS}_${METHOD}_th${THRESHOLD}.log
            
            # 检查运行结果
            if [ $? -eq 0 ]; then
                successful_runs=$((successful_runs + 1))
                echo "✅ Success: T=${STEPS}, Method=${METHOD}, Threshold=${THRESHOLD}"
            else
                failed_runs=$((failed_runs + 1))
                echo "❌ Failed: T=${STEPS}, Method=${METHOD}, Threshold=${THRESHOLD}"
            fi
            
            # 等待 GPU 释放
            echo "Waiting 5 seconds for GPU to release..."
            sleep 5
        done
    done
done

echo ""
echo "=========================================="
echo "Batch Run Summary"
echo "=========================================="
echo "Total runs: $total_runs"
echo "Successful: $successful_runs"
echo "Failed: $failed_runs"
echo "=========================================="

# 汇总结果
echo ""
echo "Collecting results..."
python3 << EOF
import json
import glob
from pathlib import Path

results_dir = Path("QATcode/quantitative_results")
json_files = list(results_dir.glob("quantitative_analysis_*.json"))

if json_files:
    print(f"Found {len(json_files)} result files")
    
    # 创建汇总
    summary = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                config = data.get('config', {})
                summary.append({
                    'steps': config.get('num_steps'),
                    'method': config.get('cache_method'),
                    'threshold': config.get('cache_threshold'),
                    'original_size_mb': data.get('original_model', {}).get('model_size_mb', 0),
                    'quantized_size_mb': data.get('quantized_model', {}).get('quantized_size_mb', 0),
                    'compression_ratio': data.get('compression_ratio', 0),
                    'original_macs_g': data.get('macs_original', {}).get('total_macs', 0) / 1e9,
                    'quantized_macs_g': data.get('macs_quantized', {}).get('total_macs', 0) / 1e9,
                    'original_time_s': data.get('time_original', {}).get('avg_time_per_image', 0),
                    'quantized_time_s': data.get('time_quantized', {}).get('avg_time_per_image', 0),
                })
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    # 保存汇总
    summary_file = results_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")
else:
    print("No result files found")
EOF

echo ""
echo "Analysis complete! Check QATcode/quantitative_results/ for detailed results."

