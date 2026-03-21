#!/usr/bin/env python3
"""
顯示 FID 實驗結果摘要
"""

import csv
from pathlib import Path
from collections import defaultdict

# 從 fid_experiments/ 執行，所以 ROOT 是上一層（DiffAE root）
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "experiment_results/fid_results.csv"


def show_summary():
    """顯示結果摘要"""
    if not CSV_PATH.exists():
        print(f"❌ CSV 檔案不存在: {CSV_PATH}")
        return
    
    # 讀取 CSV
    rows = []
    with CSV_PATH.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("沒有任何記錄")
        return
    
    print("=" * 100)
    print(f"FID 實驗結果摘要（共 {len(rows)} 筆記錄）")
    print("=" * 100)
    print()
    
    # 按 mode, steps, k 分組
    groups = defaultdict(list)
    for row in rows:
        key = (row['mode'], int(row['steps']), int(row['k']))
        groups[key].append(row)
    
    # 按 steps 和 k 排序
    sorted_groups = sorted(groups.items(), key=lambda x: (x[0][1], x[0][2], x[0][0]))
    
    # 顯示表格
    print(f"{'Mode':<15} {'T':<5} {'k':<8} {'Samples':<10} {'FID':<15} {'Status':<10}")
    print("-" * 100)
    
    for (mode, steps, k), group_rows in sorted_groups:
        for row in group_rows:
            fid_str = f"{float(row['fid']):.4f}" if row['fid'] else "N/A"
            status_symbol = "✓" if row['status'] == 'ok' else "✗"
            print(f"{row['mode']:<15} {row['steps']:<5} {row['k']}k    {row['eval_samples']:<10} {fid_str:<15} {status_symbol} {row['status']}")
    
    print()
    print("-" * 100)
    
    # 統計
    success_count = sum(1 for r in rows if r['status'] == 'ok')
    failed_count = len(rows) - success_count
    
    print(f"成功: {success_count}, 失敗: {failed_count}")
    
    # 按 T 和 k 分組比較
    print()
    print("=" * 100)
    print("Baseline vs QAT 比較")
    print("=" * 100)
    
    comparison = defaultdict(dict)
    for row in rows:
        if row['status'] == 'ok' and row['fid']:
            key = (int(row['steps']), int(row['k']))
            comparison[key][row['mode']] = float(row['fid'])
    
    if comparison:
        print()
        print(f"{'T':<5} {'k':<8} {'Baseline':<15} {'QAT':<15} {'Diff':<15}")
        print("-" * 100)
        
        for (steps, k), modes in sorted(comparison.items()):
            baseline = modes.get('baseline', None)
            qat = modes.get('float', modes.get('int', None))
            
            baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
            qat_str = f"{qat:.4f}" if qat is not None else "N/A"
            
            if baseline is not None and qat is not None:
                diff = qat - baseline
                diff_str = f"{diff:+.4f}"
            else:
                diff_str = "N/A"
            
            print(f"{steps:<5} {k}k    {baseline_str:<15} {qat_str:<15} {diff_str:<15}")
    
    print()


if __name__ == '__main__':
    show_summary()
