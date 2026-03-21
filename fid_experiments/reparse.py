#!/usr/bin/env python3
"""
重新解析 experiment_results/logs/ 中的 log，更新 CSV 和 JSONL

用於修復之前因為 FID pattern 不匹配導致的空結果
"""

import csv
import json
import re
from pathlib import Path
from typing import Optional

# 從 fid_experiments/ 執行，所以 ROOT 是上一層（DiffAE root）
ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "experiment_results/logs"
CSV_PATH = ROOT / "experiment_results/fid_results.csv"
JSONL_PATH = ROOT / "experiment_results/fid_results.jsonl"

# 完整的 FID patterns（與 run_experiments.py 同步）
FID_PATTERNS = [
    # 格式：FID@50000 100 steps score: 11.09
    re.compile(
        r"FID@\s*(?P<eval>\d+)\s+(?P<steps>\d+)\s+steps\s+score:\s*(?P<fid>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
    # 格式：fid (0): 21.126529693603516
    re.compile(
        r"\bfid\s*\(\s*\d+\s*\)\s*:\s*(?P<fid>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
    # 格式：FID: 11.09 / fid = 11.09
    re.compile(
        r"\bFID\b\s*[:=]\s*(?P<fid>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
]


def parse_fid_from_text(text: str) -> Optional[float]:
    """從 log 文字中提取最後一個 FID"""
    last_fid: Optional[float] = None
    for line in text.splitlines():
        for p in FID_PATTERNS:
            m = p.search(line)
            if m is None:
                continue
            try:
                last_fid = float(m.group("fid"))
            except Exception:
                continue
    return last_fid


def reparse_logs():
    """重新解析所有 log 檔案並更新結果"""
    if not CSV_PATH.exists():
        print(f"❌ CSV 檔案不存在: {CSV_PATH}")
        return
    
    # 讀取現有 CSV（可能沒有 header）
    rows = []
    fieldnames = [
        'timestamp', 'script', 'mode', 'steps', 'k', 'eval_samples',
        'fid', 'status', 'return_code', 'log_path', 'command'
    ]
    
    with CSV_PATH.open('r', encoding='utf-8') as f:
        # 檢查第一行是否是 header
        first_line = f.readline().strip()
        f.seek(0)
        
        if first_line.startswith('timestamp'):
            # 有 header
            reader = csv.DictReader(f)
            rows = list(reader)
        else:
            # 沒有 header，手動指定
            reader = csv.DictReader(f, fieldnames=fieldnames)
            rows = list(reader)
    
    print(f"找到 {len(rows)} 筆記錄")
    print()
    
    updated_count = 0
    for row in rows:
        log_path = ROOT / row['log_path']
        
        # 如果已經有 FID 且 status 是 ok，跳過
        if row['fid'] and row['status'] == 'ok':
            print(f"✓ 跳過（已有 FID）: {log_path.name}")
            continue
        
        # 讀取 log 並重新解析
        if not log_path.exists():
            print(f"⚠ Log 不存在: {log_path}")
            continue
        
        log_text = log_path.read_text(encoding='utf-8', errors='ignore')
        fid = parse_fid_from_text(log_text)
        
        if fid is not None:
            old_fid = row['fid']
            old_status = row['status']
            row['fid'] = f"{fid:.10f}"
            row['status'] = 'ok'
            updated_count += 1
            print(f"✓ 更新: {log_path.name}")
            print(f"  FID: {old_fid or 'N/A'} → {fid:.4f}")
            print(f"  Status: {old_status} → ok")
        else:
            print(f"✗ 未找到 FID: {log_path.name}")
        print()
    
    if updated_count == 0:
        print("沒有需要更新的記錄")
        return
    
    # 備份舊檔案
    if CSV_PATH.exists():
        backup_csv = CSV_PATH.with_suffix('.csv.backup')
        CSV_PATH.rename(backup_csv)
        print(f"已備份 CSV 到: {backup_csv}")
    
    if JSONL_PATH.exists():
        backup_jsonl = JSONL_PATH.with_suffix('.jsonl.backup')
        JSONL_PATH.rename(backup_jsonl)
        print(f"已備份 JSONL 到: {backup_jsonl}")
    
    # 寫回 CSV
    fieldnames = [
        'timestamp', 'script', 'mode', 'steps', 'k', 'eval_samples',
        'fid', 'status', 'return_code', 'log_path', 'command'
    ]
    
    with CSV_PATH.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✅ 已更新 CSV: {CSV_PATH}")
    
    # 寫回 JSONL
    with JSONL_PATH.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    print(f"✅ 已更新 JSONL: {JSONL_PATH}")
    print(f"\n總共更新了 {updated_count} 筆記錄")


if __name__ == '__main__':
    print("=" * 80)
    print("重新解析 FID 結果")
    print("=" * 80)
    print()
    reparse_logs()
