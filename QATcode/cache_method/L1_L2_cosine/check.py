import torch
import numpy as np
from pathlib import Path
import argparse

# --------- basic utils ---------
def _flatten(x: torch.Tensor) -> torch.Tensor:
    x = x.detach()
    x = x.reshape(-1).float()
    x = x[torch.isfinite(x)]
    return x

def pearson_corr(a: torch.Tensor, b: torch.Tensor, eps=1e-12) -> float:
    a = _flatten(a)
    b = _flatten(b)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.std(unbiased=False) * b.std(unbiased=False) + eps)
    return float((a * b).mean() / denom)

def rankdata_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Spearman ranking (dense ranks) using argsort twice.
    Tie-handling: not perfect average-rank, but good enough for redundancy check.
    """
    x = _flatten(x)
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(len(x), device=x.device, dtype=torch.float32)
    return ranks

def spearman_corr(a: torch.Tensor, b: torch.Tensor, eps=1e-12) -> float:
    ra = rankdata_torch(a)
    rb = rankdata_torch(b)
    return pearson_corr(ra, rb, eps=eps)

# --------- decision overlap ---------
def topk_jaccard(a: torch.Tensor, b: torch.Tensor, k: int, largest=True) -> float:
    """
    Compare the selected step indices of top-k (largest or smallest).
    """
    a = _flatten(a)
    b = _flatten(b)
    k = min(k, a.numel(), b.numel())
    ia = torch.topk(a, k=k, largest=largest).indices
    ib = torch.topk(b, k=k, largest=largest).indices
    sa = set(ia.cpu().tolist())
    sb = set(ib.cpu().tolist())
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(union, 1)

def percentile_threshold_agreement(a: torch.Tensor, b: torch.Tensor, p=90.0, larger_is_refresh=True) -> float:
    """
    threshold by percentile, compare boolean mask agreement.
    """
    a = _flatten(a)
    b = _flatten(b)

    ta = torch.quantile(a, p / 100.0)
    tb = torch.quantile(b, p / 100.0)

    if larger_is_refresh:
        ma = a >= ta
        mb = b >= tb
    else:
        ma = a <= ta
        mb = b <= tb

    agree = (ma == mb).float().mean()
    return float(agree)

# --------- main report ---------
def redundancy_report(a: torch.Tensor, b: torch.Tensor, name_a="L1rel_sym", name_b="L1rel_asym", return_str=False):
    """
    生成冗餘性報告
    
    Args:
        a: 第一個張量
        b: 第二個張量
        name_a: 第一個張量的名稱
        name_b: 第二個張量的名稱
        return_str: 如果為 True，返回字符串而不是打印
    
    Returns:
        如果 return_str=True，返回報告字符串；否則返回 None
    """
    lines = []
    lines.append(f"[Compare] {name_a} vs {name_b}")
    lines.append(f"Pearson  : {pearson_corr(a, b):.4f}")
    lines.append(f"Spearman : {spearman_corr(a, b):.4f}")
    lines.append("")

    # Top-K overlap (refresh = larger change)
    for k in [5, 10, 20, 50]:
        lines.append(f"Top-{k} Jaccard (largest): {topk_jaccard(a, b, k=k, largest=True):.4f}")

    # Threshold agreement
    lines.append("")
    for p in [80, 90, 95]:
        lines.append(f"Percentile-{p} agreement (largest): {percentile_threshold_agreement(a, b, p=p, larger_is_refresh=True):.4f}")
    
    report_str = "\n".join(lines)
    
    if return_str:
        return report_str
    else:
        print(report_str)
        return None

# ---------------- Main function ----------------
def load_npz_data(npz_path: str, block_name: str = None):
    """
    從 NPZ 文件加載 L1rel 和 L1rel Rate 數據
    
    Args:
        npz_path: NPZ 文件路徑
        block_name: 可選的 block 名稱，如果提供則只加載該 block 的數據
    
    Returns:
        dict: 包含 'l1rel' 和 'l1rel_rate' 的字典，如果 block_name 提供則返回單個 block 的數據
    """
    data = np.load(npz_path)
    
    if block_name:
        # 單個 block 的數據
        if 'l1rel' in data and 'l1rel_rate' in data:
            return {
                'l1rel': torch.from_numpy(data['l1rel']),
                'l1rel_rate': torch.from_numpy(data['l1rel_rate'])
            }
        else:
            raise ValueError(f"NPZ 文件中未找到 'l1rel' 或 'l1rel_rate' 鍵")
    else:
        # 返回所有數據
        result = {}
        for key in data.keys():
            result[key] = torch.from_numpy(data[key])
        return result

def check_single_block(npz_path: str, block_name: str, output_file=None):
    """檢查單個 block 的冗餘性"""
    header = f"\n{'='*60}\n檢查 Block: {block_name}\n數據來源: {npz_path}\n{'='*60}\n"
    
    if output_file:
        output_file.write(header)
        print(header, end='')
    else:
        print(header, end='')
    
    data = load_npz_data(npz_path, block_name=block_name)
    l1rel = data['l1rel']
    l1rel_rate = data['l1rel_rate']
    
    report_str = redundancy_report(l1rel, l1rel_rate, name_a="L1rel", name_b="L1rel Rate", return_str=True)
    
    if output_file:
        output_file.write(report_str + "\n\n")
    else:
        print(report_str)

def check_all_blocks(npz_dir: str, output_file=None):
    """檢查目錄中所有 NPZ 文件的冗餘性"""
    npz_dir = Path(npz_dir)
    npz_files = sorted(npz_dir.glob("*.npz"))
    
    if not npz_files:
        msg = f"在 {npz_dir} 中未找到 NPZ 文件"
        if output_file:
            output_file.write(msg + "\n")
        print(msg)
        return
    
    msg = f"\n找到 {len(npz_files)} 個 NPZ 文件\n"
    if output_file:
        output_file.write(msg)
    print(msg, end='')
    
    for npz_path in npz_files:
        block_name = npz_path.stem
        try:
            check_single_block(str(npz_path), block_name, output_file=output_file)
        except Exception as e:
            error_msg = f"處理 {block_name} 時出錯: {e}\n"
            if output_file:
                output_file.write(error_msg)
            print(error_msg, end='')
            continue

def main():
    parser = argparse.ArgumentParser(
        description="檢查 L1rel 和 L1rel Rate 的冗餘性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 檢查單個 block
  python check.py --npz path/to/block.npz --block model_input_blocks_0
  
  # 檢查目錄中所有 NPZ 文件
  python check.py --npz_dir path/to/npz/directory
  
  # 直接比較兩個張量（用於測試）
  python check.py --test
        """
    )
    
    parser.add_argument('--npz', type=str, help='NPZ 文件路徑')
    parser.add_argument('--block', type=str, help='Block 名稱（可選，如果提供則只檢查該 block）')
    parser.add_argument('--npz_dir', type=str, help='包含 NPZ 文件的目錄路徑')
    parser.add_argument('--test', action='store_true', help='運行測試示例')
    parser.add_argument('--output', '-o', type=str, help='輸出文件路徑（將結果保存到文件）')
    
    args = parser.parse_args()
    
    # 打開輸出文件（如果指定）
    output_file = None
    if args.output:
        output_file = open(args.output, 'w', encoding='utf-8')
        print(f"結果將保存到: {args.output}")
    
    try:
        if args.test:
            # 測試模式：生成示例數據
            msg = "運行測試模式...\n"
            if output_file:
                output_file.write(msg)
            print(msg, end='')
            a = torch.randn(100, 100)
            b = a + torch.randn(100, 100) * 0.1  # 高度相關
            report_str = redundancy_report(a, b, name_a="Test_A", name_b="Test_B", return_str=True)
            if output_file:
                output_file.write(report_str + "\n")
            else:
                print(report_str)
        
        elif args.npz_dir:
            # 檢查目錄中所有 NPZ 文件
            check_all_blocks(args.npz_dir, output_file=output_file)
        
        elif args.npz:
            # 檢查單個 NPZ 文件
            if args.block:
                check_single_block(args.npz, args.block, output_file=output_file)
            else:
                # 如果沒有指定 block，嘗試從文件名推斷
                npz_path = Path(args.npz)
                block_name = npz_path.stem
                check_single_block(args.npz, block_name, output_file=output_file)
        
        else:
            parser.print_help()
    
    finally:
        # 關閉輸出文件
        if output_file:
            output_file.close()
            print(f"\n結果已保存到: {args.output}")

if __name__ == "__main__":
    main()
