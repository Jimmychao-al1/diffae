"""
SVD vs Similarity 相關性分析

功能：
- 讀取 svd_metrics/<block_slug>.json（SVD 子空間距離）
- 讀取 L1_L2_cosine/T_100/*/result_npz/<block_slug>.npz（similarity step 曲線）
- 計算 Pearson / Spearman 相關性（L1 vs SVDdist、L1rel_rate vs SVDdist、CosDist vs SVDdist）
- 可選：畫對齊曲線圖與散點圖
- 輸出：correlation/<block_slug>.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ==================== 工具函數 ====================

def load_svd_metrics(svd_json_path: Path) -> Dict:
    """載入 SVD 指標 JSON"""
    if not svd_json_path.exists():
        raise FileNotFoundError(f"SVD JSON 不存在: {svd_json_path}")
    
    with open(svd_json_path, 'r') as f:
        data = json.load(f)
    
    return data


def load_similarity_npz(npz_path: Path) -> Dict:
    """
    載入 similarity NPZ
    
    Returns:
        dict: 包含 l1_step_mean, l1_rate_step_mean, cos_step_mean, step_idx
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Similarity NPZ 不存在: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)

    if 'l1_step_mean' not in keys:
        raise KeyError(f"{npz_path} 缺少必要欄位: l1_step_mean")
    if 'cos_step_mean' not in keys:
        raise KeyError(f"{npz_path} 缺少必要欄位: cos_step_mean")

    # 目前 similarity_calculation.py 已輸出 l1_rate_step_mean；舊檔可能不存在，做相容降級。
    if 'l1_rate_step_mean' in keys:
        l1_rate_step_mean = data['l1_rate_step_mean']
    else:
        l1_rate_step_mean = data['l1_step_mean']

    step_idx = data['step_idx'] if 'step_idx' in keys else None

    return {
        'l1_step_mean': data['l1_step_mean'],
        'l1_rate_step_mean': l1_rate_step_mean,
        'cos_step_mean': data['cos_step_mean'],
        'step_idx': step_idx,
    }


def compute_correlations(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    計算 Pearson 與 Spearman 相關性
    
    Args:
        x: 序列 1
        y: 序列 2
    
    Returns:
        dict: pearson, spearman, pearson_pvalue, spearman_pvalue
    """
    # 過濾 NaN / Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return {
            'pearson': float('nan'),
            'spearman': float('nan'),
            'pearson_pvalue': float('nan'),
            'spearman_pvalue': float('nan')
        }
    
    # Pearson
    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
    
    # Spearman
    spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
    
    return {
        'pearson': float(pearson_r),
        'spearman': float(spearman_r),
        'pearson_pvalue': float(pearson_p),
        'spearman_pvalue': float(spearman_p)
    }


def plot_alignment(
    svd_dist: np.ndarray,
    l1: np.ndarray,
    cos_dist: np.ndarray,
    block_slug: str,
    output_path: Path
):
    """
    畫對齊曲線圖
    
    Args:
        svd_dist: SVD interval distance 序列，長度 L（通常 T-1，但可能因對齊截斷而更短）
        l1: L1 或 L1rel_rate 的 step 序列，長度 L (L1)
        cos_dist: 1 - cos_step_mean 的 step 序列，長度 L
        block_slug: Block 名稱
        output_path: 輸出路徑
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    T = len(svd_dist)
    x = np.arange(T)
    # point-wise:
    # analysis index left->right is 0..T-1 (noise->clear), display label should be t=T-1..0
    xticks = list(range(0, T, 10))
    if (T - 1) not in xticks:
        xticks.append(T - 1)
    xticklabels = [str((T - 1) - i) for i in xticks]
    
    # 上圖：L1 vs SVD
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(x, l1, 'b-', linewidth=2, label='L1 relative change', alpha=0.8)
    ax1_twin.plot(x, svd_dist, 'r--', linewidth=2, label='SVD Subspace Dist', alpha=0.8)
    
    ax1.set_xlabel('DDIM timestep t') # noise (T-1) -> clear (0) 放在論文圖中說明
    ax1.set_ylabel('L1 relative change', color='b')
    ax1_twin.set_ylabel('SVD Subspace Distance', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_title(f'{block_slug} - L1 relative change vs SVD Subspace Distance', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)
    
    # 下圖：CosDist vs SVD
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    ax2.plot(x, cos_dist, 'g-', linewidth=2, label='Cosine distance (1 − cosine similarity)', alpha=0.8)
    ax2_twin.plot(x, svd_dist, 'r--', linewidth=2, label='SVD Subspace Dist', alpha=0.8)
    
    ax2.set_xlabel('DDIM timestep t') # noise (T-1) -> clear (0) 放在論文圖中說明
    ax2.set_ylabel('Cosine distance (1 − cosine similarity)', color='g')
    ax2_twin.set_ylabel('SVD Subspace Distance', color='r')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.set_title(f'{block_slug} - Cosine distance (1 − cosine similarity) vs SVD Subspace Distance', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   - 對齊曲線圖: {output_path}")


def plot_scatter(
    svd_dist: np.ndarray,
    l1: np.ndarray,
    cos_dist: np.ndarray,
    block_slug: str,
    output_path: Path,
    l1_corr: Dict,
    cos_corr: Dict
):
    """
    畫散點圖
    
    Args:
        svd_dist: SVD 子空間距離，長度 T
        l1: L1rel step mean，長度 T
        cos_dist: 1 - Cosine，長度 T
        block_slug: Block 名稱
        output_path: 輸出路徑
        l1_corr: L1 vs SVD 的相關性統計
        cos_corr: CosDist vs SVD 的相關性統計
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左圖：L1 vs SVD
    ax1 = axes[0]
    ax1.scatter(l1, svd_dist, alpha=0.6, s=30, c='blue')
    ax1.set_xlabel('L1 relative change')
    ax1.set_ylabel('SVD Subspace Distance')
    ax1.set_title(f'{block_slug} - L1 relative change vs SVD\nPearson: {l1_corr["pearson"]:.4f}, Spearman: {l1_corr["spearman"]:.4f}',
                  fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 右圖：CosDist vs SVD
    ax2 = axes[1]
    ax2.scatter(cos_dist, svd_dist, alpha=0.6, s=30, c='green')
    ax2.set_xlabel('Cosine distance (1 − cosine similarity)')
    ax2.set_ylabel('SVD Subspace Distance')
    ax2.set_title(f'{block_slug} - Cosine Distance vs SVD\nPearson: {cos_corr["pearson"]:.4f}, Spearman: {cos_corr["spearman"]:.4f}',
                  fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   - 散點圖: {output_path}")


# ==================== 主流程 ====================

def process_single_correlation(
    svd_json_path: Path,
    similarity_npz_path: Path,
    output_dir: Path,
    plot_figures: bool = True
):
    """
    處理單一 block 的相關性分析
    
    Args:
        svd_json_path: SVD 指標 JSON 路徑
        similarity_npz_path: Similarity NPZ 路徑
        output_dir: 輸出目錄
        plot_figures: 是否畫圖
    """
    # 1. 載入 SVD 指標
    svd_data = load_svd_metrics(svd_json_path)
    block_slug = svd_data['block']
    T = svd_data['T']
    svd_dist = np.array(svd_data['subspace_dist'])  # 長度 T
    
    print(f"\n{'='*60}")
    print(f"處理 Block: {block_slug}")
    print(f"{'='*60}")
    print(f"T={T}, rank_r={svd_data['rank_r']}")
    
    # 2. 載入 similarity（interval-wise: 長度通常為 T-1）
    sim_data = load_similarity_npz(similarity_npz_path)
    l1_step_mean = sim_data['l1_step_mean']
    l1_rate_step_mean = sim_data['l1_rate_step_mean']
    cos_step_mean = sim_data['cos_step_mean']
    
    # 3. 檢查長度
    interval_len = len(l1_step_mean)
    if len(l1_rate_step_mean) != interval_len or len(cos_step_mean) != interval_len:
        print(
            "警告：similarity 各序列長度不一致，將取最小長度對齊 "
            f"(l1={len(l1_step_mean)}, l1_rate={len(l1_rate_step_mean)}, cos={len(cos_step_mean)})"
        )
    min_interval_len = min(len(l1_step_mean), len(l1_rate_step_mean), len(cos_step_mean))
    if min_interval_len <= 0:
        raise ValueError("Similarity step 序列長度為 0，無法計算相關性")

    # subspace_dist[t]（t>=1）對應 interval index j=t-1，所以取 svd_dist[1:1+L]
    if len(svd_dist) - 1 < min_interval_len:
        print(
            f"警告：SVD interval 長度不足，SVD 可用={len(svd_dist)-1}, similarity={min_interval_len}，"
            "將以較短長度對齊"
        )
    L = min(min_interval_len, max(len(svd_dist) - 1, 0))
    if L <= 1:
        raise ValueError(f"對齊後有效序列太短 (L={L})，無法計算穩定相關性")

    svd_dist_seq = svd_dist[1:1 + L]
    l1_seq = l1_step_mean[:L]
    l1_rate_seq = l1_rate_step_mean[:L]
    cos_dist_seq = 1.0 - cos_step_mean[:L]  # Cosine distance = 1 - CosSim
    
    # 5. 計算相關性
    print("\n計算相關性...")
    l1_vs_svd = compute_correlations(l1_seq, svd_dist_seq)
    l1_rate_vs_svd = compute_correlations(l1_rate_seq, svd_dist_seq)
    cos_vs_svd = compute_correlations(cos_dist_seq, svd_dist_seq)
    
    print(f"L1 vs SVD:")
    print(f"  Pearson: {l1_vs_svd['pearson']:.4f} (p={l1_vs_svd['pearson_pvalue']:.4e})")
    print(f"  Spearman: {l1_vs_svd['spearman']:.4f} (p={l1_vs_svd['spearman_pvalue']:.4e})")
    
    print(f"L1rel_rate vs SVD:")
    print(f"  Pearson: {l1_rate_vs_svd['pearson']:.4f} (p={l1_rate_vs_svd['pearson_pvalue']:.4e})")
    print(f"  Spearman: {l1_rate_vs_svd['spearman']:.4f} (p={l1_rate_vs_svd['spearman_pvalue']:.4e})")
    
    print(f"Cosine Distance vs SVD:")
    print(f"  Pearson: {cos_vs_svd['pearson']:.4f} (p={cos_vs_svd['pearson_pvalue']:.4e})")
    print(f"  Spearman: {cos_vs_svd['spearman']:.4f} (p={cos_vs_svd['spearman_pvalue']:.4e})")
    
    # 6. 輸出 JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"{block_slug}.json"
    
    result = {
        "block": block_slug,
        "T_svd": int(T),
        "interval_length_used": int(L),
        "rank_r": svd_data['rank_r'],
        "correlation": {
            "L1_vs_SVD": l1_vs_svd,
            "L1relRate_vs_SVD": l1_rate_vs_svd,
            "CosDist_vs_SVD": cos_vs_svd
        }
    }
    
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ 相關性 JSON: {output_json}")
    
    # 7. 可選：畫圖
    if plot_figures:
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # 對齊曲線圖
        alignment_path = figures_dir / f"{block_slug}_alignment.png"
        plot_alignment(svd_dist_seq, l1_rate_seq, cos_dist_seq, block_slug, alignment_path)
        
        # 散點圖
        scatter_path = figures_dir / f"{block_slug}_scatter.png"
        plot_scatter(svd_dist_seq, l1_seq, cos_dist_seq, block_slug, scatter_path, l1_vs_svd, cos_vs_svd)
    
    return result


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="SVD vs Similarity Correlation Analysis")
    parser.add_argument('--svd_metrics', type=str, help='SVD 指標 JSON 路徑（單一 block）')
    parser.add_argument('--similarity_npz', type=str, help='Similarity NPZ 路徑（單一 block）')
    parser.add_argument('--svd_metrics_dir', type=str, default='QATcode/cache_method/SVD/svd_metrics',
                        help='SVD 指標目錄（批次處理）')
    parser.add_argument('--similarity_npz_dir', type=str, default='QATcode/cache_method/L1_L2_cosine/T_100/v2_latest/result_npz',
                        help='Similarity NPZ 目錄（批次處理）')
    parser.add_argument('--output_root', type=str, default='QATcode/cache_method/SVD/correlation',
                        help='輸出根目錄')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='生成對齊曲線與散點圖')
    parser.add_argument('--all', action='store_true',
                        help='批次處理所有 block')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_root)
    
    if args.all:
        # 批次處理所有 block
        svd_dir = Path(args.svd_metrics_dir)
        sim_dir = Path(args.similarity_npz_dir)
        
        if not svd_dir.exists():
            print(f"錯誤：SVD 目錄不存在: {svd_dir}")
            return
        
        if not sim_dir.exists():
            print(f"錯誤：Similarity 目錄不存在: {sim_dir}")
            return
        
        svd_jsons = sorted(svd_dir.glob("*.json"))
        print(f"找到 {len(svd_jsons)} 個 SVD JSON")
        
        success_count = 0
        for svd_json in svd_jsons:
            block_slug = svd_json.stem
            sim_npz = sim_dir / f"{block_slug}.npz"
            
            if not sim_npz.exists():
                print(f"跳過 {block_slug}：找不到對應的 similarity NPZ")
                continue
            
            try:
                process_single_correlation(svd_json, sim_npz, output_dir, plot_figures=args.plot)
                success_count += 1
            except Exception as e:
                print(f"處理 {block_slug} 時出錯: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*60}")
        print(f"批次處理完成：成功 {success_count} / {len(svd_jsons)} 個 block")
        print(f"{'='*60}")
    
    elif args.svd_metrics and args.similarity_npz:
        # 處理單一 block
        svd_json = Path(args.svd_metrics)
        sim_npz = Path(args.similarity_npz)
        
        process_single_correlation(svd_json, sim_npz, output_dir, plot_figures=args.plot)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
