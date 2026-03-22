"""
Stage-1 結果可視化

生成：
1. D_global / D_smooth 曲線 + change points
2. Zone segmentation 示意圖
3. K 分佈熱圖 (B x Z)
4. K 直方圖
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _zone_axis(z: dict):
    a0 = z.get("axis_start", z.get("t_start"))
    a1 = z.get("axis_end", z.get("t_end"))
    if a0 is None or a1 is None:
        raise KeyError(f"zone 缺少 axis_start/axis_end 或舊版 t_start/t_end: {z}")
    return int(a0), int(a1)


def load_stage1_outputs(output_dir: str):
    """載入 Stage-1 輸出"""
    p = Path(output_dir)
    
    with open(p / "scheduler_config.json") as f:
        config = json.load(f)
    
    with open(p / "scheduler_diagnostics.json") as f:
        diag = json.load(f)
    
    return config, diag


def plot_drift_and_zones(config, diag, save_path):
    """Plot 1: D_global/D_smooth 曲線 + zones"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    D_global = np.array(diag['D_global'])
    D_smooth = np.array(diag['D_smooth'])
    Delta = np.array(diag['Delta'])
    T = len(D_global) + 1  # 100
    
    # === Top: D_global + D_smooth ===
    ax1.plot(D_global, label='D_global (raw)', alpha=0.6, lw=1)
    ax1.plot(D_smooth, label='D_smooth', lw=2)
    
    # Change points（vertical lines）
    for cp in diag['change_points']:
        ax1.axvline(cp, color='red', alpha=0.5, ls='--', lw=1)
    
    # Zone backgrounds
    zones = config['zones']
    colors = plt.cm.tab10(np.linspace(0, 1, len(zones)))
    for z, color in zip(zones, colors):
        a0, a1 = _zone_axis(z)
        ax1.axvspan(a0, a1, alpha=0.15, color=color, label=f"Zone {z['id']}")
    
    ax1.set_xlabel('Interval index j (length 99; analysis axis; DDIM: t_ddim=99-axis_idx at point indices)')
    ax1.set_ylabel('Global Drift')
    ax1.set_title('FID-weighted Global Drift + Zone Segmentation (per-interval D)')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    
    # === Bottom: Delta ===
    ax2.plot(Delta, label='Δ[t] = |D_smooth[t] - D_smooth[t-1]|', color='purple')
    
    for cp in diag['change_points']:
        ax2.axvline(cp, color='red', alpha=0.5, ls='--', lw=1)
    
    ax2.set_xlabel('Same interval index j as D_smooth (analysis axis)')
    ax2.set_ylabel('Change Magnitude Δ')
    ax2.set_title('Delta (Change Magnitude)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已儲存: {save_path}")


def plot_k_heatmap(config, save_path):
    """Plot 2: K 分佈熱圖 (B x Z)"""
    blocks = config['blocks']
    B = len(blocks)
    Z = len(config['zones'])
    
    k_matrix = np.zeros((B, Z), dtype=int)
    block_names = []
    
    for b_idx, block in enumerate(blocks):
        k_matrix[b_idx, :] = block['k_per_zone']
        # 簡化 block name（太長會顯示不下）
        name = block['name'].replace('model.', '')
        block_names.append(name)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(k_matrix, cmap='YlOrRd', aspect='auto', vmin=1, vmax=8)
    
    # X-axis: zones
    ax.set_xticks(range(Z))
    ax.set_xticklabels(
        [f"Z{z['id']}\n{_zone_axis(z)[0]}..{_zone_axis(z)[1]}" for z in config['zones']],
        fontsize=8,
    )
    ax.set_xlabel('Zones')
    
    # Y-axis: blocks
    ax.set_yticks(range(B))
    ax.set_yticklabels(block_names, fontsize=7)
    ax.set_ylabel('Blocks')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('k (cache frequency)', rotation=270, labelpad=20)
    
    # 標註數值
    for b in range(B):
        for z in range(Z):
            text = ax.text(z, b, k_matrix[b, z],
                          ha="center", va="center", color="black", fontsize=7)
    
    ax.set_title('K Distribution Heatmap (per Block × Zone)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已儲存: {save_path}")


def plot_k_histogram(config, save_path):
    """Plot 3: K 直方圖（整體分佈）"""
    all_k = []
    for block in config['blocks']:
        all_k.extend(block['k_per_zone'])
    all_k = np.array(all_k)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bins = np.arange(0.5, 9.5, 1)
    ax.hist(all_k, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    
    ax.set_xlabel('k (cache frequency)')
    ax.set_ylabel('Count')
    ax.set_title(f'K Distribution Histogram (Total: {len(all_k)} entries)')
    ax.set_xticks(range(1, 9))
    ax.grid(alpha=0.3, axis='y')
    
    # 統計
    stats_text = f"Min={all_k.min()}, Max={all_k.max()}, Mean={all_k.mean():.2f}, Median={np.median(all_k):.1f}"
    ax.text(0.5, 0.95, stats_text, transform=ax.transAxes,
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已儲存: {save_path}")


def plot_zone_risk(config, diag, save_path):
    """Plot 4: Zone risk + k_max ceiling"""
    zones = config['zones']
    R_z = np.array(diag['R_z'])
    k_max_z = np.array(diag['k_max_z'])
    Z = len(zones)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    zone_ids = [z['id'] for z in zones]
    zone_labels = [f"Z{z['id']}\n{_zone_axis(z)[0]}..{_zone_axis(z)[1]}" for z in zones]
    
    # === Top: Zone risk R_z ===
    bars = ax1.bar(zone_ids, R_z, color='coral', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Zone')
    ax1.set_ylabel('Risk R_z')
    ax1.set_title('Zone Risk R_z (higher = more risky)')
    ax1.set_xticks(zone_ids)
    ax1.set_xticklabels(zone_labels)
    ax1.grid(alpha=0.3, axis='y')
    
    # 標註數值
    for bar, val in zip(bars, R_z):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # === Bottom: k_max ceiling ===
    bars = ax2.bar(zone_ids, k_max_z, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Zone')
    ax2.set_ylabel('k_max ceiling')
    ax2.set_title('Zone-level K Ceiling')
    ax2.set_xticks(zone_ids)
    ax2.set_xticklabels(zone_labels)
    ax2.set_ylim([0, 9])
    ax2.grid(alpha=0.3, axis='y')
    
    # 標註數值
    for bar, val in zip(bars, k_max_z):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已儲存: {save_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Stage-1 results")
    parser.add_argument(
        "--stage1_output_dir",
        type=str,
        default="QATcode/cache_method/Stage1/stage1_output",
        help="Stage-1 output directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="QATcode/cache_method/Stage1/stage1_figures",
        help="Visualization output directory"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Stage-1 結果可視化")
    print("=" * 80)
    
    # 載入資料
    print(f"\n讀取 Stage-1 輸出: {args.stage1_output_dir}")
    config, diag = load_stage1_outputs(args.stage1_output_dir)
    
    # 創建輸出目錄
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成圖表
    print("\n生成可視化...")
    
    plot_drift_and_zones(config, diag, out_dir / "1_drift_and_zones.png")
    plot_k_heatmap(config, out_dir / "2_k_heatmap.png")
    plot_k_histogram(config, out_dir / "3_k_histogram.png")
    plot_zone_risk(config, diag, out_dir / "4_zone_risk.png")
    
    print("\n" + "=" * 80)
    print(f"✅ 全部完成！圖表已存至: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
