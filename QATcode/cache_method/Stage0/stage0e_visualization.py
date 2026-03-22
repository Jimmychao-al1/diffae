"""
Stage-0E Visualization

讀取 Stage-0E 的 .npy 輸出，繪製：
1. 代表性 block 的三條曲線（L1 / CosDist / SVD）
2. 所有 block 的 FID weight bar chart
3. 全局 heatmap（B × T-1）

用法：
    python3 QATcode/cache_method/Stage0/stage0e_visualization.py
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ============================================================
# 一、讀取 Stage-0E 輸出
# ============================================================

def load_stage0e_outputs(output_dir: str):
    """
    從 output_dir 讀取 Stage-0E 的結果。

    Returns:
        block_names: np.ndarray, shape (B,)
        l1_norm:     np.ndarray, shape (B, T-1)
        cos_norm:    np.ndarray, shape (B, T-1)
        svd_norm:    np.ndarray, shape (B, T-1)
        fid_w:       np.ndarray, shape (B,)
    """
    p = Path(output_dir)
    block_names = np.load(p / "block_names.npy", allow_pickle=True)
    l1_norm = np.load(p / "l1_interval_norm.npy")
    cos_norm = np.load(p / "cosdist_interval_norm.npy")
    svd_norm = np.load(p / "svd_interval_norm.npy")
    fid_w = np.load(p / "fid_w_qdiffae_clip.npy")
    return block_names, l1_norm, cos_norm, svd_norm, fid_w


# ============================================================
# 二、Per-block 三曲線圖
# ============================================================

def plot_block_curves(
    block_idx: int,
    block_name: str,
    l1: np.ndarray,
    cos: np.ndarray,
    svd: np.ndarray,
    fid_w: float,
    save_path: str,
):
    """
    對單一 block 繪製 L1 / CosDist / SVD 三條 interval-wise 曲線。

    Args:
        block_idx: block 在陣列中的 index
        block_name: block 名稱（顯示在標題）
        l1, cos, svd: shape (T-1,)
        fid_w: 該 block 的 FID weight（顯示在標題）
        save_path: 輸出 PNG 路徑
    """
    T_minus_1 = len(l1)
    x = np.arange(T_minus_1)  # interval index 0..T-2

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(x, l1, label="L1rel_rate (norm)", color="#1f77b4", linewidth=1.2, alpha=0.9)
    ax.plot(x, cos, label="CosDist (norm)", color="#ff7f0e", linewidth=1.2, alpha=0.9)
    ax.plot(x, svd, label="SVD drift (norm)", color="#2ca02c", linewidth=1.2, alpha=0.9)

    ax.set_xlabel("Interval index j (analysis axis; DDIM: t_ddim (99-j)→(98-j))", fontsize=10)
    ax.set_ylabel("Normalized value  [0, 1]", fontsize=10)
    ax.set_title(
        f"{block_name}    (FID weight = {fid_w:.4f})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0, T_minus_1 - 1)
    ax.set_ylim(-0.02, min(1.05, max(l1.max(), cos.max(), svd.max()) * 1.15 + 0.02))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_selected_blocks(
    block_names: np.ndarray,
    l1_norm: np.ndarray,
    cos_norm: np.ndarray,
    svd_norm: np.ndarray,
    fid_w: np.ndarray,
    indices: List[int],
    save_dir: str,
):
    """
    對一組選定的 block 批次繪圖。

    Args:
        indices: 要繪圖的 block index 列表
        save_dir: 輸出目錄
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        name = str(block_names[idx])
        slug = name.replace(".", "_")
        out = save_path / f"{slug}_curves.png"
        plot_block_curves(
            block_idx=idx,
            block_name=name,
            l1=l1_norm[idx],
            cos=cos_norm[idx],
            svd=svd_norm[idx],
            fid_w=fid_w[idx],
            save_path=str(out),
        )
        print(f"  ✅ {name} -> {out}")


# ============================================================
# 三、FID weight bar chart
# ============================================================

def plot_fid_weight_bar(
    block_names: np.ndarray,
    fid_w: np.ndarray,
    save_path: str,
):
    """
    繪製所有 block 的 FID weight 條形圖（按 weight 遞減排序）。
    """
    B = len(block_names)
    order = np.argsort(fid_w)[::-1]

    names_sorted = [str(block_names[i]).replace("model.", "") for i in order]
    w_sorted = fid_w[order]

    # 顏色：非零 → 藍色漸變，零 → 灰色
    colors = []
    for w in w_sorted:
        if w > 0:
            # 藍色深淺與 w 成正比
            intensity = 0.3 + 0.7 * w
            colors.append((0.12, 0.47 * intensity, 0.71 * intensity + 0.29 * (1 - intensity)))
        else:
            colors.append((0.75, 0.75, 0.75))

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(range(B), w_sorted, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(B))
    ax.set_yticklabels(names_sorted, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("FID weight  w_b  [0, 1]", fontsize=10)
    ax.set_title("Per-block FID Sensitivity Weight (T=100, Q-DiffAE)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.08)
    ax.grid(axis="x", alpha=0.3)

    # 在每個 bar 右邊標數字
    for i, (w, bar) in enumerate(zip(w_sorted, bars)):
        if w > 0:
            ax.text(w + 0.01, i, f"{w:.3f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ FID weight bar -> {save_path}")


# ============================================================
# 四、全局 Heatmap（三種指標各一張）
# ============================================================

def plot_heatmap(
    data: np.ndarray,
    block_names: np.ndarray,
    title: str,
    save_path: str,
    cmap: str = "YlOrRd",
):
    """
    繪製 (B, T-1) 的 heatmap。
    X 軸 = interval index, Y 軸 = block（按原始順序）。
    """
    B, T = data.shape

    # 用簡短名稱
    short_names = [str(n).replace("model.", "") for n in block_names]

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    ax.set_xlabel("Interval index j (analysis axis; DDIM: t_ddim (99-j)→(98-j))", fontsize=10)
    ax.set_ylabel("Block", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_yticks(range(B))
    ax.set_yticklabels(short_names, fontsize=6)

    # X 軸每 10 個 tick
    xticks = list(range(0, T, 10))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Normalized value [0, 1]", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Heatmap -> {save_path}")


# ============================================================
# 五、Combined overview（3 指標 + FID weight，單張大圖）
# ============================================================

def plot_combined_overview(
    block_idx: int,
    block_name: str,
    l1: np.ndarray,
    cos: np.ndarray,
    svd: np.ndarray,
    fid_w_all: np.ndarray,
    block_names: np.ndarray,
    save_path: str,
):
    """
    上半：三條曲線（selected block）
    下半：FID weight bar（highlight selected block）
    """
    T_minus_1 = len(l1)
    x = np.arange(T_minus_1)
    B = len(block_names)
    order = np.argsort(fid_w_all)[::-1]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.3)

    # --- 上半：三曲線 ---
    ax_top = fig.add_subplot(gs[0])
    ax_top.plot(x, l1, label="L1rel_rate", color="#1f77b4", linewidth=1.3)
    ax_top.plot(x, cos, label="CosDist", color="#ff7f0e", linewidth=1.3)
    ax_top.plot(x, svd, label="SVD drift", color="#2ca02c", linewidth=1.3)
    ax_top.set_xlabel("Interval index j (analysis axis)", fontsize=10)
    ax_top.set_ylabel("Normalized [0, 1]", fontsize=10)
    ax_top.set_title(
        f"Stage-0E: {block_name}  (w_b = {fid_w_all[block_idx]:.4f})",
        fontsize=13, fontweight="bold",
    )
    ax_top.set_xlim(0, T_minus_1 - 1)
    y_max = max(l1.max(), cos.max(), svd.max())
    ax_top.set_ylim(-0.02, min(1.05, y_max * 1.15 + 0.02))
    ax_top.legend(fontsize=9)
    ax_top.grid(True, alpha=0.3)

    # --- 下半：FID weight bar ---
    ax_bot = fig.add_subplot(gs[1])
    names_sorted = [str(block_names[i]).replace("model.", "") for i in order]
    w_sorted = fid_w_all[order]

    colors = []
    for i_sorted, orig_idx in enumerate(order):
        if orig_idx == block_idx:
            colors.append("#d62728")  # 紅色 highlight
        elif w_sorted[i_sorted] > 0:
            colors.append("#1f77b4")
        else:
            colors.append("#cccccc")

    ax_bot.barh(range(B), w_sorted, color=colors, edgecolor="white", linewidth=0.3)
    ax_bot.set_yticks(range(B))
    ax_bot.set_yticklabels(names_sorted, fontsize=6)
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel("FID weight w_b", fontsize=10)
    ax_bot.set_title("FID Sensitivity (red = selected block)", fontsize=11)
    ax_bot.set_xlim(0, 1.08)
    ax_bot.grid(axis="x", alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Combined overview -> {save_path}")


# ============================================================
# 六、自動選出代表性 block
# ============================================================

def select_representative_blocks(
    block_names: np.ndarray,
    fid_w: np.ndarray,
    n_top: int = 3,
    n_bottom: int = 2,
    extra: Optional[List[str]] = None,
) -> List[int]:
    """
    自動選出代表性 block：
    - FID weight 最高的 n_top 個
    - FID weight 最低（或 = 0）的 n_bottom 個
    - 額外指定的 block（若存在）

    Returns:
        indices: 去重後的 block index 列表
    """
    order_desc = np.argsort(fid_w)[::-1]
    order_asc = np.argsort(fid_w)

    selected = set()

    # Top FID weight
    for i in range(min(n_top, len(order_desc))):
        selected.add(int(order_desc[i]))

    # Bottom FID weight
    for i in range(min(n_bottom, len(order_asc))):
        selected.add(int(order_asc[i]))

    # Extra
    if extra:
        name_to_idx = {str(n): i for i, n in enumerate(block_names)}
        for name in extra:
            if name in name_to_idx:
                selected.add(name_to_idx[name])

    return sorted(selected)


# ============================================================
# 七、主入口
# ============================================================

def main(
    input_dir: str,
    output_dir: str,
    n_top: int = 3,
    n_bottom: int = 2,
    extra_blocks: Optional[List[str]] = None,
):
    """
    Stage-0E 可視化主流程。

    Args:
        input_dir: Stage-0E 的 .npy 輸出目錄
        output_dir: 圖片輸出目錄
        n_top: 繪製 FID weight 最高的幾個 block
        n_bottom: 繪製 FID weight 最低的幾個 block
        extra_blocks: 額外指定要繪製的 block 名稱
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage-0E Visualization")
    print("=" * 60)

    # 1. 讀取
    print("\n[1] 載入 Stage-0E 輸出...")
    block_names, l1_norm, cos_norm, svd_norm, fid_w = load_stage0e_outputs(input_dir)
    B, T_minus_1 = l1_norm.shape
    print(f"    B={B}, T-1={T_minus_1}")

    # 2. 選出代表性 block
    print("\n[2] 選出代表性 block...")
    indices = select_representative_blocks(
        block_names, fid_w, n_top=n_top, n_bottom=n_bottom, extra=extra_blocks,
    )
    print(f"    選出 {len(indices)} 個 block:")
    for idx in indices:
        tag = "HIGH" if fid_w[idx] > 0.1 else ("LOW" if fid_w[idx] == 0 else "MID")
        print(f"      [{idx:2d}] {block_names[idx]:35s}  w={fid_w[idx]:.4f}  ({tag})")

    # 3. Per-block 三曲線圖
    print("\n[3] 繪製 per-block 三曲線圖...")
    curves_dir = out / "block_curves"
    plot_selected_blocks(
        block_names, l1_norm, cos_norm, svd_norm, fid_w,
        indices=indices, save_dir=str(curves_dir),
    )

    # 4. Combined overview（每個 selected block 一張大圖）
    print("\n[4] 繪製 combined overview...")
    overview_dir = out / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        slug = str(block_names[idx]).replace(".", "_")
        plot_combined_overview(
            block_idx=idx,
            block_name=str(block_names[idx]),
            l1=l1_norm[idx],
            cos=cos_norm[idx],
            svd=svd_norm[idx],
            fid_w_all=fid_w,
            block_names=block_names,
            save_path=str(overview_dir / f"{slug}_overview.png"),
        )

    # 5. FID weight bar chart
    print("\n[5] 繪製 FID weight bar chart...")
    plot_fid_weight_bar(block_names, fid_w, str(out / "fid_weight_bar.png"))

    # 6. 全局 Heatmap
    print("\n[6] 繪製全局 heatmap...")
    heatmap_dir = out / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(l1_norm, block_names, "L1rel_rate (normalized)", str(heatmap_dir / "heatmap_l1.png"), cmap="YlOrRd")
    plot_heatmap(cos_norm, block_names, "Cosine Distance (normalized)", str(heatmap_dir / "heatmap_cosdist.png"), cmap="YlOrRd")
    plot_heatmap(svd_norm, block_names, "SVD Subspace Drift (normalized)", str(heatmap_dir / "heatmap_svd.png"), cmap="YlOrRd")

    print("\n" + "=" * 60)
    print(f"✅ 所有圖片已儲存至: {out}")
    print("=" * 60)


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

    main(
        input_dir=os.path.join(project_root, "QATcode/cache_method/Stage0/stage0e_output"),
        output_dir=os.path.join(project_root, "QATcode/cache_method/Stage0/stage0e_figures"),
        n_top=3,
        n_bottom=2,
        extra_blocks=["model.middle_block", "model.input_blocks.7"],
    )
