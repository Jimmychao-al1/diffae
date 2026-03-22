"""
Stage-1: Offline Scheduler Synthesis (T=100)

從 Stage-0 的 tri-evidence（L1/Cosine/SVD similarity + FID sensitivity）
合成靜態 cache scheduler：zones + k[b,z]

時間軸（與 Stage 0、L1_L2_cosine .npz 一致）：
- **analysis axis** 索引 axis_idx ∈ [0, 99]：與 similarity 收集時的 step_counter / 圖橫軸由左到右 0→99 一致。
- **DDIM** 進模型的 timestep：**t_ddim = 99 - axis_idx**（採樣順序為 t_ddim 從 99→0）。
- 長度 99 的 interval 陣列：第 j 欄 = analysis 上 interval j（axis j 與 j+1 之間）= t_ddim (99−j)→(98−j)。

本模組內 **D_global / zones / recompute mask 的索引**皆為 **analysis axis**（或 interval 維），**不是**未加轉換的 DDIM t_ddim。

演算法流程：
1. 載入 Stage-0E 輸出
2. 計算 FID-weighted global drift D_global（per-interval）
3. 平滑 + 找 change points → 分 zones（axis 上 0..99 的點區間）
4. 計算 zone-level tri-evidence score A[b,z]
5. 映射到 k_raw[b,z]
6. Zone-level risk ceiling
7. Regularization（避免 k 跳太大）
8. 輸出 scheduler_config.json + diagnostics
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np


# ============================================================
# Logger 設定
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [Stage1] %(message)s'
)
LOGGER = logging.getLogger("Stage1Scheduler")


# ============================================================
# Data structures
# ============================================================

@dataclass
class Zone:
    """Zone 定義（shared across all blocks）。

    axis_start / axis_end：analysis axis 上的點索引 [0..99]，含端點；
    與 similarity_calculation 的 step_counter、Stage 0 圖橫軸一致。
    DDIM：t_ddim = 99 - axis_idx。
    """
    id: int
    axis_start: int  # inclusive, analysis axis index
    axis_end: int    # inclusive, analysis axis index
    
    def __post_init__(self):
        assert self.axis_start <= self.axis_end
        assert self.axis_start >= 0
    
    def length(self) -> int:
        return self.axis_end - self.axis_start + 1
    
    def axis_indices(self) -> List[int]:
        return list(range(self.axis_start, self.axis_end + 1))


@dataclass
class SchedulerConfig:
    """Scheduler 配置（用於輸出 JSON）"""
    version: str
    T: int
    t_order: str  # 與 analysis_axis_order 同值；歷史欄位名保留
    analysis_axis_order: str
    axis_convention: str
    ddim_timestep_formula: str
    params: Dict[str, Any]
    zones: List[Dict[str, Any]]
    blocks: List[Dict[str, Any]]


# ============================================================
# 一、載入 Stage-0E 輸出
# ============================================================

def load_stage0e_outputs(
    input_dir: str,
    eta: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    從 Stage-0E 輸出目錄讀取正規化後的 interval-wise 指標。
    
    Args:
        input_dir: Stage-0E 的 .npy 輸出目錄
        eta: L1 vs Cosine 穩定度加權，S_sim = eta*S_l1 + (1-eta)*S_cos，範圍 [0,1]；
             eta=1 為僅 L1（與舊版行為一致），eta=0 為僅 Cosine。
    
    Returns:
        block_names: (B,) object array
        S_sim: (B, T-1) similarity 穩定性分數（越大越穩定）
        S_svd: (B, T-1) 穩定性分數（越大越穩定）= 1 - SVD_interval_norm
        d_norm: (B, T-1) drift（越大越不穩定）= SVD_interval_norm
        FID_sens: (B,) FID 敏感度（越大越敏感）
        S_l1: (B, T-1) = 1 - l1_interval_norm
        S_cos: (B, T-1) = 1 - cosdist_interval_norm
    
    語義轉換：
    - Stage-0E 的 l1 / cosdist / svd 是「變化量」（越大越不穩定）
    - S_l1, S_cos = 1 - norm；S_sim 為兩者線性混合後 clip 至 [0,1]
    - d_norm 保留 drift 語義（直接用 svd_interval_norm）
    """
    p = Path(input_dir)
    
    if not p.exists():
        raise FileNotFoundError(f"Stage-0E 輸出目錄不存在: {p}")
    
    # 讀取檔案
    required_files = [
        "block_names.npy",
        "l1_interval_norm.npy",
        "cosdist_interval_norm.npy",
        "svd_interval_norm.npy",
        "fid_w_qdiffae_clip.npy",
    ]
    
    for f in required_files:
        if not (p / f).exists():
            raise FileNotFoundError(f"缺少必要檔案: {p / f}")
    
    block_names = np.load(p / "block_names.npy", allow_pickle=True)
    l1_norm = np.load(p / "l1_interval_norm.npy")
    cos_norm = np.load(p / "cosdist_interval_norm.npy")
    svd_norm = np.load(p / "svd_interval_norm.npy")
    fid_w = np.load(p / "fid_w_qdiffae_clip.npy")
    
    # Shape 檢查
    B = len(block_names)
    T_minus_1 = l1_norm.shape[1] if len(l1_norm.shape) == 2 else len(l1_norm)
    
    if T_minus_1 != 99:
        raise ValueError(f"期望 T-1=99（T=100），但讀到 {T_minus_1}")
    
    expected_shape = (B, T_minus_1)
    for arr, name in [(l1_norm, "l1"), (cos_norm, "cos"), (svd_norm, "svd")]:
        if arr.shape != expected_shape:
            raise ValueError(f"{name} shape {arr.shape} != 期望 {expected_shape}")
    
    if fid_w.shape != (B,):
        raise ValueError(f"fid_w shape {fid_w.shape} != 期望 ({B},)")
    
    # 數值範圍檢查（若超出 [0,1] 則 clip + 警告）
    def check_and_clip(arr, name):
        if arr.min() < 0 or arr.max() > 1:
            LOGGER.warning(f"{name} 超出 [0,1] 範圍：[{arr.min():.4f}, {arr.max():.4f}]，已 clip")
            return np.clip(arr, 0, 1)
        return arr
    
    l1_norm = check_and_clip(l1_norm, "l1_interval_norm")
    cos_norm = check_and_clip(cos_norm, "cosdist_interval_norm")
    svd_norm = check_and_clip(svd_norm, "svd_interval_norm")
    fid_w = check_and_clip(fid_w, "fid_w")
    
    if not (0.0 <= float(eta) <= 1.0):
        raise ValueError(f"eta 必須在 [0, 1] 內，收到 {eta}")
    eta_f = float(eta)
    
    # 語義轉換：變化量 → 穩定性分數；similarity 通道為 L1 / Cos 加權混合
    S_l1 = 1.0 - l1_norm
    S_cos = 1.0 - cos_norm
    S_sim = eta_f * S_l1 + (1.0 - eta_f) * S_cos
    S_sim = np.clip(S_sim, 0.0, 1.0)
    S_svd = 1.0 - svd_norm
    d_norm = svd_norm  # drift 保留原語義（越大越不穩定）
    FID_sens = fid_w
    
    LOGGER.info(f"✅ 成功載入 Stage-0E 輸出：B={B}, T-1={T_minus_1}, eta={eta_f}")
    LOGGER.info(f"   S_l1: [{S_l1.min():.4f}, {S_l1.max():.4f}], S_cos: [{S_cos.min():.4f}, {S_cos.max():.4f}]")
    LOGGER.info(f"   S_sim (stability): [{S_sim.min():.4f}, {S_sim.max():.4f}]")
    LOGGER.info(f"   S_svd (stability): [{S_svd.min():.4f}, {S_svd.max():.4f}]")
    LOGGER.info(f"   d_norm (drift): [{d_norm.min():.4f}, {d_norm.max():.4f}]")
    LOGGER.info(f"   FID_sens: [{FID_sens.min():.4f}, {FID_sens.max():.4f}], 非零數={np.sum(FID_sens > 0)}/{B}")
    
    return block_names, S_sim, S_svd, d_norm, FID_sens, S_l1, S_cos


# ============================================================
# 二、FID-weighted global drift + 平滑
# ============================================================

def compute_global_drift(
    d_norm: np.ndarray,
    FID_sens: np.ndarray,
    smooth_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算 FID-weighted global drift + moving average 平滑。

    **索引**：`d_norm[b, j]` 與 Stage 0 / .npz 一致，為 **analysis 上 interval j**（axis j 與 j+1 之間）。
    `D_global[j]` 對應同一 interval j（長度 T-1=99）。
    
    Args:
        d_norm: (B, T-1) drift（越大越不穩定）
        FID_sens: (B,) FID 敏感度權重
        smooth_window: moving average 視窗大小
    
    Returns:
        D_global: (T-1,) raw global drift（per-interval on analysis axis）
        D_smooth: (T-1,) smoothed global drift
    
    公式：
        D_global[j] = sum_b (w_b * d_norm[b,j]) / (sum_b w_b + eps)
        D_smooth = moving_avg(D_global, window)
    """
    B, T_minus_1 = d_norm.shape
    eps = 1e-8
    
    # FID weight normalization（確保 sum 不為 0）
    w_sum = FID_sens.sum() + eps
    w_norm = FID_sens / w_sum  # shape (B,)
    
    # Weighted sum: (B,) @ (B, T-1) -> (T-1)
    D_global = (w_norm[:, None] * d_norm).sum(axis=0)  # shape (T-1,)
    
    # Moving average smoothing
    if smooth_window <= 1:
        D_smooth = D_global.copy()
    else:
        D_smooth = moving_average(D_global, window=smooth_window)
    
    LOGGER.info(f"Global drift: min={D_global.min():.4f}, max={D_global.max():.4f}, mean={D_global.mean():.4f}")
    LOGGER.info(f"Smoothed drift: min={D_smooth.min():.4f}, max={D_smooth.max():.4f}, mean={D_smooth.mean():.4f}")
    
    return D_global, D_smooth


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """
    1D moving average，邊界用 same mode（補零或截斷）。
    
    Args:
        x: shape (N,)
        window: 視窗大小（奇數較佳，例如 3, 5, 7）
    
    Returns:
        x_smooth: shape (N,)
    """
    if window <= 1:
        return x.copy()
    
    # 使用 numpy convolve with mode='same'
    kernel = np.ones(window) / window
    x_smooth = np.convolve(x, kernel, mode='same')
    return x_smooth


# ============================================================
# 三、Zone segmentation（找 change points）
# ============================================================

def find_zones(
    D_smooth: np.ndarray,
    method: str = "topk",
    topk: int = 6,
    threshold_quantile: float = 0.90,
) -> List[Zone]:
    """
    根據 D_smooth 的變化找出 zones。
    
    Args:
        D_smooth: (T-1,) smoothed global drift
        method: "topk" 或 "threshold"
        topk: topk 模式下，取 Δ 最大的 K 個點
        threshold_quantile: threshold 模式下，Δ >= quantile(q) 的點
    
    Returns:
        zones: List[Zone]，覆蓋 **analysis axis 點** 0..99（T=100 個點；邊界由 interval 變化推得）
    
    演算法：
    1. 計算 Δ（沿 interval 軸）= |D_smooth[j] - D_smooth[j-1]|
    2. 選 change points（映射為 axis 上的分界點）
    3. boundaries = [0] + sorted(change_points) + [T-1]
    4. zones 的 axis_start..axis_end 為 **analysis axis 索引**，**不是**未轉換的 DDIM t_ddim
    """
    T_minus_1 = len(D_smooth)
    T = T_minus_1 + 1  # 100 個 timestep: 0..99
    
    # 1. 計算 Δ（change magnitude）
    # Δ[t] for t=1..T-2 代表 interval t-1 到 interval t 的變化
    Delta = np.zeros(T_minus_1)
    Delta[0] = 0.0  # 第一個 interval 沒有 previous
    for t in range(1, T_minus_1):
        Delta[t] = abs(D_smooth[t] - D_smooth[t - 1])
    
    # 2. 選 change points（timestep index，不是 interval index）
    # 這裡要小心：interval t 代表 timestep t → t+1
    # 如果 Delta[t] 大（interval t 變化大），change point 應該在 timestep t 或 t+1
    # 為簡化，我們把 change point 定為 interval index t+1（即下一個 timestep 的開始）
    
    if method == "topk":
        # 取 Delta 最大的 topk 個 interval，對應的 timestep
        # 排除第一個 interval（t=0），因為它沒有 previous
        valid_indices = np.arange(1, T_minus_1)
        valid_delta = Delta[1:]
        if len(valid_delta) == 0:
            change_intervals = []
        else:
            top_k = min(topk, len(valid_delta))
            top_indices_in_valid = np.argsort(valid_delta)[-top_k:]
            change_intervals = valid_indices[top_indices_in_valid]
        
        # Change point = interval i 的結束點 = timestep i+1
        change_points = sorted([int(i + 1) for i in change_intervals])
        
    elif method == "threshold":
        thresh = np.quantile(Delta[1:], threshold_quantile)
        change_intervals = np.where(Delta[1:] >= thresh)[0] + 1  # +1 因為 Delta[1:] offset
        change_points = sorted([int(i + 1) for i in change_intervals])
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    LOGGER.info(f"Change points (analysis axis index): {change_points}")
    
    # 3. 建立 zones
    boundaries = sorted(set([0] + change_points + [T - 1]))
    
    zones = []
    for i in range(len(boundaries) - 1):
        z = Zone(
            id=i,
            axis_start=boundaries[i],
            axis_end=boundaries[i + 1] - 1 if i < len(boundaries) - 2 else T - 1,
        )
        # 避免長度為 0 的 zone
        if z.length() < 1:
            continue
        zones.append(z)
    
    # 最後一個 zone 應該結束在 T-1=99
    if zones[-1].axis_end != T - 1:
        zones[-1].axis_end = T - 1
    
    LOGGER.info(f"生成 {len(zones)} 個 zones:")
    for z in zones:
        LOGGER.info(f"  Zone {z.id}: axis={z.axis_start}..{z.axis_end} (len={z.length()})")
    
    return zones, Delta


# ============================================================
# 四、Zone-level tri-evidence aggregation
# ============================================================

def compute_zone_evidence(
    zones: List[Zone],
    S_sim: np.ndarray,
    S_svd: np.ndarray,
    FID_sens: np.ndarray,
    alpha: float = 1/3,
    beta: float = 1/3,
    gamma: float = 1/3,
) -> np.ndarray:
    """
    計算每個 block、zone 的 tri-evidence score A[b,z]。
    
    Args:
        zones: Zone 列表
        S_sim: (B, T-1) similarity-based stability
        S_svd: (B, T-1) SVD-based stability
        FID_sens: (B,) FID sensitivity
        alpha, beta, gamma: 三個分數的權重（應 sum=1）
    
    Returns:
        A: (B, Z) tri-evidence score，A[b,z] in [0,1]
    
    公式：
        對 zone z 的 axis 點區間 [axis_start..axis_end]，聚合其覆蓋的 **interval 欄位**（見程式內 slice）
        S_sim[b,z] = mean_{interval i in zone z} S_sim[b,i]（S_sim 已含 eta·L1 + (1-eta)·Cos）
        S_svd[b,z] = mean_{interval i in zone z} S_svd[b,i]
        S_fid[b] = 1 - FID_sens[b]（FID-safe score）
        A[b,z] = α*S_sim[b,z] + β*S_svd[b,z] + γ*S_fid[b]
    """
    B, T_minus_1 = S_sim.shape
    Z = len(zones)
    
    # FID-safe score（越大越安全越可 cache）
    S_fid = 1.0 - FID_sens  # shape (B,)
    
    A = np.zeros((B, Z), dtype=np.float32)
    
    for z_idx, zone in enumerate(zones):
        # Zone 對應的 interval indices（與 Stage 0 陣列欄位一致）
        # axis 點區間 [axis_start..axis_end] 與 interval slice 的對應見 slice 邏輯（與舊註解可能差一欄，行為未改）
        interval_start = zone.axis_start
        interval_end = min(zone.axis_end, T_minus_1 - 1)  # 不能超過 T-2（最後一個 interval）
        
        if interval_end < interval_start:
            # Zone 只有 1 個 timestep，沒有 interval
            # 使用前一個 interval 或設為 0
            LOGGER.warning(f"Zone {zone.id} 太短（只有 1 個 timestep），使用 A=0")
            A[:, z_idx] = 0.0
            continue
        
        # 計算 zone 內的平均穩定性
        S_sim_z = S_sim[:, interval_start:interval_end+1].mean(axis=1)  # (B,)
        S_svd_z = S_svd[:, interval_start:interval_end+1].mean(axis=1)  # (B,)
        
        # Tri-evidence score
        A[:, z_idx] = alpha * S_sim_z + beta * S_svd_z + gamma * S_fid
    
    # Clip 到 [0, 1]
    A = np.clip(A, 0, 1)
    
    LOGGER.info(f"Tri-evidence A[b,z]: shape={A.shape}, range=[{A.min():.4f}, {A.max():.4f}], mean={A.mean():.4f}")
    
    return A


# ============================================================
# 五、A[b,z] → k_raw[b,z]
# ============================================================

def map_to_k_raw(
    A: np.ndarray,
    k_min: int = 1,
    k_max: int = 8,
) -> np.ndarray:
    """
    將 tri-evidence score A[b,z] 映射到 k_raw[b,z]。
    
    Args:
        A: (B, Z) tri-evidence score in [0, 1]
        k_min: 最小 k
        k_max: 最大 k
    
    Returns:
        k_raw: (B, Z) int，cache frequency
    
    公式：
        k_raw[b,z] = k_min + round(A[b,z] * (k_max - k_min))
    """
    k_range = k_max - k_min
    k_raw = k_min + np.round(A * k_range).astype(np.int32)
    k_raw = np.clip(k_raw, k_min, k_max)
    
    LOGGER.info(f"k_raw: range=[{k_raw.min()}, {k_raw.max()}], mean={k_raw.mean():.2f}")
    
    return k_raw


# ============================================================
# 六、Zone-level risk ceiling
# ============================================================

def compute_zone_risk_ceiling(
    zones: List[Zone],
    D_smooth: np.ndarray,
    k_min: int = 1,
    k_max: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算每個 zone 的 risk R_z 和對應的 k_max_z ceiling。
    
    Args:
        zones: Zone 列表
        D_smooth: (T-1,) smoothed global drift
        k_min, k_max: k 範圍
    
    Returns:
        R_z: (Z,) zone risk（越大越危險）
        k_max_z: (Z,) zone-level k ceiling
    
    公式：
        R_z = mean_{interval i in zone z} D_smooth[i]
        k_max_z = k_min + round((1 - R_z) * (k_max - k_min))
    """
    Z = len(zones)
    T_minus_1 = len(D_smooth)
    R_z = np.zeros(Z, dtype=np.float32)
    
    for z_idx, zone in enumerate(zones):
        interval_start = zone.axis_start
        interval_end = min(zone.axis_end, T_minus_1 - 1)
        
        if interval_end < interval_start:
            R_z[z_idx] = 0.0
        else:
            R_z[z_idx] = D_smooth[interval_start:interval_end+1].mean()
    
    # k_max_z = k_min + round((1 - R_z) * (k_max - k_min))
    k_range = k_max - k_min
    k_max_z = k_min + np.round((1.0 - R_z) * k_range).astype(np.int32)
    k_max_z = np.clip(k_max_z, k_min, k_max)
    
    LOGGER.info(f"Zone risk R_z: {R_z}")
    LOGGER.info(f"Zone k_max ceiling: {k_max_z}")
    
    return R_z, k_max_z


def apply_zone_ceiling(
    k_raw: np.ndarray,
    k_max_z: np.ndarray,
) -> np.ndarray:
    """
    對每個 zone 應用 k_max ceiling。
    
    Args:
        k_raw: (B, Z)
        k_max_z: (Z,)
    
    Returns:
        k_ceiling: (B, Z)，k_ceiling[b,z] = min(k_raw[b,z], k_max_z[z])
    """
    B, Z = k_raw.shape
    k_ceiling = np.minimum(k_raw, k_max_z[None, :])  # broadcast
    
    changed = np.sum(k_ceiling != k_raw)
    LOGGER.info(f"Zone ceiling 應用：{changed}/{B*Z} 個 k 被降低")
    
    return k_ceiling


# ============================================================
# 七、Regularization（避免相鄰 zone 的 k 跳太大）
# ============================================================

def regularize_k(
    k: np.ndarray,
    mode: str = "delta1",
) -> np.ndarray:
    """
    正規化 k，避免相鄰 zones 跳太大。
    
    Args:
        k: (B, Z)
        mode: "delta1" 或 "nondecreasing"
    
    Returns:
        k_reg: (B, Z)
    
    Modes:
    - delta1: |k[b,z] - k[b,z-1]| <= 1
    - nondecreasing: k[b,z] >= k[b,z-1]
    """
    B, Z = k.shape
    k_reg = k.copy()
    
    if mode == "delta1":
        for b in range(B):
            for z in range(1, Z):
                diff = k_reg[b, z] - k_reg[b, z - 1]
                if abs(diff) > 1:
                    # 往前一格的方向拉回
                    if diff > 1:
                        k_reg[b, z] = k_reg[b, z - 1] + 1
                    else:
                        k_reg[b, z] = k_reg[b, z - 1] - 1
    
    elif mode == "nondecreasing":
        for b in range(B):
            for z in range(1, Z):
                k_reg[b, z] = max(k_reg[b, z], k_reg[b, z - 1])
    
    elif mode == "none":
        pass  # 不做 regularization
    
    else:
        raise ValueError(f"Unknown regularization mode: {mode}")
    
    changed = np.sum(k_reg != k)
    LOGGER.info(f"Regularization ({mode}): {changed}/{B*Z} 個 k 被調整")
    
    return k_reg


# ============================================================
# 八、Build recompute mask（per-timestep）
# ============================================================

def build_recompute_mask(
    T: int,
    zones: List[Zone],
    k_per_zone: List[int],
) -> np.ndarray:
    """
    從 zones + k 轉成 **analysis axis** 上的 recompute mask（長度 T=100）。

    **索引語意**：mask[i]=True 表示在 **analysis axis_idx = i** 對應的 forward 步要 full compute。
    接 DDIM 時：**t_ddim = 99 - i**（單張量 timestep 張量請用此映射）。

    Args:
        T: 總步數（100），與 axis 0..99 對齊
        zones: Zone 列表（axis_start/axis_end 為 analysis axis）
        k_per_zone: 長度 Z，每個 zone 的 k

    Returns:
        mask: (T,) bool

    規則：
    - Zone 起點一定 recompute
    - 之後每隔 k 步：axis_start, axis_start+k, ... <= axis_end
    """
    mask = np.zeros(T, dtype=bool)
    
    for zone, k in zip(zones, k_per_zone):
        t = zone.axis_start
        while t <= zone.axis_end:
            if t < T:
                mask[t] = True
            t += k
    
    return mask


# ============================================================
# 九、主流程：Stage-1 synthesis
# ============================================================

def run_stage1_synthesis(
    stage0_dir: str,
    output_dir: str,
    # Tri-evidence weights
    alpha: float = 1/3,
    beta: float = 1/3,
    gamma: float = 1/3,
    eta: float = 1.0,
    # k range
    k_min: int = 1,
    k_max: int = 8,
    # Smoothing
    smooth_window: int = 5,
    # Change point detection
    cp_method: str = "topk",
    cp_topk: int = 6,
    cp_threshold_quantile: float = 0.90,
    # Regularization
    regularize: str = "delta1",
) -> Tuple[SchedulerConfig, Dict]:
    """
    Stage-1 主流程。
    
    Args:
        stage0_dir: Stage-0E 輸出目錄
        output_dir: 輸出目錄
        alpha, beta, gamma: tri-evidence 權重
        eta: similarity 證據中 L1 權重，S_sim = eta*S_l1 + (1-eta)*S_cos
        k_min, k_max: k 範圍
        smooth_window: D_global 平滑視窗
        cp_method: change point 方法（"topk" 或 "threshold"）
        cp_topk: topk 數量
        cp_threshold_quantile: threshold quantile
        regularize: 正規化模式（"delta1", "nondecreasing", "none"）
    
    Returns:
        config: SchedulerConfig
        diagnostics: Dict（詳細診斷資料）
    """
    LOGGER.info("=" * 80)
    LOGGER.info("Stage-1: Offline Scheduler Synthesis (T=100)")
    LOGGER.info("=" * 80)
    
    # === Step 1: Load Stage-0E outputs ===
    LOGGER.info("\n[Step 1] 載入 Stage-0E 輸出...")
    block_names, S_sim, S_svd, d_norm, FID_sens, S_l1, S_cos = load_stage0e_outputs(
        stage0_dir, eta=eta
    )
    B, T_minus_1 = S_sim.shape
    T = T_minus_1 + 1  # 100
    
    # === Step 2: FID-weighted global drift ===
    LOGGER.info("\n[Step 2] 計算 FID-weighted global drift...")
    D_global, D_smooth = compute_global_drift(d_norm, FID_sens, smooth_window=smooth_window)
    
    # === Step 3: Find zones ===
    LOGGER.info("\n[Step 3] Zone segmentation...")
    zones, Delta = find_zones(
        D_smooth,
        method=cp_method,
        topk=cp_topk,
        threshold_quantile=cp_threshold_quantile,
    )
    Z = len(zones)
    
    # === Step 4: Zone-level tri-evidence ===
    LOGGER.info("\n[Step 4] 計算 zone-level tri-evidence score...")
    A = compute_zone_evidence(zones, S_sim, S_svd, FID_sens, alpha, beta, gamma)
    
    # === Step 5: Map to k_raw ===
    LOGGER.info("\n[Step 5] A[b,z] → k_raw[b,z]...")
    k_raw = map_to_k_raw(A, k_min, k_max)
    
    # === Step 6: Zone-level risk ceiling ===
    LOGGER.info("\n[Step 6] Zone-level risk ceiling...")
    R_z, k_max_z = compute_zone_risk_ceiling(zones, D_smooth, k_min, k_max)
    k_ceiling = apply_zone_ceiling(k_raw, k_max_z)
    
    # === Step 7: Regularization ===
    LOGGER.info("\n[Step 7] Regularization...")
    k_final = regularize_k(k_ceiling, mode=regularize)
    
    # === Step 8: 組裝 config ===
    LOGGER.info("\n[Step 8] 組裝 scheduler config...")
    
    params = {
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "eta": float(eta),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "smooth_window": int(smooth_window),
        "cp_method": cp_method,
        "cp_topk": int(cp_topk) if cp_method == "topk" else None,
        "cp_threshold_quantile": float(cp_threshold_quantile) if cp_method == "threshold" else None,
        "regularize": regularize,
    }
    
    zones_dict = [
        {
            "id": z.id,
            "axis_start": z.axis_start,
            "axis_end": z.axis_end,
            "t_start": z.axis_start,
            "t_end": z.axis_end,
        }
        for z in zones
    ]
    
    blocks_dict = []
    for b in range(B):
        block_dict = {
            "id": int(b),
            "name": str(block_names[b]),
            "k_per_zone": k_final[b, :].tolist(),
        }
        blocks_dict.append(block_dict)
    
    axis_order_str = "analysis_axis_0_to_99_inclusive"
    config = SchedulerConfig(
        version="v_final_stage1",
        T=T,
        t_order=axis_order_str,
        analysis_axis_order=axis_order_str,
        axis_convention="analysis_axis",
        ddim_timestep_formula="t_ddim = 99 - axis_idx",
        params=params,
        zones=zones_dict,
        blocks=blocks_dict,
    )
    
    # === Diagnostics ===
    diagnostics = {
        "axis_convention": "analysis_axis",
        "ddim_timestep_formula": "t_ddim = 99 - axis_idx",
        "analysis_axis_order": axis_order_str,
        "note_D_global_length": "D_global/D_smooth/Delta length 99 = per-interval on analysis axis (interval j between axis j and j+1)",
        "D_global": D_global.tolist(),
        "D_smooth": D_smooth.tolist(),
        "Delta": Delta.tolist(),
        "change_points": [z.axis_start for z in zones[1:]],  # 不包含第一個 zone 的起點 axis=0
        "R_z": R_z.tolist(),
        "k_max_z": k_max_z.tolist(),
        "eta": float(eta),
        "S_l1_stats": {
            "mean": float(S_l1.mean()),
            "min": float(S_l1.min()),
            "max": float(S_l1.max()),
            "std": float(S_l1.std()),
        },
        "S_cos_stats": {
            "mean": float(S_cos.mean()),
            "min": float(S_cos.min()),
            "max": float(S_cos.max()),
            "std": float(S_cos.std()),
        },
        "A_stats": {
            "mean": float(A.mean()),
            "min": float(A.min()),
            "max": float(A.max()),
            "std": float(A.std()),
        },
        "k_raw_stats": {
            "mean": float(k_raw.mean()),
            "min": int(k_raw.min()),
            "max": int(k_raw.max()),
        },
        "k_final_stats": {
            "mean": float(k_final.mean()),
            "min": int(k_final.min()),
            "max": int(k_final.max()),
        },
    }
    
    # === Step 9: 存檔 ===
    LOGGER.info("\n[Step 9] 存檔...")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    config_path = out_path / "scheduler_config.json"
    diag_path = out_path / "scheduler_diagnostics.json"
    
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    with open(diag_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    
    LOGGER.info(f"✅ Config 已存至: {config_path}")
    LOGGER.info(f"✅ Diagnostics 已存至: {diag_path}")
    
    # === Step 10: 驗證 ===
    LOGGER.info("\n[Step 10] 驗證...")
    
    # 檢查 zones 覆蓋
    zone_coverage = set()
    for z in zones:
        zone_coverage.update(range(z.axis_start, z.axis_end + 1))
    expected_coverage = set(range(T))
    if zone_coverage != expected_coverage:
        LOGGER.error(f"❌ Zone 覆蓋不完整！缺少 analysis-axis index: {expected_coverage - zone_coverage}")
    else:
        LOGGER.info(f"   ✅ Zones 完整覆蓋 analysis axis 0..{T-1}")
    
    # 檢查 k 範圍
    k_in_range = (k_final >= k_min).all() and (k_final <= k_max).all()
    if not k_in_range:
        LOGGER.error(f"❌ k_final 超出 [{k_min}, {k_max}] 範圍")
    else:
        LOGGER.info(f"   ✅ k_final 全在 [{k_min}, {k_max}]")
    
    # 測試 recompute mask（隨機選一個 block）
    test_block_idx = 0
    mask = build_recompute_mask(T, zones, k_final[test_block_idx, :].tolist())
    recompute_count = mask.sum()
    LOGGER.info(f"   ✅ Block[{test_block_idx}] recompute mask: {recompute_count}/{T} analysis-axis indices")
    
    # 檢查每個 zone 起點是否 True
    for z in zones:
        if not mask[z.axis_start]:
            LOGGER.error(f"❌ Zone {z.id} 起點 axis={z.axis_start} 的 mask 不是 True！")
    
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Stage-1 完成！")
    LOGGER.info("=" * 80)
    
    return config, diagnostics


# ============================================================
# 十、Self-test（用假資料測試）
# ============================================================

def self_test():
    """用隨機假資料測試 Stage-1 pipeline。"""
    print("\n" + "=" * 60)
    print("Self-test: Stage-1 with synthetic data")
    print("=" * 60)
    
    B, T = 4, 100
    T_minus_1 = T - 1
    
    # 生成假資料
    np.random.seed(42)
    block_names = np.array([f"block_{i}" for i in range(B)], dtype=object)
    S_sim = np.random.rand(B, T_minus_1).astype(np.float32)
    S_svd = np.random.rand(B, T_minus_1).astype(np.float32)
    d_norm = np.random.rand(B, T_minus_1).astype(np.float32) * 0.3  # 較小的 drift
    FID_sens = np.random.rand(B).astype(np.float32)
    
    # 存成臨時 .npy
    tmp_dir = Path("/tmp/stage1_self_test")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(tmp_dir / "block_names.npy", block_names)
    np.save(tmp_dir / "l1_interval_norm.npy", 1.0 - S_sim)  # Stage-0E 格式是變化量
    np.save(tmp_dir / "cosdist_interval_norm.npy", np.random.rand(B, T_minus_1) * 0.1)
    np.save(tmp_dir / "svd_interval_norm.npy", d_norm)
    np.save(tmp_dir / "fid_w_qdiffae_clip.npy", FID_sens)
    
    # 執行 Stage-1
    config, diag = run_stage1_synthesis(
        stage0_dir=str(tmp_dir),
        output_dir=str(tmp_dir / "output"),
        alpha=0.4, beta=0.4, gamma=0.2,
        eta=0.5,
        k_min=1, k_max=6,
        smooth_window=3,
        cp_method="topk",
        cp_topk=4,
        regularize="delta1",
    )
    
    # 檢查
    assert config.T == T
    assert len(config.zones) > 0
    assert len(config.blocks) == B
    
    # 檢查 zones 覆蓋
    all_t = set()
    for z_dict in config.zones:
        all_t.update(range(z_dict["t_start"], z_dict["t_end"] + 1))
    assert all_t == set(range(T)), f"Zones 沒有完整覆蓋 0..{T-1}"
    
    # 檢查 k 範圍
    for block in config.blocks:
        k_list = block["k_per_zone"]
        assert all(1 <= k <= 6 for k in k_list), f"k 超出範圍: {k_list}"
    
    print("\n✅ Self-test 通過！")
    print(f"   Zones: {len(config.zones)}")
    print(f"   Blocks: {len(config.blocks)}")
    print(f"   Sample k[block_0]: {config.blocks[0]['k_per_zone']}")
    print("=" * 60)


# ============================================================
# 十一、CLI 主入口
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Stage-1: Offline Scheduler Synthesis (T=100)"
    )
    
    # I/O
    parser.add_argument(
        "--stage0_dir",
        type=str,
        default="QATcode/cache_method/Stage0/stage0e_output",
        help="Stage-0E 輸出目錄"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="QATcode/cache_method/Stage1/stage1_output",
        help="Stage-1 輸出目錄"
    )
    
    # Tri-evidence weights
    parser.add_argument("--alpha", type=float, default=0.3, help="similarity 通道權重（對 S_sim）")
    parser.add_argument("--beta", type=float, default=0.6, help="SVD stability 權重")
    parser.add_argument("--gamma", type=float, default=0.1, help="FID-safe 權重")
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="S_sim = eta*S_l1 + (1-eta)*S_cos；1=僅 L1，0=僅 Cosine",
    )
    
    # k range
    parser.add_argument("--k_min", type=int, default=1, help="最小 k")
    parser.add_argument("--k_max", type=int, default=5, help="最大 k")
    
    # Smoothing
    parser.add_argument("--smooth_window", type=int, default=3, help="D_global 平滑視窗")
    
    # Change point detection
    parser.add_argument(
        "--cp_method",
        type=str,
        default="topk",
        choices=["topk", "threshold"],
        help="Change point 檢測方法"
    )
    parser.add_argument("--cp_topk", type=int, default=25, help="topk 模式：取 top K")
    parser.add_argument(
        "--cp_threshold_quantile",
        type=float,
        default=0.90,
        help="threshold 模式：quantile 閾值"
    )
    
    # Regularization
    parser.add_argument(
        "--regularize",
        type=str,
        default="delta1",
        choices=["delta1", "nondecreasing", "none"],
        help="k 正規化模式"
    )
    
    # Self-test
    parser.add_argument("--self_test", action="store_true", help="執行 self-test（用假資料）")
    
    args = parser.parse_args()
    
    if args.self_test:
        self_test()
        return
    
    # 正常執行
    config, diagnostics = run_stage1_synthesis(
        stage0_dir=args.stage0_dir,
        output_dir=args.output_dir,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        eta=args.eta,
        k_min=args.k_min,
        k_max=args.k_max,
        smooth_window=args.smooth_window,
        cp_method=args.cp_method,
        cp_topk=args.cp_topk,
        cp_threshold_quantile=args.cp_threshold_quantile,
        regularize=args.regularize,
    )
    
    print("\n" + "=" * 80)
    print(f"✅ Stage-1 完成！")
    print(f"   Config: {args.output_dir}/scheduler_config.json")
    print(f"   Diagnostics: {args.output_dir}/scheduler_diagnostics.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
