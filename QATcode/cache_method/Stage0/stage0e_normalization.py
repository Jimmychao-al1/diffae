"""
Stage-0E: Loader + Normalization for Cache Scheduler

此模組讀取 Stage-0 的原始實驗資料（T=100）並產生正規化的 interval-wise 指標。

資料來源：
1. L1 / Cosine: QATcode/cache_method/L1_L2_cosine/T_100/Res/result_npz/*.npz
2. SVD drift: QATcode/cache_method/SVD/svd_metrics/*.json
3. FID sensitivity: QATcode/cache_method/FID/fid_cache_sensitivity/fid_sensitivity_results.json

輸出：
- 正規化的 interval-wise 指標 (B, T-1)，其中 interval i 代表 step i → i+1
- FID-based block weights w_b (B,)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# 設定 logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
LOGGER = logging.getLogger("Stage0E")


#=============================================================================
# 一、載入 interval-wise 指標
#=============================================================================

def load_interval_metrics(
    l1_cos_dir: str,
    svd_dir: str
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    掃描兩個資料夾，讀取所有 block 的 L1 / Cos / SVD。
    
    Args:
        l1_cos_dir: L1/Cosine npz 檔案目錄，例如 "QATcode/cache_method/L1_L2_cosine/T_100/Res/result_npz"
        svd_dir: SVD metrics JSON 目錄，例如 "QATcode/cache_method/SVD/svd_metrics"
    
    Returns:
        block_names: List[str]，block 名稱列表（依 npz 檔名排序）
        L1_interval: np.ndarray, shape (B, T-1)，第 j 欄 = **analysis axis 上 interval j**（與 .npz 一致）
        CosDist_interval: 同上
        SVD_interval: 同上
    
    Interval mapping（與 similarity_calculation / L1_L2_cosine .npz 一致）：
    - 欄位索引 j ∈ [0..T-2]：**interval j** = 沿 **analysis axis** 在點 j 與 j+1 之間的變化
    - 與 DDIM 進模型 timestep：**該區間對應 t_ddim 由 (99−j) 變到 (98−j)**（T=100）
    - 勿將 j 誤解為「DDIM 張量上的 t=j→t=j+1」；兩者索引方向相反，見 cache_time_axis_audit.md
    - L1: l1_rate_step_mean[j]；Cos: 1.0 - cos_step_mean[j]；SVD: subspace_dist[j+1]（見下方原始對應）
    """
    l1_cos_path = Path(l1_cos_dir)
    svd_path = Path(svd_dir)
    
    if not l1_cos_path.exists():
        raise FileNotFoundError(f"L1/Cosine 目錄不存在: {l1_cos_path}")
    if not svd_path.exists():
        raise FileNotFoundError(f"SVD 目錄不存在: {svd_path}")
    
    # 1. 掃描 npz 檔案，取得所有 block 名稱
    npz_files = sorted(l1_cos_path.glob("*.npz"))
    if len(npz_files) == 0:
        raise ValueError(f"在 {l1_cos_path} 中找不到任何 .npz 檔案")
    
    block_names = []
    L1_list = []
    CosDist_list = []
    SVD_list = []
    
    LOGGER.info(f"載入 {len(npz_files)} 個 block 的 interval-wise 指標...")
    
    for npz_file in npz_files:
        block_slug = npz_file.stem  # 例如 "model_input_blocks_0"
        
        # 轉回原始格式：model_input_blocks_0 -> model.input_blocks.0
        # 策略：只替換特定的關鍵下劃線
        block_name = block_slug
        block_name = block_name.replace('model_', 'model.', 1)  # model_ -> model.
        block_name = block_name.replace('_blocks_', '_blocks.', 1)  # _blocks_ -> _blocks.
        block_name = block_name.replace('input_', 'input_')  # 保持 input_
        block_name = block_name.replace('output_', 'output_')  # 保持 output_
        block_name = block_name.replace('middle_', 'middle_')  # 保持 middle_
        # 然後再全部的 _ 替換成 .
        # 不對，應該更小心：
        # model_input_blocks_0 -> 想要 model.input_blocks.0
        # model_middle_block -> 想要 model.middle_block
        
        # 重新設計：
        if block_slug.startswith('model_'):
            rest = block_slug[6:]  # 去掉 'model_'
            # rest 可能是：input_blocks_0, middle_block, output_blocks_5
            parts = rest.split('_')
            if len(parts) >= 3 and parts[-1].isdigit():
                # input_blocks_0 -> ['input', 'blocks', '0']
                # 合成：input_blocks.0
                block_name = 'model.' + '_'.join(parts[:-1]) + '.' + parts[-1]
            else:
                # middle_block -> ['middle', 'block']
                block_name = 'model.' + rest.replace('_', '_')
        else:
            block_name = block_slug
        
        # 2. 讀取 L1 / Cosine
        try:
            data = np.load(npz_file)
            l1_rate_step_mean = data['l1_rate_step_mean']  # shape (T-1,)
            cos_step_mean = data['cos_step_mean']  # shape (T-1,)，cosine similarity
            
            T_minus_1 = len(l1_rate_step_mean)
            
            # 3. 讀取 SVD
            svd_json_path = svd_path / f"{block_slug}.json"
            if not svd_json_path.exists():
                LOGGER.warning(f"SVD JSON 不存在，跳過 block: {block_name}")
                continue
            
            with open(svd_json_path, 'r') as f:
                svd_data = json.load(f)
            
            subspace_dist = np.array(svd_data['subspace_dist'])  # shape (T,)
            
            # 4. Interval mapping
            # interval i in [0..T-2] 代表 step i → i+1
            L1_interval = l1_rate_step_mean  # shape (T-1,)
            CosDist_interval = 1.0 - cos_step_mean  # 轉成 cosine distance
            
            # SVD: subspace_dist[t] 測量 (t-1) → t
            # 所以 interval i (step i → i+1) 對應 subspace_dist[i+1]
            SVD_interval = subspace_dist[1:]  # shape (T-1,)，取 subspace_dist[1..T-1]
            
            # 檢查長度一致性
            assert len(L1_interval) == T_minus_1
            assert len(CosDist_interval) == T_minus_1
            assert len(SVD_interval) == T_minus_1
            
            block_names.append(block_name)
            L1_list.append(L1_interval)
            CosDist_list.append(CosDist_interval)
            SVD_list.append(SVD_interval)
            
        except Exception as e:
            LOGGER.error(f"載入 {block_name} 時出錯: {e}")
            continue
    
    if len(block_names) == 0:
        raise ValueError("沒有成功載入任何 block 的資料")
    
    # 5. 堆疊成 (B, T-1) 陣列
    L1_interval_all = np.stack(L1_list, axis=0)  # (B, T-1)
    CosDist_interval_all = np.stack(CosDist_list, axis=0)  # (B, T-1)
    SVD_interval_all = np.stack(SVD_list, axis=0)  # (B, T-1)
    
    LOGGER.info(f"✅ 成功載入 {len(block_names)} 個 block")
    LOGGER.info(f"   L1_interval shape: {L1_interval_all.shape}")
    LOGGER.info(f"   CosDist_interval shape: {CosDist_interval_all.shape}")
    LOGGER.info(f"   SVD_interval shape: {SVD_interval_all.shape}")
    
    return block_names, L1_interval_all, CosDist_interval_all, SVD_interval_all


#=============================================================================
# 二、Min-max 正規化
#=============================================================================

def normalize_minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    對 x 全體元素做 min-max 正規化到 [0, 1]。
    
    Args:
        x: 任意 shape 的 numpy array，例如 (B, T-1)
        eps: 最小範圍閾值，若 max - min <= eps 則回傳全零
    
    Returns:
        x_norm: 正規化後的陣列，同 shape，值域 [0, 1]
    
    處理規則：
    - 過濾 NaN/Inf
    - 若 max - min > eps：x_norm = (x - min) / (max - min)
    - 否則：x_norm = 全零
    - 最後 clip 到 [0, 1]
    """
    # 1. 過濾 NaN/Inf，建立 mask
    valid_mask = np.isfinite(x)
    
    if not np.any(valid_mask):
        LOGGER.warning("輸入陣列全為 NaN/Inf，回傳全零")
        return np.zeros_like(x, dtype=np.float32)
    
    # 2. 計算有效值的 min/max
    x_valid = x[valid_mask]
    x_min = x_valid.min()
    x_max = x_valid.max()
    
    # 3. 判斷範圍
    if x_max - x_min <= eps:
        LOGGER.warning(f"數值範圍過小 (max - min = {x_max - x_min:.2e} <= eps)，回傳全零")
        return np.zeros_like(x, dtype=np.float32)
    
    # 4. 正規化
    x_norm = (x - x_min) / (x_max - x_min)
    
    # 5. 處理殘餘的 NaN/Inf（理論上不應該出現，但保險起見）
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 6. Clip 到 [0, 1]
    x_norm = np.clip(x_norm, 0.0, 1.0)
    
    return x_norm.astype(np.float32)


#=============================================================================
# 三、載入 FID sensitivity 資料
#=============================================================================

def load_delta_fid_qdiffae(
    fid_json_path: str
) -> Tuple[List[str], Dict[str, Dict[int, float]]]:
    """
    從 JSON 讀取 T=100 的 Delta-FID（Q-DiffAE）。
    
    Args:
        fid_json_path: FID sensitivity 結果 JSON 路徑
    
    Returns:
        block_names: List[str]，block 名稱列表
        delta_fid: Dict[str, Dict[int, float]]
            delta_fid[block_name][k] = delta_FID at T=100, hop k
            其中 k in {3, 4, 5}
    
    JSON 結構：
        results["100steps"]["baseline_fid"]
        results["100steps"]["k3" / "k4" / "k5"][layer_name]["delta"]
    """
    fid_path = Path(fid_json_path)
    if not fid_path.exists():
        raise FileNotFoundError(f"FID JSON 不存在: {fid_path}")
    
    with open(fid_path, 'r') as f:
        results = json.load(f)
    
    # 檢查是否有 T=100 的資料（可能是 "100steps" 或 "T100"）
    results_dict = results.get("results", {})
    
    if "100steps" in results_dict:
        step_results = results_dict["100steps"]
    elif "T100" in results_dict:
        step_results = results_dict["T100"]
    else:
        available_keys = list(results_dict.keys())
        raise ValueError(
            f"FID JSON 中沒有 T=100 的資料。可用的 keys: {available_keys}\n"
            f"請確認已執行 T=100 的 FID 實驗"
        )
    baseline_fid = step_results.get("baseline_fid")
    
    if baseline_fid is None:
        raise ValueError("找不到 baseline_fid (T=100)")
    
    LOGGER.info(f"Baseline FID (T=100): {baseline_fid:.4f}")
    
    # 收集所有 block 的 delta_fid
    delta_fid = {}
    k_values = [3, 4, 5]
    
    for k in k_values:
        k_key = f"k{k}"
        if k_key not in step_results:
            LOGGER.warning(f"找不到 k={k} 的資料，跳過")
            continue
        
        for layer_name, layer_data in step_results[k_key].items():
            if layer_name not in delta_fid:
                delta_fid[layer_name] = {}
            
            delta = layer_data.get("delta")
            if delta is not None:
                delta_fid[layer_name][k] = float(delta)
    
    block_names = sorted(delta_fid.keys())
    
    LOGGER.info(f"✅ 成功載入 {len(block_names)} 個 block 的 Delta-FID (T=100)")
    LOGGER.info(f"   k values: {k_values}")
    
    return block_names, delta_fid


#=============================================================================
# 四、FID-based block weight 計算
#=============================================================================

def compute_fid_weights(
    block_names: List[str],
    delta_fid: Dict[str, Dict[int, float]],
    eps_noise: float = 0.5,
    quantile: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算 FID-based block weights w_b (T=100)。
    
    Args:
        block_names: block 名稱列表，順序與輸出對應
        delta_fid: delta_fid[block_name][k] = delta_FID at hop k
        eps_noise: noise 修正閾值（預設 0.5）
        quantile: quantile clipping 閾值（預設 0.95）
    
    Returns:
        w_clip: np.ndarray, shape (B,)，quantile-clipped + normalized weights
        w_rank: np.ndarray, shape (B,)，rank-based weights（用於 ablation）
    
    步驟：
    1. Noise 修正：delta_pos = max(0, delta - eps_noise)
    2. Worst-case 聚合：S_b = max_k delta_pos[b][k]
    3. Quantile clipping：S_clip = min(S, quantile_95)
    4. Min-max 正規化：w = S_clip / max(S_clip)
    5. Rank-based（可選）：按 S 排序後線性對應映射到 [0, 1]
    """
    B = len(block_names)
    S = np.zeros(B, dtype=np.float32)
    
    # 1. Noise 修正 + Worst-case 聚合
    for i, block_name in enumerate(block_names):
        if block_name not in delta_fid:
            LOGGER.warning(f"Block {block_name} 沒有 FID 資料，使用 S_b = 0")
            continue
        
        delta_dict = delta_fid[block_name]
        delta_pos_values = []
        
        for k in [3, 4, 5]:
            if k in delta_dict:
                delta = delta_dict[k]
                delta_pos = max(0.0, delta - eps_noise)
                delta_pos_values.append(delta_pos)
        
        if len(delta_pos_values) > 0:
            S[i] = max(delta_pos_values)  # Worst-case
    
    LOGGER.info(f"原始 S 統計：min={S.min():.4f}, max={S.max():.4f}, mean={S.mean():.4f}")
    
    # 2. 檢查是否全零
    if np.all(S == 0):
        LOGGER.warning("所有 S_b = 0，回傳全零權重")
        w_clip = np.zeros(B, dtype=np.float32)
        w_rank = np.zeros(B, dtype=np.float32)
        return w_clip, w_rank
    
    # 3. Quantile clipping
    hi = np.quantile(S, quantile)
    S_clip = np.minimum(S, hi)
    
    LOGGER.info(f"Quantile clipping (q={quantile}): hi={hi:.4f}")
    LOGGER.info(f"S_clip 統計：min={S_clip.min():.4f}, max={S_clip.max():.4f}, mean={S_clip.mean():.4f}")
    
    # 4. Min-max 正規化
    h = S_clip.max()
    if h <= 0:
        LOGGER.warning("S_clip.max() <= 0，回傳全零權重")
        w_clip = np.zeros(B, dtype=np.float32)
    else:
        w_clip = S_clip / h
    
    LOGGER.info(f"w_clip 統計：min={w_clip.min():.4f}, max={w_clip.max():.4f}, mean={w_clip.mean():.4f}")
    
    # 5. Rank-based weights（用於 ablation）
    w_rank = rank_based_weights(S)
    LOGGER.info(f"w_rank 統計：min={w_rank.min():.4f}, max={w_rank.max():.4f}, mean={w_rank.mean():.4f}")
    
    return w_clip, w_rank


def rank_based_weights(S: np.ndarray) -> np.ndarray:
    """
    對 S (B,) 做排序，回傳 [0, 1] 的 rank 權重。
    
    Args:
        S: shape (B,)，任意數值
    
    Returns:
        w_rank: shape (B,)，rank-based weights in [0, 1]
    
    規則：
    - 最小的 S 對應 w_rank = 0
    - 最大的 S 對應 w_rank = 1
    - 中間線性插值
    """
    B = S.shape[0]
    order = np.argsort(S)  # ascending order
    w_rank = np.zeros_like(S, dtype=np.float32)
    
    if B > 1:
        for i, idx in enumerate(order):
            w_rank[idx] = i / (B - 1)
    else:
        w_rank[0] = 1.0  # 只有一個 block 時設為 1
    
    return w_rank


#=============================================================================
# 五、主入口函式
#=============================================================================

def run_stage0e(
    l1_cos_dir: str,
    svd_dir: str,
    fid_json_path: str,
    output_dir: str,
    eps_noise: float = 0.5,
    quantile: float = 0.95,
):
    """
    Stage-0E 主流程：讀取 + 正規化 + 輸出。
    
    Args:
        l1_cos_dir: L1/Cosine npz 檔案目錄
        svd_dir: SVD metrics JSON 目錄
        fid_json_path: FID sensitivity 結果 JSON 路徑
        output_dir: 輸出目錄
        eps_noise: FID noise 修正閾值
        quantile: FID quantile clipping 閾值
    
    輸出檔案（都存成 .npy）：
    - block_names.npy: (B,) object array，block 名稱列表
    - l1_interval_norm.npy: (B, T-1) float32，正規化的 L1rel_rate
    - cosdist_interval_norm.npy: (B, T-1) float32，正規化的 cosine distance
    - svd_interval_norm.npy: (B, T-1) float32，正規化的 SVD 子空間距離
    - fid_w_qdiffae_clip.npy: (B,) float32，quantile-clipped FID weights
    - fid_w_qdiffae_rank.npy: (B,) float32，rank-based FID weights（用於 ablation）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("=" * 80)
    LOGGER.info("Stage-0E: Loader + Normalization (T=100)")
    LOGGER.info("=" * 80)
    
    # 1. 載入 interval-wise 指標
    LOGGER.info("\n[步驟 1] 載入 L1 / Cosine / SVD interval-wise 指標...")
    block_names_metric, L1_interval_all, CosDist_interval_all, SVD_interval_all = load_interval_metrics(
        l1_cos_dir=l1_cos_dir,
        svd_dir=svd_dir
    )
    
    # 2. 正規化
    LOGGER.info("\n[步驟 2] Min-max 正規化（三種指標獨立處理）...")
    
    LOGGER.info("  正規化 L1_interval...")
    L1_interval_norm = normalize_minmax(L1_interval_all)
    LOGGER.info(f"    L1_interval_norm: min={L1_interval_norm.min():.4f}, max={L1_interval_norm.max():.4f}, "
                f"mean={L1_interval_norm.mean():.4f}, std={L1_interval_norm.std():.4f}")
    
    LOGGER.info("  正規化 CosDist_interval...")
    CosDist_interval_norm = normalize_minmax(CosDist_interval_all)
    LOGGER.info(f"    CosDist_interval_norm: min={CosDist_interval_norm.min():.4f}, max={CosDist_interval_norm.max():.4f}, "
                f"mean={CosDist_interval_norm.mean():.4f}, std={CosDist_interval_norm.std():.4f}")
    
    LOGGER.info("  正規化 SVD_interval...")
    SVD_interval_norm = normalize_minmax(SVD_interval_all)
    LOGGER.info(f"    SVD_interval_norm: min={SVD_interval_norm.min():.4f}, max={SVD_interval_norm.max():.4f}, "
                f"mean={SVD_interval_norm.mean():.4f}, std={SVD_interval_norm.std():.4f}")
    
    # 3. 載入 FID sensitivity 並計算權重
    LOGGER.info("\n[步驟 3] 載入 FID sensitivity (T=100) 並計算 block weights...")
    block_names_fid, delta_fid = load_delta_fid_qdiffae(fid_json_path=fid_json_path)
    
    # 確保 block 名稱對齊
    # FID JSON 的 layer 名稱格式：encoder_layer_X / decoder_layer_X / middle_layer
    # Metric 的 block 名稱格式：model.input_blocks.X / model.output_blocks.X / model.middle_block
    
    # 建立 mapping: metric block name -> FID layer name
    def metric_to_fid_name(block_name: str) -> str:
        """
        將 metric block 名稱轉換為 FID layer 名稱。
        
        Examples:
            model.input_blocks.0 -> encoder_layer_0
            model.middle_block -> middle_layer
            model.output_blocks.5 -> decoder_layer_5
        """
        if 'input_blocks' in block_name:
            idx = block_name.split('.')[-1]
            return f"encoder_layer_{idx}"
        elif 'middle_block' in block_name:
            return "middle_layer"
        elif 'output_blocks' in block_name:
            idx = block_name.split('.')[-1]
            return f"decoder_layer_{idx}"
        else:
            return block_name  # Fallback
    
    # 對齊到 metric block_names 的順序
    delta_fid_aligned = {}
    for block_name in block_names_metric:
        fid_layer_name = metric_to_fid_name(block_name)
        if fid_layer_name in delta_fid:
            delta_fid_aligned[block_name] = delta_fid[fid_layer_name]
        else:
            LOGGER.warning(f"Block {block_name} (FID: {fid_layer_name}) 在 FID 資料中找不到，使用空字典")
            delta_fid_aligned[block_name] = {}
    
    w_clip, w_rank = compute_fid_weights(
        block_names=block_names_metric,
        delta_fid=delta_fid_aligned,
        eps_noise=eps_noise,
        quantile=quantile
    )
    
    # 4. 檢查數值有效性
    LOGGER.info("\n[步驟 4] 檢查數值有效性...")
    
    def check_array(arr, name):
        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()
        in_range = (arr >= 0).all() and (arr <= 1).all()
        LOGGER.info(f"  {name}: NaN={has_nan}, Inf={has_inf}, in [0,1]={in_range}")
        if has_nan or has_inf or not in_range:
            LOGGER.error(f"    ❌ {name} 包含異常值！")
        else:
            LOGGER.info(f"    ✅ {name} 正常")
    
    check_array(L1_interval_norm, "L1_interval_norm")
    check_array(CosDist_interval_norm, "CosDist_interval_norm")
    check_array(SVD_interval_norm, "SVD_interval_norm")
    check_array(w_clip, "fid_w_clip")
    check_array(w_rank, "fid_w_rank")
    
    # 5. 存檔
    LOGGER.info("\n[步驟 5] 存檔...")
    
    # 轉成 numpy object array 以便儲存字串列表
    block_names_array = np.array(block_names_metric, dtype=object)
    
    np.save(output_path / "block_names.npy", block_names_array)
    np.save(output_path / "l1_interval_norm.npy", L1_interval_norm)
    np.save(output_path / "cosdist_interval_norm.npy", CosDist_interval_norm)
    np.save(output_path / "svd_interval_norm.npy", SVD_interval_norm)
    np.save(output_path / "fid_w_qdiffae_clip.npy", w_clip)
    np.save(output_path / "fid_w_qdiffae_rank.npy", w_rank)

    # ------------------------------------------------------------
    # 輸出層：時間軸對齊資訊（不改變既有數值陣列）
    # interval j (analysis axis, 0..T-2) -> 顯示 t_curr = (T-2) - j
    # 若此處 interval_len = T-1，則 t_curr = (interval_len-1) - j
    # ------------------------------------------------------------
    interval_len = int(L1_interval_norm.shape[1])  # expected 99 when T=100
    t_curr_interval = (interval_len - 1) - np.arange(interval_len, dtype=np.int32)
    np.save(output_path / "t_curr_interval.npy", t_curr_interval)
    np.save(
        output_path / "axis_interval_def.npy",
        np.array(
            "interval-wise: analysis interval index j (0..T-2) keeps internal order; display label is t_curr=(T-2)-j",
            dtype=object,
        ),
    )
    
    LOGGER.info(f"✅ 所有檔案已儲存至: {output_path}")
    LOGGER.info("\n輸出檔案列表：")
    LOGGER.info(f"  - block_names.npy: shape {block_names_array.shape}")
    LOGGER.info(f"  - l1_interval_norm.npy: shape {L1_interval_norm.shape}")
    LOGGER.info(f"  - cosdist_interval_norm.npy: shape {CosDist_interval_norm.shape}")
    LOGGER.info(f"  - svd_interval_norm.npy: shape {SVD_interval_norm.shape}")
    LOGGER.info(f"  - fid_w_qdiffae_clip.npy: shape {w_clip.shape}")
    LOGGER.info(f"  - fid_w_qdiffae_rank.npy: shape {w_rank.shape}")
    
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Stage-0E 完成！")
    LOGGER.info("=" * 80)


#=============================================================================
# 主程式入口
#=============================================================================

if __name__ == "__main__":
    # 預設路徑（使用者需要根據實際專案結構調整）
    import os
    
    # 假設從專案根目錄執行
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    
    run_stage0e(
        l1_cos_dir=os.path.join(project_root, "QATcode/cache_method/L1_L2_cosine/T_100/v2_latest/result_npz"),
        svd_dir=os.path.join(project_root, "QATcode/cache_method/SVD/svd_metrics"),
        fid_json_path=os.path.join(project_root, "QATcode/cache_method/FID/fid_cache_sensitivity/fid_sensitivity_results.json"),
        output_dir=os.path.join(project_root, "QATcode/cache_method/Stage0/stage0e_output"),
        eps_noise=0.5,
        quantile=0.95
    )
