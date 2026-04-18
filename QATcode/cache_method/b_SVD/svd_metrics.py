"""
SVD 指標計算腳本

功能：
- 讀取 svd_features/<block_slug>/ 下的 t_{0..99}.pt
- 對每個 timestep 計算 channel covariance matrix 的 eigenvalues/eigenvectors
- 用代表 timestep 的 cumulative energy 定 rank r
- 計算相鄰 timestep 的子空間距離（實作為 t-1 -> t）
- 可選：計算 energy ratio 曲線
- 輸出：svd_metrics/<block_slug>.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# ==================== 工具函數 ====================


def load_features(feature_dir: Path) -> Tuple[List[torch.Tensor], Dict]:
    """
    載入某個 block 的所有 timestep features

    Returns:
        features: List[torch.Tensor]，每個 tensor shape (N, C, H, W)
        meta: Dict，包含 block、N、T、C、H、W
    """
    meta_path = feature_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json 不存在: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    T = meta["T"]
    features = []

    for t in tqdm(range(T), desc=f"Loading {meta['block']}"):
        pt_path = feature_dir / f"t_{t}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"t_{t}.pt 不存在: {pt_path}")

        tensor_t = torch.load(pt_path, map_location="cpu")
        features.append(tensor_t)

    return features, meta


def compute_covariance_eigen(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    計算 channel second-moment matrix（uncentered covariance-like）的 eigenvalues 與 eigenvectors

    Args:
        X: shape (N, C, H, W)

    Returns:
        eigenvalues: shape (C,)，遞減排序
        eigenvectors: shape (C, C)，每欄對應一個 eigenvalue
    """
    N, C, H, W = X.shape
    M = N * H * W

    # Reshape: (C, M)
    X_reshaped = X.permute(1, 0, 2, 3).reshape(C, M).double()

    # Uncentered second-moment matrix（未做中心化）
    # Σ = (X @ X^T) / M
    Sigma = (X_reshaped @ X_reshaped.T) / M

    # Eigen decomposition（對稱矩陣，用 eigh）
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)

    # 遞減排序
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvectors = torch.flip(eigenvectors, dims=[1])

    return eigenvalues, eigenvectors


def compute_rank_r(eigenvalues: torch.Tensor, energy_threshold: float = 0.98) -> int:
    """
    根據 cumulative energy 門檻計算 rank r

    Args:
        eigenvalues: shape (C,)，遞減排序
        energy_threshold: 能量門檻（例如 0.95、0.98）

    Returns:
        r: 最小的 r 使得 sum(λ[:r]) / sum(λ) >= energy_threshold
    """
    total_energy = eigenvalues.sum()
    cumulative_energy = torch.cumsum(eigenvalues, dim=0)

    # 找到第一個 >= threshold 的位置
    r = torch.searchsorted(cumulative_energy, energy_threshold * total_energy).item() + 1

    # 確保至少為 1
    r = max(1, min(r, len(eigenvalues)))

    return r


def compute_subspace_distance(U_t: torch.Tensor, U_prev: torch.Tensor, r: int) -> float:
    """
    計算兩個 timestep 的子空間距離

    Args:
        U_t: shape (C, C)，當前 timestep 的 eigenvectors
        U_prev: shape (C, C)，前一個 timestep 的 eigenvectors
        r: rank（取前 r 個 eigenvectors）

    Returns:
        distance: d(t, t-1) = 1 - ( || U_t^{(r)T} U_{t-1}^{(r)} ||^2 / r )
    """
    # 取前 r 個 eigenvectors
    U_t_r = U_t[:, :r]  # (C, r)
    U_prev_r = U_prev[:, :r]  # (C, r)

    # 計算內積矩陣：U_t^{(r)T} U_{t-1}^{(r)}
    M = U_t_r.T @ U_prev_r  # (r, r)

    # Frobenius norm^2
    norm_sq = (M * M).sum()

    # 距離
    distance = 1.0 - (norm_sq / r).item()

    return distance


def compute_energy_ratios(
    eigenvalues_list: List[torch.Tensor], k_values: List[int] = [4, 8, 16, 32]
) -> Dict:
    """
    計算每個 timestep 的 energy ratio 曲線

    Args:
        eigenvalues_list: List[shape (C,)]，每個 timestep 的 eigenvalues
        k_values: 要計算的 k 值列表

    Returns:
        energy_ratio: Dict，key 為 f"k{k}"，value 為長度 T 的 list
    """
    T = len(eigenvalues_list)
    energy_ratio = {f"k{k}": [] for k in k_values}

    for t in range(T):
        eigenvalues = eigenvalues_list[t]
        total_energy = eigenvalues.sum()

        for k in k_values:
            k_actual = min(k, len(eigenvalues))
            energy_k = eigenvalues[:k_actual].sum()
            ratio = (energy_k / total_energy).item()
            energy_ratio[f"k{k}"].append(ratio)

    return energy_ratio


# ==================== 主流程 ====================


def process_features_in_memory(
    features: List[torch.Tensor],
    meta: Dict,
    output_dir: Path,
    representative_t: int,
    energy_threshold: float,
    compute_energy: bool = True,
) -> Dict:
    """
    直接使用記憶體中的 features 計算 SVD 指標並輸出 JSON

    Args:
        features: List[torch.Tensor]，每個 tensor shape (N, C, H, W)
        meta: Dict，至少包含 block / N / T / C / H / W
        output_dir: 輸出目錄（svd_metrics/）
        representative_t: 代表 timestep（用於定 rank r）
        energy_threshold: energy 門檻
        compute_energy: 是否計算 energy ratio 曲線
    """
    T = meta["T"]
    C = meta["C"]
    block_slug = meta["block"]

    if len(features) != T:
        raise ValueError(f"features 長度與 T 不一致: len(features)={len(features)}, T={T}")

    print(f"\n{'='*60}")
    print(f"處理 Block: {block_slug}")
    print(f"{'='*60}")
    print(f"Block: {block_slug}")
    print(f"N={meta['N']}, T={T}, C={C}, H={meta['H']}, W={meta['W']}")

    # 1. 對每個 timestep 計算 eigenvalues / eigenvectors
    print("\n計算 covariance eigenvalues/eigenvectors...")
    eigenvalues_list = []
    eigenvectors_list = []

    for t in tqdm(range(T), desc="SVD"):
        eigenvalues, eigenvectors = compute_covariance_eigen(features[t])
        eigenvalues_list.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)

    # 2. 用代表 timestep 定 rank r
    if representative_t < 0 or representative_t >= T:
        representative_t = T - 1
        print(f"\n警告：representative_t 超出範圍，改用 T-1={representative_t}")

    eigenvalues_ref = eigenvalues_list[representative_t]
    rank_r = compute_rank_r(eigenvalues_ref, energy_threshold)

    cumulative_energy_ref = torch.cumsum(eigenvalues_ref, dim=0) / eigenvalues_ref.sum()
    actual_energy = cumulative_energy_ref[rank_r - 1].item()

    print(f"\n代表 timestep: t={representative_t}")
    print(f"Energy threshold: {energy_threshold}")
    print(f"計算得到 rank r: {rank_r} / {C}")
    print(f"實際 cumulative energy: {actual_energy:.6f}")

    # 3. 計算子空間距離
    print("\n計算子空間距離...")
    subspace_dist = [0.0]  # subspace_dist[0] 為初始化；真正 interval distance 從 t=1 開始

    for t in tqdm(range(1, T), desc="Subspace distance"):
        U_t = eigenvectors_list[t]
        U_prev = eigenvectors_list[t - 1]
        dist = compute_subspace_distance(U_t, U_prev, rank_r)
        subspace_dist.append(dist)

    print(
        f"子空間距離統計: mean={np.mean(subspace_dist[1:]):.6f}, max={np.max(subspace_dist[1:]):.6f}, min={np.min(subspace_dist[1:]):.6f}"
    )
    print("註：與 similarity 對齊時使用 subspace_dist[1:]（對應 interval）")

    # 4. 可選：計算 energy ratio
    energy_ratio = None
    if compute_energy:
        print("\n計算 energy ratio 曲線...")
        energy_ratio = compute_energy_ratios(eigenvalues_list, k_values=[4, 8, 16, 32, 64])

    # 5. 輸出 JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{block_slug}.json"

    result = {
        "block": block_slug,
        "target_block_name": meta.get("target_block_name", block_slug),
        "T": T,
        "C": C,
        "N": meta["N"],
        "H": meta["H"],
        "W": meta["W"],
        "rank_r": rank_r,
        "representative_t": representative_t,
        "energy_threshold": energy_threshold,
        "actual_energy_at_r": actual_energy,
        "timesteps": list(range(T)),
        "subspace_dist": subspace_dist,
    }

    if energy_ratio:
        result["energy_ratio"] = energy_ratio

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ 輸出: {output_path}")
    return result


def process_feature_buffers_in_memory(
    feature_buffers: Dict[int, List[torch.Tensor]],
    meta: Dict,
    target_N: int,
    output_dir: Path,
    representative_t: int,
    energy_threshold: float,
    compute_energy: bool = True,
) -> Dict:
    """
    直接從 feature buffers 串流計算 SVD，避免建立第二份完整 features 清單。
    """
    T = meta["T"]
    C = meta["C"]
    block_slug = meta["block"]

    print(f"\n{'='*60}")
    print(f"處理 Block: {block_slug}")
    print(f"{'='*60}")
    print(f"Block: {block_slug}")
    print(f"N(target)={target_N}, T={T}, C={C}, H={meta['H']}, W={meta['W']}")

    print("\n計算 covariance eigenvalues/eigenvectors（streaming）...")
    eigenvalues_list: List[torch.Tensor] = []
    eigenvectors_list: List[torch.Tensor] = []
    actual_N = None

    for t in tqdm(range(T), desc="SVD"):
        chunks = feature_buffers.get(t, [])
        if len(chunks) == 0:
            raise RuntimeError(f"t={t} 無資料，無法計算 SVD")

        tensor_t = torch.cat(chunks, dim=0)
        if tensor_t.shape[0] > target_N:
            tensor_t = tensor_t[:target_N]

        actual_N_t = int(tensor_t.shape[0])
        if actual_N_t <= 0:
            raise RuntimeError(f"t={t} 無有效樣本，無法計算 SVD")
        if actual_N is None:
            actual_N = actual_N_t

        eigenvalues, eigenvectors = compute_covariance_eigen(tensor_t)
        eigenvalues_list.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)

        # 立即釋放該 timestep 的 raw feature 記憶體
        feature_buffers[t] = []
        del tensor_t

    # 代表 timestep -> rank r
    if representative_t < 0 or representative_t >= T:
        representative_t = T - 1
        print(f"\n警告：representative_t 超出範圍，改用 T-1={representative_t}")

    eigenvalues_ref = eigenvalues_list[representative_t]
    rank_r = compute_rank_r(eigenvalues_ref, energy_threshold)

    cumulative_energy_ref = torch.cumsum(eigenvalues_ref, dim=0) / eigenvalues_ref.sum()
    actual_energy = cumulative_energy_ref[rank_r - 1].item()

    print(f"\n代表 timestep: t={representative_t}")
    print(f"Energy threshold: {energy_threshold}")
    print(f"計算得到 rank r: {rank_r} / {C}")
    print(f"實際 cumulative energy: {actual_energy:.6f}")

    print("\n計算子空間距離...")
    subspace_dist = [0.0]  # subspace_dist[0] 為初始化；真正 interval distance 從 t=1 開始
    for t in tqdm(range(1, T), desc="Subspace distance"):
        dist = compute_subspace_distance(eigenvectors_list[t], eigenvectors_list[t - 1], rank_r)
        subspace_dist.append(dist)

    print(
        f"子空間距離統計: mean={np.mean(subspace_dist[1:]):.6f}, max={np.max(subspace_dist[1:]):.6f}, min={np.min(subspace_dist[1:]):.6f}"
    )
    print("註：與 similarity 對齊時使用 subspace_dist[1:]（對應 interval）")

    energy_ratio = None
    if compute_energy:
        print("\n計算 energy ratio 曲線...")
        energy_ratio = compute_energy_ratios(eigenvalues_list, k_values=[4, 8, 16, 32, 64])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{block_slug}.json"

    result = {
        "block": block_slug,
        "target_block_name": meta.get("target_block_name", block_slug),
        "T": T,
        "C": C,
        "N": int(actual_N),
        "H": meta["H"],
        "W": meta["W"],
        "rank_r": rank_r,
        "representative_t": representative_t,
        "energy_threshold": energy_threshold,
        "actual_energy_at_r": actual_energy,
        "timesteps": list(range(T)),
        "subspace_dist": subspace_dist,
    }
    if energy_ratio:
        result["energy_ratio"] = energy_ratio

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ 輸出: {output_path}")
    return result


def process_single_block(
    feature_dir: Path,
    output_dir: Path,
    representative_t: int,
    energy_threshold: float,
    compute_energy: bool = True,
) -> "Any":
    """
    處理單一 block 的 SVD 指標計算

    Args:
        feature_dir: svd_features/<block_slug>/ 的路徑
        output_dir: 輸出目錄（svd_metrics/）
        representative_t: 代表 timestep（用於定 rank r）
        energy_threshold: energy 門檻
        compute_energy: 是否計算 energy ratio 曲線
    """
    print(f"\n{'='*60}")
    print(f"處理 Block: {feature_dir.name}")
    print(f"{'='*60}")

    # 1. 載入 features
    features, meta = load_features(feature_dir)
    return process_features_in_memory(
        features=features,
        meta=meta,
        output_dir=output_dir,
        representative_t=representative_t,
        energy_threshold=energy_threshold,
        compute_energy=compute_energy,
    )


def main() -> "Any":
    """Public function main."""
    parser = argparse.ArgumentParser(description="SVD Metrics Calculation")
    parser.add_argument(
        "--feature_dir", type=str, help="單一 block 的 feature 目錄（svd_features/<block_slug>/）"
    )
    parser.add_argument(
        "--feature_root",
        type=str,
        default="QATcode/cache_method/SVD/svd_features",
        help="svd_features 根目錄（批次處理所有 block）",
    )
    parser.add_argument(
        "--output_root", type=str, default="QATcode/cache_method/SVD/svd_metrics", help="輸出根目錄"
    )
    parser.add_argument(
        "--representative-t", type=int, default=-1, help="代表 timestep（用於定 rank r），-1 表示 T-1（預設）"
    )
    parser.add_argument(
        "--energy-threshold", type=float, default=0.98, help="Cumulative energy 門檻（預設 0.98）"
    )
    parser.add_argument(
        "--compute-energy", action="store_true", default=True, help="計算 energy ratio 曲線"
    )
    parser.add_argument("--all", action="store_true", help="批次處理 feature_root 下所有 block")

    args = parser.parse_args()

    output_dir = Path(args.output_root)

    if args.all:
        # 批次處理所有 block
        feature_root = Path(args.feature_root)
        if not feature_root.exists():
            print(f"錯誤：{feature_root} 不存在")
            return

        block_dirs = sorted([d for d in feature_root.iterdir() if d.is_dir()])
        print(f"找到 {len(block_dirs)} 個 block")

        for block_dir in block_dirs:
            try:
                # representative_t = -1 表示用 T-1
                rep_t = args.representative_t
                if rep_t < 0:
                    # 從 meta.json 讀取 T
                    with open(block_dir / "meta.json", "r") as f:
                        meta_temp = json.load(f)
                    rep_t = meta_temp["T"] - 1

                process_single_block(
                    block_dir,
                    output_dir,
                    representative_t=rep_t,
                    energy_threshold=args.energy_threshold,
                    compute_energy=args.compute_energy,
                )
            except Exception as e:
                print(f"處理 {block_dir.name} 時出錯: {e}")
                import traceback

                traceback.print_exc()
                continue

        print(f"\n{'='*60}")
        print(f"批次處理完成，共 {len(block_dirs)} 個 block")
        print(f"{'='*60}")

    elif args.feature_dir:
        # 處理單一 block
        feature_dir = Path(args.feature_dir)
        if not feature_dir.exists():
            print(f"錯誤：{feature_dir} 不存在")
            return

        # representative_t = -1 表示用 T-1
        rep_t = args.representative_t
        if rep_t < 0:
            with open(feature_dir / "meta.json", "r") as f:
                meta_temp = json.load(f)
            rep_t = meta_temp["T"] - 1

        process_single_block(
            feature_dir,
            output_dir,
            representative_t=rep_t,
            energy_threshold=args.energy_threshold,
            compute_energy=args.compute_energy,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
