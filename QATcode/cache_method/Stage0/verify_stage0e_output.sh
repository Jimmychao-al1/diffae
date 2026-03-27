#!/usr/bin/env bash
# 驗證 Stage-0E 輸出的腳本（輕量 fail-fast validator）

OUTPUT_DIR="QATcode/cache_method/Stage0/stage0e_output"

echo "=============================="
echo "Stage-0E 輸出驗證"
echo "=============================="

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "❌ 輸出目錄不存在: $OUTPUT_DIR"
    echo "請先執行: python3 QATcode/cache_method/Stage0/stage0e_normalization.py"
    exit 1
fi

python3 << 'PYEOF'
import numpy as np
from pathlib import Path
import sys

output_dir = Path("QATcode/cache_method/Stage0/stage0e_output")

print("\n載入並檢查輸出檔案：")
print("=" * 60)

required_files = [
    "l1_interval_norm.npy",
    "cosdist_interval_norm.npy",
    "svd_interval_norm.npy",
    "delta_fid.npy",
    "fid_weights.npy",
    "block_names_metric.npy",
    "t_curr_interval.npy",
    "axis_interval_def.npy",
]

missing = [f for f in required_files if not (output_dir / f).exists()]
if missing:
    print(f"❌ 缺少必要檔案: {missing}")
    sys.exit(1)

# 保留原本主要檔名相容
names = np.load(output_dir / "block_names.npy", allow_pickle=True) if (output_dir / "block_names.npy").exists() else np.load(output_dir / "block_names_metric.npy", allow_pickle=True)
l1 = np.load(output_dir / "l1_interval_norm.npy")
cos = np.load(output_dir / "cosdist_interval_norm.npy")
svd = np.load(output_dir / "svd_interval_norm.npy")
w_clip = np.load(output_dir / "fid_w_qdiffae_clip.npy") if (output_dir / "fid_w_qdiffae_clip.npy").exists() else np.load(output_dir / "fid_weights.npy")
w_rank = np.load(output_dir / "fid_w_qdiffae_rank.npy") if (output_dir / "fid_w_qdiffae_rank.npy").exists() else None
t_curr = np.load(output_dir / "t_curr_interval.npy")
axis_def_obj = np.load(output_dir / "axis_interval_def.npy", allow_pickle=True)
axis_def = str(axis_def_obj.item() if hasattr(axis_def_obj, "item") else axis_def_obj)

# 基本資訊
print(f"\n📊 基本資訊：")
print(f"  Block 數量: {len(names)}")
print(f"  Interval 數量 (T-1): {l1.shape[1]}")
print(f"  前 5 個 block: {list(names[:5])}")
print(f"  axis_interval_def: {axis_def}")
print(f"  t_curr_interval 長度: {len(t_curr)}")
print(f"  t_curr_interval 前 5: {t_curr[:5].tolist()}")
print(f"  t_curr_interval 後 5: {t_curr[-5:].tolist()}")

errors = []

# shape 檢查
if l1.ndim != 2 or cos.ndim != 2 or svd.ndim != 2:
    errors.append(f"metric arrays 必須是 2D: l1={l1.shape}, cos={cos.shape}, svd={svd.shape}")
else:
    if l1.shape != cos.shape or l1.shape != svd.shape:
        errors.append(f"metric shape 不一致: l1={l1.shape}, cos={cos.shape}, svd={svd.shape}")
    if l1.shape[0] != len(names):
        errors.append(f"block 維度不一致: len(names)={len(names)}, metric B={l1.shape[0]}")
    if len(t_curr) != l1.shape[1]:
        errors.append(f"時間軸長度不一致: len(t_curr)={len(t_curr)}, T-1={l1.shape[1]}")

# 數值範圍檢查
print(f"\n📈 數值範圍檢查（應全在 [0, 1]）：")
def check_range(arr, name):
    print(f"  {name:25s}: [{arr.min():.4f}, {arr.max():.4f}], mean={arr.mean():.4f}, std={arr.std():.4f}")
    if np.isnan(arr).any() or np.isinf(arr).any():
        errors.append(f"{name} 含有 NaN/Inf")
    tol = 1e-6
    if arr.min() < -tol or arr.max() > 1.0 + tol:
        errors.append(f"{name} 超出 [0,1] 容許範圍: [{arr.min():.6f}, {arr.max():.6f}]")

check_range(l1, "L1_interval_norm")
check_range(cos, "CosDist_interval_norm")
check_range(svd, "SVD_interval_norm")
check_range(w_clip, "FID w_clip")
if w_rank is not None:
    check_range(w_rank, "FID w_rank")

# FID weights 分佈
print(f"\n🎯 FID Weights 分佈：")
nonzero_count = np.sum(w_clip > 0)
print(f"  非零 block 數: {nonzero_count}/{len(names)}")
print(f"  Top 5 敏感 block:")
top5_idx = np.argsort(w_clip)[::-1][:5]
for rank, idx in enumerate(top5_idx, 1):
    print(f"    {rank}. {names[idx]:30s} w={w_clip[idx]:.4f}")

# 採樣檢查：查看特定 block 的指標
print(f"\n🔍 採樣檢查（block=model.input_blocks.0）：")
block_idx = 0
intervals_to_check = [0, 25, 50, 75, 98]
print(f"  {'Interval':>10s} {'L1':>8s} {'Cosine':>8s} {'SVD':>8s}")
for i in intervals_to_check:
    print(f"  {i:>10d} {l1[block_idx, i]:8.4f} {cos[block_idx, i]:8.4f} {svd[block_idx, i]:8.4f}")

if errors:
    print("\n" + "=" * 60)
    print("❌ Stage0 output validation failed")
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err}")
    print("=" * 60)
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ Stage0 output validation passed")
print("=" * 60)
PYEOF

echo ""
echo "輸出檔案大小："
ls -lh "$OUTPUT_DIR"/*.npy
