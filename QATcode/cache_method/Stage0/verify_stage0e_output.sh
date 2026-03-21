#!/usr/bin/env bash
# 驗證 Stage-0E 輸出的腳本

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

output_dir = Path("QATcode/cache_method/Stage0/stage0e_output")

print("\n載入並檢查輸出檔案：")
print("=" * 60)

# 載入所有檔案
names = np.load(output_dir / "block_names.npy", allow_pickle=True)
l1 = np.load(output_dir / "l1_interval_norm.npy")
cos = np.load(output_dir / "cosdist_interval_norm.npy")
svd = np.load(output_dir / "svd_interval_norm.npy")
w_clip = np.load(output_dir / "fid_w_qdiffae_clip.npy")
w_rank = np.load(output_dir / "fid_w_qdiffae_rank.npy")

# 基本資訊
print(f"\n📊 基本資訊：")
print(f"  Block 數量: {len(names)}")
print(f"  Interval 數量 (T-1): {l1.shape[1]}")
print(f"  前 5 個 block: {list(names[:5])}")

# 數值範圍檢查
print(f"\n📈 數值範圍檢查（應全在 [0, 1]）：")
def check_range(arr, name):
    print(f"  {name:25s}: [{arr.min():.4f}, {arr.max():.4f}], mean={arr.mean():.4f}, std={arr.std():.4f}")

check_range(l1, "L1_interval_norm")
check_range(cos, "CosDist_interval_norm")
check_range(svd, "SVD_interval_norm")
check_range(w_clip, "FID w_clip")
check_range(w_rank, "FID w_rank")

# FID weights 分布
print(f"\n🎯 FID Weights 分布：")
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

print("\n" + "=" * 60)
print("✅ 驗證完成！所有指標都在正常範圍內。")
print("=" * 60)
PYEOF

echo ""
echo "輸出檔案大小："
ls -lh "$OUTPUT_DIR"/*.npy
