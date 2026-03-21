#!/usr/bin/env bash
# 快速測試單一 block 的完整流程（A → B → C）

set -e

# 測試用 block
TEST_BLOCK="model.input_blocks.0"
SAFE_NAME=$(echo "$TEST_BLOCK" | tr '.' '_')

echo "=============================="
echo "測試 SVD 完整流程"
echo "Test Block: $TEST_BLOCK"
echo "=============================="

# 階段 A：收集 feature
echo ""
echo "[階段 A] 收集 Feature..."
python QATcode/cache_method/SVD/collect_features_for_svd.py \
  --num_steps 100 \
  --svd_target_block "$TEST_BLOCK" \
  --svd_target_N 32 \
  --svd_output_root QATcode/cache_method/SVD \
  --log_file QATcode/cache_method/SVD/logs/test_${SAFE_NAME}.log

# 階段 B：計算 SVD 指標
echo ""
echo "[階段 B] 計算 SVD 指標..."
python QATcode/cache_method/SVD/svd_metrics.py \
  --feature_dir QATcode/cache_method/SVD/svd_features/${SAFE_NAME} \
  --output_root QATcode/cache_method/SVD/svd_metrics \
  --representative-t -1 \
  --energy-threshold 0.98

# 階段 C：相關性分析
echo ""
echo "[階段 C] 相關性分析..."
python QATcode/cache_method/SVD/correlate_svd_similarity.py \
  --svd_metrics QATcode/cache_method/SVD/svd_metrics/${SAFE_NAME}.json \
  --similarity_npz QATcode/cache_method/L1_L2_cosine/T_100/Res/result_npz/${SAFE_NAME}.npz \
  --output_root QATcode/cache_method/SVD/correlation \
  --plot

echo ""
echo "=============================="
echo "測試完成！"
echo "=============================="
echo "輸出位置："
echo "  Features: QATcode/cache_method/SVD/svd_features/${SAFE_NAME}/"
echo "  SVD JSON: QATcode/cache_method/SVD/svd_metrics/${SAFE_NAME}.json"
echo "  Correlation: QATcode/cache_method/SVD/correlation/${SAFE_NAME}.json"
echo "  Figures: QATcode/cache_method/SVD/correlation/figures/${SAFE_NAME}_*.png"
echo "=============================="
