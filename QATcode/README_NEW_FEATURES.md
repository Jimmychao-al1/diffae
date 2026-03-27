# 新功能使用說明

本文檔說明 Step 6 量化訓練中新增的三個主要功能及其使用方法。

## 功能概覽

### 1. Layer-by-Layer 逐層訓練
**檔案**: `layer_by_layer_trainer.py`
**功能**: 實現 144 層的逐層量化訓練策略
**預設狀態**: 關閉（zero-diff）

### 2. 代理指標驗證
**檔案**: `proxy_validation.py`
**功能**: 門檻驗證、FID/dFID 趨勢評估
**預設狀態**: 關閉（zero-diff）

## 使用方法

### 基本用法（保持原有行為）
```bash
python QATcode/quantize_diffae_step6_train.py
```
所有新功能預設關閉，行為與原版本完全一致。

### 啟用特定功能
```bash
# 啟用 Layer-by-Layer 訓練
python QATcode/quantize_diffae_step6_train.py --enable-layer-by-layer

# 啟用代理指標驗證
python QATcode/quantize_diffae_step6_train.py --enable-proxy-validation

# 同時啟用所有功能
python QATcode/quantize_diffae_step6_train.py \
    --enable-layer-by-layer \
    --enable-proxy-validation \
    --exp-dir runs/full_features_exp
```

### 完整參數列表
```bash
python QATcode/quantize_diffae_step6_train.py --help
```

- `--enable-layer-by-layer`: 啟用逐層訓練
- `--enable-proxy-validation`: 啟用代理指標驗證
- `--exp-dir`: 實驗目錄路徑（預設: `runs/default`）
- `--dry-run`: 乾執行模式（用於測試）
- `--pause`: 分析後暫停等待

## 詳細功能說明

### 1. Layer-by-Layer 逐層訓練

#### 核心概念
- **144 層量化控制**: 自動識別模型中的可量化層
- **開關語意**: 
  - 已完成層: `(W=True, A=True, requires_grad=False)`
  - 當前層: `(W=True, A=True, requires_grad=True)`
  - 未輪到層: `(W=False, A=False, requires_grad=False)`
- **Optimizer 限制**: 僅包含當前層參數
- **檢查點管理**: 每層保存最佳檢查點，下層從最佳檢查點續訓

#### 使用流程
1. 首次執行：自動從第 0 層開始
2. 層完成後：自動推進到下一層
3. 重新啟動：自動載入上次的層狀態
4. 檢查點：保存在 `{exp_dir}/layer_checkpoints/`
5. 狀態文件：`{exp_dir}/layer_states.json`

#### 輸出結構
```
runs/{exp_dir}/
├── layer_checkpoints/
│   ├── layer_0_best.pth
│   ├── layer_1_best.pth
│   └── ...
├── layer_states.json
└── ...
```

### 2. 代理指標驗證

#### 門檻驗證（每層結束立即執行）
- **樣本數**: 256-512 張無條件樣本
- **時間步**: 16-32 子集（偏後期步）
- **指標**:
  - ε-MSE: Teacher vs Student 的噪聲預測 MSE
  - Cosine 相似度: Teacher vs Student 的預測相似度
- **門檻**:
  - ε-MSE 上升 ≤ 5%
  - Cosine 下降 ≤ 5%
- **輸出**: `{exp_dir}/eval/proxy_history.jsonl`

#### 趨勢評估（門檻通過時執行）
- **2K 評估**: 2,000 張圖像
- **里程碑**: 5,000 張圖像  
- **最終**: ≥10,000 張圖像（理想 50,000）
- **指標**:
  - FID: Student vs Real（真實統計）
  - dFID: Student vs Teacher（Teacher 作參考）
- **輸出**: 
  - `{exp_dir}/eval/fid_2k.json`
  - `{exp_dir}/eval/dfid_2k.json`
  - 其他里程碑類推

#### 快取管理
- **真實統計**: `{exp_dir}/eval/real_stats/stats_ffhq128_50k.npz`
- **生成快取**: `{exp_dir}/eval/gen_cache/{tag}/`
- 支援特徵快取以加速重複計算

## 檔案結構對照

### 新增檔案
```
QATcode/
├── layer_by_layer_trainer.py    # Layer-by-Layer 管理器
├── proxy_validation.py          # 代理指標驗證
└── README_NEW_FEATURES.md       # 本說明文件
```

### 修改檔案
```
QATcode/quantize_diffae_step6_train.py  # 主訓練流程整合
metrics.py                              # 新增 dFID 實作
```

### 輸出結構
```
runs/{exp_dir}/
├── layer_checkpoints/           # Layer-by-Layer 檢查點
├── layer_states.json           # 層狀態記錄
├── eval/
│   ├── proxy_history.jsonl     # 代理指標歷史
│   ├── real_stats/             # 真實資料統計快取
│   ├── gen_cache/              # 生成特徵快取
│   ├── fid_2k.json            # FID 結果
│   └── dfid_2k.json           # dFID 結果
└── ...
```

## 等價性驗證

### Zero-Diff 保證
當所有新功能關閉時（預設狀態），程式行為與原版本完全一致：
- 相同 seed/步數下逐位對齊
- 允許極小浮點噪聲（< 1e-6）
- CLI/API/輸出路徑/日誌欄位不變

### 驗證命令
```bash
# 原始行為（參考）
python QATcode/quantize_diffae_step6_train.py --seed 42 --dry-run

# 新版本（應該完全一致）
python QATcode/quantize_diffae_step6_train.py --seed 42 --dry-run
```

## 故障排除

### 常見問題
1. **Layer-by-Layer 參數驗證失敗**
   - 檢查量化層數是否為 144
   - 確認層索引與模型結構一致

2. **代理驗證 FID 計算失敗**
   - 確認 pytorch_fid 套件已安裝
   - 檢查 Inception 模型載入

### 除錯模式
```bash
# 乾執行模式（不更新參數）
python QATcode/quantize_diffae_step6_train.py --dry-run --max-steps 5

# 啟用詳細日誌
export PYTHONPATH=.
python QATcode/quantize_diffae_step6_train.py --enable-proxy-validation 2>&1 | tee debug.log
```

## 效能考量

### 計算開銷
- **Layer-by-Layer**: 幾乎無額外開銷
- **代理驗證**: 中等開銷（每層約 2-5 分鐘）

### 記憶體使用
- 代理驗證會暫時佔用額外 GPU 記憶體用於 Inception 計算
- 快取檔案可能佔用較多磁碟空間

### 建議配置
- GPU: ≥ 8GB VRAM（代理驗證）
- 磁碟: ≥ 10GB 空間（快取檔案）
- 記憶體: ≥ 16GB RAM

## 授權與支援

本實作嚴格遵循原有程式碼的授權條款。如有技術問題，請檢查日誌文件或聯繫開發團隊。
