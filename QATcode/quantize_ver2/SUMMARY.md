# quantize_ver2：Step 6 QAT Baseline 訓練記錄

本文件記錄正式基準訓練（baseline）的完整設定，來源為訓練 log：`QATcode/quantize_ver2/log/step6_train_700.log`。

---

## 實驗設定

### Run 身分

| 項目 | 值 |
|------|-----|
| 任務 | Diff-AE EfficientDM Step 6（QAT） |
| log 檔 | `QATcode/quantize_ver2/log/step6_train_700.log` |
| TensorBoard | `QATcode/quantize_ver2/tb_runs/step6_train_700` |
| 裝置 | `cuda` |
| 總訓練時間 | `4472.67 秒` |
| 平均每 epoch | `27.95 秒` |

### 模型與資料路徑

| 項目 | 路徑 |
|------|------|
| Base Diff-AE checkpoint | `checkpoints/ffhq128_autoenc_latent/last.ckpt` |
| 量化校準資料 | `QATcode/quantize_ver2/calibration_diffae.pth` |
| 最佳 checkpoint（訓練中覆寫） | `QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth` |
| 最終 checkpoint | `QATcode/quantize_ver2/checkpoints/diffae_step6_lora_final.pth` |

### 訓練超參數

| 參數 | 值 |
|------|-----|
| DDIM steps | `100` |
| Epochs | `160` |
| Loss chunk size | `20` |
| LoRA factor | `700` |
| Weight quant LR | `1.00e-06` |
| Act quant LR | `5.00e-04` |
| Optimizer groups | LoRA=140、Act-quant(scale_list)=140 |
| 可訓練參數 | `9,803,440 / 96,366,131`（10.17%） |

### 功能旗標（此 baseline 版本）

| 旗標 | 狀態 |
|------|------|
| Teacher autocast match (A/B) | `True` |
| Scale-list debug | `False`（interval=100） |
| Timestep grad conflict debug | `False`（steps=0,80,99；interval=0 epoch，等同關閉） |
| Tail repair | `False`（outer_steps=1；t_range=0；lr_scale=0.25） |
| Quant update diag | 啟用（interval=8；topk=12；output root=`QATcode/quantize_ver2/quant_diag_runs`） |

---

## 訓練結果

| 指標 | 值 |
|------|-----|
| 最佳損失（best loss） | `0.000342` |
| 最後 epoch（160/160）平均蒸餾損失 | `0.000391` |

---

## 對外評估指標（正式 baseline）

| 設定 | FID@5k（四捨五入至小數兩位） |
|------|------------------------------|
| T=20 | `19.87` |
| T=100 | `14.95` |

> 數值來源：`experiment_results/fid_results.jsonl`（`sample_lora_intmodel_v2.py`，`mode=float`，`eval_samples=5000`）。原始值約 **19.869**（T=20）、**14.947**（T=100）；上表為標準四捨五入。非訓練 log 直接列出。

---

## 重現條件

要重現此 baseline，至少需固定以下條件：

1. 同一份程式碼版本（`quantize_diffae_step6_train_v2.py` + `diffae_trainer_v2.py`）
2. 同一條 base checkpoint（`checkpoints/ffhq128_autoenc_latent/last.ckpt`）
3. 同一組旗標狀態（特別是 `timestep grad conflict debug=False`、`tail repair=False`）
4. 同一組 LoRA / quant LR（`lora_factor=700`、`act_quant_lr=5e-4`）
5. 同一套評估流程與樣本數（FID `eval_samples=5000`）
