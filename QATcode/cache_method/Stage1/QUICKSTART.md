# Stage-1 快速開始（baseline v1）

**時間軸**：對外輸出以 **DDIM 99→0** 為準；`expanded_mask[b,i]` 的 **i=0 為 t=99**。interval ↔ reused timestep 對應見 `README.md` 與 `scheduler_diagnostics.json` 的 `mapping_note`。

## 一鍵執行

```bash
cd /home/jimmy/diffae

python3 QATcode/cache_method/Stage1/stage1_scheduler.py \
  --stage0_dir QATcode/cache_method/Stage0/stage0e_output \
  --output_dir QATcode/cache_method/Stage1/stage1_output

python3 QATcode/cache_method/Stage1/verify_scheduler.py \
  --config QATcode/cache_method/Stage1/stage1_output/scheduler_config.json

python3 QATcode/cache_method/Stage1/visualize_stage1.py \
  --stage1_output_dir QATcode/cache_method/Stage1/stage1_output \
  --output_dir QATcode/cache_method/Stage1/stage1_figures
```

## 輸出位置

```
stage1_output/
├── scheduler_config.json
├── scheduler_diagnostics.json
└── verification_summary.json

stage1_figures/
├── 1_global_cutting_and_zones.png
├── 2_k_zone_heatmap.png
├── 3_expanded_mask_heatmap.png
└── 4_candidate_selected_k.png
```

## 常用參數

- `--K`：top-K change points（內部會與 T 一齊 cap）
- `--smooth_window`：對 `G`（步序）的 moving average 視窗
- `--lambda`：`J(b,z,k)` 中 compute penalty 係數（預設 1.0）
- `--k_min` / `--k_max`：候選 k 範圍（pattern 去重後評估）

Sweep 範例：`bash QATcode/cache_method/Stage1/run_stage1_sweep.sh`（編輯腳本內 `K_LIST` 等陣列）。

## Self-test

```bash
python3 QATcode/cache_method/Stage1/stage1_scheduler.py --self_test
```
