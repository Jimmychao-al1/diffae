# Q-DiffAE Resweep Summary

## Pre-Registration Protocol

Timestamp committed: 2026-07-04T08:19:35+00:00

### Stage 1 (zone segmentation)

- Grid: K=[8, 12, 16, 20, 25], sw=[2, 3, 5], lambda=[0.25, 0.5, 1.0, 2.0], k_max=[4] = 60 configs
- Iteration order: K (outer) -> sw -> lambda -> k_max (inner), deterministic

### Stage 2 (refinement)

- Three-step pipeline (fixed, not swept):
  - Step 1: stage2_runtime_refine.py (global refine)
  - Step 2: build_blockwise_thresholds.py
  - Step 3: stage2_runtime_refine.py (2nd refine with threshold-config)
- Threshold parameters (Q-DiffAE mainline `baseline_908030` variant):
  - q_zone = 0.90
  - q_peak = 0.80
  - peak_over_zone_ratio_min = 1.3
- Rationale: Fixed threshold matches published Q-DiffAE mainline setting to ensure resweep comparability with existing thesis Q-DiffAE table (FID@50K = 10.311).

### FID evaluation

- Stage B: FID@5K on 60 configs, seed=0, mode=float, num_steps=100, quant-state=tt
- Stage D: FID@50K on top-3 configs (same sampling config)
- FID input: Step 3 refined scheduler (`stage2_refined_scheduler_config.json`)

### Selection rule

- Rank 60 configs by FID@5K ascending
- Select top-3 for FID@50K evaluation
- If sweep incomplete (< 60 success), require --allow-incomplete-selection

### Environment / fixed inputs

- Checkpoints:
  - QAT ckpt: /home/jimmy/diffae/QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth
  - Base Diff-AE ckpt: /home/jimmy/diffae/checkpoints/ffhq128_autoenc_latent/last.ckpt
  - Calibration: /home/jimmy/diffae/QATcode/quantize_ver2/calibration_diffae.pth
- Environment: /home/jimmy/anaconda3/envs/diffae_bw/bin/python, CUDA: cuda_available=True
device_count=1
- Reference stats: /home/jimmy/diffae/mycache/eval_images/ffhqlmdb256_size128_5000_5000

## Sweep Status

- Stage A (Stage 1 scheduler synthesis): 60/60
- Stage A2 (Stage 2 refinement): 60/60
- Stage B (FID@5K): 0/60
- Stage C (top-3 selection): missing
- Stage D (FID@50K on top-3): 0/3
- Sweep complete through Stage B: No

## Stage 1 Sweep Results (60 configs, ranked by FID@5K)

| Rank | Config | FID@5K | Status |
|------|--------|--------|--------|

## Top-3 FID@50K

Top-3 not available yet.

## Failed / Missing Configs

| Config ID | Failed at Stage | Failed Step | Failure Reason |
|-----------|-----------------|-------------|----------------|
| K8_sw2_lam0.25_kmax4 | B |  | missing FID@5K run |
| K8_sw2_lam0.5_kmax4 | B |  | missing FID@5K run |
| K8_sw2_lam1.0_kmax4 | B |  | missing FID@5K run |
| K8_sw2_lam2.0_kmax4 | B |  | missing FID@5K run |
| K8_sw3_lam0.25_kmax4 | B |  | missing FID@5K run |
| K8_sw3_lam0.5_kmax4 | B |  | missing FID@5K run |
| K8_sw3_lam1.0_kmax4 | B |  | missing FID@5K run |
| K8_sw3_lam2.0_kmax4 | B |  | missing FID@5K run |
| K8_sw5_lam0.25_kmax4 | B |  | missing FID@5K run |
| K8_sw5_lam0.5_kmax4 | B |  | missing FID@5K run |
| K8_sw5_lam1.0_kmax4 | B |  | missing FID@5K run |
| K8_sw5_lam2.0_kmax4 | B |  | missing FID@5K run |
| K12_sw2_lam0.25_kmax4 | B |  | missing FID@5K run |
| K12_sw2_lam0.5_kmax4 | B |  | missing FID@5K run |
| K12_sw2_lam1.0_kmax4 | B |  | missing FID@5K run |
| K12_sw2_lam2.0_kmax4 | B |  | missing FID@5K run |
| K12_sw3_lam0.25_kmax4 | B |  | missing FID@5K run |
| K12_sw3_lam0.5_kmax4 | B |  | missing FID@5K run |
| K12_sw3_lam1.0_kmax4 | B |  | missing FID@5K run |
| K12_sw3_lam2.0_kmax4 | B |  | missing FID@5K run |
| K12_sw5_lam0.25_kmax4 | B |  | missing FID@5K run |
| K12_sw5_lam0.5_kmax4 | B |  | missing FID@5K run |
| K12_sw5_lam1.0_kmax4 | B |  | missing FID@5K run |
| K12_sw5_lam2.0_kmax4 | B |  | missing FID@5K run |
| K16_sw2_lam0.25_kmax4 | B |  | missing FID@5K run |
| K16_sw2_lam0.5_kmax4 | B |  | missing FID@5K run |
| K16_sw2_lam1.0_kmax4 | B |  | missing FID@5K run |
| K16_sw2_lam2.0_kmax4 | B |  | missing FID@5K run |
| K16_sw3_lam0.25_kmax4 | B |  | missing FID@5K run |
| K16_sw3_lam0.5_kmax4 | B |  | missing FID@5K run |
| K16_sw3_lam1.0_kmax4 | B |  | missing FID@5K run |
| K16_sw3_lam2.0_kmax4 | B |  | missing FID@5K run |
| K16_sw5_lam0.25_kmax4 | B |  | missing FID@5K run |
| K16_sw5_lam0.5_kmax4 | B |  | missing FID@5K run |
| K16_sw5_lam1.0_kmax4 | B |  | missing FID@5K run |
| K16_sw5_lam2.0_kmax4 | B |  | missing FID@5K run |
| K20_sw2_lam0.25_kmax4 | B |  | missing FID@5K run |
| K20_sw2_lam0.5_kmax4 | B |  | missing FID@5K run |
| K20_sw2_lam1.0_kmax4 | B |  | missing FID@5K run |
| K20_sw2_lam2.0_kmax4 | B |  | missing FID@5K run |
| K20_sw3_lam0.25_kmax4 | B |  | missing FID@5K run |
| K20_sw3_lam0.5_kmax4 | B |  | missing FID@5K run |
| K20_sw3_lam1.0_kmax4 | B |  | missing FID@5K run |
| K20_sw3_lam2.0_kmax4 | B |  | missing FID@5K run |
| K20_sw5_lam0.25_kmax4 | B |  | missing FID@5K run |
| K20_sw5_lam0.5_kmax4 | B |  | missing FID@5K run |
| K20_sw5_lam1.0_kmax4 | B |  | missing FID@5K run |
| K20_sw5_lam2.0_kmax4 | B |  | missing FID@5K run |
| K25_sw2_lam0.25_kmax4 | B |  | missing FID@5K run |
| K25_sw2_lam0.5_kmax4 | B |  | missing FID@5K run |
| K25_sw2_lam1.0_kmax4 | B |  | missing FID@5K run |
| K25_sw2_lam2.0_kmax4 | B |  | missing FID@5K run |
| K25_sw3_lam0.25_kmax4 | B |  | missing FID@5K run |
| K25_sw3_lam0.5_kmax4 | B |  | missing FID@5K run |
| K25_sw3_lam1.0_kmax4 | B |  | missing FID@5K run |
| K25_sw3_lam2.0_kmax4 | B |  | missing FID@5K run |
| K25_sw5_lam0.25_kmax4 | B |  | missing FID@5K run |
| K25_sw5_lam0.5_kmax4 | B |  | missing FID@5K run |
| K25_sw5_lam1.0_kmax4 | B |  | missing FID@5K run |
| K25_sw5_lam2.0_kmax4 | B |  | missing FID@5K run |
