#!/usr/bin/env bash
set -euo pipefail

cd /home/jimmy/diffae

echo "=== Row 4: FP32 weights + calibration-only act scales (verify) ==="
python QATcode/quantize_ver2/sample_lora_intmodel_v2.py \
  --baseline-aq \
  --num_steps 100 \
  --eval_samples 5000 \
  --log_file QATcode/quantize_ver2/log/ablation_row4_baseline_aq_T100_5k.log

echo "=== Row 3: FP32 weights + QAT TT act scales (new) ==="
python QATcode/quantize_ver2/sample_lora_intmodel_v2.py \
  --baseline-qat-act \
  --num_steps 100 \
  --eval_samples 5000 \
  --log_file QATcode/quantize_ver2/log/ablation_row3_baseline_qat_act_T100_5k.log

echo "Done."
echo "Row 4 images: mycache/gen_images/ffhq128_autoenc_latent_BASELINE_AQ_T100/"
echo "Row 3 images: mycache/gen_images/ffhq128_autoenc_latent_BASELINE_QAT_ACT_T100/"
