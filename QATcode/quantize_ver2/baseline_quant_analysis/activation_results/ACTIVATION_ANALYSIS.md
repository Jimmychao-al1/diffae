# Activation Distribution Analysis

## Analysis goal
- Compare `FF` vs `FT` under identical `w + lora` to study activation quant effect on layer-input distributions.

## Experiment setup
- T: 100
- seed: 0
- num_samples: 32
- compared modes: ['FF', 'FT']
- effective weight: w_plus_lora

## Layer-level summary
- selected layers: 142
- Conv2d layers: 72
- Linear layers: 70
- top changed layers (by q999 ratio deviation):
  - `model.output_blocks.12.0.out_layers.3` | block=model.output_blocks.12 | q999_ratio=1.3222 | std_ratio=1.2900
  - `model.output_blocks.2.1.out_layers.3` | block=model.output_blocks.2 | q999_ratio=1.3138 | std_ratio=1.2495
  - `model.output_blocks.2.0.out_layers.3` | block=model.output_blocks.2 | q999_ratio=1.2622 | std_ratio=1.1795
  - `model.output_blocks.3.0.out_layers.3` | block=model.output_blocks.3 | q999_ratio=1.2048 | std_ratio=1.1323
  - `model.output_blocks.5.2.out_layers.3` | block=model.output_blocks.5 | q999_ratio=1.1227 | std_ratio=1.0742

## Block-level summary
- top changed blocks (by q999 ratio deviation):
  - `model.output_blocks.12` | q999_ratio=1.0929 | std_ratio=1.0719
  - `model.output_blocks.2` | q999_ratio=1.0808 | std_ratio=1.0533
  - `model.output_blocks.3` | q999_ratio=1.0553 | std_ratio=1.0393
  - `model.output_blocks.5` | q999_ratio=1.0384 | std_ratio=1.0181
  - `model.output_blocks.9` | q999_ratio=1.0355 | std_ratio=1.0107

## Initial findings
- dominant effect label across layers: `minimal_change`
- differences are typically concentrated in a subset of layers/blocks ranked above.
- use representative plots + block bars to inspect whether changes align with tail compression or variance reduction patterns.

## Output artifact paths
- compare summary: `QATcode/quantize_ver2/baseline_quant_analysis/activation_results/T_100/official/seed0/FF_vs_FT/activation_compare_summary.json`
- block summary: `QATcode/quantize_ver2/baseline_quant_analysis/activation_results/T_100/official/seed0/FF_vs_FT/block_compare_summary.json`
- rankings: `QATcode/quantize_ver2/baseline_quant_analysis/activation_results/T_100/official/seed0/FF_vs_FT/rankings.json`
- representative layers: `QATcode/quantize_ver2/baseline_quant_analysis/activation_results/T_100/official/seed0/FF_vs_FT/representative_layers.json`
- representative plots: `QATcode/quantize_ver2/baseline_quant_analysis/activation_results/T_100/official/seed0/FF_vs_FT/representative_plots`
- block plots: `QATcode/quantize_ver2/baseline_quant_analysis/activation_results/T_100/official/seed0/FF_vs_FT/block_plots`
