# Activation Distribution Analysis

## Analysis goal
- Per-layer input activation statistics during DDIM sampling; pairwise ratio/delta under `comparisons/<a>_vs_<b>/`.
- `BASELINE`: original Diff-AE `ema` (no QAT/LoRA ckpt). FF/FT/TT: quantized graph with shared `w+lora` from `--lora-ckpt`.

## Layer selection
- selected layers: 142
- Conv2d layers: 72
- Linear layers: 70

## Output layout
- run root: `QATcode/quantize_ver2/baseline_quant_analysis/activation_results/T_100/official/seed0`
- per-mode: `QATcode/quantize_ver2/baseline_quant_analysis/activation_results/T_100/official/seed0/models`
- pairwise json: `QATcode/quantize_ver2/baseline_quant_analysis/activation_results/T_100/official/seed0/comparisons`
- curves / block bars (prefixed): `QATcode/quantize_ver2/baseline_quant_analysis/activation_results/T_100/official/seed0/plots/pairwise`

## Pair: `ff_vs_ft`
- modes: ['FF', 'FT']
- T / seed / num_samples: 100 / 0 / 32
- dominant effect label (layer vote): `minimal_change`

- top changed layers (by q999 ratio deviation):
  - `model.output_blocks.2.1.out_layers.3` | block=model.output_blocks.2 | q999_ratio=1.3424 | std_ratio=1.2495
  - `model.output_blocks.2.0.out_layers.3` | block=model.output_blocks.2 | q999_ratio=1.2893 | std_ratio=1.1795
  - `model.output_blocks.12.0.out_layers.3` | block=model.output_blocks.12 | q999_ratio=1.2503 | std_ratio=1.2900
  - `model.output_blocks.3.0.out_layers.3` | block=model.output_blocks.3 | q999_ratio=1.2238 | std_ratio=1.1323
  - `model.output_blocks.4.0.out_layers.3` | block=model.output_blocks.4 | q999_ratio=1.1341 | std_ratio=1.0782

- top changed blocks:
  - `model.output_blocks.2` | q999_ratio=1.0885 | std_ratio=1.0533
  - `model.output_blocks.3` | q999_ratio=1.0753 | std_ratio=1.0393
  - `model.output_blocks.12` | q999_ratio=1.0552 | std_ratio=1.0719
  - `model.output_blocks.4` | q999_ratio=1.0394 | std_ratio=1.0213
  - `model.output_blocks.7` | q999_ratio=1.0295 | std_ratio=1.0122

- artifacts: `QATcode/quantize_ver2/baseline_quant_analysis/activation_results/T_100/official/seed0/comparisons/ff_vs_ft`
- plot filename prefix: `ff_vs_ft_` under `plots/pairwise/`

## Pair: `baseline_vs_ft`
- modes: ['BASELINE', 'FT']
- T / seed / num_samples: 100 / 0 / 32
- dominant effect label (layer vote): `minimal_change`

- top changed layers (by q999 ratio deviation):
  - `model.output_blocks.12.0.out_layers.3` | block=model.output_blocks.12 | q999_ratio=1.2441 | std_ratio=1.3049
  - `model.output_blocks.13.0.out_layers.3` | block=model.output_blocks.13 | q999_ratio=1.1602 | std_ratio=1.0955
  - `model.input_blocks.5.0.out_layers.3` | block=model.input_blocks.5 | q999_ratio=1.1494 | std_ratio=1.0448
  - `model.input_blocks.7.0.out_layers.3` | block=model.input_blocks.7 | q999_ratio=1.1019 | std_ratio=1.0277
  - `model.input_blocks.8.0.out_layers.3` | block=model.input_blocks.8 | q999_ratio=1.0900 | std_ratio=1.0555

- top changed blocks:
  - `model.output_blocks.12` | q999_ratio=1.0549 | std_ratio=1.0751
  - `model.output_blocks.13` | q999_ratio=1.0494 | std_ratio=1.0241
  - `model.input_blocks.5` | q999_ratio=1.0358 | std_ratio=1.0120
  - `model.input_blocks.11` | q999_ratio=1.0327 | std_ratio=1.0066
  - `model.input_blocks.7` | q999_ratio=1.0274 | std_ratio=1.0078

- artifacts: `QATcode/quantize_ver2/baseline_quant_analysis/activation_results/T_100/official/seed0/comparisons/baseline_vs_ft`
- plot filename prefix: `baseline_vs_ft_` under `plots/pairwise/`

## Pair: `baseline_vs_tt`
- modes: ['BASELINE', 'TT']
- T / seed / num_samples: 100 / 0 / 32
- dominant effect label (layer vote): `minimal_change`

- top changed layers (by q999 ratio deviation):
  - `model.output_blocks.12.0.out_layers.3` | block=model.output_blocks.12 | q999_ratio=1.3880 | std_ratio=1.3089
  - `model.output_blocks.13.0.out_layers.3` | block=model.output_blocks.13 | q999_ratio=1.1638 | std_ratio=1.0864
  - `model.input_blocks.5.0.out_layers.3` | block=model.input_blocks.5 | q999_ratio=1.1567 | std_ratio=1.0428
  - `model.input_blocks.8.0.out_layers.3` | block=model.input_blocks.8 | q999_ratio=1.1491 | std_ratio=1.0567
  - `model.input_blocks.9.0.out_layers.3` | block=model.input_blocks.9 | q999_ratio=1.0855 | std_ratio=1.0388

- top changed blocks:
  - `model.output_blocks.12` | q999_ratio=1.0909 | std_ratio=1.0760
  - `model.output_blocks.13` | q999_ratio=1.0521 | std_ratio=1.0218
  - `model.input_blocks.8` | q999_ratio=1.0406 | std_ratio=1.0163
  - `model.input_blocks.5` | q999_ratio=1.0323 | std_ratio=1.0115
  - `model.input_blocks.10` | q999_ratio=1.0295 | std_ratio=1.0039

- artifacts: `QATcode/quantize_ver2/baseline_quant_analysis/activation_results/T_100/official/seed0/comparisons/baseline_vs_tt`
- plot filename prefix: `baseline_vs_tt_` under `plots/pairwise/`

