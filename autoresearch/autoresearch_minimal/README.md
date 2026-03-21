# Where to place this

Recommended location in your repo:

`/home/jimmy/diffae/autoresearch/`

Not under `QATcode/quantize_ver2/`.

Reason:
- `quantize_ver2/` should remain model/training code
- `autoresearch/` should be an experiment-management layer at repo root
- this makes it easier to compare branches and keep search logic separate from training logic

## Suggested layout

/home/jimmy/diffae/
- autoresearch/
  - program.md
  - results.tsv
  - search_space.yaml
  - run_one_exp.sh
- QATcode/
  - quantize_ver2/

## Install

From repo root:

```bash
mkdir -p autoresearch
cp /path/to/these/files/* autoresearch/
```
