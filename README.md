# KTCoralCLS

Code and evaluation artifacts for **KTCoral** / **KTCoralCLS**: a 36-genus coral classification benchmark focused on taxa monitored around Koh Tao, Thailand.

This repository contains training and evaluation code, fixed train/val/test splits, patch provenance metadata, and metric summaries from the paper experiments. Image patches and model checkpoints are distributed separately (see [DATA.md](DATA.md)).

## Requirements

- Python 3.11+
- CUDA-capable GPU recommended for training
- PyTorch with torchvision

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Layout

```
KTCoralCLS/
├── src/ktcoralcls/          # training, evaluation, comparison
├── scripts/                 # split construction and experiment runners
├── datasets/ktcoral/        # splits, provenance, summaries (no patch images)
├── artifacts/               # metric tables and figures from reported runs
├── DATA.md                  # dataset contents and external downloads
└── requirements.txt
```

## Experiments

Four ImageNet-initialised backbones are supported: ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16.

| Protocol | Splits | Reported role |
|----------|--------|----------------|
| Full stratified | `datasets/ktcoral/splits` | Architecture comparison |
| Balanced stratified (≤1000/genus) | `datasets/ktcoral/splits_balanced` | Class-balance ablation |
| Full parent-image group hold-out | `datasets/ktcoral/splits_sourceaware` | Primary genus metrics |
| Growth multi-task (group hold-out) | `datasets/ktcoral/splits_growth_sourceaware` | Secondary growth-form task |

Model selection for genus classification prioritises worst-genus F1, then lookalike swap count, then macro F1.

## Recommended model

The reported genus classifier is **genus-only ConvNeXt-Tiny** trained on the **parent-image group hold-out** split (not the multi-task growth model). If you are looking for the 'final'  genus-only model, this is the one to use.

| | |
|--|--|
| Checkpoint | `artifacts/runs_sourceaware/KTCoralsCLS-ConvNeXt/best.pt` |
| Splits | `datasets/ktcoral/splits_sourceaware/` |
| Test metrics | 81.9% accuracy, macro F1 0.796, worst-genus F1 0.625 |

Weights are distributed separately from this repository (see [DATA.md](DATA.md)). Metric files for that run are under `artifacts/runs_sourceaware/KTCoralsCLS-ConvNeXt/` and `artifacts/comparison_sourceaware/`.

## Training

Place patch images under `datasets/ktcoral/patches/` (and optionally `patches_balanced/`) as described in DATA.md.

```bash
# Train all four backbones (stratified full defaults unless overridden)
python -m ktcoralcls.train_suite \
  --epochs 15 \
  --patches-dir datasets/ktcoral/patches \
  --splits-dir datasets/ktcoral/splits \
  --runs-dir artifacts/runs_full

# Group hold-out bake-off
bash scripts/run_sourceaware_full_bakeoff.sh

# Multi-task growth (ConvNeXt)
bash scripts/run_growth_sourceaware.sh
```

Single-model training:

```bash
python -m ktcoralcls.train --experiment KTCoralsCLS-ConvNeXt \
  --patches-dir datasets/ktcoral/patches \
  --splits-dir datasets/ktcoral/splits_sourceaware \
  --runs-dir artifacts/runs_sourceaware
```

## Evaluation

```bash
python -m ktcoralcls.compare \
  --runs-dir artifacts/runs_sourceaware \
  --out-dir artifacts/comparison_sourceaware \
  --patches-dir datasets/ktcoral/patches \
  --splits-dir datasets/ktcoral/splits_sourceaware
```

Precomputed summaries for the paper runs are already under `artifacts/`.

## Provenance

`datasets/ktcoral/supplement/ktcoral_patch_provenance.csv` lists one row per retained patch (genus, platform, identifiers, licence fields where available, growth-form availability, split assignments). Screening rules are in `ktcoral_screening_rules.md`.

## Citation

If you use this dataset or code, please cite the KTCoralCLS paper (title: *KTCoralCLS: A Localised Benchmark for Coral Genus Classification on Koh Tao*).
