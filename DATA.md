# Data

This repository ships splits, provenance metadata, and evaluation summaries. It does **not** include the image patches or trained weight files.

## Included here

| Path | Description |
|------|-------------|
| `datasets/ktcoral/splits/` | Full stratified 80/10/10 split |
| `datasets/ktcoral/splits_balanced/` | Balanced (≤1000/genus) stratified split |
| `datasets/ktcoral/splits_sourceaware/` | Full parent-image group hold-out split |
| `datasets/ktcoral/splits_growth/` | Growth multi-task stratified split |
| `datasets/ktcoral/splits_growth_sourceaware/` | Growth multi-task group hold-out split |
| `datasets/ktcoral/supplement/ktcoral_patch_provenance.csv` | One row per retained patch (genus, platform, identifiers, licence fields, growth-form flag, split assignments) |
| `datasets/ktcoral/supplement/ktcoral_screening_rules.md` | Inclusion / exclusion rules for web-sourced fill |
| `datasets/ktcoral/summary.csv` | Full dataset class counts |
| `datasets/ktcoral/balanced_summary.csv` | Balanced dataset class counts |
| `datasets/ktcoral/growth_label_summary.csv` | Growth-form label counts |
| `datasets/ktcoral/patch_growth_labels.csv` | Per-patch growth-form labels |
| `artifacts/` | Metric tables and figures from the reported runs |

Split CSV paths are relative to the patch root (e.g. `Acropora/<filename>.jpg`).

## Patches (external)

Training and evaluation need the 224×224 patch folders:

```
datasets/ktcoral/patches/<Genus>/...
datasets/ktcoral/patches_balanced/<Genus>/...   # optional; ≤1000/genus variant
```

| Variant | Images | Approx. size |
|---------|-------:|-------------:|
| Full cleaned | 110,833 | ~2.4 GB |
| Balanced | 35,101 | ~0.9 GB |

Download from Hugging Face [`quacky/ktcoral`](https://huggingface.co/datasets/quacky/ktcoral) (`patches/` in that dataset) and extract into those paths before training.

If `patches_balanced/` exists, some entrypoints prefer it by default. For full-dataset runs, pass `--patches-dir datasets/ktcoral/patches` (as in `scripts/`).

## Checkpoints (external)

Trained `best.pt` files are distributed with the model release, using the same run folder names as under `artifacts/`.

The recommended deployment checkpoint is the **genus-only** ConvNeXt group-hold-out model:

[`quacky/ktcoralcls-convnext`](https://huggingface.co/quacky/ktcoralcls-convnext) → download `best.pt` to  
`artifacts/runs_sourceaware/KTCoralsCLS-ConvNeXt/best.pt`

Place downloaded weights beside the metric files under `artifacts/` if you want to evaluate without retraining. The multi-task growth checkpoint is experimental and is not the primary model.

## Licences

ReefNet-derived patches follow ReefNet / CoralNet source terms. Supplementary images were restricted to openly licensed material; per-image licence fields are in the provenance CSV where available.
