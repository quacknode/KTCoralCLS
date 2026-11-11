from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ktcoralcls.genera import GENERA
from ktcoralcls.growth import GROWTH_CLASSES, IGNORE_INDEX
from ktcoralcls.metrics import summarize_predictions, top_confused_pairs
from ktcoralcls.model import create_multitask_model
from ktcoralcls.plots import plot_confusion_matrix, plot_learning_curves, plot_lookalike_swaps, plot_per_genus_f1, plot_top_confusions, _save
from ktcoralcls.train_multitask import MultiTaskPatchDataset, DEFAULT_PATCHES, DEFAULT_SPLITS
from ktcoralcls.data import eval_transforms
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'artifacts' / 'comparison_growth'

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=ROOT / 'artifacts' / 'runs_growth' / 'KTCoralsCLS-ConvNeXt-MultiTask')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--patches-dir', type=Path, default=DEFAULT_PATCHES)
    p.add_argument('--splits-dir', type=Path, default=DEFAULT_SPLITS)
    p.add_argument('--batch-size', type=int, default=48)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    return p.parse_args()

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    g_true, g_pred = ([], [])
    w_true, w_pred = ([], [])
    for x, y_g, y_w in loader:
        x = x.to(device, non_blocking=True)
        lg, lw = model(x)
        g_true.extend(y_g.tolist())
        g_pred.extend(lg.argmax(1).cpu().tolist())
        mask = y_w != IGNORE_INDEX
        if mask.any():
            w_true.extend(y_w[mask].tolist())
            w_pred.extend(lw[mask].argmax(1).cpu().tolist())
    return (np.array(g_true), np.array(g_pred), np.array(w_true) if w_true else np.array([], dtype=int), np.array(w_pred) if w_pred else np.array([], dtype=int))

def plot_growth_bars(per_class: pd.DataFrame, out: Path) -> None:
    df = per_class.sort_values('f1')
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(df))))
    ax.barh(df['genus'], df['f1'], color='#2c6e8a', edgecolor='black', linewidth=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('F1')
    ax.set_title('Growth-form F1 (labeled test subset)')
    for y, (f1, support) in enumerate(zip(df['f1'], df['support'])):
        ax.text(min(f1 + 0.02, 0.98), y, f'{f1:.2f} (n={support})', va='center', fontsize=8)
    _save(fig, out)

def plot_dual_learning_curves(hist: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hist['epoch'], hist['val_genus_acc'], label='Genus acc')
    axes[0].plot(hist['epoch'], hist['val_genus_macro_f1'], label='Genus macro F1')
    axes[0].set_title('Genus (validation)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[1].plot(hist['epoch'], hist['val_growth_acc'], label='Growth acc')
    axes[1].plot(hist['epoch'], hist['val_growth_macro_f1'], label='Growth macro F1')
    axes[1].set_title('Growth form (labeled val subset)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    fig.suptitle('KTCoralsCLS-ConvNeXt multi-task learning curves')
    _save(fig, out)

def plot_metric_summary(genus_m: dict, growth_m: dict, out: Path) -> None:
    labels = ['Genus\naccuracy', 'Genus\nmacro F1', 'Genus\nworst F1', 'Growth\naccuracy', 'Growth\nmacro F1', 'Growth\nworst F1']
    vals = [genus_m['accuracy'], genus_m['macro_f1'], genus_m['min_class_f1'], growth_m.get('accuracy', 0), growth_m.get('macro_f1', 0), growth_m.get('min_class_f1', 0)]
    colors = ['#2c6e8a'] * 3 + ['#c47b2b'] * 3
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('ConvNeXt multi-task test metrics')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f'{v:.2f}', ha='center', fontsize=9)
    _save(fig, out)

def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.run_dir / 'best.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    genus_classes = ckpt.get('genus_classes', GENERA)
    growth_classes = ckpt.get('growth_classes', GROWTH_CLASSES)
    model = create_multitask_model(len(genus_classes), len(growth_classes), backbone=ckpt['backbone'], pretrained=False)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    frame = pd.read_csv(args.splits_dir / f'{args.split}.csv')
    ds = MultiTaskPatchDataset(frame, args.patches_dir, eval_transforms())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    y_g, p_g, y_w, p_w = predict(model, loader, device)
    genus_bundle = summarize_predictions(y_g, p_g, genus_classes)
    name = 'KTCoralsCLS-ConvNeXt-MultiTask'
    genus_metrics = {'model': name, 'task': 'genus', 'split': args.split, 'n': int(len(y_g)), 'accuracy': genus_bundle.accuracy, 'macro_f1': genus_bundle.macro_f1, 'weighted_f1': genus_bundle.weighted_f1, 'min_class_f1': genus_bundle.min_class_f1, 'min_class_recall': genus_bundle.min_class_recall, 'mean_class_recall': genus_bundle.mean_class_recall}
    growth_metrics = {'model': name, 'task': 'growth', 'split': args.split, 'n': int(len(y_w))}
    growth_bundle = None
    if len(y_w):
        growth_bundle = summarize_predictions(y_w, p_w, growth_classes)
        growth_pc = growth_bundle.per_class.rename(columns={'genus': 'growth'})
        growth_metrics.update({'accuracy': growth_bundle.accuracy, 'macro_f1': growth_bundle.macro_f1, 'weighted_f1': growth_bundle.weighted_f1, 'min_class_f1': growth_bundle.min_class_f1, 'min_class_recall': growth_bundle.min_class_recall, 'mean_class_recall': growth_bundle.mean_class_recall})
        growth_pc.to_csv(args.out_dir / f'per_growth_{args.split}.csv', index=False)
        pd.DataFrame(growth_bundle.confusion, index=growth_classes, columns=growth_classes).to_csv(args.out_dir / f'confusion_growth_{args.split}.csv')
        plot_confusion_matrix(growth_bundle, args.out_dir / 'confusion_growth_norm.png', title='Growth-form confusion (row-normalized)', normalize=True)
        growth_bundle.per_class = growth_bundle.per_class.copy()
        plot_growth_bars(growth_pc.rename(columns={'growth': 'genus'}), args.out_dir / 'per_growth_f1.png')
        top_confused_pairs(growth_bundle, k=20).to_csv(args.out_dir / f'confused_pairs_growth_{args.split}.csv', index=False)
    genus_bundle.per_class.to_csv(args.out_dir / f'per_genus_{args.split}.csv', index=False)
    pd.DataFrame(genus_bundle.confusion, index=genus_classes, columns=genus_classes).to_csv(args.out_dir / f'confusion_genus_{args.split}.csv')
    plot_confusion_matrix(genus_bundle, args.out_dir / 'confusion_genus_norm.png', title='Genus confusion (row-normalized)', normalize=True)
    plot_per_genus_f1({name: genus_bundle}, args.out_dir / 'per_genus_f1.png')
    plot_top_confusions({name: genus_bundle}, args.out_dir / 'top_genus_confusions.png')
    plot_lookalike_swaps({name: genus_bundle}, args.out_dir / 'lookalike_swaps.png')
    hist_path = args.run_dir / 'history.json'
    if hist_path.exists():
        hist = pd.read_json(hist_path)
        plot_dual_learning_curves(hist, args.out_dir / 'learning_curves.png')
        hist2 = hist.rename(columns={'val_genus_acc': 'val_accuracy', 'val_genus_macro_f1': 'val_macro_f1'})
        plot_learning_curves({name: hist2}, args.out_dir / 'learning_curves_genus_only.png')
    plot_metric_summary(genus_metrics, growth_metrics, args.out_dir / 'metrics_bars.png')
    summary = pd.DataFrame([genus_metrics, growth_metrics])
    summary.to_csv(args.out_dir / 'summary.csv', index=False)
    (args.out_dir / 'summary.json').write_text(summary.to_json(orient='records', indent=2), encoding='utf-8')
    notes = {'model': name, 'backbone': ckpt.get('backbone'), 'checkpoint': str(ckpt_path), 'genus_test': genus_metrics, 'growth_test': growth_metrics, 'growth_labeling': 'Growth loss/metrics only on ReefNet patches with Growthform or Branching_growthform_specification; iNat/GBIF/OpenVerse fills train genus only.', 'figures': sorted((p.name for p in args.out_dir.glob('*.png')))}
    (args.out_dir / 'run_notes.json').write_text(json.dumps(notes, indent=2), encoding='utf-8')
    for f in args.out_dir.iterdir():
        if f.suffix in {'.png', '.csv', '.json'}:
            target = args.run_dir / f.name
            if not target.exists() or f.stat().st_mtime >= target.stat().st_mtime:
                target.write_bytes(f.read_bytes())
    print(summary.to_string(index=False))
    print(f'\nWrote outputs -> {args.out_dir}')
if __name__ == '__main__':
    main()
