from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from ktcoralcls.data import DEFAULT_PATCHES, DEFAULT_SPLITS
from ktcoralcls.evaluate import predict_split, write_eval_artifacts
from ktcoralcls.genera import GENERA
from ktcoralcls.metrics import summarize_predictions, symmetric_pair_errors, top_confused_pairs
from ktcoralcls.model import EXPERIMENTS
from ktcoralcls.plots import plot_learning_curves, plot_lookalike_swaps, plot_metric_bars, plot_per_genus_f1, plot_top_confusions
from ktcoralcls.train import DEFAULT_RUNS
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'artifacts' / 'comparison'

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-dir', type=Path, default=DEFAULT_RUNS)
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--patches-dir', type=Path, default=DEFAULT_PATCHES)
    p.add_argument('--splits-dir', type=Path, default=DEFAULT_SPLITS)
    p.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundles = {}
    summary_rows = []
    histories = {}
    for exp in EXPERIMENTS:
        ckpt_path = args.runs_dir / exp / 'best.pt'
        if not ckpt_path.exists():
            print(f'Missing checkpoint for {exp}: {ckpt_path}')
            continue
        print(f'Evaluating {exp} ...')
        y_true, y_pred, ckpt = predict_split(ckpt_path, patches_dir=args.patches_dir, splits_dir=args.splits_dir, split=args.split, batch_size=args.batch_size, num_workers=args.num_workers, device=device)
        classes = ckpt.get('classes', GENERA)
        write_eval_artifacts(y_true, y_pred, classes, args.runs_dir / exp, split=args.split, model_name=exp)
        bundle = summarize_predictions(y_true, y_pred, classes)
        bundles[exp] = bundle
        summary_rows.append({'model': exp, 'accuracy': bundle.accuracy, 'macro_f1': bundle.macro_f1, 'weighted_f1': bundle.weighted_f1, 'min_class_f1': bundle.min_class_f1, 'min_class_recall': bundle.min_class_recall, 'mean_class_recall': bundle.mean_class_recall, 'top_confusion': top_confused_pairs(bundle, 1).iloc[0]['pair'] if not bundle.confused_pairs.empty else '', 'top_confusion_count': int(top_confused_pairs(bundle, 1).iloc[0]['count']) if not bundle.confused_pairs.empty else 0, 'worst_lookalike_pair': symmetric_pair_errors(bundle, 1).iloc[0]['pair'] if not symmetric_pair_errors(bundle, 1).empty else '', 'worst_lookalike_swaps': int(symmetric_pair_errors(bundle, 1).iloc[0]['total_swap']) if not symmetric_pair_errors(bundle, 1).empty else 0})
        hist_path = args.runs_dir / exp / 'history.json'
        if hist_path.exists():
            histories[exp] = pd.read_json(hist_path)
    if not summary_rows:
        raise SystemExit('No models evaluated. Train first: python -m ktcoralcls.train_suite')
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(['min_class_f1', 'worst_lookalike_swaps', 'macro_f1'], ascending=[False, True, False]).reset_index(drop=True)
    summary.to_csv(args.out_dir / 'summary.csv', index=False)
    (args.out_dir / 'summary.json').write_text(summary.to_json(orient='records', indent=2), encoding='utf-8')
    plot_metric_bars(summary, args.out_dir / 'metrics_bars.png')
    plot_per_genus_f1(bundles, args.out_dir / 'per_genus_f1.png')
    plot_top_confusions(bundles, args.out_dir / 'top_confusions.png')
    plot_lookalike_swaps(bundles, args.out_dir / 'lookalike_swaps.png')
    if histories:
        plot_learning_curves(histories, args.out_dir / 'learning_curves.png')
    winner = summary.iloc[0]['model']
    notes = {'selected_model': winner, 'ranking_rule': 'Primary: maximize min_class_f1 (worst genus). Tie-break: minimize worst_lookalike_swaps (A↔B confusions). Then maximize macro_f1. Overall accuracy is reported but not used to pick.', 'models_evaluated': list(bundles.keys())}
    (args.out_dir / 'recommendation.json').write_text(json.dumps(notes, indent=2), encoding='utf-8')
    print()
    print(summary.to_string(index=False))
    print()
    print(f'Selected model: {winner}')
    print(f'Figures + tables -> {args.out_dir}')
if __name__ == '__main__':
    main()
