from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ktcoralcls.data import DEFAULT_PATCHES, DEFAULT_SPLITS, PatchDataset, eval_transforms, load_splits
from ktcoralcls.genera import GENERA
from ktcoralcls.metrics import summarize_predictions, symmetric_pair_errors, top_confused_pairs
from ktcoralcls.model import load_checkpoint
from ktcoralcls.plots import plot_confusion_matrix
ROOT = Path(__file__).resolve().parents[2]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--patches-dir', type=Path, default=DEFAULT_PATCHES)
    p.add_argument('--splits-dir', type=Path, default=DEFAULT_SPLITS)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()

@torch.no_grad()
def predict_split(checkpoint: Path, *, patches_dir: Path=DEFAULT_PATCHES, splits_dir: Path=DEFAULT_SPLITS, split: str='test', batch_size: int=64, num_workers: int=4, device: torch.device | None=None) -> tuple[np.ndarray, np.ndarray, dict]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt = load_checkpoint(checkpoint, device, num_classes=len(GENERA))
    frame = load_splits(splits_dir)[split]
    ds = PatchDataset(frame, patches_dir, eval_transforms())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    ys: list[int] = []
    ps: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        ys.extend(y.tolist())
        ps.extend(pred.cpu().tolist())
    return (np.array(ys), np.array(ps), ckpt)

def write_eval_artifacts(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], out_dir: Path, *, split: str, model_name: str) -> None:
    bundle = summarize_predictions(y_true, y_pred, classes)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {'model': model_name, 'split': split, 'n': int(len(y_true)), 'accuracy': bundle.accuracy, 'macro_f1': bundle.macro_f1, 'weighted_f1': bundle.weighted_f1, 'min_class_f1': bundle.min_class_f1, 'mean_class_recall': bundle.mean_class_recall, 'min_class_recall': bundle.min_class_recall}
    (out_dir / f'metrics_{split}.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    bundle.per_class.to_csv(out_dir / f'per_class_{split}.csv', index=False)
    pd.DataFrame(bundle.confusion, index=classes, columns=classes).to_csv(out_dir / f'confusion_{split}.csv')
    pd.DataFrame(bundle.confusion_norm, index=classes, columns=classes).to_csv(out_dir / f'confusion_norm_{split}.csv')
    top_confused_pairs(bundle, k=50).to_csv(out_dir / f'confused_pairs_{split}.csv', index=False)
    symmetric_pair_errors(bundle, min_count=1).to_csv(out_dir / f'lookalike_swaps_{split}.csv', index=False)
    plot_confusion_matrix(bundle, out_dir / f'confusion_norm_{split}.png', title=f'{model_name} — {split} (row-normalized)', normalize=True)
    print(f'{model_name} [{split}] acc={bundle.accuracy:.4f} macro_f1={bundle.macro_f1:.4f} min_f1={bundle.min_class_f1:.4f} min_recall={bundle.min_class_recall:.4f}')
    print('Top confusions:')
    print(top_confused_pairs(bundle, k=10).to_string(index=False))

def main() -> None:
    args = parse_args()
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    y_true, y_pred, ckpt = predict_split(args.checkpoint, patches_dir=args.patches_dir, splits_dir=args.splits_dir, split=args.split, batch_size=args.batch_size, num_workers=args.num_workers, device=device)
    classes = ckpt.get('classes', GENERA)
    model_name = ckpt.get('experiment') or args.checkpoint.parent.name
    out_dir = args.out_dir or args.checkpoint.parent
    write_eval_artifacts(y_true, y_pred, classes, out_dir, split=args.split, model_name=model_name)
    print(f'Wrote artifacts to {out_dir}')
if __name__ == '__main__':
    main()
