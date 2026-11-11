from __future__ import annotations
import argparse
from pathlib import Path
import torch
from ktcoralcls.model import EXPERIMENTS
from ktcoralcls.train import DEFAULT_RUNS, train_run
from ktcoralcls.data import DEFAULT_PATCHES, DEFAULT_SPLITS

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--runs-dir', type=Path, default=DEFAULT_RUNS)
    p.add_argument('--patches-dir', type=Path, default=DEFAULT_PATCHES)
    p.add_argument('--splits-dir', type=Path, default=DEFAULT_SPLITS)
    p.add_argument('--only', action='append', choices=sorted(EXPERIMENTS.keys()), help='Train a subset (repeatable). Default: all four.')
    p.add_argument('--skip-existing', action='store_true', help='Skip if best.pt already exists')
    p.add_argument('--force-splits', action='store_true', help='Rebuild train/val/test CSVs')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiments = args.only or list(EXPERIMENTS.keys())
    print(f'Training {len(experiments)} experiments on {device}')
    print(f'Patches: {args.patches_dir}')
    print(f'Splits:  {args.splits_dir}')
    for exp in experiments:
        run_dir = args.runs_dir / exp
        if args.skip_existing and (run_dir / 'best.pt').exists():
            print(f'Skipping {exp} (best.pt exists)')
            continue
        train_run(backbone=EXPERIMENTS[exp], run_name=exp, patches_dir=args.patches_dir, splits_dir=args.splits_dir, runs_dir=args.runs_dir, epochs=args.epochs, num_workers=args.num_workers, seed=args.seed, force_splits=args.force_splits, device=device)
    print('All requested experiments finished.')
    print('Next: pipenv run python -m ktcoralcls.compare')
if __name__ == '__main__':
    main()
