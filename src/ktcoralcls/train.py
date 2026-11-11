from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ktcoralcls.data import DEFAULT_PATCHES, DEFAULT_SPLITS, PatchDataset, build_splits, class_weights_from_frame, eval_transforms, train_transforms
from ktcoralcls.genera import GENERA
from ktcoralcls.model import EXPERIMENTS, create_model, experiment_name
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = ROOT / 'artifacts' / 'runs'
DEFAULT_BATCH = {'resnet50': 48, 'efficientnet_b0': 64, 'convnext_tiny': 48, 'vit_b_16': 32}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patches-dir', type=Path, default=DEFAULT_PATCHES)
    p.add_argument('--splits-dir', type=Path, default=DEFAULT_SPLITS)
    p.add_argument('--runs-dir', type=Path, default=DEFAULT_RUNS)
    p.add_argument('--run-name', type=str, default=None, help='Defaults to KTCoralsCLS-<Arch>')
    p.add_argument('--backbone', type=str, default='efficientnet_b0', choices=sorted(set(EXPERIMENTS.values())))
    p.add_argument('--experiment', type=str, default=None, choices=sorted(EXPERIMENTS.keys()))
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force-splits', action='store_true')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_y: list[int] = []
    all_p: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        pred = logits.argmax(dim=1)
        all_y.extend(y.cpu().tolist())
        all_p.extend(pred.cpu().tolist())
    y_true = np.array(all_y)
    y_pred = np.array(all_p)
    return {'loss': total_loss / max(len(y_true), 1), 'accuracy': float(accuracy_score(y_true, y_pred)), 'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'y_true': y_true, 'y_pred': y_pred}

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, scaler: torch.amp.GradScaler | None) -> float:
    model.train()
    running = 0.0
    n = 0
    use_amp = scaler is not None
    for x, y in tqdm(loader, desc='train', leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if use_amp:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item() * len(y)
        n += len(y)
    return running / max(n, 1)

def train_run(*, backbone: str, run_name: str | None=None, patches_dir: Path=DEFAULT_PATCHES, splits_dir: Path=DEFAULT_SPLITS, runs_dir: Path=DEFAULT_RUNS, epochs: int=15, batch_size: int | None=None, lr: float=0.0003, weight_decay: float=0.0001, num_workers: int=4, seed: int=42, force_splits: bool=False, pretrained: bool=True, device: torch.device | None=None) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits = build_splits(patches_dir=patches_dir, splits_dir=splits_dir, seed=seed, force=force_splits)
    name = run_name or experiment_name(backbone)
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    bs = batch_size or DEFAULT_BATCH.get(backbone, 32)
    train_ds = PatchDataset(splits['train'], patches_dir, train_transforms())
    val_ds = PatchDataset(splits['val'], patches_dir, eval_transforms())
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=device.type == 'cuda', persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda', persistent_workers=num_workers > 0)
    model = create_model(len(GENERA), backbone=backbone, pretrained=pretrained)
    model.to(device)
    weights = class_weights_from_frame(splits['train']).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    meta = {'experiment': name, 'backbone': backbone, 'epochs': epochs, 'batch_size': bs, 'lr': lr, 'classes': GENERA, 'train_n': len(train_ds), 'val_n': len(val_ds), 'device': str(device)}
    (run_dir / 'config.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    history: list[dict] = []
    best_f1 = -1.0
    best_path = run_dir / 'best.pt'
    print(f'=== {name} ({backbone}) ===')
    print(f'Device={device} train={len(train_ds):,} val={len(val_ds):,} batch={bs}')
    print(f'Run dir: {run_dir}')
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_metrics['loss'], 'val_accuracy': val_metrics['accuracy'], 'val_macro_f1': val_metrics['macro_f1'], 'lr': optimizer.param_groups[0]['lr'], 'seconds': time.time() - t0}
        history.append(row)
        print(f"epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}  val_macro_f1={val_metrics['macro_f1']:.4f}  ({row['seconds']:.0f}s)")
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            torch.save({'model': model.state_dict(), 'backbone': backbone, 'experiment': name, 'classes': GENERA, 'epoch': epoch, 'val_macro_f1': best_f1, 'val_accuracy': val_metrics['accuracy']}, best_path)
            print(f'  saved best -> {best_path} (macro_f1={best_f1:.4f})')
    torch.save({'model': model.state_dict(), 'backbone': backbone, 'experiment': name, 'classes': GENERA, 'epoch': epochs, 'val_macro_f1': history[-1]['val_macro_f1'] if history else None}, run_dir / 'last.pt')
    (run_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(f'Done {name}. Best val macro_f1={best_f1:.4f}')
    return run_dir

def main() -> None:
    args = parse_args()
    if args.experiment:
        backbone = EXPERIMENTS[args.experiment]
        run_name = args.run_name or args.experiment
    else:
        backbone = args.backbone
        run_name = args.run_name or experiment_name(backbone)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    train_run(backbone=backbone, run_name=run_name, patches_dir=args.patches_dir, splits_dir=args.splits_dir, runs_dir=args.runs_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, num_workers=args.num_workers, seed=args.seed, force_splits=args.force_splits, pretrained=not args.no_pretrained, device=device)
if __name__ == '__main__':
    main()
