from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ktcoralcls.data import class_weights_from_frame, eval_transforms, train_transforms
from ktcoralcls.genera import GENERA
from ktcoralcls.growth import GROWTH_CLASSES, IGNORE_INDEX
from ktcoralcls.model import create_multitask_model
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATCHES = ROOT / 'datasets' / 'ktcoral' / 'patches'
DEFAULT_SPLITS = ROOT / 'datasets' / 'ktcoral' / 'splits_growth'
DEFAULT_RUNS = ROOT / 'artifacts' / 'runs_growth'

class MultiTaskPatchDataset(Dataset):

    def __init__(self, frame: pd.DataFrame, patches_dir: Path, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.patches_dir = Path(patches_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        from PIL import Image
        row = self.frame.iloc[idx]
        path = self.patches_dir / row['path']
        with Image.open(path) as img:
            img = img.convert('RGB')
            x = self.transform(img)
        genus = int(row['label'])
        growth = int(row['growth_id']) if 'growth_id' in row and pd.notna(row['growth_id']) else IGNORE_INDEX
        return (x, genus, growth)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patches-dir', type=Path, default=DEFAULT_PATCHES)
    p.add_argument('--splits-dir', type=Path, default=DEFAULT_SPLITS)
    p.add_argument('--runs-dir', type=Path, default=DEFAULT_RUNS)
    p.add_argument('--run-name', type=str, default='KTCoralsCLS-ConvNeXt-MultiTask')
    p.add_argument('--backbone', type=str, default='convnext_tiny')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=48)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--growth-loss-weight', type=float, default=0.5)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()

def growth_class_weights(frame: pd.DataFrame, num_classes: int) -> torch.Tensor:
    labeled = frame.loc[frame['growth_id'] != IGNORE_INDEX, 'growth_id']
    weights = torch.ones(num_classes, dtype=torch.float32)
    if labeled.empty:
        return weights
    counts = labeled.value_counts()
    n = len(labeled)
    for label, count in counts.items():
        weights[int(label)] = n / (num_classes * float(count))
    return weights

@torch.no_grad()
def evaluate(model, loader, device, genus_crit, growth_crit, growth_w: float) -> dict:
    model.eval()
    g_true, g_pred = ([], [])
    w_true, w_pred = ([], [])
    total_loss = 0.0
    n = 0
    for x, y_g, y_w in loader:
        x = x.to(device, non_blocking=True)
        y_g = y_g.to(device, non_blocking=True)
        y_w = y_w.to(device, non_blocking=True)
        logits_g, logits_w = model(x)
        loss_g = genus_crit(logits_g, y_g)
        loss_w = growth_crit(logits_w, y_w)
        loss = loss_g + growth_w * loss_w
        bs = len(y_g)
        total_loss += loss.item() * bs
        n += bs
        g_true.extend(y_g.cpu().tolist())
        g_pred.extend(logits_g.argmax(1).cpu().tolist())
        mask = y_w != IGNORE_INDEX
        if mask.any():
            w_true.extend(y_w[mask].cpu().tolist())
            w_pred.extend(logits_w[mask].argmax(1).cpu().tolist())
    g_true_a = np.array(g_true)
    g_pred_a = np.array(g_pred)
    out = {'loss': total_loss / max(n, 1), 'genus_accuracy': float(accuracy_score(g_true_a, g_pred_a)), 'genus_macro_f1': float(f1_score(g_true_a, g_pred_a, average='macro', zero_division=0)), 'y_genus_true': g_true_a, 'y_genus_pred': g_pred_a}
    if w_true:
        w_true_a = np.array(w_true)
        w_pred_a = np.array(w_pred)
        out['growth_accuracy'] = float(accuracy_score(w_true_a, w_pred_a))
        out['growth_macro_f1'] = float(f1_score(w_true_a, w_pred_a, average='macro', zero_division=0))
        out['growth_n'] = int(len(w_true_a))
        out['y_growth_true'] = w_true_a
        out['y_growth_pred'] = w_pred_a
    else:
        out['growth_accuracy'] = 0.0
        out['growth_macro_f1'] = 0.0
        out['growth_n'] = 0
    return out

def train_one_epoch(model, loader, optimizer, genus_crit, growth_crit, growth_w, device, scaler) -> float:
    model.train()
    running = 0.0
    n = 0
    use_amp = scaler is not None
    for x, y_g, y_w in tqdm(loader, desc='train', leave=False):
        x = x.to(device, non_blocking=True)
        y_g = y_g.to(device, non_blocking=True)
        y_w = y_w.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits_g, logits_w = model(x)
            loss = genus_crit(logits_g, y_g) + growth_w * growth_crit(logits_w, y_w)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item() * len(y_g)
        n += len(y_g)
    return running / max(n, 1)

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    splits = {s: pd.read_csv(args.splits_dir / f'{s}.csv') for s in ('train', 'val', 'test')}
    run_dir = args.runs_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_ds = MultiTaskPatchDataset(splits['train'], args.patches_dir, train_transforms())
    val_ds = MultiTaskPatchDataset(splits['val'], args.patches_dir, eval_transforms())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0)
    model = create_multitask_model(len(GENERA), len(GROWTH_CLASSES), backbone=args.backbone, pretrained=True).to(device)
    genus_w = class_weights_from_frame(splits['train']).to(device)
    growth_w_vec = growth_class_weights(splits['train'], len(GROWTH_CLASSES)).to(device)
    genus_crit = nn.CrossEntropyLoss(weight=genus_w)
    growth_crit = nn.CrossEntropyLoss(weight=growth_w_vec, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    meta = {'experiment': args.run_name, 'backbone': args.backbone, 'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr, 'growth_loss_weight': args.growth_loss_weight, 'genus_classes': GENERA, 'growth_classes': GROWTH_CLASSES, 'train_n': len(train_ds), 'val_n': len(val_ds), 'train_growth_n': int((splits['train']['growth_id'] != IGNORE_INDEX).sum()), 'device': str(device)}
    (run_dir / 'config.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    history = []
    best_score = -1.0
    best_path = run_dir / 'best.pt'
    print(f'=== {args.run_name} ({args.backbone}) multi-task ===')
    print(f"Device={device} train={len(train_ds):,} val={len(val_ds):,} growth-labeled train={(splits['train']['growth_id'] != IGNORE_INDEX).sum():,}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, genus_crit, growth_crit, args.growth_loss_weight, device, scaler)
        val = evaluate(model, val_loader, device, genus_crit, growth_crit, args.growth_loss_weight)
        scheduler.step()
        score = val['genus_macro_f1'] + 0.25 * val['growth_macro_f1']
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val['loss'], 'val_genus_acc': val['genus_accuracy'], 'val_genus_macro_f1': val['genus_macro_f1'], 'val_growth_acc': val['growth_accuracy'], 'val_growth_macro_f1': val['growth_macro_f1'], 'val_growth_n': val['growth_n'], 'selection_score': score, 'lr': optimizer.param_groups[0]['lr'], 'seconds': time.time() - t0}
        history.append(row)
        print(f"epoch {epoch:02d}/{args.epochs}  loss={train_loss:.4f}  genus_acc={val['genus_accuracy']:.4f} genus_f1={val['genus_macro_f1']:.4f}  growth_acc={val['growth_accuracy']:.4f} growth_f1={val['growth_macro_f1']:.4f} (n={val['growth_n']})  ({row['seconds']:.0f}s)")
        if score > best_score:
            best_score = score
            torch.save({'model': model.state_dict(), 'backbone': args.backbone, 'experiment': args.run_name, 'genus_classes': GENERA, 'growth_classes': GROWTH_CLASSES, 'epoch': epoch, 'val_genus_macro_f1': val['genus_macro_f1'], 'val_growth_macro_f1': val['growth_macro_f1'], 'selection_score': score}, best_path)
            print(f'  saved best -> {best_path} (score={best_score:.4f})')
    torch.save({'model': model.state_dict(), 'backbone': args.backbone, 'experiment': args.run_name, 'genus_classes': GENERA, 'growth_classes': GROWTH_CLASSES, 'epoch': args.epochs}, run_dir / 'last.pt')
    (run_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(f'Done. Best selection_score={best_score:.4f}')
    print(f'Next: pipenv run python -m ktcoralcls.report_multitask --run-dir {run_dir}')
if __name__ == '__main__':
    main()
