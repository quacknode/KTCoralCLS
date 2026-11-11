from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from ktcoralcls.genera import GENERA, IMAGE_EXTS, PATCH_SIZE, SPLIT_SEED
ROOT = Path(__file__).resolve().parents[2]
_BALANCED = ROOT / 'datasets' / 'ktcoral' / 'patches_balanced'
_FULL = ROOT / 'datasets' / 'ktcoral' / 'patches'
DEFAULT_PATCHES = _BALANCED if _BALANCED.exists() else _FULL
DEFAULT_SPLITS = ROOT / 'datasets' / 'ktcoral' / 'splits_balanced' if _BALANCED.exists() else ROOT / 'datasets' / 'ktcoral' / 'splits'

def genus_to_index() -> dict[str, int]:
    return {g: i for i, g in enumerate(GENERA)}

def index_to_genus() -> dict[int, str]:
    return {i: g for i, g in enumerate(GENERA)}

def discover_patches(patches_dir: Path=DEFAULT_PATCHES) -> pd.DataFrame:
    label_map = genus_to_index()
    rows: list[dict] = []
    for genus in GENERA:
        gdir = patches_dir / genus
        if not gdir.is_dir():
            continue
        for p in gdir.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                rows.append({'path': str(p.relative_to(patches_dir)).replace('\\', '/'), 'genus': genus, 'label': label_map[genus]})
    if not rows:
        raise FileNotFoundError(f'No patches found under {patches_dir}')
    return pd.DataFrame(rows)

def build_splits(patches_dir: Path=DEFAULT_PATCHES, splits_dir: Path=DEFAULT_SPLITS, seed: int=SPLIT_SEED, train_ratio: float=0.8, val_ratio: float=0.1, force: bool=False) -> dict[str, pd.DataFrame]:
    splits_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: splits_dir / f'{s}.csv' for s in ('train', 'val', 'test')}
    if not force and all((p.exists() for p in paths.values())):
        return {s: pd.read_csv(p) for s, p in paths.items()}
    df = discover_patches(patches_dir)
    test_size = 1.0 - train_ratio - val_ratio
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df['label'])
    val_frac = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=val_frac, random_state=seed, stratify=train_val['label'])
    out = {'train': train.reset_index(drop=True), 'val': val.reset_index(drop=True), 'test': test.reset_index(drop=True)}
    for name, part in out.items():
        part = part.copy()
        part['split'] = name
        part.to_csv(paths[name], index=False)
        out[name] = part
    return out

def load_splits(splits_dir: Path=DEFAULT_SPLITS) -> dict[str, pd.DataFrame]:
    return {s: pd.read_csv(splits_dir / f'{s}.csv') for s in ('train', 'val', 'test')}

def class_weights_from_frame(df: pd.DataFrame, num_classes: int=len(GENERA)) -> torch.Tensor:
    counts = df['label'].value_counts()
    weights = torch.ones(num_classes, dtype=torch.float32)
    n = len(df)
    for label, count in counts.items():
        weights[int(label)] = n / (num_classes * float(count))
    return weights

def train_transforms(size: int=PATCH_SIZE) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(size + 32), transforms.RandomCrop(size), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def eval_transforms(size: int=PATCH_SIZE) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class PatchDataset(Dataset):

    def __init__(self, frame: pd.DataFrame, patches_dir: Path, transform: transforms.Compose | None=None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.patches_dir = Path(patches_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[idx]
        path = self.patches_dir / row['path']
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                tensor = transforms.ToTensor()(img)
        return (tensor, int(row['label']))
