from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ktcoralcls.metrics import EvalBundle, symmetric_pair_errors, top_confused_pairs
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10, 'legend.fontsize': 9, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.grid': False})

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

def plot_metric_bars(summary: pd.DataFrame, out: Path) -> None:
    metrics = [('accuracy', 'Overall accuracy'), ('macro_f1', 'Macro F1'), ('min_class_f1', 'Worst-genus F1'), ('min_class_recall', 'Worst-genus recall')]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    names = summary['model'].tolist()
    x = np.arange(len(names))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        vals = summary[col].to_numpy()
        bars = ax.bar(x, vals, color='#2c6e8a', edgecolor='black', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel('Score')
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    fig.suptitle('KTCoralsCLS backbone comparison', fontsize=13)
    _save(fig, out)

def plot_learning_curves(histories: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, hist in histories.items():
        axes[0].plot(hist['epoch'], hist['val_accuracy'], label=name)
        axes[1].plot(hist['epoch'], hist['val_macro_f1'], label=name)
    axes[0].set_title('Validation accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_title('Validation macro F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro F1')
    for ax in axes:
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)
    _save(fig, out)

def plot_confusion_matrix(bundle: EvalBundle, out: Path, title: str, normalize: bool=True) -> None:
    mat = bundle.confusion_norm if normalize else bundle.confusion.astype(float)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1.0 if normalize else None, aspect='auto')
    ax.set_xticks(range(len(bundle.classes)))
    ax.set_yticks(range(len(bundle.classes)))
    ax.set_xticklabels(bundle.classes, rotation=90, fontsize=7)
    ax.set_yticklabels(bundle.classes, fontsize=7)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out)

def plot_per_genus_f1(bundles: dict[str, EvalBundle], out: Path) -> None:
    frames = []
    for name, b in bundles.items():
        df = b.per_class[['genus', 'f1']].copy()
        df['model'] = name
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    mean_f1 = long.groupby('genus')['f1'].mean().sort_values()
    genera = mean_f1.index.tolist()
    models = list(bundles.keys())
    fig, ax = plt.subplots(figsize=(12, max(6, 0.28 * len(genera))))
    y = np.arange(len(genera))
    height = 0.8 / max(len(models), 1)
    colors = ['#2c6e8a', '#c47b2b', '#3d7a4a', '#8b3a4a']
    for i, model in enumerate(models):
        vals = long[long['model'] == model].set_index('genus').reindex(genera)['f1'].fillna(0).to_numpy()
        ax.barh(y + i * height, vals, height=height, label=model, color=colors[i % len(colors)], edgecolor='none')
    ax.set_yticks(y + height * (len(models) - 1) / 2)
    ax.set_yticklabels(genera, fontsize=8)
    ax.set_xlabel('F1')
    ax.set_xlim(0, 1.05)
    ax.set_title('Per-genus F1 (hardest genera at top)')
    ax.legend(loc='lower right', fontsize=8)
    _save(fig, out)

def plot_top_confusions(bundles: dict[str, EvalBundle], out: Path, k: int=12) -> None:
    n = len(bundles)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (name, bundle) in zip(axes, bundles.items()):
        top = top_confused_pairs(bundle, k=k)
        if top.empty:
            ax.set_title(name)
            continue
        labels = top['pair'].tolist()[::-1]
        counts = top['count'].tolist()[::-1]
        ax.barh(labels, counts, color='#8b3a4a')
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Errors')
        ax.tick_params(axis='y', labelsize=7)
    fig.suptitle('Top genus confusions (true → predicted)', fontsize=12)
    _save(fig, out)

def plot_lookalike_swaps(bundles: dict[str, EvalBundle], out: Path, top_n: int=10) -> None:
    pair_scores: dict[str, int] = {}
    pair_frames: dict[str, pd.DataFrame] = {}
    for name, b in bundles.items():
        df = symmetric_pair_errors(b, min_count=1)
        pair_frames[name] = df
        for _, row in df.head(30).iterrows():
            pair_scores[row['pair']] = max(pair_scores.get(row['pair'], 0), int(row['total_swap']))
    if not pair_scores:
        return
    top_pairs = [p for p, _ in sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
    models = list(bundles.keys())
    data = np.zeros((len(top_pairs), len(models)))
    for j, model in enumerate(models):
        df = pair_frames[model]
        lookup = {r['pair']: int(r['total_swap']) for _, r in df.iterrows()} if not df.empty else {}
        for i, pair in enumerate(top_pairs):
            data[i, j] = lookup.get(pair, 0)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(top_pairs))))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_yticks(range(len(top_pairs)))
    ax.set_yticklabels(top_pairs, fontsize=8)
    ax.set_title('Lookalike genus swaps (A↔B error counts)')
    for i in range(len(top_pairs)):
        for j in range(len(models)):
            ax.text(j, i, f'{int(data[i, j])}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save(fig, out)
