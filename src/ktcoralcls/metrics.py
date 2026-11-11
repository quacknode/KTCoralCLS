from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

@dataclass
class EvalBundle:
    y_true: np.ndarray
    y_pred: np.ndarray
    classes: list[str]
    accuracy: float
    macro_f1: float
    weighted_f1: float
    min_class_f1: float
    mean_class_recall: float
    min_class_recall: float
    confusion: np.ndarray
    confusion_norm: np.ndarray
    per_class: pd.DataFrame
    confused_pairs: pd.DataFrame

def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> EvalBundle:
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sums, 1), dtype=np.float64)
    report = classification_report(y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0)
    rows = []
    for i, name in enumerate(classes):
        r = report[name]
        rows.append({'genus': name, 'precision': r['precision'], 'recall': r['recall'], 'f1': r['f1-score'], 'support': int(r['support'])})
    per_class = pd.DataFrame(rows)
    pair_rows = []
    for i, true_g in enumerate(classes):
        for j, pred_g in enumerate(classes):
            if i == j:
                continue
            count = int(cm[i, j])
            if count <= 0:
                continue
            rate = float(cm_norm[i, j])
            pair_rows.append({'true': true_g, 'predicted': pred_g, 'count': count, 'rate_given_true': rate, 'pair': f'{true_g}→{pred_g}'})
    confused = pd.DataFrame(pair_rows)
    if not confused.empty:
        confused = confused.sort_values(['count', 'rate_given_true'], ascending=False).reset_index(drop=True)
    f1s = per_class['f1'].to_numpy()
    recalls = per_class['recall'].to_numpy()
    return EvalBundle(y_true=y_true, y_pred=y_pred, classes=classes, accuracy=float(accuracy_score(y_true, y_pred)), macro_f1=float(f1_score(y_true, y_pred, average='macro', zero_division=0)), weighted_f1=float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), min_class_f1=float(f1s.min()) if len(f1s) else 0.0, mean_class_recall=float(recalls.mean()) if len(recalls) else 0.0, min_class_recall=float(recalls.min()) if len(recalls) else 0.0, confusion=cm, confusion_norm=cm_norm, per_class=per_class, confused_pairs=confused)

def top_confused_pairs(bundle: EvalBundle, k: int=15) -> pd.DataFrame:
    if bundle.confused_pairs.empty:
        return bundle.confused_pairs
    return bundle.confused_pairs.head(k).copy()

def symmetric_pair_errors(bundle: EvalBundle, min_count: int=5) -> pd.DataFrame:
    cm = bundle.confusion
    classes = bundle.classes
    rows = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ab = int(cm[i, j])
            ba = int(cm[j, i])
            total = ab + ba
            if total < min_count:
                continue
            rows.append({'genus_a': classes[i], 'genus_b': classes[j], 'a_as_b': ab, 'b_as_a': ba, 'total_swap': total, 'pair': f'{classes[i]} ↔ {classes[j]}'})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values('total_swap', ascending=False).reset_index(drop=True)
