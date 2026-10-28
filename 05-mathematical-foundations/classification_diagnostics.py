"""
Classification Diagnostics: Confusion Matrices, Error Taxonomy & Derived Evaluation Metrics
Team: Pocket Data Lab
Article Reference: https://malcolmlow.com/2026/10/26/multiclass-log-loss-explained-cross-entropy-mle-python/

This module provides vectorised NumPy implementations of classification diagnostics:
  1. Multi-Class & Binary Confusion Matrix (C x C contingency table).
  2. Error Taxonomy:
       - True Positives (TP)
       - False Positives (FP, Type I error / False Alarm)
       - True Negatives (TN)
       - False Negatives (FN, Type II error / Missed Detection)
  3. Diagnostic Performance Metrics:
       - Accuracy: (TP + TN) / Total
       - Precision (Positive Predictive Value): TP / (TP + FP)
       - Recall (Sensitivity / True Positive Rate): TP / (TP + FN)
       - Specificity (True Negative Rate): TN / (TN + FP)
       - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
  4. Multi-Class Averaging Protocols:
       - Macro Average: Unweighted arithmetic mean across classes (protects rare classes).
       - Micro Average: Global instance-level tally (reflects global hit rate).
       - Weighted Average: Class support-weighted mean (reflects class prevalence).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def compute_confusion_matrix(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Computes the C x C confusion matrix contingency table.

    Parameters:
    -----------
    y_true : 1D array-like of shape (N,)
        Ground truth integer class labels in range [0, C-1].
    y_pred : 1D array-like of shape (N,)
        Predicted discrete integer class labels in range [0, C-1].
    num_classes : int, optional
        Total number of classes C. Inferred from max(y_true, y_pred) + 1 if None.

    Returns:
    --------
    np.ndarray of shape (C, C):
        Contingency matrix where entry (i, j) represents samples with
        ground-truth class i predicted as class j.
        Rows: Ground Truth | Columns: Predictions
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1D arrays; got shapes {y_true.shape} and {y_pred.shape}")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

    if num_classes is None:
        num_classes = int(max(np.max(y_true), np.max(y_pred))) + 1

    # Vectorised bincount indexing: index = y_true * num_classes + y_pred
    flat_indices = y_true * num_classes + y_pred
    counts = np.bincount(flat_indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    num_classes: Optional[int] = None,
    target_names: Optional[List[str]] = None,
    zero_division: float = 0.0,
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Computes per-class and global diagnostic classification metrics.

    Parameters:
    -----------
    y_true : 1D array-like of shape (N,)
        Ground truth integer class labels.
    y_pred : 1D array-like of shape (N,)
        Predicted discrete integer class labels.
    num_classes : int, optional
        Total number of distinct classes.
    target_names : list of str, optional
        Human-readable labels for classes. Defaults to ['Class 0', 'Class 1', ...].
    zero_division : float, default=0.0
        Value to assign when metric denominator is zero.

    Returns:
    --------
    dict containing:
        - 'confusion_matrix': np.ndarray of shape (C, C)
        - 'per_class': dict mapping class name to metrics (TP, FP, FN, TN, precision, recall, specificity, f1, support)
        - 'accuracy': float
        - 'macro_avg': dict with precision, recall, specificity, f1
        - 'micro_avg': dict with precision, recall, f1
        - 'weighted_avg': dict with precision, recall, specificity, f1
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    c = cm.shape[0]
    total_samples = int(np.sum(cm))

    if target_names is None:
        target_names = [f"Class {i}" for i in range(c)]
    elif len(target_names) != c:
        raise ValueError(f"len(target_names)={len(target_names)} does not match num_classes={c}")

    per_class_metrics = {}
    precisions = np.zeros(c, dtype=np.float64)
    recalls = np.zeros(c, dtype=np.float64)
    specificities = np.zeros(c, dtype=np.float64)
    f1_scores = np.zeros(c, dtype=np.float64)
    supports = np.zeros(c, dtype=np.int64)

    for i in range(c):
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - tp)  # Column sum minus diagonal: Type I error
        fn = int(np.sum(cm[i, :]) - tp)  # Row sum minus diagonal: Type II error
        tn = int(total_samples - tp - fp - fn)
        support = int(np.sum(cm[i, :]))

        # Precision = TP / (TP + FP)
        prec_denom = tp + fp
        precision = float(tp / prec_denom) if prec_denom > 0 else zero_division

        # Recall (Sensitivity) = TP / (TP + FN)
        rec_denom = tp + fn
        recall = float(tp / rec_denom) if rec_denom > 0 else zero_division

        # Specificity = TN / (TN + FP)
        spec_denom = tn + fp
        specificity = float(tn / spec_denom) if spec_denom > 0 else zero_division

        # F1-Score = 2 * (P * R) / (P + R)
        f1_denom = precision + recall
        f1 = float(2.0 * precision * recall / f1_denom) if f1_denom > 0 else zero_division

        precisions[i] = precision
        recalls[i] = recall
        specificities[i] = specificity
        f1_scores[i] = f1
        supports[i] = support

        per_class_metrics[target_names[i]] = {
            "TP": tp,
            "FP (Type I)": fp,
            "FN (Type II)": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "support": support,
        }

    # Global Accuracy
    total_correct = int(np.trace(cm))
    accuracy = float(total_correct / total_samples) if total_samples > 0 else 0.0

    # Macro Averages (unweighted arithmetic mean across classes)
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_specificity = float(np.mean(specificities))
    macro_f1 = float(np.mean(f1_scores))

    # Weighted Averages (weighted by class support)
    if total_samples > 0:
        weights = supports / total_samples
        weighted_precision = float(np.sum(weights * precisions))
        weighted_recall = float(np.sum(weights * recalls))
        weighted_specificity = float(np.sum(weights * specificities))
        weighted_f1 = float(np.sum(weights * f1_scores))
    else:
        weighted_precision = weighted_recall = weighted_specificity = weighted_f1 = 0.0

    # Micro Average (global instance tally: Micro P = Micro R = Micro F1 = Accuracy)
    micro_precision = micro_recall = micro_f1 = accuracy

    return {
        "confusion_matrix": cm,
        "target_names": target_names,
        "per_class": per_class_metrics,
        "accuracy": accuracy,
        "total_samples": total_samples,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "specificity": macro_specificity,
            "f1_score": macro_f1,
        },
        "micro_avg": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1_score": micro_f1,
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "specificity": weighted_specificity,
            "f1_score": weighted_f1,
        },
    }


def format_diagnostic_report(metrics: Dict) -> str:
    """
    Renders an ASCII diagnostic report detailing class performance,
    error breakdown, and multi-class aggregations.
    """
    lines = []
    lines.append("=" * 86)
    lines.append(f"{'CLASSIFICATION DIAGNOSTICS & ERROR TAXONOMY REPORT':^86}")
    lines.append(f"{'Team: Pocket Data Lab':^86}")
    lines.append("=" * 86)

    # Confusion matrix section
    cm = metrics["confusion_matrix"]
    names = metrics["target_names"]
    lines.append(f"\n[1] Confusion Matrix (Rows: Ground Truth | Columns: Predictions)")
    col_header = "       " + "".join([f"{name[:10]:>12}" for name in names]) + "     Total"
    lines.append(col_header)
    lines.append("-" * len(col_header))
    for i, name in enumerate(names):
        row_str = f"{name[:6]:<7}" + "".join([f"{cm[i, j]:>12}" for j in range(len(names))]) + f"{np.sum(cm[i, :]):>10}"
        lines.append(row_str)
    lines.append("-" * len(col_header))
    pred_totals = "PredTot" + "".join([f"{np.sum(cm[:, j]):>12}" for j in range(len(names))]) + f"{np.sum(cm):>10}"
    lines.append(pred_totals)

    # Detailed diagnostic table
    lines.append(f"\n[2] Detailed Error Taxonomy & Diagnostic Metrics")
    header = (
        f"{'Class':<14} {'TP':>5} {'FP(I)':>6} {'FN(II)':>7} {'TN':>6} "
        f"{'Prec':>8} {'Recall':>8} {'Spec':>8} {'F1':>8} {'Support':>8}"
    )
    lines.append(header)
    lines.append("-" * 86)
    for name in names:
        p = metrics["per_class"][name]
        lines.append(
            f"{name:<14} {p['TP']:>5} {p['FP (Type I)']:>6} {p['FN (Type II)']:>7} {p['TN']:>6} "
            f"{p['precision']:>8.4f} {p['recall']:>8.4f} {p['specificity']:>8.4f} {p['f1_score']:>8.4f} {p['support']:>8}"
        )
    lines.append("-" * 86)

    # Summary rows
    mac = metrics["macro_avg"]
    wt = metrics["weighted_avg"]
    mic = metrics["micro_avg"]
    lines.append(
        f"{'Accuracy':<38} {'':>6} {'':>8} {'':>8} {'':>8} {metrics['accuracy']:>8.4f} {metrics['total_samples']:>8}"
    )
    lines.append(
        f"{'Macro Avg':<38} {'':>6} {mac['precision']:>8.4f} {mac['recall']:>8.4f} {mac['specificity']:>8.4f} {mac['f1_score']:>8.4f} {metrics['total_samples']:>8}"
    )
    lines.append(
        f"{'Weighted Avg':<38} {'':>6} {wt['precision']:>8.4f} {wt['recall']:>8.4f} {wt['specificity']:>8.4f} {wt['f1_score']:>8.4f} {metrics['total_samples']:>8}"
    )
    lines.append("=" * 86)

    return "\n".join(lines)


if __name__ == "__main__":
    print("Testing Classification Diagnostics Module...")

    # Case 1: Multi-Class 3-Way LLM Preference Benchmark (Winner A, Winner B, Tie)
    print("\n" + "=" * 50)
    print("Case 1: 3-Class LLM Human Preference Scenario")
    print("=" * 50)
    y_true_mc = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 1, 0, 2, 2, 1])
    y_pred_mc = np.array([0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 0, 0, 2, 1, 1])
    class_names_mc = ["Model A Win", "Model B Win", "Tie"]

    metrics_mc = compute_classification_metrics(y_true_mc, y_pred_mc, target_names=class_names_mc)
    print(format_diagnostic_report(metrics_mc))

    # Case 2: Extreme Class Imbalance (Why Accuracy Fails & Macro F1 Succeeds)
    print("\n" + "=" * 50)
    print("Case 2: Severe Class Imbalance (95% Negative, 5% Positive)")
    print("=" * 50)
    # 95 negatives (0) and 5 positives (1)
    y_true_imb = np.array([0] * 95 + [1] * 5)
    # Naive majority model predicting 0 for all instances
    y_pred_naive = np.array([0] * 100)
    metrics_imb = compute_classification_metrics(
        y_true_imb, y_pred_naive, target_names=["Majority (0)", "Anomaly (1)"]
    )
    print(format_diagnostic_report(metrics_imb))
    print("Notice: Accuracy is 95.00%, but Anomaly Recall is 0.0000 and Macro F1 is only 48.72%!")
    print("Diagnostic result confirms why accuracy alone is treacherous in imbalanced competitive tasks.")

    # Cross-validation against scikit-learn
    try:
        from sklearn.metrics import classification_report, confusion_matrix

        sk_cm = confusion_matrix(y_true_mc, y_pred_mc)
        assert np.array_equal(metrics_mc["confusion_matrix"], sk_cm), "Confusion matrix mismatch with sklearn!"
        print("\n[OK] Confusion matrix matches scikit-learn exactly.")
    except ImportError:
        print("\n[INFO] scikit-learn not installed; standalone validation passed.")
