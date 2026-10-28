# Track 5: Mathematical Foundations & Evaluation Metrics

This module provides formal mathematical derivations, statistical theories, and vectorised NumPy implementations underlying competitive loss functions and discrete diagnostic evaluation.

---

## 📖 Module Scripts & Architecture

### 1. Continuous Probabilistic Calibration: Multi-Class Log Loss
* **File:** [`multiclass_log_loss.py`](./multiclass_log_loss.py)
* **Core Concepts:**
  * Maximum Likelihood Estimation (MLE) to Negative Log-Likelihood (NLL) derivation.
  * Why confident false predictions trigger catastrophic infinite penalties (`ln(0) -> -inf`).
  * Probability calibration, probability smoothing, and numerical clipping bounds (`eps = 1e-15`).
  * Reference Primer: [Multi-Class Log Loss Explained: Math, MLE & Python Code](https://malcolmlow.com/2026/10/26/multiclass-log-loss-explained-cross-entropy-mle-python/)

### 2. Discrete Diagnostic Auditing: Classification Diagnostics & Confusion Matrices
* **File:** [`classification_diagnostics.py`](./classification_diagnostics.py)
* **Core Concepts:**
  * **`C x C` Confusion Matrix:** Vectorised contingency table mapping ground-truth classes against discrete model predictions.
  * **Error Taxonomy:**
    * True Positives (`TP`): Accurate detections.
    * False Positives (`FP`, Type I error): False alarms / commission errors.
    * True Negatives (`TN`): Accurate exclusions.
    * False Negatives (`FN`, Type II error): Missed detections / omission errors.
  * **Diagnostic Metrics:**
    * `Precision = TP / (TP + FP)`: Confidence in positive flags; crucial when false alarms carry operational cost.
    * `Recall = TP / (TP + FN)`: Coverage of actual events; critical when missed detections carry severe danger.
    * `Specificity = TN / (TN + FP)`: Exclusion rate protecting baseline operations.
    * `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`: Harmonic mean balancing precision and recall.
  * **Multi-Class Aggregation Protocols:**
    * `Macro Average`: Unweighted arithmetic mean across classes; protects rare minority classes under severe class imbalance.
    * `Micro Average`: Global instance-level tally; directly mirrors overall accuracy.
    * `Weighted Average`: Support-weighted mean reflecting class prevalence in the dataset.

---

## 🚀 Quickstart & Usage

### Running Multi-Class Log Loss:
```bash
python3 multiclass_log_loss.py
```

### Running Classification Diagnostics:
```bash
python3 classification_diagnostics.py
```

### Programmatic Integration:
```python
import numpy as np
from classification_diagnostics import compute_classification_metrics, format_diagnostic_report

# 3-class ground truth and discrete predictions
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2])
y_pred = np.array([0, 1, 1, 0, 1, 2, 2, 0, 1, 2])
target_names = ["Winner A", "Winner B", "Tie"]

metrics = compute_classification_metrics(y_true, y_pred, target_names=target_names)
print(format_diagnostic_report(metrics))
```
