# Track 5: Mathematical Foundations & Evaluation Metrics

This module provides the formal mathematical derivations, statistical theories, and vectorised implementations underlying discrete diagnostic evaluation and continuous loss calibration.

---

## 📖 Module Structure & Curated Primers

### 1. Discrete Diagnostic Auditing: Classification Diagnostics & Confusion Matrices
* **File:** [`classification_diagnostics.py`](./classification_diagnostics.py)
* **Mathematical Theory & Architecture:**
  * **`C x C` Confusion Matrix:** Contingency table mapping ground-truth targets against discrete model predictions.
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

### 2. Continuous Probabilistic Calibration: Multi-Class Log Loss
* **File:** [`multiclass_log_loss.py`](./multiclass_log_loss.py)
* **Mathematical Theory & Architecture:**
  * **Maximum Likelihood Estimation (MLE) to Negative Log-Likelihood (NLL):** Derivation from joint likelihood to additive loss.
  * **Logarithmic Penalty Curvature:** Why confident false predictions trigger catastrophic infinite penalties (`ln(0) -> -inf`).
  * **Probability Calibration & Clipping:** Numerical stability bounds (`eps = 1e-15`) and multi-class probability normalization.
  * **Reference Primer:** [Multi-Class Log Loss Explained: Cross-Entropy & Maximum Likelihood Estimation](https://malcolmlow.com/2026/10/26/multiclass-log-loss-explained-cross-entropy-mle-python/)

---

## 🚀 Quickstart & Usage

### 1. Classification Diagnostics
```bash
python3 classification_diagnostics.py
```

### 2. Multi-Class Log Loss Calibration
```bash
python3 multiclass_log_loss.py
```

### Programmatic Integration
```python
import numpy as np
from classification_diagnostics import compute_classification_metrics, format_diagnostic_report
from multiclass_log_loss import safe_multiclass_log_loss

# 1. Discrete Diagnostic Evaluation (3-Class)
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2])
y_pred = np.array([0, 1, 1, 0, 1, 2, 2, 0, 1, 2])
target_names = ["Winner A", "Winner B", "Tie"]

metrics = compute_classification_metrics(y_true, y_pred, target_names=target_names)
print(format_diagnostic_report(metrics))

# 2. Continuous Probabilistic Log Loss
y_true_oh = np.eye(3)[y_true]
y_probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7],
                    [0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8],
                    [0.5, 0.3, 0.2], [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
loss = safe_multiclass_log_loss(y_true_oh, y_probs)
print(f"Multi-Class Log Loss: {loss:.5f}")
```
