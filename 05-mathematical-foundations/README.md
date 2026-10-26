# Track 5: Mathematical Foundations & Evaluation Metrics

This module provides the formal mathematical derivations, statistical theories, and numerical clipping implementations underlying competitive loss functions.

---

## 📖 Topics & Primers

* **Multi-Class Log Loss (Cross-Entropy):**
  * Maximum Likelihood Estimation (MLE) to Negative Log-Likelihood (NLL) derivation.
  * Why confident false predictions trigger catastrophic infinite penalties (`ln(0) &rarr; -\infty`).
  * Probability calibration, smoothing, and numerical clipping bounds (`eps = 1e-15`).
  * Full Reference Article: [Multi-Class Log Loss Explained: Math, MLE & Python Code](https://malcolmlow.com/2026/10/26/multiclass-log-loss-explained-cross-entropy-mle-python/)
* **Classification Diagnostics:**
  * Confusion Matrix error taxonomy (Type I vs Type II errors).
  * Precision, Recall, and F1 formulations (macro vs micro vs weighted).
