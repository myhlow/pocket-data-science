# Part X: Mobile GBDT Machine Learning for LMSYS Preference

**Track 6 &middot; Module 11** &nbsp;|&nbsp; **Kaggle Team:** Pocket Data Lab &nbsp;|&nbsp; **Live Article:** [Pocket Data Science X: Mobile GBDT for LLM Preferences](https://malcolmlow.com/2026/10/26/pocket-data-science-10-kaggle-llm-classification-gbdt-android-termux/)

This module advances from surface-level heuristics (Part IX) to full supervised machine learning on mobile ARM64 hardware (Android Termux PRoot). We engineer **58 vectorized linguistic, structural, readability, and prompt-overlap features** across all 57,477 pairwise battles and train calibrated linear and gradient-boosted tree architectures.

---

## 🏆 Full 4-Experiment Kaggle Benchmark Scorecard

| Experiment & Architecture | Model Hyperparameters | 5-Fold Local CV Loss | Kaggle Public LB Score | Submission Ref |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Calibrated Logistic Regression** | $L_2$ regularized multinomial logistic regression (`C=0.1`) | `1.05613` | **`1.06173`** | `57667634` |
| **Exp 2: Standalone LightGBM GBDT** | `max_depth=6`, `num_leaves=31`, `lr=0.05`, 160 estimators | `1.03601` | **`1.04081`** | `57667635` |
| **Exp 3: Standalone CatBoost Classifier** | Oblivious symmetric trees, `depth=5`, `lr=0.06`, 300 iterations | `1.03594` | **`1.04455`** | `57667637` |
| **Exp 4: Dual GBDT Blended Ensemble** | SLSQP-optimized convex blend (50.6% LightGBM + 49.4% CatBoost) | **`1.03493`** | **`1.04158`** | `57657442` |

*Official Leaderboard Standing: **Rank #170 out of 290 teams** (+52 positions gained over Part IX).*

---

## 🚀 Standalone Kaggle Notebook Kernels
Each experiment is available as an independent, self-contained Kaggle Code competition notebook:
* **Exp 1:** [`myhlow/lmsys-part-x-exp1-logistic-regression`](https://www.kaggle.com/code/myhlow/lmsys-part-x-exp1-logistic-regression)
* **Exp 2:** [`myhlow/lmsys-part-x-exp2-lightgbm`](https://www.kaggle.com/code/myhlow/lmsys-part-x-exp2-lightgbm)
* **Exp 3:** [`myhlow/lmsys-part-x-exp3-catboost`](https://www.kaggle.com/code/myhlow/lmsys-part-x-exp3-catboost)
* **Exp 4 (Dual Blend):** [`myhlow/llm-preference-mobile-gbdt-pocket-data-lab`](https://www.kaggle.com/code/myhlow/llm-preference-mobile-gbdt-pocket-data-lab)

---

## 📂 Source Code Structure
* `main.py`: Full stand-alone Python pipeline executing feature extraction, model fitting, and test inference.
* `main.ipynb`: Self-contained Jupyter notebook formatted for Kaggle competition runner.
* `run_ml_experiments.py`: Local 5-fold cross-validation benchmarking harness with SLSQP blend optimizer.
* `test_feature_extraction.py`: Unit test verifying matrix generation runtime (14.36s for 57,477 rows).
* `kernel-metadata.json`: Kaggle API configuration metadata.
