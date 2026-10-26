# Track 3: Mobile Edge Vision & NLP on RAM Budgets

This track benchmarks computer vision and natural language processing on handheld hardware strictly constrained to 2 GB memory ceilings without dedicated GPUs.

---

## 🏆 Benchmark Scorecard

| Competition & Module | Methodology & Models | Memory / Standing | Article Guide |
| :--- | :--- | :--- | :--- |
| **MNIST Digit Recognizer (Part IV)** | PCA Orthogonal Subspace Projection + RBF Support Vector Machines | 10-Class Edge Bound (2 GB RAM) | [Part IV Guide](https://malcolmlow.com/2026/09/27/pocket-data-science-4-mnist-digit-recognizer-subspace-svm-android-termux/) |
| **NLP Disaster Tweets (Part V)** | Sparse TF-IDF N-grams + Regularized Linear Models | Edge NLP Baseline | [Part V Guide](https://malcolmlow.com/2026/10/08/pocket-data-science-5-nlp-disaster-tweets-baselines/) |

---

## 📂 Subdirectories
* `01-mnist-subspace-svm/`: PCA dimensionality reduction and Support Vector Machines.
* `02-nlp-disaster-tweets/`: Sparse n-gram text vectorization and calibrated classifiers.
