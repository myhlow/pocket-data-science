# 📱 Pocket Data Science: Edge Machine Learning & Kaggle Monorepo

[![Platform](https://img.shields.io/badge/Platform-Android%20Termux%20PRoot-38bdf8.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](#)
[![Kaggle Team](https://img.shields.io/badge/Kaggle-Pocket%20Data%20Lab-20beff.svg)](https://www.kaggle.com/pocketdatalab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical research series and open-source codebase proving that competitive machine learning, feature engineering, probabilistic calibration, and top-tier Kaggle standings can be developed, tested, and dispatched entirely from handheld Android smartphones.

Developed by team **Pocket Data Lab** using **Android Termux**, **PRoot Linux**, and **Google Antigravity CLI** (`agy`).

---

## 🗺️ Curriculum & Roadmap Overview

For the complete interactive learning roadmap and visual curriculum, visit:  
👉 **[Pocket Data Science: Complete Learning Path & 6-Track Roadmap](https://malcolmlow.com/2026/10/26/pocket-data-science-complete-learning-path-mobile-kaggle-roadmap/)**

```
pocket-data-science/
├── 00-infrastructure-setup/        # Android PRoot environment, compilers, and CLI orchestration
├── 01-tabular-classification/      # Titanic (Top 3%) & Spaceship Titanic (Top 6%)
├── 02-regression-geometry/          # House Prices advanced regression & target transformations
├── 03-edge-vision-nlp/             # MNIST Subspace SVM & NLP Disaster Tweets under 2 GB RAM
├── 04-time-series-forecasting/     # Store Sales Quad-Ensemble (Global Top 100 · Rank #99)
├── 05-mathematical-foundations/    # Multi-class log loss MLE derivations & evaluation metrics
└── 06-llm-human-preference/        # LMSYS Chatbot Arena: Heuristics & GBDTs (Rank #170)
```

---

## 🏆 Master Benchmark & Leaderboard Scorecard

All pipelines below were engineered, cross-validated, and submitted directly from a smartphone:

| Competition & Module | Methodology & Architecture | Official Standing | Track & Code |
| :--- | :--- | :--- | :--- |
| **Titanic (Part I)** | 10-Fold GBDT Blended Ensemble (RF + LGB + CB) | **Top 3% Global** (0.79665) | [`01-tabular-classification/`](01-tabular-classification/) |
| **Spaceship Titanic (Part II)** | CryoSleep Invariants & Native CatBoost | **Top 6% Global** (0.80687) | [`01-tabular-classification/`](01-tabular-classification/) |
| **House Prices (Part III)** | Log-Transformed Regularized Regression | Continuous RMSLE Benchmarked | [`02-regression-geometry/`](02-regression-geometry/) |
| **MNIST Digits (Part IV)** | PCA Subspace Projection + Support Vector Machine | RAM Bound (2 GB Budget) | [`03-edge-vision-nlp/`](03-edge-vision-nlp/) |
| **Disaster Tweets (Part V)** | Sparse TF-IDF N-grams + Linear Models | Edge NLP Baseline | [`03-edge-vision-nlp/`](03-edge-vision-nlp/) |
| **Store Sales Baseline (Part VI)** | Hierarchical Non-ML Moving Averages | **Top 33.1% Global** (0.44975) | [`04-time-series-forecasting/`](04-time-series-forecasting/) |
| **Store Sales Hybrid (Part VII)** | Deterministic Trend + LightGBM Residuals | **Top 21.0% Global** (0.41843) | [`04-time-series-forecasting/`](04-time-series-forecasting/) |
| **Store Sales Capstone (Part VIII)** | 4-Way Convex Ensemble (Trend + Calendar + GBDT) | **Global Top 100 &middot; Rank #99** (0.40455) | [`04-time-series-forecasting/`](04-time-series-forecasting/) |
| **LMSYS Baselines (Part IX)** | Calibrated Response Length Difference Heuristic | **Rank #222 of 291** (1.07823) | [`06-llm-human-preference/`](06-llm-human-preference/) |
| **LMSYS Machine Learning (Part X)** | 58 Linguistic Feats + Dual GBDT Ensemble | **Rank #170 of 290** (1.04158) | [`06-llm-human-preference/`](06-llm-human-preference/) |

---

## 🚀 Quick Start on Mobile

To clone and set up the repository on your Android Termux device:

```bash
# 1. Clone the repository
git clone https://github.com/myhlow/pocket-data-science.git
cd pocket-data-science

# 2. Bootstrap the PRoot environment
bash 00-infrastructure-setup/bootstrap_environment.sh

# 3. Run any experiment module (e.g. LMSYS Part X ML Benchmark)
python3 06-llm-human-preference/02-part-x-mobile-gbdt-ensemble/run_ml_experiments.py
```

---

## 📄 License & Attribution

Distributed under the MIT License. Developed by team **Pocket Data Lab** as part of the **Pocket Data Science** series on [malcolmlow.com](https://malcolmlow.com).
