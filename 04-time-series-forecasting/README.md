# Track 4: Time Series Forecasting & Quad-Ensembles

This track documents the Store Sales time-series forecasting campaign: advancing from non-ML hierarchical moving averages to hybrid residual learning, culminating in a convex 4-way ensemble reaching the global top 100.

---

## 🏆 Benchmark Scorecard

| Module & Milestone | Methodology & Models | Kaggle Standing | Article Guide |
| :--- | :--- | :--- | :--- |
| **Part VI &mdash; Non-ML Baselines** | Hierarchical rolling moving averages (zero ML weights) | **Top 33.1% Global** (0.44975) | [Part VI Guide](https://malcolmlow.com/2026/10/15/pocket-data-science-6-store-sales-time-series-non-ml-baselines/) |
| **Part VII &mdash; Residual GBDT** | Moving average trend + LightGBM residual learning | **Top 21.0% Global** (0.41843) | [Part VII Guide](https://malcolmlow.com/2026/10/16/pocket-data-science-7-store-sales-gradient-boosting-android-termux/) |
| **Part VIII &mdash; The Quad-Ensemble** | 4-Way Convex Blend (Trend + Calendar + GBDT Residuals) | **Global Top 100 &middot; Rank #99** (0.40455) | [Part VIII Guide](https://malcolmlow.com/2026/10/17/pocket-data-science-8-store-sales-quad-ensemble-top-100-android-termux/) |

---

## 📂 Subdirectories
* `01-store-sales-quad-ensemble/`: 11 progressive forecasting scripts from simple Ridge baselines to the grandmaster Quad-Ensemble.
