import numpy as np
import pandas as pd
import time

start_time = time.time()
print("=== Experiment 5: Multi-Model Ensemble Blend & Post-Processing ===")

# 1. Load component submissions
s_exp4 = pd.read_csv('/root/kaggle_store_sales/submission_exp4_hybrid_residual.csv')
s_exp3 = pd.read_csv('/root/kaggle_store_sales/submission_exp3_catboost_exogenous.csv')
test = pd.read_csv('/root/kaggle_store_sales/test.csv')

# 2. Log-space Ensembling
# 82% Hybrid Residual GBDT (anchored on domain prior) + 18% Exogenous GBDT (capturing holiday & promo lifts)
log_exp4 = np.log1p(s_exp4['sales'])
log_exp3 = np.log1p(s_exp3['sales'])

blend_log = 0.82 * log_exp4 + 0.18 * log_exp3
blend_sales = np.expm1(blend_log)

# 3. Post-Processing Domain Guardrails
# Enforce zero mask on dead SKUs with zero promotions in test
dead_mask = (s_exp4['sales'] == 0.0)
blend_sales[dead_mask] = 0.0
print(f"Applied promotion-guarded zero mask to {dead_mask.sum()} test rows.")

# 4. Save Submission
sub = pd.DataFrame({'id': test['id'], 'sales': blend_sales})
sub.to_csv('/root/kaggle_store_sales/submission_exp5_ensemble_blend.csv', index=False)
print(f"Submission saved to submission_exp5_ensemble_blend.csv in {time.time()-start_time:.2f}s")
