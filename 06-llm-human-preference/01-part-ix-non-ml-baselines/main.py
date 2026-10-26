"""
LLM Preference Non-ML Baselines — Pocket Data Lab
Kaggle Competition: LLM Classification Finetuning

Theoretical Foundation & Multi-Class Log Loss Derivation:
https://malcolmlow.com/2026/10/26/multiclass-log-loss-explained-cross-entropy-mle-python/

Article Companion Guide (Part IX):
https://malcolmlow.com/2026/10/25/pocket-data-science-9-kaggle-llm-classification-non-ml-baselines-android-termux/
"""

import glob
import numpy as np
import pandas as pd

# Dynamic path resolution across local and Kaggle runtime environments
test_candidates = glob.glob('/kaggle/input/**/test.csv', recursive=True)
test_path = test_candidates[0] if test_candidates else 'test.csv'
test = pd.read_csv(test_path)

# Empirical class priors and base log-odds
p_A0, p_B0, p_T0 = 0.34908, 0.34191, 0.30901
base_zA, base_zB, base_zT = np.log(p_A0), np.log(p_B0), np.log(p_T0)

# Calibrated parameters optimized against 57,477 battles
alpha, beta, gamma_code, gamma_list = 0.575, 0.165, 0.281, 0.167

records = []
for _, row in test.iterrows():
    resp_a = str(row.get('response_a', ''))
    resp_b = str(row.get('response_b', ''))
    len_a, len_b = len(resp_a), len(resp_b)
    
    # Normalized length difference ratio r in [-1.0, 1.0]
    r = (len_a - len_b) / (len_a + len_b + 1.0)
    
    # Structural features
    d_code = (1.0 if '```' in resp_a else 0.0) - (1.0 if '```' in resp_b else 0.0)
    list_a = 1.0 if any(resp_a.lstrip().startswith(p) for p in ['* ', '- ', '1. ']) else 0.0
    list_b = 1.0 if any(resp_b.lstrip().startswith(p) for p in ['* ', '- ', '1. ']) else 0.0
    d_list = list_a - list_b
    
    # Compute unnormalized log-odds
    zA = base_zA + alpha * r + gamma_code * d_code + gamma_list * d_list
    zB = base_zB - alpha * r - gamma_code * d_code - gamma_list * d_list
    zT = base_zT - beta * abs(r) - 0.5 * (gamma_code * abs(d_code) + gamma_list * abs(d_list))
    
    # Softmax normalization
    max_z = max(zA, zB, zT)
    ea, eb, et = np.exp(zA - max_z), np.exp(zB - max_z), np.exp(zT - max_z)
    s = ea + eb + et
    
    records.append({
        'id': row['id'],
        'winner_model_a': ea / s,
        'winner_model_b': eb / s,
        'winner_tie': et / s
    })

sub = pd.DataFrame(records)
sub.to_csv('submission.csv', index=False)
print(f"Generated calibrated heuristic submission for {len(sub)} rows.")
