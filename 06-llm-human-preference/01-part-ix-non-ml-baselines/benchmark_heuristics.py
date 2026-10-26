#!/usr/bin/env python3
"""
Benchmark Non-ML Heuristics for LLM Human Preference Prediction
Competition: llm-classification-finetuning (Chatbot Arena)
Team: Pocket Data Lab
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

print("Loading train dataset (57,477 rows)...")
cols = ['winner_model_a', 'winner_model_b', 'winner_tie', 'response_a', 'response_b']
df = pd.read_csv('/root/llm_classification_finetuning/train.csv', usecols=cols)

# Extract non-ML features
print("Extracting non-ML heuristic signals...")
df['len_a'] = df['response_a'].fillna('').apply(len)
df['len_b'] = df['response_b'].fillna('').apply(len)
df['len_diff'] = (df['len_a'] - df['len_b']) / (df['len_a'] + df['len_b'] + 1e-5)

def has_code(s):
    return 1.0 if '```' in str(s) else 0.0

def has_list(s):
    t = str(s)
    return 1.0 if ('- ' in t or '* ' in t or '1. ' in t) else 0.0

df['code_diff'] = df['response_a'].apply(has_code) - df['response_b'].apply(has_code)
df['list_diff'] = df['response_a'].apply(has_list) - df['response_b'].apply(has_list)

y_true = df[['winner_model_a', 'winner_model_b', 'winner_tie']].values

# Baseline 1: Uniform Prior
p_uniform = np.full_like(y_true, 1.0 / 3.0)
loss_uniform = log_loss(y_true, p_uniform)

# Baseline 2: Empirical Class Prior
prior_probs = y_true.mean(axis=0)
p_prior = np.tile(prior_probs, (len(df), 1))
loss_prior = log_loss(y_true, p_prior)

# Baseline 3: Pure Calibrated Length Heuristic
def compute_length_probs(params, len_diff):
    alpha, beta = params
    base_a = np.log(prior_probs[0])
    base_b = np.log(prior_probs[1])
    base_tie = np.log(prior_probs[2])
    
    logit_a = base_a + alpha * len_diff
    logit_b = base_b - alpha * len_diff
    logit_tie = base_tie - beta * np.abs(len_diff)
    
    logits = np.column_stack([logit_a, logit_b, logit_tie])
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

res_len = minimize(lambda p: log_loss(y_true, compute_length_probs(p, df['len_diff'].values)), [0.65, 0.15], method='Nelder-Mead')
loss_len = res_len.fun
alpha_opt, beta_opt = res_len.x

# Baseline 4: Multi-Feature Heuristic (Length + Code + Markdown Lists)
def compute_multi_probs(params):
    alpha, beta, gamma_code, gamma_list = params
    base_a = np.log(prior_probs[0])
    base_b = np.log(prior_probs[1])
    base_tie = np.log(prior_probs[2])
    
    delta = (alpha * df['len_diff'].values + 
             gamma_code * df['code_diff'].values + 
             gamma_list * df['list_diff'].values)
    
    logit_a = base_a + delta
    logit_b = base_b - delta
    logit_tie = base_tie - beta * np.abs(df['len_diff'].values)
    
    logits = np.column_stack([logit_a, logit_b, logit_tie])
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

res_multi = minimize(lambda p: log_loss(y_true, compute_multi_probs(p)), [alpha_opt, beta_opt, 0.05, 0.05], method='Nelder-Mead')
loss_multi = res_multi.fun
opt_params = res_multi.x

print("=" * 70)
print("NON-ML HEURISTIC BENCHMARK SUMMARY (57,477 Battles)")
print("=" * 70)
print(f"1. Uniform Random Baseline [1/3, 1/3, 1/3]:  Log Loss = {loss_uniform:.5f}")
print(f"2. Empirical Class Prior [{prior_probs[0]:.4f}, {prior_probs[1]:.4f}, {prior_probs[2]:.4f}]: Log Loss = {loss_prior:.5f} (Δ = {loss_uniform - loss_prior:+.5f})")
print(f"3. Calibrated Length Heuristic (α={alpha_opt:.3f}, β={beta_opt:.3f}): Log Loss = {loss_len:.5f} (Δ = {loss_prior - loss_len:+.5f})")
print(f"4. Multi-Feature Structure Heuristic:         Log Loss = {loss_multi:.5f} (Δ = {loss_prior - loss_multi:+.5f})")
print(f"   Optimal weights: Length α={opt_params[0]:.3f}, Tie β={opt_params[1]:.3f}, Code γ1={opt_params[2]:.3f}, Lists γ2={opt_params[3]:.3f}")
print("=" * 70)
