import time
import os
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import lightgbm as lgb
from catboost import CatBoostClassifier

print("=" * 75)
print("POCKET DATA SCIENCE PART X: MACHINE LEARNING FOR LMSYS PREFERENCE")
print("=" * 75)

t0 = time.time()
print("\n[Step 1] Loading full dataset (57,477 rows)...")
cols = ['id', 'prompt', 'response_a', 'response_b', 'winner_model_a', 'winner_model_b', 'winner_tie']
df = pd.read_csv('/root/llm_classification_finetuning/train.csv', usecols=cols)

y_multiclass = np.argmax(df[['winner_model_a', 'winner_model_b', 'winner_tie']].values, axis=1)
y_onehot = df[['winner_model_a', 'winner_model_b', 'winner_tie']].values

print(f"Dataset loaded: {len(df)} rows. Target class counts:")
for i, name in enumerate(['Model A', 'Model B', 'Tie']):
    cnt = np.sum(y_multiclass == i)
    print(f"  {name}: {cnt} ({cnt / len(df) * 100:.2f}%)")

print("\n[Step 2] Extracting engineered feature matrix...")
t_feat = time.time()

prompt = df['prompt'].fillna('').astype(str)
ra = df['response_a'].fillna('').astype(str)
rb = df['response_b'].fillna('').astype(str)

# 1. Surface Length Features
len_p = prompt.str.len().values.astype(np.float32)
len_a = ra.str.len().values.astype(np.float32)
len_b = rb.str.len().values.astype(np.float32)

len_diff = len_a - len_b
len_ratio = (len_a - len_b) / (len_a + len_b + 1.0)
len_ratio_p_a = len_a / (len_p + 1.0)
len_ratio_p_b = len_b / (len_p + 1.0)
len_ratio_p_diff = len_ratio_p_a - len_ratio_p_b

# 2. Word Count Features
words_p = prompt.str.split().str.len().values.astype(np.float32)
words_a = ra.str.split().str.len().values.astype(np.float32)
words_b = rb.str.split().str.len().values.astype(np.float32)

words_diff = words_a - words_b
words_ratio = (words_a - words_b) / (words_a + words_b + 1.0)

mean_word_len_a = len_a / (words_a + 1.0)
mean_word_len_b = len_b / (words_b + 1.0)
mean_word_len_diff = mean_word_len_a - mean_word_len_b

# 3. Structural & Markdown Features
has_code_a = ra.str.contains('```', regex=False).astype(np.float32).values
has_code_b = rb.str.contains('```', regex=False).astype(np.float32).values
code_diff = has_code_a - has_code_b

count_code_a = ra.str.count('```').values.astype(np.float32)
count_code_b = rb.str.count('```').values.astype(np.float32)
count_code_diff = count_code_a - count_code_b

list_a = ra.str.count(r'(?m)^[\s]*[-*•]\s').values.astype(np.float32)
list_b = rb.str.count(r'(?m)^[\s]*[-*•]\s').values.astype(np.float32)
list_diff = list_a - list_b
has_list_a = (list_a > 0).astype(np.float32)
has_list_b = (list_b > 0).astype(np.float32)
has_list_diff = has_list_a - has_list_b

num_list_a = ra.str.count(r'(?m)^[\s]*\d+\.\s').values.astype(np.float32)
num_list_b = rb.str.count(r'(?m)^[\s]*\d+\.\s').values.astype(np.float32)
num_list_diff = num_list_a - num_list_b

header_a = ra.str.count(r'(?m)^#{1,4}\s').values.astype(np.float32)
header_b = rb.str.count(r'(?m)^#{1,4}\s').values.astype(np.float32)
header_diff = header_a - header_b

bold_a = ra.str.count(r'\*\*[^*]+\*\*').values.astype(np.float32)
bold_b = rb.str.count(r'\*\*[^*]+\*\*').values.astype(np.float32)
bold_diff = bold_a - bold_b

newline_a = ra.str.count('\n').values.astype(np.float32)
newline_b = rb.str.count('\n').values.astype(np.float32)
newline_diff = newline_a - newline_b

# 4. Punctuation & Tone Features
excl_a = ra.str.count('!').values.astype(np.float32)
excl_b = rb.str.count('!').values.astype(np.float32)
excl_diff = excl_a - excl_b

quest_a = ra.str.count(r'\?').values.astype(np.float32)
quest_b = rb.str.count(r'\?').values.astype(np.float32)
quest_diff = quest_a - quest_b

sorry_a = ra.str.lower().str.contains(r'sorry|apologize|apologies', regex=True).astype(np.float32).values
sorry_b = rb.str.lower().str.contains(r'sorry|apologize|apologies', regex=True).astype(np.float32).values
sorry_diff = sorry_a - sorry_b

sure_a = ra.str.lower().str.contains(r'sure|certainly|gladly|of course|here is|here are', regex=True).astype(np.float32).values
sure_b = rb.str.lower().str.contains(r'sure|certainly|gladly|of course|here is|here are', regex=True).astype(np.float32).values
sure_diff = sure_a - sure_b

# 5. Overlap & Jaccard Lexical Matching
print("Computing prompt-response lexical overlap and Jaccard similarity...")
def get_word_set(text):
    return set(text.lower().split())

prompts = [get_word_set(x) for x in prompt]
ras = [get_word_set(x) for x in ra]
rbs = [get_word_set(x) for x in rb]

jaccard_a = np.zeros(len(df), dtype=np.float32)
jaccard_b = np.zeros(len(df), dtype=np.float32)
cov_a = np.zeros(len(df), dtype=np.float32)
cov_b = np.zeros(len(df), dtype=np.float32)

for i in range(len(df)):
    p, a, b = prompts[i], ras[i], rbs[i]
    lp = len(p)
    if lp == 0:
        continue
    inter_a = len(p.intersection(a))
    inter_b = len(p.intersection(b))
    union_a = len(p.union(a))
    union_b = len(p.union(b))
    
    jaccard_a[i] = inter_a / union_a if union_a > 0 else 0.0
    jaccard_b[i] = inter_b / union_b if union_b > 0 else 0.0
    cov_a[i] = inter_a / lp
    cov_b[i] = inter_b / lp

jaccard_diff = jaccard_a - jaccard_b
cov_diff = cov_a - cov_b

feature_dict = {
    'len_p': len_p, 'len_a': len_a, 'len_b': len_b, 'len_diff': len_diff, 'len_ratio': len_ratio,
    'len_ratio_p_a': len_ratio_p_a, 'len_ratio_p_b': len_ratio_p_b, 'len_ratio_p_diff': len_ratio_p_diff,
    'words_p': words_p, 'words_a': words_a, 'words_b': words_b, 'words_diff': words_diff, 'words_ratio': words_ratio,
    'mean_word_len_a': mean_word_len_a, 'mean_word_len_b': mean_word_len_b, 'mean_word_len_diff': mean_word_len_diff,
    'has_code_a': has_code_a, 'has_code_b': has_code_b, 'code_diff': code_diff,
    'count_code_a': count_code_a, 'count_code_b': count_code_b, 'count_code_diff': count_code_diff,
    'list_a': list_a, 'list_b': list_b, 'list_diff': list_diff,
    'has_list_a': has_list_a, 'has_list_b': has_list_b, 'has_list_diff': has_list_diff,
    'num_list_a': num_list_a, 'num_list_b': num_list_b, 'num_list_diff': num_list_diff,
    'header_a': header_a, 'header_b': header_b, 'header_diff': header_diff,
    'bold_a': bold_a, 'bold_b': bold_b, 'bold_diff': bold_diff,
    'newline_a': newline_a, 'newline_b': newline_b, 'newline_diff': newline_diff,
    'excl_a': excl_a, 'excl_b': excl_b, 'excl_diff': excl_diff,
    'quest_a': quest_a, 'quest_b': quest_b, 'quest_diff': quest_diff,
    'sorry_a': sorry_a, 'sorry_b': sorry_b, 'sorry_diff': sorry_diff,
    'sure_a': sure_a, 'sure_b': sure_b, 'sure_diff': sure_diff,
    'jaccard_a': jaccard_a, 'jaccard_b': jaccard_b, 'jaccard_diff': jaccard_diff,
    'cov_a': cov_a, 'cov_b': cov_b, 'cov_diff': cov_diff
}

X = pd.DataFrame(feature_dict)
print(f"Feature matrix built in {time.time() - t_feat:.2f}s! Shape: {X.shape}")

# Save feature names
feature_names = list(X.columns)
np.save('/root/llm_classification_finetuning/feature_names.npy', feature_names)

# Free text objects to conserve memory
del prompt, ra, rb, prompts, ras, rbs
gc.collect()

# Setup 5-Fold Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 75)
print("[Step 3] Running 5-Fold Stratified Cross-Validation Benchmarks...")
print("=" * 75)

# -------------------------------------------------------------
# Part IX Baseline check
# -------------------------------------------------------------
prior_probs = y_onehot.mean(axis=0)
loss_prior = log_loss(y_onehot, np.tile(prior_probs, (len(df), 1)))

# Calibrated Part IX heuristic prediction
def get_part_ix_heuristic_preds(X_df):
    r = X_df['len_ratio'].values
    d_code = X_df['code_diff'].values
    d_list = X_df['has_list_diff'].values
    base_zA, base_zB, base_zT = np.log(prior_probs[0]), np.log(prior_probs[1]), np.log(prior_probs[2])
    alpha, beta, gamma_code, gamma_list = 0.575, 0.165, 0.281, 0.167
    
    zA = base_zA + alpha * r + gamma_code * d_code + gamma_list * d_list
    zB = base_zB - alpha * r - gamma_code * d_code - gamma_list * d_list
    zT = base_zT - beta * np.abs(r) - 0.5 * (gamma_code * np.abs(d_code) + gamma_list * np.abs(d_list))
    
    logits = np.column_stack([zA, zB, zT])
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

oof_part_ix = get_part_ix_heuristic_preds(X)
loss_part_ix = log_loss(y_onehot, oof_part_ix)
print(f"Part IX Non-ML Baseline Multi-Heuristic: Log Loss = {loss_part_ix:.5f}")

# -------------------------------------------------------------
# Experiment 1: Regularized Logistic Regression (L2)
# -------------------------------------------------------------
print("\n--- Experiment 1: Calibrated Logistic Regression (L2) ---")
oof_lr = np.zeros_like(y_onehot, dtype=np.float64)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_multiclass)):
    clf = LogisticRegression(C=0.1, max_iter=200, multi_class='multinomial', random_state=42)
    clf.fit(X_scaled[train_idx], y_multiclass[train_idx])
    oof_lr[val_idx] = clf.predict_proba(X_scaled[val_idx])

loss_lr = log_loss(y_onehot, oof_lr)
print(f"5-Fold OOF Logistic Regression Log Loss: {loss_lr:.5f} (Δ vs Part IX = {loss_part_ix - loss_lr:+.5f})")

# -------------------------------------------------------------
# Experiment 2: LightGBM MultiClass (GBDT)
# -------------------------------------------------------------
print("\n--- Experiment 2: LightGBM MultiClass GBDT ---")
oof_lgb = np.zeros_like(y_onehot, dtype=np.float64)
lgb_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 300,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': 4
}

feature_importances = np.zeros(X.shape[1], dtype=np.float64)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_multiclass)):
    X_train, y_train = X.iloc[train_idx], y_multiclass[train_idx]
    X_val, y_val = X.iloc[val_idx], y_multiclass[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    oof_lgb[val_idx] = model.predict_proba(X_val)
    feature_importances += model.feature_importances_ / 5.0
    fold_loss = log_loss(y_onehot[val_idx], oof_lgb[val_idx])
    print(f"  Fold {fold + 1} Log Loss: {fold_loss:.5f} (Best Iter: {model.best_iteration_})")

loss_lgb = log_loss(y_onehot, oof_lgb)
print(f"5-Fold OOF LightGBM Log Loss: {loss_lgb:.5f} (Δ vs Part IX = {loss_part_ix - loss_lgb:+.5f})")

# Top 10 Features
fi_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances}).sort_values('importance', ascending=False)
print("\nTop 10 Most Influential Features (LightGBM):")
for idx, r in fi_df.head(10).iterrows():
    print(f"  - {r['feature']:<20}: {r['importance']:.1f}")

# -------------------------------------------------------------
# Experiment 3: CatBoost MultiClass
# -------------------------------------------------------------
print("\n--- Experiment 3: CatBoost MultiClass Classifier ---")
oof_cb = np.zeros_like(y_onehot, dtype=np.float64)
cb_params = {
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'iterations': 400,
    'learning_rate': 0.06,
    'depth': 5,
    'random_seed': 42,
    'verbose': 0,
    'thread_count': 4
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_multiclass)):
    X_train, y_train = X.iloc[train_idx], y_multiclass[train_idx]
    X_val, y_val = X.iloc[val_idx], y_multiclass[val_idx]
    
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)
    oof_cb[val_idx] = cb_model.predict_proba(X_val)
    fold_loss = log_loss(y_onehot[val_idx], oof_cb[val_idx])
    print(f"  Fold {fold + 1} CatBoost Log Loss: {fold_loss:.5f}")

loss_cb = log_loss(y_onehot, oof_cb)
print(f"5-Fold OOF CatBoost Log Loss: {loss_cb:.5f} (Δ vs Part IX = {loss_part_ix - loss_cb:+.5f})")

# -------------------------------------------------------------
# Experiment 4: Multi-Model Ensemble Blend
# -------------------------------------------------------------
print("\n--- Experiment 4: Blended Multi-Model Ensemble ---")
# Optimize blend weights: w_lgb, w_cb, w_lr, w_heuristic
from scipy.optimize import minimize

def ensemble_loss(weights):
    w1, w2, w3, w4 = weights
    w_sum = w1 + w2 + w3 + w4
    if w_sum <= 0:
        return 999.0
    p = (w1 * oof_lgb + w2 * oof_cb + w3 * oof_lr + w4 * oof_part_ix) / w_sum
    # Clip for safety
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    p = p / np.sum(p, axis=1, keepdims=True)
    return log_loss(y_onehot, p)

res_ens = minimize(ensemble_loss, [0.4, 0.4, 0.1, 0.1], bounds=[(0, 1)]*4, method='SLSQP')
best_weights = res_ens.x / np.sum(res_ens.x)
loss_ensemble = res_ens.fun

print(f"Optimized Blend Weights:")
print(f"  LightGBM:  {best_weights[0]*100:.1f}%")
print(f"  CatBoost:  {best_weights[1]*100:.1f}%")
print(f"  Logistic:  {best_weights[2]*100:.1f}%")
print(f"  Heuristic: {best_weights[3]*100:.1f}%")
print(f"\n5-Fold OOF Ensemble Blend Log Loss: {loss_ensemble:.5f} (Total gain over Part IX = {loss_part_ix - loss_ensemble:+.5f})")

print("\n" + "=" * 75)
print("EXPERIMENT BENCHMARK SUMMARY TABLE")
print("=" * 75)
print(f"{'Method / Experiment':<35} | {'Log Loss':<10} | {'Gain vs Prior':<12}")
print("-" * 75)
print(f"{'1. Uniform Prior [1/3, 1/3, 1/3]':<35} | {1.09861:<10.5f} | {'Baseline':<12}")
print(f"{'2. Empirical Class Prior':<35} | {loss_prior:<10.5f} | {1.09861 - loss_prior:+.5f}")
print(f"{'3. Part IX Calibrated Non-ML':<35} | {loss_part_ix:<10.5f} | {loss_prior - loss_part_ix:+.5f}")
print(f"{'4. Calibrated Logistic Regression':<35} | {loss_lr:<10.5f} | {loss_prior - loss_lr:+.5f}")
print(f"{'5. CatBoost MultiClass':<35} | {loss_cb:<10.5f} | {loss_prior - loss_cb:+.5f}")
print(f"{'6. LightGBM MultiClass GBDT':<35} | {loss_lgb:<10.5f} | {loss_prior - loss_lgb:+.5f}")
print(f"{'7. Blended GBDT + Heuristic Ensemble':<35} | {loss_ensemble:<10.5f} | {loss_prior - loss_ensemble:+.5f}")
print("=" * 75)
print(f"Total benchmark completed in {time.time() - t0:.2f}s!")

# Save OOF predictions and models for submission script
np.save('/root/llm_classification_finetuning/oof_ensemble.npy', (best_weights[0]*oof_lgb + best_weights[1]*oof_cb + best_weights[2]*oof_lr + best_weights[3]*oof_part_ix))
np.save('/root/llm_classification_finetuning/best_weights.npy', best_weights)
