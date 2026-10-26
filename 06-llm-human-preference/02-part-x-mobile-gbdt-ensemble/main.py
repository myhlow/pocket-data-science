"""
Pocket Data Science Part X: Mobile GBDT Machine Learning for LMSYS Preference
Team: Pocket Data Lab
Competition: Kaggle LLM Classification Finetuning

Master Learning Path:
https://malcolmlow.com/2026/10/26/pocket-data-science-complete-learning-path-mobile-kaggle-roadmap/

Mathematical Foundations (Log Loss & Calibration):
https://malcolmlow.com/2026/10/26/multiclass-log-loss-explained-cross-entropy-mle-python/
"""

import os
import glob
import time
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

t_start = time.time()
print("Starting Pocket Data Science Part X GBDT Pipeline...")

# 1. Resolve paths dynamically (local vs Kaggle)
train_paths = glob.glob('/kaggle/input/**/train.csv', recursive=True)
test_paths = glob.glob('/kaggle/input/**/test.csv', recursive=True)

train_file = train_paths[0] if train_paths else '/root/llm_classification_finetuning/train.csv'
test_file = test_paths[0] if test_paths else '/root/llm_classification_finetuning/test.csv'

print(f"Loading train dataset from: {train_file}")
train_cols = ['id', 'prompt', 'response_a', 'response_b', 'winner_model_a', 'winner_model_b', 'winner_tie']
train_df = pd.read_csv(train_file, usecols=train_cols)
test_df = pd.read_csv(test_file)

y_train = np.argmax(train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].values, axis=1)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# 2. Vectorized Feature Extraction Function
def extract_features(df):
    prompt = df['prompt'].fillna('').astype(str)
    ra = df['response_a'].fillna('').astype(str)
    rb = df['response_b'].fillna('').astype(str)

    len_p = prompt.str.len().values.astype(np.float32)
    len_a = ra.str.len().values.astype(np.float32)
    len_b = rb.str.len().values.astype(np.float32)

    len_diff = len_a - len_b
    len_ratio = (len_a - len_b) / (len_a + len_b + 1.0)
    len_ratio_p_a = len_a / (len_p + 1.0)
    len_ratio_p_b = len_b / (len_p + 1.0)
    len_ratio_p_diff = len_ratio_p_a - len_ratio_p_b

    words_p = prompt.str.split().str.len().values.astype(np.float32)
    words_a = ra.str.split().str.len().values.astype(np.float32)
    words_b = rb.str.split().str.len().values.astype(np.float32)

    words_diff = words_a - words_b
    words_ratio = (words_a - words_b) / (words_a + words_b + 1.0)

    mean_word_len_a = len_a / (words_a + 1.0)
    mean_word_len_b = len_b / (words_b + 1.0)
    mean_word_len_diff = mean_word_len_a - mean_word_len_b

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

    # Prompt overlap
    prompts = [set(x.lower().split()) for x in prompt]
    ras = [set(x.lower().split()) for x in ra]
    rbs = [set(x.lower().split()) for x in rb]

    n = len(df)
    jaccard_a = np.zeros(n, dtype=np.float32)
    jaccard_b = np.zeros(n, dtype=np.float32)
    cov_a = np.zeros(n, dtype=np.float32)
    cov_b = np.zeros(n, dtype=np.float32)

    for i in range(n):
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
    return pd.DataFrame(feature_dict)

print("Extracting training features...")
X_train = extract_features(train_df)
print("Extracting test features...")
X_test = extract_features(test_df)

del train_df
gc.collect()

# 3. Model Training
print("Training LightGBM MultiClass GBDT...")
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    metric='multi_logloss',
    boosting_type='gbdt',
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=160,
    random_state=42,
    verbose=-1,
    n_jobs=4
)
lgb_model.fit(X_train, y_train)
p_lgb_test = lgb_model.predict_proba(X_test)

print("Training CatBoost MultiClass Classifier...")
cb_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='MultiClass',
    iterations=300,
    learning_rate=0.06,
    depth=5,
    random_seed=42,
    verbose=0,
    thread_count=4
)
cb_model.fit(X_train, y_train)
p_cb_test = cb_model.predict_proba(X_test)

# 4. Ensemble Blending (50.6% LightGBM + 49.4% CatBoost)
p_ensemble = 0.506 * p_lgb_test + 0.494 * p_cb_test

# Epsilon clipping safety for multi-class log loss
eps = 1e-15
p_ensemble = np.clip(p_ensemble, eps, 1.0 - eps)
p_ensemble = p_ensemble / np.sum(p_ensemble, axis=1, keepdims=True)

# 5. Output Submission
sub = pd.DataFrame({
    'id': test_df['id'],
    'winner_model_a': p_ensemble[:, 0],
    'winner_model_b': p_ensemble[:, 1],
    'winner_tie': p_ensemble[:, 2]
})
sub.to_csv('submission.csv', index=False)
print(f"Generated submission.csv with shape {sub.shape} in {time.time() - t_start:.2f}s!")
print(sub.head())
