import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import time

start_time = time.time()
print("=== Experiment 1: Regularized Linear Model (Ridge) ===")

# 1. Load data (2017 slice for fast, lightweight edge execution)
print("Loading datasets...")
train = pd.read_csv('/root/kaggle_store_sales/train.csv', parse_dates=['date'])
test = pd.read_csv('/root/kaggle_store_sales/test.csv', parse_dates=['date'])

# Filter train to 2017 to conserve RAM and avoid historical structural breaks
train_2017 = train[train['date'] >= '2017-01-01'].copy()

# Log1p target transform
train_2017['target'] = np.log1p(train_2017['sales'])

# 2. Build calendar features
for df in [train_2017, test]:
    df['dayofweek'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# 3. Create historical 16-day and 28-day lag priors per (store_nbr, family, dayofweek)
# We use 2017-07-05 to 2017-08-15 as historical prior
prior_42d = train_2017[(train_2017['date'] >= '2017-07-05') & (train_2017['date'] <= '2017-08-15')].copy()
prior_mean = prior_42d.groupby(['store_nbr', 'family', 'dayofweek'])['target'].mean().reset_index()
prior_mean.rename(columns={'target': 'prior_log_sales'}, inplace=True)

train_2017 = train_2017.merge(prior_mean, on=['store_nbr', 'family', 'dayofweek'], how='left')
train_2017['prior_log_sales'] = train_2017['prior_log_sales'].fillna(0.0)

test = test.merge(prior_mean, on=['store_nbr', 'family', 'dayofweek'], how='left')
test['prior_log_sales'] = test['prior_log_sales'].fillna(0.0)

# Validation slice: 2017-08-01 to 2017-08-15 (15 days mirroring test)
val_mask = train_2017['date'] >= '2017-08-01'
tr_mask = (train_2017['date'] >= '2017-03-01') & (train_2017['date'] < '2017-08-01')

# One-hot encode store_nbr and family
cat_cols = ['store_nbr', 'family', 'dayofweek']
num_cols = ['onpromotion', 'is_weekend', 'day', 'prior_log_sales']

X_all = pd.get_dummies(pd.concat([train_2017[cat_cols + num_cols], test[cat_cols + num_cols]]), columns=cat_cols, drop_first=True)
n_train = len(train_2017)
X_train_full = X_all.iloc[:n_train]
X_test = X_all.iloc[n_train:]

X_tr = X_train_full[tr_mask]
y_tr = train_2017.loc[tr_mask, 'target']
X_val = X_train_full[val_mask]
y_val = train_2017.loc[val_mask, 'target']

print(f"Train rows: {len(X_tr):,}, Val rows: {len(X_val):,}, Features: {X_tr.shape[1]}")

# 4. Fit Ridge
ridge = Ridge(alpha=10.0)
ridge.fit(X_tr, y_tr)

val_preds = np.expm1(np.clip(ridge.predict(X_val), 0, None))
val_actual = train_2017.loc[val_mask, 'sales'].values
val_rmsle = np.sqrt(np.mean((np.log1p(val_preds) - np.log1p(val_actual)) ** 2))
print(f"Validation RMSLE (2017-08-01 to 2017-08-15): {val_rmsle:.5f}")

# 5. Fit on full recent training and predict test
ridge_full = Ridge(alpha=10.0)
X_tr_full = X_train_full[train_2017['date'] >= '2017-03-01']
y_tr_full = train_2017.loc[train_2017['date'] >= '2017-03-01', 'target']
ridge_full.fit(X_tr_full, y_tr_full)

test_preds = np.expm1(np.clip(ridge_full.predict(X_test), 0, None))

sub = test[['id']].copy()
sub['sales'] = test_preds
sub.to_csv('/root/kaggle_store_sales/submission_exp1_ridge.csv', index=False)
print(f"Submission saved to submission_exp1_ridge.csv. Elapsed: {time.time()-start_time:.1f}s")
