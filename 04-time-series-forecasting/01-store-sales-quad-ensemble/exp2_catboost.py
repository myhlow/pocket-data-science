import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import time

start_time = time.time()
print("=== Experiment 2: Native Categorical CatBoost GBDT ===")

# 1. Load Data
print("Loading datasets...")
train = pd.read_csv('/root/kaggle_store_sales/train.csv', parse_dates=['date'])
test = pd.read_csv('/root/kaggle_store_sales/test.csv', parse_dates=['date'])
stores = pd.read_csv('/root/kaggle_store_sales/stores.csv')

# Restrict train to 2017 to conserve RAM and avoid pre-2016 regime shifts
train_2017 = train[train['date'] >= '2017-01-01'].copy()
train_2017['target'] = np.log1p(train_2017['sales'])

# Merge store metadata
train_2017 = train_2017.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')

# 2. Combine to compute leakage-free lag and calendar features
df_all = pd.concat([train_2017[['date', 'store_nbr', 'family', 'target', 'onpromotion', 'city', 'state', 'type', 'cluster']], 
                    test[['date', 'store_nbr', 'family', 'onpromotion', 'city', 'state', 'type', 'cluster']]], 
                   ignore_index=True)

df_all['dayofweek'] = df_all['date'].dt.dayofweek
df_all['day'] = df_all['date'].dt.day
df_all['is_weekend'] = (df_all['dayofweek'] >= 5).astype(int)

# Sort for sequential rolling features
df_all.sort_values(['store_nbr', 'family', 'date'], inplace=True)
df_all.reset_index(drop=True, inplace=True)

print("Constructing lag features (horizon >= 16 days)...")
# Since the test set is 16 days, we create lag 16, 21, 28
for lag in [16, 21, 28]:
    df_all[f'lag_{lag}'] = df_all.groupby(['store_nbr', 'family'])['target'].shift(lag)

# Rolling stats on lag 16
grouped = df_all.groupby(['store_nbr', 'family'])['lag_16']
df_all['rolling_mean_7'] = grouped.transform(lambda x: x.rolling(7, min_periods=1).mean())
df_all['rolling_mean_14'] = grouped.transform(lambda x: x.rolling(14, min_periods=1).mean())

# Fill NaNs for early dates
num_features = ['onpromotion', 'is_weekend', 'day', 'dayofweek', 'lag_16', 'lag_21', 'lag_28', 'rolling_mean_7', 'rolling_mean_14']
for col in num_features:
    df_all[col] = df_all[col].fillna(0.0)

cat_features = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster']
for col in cat_features:
    df_all[col] = df_all[col].astype(str)

features = cat_features + num_features
print(f"Total features: {len(features)} ({len(cat_features)} categoricals, {len(num_features)} numericals)")

# Separate back into train and test
train_df = df_all[df_all['date'] <= '2017-08-15'].copy()
test_df = df_all[df_all['date'] >= '2017-08-16'].copy()

# Filter to where lag_28 is available (from 2017-02-01 onwards)
train_df = train_df[train_df['date'] >= '2017-02-01'].copy()

# Validation split (2017-08-01 to 2017-08-15)
val_mask = train_df['date'] >= '2017-08-01'
tr_mask = train_df['date'] < '2017-08-01'

X_tr = train_df.loc[tr_mask, features]
y_tr = train_df.loc[tr_mask, 'target']
X_val = train_df.loc[val_mask, features]
y_val = train_df.loc[val_mask, 'target']

print(f"Training rows: {len(X_tr):,}, Validation rows: {len(X_val):,}")

# 3. Train CatBoost
cb_val = CatBoostRegressor(
    iterations=350,
    learning_rate=0.08,
    depth=6,
    loss_function='RMSE',
    cat_features=cat_features,
    random_seed=42,
    verbose=50,
    thread_count=4
)

print("Fitting CatBoost on validation split...")
cb_val.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=40)

val_preds = np.expm1(np.clip(cb_val.predict(X_val), 0, None))
val_actual = np.expm1(y_val.values)
val_rmsle = np.sqrt(np.mean((np.log1p(val_preds) - np.log1p(val_actual)) ** 2))
print(f"\n--> Validation RMSLE: {val_rmsle:.5f}")

# 4. Retrain on full dataset and predict test
print("\nRetraining on full training set (through 2017-08-15)...")
cb_full = CatBoostRegressor(
    iterations=cb_val.get_best_iteration() or 350,
    learning_rate=0.08,
    depth=6,
    loss_function='RMSE',
    cat_features=cat_features,
    random_seed=42,
    verbose=0,
    thread_count=4
)
cb_full.fit(train_df[features], train_df['target'])

# Predict test
test_preds = np.expm1(np.clip(cb_full.predict(test_df[features]), 0, None))

# Merge with test ID to preserve exact submission order
test_orig = pd.read_csv('/root/kaggle_store_sales/test.csv')
test_df['sales'] = test_preds
test_df['store_nbr'] = test_df['store_nbr'].astype(int)
test_df['date'] = test_df['date'].astype(str)
sub = test_orig[['id', 'store_nbr', 'family', 'date']].merge(
    test_df[['store_nbr', 'family', 'date', 'sales']],
    on=['store_nbr', 'family', 'date'],
    how='left'
)

sub[['id', 'sales']].to_csv('/root/kaggle_store_sales/submission_exp2_catboost.csv', index=False)
print(f"Submission saved to submission_exp2_catboost.csv. Total time: {time.time()-start_time:.1f}s")
