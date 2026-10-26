import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import time

start_time = time.time()
print("=== Experiment 3: CatBoost GBDT with Exogenous Features (Holidays, Oil, Paydays) ===")

# 1. Load Data
print("Loading datasets...")
train = pd.read_csv('/root/kaggle_store_sales/train.csv', parse_dates=['date'])
test = pd.read_csv('/root/kaggle_store_sales/test.csv', parse_dates=['date'])
stores = pd.read_csv('/root/kaggle_store_sales/stores.csv')
oil = pd.read_csv('/root/kaggle_store_sales/oil.csv', parse_dates=['date'])
holidays = pd.read_csv('/root/kaggle_store_sales/holidays_events.csv', parse_dates=['date'])

# Restrict train to 2017 to conserve RAM and avoid regime shifts
train_2017 = train[train['date'] >= '2017-01-01'].copy()
train_2017['target'] = np.log1p(train_2017['sales'])

# Merge store metadata
train_2017 = train_2017.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')

# 2. Exogenous Feature Engineering

# A. Oil Price Processing
# Create a full date range to interpolate missing oil prices
all_dates = pd.date_range(start='2017-01-01', end='2017-08-31', freq='D')
oil_full = pd.DataFrame({'date': all_dates})
oil_full = oil_full.merge(oil, on='date', how='left')
oil_full['oil_price'] = oil_full['dcoilwtico'].ffill().bfill()
oil_full['oil_roll7'] = oil_full['oil_price'].rolling(7, min_periods=1).mean()
oil_full['oil_diff7'] = oil_full['oil_price'] - oil_full['oil_roll7']
oil_features = oil_full[['date', 'oil_price', 'oil_roll7', 'oil_diff7']]

# B. Holidays Processing
# In Ecuador, transferred holidays are moved to a transfer date. Only active (not transferred) holidays count.
active_holidays = holidays[holidays['transferred'] == False].copy()

# National holidays apply to all stores
nat_holidays = active_holidays[active_holidays['locale'] == 'National'].drop_duplicates(subset=['date'])
nat_dates = set(nat_holidays['date'])

# Regional and Local holidays
reg_holidays = active_holidays[active_holidays['locale'] == 'Regional'].copy()
loc_holidays = active_holidays[active_holidays['locale'] == 'Local'].copy()

# Combine train and test for uniform feature engineering
df_all = pd.concat([
    train_2017[['date', 'store_nbr', 'family', 'target', 'onpromotion', 'city', 'state', 'type', 'cluster']], 
    test[['date', 'store_nbr', 'family', 'onpromotion', 'city', 'state', 'type', 'cluster']]
], ignore_index=True)

# Merge oil
df_all = df_all.merge(oil_features, on='date', how='left')

# Holiday flags
df_all['is_national_holiday'] = df_all['date'].isin(nat_dates).astype(int)

# Match regional holidays by state
reg_sub = reg_holidays[['date', 'locale_name']].drop_duplicates().rename(columns={'locale_name': 'state'})
reg_sub['is_regional_holiday'] = 1
df_all = df_all.merge(reg_sub, on=['date', 'state'], how='left')
df_all['is_regional_holiday'] = df_all['is_regional_holiday'].fillna(0).astype(int)

# Match local holidays by city
loc_sub = loc_holidays[['date', 'locale_name']].drop_duplicates().rename(columns={'locale_name': 'city'})
loc_sub['is_local_holiday'] = 1
df_all = df_all.merge(loc_sub, on=['date', 'city'], how='left')
df_all['is_local_holiday'] = df_all['is_local_holiday'].fillna(0).astype(int)

df_all['is_any_holiday'] = ((df_all['is_national_holiday'] == 1) | 
                            (df_all['is_regional_holiday'] == 1) | 
                            (df_all['is_local_holiday'] == 1)).astype(int)

# C. Payday (Quincena) Feature
# In Ecuador wages are paid on the 15th and the end of the month
df_all['day'] = df_all['date'].dt.day
df_all['dayofweek'] = df_all['date'].dt.dayofweek
df_all['is_weekend'] = (df_all['dayofweek'] >= 5).astype(int)
df_all['is_payday'] = ((df_all['day'] == 15) | (df_all['date'].dt.is_month_end)).astype(int)
df_all['dist_to_payday'] = np.minimum(np.abs(df_all['day'] - 15), np.minimum(df_all['day'] - 1, 31 - df_all['day']))

# Sort for sequential rolling features
df_all.sort_values(['store_nbr', 'family', 'date'], inplace=True)
df_all.reset_index(drop=True, inplace=True)

print("Constructing lag and rolling features (horizon >= 16 days)...")
for lag in [16, 21, 28]:
    df_all[f'lag_{lag}'] = df_all.groupby(['store_nbr', 'family'])['target'].shift(lag)

grouped = df_all.groupby(['store_nbr', 'family'])['lag_16']
df_all['rolling_mean_7'] = grouped.transform(lambda x: x.rolling(7, min_periods=1).mean())
df_all['rolling_mean_14'] = grouped.transform(lambda x: x.rolling(14, min_periods=1).mean())
df_all['rolling_std_7'] = grouped.transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0.0)

# Fill NaNs
num_features = [
    'onpromotion', 'is_weekend', 'day', 'dayofweek', 'is_payday', 'dist_to_payday',
    'is_national_holiday', 'is_regional_holiday', 'is_local_holiday', 'is_any_holiday',
    'oil_price', 'oil_roll7', 'oil_diff7',
    'lag_16', 'lag_21', 'lag_28', 'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7'
]
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
    iterations=400,
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
best_iters = cb_val.get_best_iteration() or 380
cb_full = CatBoostRegressor(
    iterations=best_iters,
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

sub[['id', 'sales']].to_csv('/root/kaggle_store_sales/submission_exp3_catboost_exogenous.csv', index=False)
print(f"Submission saved to submission_exp3_catboost_exogenous.csv. Total time: {time.time()-start_time:.1f}s")
