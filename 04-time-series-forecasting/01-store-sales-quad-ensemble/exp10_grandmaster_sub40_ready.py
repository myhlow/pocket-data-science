import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import time

start_time = time.time()
print("=" * 75)
print("=== Experiment 10: Local Stacking Pipeline & Sub-0.40 Submission Prep ===")
print("=" * 75)

# 1. Load Data
print("\n[Step 1/5] Ingesting training and test sets...")
t0 = time.time()
train = pd.read_csv('/root/kaggle_store_sales/train.csv', parse_dates=['date'])
test = pd.read_csv('/root/kaggle_store_sales/test.csv', parse_dates=['date'])
stores = pd.read_csv('/root/kaggle_store_sales/stores.csv')
oil = pd.read_csv('/root/kaggle_store_sales/oil.csv', parse_dates=['date'])
holidays = pd.read_csv('/root/kaggle_store_sales/holidays_events.csv', parse_dates=['date'])

train_2017 = train[train['date'] >= '2017-01-01'].copy()
train_2017 = train_2017[~((train_2017['store_nbr'] == 52) & (train_2017['date'] < '2017-04-20'))].copy()
train_2017['target'] = np.log1p(train_2017['sales'])

train_2017 = train_2017.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')
print(f"Data loaded in {time.time()-t0:.1f}s. Train rows: {len(train_2017):,}, Test rows: {len(test):,}")

# 2. Exogenous Engineering
print("\n[Step 2/5] Constructing calendar, oil, and promotional feature sets...")
t0 = time.time()
all_dates = pd.date_range(start='2017-01-01', end='2017-08-31', freq='D')
oil_full = pd.DataFrame({'date': all_dates}).merge(oil, on='date', how='left')
oil_full['oil_price'] = oil_full['dcoilwtico'].ffill().bfill().astype('float32')
oil_full['oil_roll7'] = oil_full['oil_price'].rolling(7, min_periods=1).mean().astype('float32')
oil_full['oil_diff7'] = (oil_full['oil_price'] - oil_full['oil_roll7']).astype('float32')
oil_features = oil_full[['date', 'oil_price', 'oil_roll7', 'oil_diff7']]

active_holidays = holidays[holidays['transferred'] == False].copy()
nat_dates = set(active_holidays[active_holidays['locale'] == 'National']['date'])
reg_sub = active_holidays[active_holidays['locale'] == 'Regional'][['date', 'locale_name']].drop_duplicates().rename(columns={'locale_name': 'state'})
reg_sub['is_reg_holiday'] = 1
loc_sub = active_holidays[active_holidays['locale'] == 'Local'][['date', 'locale_name']].drop_duplicates().rename(columns={'locale_name': 'city'})
loc_sub['is_loc_holiday'] = 1

df_all = pd.concat([
    train_2017[['date', 'store_nbr', 'family', 'target', 'sales', 'onpromotion', 'city', 'state', 'type', 'cluster']],
    test[['date', 'store_nbr', 'family', 'onpromotion', 'city', 'state', 'type', 'cluster']]
], ignore_index=True)

df_all = df_all.merge(oil_features, on='date', how='left')
df_all['is_nat_holiday'] = df_all['date'].isin(nat_dates).astype('int8')
df_all = df_all.merge(reg_sub, on=['date', 'state'], how='left').fillna({'is_reg_holiday': 0})
df_all['is_reg_holiday'] = df_all['is_reg_holiday'].astype('int8')
df_all = df_all.merge(loc_sub, on=['date', 'city'], how='left').fillna({'is_loc_holiday': 0})
df_all['is_loc_holiday'] = df_all['is_loc_holiday'].astype('int8')
df_all['is_holiday'] = ((df_all['is_nat_holiday'] == 1) | (df_all['is_reg_holiday'] == 1) | (df_all['is_loc_holiday'] == 1)).astype('int8')

df_all['day'] = df_all['date'].dt.day.astype('int8')
df_all['dayofweek'] = df_all['date'].dt.dayofweek.astype('int8')
df_all['is_weekend'] = (df_all['dayofweek'] >= 5).astype('int8')
df_all['is_payday'] = ((df_all['day'] == 15) | (df_all['date'].dt.is_month_end)).astype('int8')
df_all['dist_to_payday'] = np.minimum(np.abs(df_all['day'] - 15), np.minimum(df_all['day'] - 1, 31 - df_all['day'])).astype('int8')
df_all['is_payday_window'] = df_all['day'].isin([14, 15, 16, 17, 30, 31]).astype('int8')

family_avg_promo = df_all.groupby('family')['onpromotion'].transform('mean').clip(lower=1e-3).astype('float32')
df_all['promo_ratio'] = (df_all['onpromotion'] / family_avg_promo).astype('float32')
df_all['is_promoted'] = (df_all['onpromotion'] > 0).astype('int8')

df_all.sort_values(['store_nbr', 'family', 'date'], inplace=True)
df_all.reset_index(drop=True, inplace=True)

df_all['promo_lag_16'] = df_all.groupby(['store_nbr', 'family'])['onpromotion'].shift(16).fillna(0).astype('int16')
df_all['promo_delta_16'] = (df_all['onpromotion'] - df_all['promo_lag_16']).astype('float32')

for lag in [16, 21, 28]:
    df_all[f'lag_{lag}'] = df_all.groupby(['store_nbr', 'family'])['target'].shift(lag).astype('float32')

df_all['roll_mean_7_l16'] = df_all.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(7, min_periods=1).mean()).astype('float32')
df_all['roll_mean_14_l16'] = df_all.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(14, min_periods=1).mean()).astype('float32')

cat_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster']
features = cat_cols + [
    'onpromotion', 'promo_ratio', 'is_promoted', 'promo_delta_16',
    'is_weekend', 'day', 'dayofweek', 'is_payday', 'dist_to_payday', 'is_payday_window',
    'is_nat_holiday', 'is_reg_holiday', 'is_loc_holiday', 'is_holiday',
    'oil_price', 'oil_roll7', 'oil_diff7',
    'lag_16', 'lag_21', 'lag_28', 'roll_mean_7_l16', 'roll_mean_14_l16'
]

# Separate dataframes for categorical handling
df_cat = df_all.copy()
for c in cat_cols:
    df_cat[c] = df_cat[c].astype('category')
for col in features:
    if col not in cat_cols:
        df_cat[col] = df_cat[col].fillna(0.0)

df_str = df_all.copy()
for c in cat_cols:
    df_str[c] = df_str[c].astype(str)
for col in features:
    if col not in cat_cols:
        df_str[col] = df_str[col].fillna(0.0)

train_cat = df_cat[(df_cat['date'] >= '2017-02-01') & (df_cat['date'] <= '2017-08-15')].copy()
test_cat = df_cat[df_cat['date'] >= '2017-08-16'].copy()

train_str = df_str[(df_str['date'] >= '2017-02-01') & (df_str['date'] <= '2017-08-15')].copy()
test_str = df_str[df_str['date'] >= '2017-08-16'].copy()

print(f"Features ready in {time.time()-t0:.1f}s. Training rows: {len(train_cat):,}, Features: {len(features)}")

# 3. Fit Models on Full Training Data
print("\n[Step 3/5] Training Tri-GBDT models on complete training dataset...")

# LightGBM
print(">>> Training LightGBM Regressor (650 trees)...")
t_m = time.time()
m_lgb = lgb.LGBMRegressor(
    n_estimators=650,
    learning_rate=0.06,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=40,
    random_state=42,
    n_jobs=4,
    verbose=-1
)
m_lgb.fit(train_cat[features], train_cat['target'])
pred_lgb = np.clip(m_lgb.predict(test_cat[features]), 0, None)
print(f"LightGBM trained in {time.time()-t_m:.1f}s.")

# XGBoost
print(">>> Training XGBoost Regressor (650 trees)...")
t_m = time.time()
m_xgb = xgb.XGBRegressor(
    n_estimators=650,
    learning_rate=0.06,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True,
    tree_method='hist',
    random_state=42,
    n_jobs=4
)
m_xgb.fit(train_cat[features], train_cat['target'])
pred_xgb = np.clip(m_xgb.predict(test_cat[features]), 0, None)
print(f"XGBoost trained in {time.time()-t_m:.1f}s.")

# CatBoost
print(">>> Training CatBoost Regressor (550 iterations)...")
t_m = time.time()
m_cb = CatBoostRegressor(
    iterations=550,
    learning_rate=0.07,
    depth=6,
    cat_features=cat_cols,
    random_seed=42,
    verbose=0,
    thread_count=4
)
m_cb.fit(train_str[features], train_str['target'])
pred_cb = np.clip(m_cb.predict(test_str[features]), 0, None)
print(f"CatBoost trained in {time.time()-t_m:.1f}s.")

# 4. Align Component Predictions to test.csv ID
print("\n[Step 4/5] Aligning predictions and applying dead-series guardrails...")
test_orig = pd.read_csv('/root/kaggle_store_sales/test.csv')

# Load Exp 5 Base (0.41072 Kaggle benchmark)
sub_exp5 = pd.read_csv('/root/kaggle_store_sales/submission_exp5_ensemble_blend.csv')
log_exp5 = np.log1p(sub_exp5['sales'].values)

test_cat['pred_lgb'] = pred_lgb
test_cat['pred_xgb'] = pred_xgb
test_str['pred_cb'] = pred_cb

test_cat['store_nbr'] = test_cat['store_nbr'].astype(int)
test_cat['date'] = test_cat['date'].astype(str)
test_str['store_nbr'] = test_str['store_nbr'].astype(int)
test_str['date'] = test_str['date'].astype(str)

merged_preds = test_orig[['id', 'store_nbr', 'family', 'date']].merge(
    test_cat[['store_nbr', 'family', 'date', 'pred_lgb', 'pred_xgb']],
    on=['store_nbr', 'family', 'date'],
    how='left'
).merge(
    test_str[['store_nbr', 'family', 'date', 'pred_cb']],
    on=['store_nbr', 'family', 'date'],
    how='left'
)

log_lgb = merged_preds['pred_lgb'].values
log_xgb = merged_preds['pred_xgb'].values
log_cb = merged_preds['pred_cb'].values

# Dead series mask
recent_14d = train_cat[train_cat['date'] >= '2017-08-02']
sales_recent = recent_14d.groupby(['store_nbr', 'family'], observed=False)['sales'].sum().reset_index()
dead_pairs = set(sales_recent[sales_recent['sales'] == 0][['store_nbr', 'family']].itertuples(index=False, name=None))
test_dead_mask = test_orig.apply(lambda r: (int(r['store_nbr']), r['family']) in dead_pairs and r['onpromotion'] == 0, axis=1)
print(f"Identified {len(dead_pairs)} dead store-family pairs ({test_dead_mask.sum():,} test rows).")

# 5. Build Grandmaster Stacking Ensemble
print("\n[Step 5/5] Generating Grandmaster Sub-0.40 Candidate Submissions...")

# Ensemble Configuration:
# 60% Exp 5 Domain Base + 25% LightGBM + 10% XGBoost + 5% CatBoost
log_ensemble = 0.60 * log_exp5 + 0.25 * log_lgb + 0.10 * log_xgb + 0.05 * log_cb
sales_ensemble = np.clip(np.expm1(log_ensemble), 0, None)
sales_ensemble[test_dead_mask.values] = 0.0

sub_exp10 = pd.DataFrame({'id': test_orig['id'], 'sales': sales_ensemble})
exp10_path = '/root/kaggle_store_sales/submission_exp10_grandmaster_sub40_ready.csv'
sub_exp10.to_csv(exp10_path, index=False)
print(f"✓ Saved Experiment 10 Submission: {exp10_path}")
print(f"Exp 10 Mean sales: {sub_exp10['sales'].mean():.2f}")

# Sanity checks
assert len(sub_exp10) == 28512
assert not sub_exp10['sales'].isnull().any()
assert (sub_exp10['sales'] >= 0).all()
assert (sub_exp10.loc[test_dead_mask.values, 'sales'] == 0.0).all()
print(f"✓ Sanity verification passed: rows={len(sub_exp10):,}, zeros={(sub_exp10['sales']==0).sum():,}, min={sub_exp10['sales'].min():.2f}, max={sub_exp10['sales'].max():.2f}, mean={sub_exp10['sales'].mean():.2f}")

total_elapsed = time.time() - start_time
print("=" * 75)
print(f"Experiment 10 pipeline complete in {total_elapsed:.1f}s ({total_elapsed/60:.2f} min).")
print("=" * 75)
