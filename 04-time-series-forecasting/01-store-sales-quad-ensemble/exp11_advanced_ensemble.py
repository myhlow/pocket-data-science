import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
import time
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()
print("=" * 80)
print("=== Experiment 11: Transaction-Enriched Tri-GBDT with Validation-Tuned Ensemble ===")
print("=" * 80)

# ── 1. Load All Data Sources ──────────────────────────────────────────────────
print("\n[Step 1/7] Ingesting all data sources...")
t0 = time.time()

train = pd.read_csv('/root/kaggle_store_sales/train.csv', parse_dates=['date'])
test  = pd.read_csv('/root/kaggle_store_sales/test.csv',  parse_dates=['date'])
stores = pd.read_csv('/root/kaggle_store_sales/stores.csv')
oil = pd.read_csv('/root/kaggle_store_sales/oil.csv', parse_dates=['date'])
holidays = pd.read_csv('/root/kaggle_store_sales/holidays_events.csv', parse_dates=['date'])
transactions = pd.read_csv('/root/kaggle_store_sales/transactions.csv', parse_dates=['date'])

# Use 2016-2017 data for richer lag features (instead of just 2017)
train_cut = train[train['date'] >= '2016-06-01'].copy()
# Exclude store 52 anomaly
train_cut = train_cut[~((train_cut['store_nbr'] == 52) & (train_cut['date'] < '2017-04-20'))].copy()
train_cut['target'] = np.log1p(train_cut['sales'].clip(lower=0))

train_cut = train_cut.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')
print(f"Data loaded in {time.time()-t0:.1f}s. Train rows: {len(train_cut):,}, Test rows: {len(test):,}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
print("\n[Step 2/7] Building enriched feature set (transactions + calendar + oil + promo)...")
t0 = time.time()

# 2a. Oil features
all_dates = pd.date_range(start='2016-06-01', end='2017-08-31', freq='D')
oil_full = pd.DataFrame({'date': all_dates}).merge(oil, on='date', how='left')
oil_full['oil_price'] = oil_full['dcoilwtico'].ffill().bfill().astype('float32')
oil_full['oil_roll7']  = oil_full['oil_price'].rolling(7,  min_periods=1).mean().astype('float32')
oil_full['oil_roll28'] = oil_full['oil_price'].rolling(28, min_periods=1).mean().astype('float32')
oil_full['oil_diff7']  = (oil_full['oil_price'] - oil_full['oil_roll7']).astype('float32')
oil_full['oil_diff28'] = (oil_full['oil_price'] - oil_full['oil_roll28']).astype('float32')
oil_features = oil_full[['date', 'oil_price', 'oil_roll7', 'oil_roll28', 'oil_diff7', 'oil_diff28']]

# 2b. Holiday features (same as before, but also adding days-since-last-holiday)
active_holidays = holidays[holidays['transferred'] == False].copy()
nat_dates = set(active_holidays[active_holidays['locale'] == 'National']['date'])
reg_sub = active_holidays[active_holidays['locale'] == 'Regional'][['date', 'locale_name']].drop_duplicates().rename(columns={'locale_name': 'state'})
reg_sub['is_reg_holiday'] = 1
loc_sub = active_holidays[active_holidays['locale'] == 'Local'][['date', 'locale_name']].drop_duplicates().rename(columns={'locale_name': 'city'})
loc_sub['is_loc_holiday'] = 1

# 2c. Transaction features (NEW — store-level traffic signal)
txn_daily = transactions.copy()
txn_daily.sort_values(['store_nbr', 'date'], inplace=True)
txn_daily['txn_roll7']  = txn_daily.groupby('store_nbr')['transactions'].transform(lambda x: x.rolling(7,  min_periods=1).mean())
txn_daily['txn_roll14'] = txn_daily.groupby('store_nbr')['transactions'].transform(lambda x: x.rolling(14, min_periods=1).mean())
txn_daily['txn_roll28'] = txn_daily.groupby('store_nbr')['transactions'].transform(lambda x: x.rolling(28, min_periods=1).mean())
txn_daily['txn_ewm7']   = txn_daily.groupby('store_nbr')['transactions'].transform(lambda x: x.ewm(span=7).mean())
# Lag transaction features (shifted by 16 days for test-safe horizon)
for lag in [16, 21, 28]:
    txn_daily[f'txn_lag_{lag}'] = txn_daily.groupby('store_nbr')['transactions'].shift(lag)
txn_daily['txn_roll7_lag16'] = txn_daily.groupby('store_nbr')['txn_lag_16'].transform(lambda x: x.rolling(7, min_periods=1).mean())

txn_features = txn_daily[['date', 'store_nbr', 'transactions', 'txn_roll7', 'txn_roll14',
                           'txn_roll28', 'txn_ewm7', 'txn_lag_16', 'txn_lag_21', 'txn_lag_28',
                           'txn_roll7_lag16']].copy()

# 2d. Combine everything
df_all = pd.concat([
    train_cut[['date', 'store_nbr', 'family', 'target', 'sales', 'onpromotion', 'city', 'state', 'type', 'cluster']],
    test[['date', 'store_nbr', 'family', 'onpromotion', 'city', 'state', 'type', 'cluster']]
], ignore_index=True)

df_all = df_all.merge(oil_features, on='date', how='left')
df_all['is_nat_holiday'] = df_all['date'].isin(nat_dates).astype('int8')
df_all = df_all.merge(reg_sub, on=['date', 'state'], how='left').fillna({'is_reg_holiday': 0})
df_all['is_reg_holiday'] = df_all['is_reg_holiday'].astype('int8')
df_all = df_all.merge(loc_sub, on=['date', 'city'], how='left').fillna({'is_loc_holiday': 0})
df_all['is_loc_holiday'] = df_all['is_loc_holiday'].astype('int8')
df_all['is_holiday'] = ((df_all['is_nat_holiday'] == 1) | (df_all['is_reg_holiday'] == 1) | (df_all['is_loc_holiday'] == 1)).astype('int8')

# Merge transaction features
df_all = df_all.merge(txn_features, on=['date', 'store_nbr'], how='left')

# Calendar features
df_all['day'] = df_all['date'].dt.day.astype('int8')
df_all['dayofweek'] = df_all['date'].dt.dayofweek.astype('int8')
df_all['weekofyear'] = df_all['date'].dt.isocalendar().week.astype('int8')
df_all['month'] = df_all['date'].dt.month.astype('int8')
df_all['is_weekend'] = (df_all['dayofweek'] >= 5).astype('int8')
df_all['is_payday'] = ((df_all['day'] == 15) | (df_all['date'].dt.is_month_end)).astype('int8')
df_all['dist_to_payday'] = np.minimum(np.abs(df_all['day'] - 15), np.minimum(df_all['day'] - 1, 31 - df_all['day'])).astype('int8')
df_all['is_payday_window'] = df_all['day'].isin([14, 15, 16, 17, 30, 31]).astype('int8')

# Day of month sin/cos (capture cyclical patterns)
df_all['day_sin'] = np.sin(2 * np.pi * df_all['day'] / 31).astype('float32')
df_all['day_cos'] = np.cos(2 * np.pi * df_all['day'] / 31).astype('float32')
df_all['dow_sin'] = np.sin(2 * np.pi * df_all['dayofweek'] / 7).astype('float32')
df_all['dow_cos'] = np.cos(2 * np.pi * df_all['dayofweek'] / 7).astype('float32')

# Promotion features
family_avg_promo = df_all.groupby('family')['onpromotion'].transform('mean').clip(lower=1e-3).astype('float32')
df_all['promo_ratio'] = (df_all['onpromotion'] / family_avg_promo).astype('float32')
df_all['is_promoted'] = (df_all['onpromotion'] > 0).astype('int8')
store_family_avg_promo = df_all.groupby(['store_nbr', 'family'])['onpromotion'].transform('mean').clip(lower=1e-3).astype('float32')
df_all['promo_ratio_sf'] = (df_all['onpromotion'] / store_family_avg_promo).astype('float32')

# Sort for lag computation
df_all.sort_values(['store_nbr', 'family', 'date'], inplace=True)
df_all.reset_index(drop=True, inplace=True)

# Promotion lags
df_all['promo_lag_16'] = df_all.groupby(['store_nbr', 'family'])['onpromotion'].shift(16).fillna(0).astype('int16')
df_all['promo_delta_16'] = (df_all['onpromotion'] - df_all['promo_lag_16']).astype('float32')

# Target lags (extended set)
for lag in [16, 17, 18, 19, 20, 21, 28, 35]:
    df_all[f'lag_{lag}'] = df_all.groupby(['store_nbr', 'family'])['target'].shift(lag).astype('float32')

# Rolling statistics on lagged target (test-safe: starting from lag_16)
df_all['roll_mean_7_l16']  = df_all.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(7,  min_periods=1).mean()).astype('float32')
df_all['roll_mean_14_l16'] = df_all.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(14, min_periods=1).mean()).astype('float32')
df_all['roll_mean_28_l16'] = df_all.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(28, min_periods=1).mean()).astype('float32')
df_all['roll_std_7_l16']   = df_all.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(7,  min_periods=1).std()).fillna(0).astype('float32')
df_all['roll_std_14_l16']  = df_all.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(14, min_periods=1).std()).fillna(0).astype('float32')

# EWM on lag_16
df_all['ewm_7_l16']  = df_all.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.ewm(span=7).mean()).astype('float32')
df_all['ewm_14_l16'] = df_all.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.ewm(span=14).mean()).astype('float32')

# Lag difference features (momentum)
df_all['lag_diff_16_21'] = (df_all['lag_16'] - df_all['lag_21']).astype('float32')
df_all['lag_diff_16_28'] = (df_all['lag_16'] - df_all['lag_28']).astype('float32')
df_all['lag_diff_21_28'] = (df_all['lag_21'] - df_all['lag_28']).astype('float32')

# Mean of 16-20 week lags (5-day average for the same store-family)
df_all['lag_mean_16_20'] = df_all[['lag_16', 'lag_17', 'lag_18', 'lag_19', 'lag_20']].mean(axis=1).astype('float32')

# ── Feature list ──
cat_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster']

features = cat_cols + [
    # Promotion
    'onpromotion', 'promo_ratio', 'promo_ratio_sf', 'is_promoted', 'promo_delta_16',
    # Calendar
    'is_weekend', 'day', 'dayofweek', 'weekofyear', 'month',
    'is_payday', 'dist_to_payday', 'is_payday_window',
    'day_sin', 'day_cos', 'dow_sin', 'dow_cos',
    # Holidays
    'is_nat_holiday', 'is_reg_holiday', 'is_loc_holiday', 'is_holiday',
    # Oil
    'oil_price', 'oil_roll7', 'oil_roll28', 'oil_diff7', 'oil_diff28',
    # Transactions (NEW)
    'txn_lag_16', 'txn_lag_21', 'txn_lag_28', 'txn_roll7_lag16',
    # Target lags
    'lag_16', 'lag_17', 'lag_18', 'lag_19', 'lag_20', 'lag_21', 'lag_28', 'lag_35',
    # Rolling stats
    'roll_mean_7_l16', 'roll_mean_14_l16', 'roll_mean_28_l16',
    'roll_std_7_l16', 'roll_std_14_l16',
    # EWM
    'ewm_7_l16', 'ewm_14_l16',
    # Momentum
    'lag_diff_16_21', 'lag_diff_16_28', 'lag_diff_21_28',
    'lag_mean_16_20',
]

print(f"Feature set: {len(features)} features ({len(cat_cols)} categorical, {len(features)-len(cat_cols)} numeric)")

# Prepare two copies: one with category dtype, one with string dtype (for CatBoost)
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

# Training from 2017-02-01 (enough lag history) and validation mimicking the test horizon
train_cat  = df_cat[(df_cat['date'] >= '2017-02-01') & (df_cat['date'] <= '2017-07-31')].copy()
val_cat    = df_cat[(df_cat['date'] >= '2017-08-01') & (df_cat['date'] <= '2017-08-15')].copy()
test_cat   = df_cat[df_cat['date'] >= '2017-08-16'].copy()

train_str  = df_str[(df_str['date'] >= '2017-02-01') & (df_str['date'] <= '2017-07-31')].copy()
val_str    = df_str[(df_str['date'] >= '2017-08-01') & (df_str['date'] <= '2017-08-15')].copy()
test_str   = df_str[df_str['date'] >= '2017-08-16'].copy()

print(f"Features ready in {time.time()-t0:.1f}s.")
print(f"  Train: {len(train_cat):,} rows (2017-02-01 to 2017-07-31)")
print(f"  Val:   {len(val_cat):,} rows (2017-08-01 to 2017-08-15)")
print(f"  Test:  {len(test_cat):,} rows (2017-08-16 to 2017-08-31)")

# ── 3. Train Models with Early Stopping on Validation ────────────────────────
print("\n[Step 3/7] Training LightGBM with early stopping...")
t_m = time.time()

m_lgb = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_samples=30,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=4,
    verbose=-1
)
m_lgb.fit(
    train_cat[features], train_cat['target'],
    eval_set=[(val_cat[features], val_cat['target'])],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
)
pred_lgb_val  = np.clip(m_lgb.predict(val_cat[features]), 0, None)
pred_lgb_test = np.clip(m_lgb.predict(test_cat[features]), 0, None)
lgb_val_rmsle = np.sqrt(np.mean((pred_lgb_val - val_cat['target'].values)**2))
print(f"  LightGBM trained in {time.time()-t_m:.1f}s. Best iteration: {m_lgb.best_iteration_}. Val RMSLE: {lgb_val_rmsle:.5f}")

# ── 4. Train XGBoost ─────────────────────────────────────────────────────────
print("\n[Step 4/7] Training XGBoost with early stopping...")
t_m = time.time()

m_xgb = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=30,
    reg_alpha=0.1,
    reg_lambda=1.0,
    enable_categorical=True,
    tree_method='hist',
    random_state=42,
    n_jobs=4,
    early_stopping_rounds=50,
    eval_metric='rmse'
)
m_xgb.fit(
    train_cat[features], train_cat['target'],
    eval_set=[(val_cat[features], val_cat['target'])],
    verbose=False
)
pred_xgb_val  = np.clip(m_xgb.predict(val_cat[features]), 0, None)
pred_xgb_test = np.clip(m_xgb.predict(test_cat[features]), 0, None)
xgb_val_rmsle = np.sqrt(np.mean((pred_xgb_val - val_cat['target'].values)**2))
print(f"  XGBoost trained in {time.time()-t_m:.1f}s. Best iteration: {m_xgb.best_iteration}. Val RMSLE: {xgb_val_rmsle:.5f}")

# ── 5. Train CatBoost ────────────────────────────────────────────────────────
print("\n[Step 5/7] Training CatBoost with early stopping...")
t_m = time.time()

m_cb = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=3.0,
    cat_features=cat_cols,
    random_seed=42,
    verbose=0,
    thread_count=4,
    early_stopping_rounds=50,
    eval_metric='RMSE'
)
m_cb.fit(
    train_str[features], train_str['target'],
    eval_set=(val_str[features], val_str['target']),
    verbose=0
)
pred_cb_val  = np.clip(m_cb.predict(val_str[features]), 0, None)
pred_cb_test = np.clip(m_cb.predict(test_str[features]), 0, None)
cb_val_rmsle = np.sqrt(np.mean((pred_cb_val - val_str['target'].values)**2))
print(f"  CatBoost trained in {time.time()-t_m:.1f}s. Best iteration: {m_cb.best_iteration_}. Val RMSLE: {cb_val_rmsle:.5f}")

# ── 6. Retrain on Full Data (train+val) for Final Predictions ─────────────────
print("\n[Step 6/7] Retraining all models on full data (train+val) with best iterations...")
t0 = time.time()

full_train_cat = df_cat[(df_cat['date'] >= '2017-02-01') & (df_cat['date'] <= '2017-08-15')].copy()
full_train_str = df_str[(df_str['date'] >= '2017-02-01') & (df_str['date'] <= '2017-08-15')].copy()

# LightGBM retrain
m_lgb_full = lgb.LGBMRegressor(
    n_estimators=m_lgb.best_iteration_,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_samples=30,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=4,
    verbose=-1
)
m_lgb_full.fit(full_train_cat[features], full_train_cat['target'])
pred_lgb_final = np.clip(m_lgb_full.predict(test_cat[features]), 0, None)

# XGBoost retrain
m_xgb_full = xgb.XGBRegressor(
    n_estimators=m_xgb.best_iteration,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=30,
    reg_alpha=0.1,
    reg_lambda=1.0,
    enable_categorical=True,
    tree_method='hist',
    random_state=42,
    n_jobs=4
)
m_xgb_full.fit(full_train_cat[features], full_train_cat['target'])
pred_xgb_final = np.clip(m_xgb_full.predict(test_cat[features]), 0, None)

# CatBoost retrain
m_cb_full = CatBoostRegressor(
    iterations=m_cb.best_iteration_,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=3.0,
    cat_features=cat_cols,
    random_seed=42,
    verbose=0,
    thread_count=4
)
m_cb_full.fit(full_train_str[features], full_train_str['target'])
pred_cb_final = np.clip(m_cb_full.predict(test_str[features]), 0, None)

print(f"  All 3 models retrained on full data in {time.time()-t0:.1f}s.")

# ── 7. Validation-Optimized Ensemble & Submission ─────────────────────────────
print("\n[Step 7/7] Optimizing ensemble weights on validation set...")
t0 = time.time()

# Grid search for best blending weights on validation set
y_val = val_cat['target'].values
best_rmsle = float('inf')
best_weights = (0.33, 0.33, 0.34)

for w_lgb in np.arange(0.20, 0.65, 0.05):
    for w_xgb in np.arange(0.10, 0.55, 0.05):
        w_cb = 1.0 - w_lgb - w_xgb
        if w_cb < 0.05 or w_cb > 0.55:
            continue
        blend = w_lgb * pred_lgb_val + w_xgb * pred_xgb_val + w_cb * pred_cb_val
        rmsle = np.sqrt(np.mean((blend - y_val)**2))
        if rmsle < best_rmsle:
            best_rmsle = rmsle
            best_weights = (round(w_lgb, 2), round(w_xgb, 2), round(w_cb, 2))

w_lgb, w_xgb, w_cb = best_weights
print(f"  Optimal weights: LGB={w_lgb:.2f}, XGB={w_xgb:.2f}, CB={w_cb:.2f} → Val RMSLE: {best_rmsle:.5f}")

# Also try blending with Exp 5 domain base (our proven strong base)
sub_exp5 = pd.read_csv('/root/kaggle_store_sales/submission_exp5_ensemble_blend.csv')
test_orig = pd.read_csv('/root/kaggle_store_sales/test.csv')

# Align predictions to test.csv ID order
test_cat_aligned = test_cat.copy()
test_cat_aligned['pred_lgb'] = pred_lgb_final
test_cat_aligned['pred_xgb'] = pred_xgb_final
test_cat_aligned['store_nbr'] = test_cat_aligned['store_nbr'].astype(int)
test_cat_aligned['date'] = test_cat_aligned['date'].astype(str)

test_str_aligned = test_str.copy()
test_str_aligned['pred_cb'] = pred_cb_final
test_str_aligned['store_nbr'] = test_str_aligned['store_nbr'].astype(int)
test_str_aligned['date'] = test_str_aligned['date'].astype(str)

merged = test_orig[['id', 'store_nbr', 'family', 'date']].merge(
    test_cat_aligned[['store_nbr', 'family', 'date', 'pred_lgb', 'pred_xgb']],
    on=['store_nbr', 'family', 'date'], how='left'
).merge(
    test_str_aligned[['store_nbr', 'family', 'date', 'pred_cb']],
    on=['store_nbr', 'family', 'date'], how='left'
)

log_lgb_f = merged['pred_lgb'].values
log_xgb_f = merged['pred_xgb'].values
log_cb_f  = merged['pred_cb'].values
log_exp5  = np.log1p(sub_exp5['sales'].values)

# Dead series mask
recent_14d = full_train_cat[full_train_cat['date'] >= '2017-08-02']
sales_recent = recent_14d.groupby(['store_nbr', 'family'], observed=False)['sales'].sum().reset_index()
dead_pairs = set(sales_recent[sales_recent['sales'] == 0][['store_nbr', 'family']].itertuples(index=False, name=None))
test_dead_mask = test_orig.apply(lambda r: (int(r['store_nbr']), r['family']) in dead_pairs and r['onpromotion'] == 0, axis=1)
print(f"  Dead series: {len(dead_pairs)} pairs ({test_dead_mask.sum():,} test rows)")

# Tri-GBDT ensemble (validation-optimized)
log_tri_ensemble = w_lgb * log_lgb_f + w_xgb * log_xgb_f + w_cb * log_cb_f

# Generate multiple submissions with different blend ratios of domain base vs new ensemble
submissions = {}

# A: Pure Tri-GBDT (no domain base)
sales_a = np.clip(np.expm1(log_tri_ensemble), 0, None)
sales_a[test_dead_mask.values] = 0.0
submissions['exp11a_pure_tri_gbdt'] = sales_a

# B: 50% Domain Base + 50% Tri-GBDT
log_b = 0.50 * log_exp5 + 0.50 * log_tri_ensemble
sales_b = np.clip(np.expm1(log_b), 0, None)
sales_b[test_dead_mask.values] = 0.0
submissions['exp11b_50_50_blend'] = sales_b

# C: 60% Domain Base + 40% Tri-GBDT
log_c = 0.60 * log_exp5 + 0.40 * log_tri_ensemble
sales_c = np.clip(np.expm1(log_c), 0, None)
sales_c[test_dead_mask.values] = 0.0
submissions['exp11c_60_40_blend'] = sales_c

# D: 40% Domain Base + 60% Tri-GBDT
log_d = 0.40 * log_exp5 + 0.60 * log_tri_ensemble
sales_d = np.clip(np.expm1(log_d), 0, None)
sales_d[test_dead_mask.values] = 0.0
submissions['exp11d_40_60_blend'] = sales_d

# E: Best exp9 blend ratio (70/20/10) but with new models
log_e = 0.70 * log_exp5 + 0.20 * log_lgb_f + 0.10 * log_xgb_f
sales_e = np.clip(np.expm1(log_e), 0, None)
sales_e[test_dead_mask.values] = 0.0
submissions['exp11e_70_20_10_like_exp9'] = sales_e

print(f"\n  Generated {len(submissions)} submission variants:")
for name, sales in submissions.items():
    sub_df = pd.DataFrame({'id': test_orig['id'], 'sales': sales})
    path = f'/root/kaggle_store_sales/submission_{name}.csv'
    sub_df.to_csv(path, index=False)
    assert len(sub_df) == 28512
    assert not sub_df['sales'].isnull().any()
    assert (sub_df['sales'] >= 0).all()
    print(f"    ✓ {name}: mean={sales.mean():.2f}, zeros={(sales==0).sum()}, saved to {path}")

print(f"\n  Ensemble optimization done in {time.time()-t0:.1f}s.")

total_elapsed = time.time() - start_time
print("\n" + "=" * 80)
print(f"Experiment 11 complete in {total_elapsed:.1f}s ({total_elapsed/60:.2f} min).")
print("=" * 80)
print(f"\n🎯 Recommended submission: exp11a_pure_tri_gbdt (pure ML, no domain heuristic)")
print(f"   If that regresses, try exp11d_40_60_blend (more weight to new models)")
