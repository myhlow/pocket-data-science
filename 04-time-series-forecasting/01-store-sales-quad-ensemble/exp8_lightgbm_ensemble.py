import numpy as np
import pandas as pd
import lightgbm as lgb
import time

start_time = time.time()
print("=" * 70)
print("=== Experiment 8: LightGBM Leaf-Wise GBDT & Tri-Model Ensemble ===")
print("=" * 70)

# 1. Load Data
print("\n[Step 1/5] Loading datasets...")
t0 = time.time()
train = pd.read_csv('/root/kaggle_store_sales/train.csv', parse_dates=['date'])
test = pd.read_csv('/root/kaggle_store_sales/test.csv', parse_dates=['date'])
stores = pd.read_csv('/root/kaggle_store_sales/stores.csv')
oil = pd.read_csv('/root/kaggle_store_sales/oil.csv', parse_dates=['date'])
holidays = pd.read_csv('/root/kaggle_store_sales/holidays_events.csv', parse_dates=['date'])

# Restrict train to 2017 & filter Store 52 pre-opening
train_2017 = train[train['date'] >= '2017-01-01'].copy()
train_2017 = train_2017[~((train_2017['store_nbr'] == 52) & (train_2017['date'] < '2017-04-20'))].copy()
train_2017['target'] = np.log1p(train_2017['sales'])

train_2017 = train_2017.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')
print(f"Data loaded in {time.time()-t0:.1f}s. Train rows: {len(train_2017):,}, Test rows: {len(test):,}")

# 2. Exogenous Features
print("\n[Step 2/5] Engineering exogenous features...")
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

# Promotion Features
family_avg_promo = df_all.groupby('family')['onpromotion'].transform('mean').clip(lower=1e-3).astype('float32')
df_all['promo_ratio'] = (df_all['onpromotion'] / family_avg_promo).astype('float32')
df_all['is_promoted'] = (df_all['onpromotion'] > 0).astype('int8')

# Lags
df_all.sort_values(['store_nbr', 'family', 'date'], inplace=True)
df_all.reset_index(drop=True, inplace=True)

df_all['promo_lag_16'] = df_all.groupby(['store_nbr', 'family'])['onpromotion'].shift(16).fillna(0).astype('int16')
df_all['promo_delta_16'] = (df_all['onpromotion'] - df_all['promo_lag_16']).astype('float32')

for lag in [16, 21, 28]:
    df_all[f'lag_{lag}'] = df_all.groupby(['store_nbr', 'family'])['target'].shift(lag).astype('float32')

df_all['roll_mean_7_l16'] = df_all.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(7, min_periods=1).mean()).astype('float32')
df_all['roll_mean_14_l16'] = df_all.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(14, min_periods=1).mean()).astype('float32')

cat_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster']
for c in cat_cols:
    df_all[c] = df_all[c].astype('category')

features = cat_cols + [
    'onpromotion', 'promo_ratio', 'is_promoted', 'promo_delta_16',
    'is_weekend', 'day', 'dayofweek', 'is_payday', 'dist_to_payday', 'is_payday_window',
    'is_nat_holiday', 'is_reg_holiday', 'is_loc_holiday', 'is_holiday',
    'oil_price', 'oil_roll7', 'oil_diff7',
    'lag_16', 'lag_21', 'lag_28', 'roll_mean_7_l16', 'roll_mean_14_l16'
]

for col in features:
    if col not in cat_cols:
        df_all[col] = df_all[col].fillna(0.0)

train_df = df_all[(df_all['date'] >= '2017-02-01') & (df_all['date'] <= '2017-08-15')].copy()
test_df = df_all[df_all['date'] >= '2017-08-16'].copy()
print(f"Features prepared in {time.time()-t0:.1f}s. Training rows: {len(train_df):,}, Features: {len(features)}")

# 3. Train LightGBM Model
print("\n[Step 3/5] Fitting Leaf-Wise LightGBM Regressor...")
t0 = time.time()
lgb_model = lgb.LGBMRegressor(
    n_estimators=550,
    learning_rate=0.07,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=40,
    random_state=42,
    n_jobs=4,
    verbose=-1
)
lgb_model.fit(train_df[features], train_df['target'])
test_pred_lgb_log = np.clip(lgb_model.predict(test_df[features]), 0, None)
test_pred_lgb = np.expm1(test_pred_lgb_log)
print(f"LightGBM trained in {time.time()-t0:.1f}s.")

# 4. Dead Series Guardrails
print("\n[Step 4/5] Applying promotion-guarded dead-series zero mask...")
recent_14d = train_df[train_df['date'] >= '2017-08-02']
sales_recent = recent_14d.groupby(['store_nbr', 'family'], observed=False)['sales'].sum().reset_index()
dead_pairs = set(sales_recent[sales_recent['sales'] == 0][['store_nbr', 'family']].itertuples(index=False, name=None))

test_orig = pd.read_csv('/root/kaggle_store_sales/test.csv')
test_dead_mask = test_orig.apply(lambda r: (int(r['store_nbr']), r['family']) in dead_pairs and r['onpromotion'] == 0, axis=1)
print(f"Applied dead series zero mask to {test_dead_mask.sum():,} test rows.")

# Align test_df with test_orig ID order
test_df['sales_lgb'] = test_pred_lgb
test_df['store_nbr'] = test_df['store_nbr'].astype(int)
test_df['date'] = test_df['date'].astype(str)

sub_lgb = test_orig[['id', 'store_nbr', 'family', 'date']].merge(
    test_df[['store_nbr', 'family', 'date', 'sales_lgb']],
    on=['store_nbr', 'family', 'date'],
    how='left'
).rename(columns={'sales_lgb': 'sales'})

sub_lgb['sales'] = np.clip(sub_lgb['sales'], 0, None)
sub_lgb.loc[test_dead_mask, 'sales'] = 0.0

# Save Standalone LightGBM
sub_lgb_path = '/root/kaggle_store_sales/submission_exp8_lgb_standalone.csv'
sub_lgb[['id', 'sales']].to_csv(sub_lgb_path, index=False)
print(f"Saved Standalone LightGBM: {sub_lgb_path} (mean sales: {sub_lgb['sales'].mean():.2f})")

# 5. Build Ensembles with Champion Exp 5 Base (0.41072)
print("\n[Step 5/5] Building Multi-Paradigm Ensembles with Exp 5 Base (0.41072)...")
sub_exp5 = pd.read_csv('/root/kaggle_store_sales/submission_exp5_ensemble_blend.csv')
log_s5 = np.log1p(sub_exp5['sales'])
log_lgb = np.log1p(sub_lgb['sales'])

# Blend 1: 75% Exp 5 Base + 25% LightGBM (Conservative, low-variance enhancement)
blend_75_25 = np.clip(np.expm1(0.75 * log_s5 + 0.25 * log_lgb), 0, None)
blend_75_25[test_dead_mask.values] = 0.0
sub_blend_75_25 = pd.DataFrame({'id': test_orig['id'], 'sales': blend_75_25})
p1 = '/root/kaggle_store_sales/submission_exp8_blend_75_25.csv'
sub_blend_75_25.to_csv(p1, index=False)
print(f"✓ Saved 75% Exp5 + 25% LightGBM: {p1} (mean sales: {sub_blend_75_25['sales'].mean():.2f})")

# Blend 2: 60% Exp 5 Base + 40% LightGBM (Balanced, higher non-linear boost)
blend_60_40 = np.clip(np.expm1(0.60 * log_s5 + 0.40 * log_lgb), 0, None)
blend_60_40[test_dead_mask.values] = 0.0
sub_blend_60_40 = pd.DataFrame({'id': test_orig['id'], 'sales': blend_60_40})
p2 = '/root/kaggle_store_sales/submission_exp8_blend_60_40.csv'
sub_blend_60_40.to_csv(p2, index=False)
print(f"✓ Saved 60% Exp5 + 40% LightGBM: {p2} (mean sales: {sub_blend_60_40['sales'].mean():.2f})")

# Sanity checks
for name, p in [("LightGBM Standalone", sub_lgb_path), ("Blend 75/25", p1), ("Blend 60/40", p2)]:
    df = pd.read_csv(p)
    assert len(df) == 28512
    assert (df.loc[test_dead_mask.values, 'sales'] == 0.0).all()
    print(f"✓ {name}: rows={len(df):,}, zeros={(df['sales']==0).sum():,}, min={df['sales'].min():.2f}, max={df['sales'].max():.2f}, mean={df['sales'].mean():.2f}")

print("\n" + "=" * 70)
print(f"Experiment 8 complete in {time.time()-start_time:.1f}s.")
print("=" * 70)
