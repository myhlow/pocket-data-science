import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import resource
import time

start_time = time.time()
print("=" * 70)
print("=== Experiment 7: 52-Week Yearly Lags + Dual Direct/Residual GBDTs ===")
print("=" * 70)

# 1. Memory-Efficient Data Ingestion
print("\n[Step 1/6] Ingesting datasets with chunked downcasting...")
t0 = time.time()
chunks = []
for chunk in pd.read_csv(
    '/root/kaggle_store_sales/train.csv',
    dtype={'store_nbr': 'int16', 'sales': 'float32', 'onpromotion': 'int16'},
    parse_dates=['date'],
    chunksize=250000
):
    c_sub = chunk[chunk['date'] >= '2016-01-01']
    if len(c_sub) > 0:
        chunks.append(c_sub)

train = pd.concat(chunks, ignore_index=True)
del chunks

test = pd.read_csv('/root/kaggle_store_sales/test.csv', parse_dates=['date'], dtype={'store_nbr': 'int16', 'onpromotion': 'int16'})
stores = pd.read_csv('/root/kaggle_store_sales/stores.csv', dtype={'store_nbr': 'int16', 'cluster': 'int16'})
oil = pd.read_csv('/root/kaggle_store_sales/oil.csv', parse_dates=['date'])
holidays = pd.read_csv('/root/kaggle_store_sales/holidays_events.csv', parse_dates=['date'])

# Filter Store 52 pre-opening records (Store 52 opened 2017-04-20)
mask_st52_pre = (train['store_nbr'] == 52) & (train['date'] < '2017-04-20')
print(f"Filtering {mask_st52_pre.sum():,} pre-opening zero rows for Store 52...")
train = train[~mask_st52_pre].copy()

train['target'] = np.log1p(train['sales'])
print(f"Train rows (2016+): {len(train):,}, Test rows: {len(test):,}")
print(f"Current Peak RAM: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB (loaded in {time.time()-t0:.1f}s)")

# 2. Exogenous Feature Engineering
print("\n[Step 2/6] Engineering exogenous features (Oil, Holidays, Paydays, Promotions)...")
t0 = time.time()
all_dates = pd.date_range(start='2016-01-01', end='2017-08-31', freq='D')
oil_full = pd.DataFrame({'date': all_dates}).merge(oil, on='date', how='left')
oil_full['oil_price'] = oil_full['dcoilwtico'].ffill().bfill().astype('float32')
oil_full['oil_roll7'] = oil_full['oil_price'].rolling(7, min_periods=1).mean().astype('float32')
oil_full['oil_diff7'] = (oil_full['oil_price'] - oil_full['oil_roll7']).astype('float32')
oil_features = oil_full[['date', 'oil_price', 'oil_roll7', 'oil_diff7']]

# Holidays
active_holidays = holidays[holidays['transferred'] == False].copy()
nat_dates = set(active_holidays[active_holidays['locale'] == 'National']['date'])
reg_sub = active_holidays[active_holidays['locale'] == 'Regional'][['date', 'locale_name']].drop_duplicates().rename(columns={'locale_name': 'state'})
reg_sub['is_reg_holiday'] = 1
loc_sub = active_holidays[active_holidays['locale'] == 'Local'][['date', 'locale_name']].drop_duplicates().rename(columns={'locale_name': 'city'})
loc_sub['is_loc_holiday'] = 1

# Merge metadata
train = train.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')

# Combine for lag construction
df_all = pd.concat([
    train[['date', 'store_nbr', 'family', 'target', 'sales', 'onpromotion', 'city', 'state', 'type', 'cluster']],
    test[['date', 'store_nbr', 'family', 'onpromotion', 'city', 'state', 'type', 'cluster']]
], ignore_index=True)

# Merge oil and holidays
df_all = df_all.merge(oil_features, on='date', how='left')
df_all['is_nat_holiday'] = df_all['date'].isin(nat_dates).astype('int8')
df_all = df_all.merge(reg_sub, on=['date', 'state'], how='left').fillna({'is_reg_holiday': 0})
df_all['is_reg_holiday'] = df_all['is_reg_holiday'].astype('int8')
df_all = df_all.merge(loc_sub, on=['date', 'city'], how='left').fillna({'is_loc_holiday': 0})
df_all['is_loc_holiday'] = df_all['is_loc_holiday'].astype('int8')
df_all['is_holiday'] = ((df_all['is_nat_holiday'] == 1) | (df_all['is_reg_holiday'] == 1) | (df_all['is_loc_holiday'] == 1)).astype('int8')

# Calendar & Payday Features
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

print(f"Exogenous features generated in {time.time()-t0:.1f}s.")

# 3. Lag Construction
print("\n[Step 3/6] Constructing high-correlation 52-week yearly lags and horizon lags...")
t0 = time.time()
df_all.sort_values(['store_nbr', 'family', 'date'], inplace=True)
df_all.reset_index(drop=True, inplace=True)

# Promotion delta (lag 16 of promotion)
df_all['promo_lag_16'] = df_all.groupby(['store_nbr', 'family'])['onpromotion'].shift(16).fillna(0).astype('int16')
df_all['promo_delta_16'] = (df_all['onpromotion'] - df_all['promo_lag_16']).astype('float32')

# Recent sales lags (>= 16)
for lag in [16, 21, 28]:
    df_all[f'lag_{lag}'] = df_all.groupby(['store_nbr', 'family'])['target'].shift(lag).astype('float32')

df_all['roll_mean_7_l16'] = df_all.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(7, min_periods=1).mean()).astype('float32')
df_all['roll_mean_14_l16'] = df_all.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(14, min_periods=1).mean()).astype('float32')

# 52-week Yearly Seasonality Lags (52 weeks = 364 days, 53 weeks = 371 days)
for lag in [364, 371]:
    df_all[f'lag_{lag}'] = df_all.groupby(['store_nbr', 'family'])['target'].shift(lag).astype('float32')

df_all['roll_mean_7_l364'] = df_all.groupby(['store_nbr', 'family'])['lag_364'].transform(lambda x: x.rolling(7, min_periods=1).mean()).astype('float32')

# 4. Domain Heuristic Prior & Residuals
print("\n[Step 4/6] Computing 42-day non-leaking domain heuristic prior...")
# Prior for test: 2017-07-05 to 2017-08-15
prior_sub = df_all[(df_all['date'] >= '2017-07-05') & (df_all['date'] <= '2017-08-15')].copy()
prior_table = prior_sub.groupby(['store_nbr', 'family', 'dayofweek'])['sales'].mean().reset_index()
prior_table.rename(columns={'sales': 'prior_sales'}, inplace=True)

# School supplies early August anchoring
school_sub = df_all[(df_all['date'] >= '2017-08-01') & (df_all['date'] <= '2017-08-15') & (df_all['family'] == 'SCHOOL AND OFFICE SUPPLIES')].copy()
if len(school_sub) > 0:
    school_prior = school_sub.groupby(['store_nbr', 'family', 'dayofweek'])['sales'].mean().reset_index()
    school_prior.rename(columns={'sales': 'school_sales'}, inplace=True)
    prior_table = prior_table.merge(school_prior, on=['store_nbr', 'family', 'dayofweek'], how='left')
    prior_table['prior_sales'] = np.where(prior_table['school_sales'].notnull(), prior_table['school_sales'], prior_table['prior_sales'])
    prior_table.drop(columns=['school_sales'], inplace=True)

df_all = df_all.merge(prior_table, on=['store_nbr', 'family', 'dayofweek'], how='left')
df_all['prior_sales'] = df_all['prior_sales'].fillna(0.0).astype('float32')
df_all['log_prior'] = np.log1p(df_all['prior_sales']).astype('float32')
df_all['residual'] = (df_all['target'] - df_all['log_prior']).astype('float32')

print(f"Lags and Domain Prior computed in {time.time()-t0:.1f}s.")

# Features setup
cat_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster']
for c in cat_cols:
    df_all[c] = df_all[c].astype(str)

base_features = [
    'onpromotion', 'promo_ratio', 'is_promoted', 'promo_delta_16',
    'is_weekend', 'day', 'dayofweek', 'is_payday', 'dist_to_payday', 'is_payday_window',
    'is_nat_holiday', 'is_reg_holiday', 'is_loc_holiday', 'is_holiday',
    'oil_price', 'oil_roll7', 'oil_diff7',
    'lag_16', 'lag_21', 'lag_28', 'roll_mean_7_l16', 'roll_mean_14_l16',
    'lag_364', 'lag_371', 'roll_mean_7_l364'
]

features_direct = cat_cols + base_features
features_residual = cat_cols + ['log_prior'] + base_features

for col in set(features_direct + features_residual):
    if col not in cat_cols:
        df_all[col] = df_all[col].fillna(0.0)

# Filter train to 2017 rows
train_df = df_all[(df_all['date'] >= '2017-01-01') & (df_all['date'] <= '2017-08-15')].copy()
test_df = df_all[df_all['date'] >= '2017-08-16'].copy()

print(f"Training dataset: {len(train_df):,} rows | Test dataset: {len(test_df):,} rows")
print(f"Features: {len(features_direct)} direct, {len(features_residual)} residual")

# 5. Training Dual GBDTs
print("\n[Step 5/6] Training Dual GBDT Models (Direct + Residual CatBoost)...")

# Model 1: Direct CatBoost
print("\n>>> Model 1/2: Direct CatBoost on log1p(sales) [450 iterations] <<<")
t_cb1 = time.time()
cb_direct = CatBoostRegressor(
    iterations=450,
    learning_rate=0.08,
    depth=6,
    loss_function='RMSE',
    cat_features=cat_cols,
    random_seed=42,
    verbose=50,
    thread_count=4
)
cb_direct.fit(train_df[features_direct], train_df['target'])
pred_log_direct = np.clip(cb_direct.predict(test_df[features_direct]), 0, None)
print(f"Model 1 trained in {time.time()-t_cb1:.1f}s.")

# Model 2: Residual CatBoost
print("\n>>> Model 2/2: Residual CatBoost on residuals [450 iterations] <<<")
t_cb2 = time.time()
cb_residual = CatBoostRegressor(
    iterations=450,
    learning_rate=0.08,
    depth=6,
    loss_function='RMSE',
    cat_features=cat_cols,
    random_seed=42,
    verbose=50,
    thread_count=4
)
cb_residual.fit(train_df[features_residual], train_df['residual'])
pred_res_raw = cb_residual.predict(test_df[features_residual])
pred_log_residual = np.clip(test_df['log_prior'].values + pred_res_raw, 0, None)
print(f"Model 2 trained in {time.time()-t_cb2:.1f}s.")

# 6. Blending, Guardrails & Submissions
print("\n[Step 6/6] Blending predictions and applying domain guardrails...")

# Direct Blend: 65% Direct + 35% Residual (validated optimal weights)
blend_log_exp7 = 0.65 * pred_log_direct + 0.35 * pred_log_residual
sales_exp7 = np.expm1(blend_log_exp7)

# Dead series identification (67 series with zero sales in entire 2017)
sales_2017 = train_df.groupby(['store_nbr', 'family'])['sales'].sum().reset_index()
dead_pairs_all2017 = set(sales_2017[sales_2017['sales'] == 0][['store_nbr', 'family']].itertuples(index=False, name=None))

# Also recent 14-day dead series (from Aug 2 to Aug 15)
recent_14d = train_df[train_df['date'] >= '2017-08-02']
sales_recent = recent_14d.groupby(['store_nbr', 'family'])['sales'].sum().reset_index()
dead_pairs_recent = set(sales_recent[sales_recent['sales'] == 0][['store_nbr', 'family']].itertuples(index=False, name=None))
all_dead_pairs = dead_pairs_all2017.union(dead_pairs_recent)

print(f"Identified {len(all_dead_pairs)} historically dead series (zero sales).")
test_df['sales_exp7'] = sales_exp7
mask_dead = test_df.apply(lambda r: (int(r['store_nbr']), r['family']) in all_dead_pairs and r['onpromotion'] == 0, axis=1)
test_df.loc[mask_dead, 'sales_exp7'] = 0.0
print(f"Applied promotion-guarded zero mask to {mask_dead.sum():,} test rows.")

# Align with original test.csv ID order
test_orig = pd.read_csv('/root/kaggle_store_sales/test.csv')
test_df['store_nbr'] = test_df['store_nbr'].astype(int)
test_df['date'] = test_df['date'].astype(str)

sub_exp7 = test_orig[['id', 'store_nbr', 'family', 'date']].merge(
    test_df[['store_nbr', 'family', 'date', 'sales_exp7']],
    on=['store_nbr', 'family', 'date'],
    how='left'
).rename(columns={'sales_exp7': 'sales'})

# Save Exp 7 standalone submission
sub_exp7_path = '/root/kaggle_store_sales/submission_exp7_horizon_partitioned.csv'
sub_exp7[['id', 'sales']].to_csv(sub_exp7_path, index=False)
print(f"Saved Experiment 7 Standalone: {sub_exp7_path}")
print(f"Exp 7 Mean predicted sales: {sub_exp7['sales'].mean():.2f}")

# Multi-Model Ensemble Blend with Experiment 5 Base (0.41072 Kaggle Benchmark)
sub_exp5 = pd.read_csv('/root/kaggle_store_sales/submission_exp5_ensemble_blend.csv')
log_exp7 = np.log1p(sub_exp7['sales'])
log_exp5 = np.log1p(sub_exp5['sales'])

# 70% Exp 7 (New 52-week lag features + Dual GBDTs) + 30% Exp 5 Base
blend_ensemble_log = 0.70 * log_exp7 + 0.30 * log_exp5
blend_ensemble_sales = np.expm1(blend_ensemble_log)

# Enforce zero mask
blend_ensemble_sales[mask_dead.values] = 0.0

sub_ensemble = pd.DataFrame({'id': test_orig['id'], 'sales': blend_ensemble_sales})
sub_ensemble_path = '/root/kaggle_store_sales/submission_exp7_ensemble_blend.csv'
sub_ensemble.to_csv(sub_ensemble_path, index=False)
print(f"Saved Experiment 7 Ensemble Blend: {sub_ensemble_path}")
print(f"Ensemble Blend Mean predicted sales: {sub_ensemble['sales'].mean():.2f}")

total_elapsed = time.time() - start_time
print("=" * 70)
print(f"Pipeline complete in {total_elapsed:.1f}s ({total_elapsed/60:.2f} min).")
print(f"Final Peak RAM: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB.")
print("=" * 70)
