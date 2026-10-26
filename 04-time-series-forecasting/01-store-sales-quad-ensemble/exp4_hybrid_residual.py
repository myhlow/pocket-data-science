import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import time

start_time = time.time()
print("=== Experiment 4: Hybrid Residual GBDT (Domain Prior + CatBoost Residuals) ===")

# 1. Load Data
print("Loading datasets...")
train = pd.read_csv('/root/kaggle_store_sales/train.csv', parse_dates=['date'])
test = pd.read_csv('/root/kaggle_store_sales/test.csv', parse_dates=['date'])
stores = pd.read_csv('/root/kaggle_store_sales/stores.csv')
oil = pd.read_csv('/root/kaggle_store_sales/oil.csv', parse_dates=['date'])
holidays = pd.read_csv('/root/kaggle_store_sales/holidays_events.csv', parse_dates=['date'])

# Restrict train to 2017 to conserve RAM and avoid pre-2016 regime shifts
train_2017 = train[train['date'] >= '2017-01-01'].copy()
train_2017['target'] = np.log1p(train_2017['sales'])

# Merge store metadata
train_2017 = train_2017.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')

# Exogenous Features
# Oil
all_dates = pd.date_range(start='2017-01-01', end='2017-08-31', freq='D')
oil_full = pd.DataFrame({'date': all_dates})
oil_full = oil_full.merge(oil, on='date', how='left')
oil_full['oil_price'] = oil_full['dcoilwtico'].ffill().bfill()
oil_full['oil_roll7'] = oil_full['oil_price'].rolling(7, min_periods=1).mean()
oil_features = oil_full[['date', 'oil_price', 'oil_roll7']]

# Holidays
active_holidays = holidays[holidays['transferred'] == False].copy()
nat_dates = set(active_holidays[active_holidays['locale'] == 'National']['date'])
reg_sub = active_holidays[active_holidays['locale'] == 'Regional'][['date', 'locale_name']].drop_duplicates().rename(columns={'locale_name': 'state'})
reg_sub['is_reg_holiday'] = 1
loc_sub = active_holidays[active_holidays['locale'] == 'Local'][['date', 'locale_name']].drop_duplicates().rename(columns={'locale_name': 'city'})
loc_sub['is_loc_holiday'] = 1

# Combine for uniform processing
df_all = pd.concat([
    train_2017[['date', 'store_nbr', 'family', 'target', 'onpromotion', 'city', 'state', 'type', 'cluster']], 
    test[['date', 'store_nbr', 'family', 'onpromotion', 'city', 'state', 'type', 'cluster']]
], ignore_index=True)

df_all = df_all.merge(oil_features, on='date', how='left')
df_all['is_nat_holiday'] = df_all['date'].isin(nat_dates).astype(int)
df_all = df_all.merge(reg_sub, on=['date', 'state'], how='left')
df_all['is_reg_holiday'] = df_all['is_reg_holiday'].fillna(0).astype(int)
df_all = df_all.merge(loc_sub, on=['date', 'city'], how='left')
df_all['is_loc_holiday'] = df_all['is_loc_holiday'].fillna(0).astype(int)
df_all['is_holiday'] = ((df_all['is_nat_holiday'] == 1) | (df_all['is_reg_holiday'] == 1) | (df_all['is_loc_holiday'] == 1)).astype(int)

df_all['day'] = df_all['date'].dt.day
df_all['dayofweek'] = df_all['date'].dt.dayofweek
df_all['is_weekend'] = (df_all['dayofweek'] >= 5).astype(int)
df_all['is_payday'] = ((df_all['day'] == 15) | (df_all['date'].dt.is_month_end)).astype(int)

# 2. Compute 42-day Grouped Mean Prior
def get_prior_table(source_train, end_date, window_days=42):
    start_date = pd.to_datetime(end_date) - pd.Timedelta(days=window_days)
    sub_df = source_train[(source_train['date'] >= start_date) & (source_train['date'] <= end_date)].copy()
    sub_df['dayofweek'] = sub_df['date'].dt.dayofweek
    
    prior = sub_df.groupby(['store_nbr', 'family', 'dayofweek'])['sales'].mean().reset_index()
    prior.rename(columns={'sales': 'prior_sales'}, inplace=True)
    
    # Back-to-school correction for Sierra August rush
    school_sub = source_train[(source_train['date'] >= '2017-08-01') & 
                              (source_train['date'] <= end_date) & 
                              (source_train['family'] == 'SCHOOL AND OFFICE SUPPLIES')].copy()
    if len(school_sub) > 0:
        school_sub['dayofweek'] = school_sub['date'].dt.dayofweek
        school_prior = school_sub.groupby(['store_nbr', 'family', 'dayofweek'])['sales'].mean().reset_index()
        school_prior.rename(columns={'sales': 'school_sales'}, inplace=True)
        prior = prior.merge(school_prior, on=['store_nbr', 'family', 'dayofweek'], how='left')
        prior['prior_sales'] = np.where(prior['school_sales'].notnull(), prior['school_sales'], prior['prior_sales'])
        prior.drop(columns=['school_sales'], inplace=True)
        
    return prior

print("Computing domain heuristic priors...")
# Prior for Test (using 2017-07-05 to 2017-08-15)
test_prior_table = get_prior_table(train_2017, '2017-08-15', window_days=42)

# Prior for Validation (using 2017-06-20 to 2017-07-31)
val_prior_table = get_prior_table(train_2017, '2017-07-31', window_days=42)

# Merge test prior as default prior
df_all = df_all.merge(test_prior_table, on=['store_nbr', 'family', 'dayofweek'], how='left')
df_all['prior_sales'] = df_all['prior_sales'].fillna(0.0)
df_all['log_prior'] = np.log1p(df_all['prior_sales'])
df_all['residual'] = df_all['target'] - df_all['log_prior']

# Lags
df_all.sort_values(['store_nbr', 'family', 'date'], inplace=True)
df_all.reset_index(drop=True, inplace=True)

print("Constructing lag features...")
for lag in [16, 21, 28]:
    df_all[f'lag_{lag}'] = df_all.groupby(['store_nbr', 'family'])['target'].shift(lag)
    df_all[f'res_lag_{lag}'] = df_all.groupby(['store_nbr', 'family'])['residual'].shift(lag)

grouped = df_all.groupby(['store_nbr', 'family'])['lag_16']
df_all['rolling_mean_7'] = grouped.transform(lambda x: x.rolling(7, min_periods=1).mean())
df_all['rolling_mean_14'] = grouped.transform(lambda x: x.rolling(14, min_periods=1).mean())

num_features = [
    'log_prior', 'onpromotion', 'is_weekend', 'day', 'dayofweek', 'is_payday', 'is_holiday',
    'oil_price', 'oil_roll7',
    'lag_16', 'lag_21', 'lag_28', 'res_lag_16', 'res_lag_21', 'res_lag_28',
    'rolling_mean_7', 'rolling_mean_14'
]
for col in num_features:
    df_all[col] = df_all[col].fillna(0.0)

cat_features = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster']
for col in cat_features:
    df_all[col] = df_all[col].astype(str)

features = cat_features + num_features
print(f"Total features: {len(features)}")

# Split
train_df = df_all[df_all['date'] <= '2017-08-15'].copy()
test_df = df_all[df_all['date'] >= '2017-08-16'].copy()
train_df = train_df[train_df['date'] >= '2017-02-01'].copy()

val_mask = train_df['date'] >= '2017-08-01'
tr_mask = train_df['date'] < '2017-08-01'

X_tr = train_df.loc[tr_mask, features]
y_tr = train_df.loc[tr_mask, 'residual']
X_val = train_df.loc[val_mask, features]
y_val = train_df.loc[val_mask, 'residual']

print(f"Training rows: {len(X_tr):,}, Validation rows: {len(X_val):,}")

# 3. Fit CatBoost on Residuals
cb_res = CatBoostRegressor(
    iterations=400,
    learning_rate=0.07,
    depth=6,
    loss_function='RMSE',
    cat_features=cat_features,
    random_seed=42,
    verbose=50,
    thread_count=4
)

print("Fitting CatBoost on residuals...")
cb_res.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=40)

# Evaluate Validation
val_res_pred = cb_res.predict(X_val)
val_pred_log = np.clip(X_val['log_prior'].values + val_res_pred, 0, None)
val_preds = np.expm1(val_pred_log)
val_actual = np.expm1(train_df.loc[val_mask, 'target'].values)
val_rmsle = np.sqrt(np.mean((np.log1p(val_preds) - np.log1p(val_actual)) ** 2))
print(f"\n--> Hybrid Validation RMSLE: {val_rmsle:.5f}")

# 4. Retrain Full on Full Training Residuals
print("\nRetraining on full training set...")
best_iters = cb_res.get_best_iteration() or 350
cb_full = CatBoostRegressor(
    iterations=best_iters,
    learning_rate=0.07,
    depth=6,
    loss_function='RMSE',
    cat_features=cat_features,
    random_seed=42,
    verbose=0,
    thread_count=4
)
cb_full.fit(train_df[features], train_df['residual'])

# 5. Predict Test & Apply Domain Guardrails
test_res_pred = cb_full.predict(test_df[features])
test_pred_log = np.clip(test_df['log_prior'].values + test_res_pred, 0, None)
test_preds = np.expm1(test_pred_log)

# Dead catalog zero mask
last_14d = train_2017[train_2017['date'] >= '2017-08-02']
recent_sales = last_14d.groupby(['store_nbr', 'family'])['sales'].sum().reset_index()
dead_pairs = set(recent_sales[recent_sales['sales'] == 0][['store_nbr', 'family']].itertuples(index=False, name=None))
print(f"Identified {len(dead_pairs)} historically dead series in final 14 days.")

test_df['sales'] = test_preds
mask_dead = test_df.apply(lambda r: (int(r['store_nbr']), r['family']) in dead_pairs and r['onpromotion'] == 0, axis=1)
test_df.loc[mask_dead, 'sales'] = 0.0
print(f"Applied promotion-guarded zero mask to {mask_dead.sum()} test rows.")

# Merge to preserve test ID order
test_orig = pd.read_csv('/root/kaggle_store_sales/test.csv')
test_df['store_nbr'] = test_df['store_nbr'].astype(int)
test_df['date'] = test_df['date'].astype(str)
sub = test_orig[['id', 'store_nbr', 'family', 'date']].merge(
    test_df[['store_nbr', 'family', 'date', 'sales']],
    on=['store_nbr', 'family', 'date'],
    how='left'
)

sub[['id', 'sales']].to_csv('/root/kaggle_store_sales/submission_exp4_hybrid_residual.csv', index=False)
print(f"Submission saved to submission_exp4_hybrid_residual.csv. Total time: {time.time()-start_time:.1f}s")
