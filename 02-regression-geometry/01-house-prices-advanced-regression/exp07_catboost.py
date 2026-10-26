import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

def main():
    print("Loading data for CatBoost...")
    train = pd.read_csv('/root/house-prices/train.csv')
    test = pd.read_csv('/root/house-prices/test.csv')
    sample_sub = pd.read_csv('/root/house-prices/sample_submission.csv')

    train['MSSubClass'] = train['MSSubClass'].astype(str)
    test['MSSubClass'] = test['MSSubClass'].astype(str)

    y = train['SalePrice']
    log_y = np.log1p(y)
    X = train.drop(columns=['Id', 'SalePrice'])
    X_test = test.drop(columns=['Id'])

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Fill NA in categoricals for CatBoost native categorical handling
    for c in cat_cols:
        X[c] = X[c].fillna('Missing').astype(str)
        X_test[c] = X_test[c].fillna('Missing').astype(str)

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    print(f"Running 5-fold Cross-Validation with {len(cat_cols)} native categorical features...")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X.iloc[tr_idx], log_y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], log_y.iloc[va_idx]

        model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            loss_function='RMSE',
            eval_metric='RMSE',
            cat_features=cat_cols,
            random_seed=42 + fold,
            verbose=0
        )
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=False
        )

        pred_va = model.predict(X_va)
        oof[va_idx] = pred_va
        test_preds += model.predict(X_test) / kf.n_splits

        fold_rmsle = np.sqrt(np.mean((pred_va - y_va)**2))
        print(f"  Fold {fold+1} RMSLE: {fold_rmsle:.5f} (Best iteration: {model.get_best_iteration()})")

    cv_rmsle = np.sqrt(np.mean((oof - log_y)**2))
    print(f"\nOverall CatBoost 5-Fold OOF RMSLE: {cv_rmsle:.5f}")

    # Exponentiate back to dollars
    final_preds = np.expm1(test_preds)

    sub = pd.DataFrame({'Id': test['Id'], 'SalePrice': final_preds})
    assert len(sub) == len(sample_sub)
    assert not sub['SalePrice'].isnull().any()

    out_path = '/root/house-prices/submission_catboost.csv'
    sub.to_csv(out_path, index=False)
    print(f"Saved CatBoost submission to {out_path} ({len(sub)} rows).")

if __name__ == '__main__':
    main()
