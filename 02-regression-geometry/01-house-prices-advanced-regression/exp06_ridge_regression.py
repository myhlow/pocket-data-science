import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def main():
    print("Loading data...")
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
    num_cols = X.select_dtypes(include=['number']).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    # 5-fold CV to evaluate and generate out-of-fold and test predictions
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    alphas = np.logspace(-2, 3, 50)

    print("Running 5-fold Cross-Validation for Ridge Regression...")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X.iloc[tr_idx], log_y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], log_y.iloc[va_idx]

        pipe = Pipeline([
            ('prep', preprocessor),
            ('ridge', RidgeCV(alphas=alphas))
        ])
        pipe.fit(X_tr, y_tr)
        best_alpha = pipe.named_steps['ridge'].alpha_
        
        pred_va = pipe.predict(X_va)
        oof[va_idx] = pred_va
        test_preds += pipe.predict(X_test) / kf.n_splits

        fold_rmsle = np.sqrt(np.mean((pred_va - y_va)**2))
        print(f"  Fold {fold+1} RMSLE: {fold_rmsle:.5f} (Best alpha: {best_alpha:.2f})")

    cv_rmsle = np.sqrt(np.mean((oof - log_y)**2))
    print(f"\nOverall Ridge 5-Fold OOF RMSLE: {cv_rmsle:.5f}")

    # Exponentiate back to dollars
    final_preds = np.expm1(test_preds)

    sub = pd.DataFrame({'Id': test['Id'], 'SalePrice': final_preds})
    assert len(sub) == len(sample_sub)
    assert not sub['SalePrice'].isnull().any()
    
    out_path = '/root/house-prices/submission_ridge.csv'
    sub.to_csv(out_path, index=False)
    print(f"Saved Ridge submission to {out_path} ({len(sub)} rows).")

if __name__ == '__main__':
    main()
