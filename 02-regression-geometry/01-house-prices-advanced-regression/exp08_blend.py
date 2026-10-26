import pandas as pd
import numpy as np

def main():
    print("Loading test predictions from Ridge and CatBoost...")
    sub_ridge = pd.read_csv('/root/house-prices/submission_ridge.csv')
    sub_cat = pd.read_csv('/root/house-prices/submission_catboost.csv')
    sample_sub = pd.read_csv('/root/house-prices/sample_submission.csv')

    # Geometric blend in log space (80% CatBoost + 20% Ridge)
    log_ridge = np.log1p(sub_ridge['SalePrice'])
    log_cat = np.log1p(sub_cat['SalePrice'])

    w_ridge = 0.20
    w_cat = 0.80
    log_blend = w_ridge * log_ridge + w_cat * log_cat
    final_preds = np.expm1(log_blend)

    sub_blend = pd.DataFrame({
        'Id': sub_ridge['Id'],
        'SalePrice': final_preds
    })

    assert len(sub_blend) == len(sample_sub)
    assert not sub_blend['SalePrice'].isnull().any()

    out_path = '/root/house-prices/submission_blend.csv'
    sub_blend.to_csv(out_path, index=False)
    print(f"Saved 80/20 CatBoost/Ridge blend to {out_path} ({len(sub_blend)} rows).")

if __name__ == '__main__':
    main()
