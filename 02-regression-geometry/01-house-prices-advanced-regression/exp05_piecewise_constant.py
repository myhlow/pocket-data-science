import pandas as pd
import numpy as np

def main():
    # 1. Load data
    train = pd.read_csv('/root/house-prices/train.csv')
    test = pd.read_csv('/root/house-prices/test.csv')
    sample_sub = pd.read_csv('/root/house-prices/sample_submission.csv')

    train['log_SalePrice'] = np.log1p(train['SalePrice'])
    global_log_mean = train['log_SalePrice'].mean()

    # --- Model A: 1D Piecewise Constant on OverallQual ---
    qual_log_mean = train.groupby('OverallQual')['log_SalePrice'].mean()
    pred_qual_log = test['OverallQual'].map(qual_log_mean).fillna(global_log_mean)
    pred_qual = np.expm1(pred_qual_log)

    sub_qual = pd.DataFrame({'Id': test['Id'], 'SalePrice': pred_qual})
    assert len(sub_qual) == len(sample_sub)
    assert not sub_qual['SalePrice'].isnull().any()
    sub_qual.to_csv('/root/house-prices/submission_piecewise_qual.csv', index=False)
    print("Saved 1D Piecewise (OverallQual) to /root/house-prices/submission_piecewise_qual.csv")

    # --- Model B: 2D Piecewise Constant on (Neighborhood, OverallQual) ---
    combo_log_mean = train.groupby(['Neighborhood', 'OverallQual'])['log_SalePrice'].mean()
    
    # Map multi-index
    test_idx = test.set_index(['Neighborhood', 'OverallQual']).index
    pred_combo_log = pd.Series(test_idx.map(combo_log_mean), index=test.index)
    # Fallback 1: OverallQual log mean; Fallback 2: Global log mean
    pred_combo_log = pred_combo_log.fillna(test['OverallQual'].map(qual_log_mean)).fillna(global_log_mean)
    pred_combo = np.expm1(pred_combo_log)

    sub_combo = pd.DataFrame({'Id': test['Id'], 'SalePrice': pred_combo})
    assert len(sub_combo) == len(sample_sub)
    assert not sub_combo['SalePrice'].isnull().any()
    sub_combo.to_csv('/root/house-prices/submission_piecewise_combo.csv', index=False)
    print("Saved 2D Piecewise (Neighborhood + OverallQual) to /root/house-prices/submission_piecewise_combo.csv")

if __name__ == '__main__':
    main()
