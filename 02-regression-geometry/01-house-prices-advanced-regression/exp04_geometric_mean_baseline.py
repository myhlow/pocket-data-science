import pandas as pd
import numpy as np

def main():
    # 1. Load datasets
    train_df = pd.read_csv('/root/house-prices/train.csv')
    test_df = pd.read_csv('/root/house-prices/test.csv')

    # 2. Compute Geometric Mean in log1p space (theoretical minimizer of RMSLE)
    log_y = np.log1p(train_df['SalePrice'])
    log_mean = float(log_y.mean())
    geom_price = float(np.expm1(log_mean))
    print(f"Mean log1p(SalePrice): {log_mean:.6f}")
    print(f"Optimal Constant Geometric Mean: ${geom_price:,.2f}")

    # Local evaluation on train set
    log_pred = np.full_like(log_y, log_mean)
    rmsle = np.sqrt(np.mean((log_pred - log_y) ** 2))
    print(f"Training RMSLE for Geometric Mean: {rmsle:.5f}")

    # 3. Create submission dataframe
    sub_df = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': geom_price
    })

    # Validate against sample_submission
    sample_sub = pd.read_csv('/root/house-prices/sample_submission.csv')
    assert len(sub_df) == len(sample_sub), f"Expected {len(sample_sub)} rows, got {len(sub_df)}"
    assert (sub_df['Id'].values == sample_sub['Id'].values).all(), "Id mismatch with sample submission!"
    assert not sub_df['SalePrice'].isnull().any(), "Found null values in predictions!"

    out_path = '/root/house-prices/submission_geometric_mean.csv'
    sub_df.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path} ({len(sub_df)} rows).")

if __name__ == '__main__':
    main()
