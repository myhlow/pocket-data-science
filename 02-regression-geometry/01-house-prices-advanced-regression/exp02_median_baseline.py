import pandas as pd
import numpy as np

def main():
    # 1. Load data
    train_df = pd.read_csv('/root/house-prices/train.csv')
    test_df = pd.read_csv('/root/house-prices/test.csv')

    # 2. Compute median SalePrice
    median_price = float(train_df['SalePrice'].median())
    print(f"Training median SalePrice: ${median_price:,.2f}")

    # Local validation score of predicting constant median on training set:
    log_actual = np.log1p(train_df['SalePrice'])
    log_pred = np.log1p(np.full_like(train_df['SalePrice'], median_price))
    rmsle = np.sqrt(np.mean((log_pred - log_actual) ** 2))
    print(f"Training RMSLE for constant median: {rmsle:.5f}")

    # 3. Create submission dataframe
    sub_df = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': median_price
    })

    # Validate against sample_submission
    sample_sub = pd.read_csv('/root/house-prices/sample_submission.csv')
    assert len(sub_df) == len(sample_sub), f"Expected {len(sample_sub)} rows, got {len(sub_df)}"
    assert (sub_df['Id'].values == sample_sub['Id'].values).all(), "Id mismatch with sample submission!"
    assert not sub_df['SalePrice'].isnull().any(), "Found null values in predictions!"

    out_path = '/root/house-prices/submission_median.csv'
    sub_df.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path} ({len(sub_df)} rows).")

if __name__ == '__main__':
    main()
