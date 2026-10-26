"""Exp 01: Random Baseline for House Prices: Advanced Regression Techniques

Generates reproducible random predictions sampled from the empirical training distribution.
This establishes the bottom baseline benchmark for the Pocket Data Science series.
"""

import numpy as np
import pandas as pd
import os

def main():
    base_dir = "/root/house-prices"
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")
    sample_sub_path = os.path.join(base_dir, "sample_submission.csv")
    out_path = os.path.join(base_dir, "submission_random.csv")

    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    print(f"Train set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")

    # Empirical target statistics
    sale_prices = train_df["SalePrice"].values
    min_p, max_p = sale_prices.min(), sale_prices.max()
    mean_p, median_p = sale_prices.mean(), np.median(sale_prices)
    std_p = sale_prices.std()

    print(f"Empirical SalePrice Stats:")
    print(f"  Min:    ${min_p:,.2f}")
    print(f"  Median: ${median_p:,.2f}")
    print(f"  Mean:   ${mean_p:,.2f}")
    print(f"  Max:    ${max_p:,.2f}")
    print(f"  Std:    ${std_p:,.2f}")

    # Set reproducible random seed
    seed = 42
    rng = np.random.RandomState(seed)

    # Strategy: Stratified Empirical Random Sampling (Bootstrap from training distribution)
    # This reflects a true random baseline that respects the target domain's observed marginal distribution.
    random_preds = rng.choice(sale_prices, size=len(test_df), replace=True)

    # Build submission dataframe
    sub_df = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": random_preds
    })

    # Rigorous validation checks
    assert len(sub_df) == len(sample_sub), f"Row count mismatch: {len(sub_df)} vs {len(sample_sub)}"
    assert (sub_df["Id"].values == sample_sub["Id"].values).all(), "Id column mismatch!"
    assert sub_df["SalePrice"].isnull().sum() == 0, "Found NaN values in predictions!"
    assert (sub_df["SalePrice"] > 0).all(), "Found non-positive price predictions!"

    sub_df.to_csv(out_path, index=False)
    print(f"\nSuccessfully generated and saved random predictions to:\n  {out_path}")

    print("\nSubmission Summary Statistics:")
    print(sub_df["SalePrice"].describe())

    print("\nFirst 10 sample predictions:")
    print(sub_df.head(10))

if __name__ == "__main__":
    main()
