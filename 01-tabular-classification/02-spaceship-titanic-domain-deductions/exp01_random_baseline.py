"""Exp 01: Stratified Random Baseline for Spaceship Titanic

Generates predictions based on the empirical class distribution of the training set.
"""

import numpy as np
import pandas as pd

def main():
    train_path = "/root/spaceship-titanic/train.csv"
    test_path = "/root/spaceship-titanic/test.csv"
    sample_sub_path = "/root/spaceship-titanic/sample_submission.csv"
    out_path = "/root/spaceship-titanic/submission_random.csv"

    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    # Calculate empirical prior
    p_true = float(train_df["Transported"].mean())
    print(f"Empirical P(Transported=True) in train: {p_true:.6f} ({train_df['Transported'].sum()}/{len(train_df)})")

    # Generate random predictions with reproducible seed
    rng = np.random.RandomState(42)
    random_preds = rng.choice([True, False], size=len(test_df), p=[p_true, 1.0 - p_true])

    sub_df = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": random_preds
    })

    # Validate against sample submission
    assert len(sub_df) == len(sample_sub), f"Length mismatch: {len(sub_df)} vs {len(sample_sub)}"
    assert (sub_df["PassengerId"].values == sample_sub["PassengerId"].values).all(), "PassengerId order mismatch!"
    assert sub_df["Transported"].isnull().sum() == 0, "Found nulls in predictions!"

    sub_df.to_csv(out_path, index=False)
    print(f"Saved random baseline predictions to {out_path}")
    print(f"Distribution in submission:\n{sub_df['Transported'].value_counts(normalize=True)}")
    print("\nFirst 5 rows:")
    print(sub_df.head())

if __name__ == "__main__":
    main()
