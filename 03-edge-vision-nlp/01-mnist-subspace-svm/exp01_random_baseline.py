import pandas as pd
import numpy as np

def main():
    print("Loading data for MNIST Random Baseline...")
    train = pd.read_csv('/root/digit-recognizer/train.csv')
    test = pd.read_csv('/root/digit-recognizer/test.csv')
    sample_sub = pd.read_csv('/root/digit-recognizer/sample_submission.csv')

    # Compute empirical digit frequencies in train set
    digit_probs = train['label'].value_counts(normalize=True).sort_index().values
    print("Empirical training class probabilities:")
    for d, p in enumerate(digit_probs):
        print(f"  Digit {d}: {p*100:.2f}%")

    # Generate random predictions using empirical training distribution
    np.random.seed(42)
    random_preds = np.random.choice(10, size=len(test), p=digit_probs)

    # Local expectation check
    expected_acc = np.sum(digit_probs ** 2)
    print(f"\nExpected accuracy for empirical random guessing: {expected_acc*100:.2f}%")

    sub = pd.DataFrame({
        'ImageId': np.arange(1, len(test) + 1),
        'Label': random_preds
    })

    assert len(sub) == len(sample_sub), "Row count mismatch!"
    assert (sub['ImageId'].values == sample_sub['ImageId'].values).all(), "ImageId mismatch!"
    assert not sub['Label'].isnull().any(), "Found null values!"

    out_path = '/root/digit-recognizer/submission_random.csv'
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path} ({len(sub)} rows).")

if __name__ == '__main__':
    main()
