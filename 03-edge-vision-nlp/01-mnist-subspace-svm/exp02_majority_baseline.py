import pandas as pd
import numpy as np

def main():
    print("Loading data for MNIST Majority Class Baseline...")
    train = pd.read_csv('/root/digit-recognizer/train.csv')
    test = pd.read_csv('/root/digit-recognizer/test.csv')
    sample_sub = pd.read_csv('/root/digit-recognizer/sample_submission.csv')

    # Find the single most frequent digit in train
    top_digit = int(train['label'].mode()[0])
    count = int(train['label'].value_counts().max())
    freq = count / len(train)
    print(f"Majority digit in training set: {top_digit} ({count:,} occurrences, {freq*100:.2f}%)")

    # Predict constant majority digit for all test images
    sub = pd.DataFrame({
        'ImageId': np.arange(1, len(test) + 1),
        'Label': top_digit
    })

    assert len(sub) == len(sample_sub), "Row count mismatch!"
    assert (sub['ImageId'].values == sample_sub['ImageId'].values).all(), "ImageId mismatch!"
    assert not sub['Label'].isnull().any(), "Found null values!"

    out_path = '/root/digit-recognizer/submission_majority.csv'
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path} ({len(sub)} rows).")

if __name__ == '__main__':
    main()
