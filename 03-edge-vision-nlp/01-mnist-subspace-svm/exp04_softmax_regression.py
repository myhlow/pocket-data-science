import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression

def main():
    print("Loading data for MNIST Multinomial Softmax Regression (Linear Model)...")
    train = pd.read_csv('/root/digit-recognizer/train.csv')
    test = pd.read_csv('/root/digit-recognizer/test.csv')
    sample_sub = pd.read_csv('/root/digit-recognizer/sample_submission.csv')

    X_train = train.drop(columns=['label']).values / 255.0
    y_train = train['label'].values
    X_test = test.values / 255.0

    print(f"Training on {X_train.shape[0]} images with 784 normalized pixel features...")
    t0 = time.time()
    clf = LogisticRegression(
        solver='lbfgs',
        max_iter=150,
        C=0.1,
        random_state=42,
        verbose=1
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s.")

    # In-sample accuracy check
    train_acc = clf.score(X_train, y_train)
    print(f"Training Accuracy: {train_acc*100:.2f}%")

    # Predict test labels
    test_preds = clf.predict(X_test)

    # Check predicted test distribution
    unique, counts = np.unique(test_preds, return_counts=True)
    print("\nPredicted test digit distribution:")
    for u, c in zip(unique, counts):
        print(f"  Digit {u}: {c:,} ({c/len(test)*100:.2f}%)")

    # Save submission
    sub = pd.DataFrame({
        'ImageId': np.arange(1, len(test) + 1),
        'Label': test_preds
    })

    assert len(sub) == len(sample_sub), "Row count mismatch!"
    assert (sub['ImageId'].values == sample_sub['ImageId'].values).all(), "ImageId mismatch!"
    assert not sub['Label'].isnull().any(), "Found null values!"

    out_path = '/root/digit-recognizer/submission_softmax.csv'
    sub.to_csv(out_path, index=False)
    print(f"\nSaved submission to {out_path} ({len(sub)} rows).")

if __name__ == '__main__':
    main()
