import pandas as pd
import numpy as np

def main():
    print("Loading data for MNIST Nearest Centroid ('Ghost Templates')...")
    train = pd.read_csv('/root/digit-recognizer/train.csv')
    test = pd.read_csv('/root/digit-recognizer/test.csv')
    sample_sub = pd.read_csv('/root/digit-recognizer/sample_submission.csv')

    X_train = train.drop(columns=['label']).values / 255.0
    y_train = train['label'].values
    X_test = test.values / 255.0

    # 1. Compute Centroid ("Ghost Template") for each digit 0 to 9
    centroids = np.zeros((10, 784))
    for d in range(10):
        centroids[d] = X_train[y_train == d].mean(axis=0)

    # 2. Local Training Evaluation
    # Cosine distance (normalizes stroke intensity differences)
    X_train_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    
    sims_train = np.dot(X_train_norm, c_norm.T)
    preds_train = np.argmax(sims_train, axis=1)
    train_acc = np.mean(preds_train == y_train)
    print(f"Training Accuracy (Cosine Nearest Centroid): {train_acc*100:.2f}%")

    # 3. Predict on Test Set
    X_test_norm = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)
    sims_test = np.dot(X_test_norm, c_norm.T)
    test_preds = np.argmax(sims_test, axis=1)

    # Check predicted test distribution
    unique, counts = np.unique(test_preds, return_counts=True)
    print("\nPredicted test digit distribution:")
    for u, c in zip(unique, counts):
        print(f"  Digit {u}: {c:,} ({c/len(test)*100:.2f}%)")

    # 4. Save Submission
    sub = pd.DataFrame({
        'ImageId': np.arange(1, len(test) + 1),
        'Label': test_preds
    })

    assert len(sub) == len(sample_sub), "Row count mismatch!"
    assert (sub['ImageId'].values == sample_sub['ImageId'].values).all(), "ImageId mismatch!"
    assert not sub['Label'].isnull().any(), "Found null values!"

    out_path = '/root/digit-recognizer/submission_centroid.csv'
    sub.to_csv(out_path, index=False)
    print(f"\nSaved submission to {out_path} ({len(sub)} rows).")

if __name__ == '__main__':
    main()
