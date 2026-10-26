import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def main():
    print("=" * 60)
    print("Experiment 01: Disaster Tweets Naive Baselines (Random & Majority)")
    print("=" * 60)

    train_path = '/root/nlp-getting-started/train.csv'
    test_path = '/root/nlp-getting-started/test.csv'
    sample_sub_path = '/root/nlp-getting-started/sample_submission.csv'

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    print(f"Train samples: {len(train):,}")
    print(f"Test samples:  {len(test):,}")

    # Target class distribution
    counts = train['target'].value_counts()
    probs = train['target'].value_counts(normalize=True)
    print("\nClass distribution in train set:")
    print(f"  Class 0 (Not Disaster): {counts[0]:,} ({probs[0]*100:.2f}%)")
    print(f"  Class 1 (Real Disaster): {counts[1]:,} ({probs[1]*100:.2f}%)")

    # 1. Majority baseline (all 0s)
    majority_preds_train = np.zeros(len(train), dtype=int)
    maj_f1 = f1_score(train['target'], majority_preds_train, zero_division=0)
    maj_acc = accuracy_score(train['target'], majority_preds_train)
    print(f"\n[Baseline 1] Majority Class (All 0s):")
    print(f"  Train Accuracy: {maj_acc*100:.2f}%")
    print(f"  Train F1-Score: {maj_f1:.4f}")

    sub_maj = pd.DataFrame({
        'id': test['id'],
        'target': np.zeros(len(test), dtype=int)
    })
    assert len(sub_maj) == len(sample_sub), "Row count mismatch!"
    assert (sub_maj['id'].values == sample_sub['id'].values).all(), "ID order mismatch!"
    sub_maj.to_csv('/root/nlp-getting-started/submission_majority.csv', index=False)
    print("  Saved: /root/nlp-getting-started/submission_majority.csv")

    # 2. Random Empirical Baseline
    np.random.seed(42)
    p1 = probs[1]
    random_preds_train = np.random.binomial(1, p1, size=len(train))
    rnd_f1 = f1_score(train['target'], random_preds_train)
    rnd_acc = accuracy_score(train['target'], random_preds_train)
    print(f"\n[Baseline 2] Empirical Random Guessing (p={p1:.3f}):")
    print(f"  Train Accuracy: {rnd_acc*100:.2f}%")
    print(f"  Train F1-Score: {rnd_f1:.4f}")

    random_preds_test = np.random.binomial(1, p1, size=len(test))
    sub_rnd = pd.DataFrame({
        'id': test['id'],
        'target': random_preds_test
    })
    assert len(sub_rnd) == len(sample_sub), "Row count mismatch!"
    assert (sub_rnd['id'].values == sample_sub['id'].values).all(), "ID order mismatch!"
    sub_rnd.to_csv('/root/nlp-getting-started/submission_random.csv', index=False)
    print("  Saved: /root/nlp-getting-started/submission_random.csv")

if __name__ == '__main__':
    main()
