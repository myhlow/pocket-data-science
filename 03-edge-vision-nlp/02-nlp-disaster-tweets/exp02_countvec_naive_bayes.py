import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, classification_report

def preprocess_text(df):
    keywords = df['keyword'].fillna('').astype(str).str.replace('%20', ' ')
    locations = df['location'].fillna('').astype(str)
    # Combine keyword and text
    combined = np.where(keywords != '', keywords + ' ' + df['text'].fillna(''), df['text'].fillna(''))
    return combined

def main():
    print("=" * 60)
    print("Experiment 02: CountVectorizer + Multinomial Naive Bayes Baseline")
    print("=" * 60)

    train = pd.read_csv('/root/nlp-getting-started/train.csv')
    test = pd.read_csv('/root/nlp-getting-started/test.csv')
    sample_sub = pd.read_csv('/root/nlp-getting-started/sample_submission.csv')

    train_text = preprocess_text(train)
    test_text = preprocess_text(test)
    y = train['target'].values

    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train), dtype=int)
    oof_probs = np.zeros(len(train), dtype=float)
    test_probs = np.zeros(len(test), dtype=float)

    print(f"Running {n_splits}-Fold Stratified Cross-Validation...")
    t0 = time.time()

    for fold, (trn_idx, val_idx) in enumerate(cv.split(train_text, y)):
        f_t0 = time.time()
        vec = CountVectorizer(ngram_range=(1, 2), min_df=2)
        X_trn = vec.fit_transform(train_text[trn_idx])
        X_val = vec.transform(train_text[val_idx])
        X_test_fold = vec.transform(test_text)

        clf = MultinomialNB(alpha=1.0)
        clf.fit(X_trn, y[trn_idx])

        val_probs = clf.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        oof_probs[val_idx] = val_probs
        oof_preds[val_idx] = val_preds
        test_probs += clf.predict_proba(X_test_fold)[:, 1] / n_splits

        fold_f1 = f1_score(y[val_idx], val_preds)
        fold_acc = accuracy_score(y[val_idx], val_preds)
        f_time = time.time() - f_t0
        print(f"  Fold {fold+1}/{n_splits} | F1: {fold_f1:.4f} | Acc: {fold_acc*100:.2f}% ({f_time:.2f}s)")

    total_time = time.time() - t0
    overall_f1 = f1_score(y, oof_preds)
    overall_acc = accuracy_score(y, oof_preds)

    print("-" * 60)
    print(f"Overall 5-Fold OOF F1-Score: {overall_f1:.4f}")
    print(f"Overall 5-Fold OOF Accuracy: {overall_acc*100:.2f}%")
    print(f"Completed in {total_time:.2f} seconds.")
    print("-" * 60)
    print("\nOOF Classification Report:")
    print(classification_report(y, oof_preds, target_names=['Not Disaster', 'Disaster'], digits=4))

    # Test predictions using default threshold 0.5
    test_preds = (test_probs >= 0.5).astype(int)
    print(f"Predicted Disaster ratio on test set: {np.mean(test_preds)*100:.2f}%")

    sub = pd.DataFrame({
        'id': test['id'],
        'target': test_preds
    })
    assert len(sub) == len(sample_sub), "Row count mismatch!"
    assert (sub['id'].values == sample_sub['id'].values).all(), "ID mismatch!"
    assert not sub['target'].isnull().any(), "Found null values!"

    out_path = '/root/nlp-getting-started/submission_countvec_nb.csv'
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path} ({len(sub)} rows).")

if __name__ == '__main__':
    main()
