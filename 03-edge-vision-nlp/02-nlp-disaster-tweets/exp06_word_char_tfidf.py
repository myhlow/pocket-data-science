import pandas as pd
import numpy as np
import re
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', ' http_url ', text)
    text = re.sub(r'@\w+', ' @user ', text)
    return text

def preprocess_df(df):
    kw = df['keyword'].fillna('').astype(str).str.replace('%20', ' ')
    cleaned = df['text'].fillna('').apply(clean_text)
    return np.where(kw != '', kw + ' ' + cleaned, cleaned)

def main():
    print("=" * 65)
    print("Experiment 05: Word + Character N-Gram FeatureUnion + Logistic Regression")
    print("=" * 65)

    train = pd.read_csv('/root/nlp-getting-started/train.csv')
    test = pd.read_csv('/root/nlp-getting-started/test.csv')
    sample_sub = pd.read_csv('/root/nlp-getting-started/sample_submission.csv')

    X_train = preprocess_df(train)
    X_test = preprocess_df(test)
    y = train['target'].values

    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(train))
    test_probs = np.zeros(len(test))

    print("Building Word (1,2) + Char_wb (3,5) FeatureUnion with Logistic Regression (C=1.5)...")
    t0 = time.time()

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_train, y)):
        f_t0 = time.time()
        union = FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=5, sublinear_tf=True))
        ])

        X_trn = union.fit_transform(X_train[trn_idx])
        X_val = union.transform(X_train[val_idx])
        X_tst = union.transform(X_test)

        clf = LogisticRegression(C=1.5, max_iter=300, random_state=42)
        clf.fit(X_trn, y[trn_idx])

        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]
        test_probs += clf.predict_proba(X_tst)[:, 1] / n_splits
        print(f"  Fold {fold+1}/{n_splits} complete ({time.time() - f_t0:.2f}s) | Features: {X_trn.shape[1]}")

    elapsed = time.time() - t0

    # Evaluate across thresholds
    best_t = 0.50
    best_acc = 0.0
    for t in np.linspace(0.40, 0.60, 41):
        acc = accuracy_score(y, (oof_probs >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_t = t

    def_acc = accuracy_score(y, (oof_probs >= 0.50).astype(int))
    def_f1 = f1_score(y, (oof_probs >= 0.50).astype(int))
    best_f1 = f1_score(y, (oof_probs >= best_t).astype(int))

    print("-" * 65)
    print(f"Default Threshold (0.50): OOF Acc: {def_acc*100:.2f}% | F1: {def_f1:.4f}")
    print(f"Optimal Threshold ({best_t:.3f}): OOF Acc: {best_acc*100:.2f}% | F1: {best_f1:.4f}")
    print(f"Total CV Runtime:        {elapsed:.2f}s")
    print("-" * 65)

    test_preds = (test_probs >= best_t).astype(int)
    pos_ratio = np.mean(test_preds) * 100
    print(f"Predicted Disaster ratio on test set: {pos_ratio:.2f}% ({np.sum(test_preds)} / {len(test_preds)})")

    sub = pd.DataFrame({'id': test['id'], 'target': test_preds})
    assert len(sub) == len(sample_sub)
    assert (sub['id'].values == sample_sub['id'].values).all()
    assert not sub['target'].isnull().any()

    out_file = '/root/nlp-getting-started/submission_word_char_tfidf.csv'
    sub.to_csv(out_file, index=False)
    print(f"Saved submission to {out_file} ({len(sub)} rows).")

if __name__ == '__main__':
    main()
