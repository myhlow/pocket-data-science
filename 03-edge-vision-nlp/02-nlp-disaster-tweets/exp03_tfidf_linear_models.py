import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report

def preprocess_text(df):
    keywords = df['keyword'].fillna('').astype(str).str.replace('%20', ' ')
    combined = np.where(keywords != '', keywords + ' ' + df['text'].fillna(''), df['text'].fillna(''))
    return combined

def find_best_threshold(y_true, probs):
    best_t = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.30, 0.70, 41):
        score = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t
    return best_t, best_f1

def main():
    print("=" * 65)
    print("Experiment 03: Sublinear TF-IDF + Linear Models (Logistic & Ridge)")
    print("=" * 65)

    train = pd.read_csv('/root/nlp-getting-started/train.csv')
    test = pd.read_csv('/root/nlp-getting-started/test.csv')
    sample_sub = pd.read_csv('/root/nlp-getting-started/sample_submission.csv')

    train_text = preprocess_text(train)
    test_text = preprocess_text(test)
    y = train['target'].values

    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 1. Logistic Regression with probability calibration & threshold tuning
    print("\n[Model 1] TF-IDF (1,2-grams) + LogisticRegression (C=1.5)...")
    oof_lr_probs = np.zeros(len(train))
    test_lr_probs = np.zeros(len(test))
    t0 = time.time()

    for fold, (trn_idx, val_idx) in enumerate(cv.split(train_text, y)):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X_trn = vec.fit_transform(train_text[trn_idx])
        X_val = vec.transform(train_text[val_idx])
        X_tst = vec.transform(test_text)

        clf = LogisticRegression(C=1.5, max_iter=300, random_state=42)
        clf.fit(X_trn, y[trn_idx])

        val_probs = clf.predict_proba(X_val)[:, 1]
        oof_lr_probs[val_idx] = val_probs
        test_lr_probs += clf.predict_proba(X_tst)[:, 1] / n_splits

    lr_time = time.time() - t0
    def_lr_f1 = f1_score(y, (oof_lr_probs >= 0.5).astype(int))
    def_lr_acc = accuracy_score(y, (oof_lr_probs >= 0.5).astype(int))
    best_t_lr, best_lr_f1 = find_best_threshold(y, oof_lr_probs)
    best_lr_acc = accuracy_score(y, (oof_lr_probs >= best_t_lr).astype(int))

    print(f"  Default Threshold (0.50): OOF F1: {def_lr_f1:.4f} | Acc: {def_lr_acc*100:.2f}%")
    print(f"  Optimized Threshold ({best_t_lr:.2f}): OOF F1: {best_lr_f1:.4f} | Acc: {best_lr_acc*100:.2f}%")
    print(f"  Completed in {lr_time:.2f} seconds.")

    test_preds_lr = (test_lr_probs >= best_t_lr).astype(int)
    sub_lr = pd.DataFrame({'id': test['id'], 'target': test_preds_lr})
    assert len(sub_lr) == len(sample_sub)
    sub_lr.to_csv('/root/nlp-getting-started/submission_tfidf_logistic.csv', index=False)
    print("  Saved: /root/nlp-getting-started/submission_tfidf_logistic.csv")

    # 2. Ridge Classifier
    print("\n[Model 2] TF-IDF (1,2-grams) + RidgeClassifier (alpha=1.0)...")
    oof_ridge_preds = np.zeros(len(train), dtype=int)
    test_ridge_scores = np.zeros(len(test))
    t0 = time.time()

    for fold, (trn_idx, val_idx) in enumerate(cv.split(train_text, y)):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X_trn = vec.fit_transform(train_text[trn_idx])
        X_val = vec.transform(train_text[val_idx])
        X_tst = vec.transform(test_text)

        clf = RidgeClassifier(alpha=1.0, random_state=42)
        clf.fit(X_trn, y[trn_idx])

        oof_ridge_preds[val_idx] = clf.predict(X_val)
        test_ridge_scores += clf.decision_function(X_tst) / n_splits

    ridge_time = time.time() - t0
    ridge_f1 = f1_score(y, oof_ridge_preds)
    ridge_acc = accuracy_score(y, oof_ridge_preds)
    print(f"  Ridge OOF F1: {ridge_f1:.4f} | Acc: {ridge_acc*100:.2f}%")
    print(f"  Completed in {ridge_time:.2f} seconds.")

    test_preds_ridge = (test_ridge_scores > 0).astype(int)
    sub_ridge = pd.DataFrame({'id': test['id'], 'target': test_preds_ridge})
    assert len(sub_ridge) == len(sample_sub)
    sub_ridge.to_csv('/root/nlp-getting-started/submission_tfidf_ridge.csv', index=False)
    print("  Saved: /root/nlp-getting-started/submission_tfidf_ridge.csv")

if __name__ == '__main__':
    main()
