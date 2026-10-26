import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, classification_report

def preprocess_text(df):
    keywords = df['keyword'].fillna('').astype(str).str.replace('%20', ' ')
    combined = np.where(keywords != '', keywords + ' ' + df['text'].fillna(''), df['text'].fillna(''))
    return combined

def find_best_threshold(y_true, probs):
    best_t = 0.5
    best_acc = 0.0
    for t in np.linspace(0.35, 0.65, 61):
        score = accuracy_score(y_true, (probs >= t).astype(int))
        if score > best_acc:
            best_acc = score
            best_t = t
    return best_t, best_acc


def main():
    print("=" * 65)
    print("Experiment 05: Multi-Model NLP Ensemble (Logistic + NB + Ridge)")
    print("=" * 65)

    train = pd.read_csv('/root/nlp-getting-started/train.csv')
    test = pd.read_csv('/root/nlp-getting-started/test.csv')
    sample_sub = pd.read_csv('/root/nlp-getting-started/sample_submission.csv')

    train_text = preprocess_text(train)
    test_text = preprocess_text(test)
    y = train['target'].values

    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lr = np.zeros(len(train))
    test_lr = np.zeros(len(test))

    oof_nb = np.zeros(len(train))
    test_nb = np.zeros(len(test))

    oof_ridge = np.zeros(len(train))
    test_ridge = np.zeros(len(test))

    print(f"Training 5-Fold Stratified Ensemble Models...")
    t0 = time.time()

    for fold, (trn_idx, val_idx) in enumerate(cv.split(train_text, y)):
        f_t0 = time.time()
        # 1. TF-IDF for Logistic Regression and Ridge
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X_trn_tfidf = tfidf.fit_transform(train_text[trn_idx])
        X_val_tfidf = tfidf.transform(train_text[val_idx])
        X_tst_tfidf = tfidf.transform(test_text)

        # 2. CountVec for Naive Bayes
        cnt = CountVectorizer(ngram_range=(1, 2), min_df=2)
        X_trn_cnt = cnt.fit_transform(train_text[trn_idx])
        X_val_cnt = cnt.transform(train_text[val_idx])
        X_tst_cnt = cnt.transform(test_text)

        # Model A: Logistic Regression (C=1.5)
        clf_lr = LogisticRegression(C=1.5, max_iter=300, random_state=42)
        clf_lr.fit(X_trn_tfidf, y[trn_idx])
        oof_lr[val_idx] = clf_lr.predict_proba(X_val_tfidf)[:, 1]
        test_lr += clf_lr.predict_proba(X_tst_tfidf)[:, 1] / n_splits

        # Model B: Multinomial Naive Bayes
        clf_nb = MultinomialNB(alpha=1.0)
        clf_nb.fit(X_trn_cnt, y[trn_idx])
        oof_nb[val_idx] = clf_nb.predict_proba(X_val_cnt)[:, 1]
        test_nb += clf_nb.predict_proba(X_tst_cnt)[:, 1] / n_splits

        # Model C: Ridge Classifier (Sigmoid transformed decision function)
        base_ridge = RidgeClassifier(alpha=1.0, random_state=42)
        base_ridge.fit(X_trn_tfidf, y[trn_idx])
        oof_ridge[val_idx] = 1.0 / (1.0 + np.exp(-base_ridge.decision_function(X_val_tfidf)))
        test_ridge += (1.0 / (1.0 + np.exp(-base_ridge.decision_function(X_tst_tfidf)))) / n_splits

        f_time = time.time() - f_t0
        print(f"  Fold {fold+1}/{n_splits} complete ({f_time:.2f}s)")

    train_time = time.time() - t0
    print(f"\nIndividual Out-Of-Fold F1 Scores (Default 0.50 Threshold):")
    print(f"  Logistic Regression: {f1_score(y, (oof_lr >= 0.5).astype(int)):.4f}")
    print(f"  Multinomial NB:      {f1_score(y, (oof_nb >= 0.5).astype(int)):.4f}")
    print(f"  Sigmoid Ridge:       {f1_score(y, (oof_ridge >= 0.5).astype(int)):.4f}")

    # Optimize weights
    print("\nGrid Search over Ensemble Weights (Optimizing Categorical Accuracy)...")
    best_weights = None
    best_acc = 0.0
    best_t = 0.5

    for w_lr in np.linspace(0.2, 0.8, 7):
        for w_nb in np.linspace(0.0, 0.4, 5):
            w_rd = 1.0 - w_lr - w_nb
            if w_rd < -1e-5:
                continue
            w_rd = max(0.0, w_rd)
            blend_oof = w_lr * oof_lr + w_nb * oof_nb + w_rd * oof_ridge
            t_cand, acc_cand = find_best_threshold(y, blend_oof)
            if acc_cand > best_acc:
                best_acc = acc_cand
                best_t = t_cand
                best_weights = (w_lr, w_nb, w_rd)

    print("-" * 65)
    print(f"Optimal Weights: LR={best_weights[0]:.2f}, NB={best_weights[1]:.2f}, Ridge={best_weights[2]:.2f}")
    print(f"Optimal Decision Threshold: {best_t:.3f}")
    blend_oof_final = best_weights[0] * oof_lr + best_weights[1] * oof_nb + best_weights[2] * oof_ridge
    oof_preds_final = (blend_oof_final >= best_t).astype(int)
    final_f1 = f1_score(y, oof_preds_final)
    final_acc = accuracy_score(y, oof_preds_final)
    print(f"Ensemble 5-Fold OOF Accuracy: {final_acc*100:.2f}%")
    print(f"Ensemble 5-Fold OOF F1-Score: {final_f1:.4f}")
    print(f"Total Ensemble Run Time:     {train_time:.2f}s")
    print("-" * 65)

    print("\nOOF Classification Report:")
    print(classification_report(y, oof_preds_final, target_names=['Not Disaster', 'Disaster'], digits=4))

    # Blend test predictions
    test_blend = best_weights[0] * test_lr + best_weights[1] * test_nb + best_weights[2] * test_ridge
    test_preds = (test_blend >= best_t).astype(int)
    print(f"Predicted Disaster ratio on test set: {np.mean(test_preds)*100:.2f}%")

    sub = pd.DataFrame({'id': test['id'], 'target': test_preds})
    assert len(sub) == len(sample_sub), "Row count mismatch!"
    assert (sub['id'].values == sample_sub['id'].values).all(), "ID mismatch!"
    assert not sub['target'].isnull().any(), "Found null values!"

    out_path = '/root/nlp-getting-started/submission_ensemble.csv'
    sub.to_csv(out_path, index=False)
    print(f"Saved ensemble submission to {out_path} ({len(sub)} rows).")

if __name__ == '__main__':
    main()
