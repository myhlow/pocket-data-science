import pandas as pd
import numpy as np
import time
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report

def preprocess_text(df):
    keywords = df['keyword'].fillna('').astype(str).str.replace('%20', ' ')
    combined = np.where(keywords != '', keywords + ' ' + df['text'].fillna(''), df['text'].fillna(''))
    return pd.DataFrame({'text_clean': combined})

def main():
    print("=" * 65)
    print("Experiment 04: CatBoost Native Text Gradient Boosted Trees")
    print("=" * 65)

    train = pd.read_csv('/root/nlp-getting-started/train.csv')
    test = pd.read_csv('/root/nlp-getting-started/test.csv')
    sample_sub = pd.read_csv('/root/nlp-getting-started/sample_submission.csv')

    X_train = preprocess_text(train)
    X_test = preprocess_text(test)
    y = train['target'].values

    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(train))
    test_probs = np.zeros(len(test))

    print(f"Training 5-Fold CatBoost with native text features...")
    t0 = time.time()

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_train, y)):
        f_t0 = time.time()
        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.08,
            depth=6,
            text_features=['text_clean'],
            random_seed=42 + fold,
            verbose=0,
            thread_count=4
        )

        model.fit(
            X_train.iloc[trn_idx], y[trn_idx],
            eval_set=(X_train.iloc[val_idx], y[val_idx]),
            early_stopping_rounds=35,
            verbose=False
        )

        val_preds_prob = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        oof_probs[val_idx] = val_preds_prob
        test_probs += model.predict_proba(X_test)[:, 1] / n_splits

        fold_f1 = f1_score(y[val_idx], (val_preds_prob >= 0.5).astype(int))
        fold_time = time.time() - f_t0
        print(f"  Fold {fold+1}/{n_splits} | Best Iter: {model.get_best_iteration()} | F1: {fold_f1:.4f} ({fold_time:.1f}s)")

    total_time = time.time() - t0
    def_f1 = f1_score(y, (oof_probs >= 0.5).astype(int))
    def_acc = accuracy_score(y, (oof_probs >= 0.5).astype(int))

    # Find optimal threshold
    best_t = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.30, 0.70, 41):
        score = f1_score(y, (oof_probs >= t).astype(int))
        if score > best_f1:
            best_f1 = score
            best_t = t
    best_acc = accuracy_score(y, (oof_probs >= best_t).astype(int))

    print("-" * 65)
    print(f"CatBoost Default Threshold (0.50): OOF F1: {def_f1:.4f} | Acc: {def_acc*100:.2f}%")
    print(f"CatBoost Optimal Threshold ({best_t:.2f}): OOF F1: {best_f1:.4f} | Acc: {best_acc*100:.2f}%")
    print(f"Total time elapsed: {total_time:.1f}s")
    print("-" * 65)

    test_preds = (test_probs >= best_t).astype(int)
    sub = pd.DataFrame({'id': test['id'], 'target': test_preds})
    assert len(sub) == len(sample_sub)
    assert (sub['id'].values == sample_sub['id'].values).all()

    out_path = '/root/nlp-getting-started/submission_catboost.csv'
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path} ({len(sub)} rows).")

if __name__ == '__main__':
    main()
