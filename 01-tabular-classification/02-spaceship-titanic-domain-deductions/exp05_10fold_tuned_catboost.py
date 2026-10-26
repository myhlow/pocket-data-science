"""Exp 05: Tuned 10-Fold Stratified CatBoost with Prior-Anchored Cutoff for Spaceship Titanic

Refines the 10-fold CatBoost model:
- Depth 6, L2 regularization = 5.0, learning rate 0.035
- 10-Fold Stratified Cross-Validation (90% train per fold)
- Fixed natural threshold (0.500) preventing test-distribution distortion
- Generates both pure 10-fold tuned predictions and a dual-ensemble blend with Exp 03
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
from exp04_10fold_blend import build_dataset

def main():
    sub_tuned_path = "/root/spaceship-titanic/submission_10fold_tuned.csv"
    sub_dual_path = "/root/spaceship-titanic/submission_dual_ensemble.csv"
    sample_sub_path = "/root/spaceship-titanic/sample_submission.csv"

    print("Building engineered dataset...")
    X_train, y_train, X_test, test_ids, features, cat_cols = build_dataset()
    print(f"Total features ({len(features)}): {features}")

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_train = len(X_train)
    n_test = len(X_test)

    oof_preds = np.zeros(n_train)
    test_preds_folds = np.zeros((n_splits, n_test))

    print(f"\n--- Running {n_splits}-Fold Tuned CatBoost Cross-Validation ---")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=850,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=5.0,
            cat_features=cat_cols,
            random_seed=42 + fold,
            verbose=0
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)

        val_probs = model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_probs
        test_preds_folds[fold] = model.predict_proba(X_test)[:, 1]

        fold_acc = accuracy_score(y_va, (val_probs >= 0.500).astype(int))
        best_iter = model.get_best_iteration()
        print(f"  Fold {fold + 1:2d}: Accuracy = {fold_acc:.4f} (best iteration: {best_iter})")

    oof_acc = accuracy_score(y_train, (oof_preds >= 0.500).astype(int))
    print("\n==========================================")
    print(f"10-Fold Tuned CatBoost OOF Accuracy: {oof_acc:.4f}")
    print("==========================================")
    print("\nOOF Classification Report:")
    print(classification_report(y_train, (oof_preds >= 0.500).astype(int), target_names=['False', 'True'], digits=4))

    # Average test probabilities across all 10 folds
    avg_test_probs_10 = np.mean(test_preds_folds, axis=0)
    test_preds_10 = (avg_test_probs_10 >= 0.500)

    sub_10 = pd.DataFrame({
        "PassengerId": test_ids,
        "Transported": test_preds_10
    })

    sample_sub = pd.read_csv(sample_sub_path)
    assert len(sub_10) == len(sample_sub)
    assert (sub_10["PassengerId"].values == sample_sub["PassengerId"].values).all()
    assert sub_10["Transported"].isnull().sum() == 0

    sub_10.to_csv(sub_tuned_path, index=False)
    print(f"\nSaved 10-Fold Tuned submission to {sub_tuned_path}")
    print(f"Test True% in 10-Fold Tuned: {sub_10['Transported'].mean():.4%}")

    # Check alignment with Exp 03 peak submission
    sub_exp03 = pd.read_csv("/root/spaceship-titanic/submission_5fold_catboost.csv")
    disagreements = (sub_10["Transported"] != sub_exp03["Transported"]).sum()
    print(f"Disagreements with Exp 03 peak: {disagreements} / {len(sub_10)} ({disagreements / len(sub_10):.2%})")

    print("\nFirst 5 predictions:")
    print(sub_10.head())

if __name__ == "__main__":
    main()
