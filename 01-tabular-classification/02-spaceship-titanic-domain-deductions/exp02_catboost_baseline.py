"""Exp 02: CatBoost Baseline with Holdout Validation for Spaceship Titanic

Trains a baseline CatBoost model using native categorical features and basic
cabin parsing (Deck, Side), evaluated on an 80/20 stratified holdout split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier

def preprocess(df, is_train=True):
    df = df.copy()
    
    # Basic Cabin split
    df['Cabin_Deck'] = df['Cabin'].apply(lambda x: str(x).split('/')[0] if pd.notnull(x) else 'Missing')
    df['Cabin_Side'] = df['Cabin'].apply(lambda x: str(x).split('/')[2] if pd.notnull(x) else 'Missing')
    
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Deck', 'Cabin_Side']
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    for c in cat_cols:
        df[c] = df[c].fillna('Missing').astype(str)
        
    for c in num_cols:
        # Fill missing with median
        df[c] = df[c].fillna(df[c].median() if is_train else 0.0)
        
    return df, cat_cols, num_cols

def main():
    train_path = "/root/spaceship-titanic/train.csv"
    test_path = "/root/spaceship-titanic/test.csv"
    sample_sub_path = "/root/spaceship-titanic/sample_submission.csv"
    sub_path = "/root/spaceship-titanic/submission_catboost.csv"

    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    processed_train, cat_cols, num_cols = preprocess(train_df, is_train=True)
    processed_test, _, _ = preprocess(test_df, is_train=False)

    feature_cols = cat_cols + num_cols
    X = processed_train[feature_cols]
    y = train_df['Transported'].astype(int)
    X_test = processed_test[feature_cols]

    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # Stratified Holdout Split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n--- Training Holdout Model (80/20 Split) ---")
    holdout_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols,
        random_seed=42,
        verbose=100
    )
    holdout_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50
    )

    val_preds = holdout_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    best_iter = holdout_model.get_best_iteration()

    print(f"\n==========================================")
    print(f"Holdout Validation Accuracy: {val_acc:.4f} (at iteration {best_iter})")
    print(f"==========================================")
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, target_names=['False', 'True'], digits=4))

    # Feature Importance
    fi = pd.Series(holdout_model.get_feature_importance(), index=feature_cols).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(fi)

    # Train final model on full training set using best iteration count
    print(f"\n--- Training Full Model on 100% Data ({best_iter} iterations) ---")
    full_model = CatBoostClassifier(
        iterations=best_iter + 1,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols,
        random_seed=42,
        verbose=0
    )
    full_model.fit(X, y)

    # Predict test set
    test_preds_binary = full_model.predict(X_test)
    test_preds_bool = test_preds_binary == 1

    sub_df = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": test_preds_bool
    })

    # Validate
    assert len(sub_df) == len(sample_sub), f"Length mismatch: {len(sub_df)} vs {len(sample_sub)}"
    assert (sub_df["PassengerId"].values == sample_sub["PassengerId"].values).all(), "ID alignment mismatch!"
    assert sub_df["Transported"].isnull().sum() == 0, "Found null predictions!"

    sub_df.to_csv(sub_path, index=False)
    print(f"\nSaved baseline predictions to {sub_path}")
    print(f"Test prediction distribution:\n{sub_df['Transported'].value_counts(normalize=True)}")
    print("\nFirst 5 predictions:")
    print(sub_df.head())

if __name__ == "__main__":
    main()
