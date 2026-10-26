"""Exp 03: 5-Fold Stratified CatBoost with Engineered Domain Features for Spaceship Titanic

Includes:
- Deterministic domain imputations (Surname-Planet mapping, Deck-Planet rules, CryoSleep-Spend links)
- Spatial cabin layout extraction (Deck, Num, Side, Region)
- Luxury vs amenity spend breakdown and zero-spend indicators
- Travel group dynamics (GroupId, GroupSize, IsSolo)
- 5-Fold Stratified K-Fold CV with Out-Of-Fold (OOF) blending
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier

def build_features():
    train_path = "/root/spaceship-titanic/train.csv"
    test_path = "/root/spaceship-titanic/test.csv"
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    n_train = len(train)
    full = pd.concat([train, test], ignore_index=True)

    # 1. Travel Group Features
    full['GroupId'] = full['PassengerId'].apply(lambda x: x.split('_')[0])
    full['GroupNum'] = full['PassengerId'].apply(lambda x: int(x.split('_')[1]))
    full['GroupSize'] = full['GroupId'].map(full['GroupId'].value_counts())
    full['IsSolo'] = (full['GroupSize'] == 1).astype(int)

    # 2. Family / Surname Extraction
    full['Surname'] = full['Name'].apply(lambda x: str(x).split()[-1] if pd.notnull(x) else None)
    full['FamilySize'] = full['Surname'].map(full['Surname'].value_counts()).fillna(1)

    # 3. Spatial Cabin Features
    full['Cabin_Deck'] = full['Cabin'].apply(lambda x: str(x).split('/')[0] if pd.notnull(x) else None)
    full['Cabin_Num'] = full['Cabin'].apply(lambda x: float(x.split('/')[1]) if pd.notnull(x) else np.nan)
    full['Cabin_Side'] = full['Cabin'].apply(lambda x: str(x).split('/')[2] if pd.notnull(x) else None)

    # Deterministic HomePlanet rules based on Cabin Decks
    full.loc[full['Cabin_Deck'].isin(['A', 'B', 'C', 'T']), 'HomePlanet'] = full.loc[full['Cabin_Deck'].isin(['A', 'B', 'C', 'T']), 'HomePlanet'].fillna('Europa')
    full.loc[full['Cabin_Deck'] == 'G', 'HomePlanet'] = full.loc[full['Cabin_Deck'] == 'G', 'HomePlanet'].fillna('Earth')

    # Group-level HomePlanet recovery
    for gid, grp in full.groupby('GroupId'):
        planets = grp['HomePlanet'].dropna().unique()
        if len(planets) == 1:
            full.loc[full['GroupId'] == gid, 'HomePlanet'] = full.loc[full['GroupId'] == gid, 'HomePlanet'].fillna(planets[0])

    # Surname-level HomePlanet recovery
    surname_to_planet = full.dropna(subset=['Surname', 'HomePlanet']).drop_duplicates(subset=['Surname']).set_index('Surname')['HomePlanet'].to_dict()
    full['HomePlanet'] = full['HomePlanet'].fillna(full['Surname'].map(surname_to_planet))
    full['HomePlanet'] = full['HomePlanet'].fillna('Earth')

    # Cabin Side & Deck group recovery
    for gid, grp in full.groupby('GroupId'):
        sides = grp['Cabin_Side'].dropna().unique()
        if len(sides) == 1:
            full.loc[full['GroupId'] == gid, 'Cabin_Side'] = full.loc[full['GroupId'] == gid, 'Cabin_Side'].fillna(sides[0])
        decks = grp['Cabin_Deck'].dropna().unique()
        if len(decks) == 1:
            full.loc[full['GroupId'] == gid, 'Cabin_Deck'] = full.loc[full['GroupId'] == gid, 'Cabin_Deck'].fillna(decks[0])

    # 4. Expenditure Dynamics
    amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    full['TotalSpend_raw'] = full[amenities].sum(axis=1)
    # If TotalSpend > 0, CryoSleep must be False
    full.loc[full['TotalSpend_raw'] > 0, 'CryoSleep'] = full.loc[full['TotalSpend_raw'] > 0, 'CryoSleep'].fillna(False)

    # CryoSleep passengers and children under 13 cannot spend
    for col in amenities:
        full.loc[full['CryoSleep'] == True, col] = full.loc[full['CryoSleep'] == True, col].fillna(0.0)
        full.loc[full['Age'] < 13, col] = full.loc[full['Age'] < 13, col].fillna(0.0)
        full[col] = full[col].fillna(full[col].median())

    full['TotalSpend'] = full[amenities].sum(axis=1)
    full['ZeroSpend'] = (full['TotalSpend'] == 0).astype(int)
    full['ServiceSpend'] = full['RoomService'] + full['Spa'] + full['VRDeck']
    full['ConsumableSpend'] = full['FoodCourt'] + full['ShoppingMall']
    full['LogTotalSpend'] = np.log1p(full['TotalSpend'])
    full['LogServiceSpend'] = np.log1p(full['ServiceSpend'])

    for a in amenities:
        full[f'{a}_Share'] = full[a] / (full['TotalSpend'] + 1.0)

    # 5. VIP & Demographic Rules
    full.loc[full['HomePlanet'] == 'Earth', 'VIP'] = full.loc[full['HomePlanet'] == 'Earth', 'VIP'].fillna(False)
    full.loc[full['Age'] < 18, 'VIP'] = full.loc[full['Age'] < 18, 'VIP'].fillna(False)
    full['VIP'] = full['VIP'].fillna(False).astype(str)
    full['CryoSleep'] = full['CryoSleep'].fillna(False).astype(str)

    full['Age'] = full['Age'].fillna(full['Age'].median())
    full['IsChild'] = (full['Age'] < 13).astype(int)
    full['Cabin_Region'] = (full['Cabin_Num'] // 300).fillna(-1).astype(int)
    full['Cabin_Num'] = full['Cabin_Num'].fillna(full['Cabin_Num'].median())

    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Deck', 'Cabin_Side', 'Cabin_Region']
    for c in cat_cols:
        full[c] = full[c].fillna('Missing').astype(str)

    num_cols = [
        'Age', 'GroupSize', 'IsSolo', 'FamilySize', 'Cabin_Num', 'IsChild',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'TotalSpend', 'ServiceSpend', 'ConsumableSpend', 'ZeroSpend',
        'LogTotalSpend', 'LogServiceSpend'
    ] + [f'{a}_Share' for a in amenities]

    features = cat_cols + num_cols
    
    X_train = full.iloc[:n_train][features].copy()
    y_train = train['Transported'].astype(int)
    X_test = full.iloc[n_train:][features].copy()
    test_ids = test['PassengerId'].copy()

    return X_train, y_train, X_test, test_ids, features, cat_cols

def main():
    sub_path = "/root/spaceship-titanic/submission_5fold_catboost.csv"
    sample_sub_path = "/root/spaceship-titanic/sample_submission.csv"

    print("Building engineered features...")
    X_train, y_train, X_test, test_ids, features, cat_cols = build_features()
    print(f"Total features: {len(features)}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_train = len(X_train)
    oof_preds = np.zeros(n_train)
    test_preds_folds = np.zeros((5, len(X_test)))

    print("\n--- Running 5-Fold Stratified Cross-Validation ---")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=700,
            learning_rate=0.04,
            depth=6,
            cat_features=cat_cols,
            random_seed=42 + fold,
            verbose=0
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)

        val_probs = model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_probs
        fold_acc = accuracy_score(y_va, (val_probs >= 0.5).astype(int))
        best_iter = model.get_best_iteration()
        print(f"  Fold {fold + 1}: Accuracy = {fold_acc:.4f} (best iteration: {best_iter})")

        # Test predictions for this fold
        test_preds_folds[fold] = model.predict_proba(X_test)[:, 1]

    oof_binary = (oof_preds >= 0.5).astype(int)
    overall_oof_acc = accuracy_score(y_train, oof_binary)

    print("\n==========================================")
    print(f"Overall 5-Fold OOF Accuracy: {overall_oof_acc:.4f}")
    print("==========================================")
    print("\nOOF Classification Report:")
    print(classification_report(y_train, oof_binary, target_names=['False', 'True'], digits=4))

    # Ensemble test predictions across all 5 folds
    avg_test_probs = np.mean(test_preds_folds, axis=0)
    final_test_preds = avg_test_probs >= 0.5

    sub_df = pd.DataFrame({
        "PassengerId": test_ids,
        "Transported": final_test_preds
    })

    sample_sub = pd.read_csv(sample_sub_path)
    assert len(sub_df) == len(sample_sub), "Row count mismatch!"
    assert (sub_df["PassengerId"].values == sample_sub["PassengerId"].values).all(), "ID alignment mismatch!"
    assert sub_df["Transported"].isnull().sum() == 0, "Null values detected!"

    sub_df.to_csv(sub_path, index=False)
    print(f"\nSaved 5-Fold blended submission to {sub_path}")
    print(f"Test prediction distribution:\n{sub_df['Transported'].value_counts(normalize=True)}")
    print("\nFirst 5 predictions:")
    print(sub_df.head())

if __name__ == "__main__":
    main()
