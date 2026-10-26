"""Exp 04: 10-Fold Stratified Blended Ensemble (CatBoost + HistGB) for Spaceship Titanic

Features:
- Deterministic domain imputations (Surname-Planet mapping, Deck-Planet rules, CryoSleep-Spend links)
- Spatial & interaction features (Deck_Side, Route, Cabin_Region, Cabin_Num)
- Granular financial breakdown (ServiceSpend, ConsumableSpend, ZeroSpend, LogSpends, AmenityShares)
- 10-Fold Stratified Cross-Validation with multi-model probability blending
- Calibrated decision threshold on Out-Of-Fold probabilities
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def build_dataset():
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
    full.loc[full['TotalSpend_raw'] > 0, 'CryoSleep'] = full.loc[full['TotalSpend_raw'] > 0, 'CryoSleep'].fillna(False)

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
    full['LogConsumableSpend'] = np.log1p(full['ConsumableSpend'])

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

    # 6. Interaction Features
    full['Deck_Side'] = full['Cabin_Deck'].fillna('M') + '_' + full['Cabin_Side'].fillna('M')
    full['Route'] = full['HomePlanet'].fillna('M') + '_' + full['Destination'].fillna('M')

    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Deck', 'Cabin_Side', 'Cabin_Region', 'Deck_Side', 'Route']
    for c in cat_cols:
        full[c] = full[c].fillna('Missing').astype(str)

    num_cols = [
        'Age', 'GroupSize', 'IsSolo', 'FamilySize', 'Cabin_Num', 'IsChild',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'TotalSpend', 'ServiceSpend', 'ConsumableSpend', 'ZeroSpend',
        'LogTotalSpend', 'LogServiceSpend', 'LogConsumableSpend'
    ] + [f'{a}_Share' for a in amenities]

    features = cat_cols + num_cols
    
    X_train = full.iloc[:n_train][features].copy()
    y_train = train['Transported'].astype(int)
    X_test = full.iloc[n_train:][features].copy()
    test_ids = test['PassengerId'].copy()

    return X_train, y_train, X_test, test_ids, features, cat_cols

def main():
    sub_path = "/root/spaceship-titanic/submission_10fold_blend.csv"
    sample_sub_path = "/root/spaceship-titanic/sample_submission.csv"

    print("Building engineered features for 10-fold ensemble...")
    X_train, y_train, X_test, test_ids, features, cat_cols = build_dataset()
    print(f"Total features: {len(features)}")
    print(f"Categorical features: {len(cat_cols)}")

    # Encoded version for HistGB
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    for col in cat_cols:
        codes, _ = pd.factorize(pd.concat([X_train[col], X_test[col]]))
        X_train_enc[col] = codes[:len(X_train)]
        X_test_enc[col] = codes[len(X_train):]

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_train = len(X_train)
    n_test = len(X_test)

    cb_oof = np.zeros(n_train)
    hgb_oof = np.zeros(n_train)

    cb_test_preds = np.zeros((n_splits, n_test))
    hgb_test_preds = np.zeros((n_splits, n_test))

    print(f"\n--- Running {n_splits}-Fold Stratified Blended Cross-Validation ---")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Model 1: CatBoost
        cb = CatBoostClassifier(
            iterations=750,
            learning_rate=0.035,
            depth=6,
            cat_features=cat_cols,
            random_seed=42 + fold,
            verbose=0
        )
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
        cb_val_probs = cb.predict_proba(X_va)[:, 1]
        cb_oof[val_idx] = cb_val_probs
        cb_test_preds[fold] = cb.predict_proba(X_test)[:, 1]

        # Model 2: HistGradientBoosting
        X_tr_enc, X_va_enc = X_train_enc.iloc[tr_idx], X_train_enc.iloc[val_idx]
        hgb = HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.035,
            max_depth=6,
            random_state=42 + fold
        )
        hgb.fit(X_tr_enc, y_tr)
        hgb_val_probs = hgb.predict_proba(X_va_enc)[:, 1]
        hgb_oof[val_idx] = hgb_val_probs
        hgb_test_preds[fold] = hgb.predict_proba(X_test_enc)[:, 1]

        fold_blend = 0.70 * cb_val_probs + 0.30 * hgb_val_probs
        fold_acc = accuracy_score(y_va, (fold_blend >= 0.5).astype(int))
        print(f"  Fold {fold + 1:2d}: Blend Accuracy = {fold_acc:.4f} (CB best iter: {cb.get_best_iteration()})")

    cb_acc = accuracy_score(y_train, (cb_oof >= 0.5).astype(int))
    hgb_acc = accuracy_score(y_train, (hgb_oof >= 0.5).astype(int))
    raw_blend_oof = 0.70 * cb_oof + 0.30 * hgb_oof
    raw_blend_acc = accuracy_score(y_train, (raw_blend_oof >= 0.5).astype(int))

    print("\n==========================================")
    print(f"10-Fold CatBoost OOF: {cb_acc:.4f}")
    print(f"10-Fold HistGB OOF:   {hgb_acc:.4f}")
    print(f"10-Fold Raw Blend:   {raw_blend_acc:.4f}")
    print("==========================================")

    # Threshold optimization
    best_thresh = 0.50
    best_acc = raw_blend_acc
    for t in np.arange(0.46, 0.55, 0.005):
        acc = accuracy_score(y_train, (raw_blend_oof >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print(f"Optimized Threshold: {best_thresh:.3f} -> Calibrated OOF Accuracy: {best_acc:.4f}")
    print("\nOOF Classification Report (Calibrated Blend):")
    final_oof_preds = (raw_blend_oof >= best_thresh).astype(int)
    print(classification_report(y_train, final_oof_preds, target_names=['False', 'True'], digits=4))

    # Test predictions: average across all 10 folds for each model, then blend
    avg_cb_test = np.mean(cb_test_preds, axis=0)
    avg_hgb_test = np.mean(hgb_test_preds, axis=0)
    final_blend_probs = 0.70 * avg_cb_test + 0.30 * avg_hgb_test

    test_preds_binary = (final_blend_probs >= best_thresh)

    sub_df = pd.DataFrame({
        "PassengerId": test_ids,
        "Transported": test_preds_binary
    })

    sample_sub = pd.read_csv(sample_sub_path)
    assert len(sub_df) == len(sample_sub), "Row count mismatch!"
    assert (sub_df["PassengerId"].values == sample_sub["PassengerId"].values).all(), "ID alignment mismatch!"
    assert sub_df["Transported"].isnull().sum() == 0, "Null values detected!"

    sub_df.to_csv(sub_path, index=False)
    print(f"\nSaved 10-Fold calibrated blend submission to {sub_path}")
    print(f"Test prediction distribution:\n{sub_df['Transported'].value_counts(normalize=True)}")
    print("\nFirst 5 predictions:")
    print(sub_df.head())

if __name__ == "__main__":
    main()
