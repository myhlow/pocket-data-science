import os
import re
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier, Pool

print("=== TITANIC TOP 3% TARGET PIPELINE (Target: >= 0.80382) ===")

train_df = pd.read_csv("/root/titanic/train.csv")
test_df = pd.read_csv("/root/titanic/test.csv")

n_train = len(train_df)
n_test = len(test_df)
print(f"Loaded train: {n_train}, test: {n_test}")

df_all = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

# 1. Clean Surname
df_all['Surname'] = df_all['Name'].apply(lambda x: x.split(',')[0].strip())

# 2. Refined Title Extraction
def get_clean_title(name):
    match = re.search(r' ([A-Za-z]+)\.', name)
    if match:
        t = match.group(1)
        if t in ['Mlle', 'Ms']:
            return 'Miss'
        elif t in ['Mme', 'Lady', 'Countess', 'Dona']:
            return 'Mrs'
        elif t in ['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']:
            return 'Rare'
        elif t in ['Mr', 'Mrs', 'Miss', 'Master']:
            return t
    return 'Rare'

df_all['Title'] = df_all['Name'].apply(get_clean_title)

# 3. Family Size & Sizing Categories
df_all['FamilySize'] = df_all['SibSp'] + df_all['Parch'] + 1
df_all['IsAlone'] = (df_all['FamilySize'] == 1).astype(int)
df_all['SmallFamily'] = ((df_all['FamilySize'] >= 2) & (df_all['FamilySize'] <= 4)).astype(int)
df_all['LargeFamily'] = (df_all['FamilySize'] >= 5).astype(int)

# 4. Shared Ticket Group Size
ticket_counts = df_all['Ticket'].value_counts().to_dict()
df_all['Ticket_Count'] = df_all['Ticket'].map(ticket_counts)

# 5. Precise Fare Per Person
p3_median_fare = df_all[df_all['Pclass'] == 3]['Fare'].median()
df_all['Fare'] = df_all['Fare'].fillna(p3_median_fare)
df_all['Fare_Per_Person'] = df_all['Fare'] / df_all['Ticket_Count']
df_all['Fare_Bin'] = pd.qcut(df_all['Fare_Per_Person'], q=5, labels=False)

# 6. Age Imputation by (Title, Pclass) medians
age_medians = df_all.groupby(['Title', 'Pclass'])['Age'].median().to_dict()
overall_median_age = df_all['Age'].median()

def impute_age(row):
    if pd.isna(row['Age']):
        return age_medians.get((row['Title'], row['Pclass']), overall_median_age)
    return row['Age']

df_all['Age'] = df_all.apply(impute_age, axis=1)

# 7. Age Group Categories (Children vs Elderly vs Adult)
df_all['IsChild'] = (df_all['Age'] <= 12).astype(int)
df_all['IsSenior'] = (df_all['Age'] >= 60).astype(int)

# 8. Cabin & Deck
df_all['Deck'] = df_all['Cabin'].fillna('U').apply(lambda x: str(x)[0])
df_all['HasCabin'] = (df_all['Deck'] != 'U').astype(int)

# 9. Embarked (Fill 2 missing with 'S')
df_all['Embarked'] = df_all['Embarked'].fillna('S')

# 10. High-Precision Woman-Child Group Survival (WCGS)
# Prevent false positives by grouping strictly by (Surname, Pclass) AND Ticket
df_all['Group_Survival'] = 0.5  # Neutral default

# Pass 1: Identical Ticket Groups
for ticket, grp in df_all.groupby('Ticket'):
    if len(grp) > 1:
        train_grp = grp[grp['Survived'].notna()]
        wc = train_grp[(train_grp['Sex'] == 'female') | (train_grp['IsChild'] == 1)]
        if len(wc) > 0:
            max_s = wc['Survived'].max()
            min_s = wc['Survived'].min()
            for idx in grp.index:
                if max_s == 1.0:
                    df_all.loc[idx, 'Group_Survival'] = 1.0
                elif min_s == 0.0:
                    df_all.loc[idx, 'Group_Survival'] = 0.0

# Pass 2: (Surname, Pclass) Families for remaining 0.5 cases
for (surname, pclass), grp in df_all.groupby(['Surname', 'Pclass']):
    if len(grp) > 1:
        # only update if still unassigned (0.5)
        train_grp = grp[grp['Survived'].notna()]
        wc = train_grp[(train_grp['Sex'] == 'female') | (train_grp['IsChild'] == 1)]
        if len(wc) > 0:
            max_s = wc['Survived'].max()
            min_s = wc['Survived'].min()
            for idx in grp.index:
                if df_all.loc[idx, 'Group_Survival'] == 0.5:
                    if max_s == 1.0:
                        df_all.loc[idx, 'Group_Survival'] = 1.0
                    elif min_s == 0.0:
                        df_all.loc[idx, 'Group_Survival'] = 0.0

print(f"Group Survival distribution:\n{df_all['Group_Survival'].value_counts()}")

# Split back to train and test
train_clean = df_all.iloc[:n_train].copy()
test_clean = df_all.iloc[n_train:].copy()

# Features for CatBoost
cb_feature_cols = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'FamilySize', 'IsAlone',
    'SmallFamily', 'LargeFamily', 'Fare', 'Fare_Per_Person', 'Fare_Bin',
    'Ticket_Count', 'Embarked', 'Title', 'Deck', 'HasCabin', 'IsChild',
    'IsSenior', 'Group_Survival'
]

cb_cat_features = [
    'Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'IsAlone', 'SmallFamily',
    'LargeFamily', 'HasCabin', 'IsChild', 'IsSenior', 'Fare_Bin', 'Group_Survival'
]

for c in cb_cat_features:
    train_clean[c] = train_clean[c].astype(str)
    test_clean[c] = test_clean[c].astype(str)

X = train_clean[cb_feature_cols]
y = train_clean['Survived'].astype(int)
X_test = test_clean[cb_feature_cols]

# Numeric One-Hot Encoded features for Random Forest & Extra Trees
df_ohe = pd.get_dummies(df_all[cb_feature_cols], drop_first=True)
X_ohe = df_ohe.iloc[:n_train].copy()
X_test_ohe = df_ohe.iloc[n_train:].copy()

# 10-Fold Stratified Cross Validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_cb = np.zeros(n_train)
oof_rf = np.zeros(n_train)
oof_et = np.zeros(n_train)

test_preds_cb = np.zeros(n_test)
test_preds_rf = np.zeros(n_test)
test_preds_et = np.zeros(n_test)

print(f"\n--- Training 10-Fold Diverse Ensemble (CatBoost + RF + ExtraTrees) ---")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # 1. CatBoost (depth=4 for high generalization)
    cb = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=4.0,
        eval_metric='Accuracy',
        random_seed=42 + fold,
        verbose=False,
        cat_features=cb_cat_features,
        early_stopping_rounds=40
    )
    cb.fit(Pool(X_tr, y_tr, cat_features=cb_cat_features), eval_set=Pool(X_va, y_va, cat_features=cb_cat_features), use_best_model=True)
    oof_cb[val_idx] = cb.predict_proba(X_va)[:, 1]
    test_preds_cb += cb.predict_proba(X_test)[:, 1] / n_splits
    
    # 2. Random Forest
    X_ohe_tr, X_ohe_va = X_ohe.iloc[train_idx], X_ohe.iloc[val_idx]
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42 + fold, n_jobs=-1)
    rf.fit(X_ohe_tr, y_tr)
    oof_rf[val_idx] = rf.predict_proba(X_ohe_va)[:, 1]
    test_preds_rf += rf.predict_proba(X_test_ohe)[:, 1] / n_splits
    
    # 3. Extra Trees
    et = ExtraTreesClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42 + fold, n_jobs=-1)
    et.fit(X_ohe_tr, y_tr)
    oof_et[val_idx] = et.predict_proba(X_ohe_va)[:, 1]
    test_preds_et += et.predict_proba(X_test_ohe)[:, 1] / n_splits
    
    fold_blend = 0.60 * oof_cb[val_idx] + 0.25 * oof_rf[val_idx] + 0.15 * oof_et[val_idx]
    acc_f = accuracy_score(y_va, (fold_blend >= 0.50).astype(int))
    print(f"  Fold {fold:2d}: Blend Acc = {acc_f:.4f} (CB Acc = {accuracy_score(y_va, (oof_cb[val_idx]>=0.5).astype(int)):.4f})")

# Evaluate Individual OOF
print(f"\n--- OOF ACCURACY COMPARISON ---")
print(f"CatBoost 10-Fold OOF Accuracy:   {accuracy_score(y, (oof_cb >= 0.50).astype(int)):.4f}")
print(f"RandomForest 10-Fold OOF Acc:    {accuracy_score(y, (oof_rf >= 0.50).astype(int)):.4f}")
print(f"ExtraTrees 10-Fold OOF Acc:      {accuracy_score(y, (oof_et >= 0.50).astype(int)):.4f}")

# Weighted Blend: 60% CatBoost + 25% Random Forest + 15% Extra Trees
oof_ensemble = 0.60 * oof_cb + 0.25 * oof_rf + 0.15 * oof_et
blend_acc = accuracy_score(y, (oof_ensemble >= 0.50).astype(int))
blend_auc = roc_auc_score(y, oof_ensemble)

print(f"\n>>> 10-FOLD ENSEMBLE OOF ACCURACY: {blend_acc:.4f} ({blend_acc*100:.2f}%) | ROC-AUC: {blend_auc:.4f} <<<")

# Final Test Predictions
test_ensemble = 0.60 * test_preds_cb + 0.25 * test_preds_rf + 0.15 * test_preds_et
test_final = (test_ensemble >= 0.50).astype(int)

sub_path = "/root/titanic/submission_top3.csv"
sub = pd.DataFrame({
    'PassengerId': test_clean['PassengerId'].astype(int),
    'Survived': test_final
})
sub.to_csv(sub_path, index=False)

print(f"\nSaved Top 3% Candidate to: {sub_path} ({len(sub)} predictions)")
print(f"Predicted Test Survival Rate: {sub['Survived'].mean():.4f} ({sub['Survived'].sum()}/{len(sub)} survived)")

# Submit to Kaggle
print("\nSubmitting to Kaggle competition 'titanic'...")
submit_cmd = [
    "kaggle", "competitions", "submit",
    "-c", "titanic",
    "-f", sub_path,
    "-m", f"10-Fold Blended Ensemble (CatBoost+RF+ET, OOF: {blend_acc:.4f})"
]

res = subprocess.run(submit_cmd, capture_output=True, text=True)
print("Kaggle submit exit code:", res.returncode)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)

# Fetch latest submission status
print("\nFetching latest submission status from Kaggle...")
res_sub = subprocess.run(["kaggle", "competitions", "submissions", "-c", "titanic"], capture_output=True, text=True)
for l in res_sub.stdout.split('\n')[:7]:
    print(" ", l)
