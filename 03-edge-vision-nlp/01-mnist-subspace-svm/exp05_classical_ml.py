import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

def main():
    print("=" * 70)
    print("MNIST Experiment 05: Classical Non-Linear Machine Learning")
    print("Models: PCA(55) + k-NN(k=5) & Extra Trees Classifier Ensemble")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1/5] Loading datasets...")
    train = pd.read_csv('/root/digit-recognizer/train.csv')
    test = pd.read_csv('/root/digit-recognizer/test.csv')
    sample_sub = pd.read_csv('/root/digit-recognizer/sample_submission.csv')

    X = train.drop(columns=['label']).values / 255.0
    y = train['label'].values
    X_test = test.values / 255.0

    print(f"  Training set : {X.shape[0]:,} images, {X.shape[1]} features (pixels)")
    print(f"  Test set     : {X_test.shape[0]:,} images, {X_test.shape[1]} features (pixels)")

    # 2. Holdout Validation Benchmark
    print("\n[Step 2/5] Running Holdout Validation Benchmark (5,000 stratified samples)...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=5000, random_state=42, stratify=y)

    # Component A: PCA + k-NN
    t0 = time.time()
    pca_val = PCA(n_components=55, random_state=42)
    X_tr_pca = pca_val.fit_transform(X_tr)
    X_val_pca = pca_val.transform(X_val)
    explained_var = pca_val.explained_variance_ratio_.sum() * 100

    knn_val = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    knn_val.fit(X_tr_pca, y_tr)
    knn_val_prob = knn_val.predict_proba(X_val_pca)
    knn_val_acc = (knn_val_prob.argmax(axis=1) == y_val).mean() * 100
    t_knn = time.time() - t0
    print(f"  PCA(55, var={explained_var:.1f}%) + k-NN(k=5): {knn_val_acc:.2f}% accuracy ({t_knn:.1f}s)")

    # Component B: Extra Trees
    t0 = time.time()
    et_val = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    et_val.fit(X_tr, y_tr)
    et_val_prob = et_val.predict_proba(X_val)
    et_val_acc = (et_val_prob.argmax(axis=1) == y_val).mean() * 100
    t_et = time.time() - t0
    print(f"  ExtraTrees(150 trees)               : {et_val_acc:.2f}% accuracy ({t_et:.1f}s)")

    # Component C: Random Forest
    t0 = time.time()
    rf_val = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf_val.fit(X_tr, y_tr)
    rf_val_prob = rf_val.predict_proba(X_val)
    rf_val_acc = (rf_val_prob.argmax(axis=1) == y_val).mean() * 100
    t_rf = time.time() - t0
    print(f"  RandomForest(150 trees)             : {rf_val_acc:.2f}% accuracy ({t_rf:.1f}s)")

    # Ensemble Blend
    blend_val_prob = 0.5 * knn_val_prob + 0.5 * et_val_prob
    blend_val_acc = (blend_val_prob.argmax(axis=1) == y_val).mean() * 100
    print(f"  --> Blended Ensemble (k-NN + ET)   : {blend_val_acc:.2f}% accuracy (+{blend_val_acc - max(knn_val_acc, et_val_acc):.2f}% gain)")

    # 3. Train on Full Dataset
    print("\n[Step 3/5] Fitting final models on all 42,000 training images...")
    
    # 3a. Full PCA + k-NN
    t0 = time.time()
    print("  Fitting PCA(55)...")
    pca_full = PCA(n_components=55, random_state=42)
    X_full_pca = pca_full.fit_transform(X)
    X_test_pca = pca_full.transform(X_test)
    full_var = pca_full.explained_variance_ratio_.sum() * 100

    print(f"  Fitting k-NN(k=5) on PCA space ({full_var:.1f}% variance)...")
    knn_full = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    knn_full.fit(X_full_pca, y)
    t_knn_full = time.time() - t0
    print(f"  PCA + k-NN fit completed in {t_knn_full:.1f}s.")

    # 3b. Full Extra Trees
    t0 = time.time()
    print("  Fitting ExtraTrees(150 trees) on raw pixel space...")
    et_full = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    et_full.fit(X, y)
    t_et_full = time.time() - t0
    print(f"  ExtraTrees fit completed in {t_et_full:.1f}s.")

    # 4. Generate Predictions on Test Set
    print("\n[Step 4/5] Generating test set predictions (28,000 images)...")
    t0 = time.time()
    test_prob_knn = knn_full.predict_proba(X_test_pca)
    print(f"  k-NN inference completed in {time.time()-t0:.1f}s.")

    t0 = time.time()
    test_prob_et = et_full.predict_proba(X_test)
    print(f"  ExtraTrees inference completed in {time.time()-t0:.1f}s.")

    test_preds_knn = test_prob_knn.argmax(axis=1)
    test_preds_et = test_prob_et.argmax(axis=1)
    agreement = (test_preds_knn == test_preds_et).mean() * 100
    print(f"  Prediction agreement between k-NN and ExtraTrees: {agreement:.2f}%")

    # Combined soft-voting ensemble
    test_prob_blend = 0.5 * test_prob_knn + 0.5 * test_prob_et
    final_preds = test_prob_blend.argmax(axis=1)

    # Inspect predicted label distributions
    unique, counts = np.unique(final_preds, return_counts=True)
    print("\n  Predicted test digit distribution:")
    for u, c in zip(unique, counts):
        print(f"    Digit {u}: {c:,} ({c/len(test)*100:.2f}%)")

    # 5. Format and Save Submission
    print("\n[Step 5/5] Formatting and verifying submission file...")
    sub = pd.DataFrame({
        'ImageId': np.arange(1, len(test) + 1),
        'Label': final_preds
    })

    assert len(sub) == len(sample_sub), f"Length mismatch: {len(sub)} vs {len(sample_sub)}"
    assert (sub['ImageId'].values == sample_sub['ImageId'].values).all(), "ImageId mismatch!"
    assert not sub['Label'].isnull().any(), "Found null values in predictions!"
    assert set(sub['Label'].unique()).issubset(set(range(10))), "Invalid label values detected!"

    out_path = '/root/digit-recognizer/submission_classical_ml.csv'
    sub.to_csv(out_path, index=False)
    print(f"  Successfully saved {len(sub):,} predictions to {out_path}.")
    print("=" * 70)

if __name__ == '__main__':
    main()
