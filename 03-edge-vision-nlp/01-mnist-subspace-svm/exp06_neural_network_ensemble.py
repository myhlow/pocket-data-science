import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

def main():
    print("=" * 70)
    print("MNIST Experiment 06 (Option B): Deep Multi-Layer Perceptron & Tri-Blend")
    print("Architectures: MLP(256, 128) + PCA(55)+k-NN(5) + ExtraTrees(150)")
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

    knn_val = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    knn_val.fit(X_tr_pca, y_tr)
    p_knn_val = knn_val.predict_proba(X_val_pca)
    acc_knn = (p_knn_val.argmax(axis=1) == y_val).mean() * 100
    print(f"  1. PCA(55) + k-NN(5)           : {acc_knn:.2f}% accuracy ({time.time()-t0:.1f}s)")

    # Component B: Extra Trees
    t0 = time.time()
    et_val = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    et_val.fit(X_tr, y_tr)
    p_et_val = et_val.predict_proba(X_val)
    acc_et = (p_et_val.argmax(axis=1) == y_val).mean() * 100
    print(f"  2. ExtraTrees(150 trees)        : {acc_et:.2f}% accuracy ({time.time()-t0:.1f}s)")

    # Component C: Multi-Layer Perceptron (MLP)
    t0 = time.time()
    mlp_val = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', max_iter=25, random_state=42)
    mlp_val.fit(X_tr, y_tr)
    p_mlp_val = mlp_val.predict_proba(X_val)
    acc_mlp = (p_mlp_val.argmax(axis=1) == y_val).mean() * 100
    print(f"  3. MLP(256, 128) Neural Network : {acc_mlp:.2f}% accuracy ({time.time()-t0:.1f}s)")

    # Ensembles
    p_exp05 = 0.5 * p_knn_val + 0.5 * p_et_val
    acc_exp05 = (p_exp05.argmax(axis=1) == y_val).mean() * 100
    print(f"\n  Baseline Exp 05 (k-NN + ET)     : {acc_exp05:.2f}%")

    p_blend_val = 0.40 * p_mlp_val + 0.30 * p_knn_val + 0.30 * p_et_val
    acc_blend = (p_blend_val.argmax(axis=1) == y_val).mean() * 100
    print(f"  --> Tri-Blend (40% MLP, 30% k-NN, 30% ET): {acc_blend:.2f}% accuracy (+{acc_blend - acc_exp05:.2f}% gain)")

    # 3. Fit on Full 42,000 Training Samples
    print("\n[Step 3/5] Fitting final models on all 42,000 training images...")
    
    # 3a. PCA + k-NN
    t0 = time.time()
    print("  Fitting PCA(55) + k-NN(5)...")
    pca_full = PCA(n_components=55, random_state=42)
    X_full_pca = pca_full.fit_transform(X)
    X_test_pca = pca_full.transform(X_test)
    knn_full = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    knn_full.fit(X_full_pca, y)
    print(f"  k-NN fit completed in {time.time()-t0:.1f}s.")

    # 3b. Extra Trees
    t0 = time.time()
    print("  Fitting ExtraTrees(150)...")
    et_full = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    et_full.fit(X, y)
    print(f"  ExtraTrees fit completed in {time.time()-t0:.1f}s.")

    # 3c. MLP Neural Network
    t0 = time.time()
    print("  Fitting MLP(256, 128) Neural Network...")
    mlp_full = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', max_iter=25, random_state=42)
    mlp_full.fit(X, y)
    print(f"  MLP fit completed in {time.time()-t0:.1f}s.")

    # 4. Generate Predictions on Test Set
    print("\n[Step 4/5] Generating test set predictions (28,000 images)...")
    t0 = time.time()
    p_test_knn = knn_full.predict_proba(X_test_pca)
    print(f"  k-NN inference: {time.time()-t0:.1f}s.")

    t0 = time.time()
    p_test_et = et_full.predict_proba(X_test)
    print(f"  ExtraTrees inference: {time.time()-t0:.1f}s.")

    t0 = time.time()
    p_test_mlp = mlp_full.predict_proba(X_test)
    print(f"  MLP inference: {time.time()-t0:.1f}s.")

    # Tri-blend predictions
    p_test_blend = 0.40 * p_test_mlp + 0.30 * p_test_knn + 0.30 * p_test_et
    final_preds = p_test_blend.argmax(axis=1)

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

    out_path = '/root/digit-recognizer/submission_mlp_ensemble.csv'
    sub.to_csv(out_path, index=False)
    print(f"  Successfully saved {len(sub):,} predictions to {out_path}.")
    print("=" * 70)

if __name__ == '__main__':
    main()
