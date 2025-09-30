# quick_synthetic_validation.py - FAST validation with reduced permutations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
import time

print("🚀 QUICK SYNTHETIC VALIDATION (30 permutations)")
print("=" * 50)

# Load perfectly clean synthetic data
df = pd.read_csv('synthetic_perfectly_clean.csv')
feature_cols = [col for col in df.columns if col.startswith('f')]
X = df[feature_cols].values
y = df['target'].values

# Generate block structure (3 equal blocks as created by synthetic generator)
n_samples = len(y)
block_size = n_samples // 3
blocks = np.zeros(n_samples, dtype=int)
blocks[:block_size] = 0
blocks[block_size:2*block_size] = 1
blocks[2*block_size:] = 2

print(f"Loaded: Features={len(feature_cols)}, Samples={len(X)}, Classes={len(np.unique(y))}, Blocks={len(np.unique(blocks))}")
print(f"Block sizes: {[np.sum(blocks == i) for i in range(3)]}")

def drop_block_predictive_features(X_train, blocks_train, k_block=10):
    """Drop k_block most block-predictive features"""
    F, p = f_classif(X_train, blocks_train)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    idx_sorted = np.argsort(F)  # ascending
    keep_idx = idx_sorted[:-k_block] if k_block > 0 else idx_sorted
    if len(keep_idx) == 0:
        keep_idx = idx_sorted[:max(1, X_train.shape[1] - k_block)]
    return np.sort(keep_idx)

def pipeline_cv_accuracy(X, y, blocks, k_block=10, n_splits=3, random_state=0):
    """Run proper pipeline CV with block-agnostic feature selection"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies = []
    
    for train_idx, test_idx in skf.split(X, y):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        blocks_train = blocks[train_idx]
        
        # Feature selection on training data only
        keep_idx = drop_block_predictive_features(X_train, blocks_train, k_block=k_block)
        X_train_selected = X_train[:, keep_idx]
        X_test_selected = X_test[:, keep_idx]
        
        # Train and evaluate
        clf = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=random_state)  # reduced trees
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    return np.array(accuracies)

# Quick validation with single seed
print("\n🧪 SINGLE SEED VALIDATION (seed=42)")
print("-" * 40)

# 1. Block predictor test
print("1. Block Predictor Test...")
block_accs = pipeline_cv_accuracy(X, blocks, blocks, k_block=10, random_state=42)  # predict blocks
block_mean = block_accs.mean()
print(f"   Block prediction accuracy: {block_mean:.4f} ± {block_accs.std():.4f}")
print(f"   Expected for 3 blocks: ~0.333")
print(f"   ✅ PASS" if block_mean <= 0.36 else "❌ FAIL")

# 2. Target baseline
print("2. Target Baseline...")
target_accs = pipeline_cv_accuracy(X, y, blocks, k_block=10, random_state=42)
target_baseline = target_accs.mean()
print(f"   Target CV accuracy: {target_baseline:.4f} ± {target_accs.std():.4f}")

# 3. Reduced permutation test (30 permutations)
print("3. Permutation Test (30 permutations)...")
n_perms = 30
rng = np.random.RandomState(42)
perm_accs = []

start_time = time.time()
for i in range(n_perms):
    y_perm = rng.permutation(y)
    perm_acc = pipeline_cv_accuracy(X, y_perm, blocks, k_block=10, random_state=42+i)
    perm_accs.append(perm_acc.mean())
    if (i + 1) % 10 == 0:
        print(f"   Completed {i+1}/{n_perms} permutations...")

perm_accs = np.array(perm_accs)
perm_mean = perm_accs.mean()
perm_std = perm_accs.std()
p_value = (np.sum(perm_accs >= target_baseline) + 1) / (n_perms + 1)
effect_size = (target_baseline - perm_mean) / perm_std if perm_std > 0 else 0

print(f"\n📊 RESULTS SUMMARY:")
print(f"   Baseline: {target_baseline:.4f}")
print(f"   Null mean: {perm_mean:.4f} ± {perm_std:.4f}")
print(f"   P-value: {p_value:.4f}")
print(f"   Effect size: {effect_size:.2f}")
print(f"   Time: {time.time() - start_time:.1f} seconds")

# Decision
print(f"\n🎯 PIPELINE VALIDATION:")
block_pass = block_mean <= 0.36
p_pass = p_value >= 0.05
effect_pass = abs(effect_size) < 2.0

print(f"   Block predictor ≤ 0.36: {block_pass} {'✅' if block_pass else '❌'}")
print(f"   P-value ≥ 0.05: {p_pass} {'✅' if p_pass else '❌'}")
print(f"   |Effect size| < 2.0: {effect_pass} {'✅' if effect_pass else '❌'}")

overall_pass = block_pass and p_pass and effect_pass
print(f"\n{'🎉 PIPELINE VALIDATION PASSED!' if overall_pass else '⚠️  PIPELINE ISSUES DETECTED'}")

if overall_pass:
    print("✅ Pipeline is working correctly on clean synthetic data")
    print("✅ Ready to proceed with real data remediation")
else:
    print("❌ Pipeline has issues that need fixing before proceeding")