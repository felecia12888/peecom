#!/usr/bin/env python3
"""
SIMPLE SYNTHETIC PIPELINE VALIDATION
====================================
Lightweight test of our diagnostic pipeline on truly clean synthetic data.
Focus: RandomForest only, no PEECOM spam.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def drop_block_predictive_features(X_train, blocks_train, k_block=10):
    """Drop features most predictive of block structure"""
    if k_block <= 0:
        return np.arange(X_train.shape[1])
    F, p = f_classif(X_train, blocks_train)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    idx_sorted = np.argsort(F)  # ascending
    keep_idx = idx_sorted[:-k_block]
    if len(keep_idx) == 0:
        keep_idx = idx_sorted[:max(1, X_train.shape[1]-k_block)]
    return np.sort(keep_idx)

def cv_with_block_filter(X, y, blocks, n_splits=3, k_block=10, random_state=0):
    """Cross-validation with block-agnostic feature selection"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        btr, bte = blocks[train_idx], blocks[test_idx]
        
        # Feature selection on training data only
        keep_idx = drop_block_predictive_features(Xtr, btr, k_block=k_block)
        Xtr_sel = Xtr[:, keep_idx]
        Xte_sel = Xte[:, keep_idx]
        
        # Train and predict
        clf = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=42)
        clf.fit(Xtr_sel, ytr)
        pred = clf.predict(Xte_sel)
        accs.append(accuracy_score(yte, pred))
    
    return np.array(accs)

def block_predictor_test(X, blocks, n_splits=3, random_state=0):
    """Test if features can predict blocks"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for train_idx, test_idx in skf.split(X, blocks):
        Xtr, Xte = X[train_idx], X[test_idx]
        btr, bte = blocks[train_idx], blocks[test_idx]
        clf = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=42)
        clf.fit(Xtr, btr)
        pred = clf.predict(Xte)
        accs.append(accuracy_score(bte, pred))
    return np.array(accs)

def permutation_test_simple(X, y, blocks, n_perms=100, k_block=10, random_state=0):
    """Simple permutation test"""
    # Baseline
    baseline_accs = cv_with_block_filter(X, y, blocks, k_block=k_block, random_state=random_state)
    baseline = baseline_accs.mean()
    
    # Permutations
    rng = np.random.RandomState(random_state)
    perms = []
    for i in range(n_perms):
        y_perm = rng.permutation(y)
        accs_p = cv_with_block_filter(X, y_perm, blocks, k_block=k_block, random_state=42+i)
        perms.append(accs_p.mean())
    
    perms = np.array(perms)
    pval = (np.sum(perms >= baseline) + 1) / (n_perms + 1)
    effect_size = (baseline - perms.mean()) / perms.std() if perms.std() > 0 else 0
    
    return baseline, perms, pval, effect_size

# Load clean synthetic data
print("🧪 SIMPLE SYNTHETIC VALIDATION")
print("=" * 40)

df = pd.read_csv('synthetic_truly_clean_validation.csv')
print(f"Loaded: {df.shape}")

# Extract data
feature_cols = [col for col in df.columns if col.startswith('f')]
X = df[feature_cols].values
y = df['target'].values

# Create 3 equal blocks
block_size = len(X) // 3
blocks = np.zeros(len(X), dtype=int)
blocks[:block_size] = 0
blocks[block_size:2*block_size] = 1
blocks[2*block_size:] = 2

print(f"Features: {len(feature_cols)}, Classes: {len(np.unique(y))}, Blocks: {len(np.unique(blocks))}")

# Quick validation across 2 seeds
results = []
for seed in [0, 1]:
    print(f"\n🎲 Seed {seed}:")
    
    # Block predictor test
    block_accs = block_predictor_test(X, blocks, random_state=seed)
    block_mean = block_accs.mean()
    print(f"   Block predictor: {block_mean:.4f}")
    
    # Target permutation test (reduced perms for speed)
    baseline, perms, pval, effect = permutation_test_simple(
        X, y, blocks, n_perms=50, k_block=10, random_state=seed)
    
    print(f"   Target baseline: {baseline:.4f}")
    print(f"   Null mean: {perms.mean():.4f}")
    print(f"   P-value: {pval:.4f}")
    print(f"   Effect: {effect:.2f}")
    
    results.append({
        'block_acc': block_mean,
        'pval': pval,
        'effect': abs(effect)
    })

# Summary
print(f"\n📊 SUMMARY:")
block_accs = [r['block_acc'] for r in results]
pvals = [r['pval'] for r in results]
effects = [r['effect'] for r in results]

print(f"Block accuracy: {np.mean(block_accs):.4f} (target: ~0.33)")
print(f"P-values: {pvals} (target: all >= 0.05)")
print(f"Effect sizes: {[f'{e:.2f}' for e in effects]} (target: small)")

# Validation check
block_ok = np.mean(block_accs) <= 0.40
pval_ok = all(p >= 0.05 for p in pvals)
effect_ok = np.mean(effects) <= 2.0

if block_ok and pval_ok and effect_ok:
    print(f"\n✅ PIPELINE VALIDATION SUCCESS!")
    print("Ready for real data testing.")
else:
    print(f"\n❌ VALIDATION ISSUES:")
    print(f"   Block test: {block_ok}")
    print(f"   P-value test: {pval_ok}")
    print(f"   Effect test: {effect_ok}")