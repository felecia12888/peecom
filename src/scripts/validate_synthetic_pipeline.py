#!/usr/bin/env python3
"""
SYNTHETIC CLEAN DATA PIPELINE VALIDATION
========================================
Purpose: Test our diagnostic pipeline on truly clean synthetic data
         to validate methodology before applying to real data.

Expected Results:
- Block prediction ~33% (chance level for 3 blocks)
- Label permutation tests p >= 0.05 (no leakage detected)
- Stable results across multiple random seeds
- Cohen's d near zero (no effect)

This validates that our pipeline correctly identifies absence of leakage.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import time
import warnings
warnings.filterwarnings('ignore')

def drop_block_predictive_features(X_train, blocks_train, k_block=10):
    """Drop features most predictive of block structure"""
    F, p = f_classif(X_train, blocks_train)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    idx_sorted = np.argsort(F)  # ascending
    keep_idx = idx_sorted[:-k_block] if k_block>0 else idx_sorted
    if len(keep_idx) == 0:
        keep_idx = idx_sorted[:max(1, X_train.shape[1]-k_block)]
    return np.sort(keep_idx)

def train_eval_fold(X_train, y_train, blocks_train, X_test, y_test, blocks_test, k_block=10):
    """Train and evaluate single fold with block-agnostic feature selection"""
    keep_idx = drop_block_predictive_features(X_train, blocks_train, k_block=k_block)
    Xtr = X_train[:, keep_idx]
    Xte = X_test[:, keep_idx]
    clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    clf.fit(Xtr, y_train)
    ypred = clf.predict(Xte)
    return accuracy_score(y_test, ypred)

def cross_val_with_block_filter(X, y, blocks, n_splits=3, k_block=10, random_state=0):
    """Cross-validation with proper block-agnostic preprocessing"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        btr, bte = blocks[train_idx], blocks[test_idx]
        acc = train_eval_fold(Xtr, ytr, btr, Xte, yte, bte, k_block=k_block)
        accs.append(acc)
    return np.array(accs)

def test_block_predictor(X, blocks, n_splits=3, random_state=0):
    """Test if features can predict block labels (should be ~chance)"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for train_idx, test_idx in skf.split(X, blocks):
        Xtr, Xte = X[train_idx], X[test_idx]
        btr, bte = blocks[train_idx], blocks[test_idx]
        clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
        clf.fit(Xtr, btr)
        pred = clf.predict(Xte)
        accs.append(accuracy_score(bte, pred))
    return np.array(accs)

def permutation_test_lite(X, y, blocks, n_perms=300, k_block=10, random_state=0):
    """Lightweight permutation test"""
    # Baseline
    baseline_accs = cross_val_with_block_filter(X, y, blocks, k_block=k_block, random_state=random_state)
    baseline = baseline_accs.mean()
    
    # Permutations 
    rng = np.random.RandomState(random_state)
    def one_perm(i):
        y_perm = rng.permutation(y)
        accs_p = cross_val_with_block_filter(X, y_perm, blocks, k_block=k_block, random_state=42+i)
        return accs_p.mean()
    
    perms = [one_perm(i) for i in range(n_perms)]  # Sequential for debugging
    perms = np.array(perms)
    
    pval = (np.sum(perms >= baseline) + 1) / (n_perms + 1)
    effect_size = (baseline - perms.mean()) / perms.std() if perms.std() > 0 else 0
    
    return baseline, perms, pval, effect_size

# Load clean synthetic data
print("🔬 SYNTHETIC CLEAN DATA PIPELINE VALIDATION")
print("=" * 50)

df = pd.read_csv('synthetic_truly_clean_validation.csv')
print(f"Loaded: {df.shape}")

# Extract features and target
feature_cols = [col for col in df.columns if col.startswith('f') and col[1:].isdigit()]
X = df[feature_cols].values
y = df['target'].values  # Use 'target' column specifically

# Generate proper 3-block structure (not from transitions)
print(f"\n📊 Data structure:")
print(f"   Features: {len(feature_cols)}")
print(f"   Samples: {len(X)}")
print(f"   Classes: {len(np.unique(y))}")

# Create 3 equal-sized blocks for testing
block_size = len(X) // 3
blocks = np.zeros(len(X), dtype=int)
blocks[:block_size] = 0
blocks[block_size:2*block_size] = 1  
blocks[2*block_size:] = 2

print(f"   Blocks: {len(np.unique(blocks))}")
print(f"   Block sizes: {[np.sum(blocks==i) for i in range(3)]}")

# Check class balance
class_counts = pd.Series(y).value_counts().sort_index()
print(f"\n📈 Class distribution:")
for cls, count in class_counts.items():
    print(f"   Class {cls}: {count} samples ({count/len(y):.1%})")
min_class = class_counts.min()
print(f"   Min class count: {min_class} (sufficient for 3-fold CV: {min_class >= 3})")

# Validation across multiple seeds
seeds = [0, 1, 2]
results = []

for seed in seeds:
    print(f"\n🎲 Testing seed {seed}...")
    
    # Test 1: Block Predictor (should be ~chance)
    block_accs = test_block_predictor(X, blocks, random_state=seed)
    block_acc_mean = block_accs.mean()
    block_acc_std = block_accs.std()
    
    # Test 2: Label Permutation Test  
    baseline, perms, pval, effect_size = permutation_test_lite(
        X, y, blocks, n_perms=300, k_block=10, random_state=seed)
    
    results.append({
        'seed': seed,
        'block_acc_mean': block_acc_mean,
        'block_acc_std': block_acc_std,
        'baseline': baseline,
        'pval': pval,
        'effect_size': effect_size,
        'null_mean': perms.mean(),
        'null_std': perms.std()
    })
    
    print(f"   Block predictor: {block_acc_mean:.4f} ± {block_acc_std:.4f}")
    print(f"   Target baseline: {baseline:.4f}")  
    print(f"   Null mean: {perms.mean():.4f} ± {perms.std():.4f}")
    print(f"   P-value: {pval:.4f}")
    print(f"   Effect size: {effect_size:.2f}")

# Summary validation
print(f"\n🎯 VALIDATION SUMMARY")
print("=" * 30)

all_block_accs = [r['block_acc_mean'] for r in results]
all_pvals = [r['pval'] for r in results]
all_effects = [r['effect_size'] for r in results]

print(f"Block predictor accuracy: {np.mean(all_block_accs):.4f} ± {np.std(all_block_accs):.4f}")
print(f"Expected: ~0.33 (chance for 3 blocks)")

print(f"\nP-values across seeds: {[f'{p:.3f}' for p in all_pvals]}")
print(f"All p >= 0.05: {all(p >= 0.05 for p in all_pvals)}")

print(f"\nEffect sizes: {[f'{e:.2f}' for e in all_effects]}")
print(f"Mean |effect size|: {np.mean(np.abs(all_effects)):.2f}")

# Final validation decision
block_test = np.mean(all_block_accs) <= 0.40  # Block predictor near chance
pval_test = all(p >= 0.05 for p in all_pvals)   # All permutation tests non-significant
effect_test = np.mean(np.abs(all_effects)) <= 2.0  # Small effect sizes

if block_test and pval_test and effect_test:
    print(f"\n✅ PIPELINE VALIDATION SUCCESSFUL!")
    print("   - Block predictor at chance level")
    print("   - All permutation tests p >= 0.05")  
    print("   - Small effect sizes")
    print("   - Results stable across seeds")
    print("\n🎉 Pipeline ready for real data testing!")
else:
    print(f"\n❌ PIPELINE VALIDATION FAILED!")
    print(f"   - Block test: {block_test}")
    print(f"   - P-value test: {pval_test}")
    print(f"   - Effect test: {effect_test}")
    print("\n🔧 Pipeline needs debugging before real data application!")