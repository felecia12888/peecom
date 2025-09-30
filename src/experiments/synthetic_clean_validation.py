#!/usr/bin/env python3
"""
SYNTHETIC CLEAN DATA PIPELINE VALIDATION
========================================

Purpose: Validate our diagnostic pipeline works correctly on clean synthetic data.

Test Criteria (must all pass):
A. Block predictor ≈ chance (≤ 0.36 for 3 blocks)
B. Target models at chance level (p >= 0.05 for permutation tests)  
C. Reproducible across 3 seeds [0, 1, 2]

Expected Results:
- Block prediction: ~33% accuracy (chance level)
- RandomForest: p >= 0.05 (no significant leakage)
- Effect sizes near zero
- Consistent across seeds

If this passes → pipeline validated, proceed to real data remediation
If this fails → pipeline bug detected, must fix before continuing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import time
import matplotlib.pyplot as plt
import seaborn as sns

def drop_block_predictive_features(X_train, blocks_train, k_block=10):
    """Drop k_block most block-predictive features (same as pilot)"""
    F, p = f_classif(X_train, blocks_train)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    idx_sorted = np.argsort(F)  # ascending
    keep_idx = idx_sorted[:-k_block] if k_block>0 else idx_sorted
    if len(keep_idx) == 0:
        keep_idx = idx_sorted[:max(1, X_train.shape[1]-k_block)]
    return np.sort(keep_idx)

def train_eval_fold(X_train, y_train, blocks_train, X_test, y_test, blocks_test, k_block=10, clf_class=None):
    """Train and evaluate single fold with proper preprocessing"""
    keep_idx = drop_block_predictive_features(X_train, blocks_train, k_block=k_block)
    Xtr = X_train[:, keep_idx]
    Xte = X_test[:, keep_idx]
    clf = clf_class() if clf_class else RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=0)
    clf.fit(Xtr, y_train)
    ypred = clf.predict(Xte)
    return accuracy_score(y_test, ypred)

def cross_val_with_block_filter(X, y, blocks, n_splits=3, k_block=10, clf_class=None, random_state=0):
    """Cross-validation with proper block-aware feature filtering"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        btr, bte = blocks[train_idx], blocks[test_idx]
        acc = train_eval_fold(Xtr, ytr, btr, Xte, yte, bte, k_block=k_block, clf_class=clf_class)
        accs.append(acc)
    return np.array(accs)

def test_block_predictor(X, blocks, n_splits=3, random_state=0):
    """Test if block predictor achieves chance level performance"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    
    for train_idx, test_idx in skf.split(X, blocks):  # Use blocks as target for stratification
        Xtr, Xte = X[train_idx], X[test_idx]
        btr, bte = blocks[train_idx], blocks[test_idx]
        
        # Simple RF for block prediction
        clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=0)
        clf.fit(Xtr, btr)
        bpred = clf.predict(Xte)
        acc = accuracy_score(bte, bpred)
        accs.append(acc)
    
    return np.array(accs)

def run_permutation_test(X, y, blocks, n_perms=300, k_block=10, random_state=0):
    """Run label permutation test with reduced permutation count"""
    print(f"Running {n_perms} permutation tests...")
    
    # Baseline
    baseline_accs = cross_val_with_block_filter(X, y, blocks, k_block=k_block, random_state=random_state)
    baseline = baseline_accs.mean()
    
    # Permutations
    rng = np.random.RandomState(random_state)
    
    def one_perm(i):
        seed = random_state + i + 1000  # Ensure different seeds
        y_perm = rng.permutation(y)
        accs_p = cross_val_with_block_filter(X, y_perm, blocks, k_block=k_block, random_state=seed)
        return accs_p.mean()
    
    t0 = time.time()
    perms = Parallel(n_jobs=4)(delayed(one_perm)(i) for i in range(n_perms))
    perms = np.array(perms)
    elapsed = time.time() - t0
    
    # Calculate p-value and effect size
    p_value = (np.sum(perms >= baseline) + 1) / (n_perms + 1)
    effect_size = (baseline - perms.mean()) / perms.std() if perms.std() > 0 else 0
    
    print(f"Permutation test completed in {elapsed:.1f}s")
    
    return baseline, perms, p_value, effect_size

def validate_pipeline_on_clean_data(data_file, seeds=[0, 1, 2], k_block=10, n_perms=300):
    """Main validation function"""
    
    print("=" * 80)
    print("🧪 SYNTHETIC CLEAN DATA PIPELINE VALIDATION")
    print("=" * 80)
    print(f"Dataset: {data_file}")
    print(f"Seeds: {seeds}")
    print(f"Features to drop: {k_block}")
    print(f"Permutations: {n_perms}")
    print()
    
    # Load clean synthetic data
    df = pd.read_csv(data_file)
    print(f"Loaded clean synthetic data: {df.shape}")
    
    feature_cols = [col for col in df.columns if col.startswith('f') and col[1:].isdigit()]
    X = df[feature_cols].values
    y = df['target'].values
    
    # Generate blocks using ORIGINAL structure (not class transitions for clean data)
    # Clean data has randomized classes, so transitions don't reflect true blocks
    print("Using original 3-block structure from synthetic generation...")
    blocks = np.zeros(len(y), dtype=int)
    
    # Original block boundaries: [733, 731, 741] samples
    block_boundaries = [0, 733, 1464, 2205]
    for block_idx in range(3):
        start_idx = block_boundaries[block_idx]
        end_idx = block_boundaries[block_idx + 1]
        blocks[start_idx:end_idx] = block_idx
    
    print(f"Features: {len(feature_cols)}, Classes: {len(np.unique(y))}, Original Blocks: {len(np.unique(blocks))}")
    
    # Check baseline block-class correlation
    block_class_acc = np.mean(blocks == y)
    print(f"Block==Class accuracy: {block_class_acc:.4f} (should be ~0.33 for clean data)")
    
    results = {}
    
    # Test across multiple seeds
    for seed in seeds:
        print(f"\n" + "="*50)
        print(f"🔬 TESTING SEED {seed}")
        print(f"="*50)
        
        # A. Block predictor test
        print(f"\n📍 A. Block Predictor Test (seed={seed})")
        block_accs = test_block_predictor(X, blocks, random_state=seed)
        block_mean = block_accs.mean()
        block_std = block_accs.std()
        
        print(f"Block prediction accuracy: {block_mean:.4f} ± {block_std:.4f}")
        
        # Check criteria A: ≤ 0.36 for 3 blocks
        block_pass = block_mean <= 0.36
        print(f"Criteria A (≤ 0.36): {'✅ PASS' if block_pass else '❌ FAIL'}")
        
        # B & C. Target model permutation test
        print(f"\n🎯 B & C. Target Model Permutation Test (seed={seed})")
        baseline, perms, p_value, effect_size = run_permutation_test(
            X, y, blocks, n_perms=n_perms, k_block=k_block, random_state=seed
        )
        
        print(f"Baseline accuracy: {baseline:.4f}")
        print(f"Null mean accuracy: {perms.mean():.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Effect size (Cohen's d): {effect_size:.2f}")
        
        # Check criteria B & C: p >= 0.05 and small effect size
        p_pass = p_value >= 0.05
        effect_pass = abs(effect_size) <= 1.0  # Small to medium effect
        
        print(f"Criteria B (p >= 0.05): {'✅ PASS' if p_pass else '❌ FAIL'}")
        print(f"Criteria C (|effect| <= 1.0): {'✅ PASS' if effect_pass else '❌ FAIL'}")
        
        # Store results
        results[seed] = {
            'block_acc': block_mean,
            'block_std': block_std,
            'block_pass': block_pass,
            'baseline_acc': baseline,
            'null_mean': perms.mean(),
            'null_std': perms.std(),
            'p_value': p_value,
            'effect_size': effect_size,
            'p_pass': p_pass,
            'effect_pass': effect_pass,
            'all_pass': block_pass and p_pass and effect_pass
        }
    
    # Summary across all seeds
    print(f"\n" + "="*80)
    print("📊 SUMMARY ACROSS ALL SEEDS")
    print(f"="*80)
    
    all_pass_seeds = [seed for seed, res in results.items() if res['all_pass']]
    
    print(f"\nSeed Results:")
    for seed in seeds:
        res = results[seed]
        status = "✅ ALL PASS" if res['all_pass'] else "❌ SOME FAIL"
        print(f"  Seed {seed}: Block={res['block_acc']:.3f}, p={res['p_value']:.3f}, effect={res['effect_size']:.2f} → {status}")
    
    # Overall validation result
    overall_pass = len(all_pass_seeds) == len(seeds)
    
    print(f"\n🏆 PIPELINE VALIDATION RESULT:")
    if overall_pass:
        print("✅ SUCCESS: Pipeline validated on clean data!")
        print("   → All criteria (A, B, C, D) met across all seeds")
        print("   → Safe to proceed with real data remediation")
    else:
        print("❌ FAILURE: Pipeline validation failed!")
        print("   → Must fix pipeline bugs before proceeding")
        print(f"   → Passed seeds: {all_pass_seeds} / {seeds}")
    
    return results, overall_pass

if __name__ == "__main__":
    # Run validation on TRULY clean synthetic data
    results, success = validate_pipeline_on_clean_data(
        'synthetic_truly_clean_validation.csv',
        seeds=[0, 1, 2],
        k_block=10,
        n_perms=300
    )
    
    if success:
        print("\n🚀 Ready to proceed with real data remediation!")
    else:
        print("\n🔧 Pipeline needs debugging before real data work!")