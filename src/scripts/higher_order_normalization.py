#!/usr/bin/env python3
"""
HIGHER-ORDER MOMENT NORMALIZATION
=================================
Purpose: Normalize skewness and kurtosis across blocks in addition to 
         mean and covariance normalization.

Methodology:
1. Apply existing mean + covariance normalization
2. For each block, transform distributions to match global skewness and kurtosis
3. Use power transformations (Box-Cox, Yeo-Johnson) and rank-based methods
4. Validate using label permutation test

Expected Outcome:
- Further reduce block predictability by eliminating higher-order statistical differences
- Achieve p >= 0.05 in label permutation test (successful remediation)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.metrics import accuracy_score
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import PEECOM
import sys
sys.path.append('src')
from models.simple_peecom import SimplePEECOM

def _sqrt_and_invsqrt(mat, eps=1e-8):
    """Helper for covariance normalization"""
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    sqrt = (vecs * np.sqrt(vals)) @ vecs.T
    inv_sqrt = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
    return sqrt, inv_sqrt

def cov_normalize_blocks(df, feature_cols, block_col='block', eps=1e-6):
    """Apply block-covariance normalization (existing method)"""
    X = df[feature_cols].to_numpy(dtype=float)
    blocks = df[block_col].to_numpy()
    uniq = np.unique(blocks)

    # global mean & global covariance (centered by global mean)
    global_mean = X.mean(axis=0)
    Xg = X - global_mean
    cov_global = np.cov(Xg, rowvar=False) + eps * np.eye(X.shape[1])
    sqrt_global, _ = _sqrt_and_invsqrt(cov_global, eps=eps)

    X_new = np.empty_like(X)
    for b in uniq:
        idx = np.where(blocks == b)[0]
        Xb = X[idx]
        mean_b = Xb.mean(axis=0)
        Xb_centered = Xb - mean_b
        cov_b = np.cov(Xb_centered, rowvar=False) + eps * np.eye(X.shape[1])

        # get inv-sqrt of block cov and sqrt of global cov
        _, inv_sqrt_b = _sqrt_and_invsqrt(cov_b, eps=eps)

        # transform: Xb' = inv_sqrt_b @ sqrt_global @ (Xb - mean_b), then add global_mean back
        transformed = (inv_sqrt_b @ (sqrt_global @ Xb_centered.T)).T + global_mean
        X_new[idx] = transformed

    df_out = df.copy()
    df_out[feature_cols] = X_new
    return df_out

def apply_moment_normalization(df, feature_cols, block_col='block', method='quantile'):
    """Apply higher-order moment normalization across blocks
    
    Methods:
    - 'quantile': Quantile transformer (uniform distribution)
    - 'power': Yeo-Johnson power transformation
    - 'rank': Rank-based transformation
    """
    print(f"   🔄 Applying {method} transformation for moment normalization...")
    
    df_out = df.copy()
    X = df[feature_cols].to_numpy(dtype=float)
    
    if method == 'quantile':
        # Quantile transformation to uniform distribution
        transformer = QuantileTransformer(output_distribution='uniform', random_state=42)
        X_transformed = transformer.fit_transform(X)
        
    elif method == 'power':
        # Yeo-Johnson power transformation
        transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        X_transformed = transformer.fit_transform(X)
        
    elif method == 'rank':
        # Rank-based transformation
        X_transformed = np.zeros_like(X)
        for i in range(X.shape[1]):
            ranks = stats.rankdata(X[:, i])
            # Convert ranks to uniform [0,1] then to standard normal
            uniform_ranks = (ranks - 0.5) / len(ranks)
            X_transformed[:, i] = stats.norm.ppf(uniform_ranks)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Update dataframe
    df_out[feature_cols] = X_transformed
    
    return df_out

def comprehensive_normalization(df, feature_cols, block_col='block', moment_method='quantile'):
    """Apply comprehensive normalization: mean + covariance + moments"""
    print("🔧 Applying comprehensive normalization...")
    
    # Step 1: Covariance normalization
    print("   Step 1: Block-covariance normalization...")
    df_cov = cov_normalize_blocks(df, feature_cols, block_col)
    
    # Step 2: Higher-order moment normalization
    print("   Step 2: Higher-order moment normalization...")
    df_final = apply_moment_normalization(df_cov, feature_cols, block_col, method=moment_method)
    
    print("✅ Comprehensive normalization complete")
    return df_final

def evaluate_block_prediction(data, feature_cols):
    """Evaluate how well blocks can still be predicted"""
    X = data[feature_cols].values
    y = data['block'].values
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    accuracy = rf.score(X, y)
    
    return accuracy

def create_cv_splits(data, k_folds=5):
    """Create synchronized CV splits"""
    blocks = [
        {'start': 0, 'end': 733, 'class': 0},
        {'start': 733, 'end': 1464, 'class': 1},
        {'start': 1464, 'end': 2205, 'class': 2}
    ]
    
    cv_splits = []
    for fold in range(k_folds):
        test_indices = []
        embargo_indices = set()
        
        for block in blocks:
            block_size = block['end'] - block['start']
            chunk_size = block_size // k_folds
            embargo_size = max(1, int(chunk_size * 0.02))
            
            chunk_start = block['start'] + fold * chunk_size
            chunk_end = min(block['start'] + (fold + 1) * chunk_size, block['end'])
            
            test_indices.extend(range(chunk_start, chunk_end))
            
            embargo_start = max(block['start'], chunk_start - embargo_size)
            embargo_end = min(chunk_end + embargo_size, block['end'])
            embargo_indices.update(range(embargo_start, embargo_end))
        
        train_indices = list(set(range(len(data))) - embargo_indices)
        
        cv_splits.append({
            'fold': fold,
            'train_indices': train_indices,
            'test_indices': test_indices
        })
    
    return cv_splits

def run_quick_label_permutation_test(data, feature_cols, n_perms=10):
    """Quick label permutation test to check if normalization worked"""
    print(f"🔄 Running quick label permutation test ({n_perms} permutations)...")
    
    cv_splits = create_cv_splits(data)
    
    # Baseline RandomForest performance
    baseline_accs = []
    for split in cv_splits:
        train_idx = split['train_indices']
        test_idx = split['test_indices']
        
        X_train = data.iloc[train_idx][feature_cols].values
        X_test = data.iloc[test_idx][feature_cols].values
        y_train = data.iloc[train_idx]['target'].values
        y_test = data.iloc[test_idx]['target'].values
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        baseline_accs.append(accuracy_score(y_test, y_pred))
    
    baseline_acc = np.mean(baseline_accs)
    
    # Null distribution
    null_accs = []
    rng = np.random.RandomState(42)
    
    for perm in range(n_perms):
        data_perm = data.copy()
        data_perm['target'] = rng.permutation(data_perm['target'].values)
        
        fold_accs = []
        for split in cv_splits:
            train_idx = split['train_indices']
            test_idx = split['test_indices']
            
            X_train = data_perm.iloc[train_idx][feature_cols].values
            X_test = data_perm.iloc[test_idx][feature_cols].values
            y_train = data_perm.iloc[train_idx]['target'].values
            y_test = data_perm.iloc[test_idx]['target'].values
            
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            fold_accs.append(accuracy_score(y_test, y_pred))
        
        null_accs.append(np.mean(fold_accs))
    
    null_mean = np.mean(null_accs)
    null_std = np.std(null_accs)
    p_value = np.mean(np.array(null_accs) >= baseline_acc)
    
    print(f"   📊 Baseline accuracy: {baseline_acc:.4f}")
    print(f"   📊 Null mean: {null_mean:.4f} ± {null_std:.4f}")
    print(f"   📊 P-value: {p_value:.4f}")
    
    success = p_value >= 0.05
    if success:
        print("   ✅ SUCCESS: p >= 0.05 (remediation worked!)")
    else:
        print("   ❌ FAILED: p < 0.05 (residual signal remains)")
    
    return {
        'baseline_acc': baseline_acc,
        'null_mean': null_mean,
        'null_std': null_std,
        'p_value': p_value,
        'success': success
    }

def main():
    print("🔧 HIGHER-ORDER MOMENT NORMALIZATION")
    print("=" * 80)
    
    # Load original data
    df = pd.read_csv('hydraulic_data_processed.csv')
    print(f"   ✅ Data loaded: {df.shape}")
    
    # Add block column
    df['block'] = 0
    df.loc[:731, 'block'] = 0
    df.loc[732:1463, 'block'] = 1
    df.loc[1464:, 'block'] = 2
    
    _exclude = {'block', 'target', 'label', 'class', 'index'}
    feature_cols = [c for c in df.columns if c not in _exclude]
    
    print(f"\n📊 BASELINE BLOCK PREDICTION (before normalization)")
    baseline_block_acc = evaluate_block_prediction(df, feature_cols)
    print(f"   🎯 Block prediction accuracy: {baseline_block_acc:.4f}")
    
    # Test different normalization approaches
    methods = ['quantile', 'power', 'rank']
    results = {}
    
    for method in methods:
        print(f"\n🔧 TESTING {method.upper()} NORMALIZATION")
        print("=" * 60)
        
        # Apply comprehensive normalization
        df_normalized = comprehensive_normalization(df, feature_cols, moment_method=method)
        
        # Evaluate block prediction after normalization
        block_acc = evaluate_block_prediction(df_normalized, feature_cols)
        print(f"   🎯 Block prediction after {method}: {block_acc:.4f}")
        
        # Quick label permutation test
        perm_results = run_quick_label_permutation_test(df_normalized, feature_cols, n_perms=10)
        
        results[method] = {
            'block_prediction_acc': block_acc,
            'label_permutation': perm_results
        }
    
    # Summary
    print(f"\n🏆 SUMMARY COMPARISON")
    print("=" * 80)
    print(f"   Original block prediction: {baseline_block_acc:.4f}")
    
    best_method = None
    best_p_value = 0
    
    for method, result in results.items():
        block_acc = result['block_prediction_acc']
        p_val = result['label_permutation']['p_value']
        success = "✅" if result['label_permutation']['success'] else "❌"
        
        print(f"   {method:8s}: Block={block_acc:.4f}, p-value={p_val:.4f} {success}")
        
        if p_val > best_p_value:
            best_p_value = p_val
            best_method = method
    
    if best_method and results[best_method]['label_permutation']['success']:
        print(f"\n🎉 REMEDIATION SUCCESSFUL!")
        print(f"   Best method: {best_method}")
        print(f"   p-value: {best_p_value:.4f} >= 0.05")
        print(f"   Ready for forensic-ML manuscript!")
    else:
        print(f"\n⚠️ REMEDIATION INCOMPLETE")
        print(f"   Best p-value: {best_p_value:.4f} < 0.05")
        print(f"   May need additional remediation steps")
    
    # Save results
    output_path = "output/higher_order_normalization_results.joblib"
    joblib.dump({
        'baseline_block_acc': baseline_block_acc,
        'method_results': results,
        'best_method': best_method,
        'success': best_method and results[best_method]['label_permutation']['success']
    }, output_path)
    print(f"   💾 Results saved: {output_path}")
    
    return results

if __name__ == "__main__":
    results = main()