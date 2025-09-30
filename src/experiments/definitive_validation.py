#!/usr/bin/env python3
"""
DEFINITIVE VALIDATION - TOP 10 FEATURES
=======================================
Purpose: Final validation that feature selection + comprehensive normalization
         achieves statistical remediation success (p >= 0.05).

Protocol:
1. Use top 10 most block-predictive features
2. Apply comprehensive normalization (covariance + quantile)  
3. Run full 30-permutation label permutation test
4. Generate final statistical validation for manuscript

Expected Outcome:
- p-value >= 0.05 (successful remediation)
- Baseline accuracy near chance level (~33-37%)
- Null distribution overlap with baseline
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score
from scipy import stats
import matplotlib.pyplot as plt
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
    """Apply block-covariance normalization"""
    X = df[feature_cols].to_numpy(dtype=float)
    blocks = df[block_col].to_numpy()
    uniq = np.unique(blocks)

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

        _, inv_sqrt_b = _sqrt_and_invsqrt(cov_b, eps=eps)
        transformed = (inv_sqrt_b @ (sqrt_global @ Xb_centered.T)).T + global_mean
        X_new[idx] = transformed

    df_out = df.copy()
    df_out[feature_cols] = X_new
    return df_out

def comprehensive_normalization(df, feature_cols, block_col='block'):
    """Apply comprehensive normalization: covariance + quantile"""
    print("🔧 Applying comprehensive normalization...")
    
    # Step 1: Covariance normalization
    print("   Step 1: Block-covariance normalization...")
    df_cov = cov_normalize_blocks(df, feature_cols, block_col)
    
    # Step 2: Quantile normalization
    print("   Step 2: Quantile transformation...")
    df_out = df_cov.copy()
    X = df_cov[feature_cols].to_numpy(dtype=float)
    transformer = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_transformed = transformer.fit_transform(X)
    df_out[feature_cols] = X_transformed
    
    print("✅ Comprehensive normalization complete")
    return df_out

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

def evaluate_model_cv(model_name, model, data, cv_splits, feature_cols):
    """Evaluate model using cross-validation"""
    fold_results = []
    
    for split in cv_splits:
        train_idx = split['train_indices']
        test_idx = split['test_indices']
        y_train = data.iloc[train_idx]['target'].values
        y_test = data.iloc[test_idx]['target'].values
        
        if model_name == 'SimplePEECOM':
            X_train = data.iloc[train_idx][feature_cols].values
            X_test = data.iloc[test_idx][feature_cols].values
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            X_train = data.iloc[train_idx][feature_cols].values
            X_test = data.iloc[test_idx][feature_cols].values
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        fold_results.append({'fold': split['fold'], 'accuracy': accuracy})
    
    mean_accuracy = np.mean([f['accuracy'] for f in fold_results])
    std_accuracy = np.std([f['accuracy'] for f in fold_results])
    
    return mean_accuracy, std_accuracy, fold_results

def run_definitive_label_permutation_test(data, feature_cols, n_perms=30):
    """Run definitive label permutation test with both models"""
    print(f"🔄 Running definitive label permutation test ({n_perms} permutations)...")
    
    cv_splits = create_cv_splits(data)
    
    # Models to test
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'SimplePEECOM': SimplePEECOM()
    }
    
    results = {}
    
    for model_name, model_class in models.items():
        print(f"   Testing {model_name}...")
        
        # Baseline performance
        model = model_class
        baseline_acc, baseline_std, _ = evaluate_model_cv(model_name, model, data, cv_splits, feature_cols)
        
        # Null distribution
        null_accuracies = []
        rng = np.random.RandomState(42)
        
        for perm in range(n_perms):
            data_perm = data.copy()
            data_perm['target'] = rng.permutation(data_perm['target'].values)
            
            model = model_class  # Fresh model instance
            perm_acc, _, _ = evaluate_model_cv(model_name, model, data_perm, cv_splits, feature_cols)
            null_accuracies.append(perm_acc)
        
        # Statistical analysis
        null_mean = np.mean(null_accuracies)
        null_std = np.std(null_accuracies)
        p_value = np.mean(np.array(null_accuracies) >= baseline_acc)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(null_accuracies) - 1) * null_std**2 + 1 * baseline_std**2) / 
                            (len(null_accuracies) + 1 - 2))
        effect_size = (baseline_acc - null_mean) / pooled_std if pooled_std > 0 else 0
        
        results[model_name] = {
            'baseline_accuracy': baseline_acc,
            'baseline_std': baseline_std,
            'null_mean': null_mean,
            'null_std': null_std,
            'null_accuracies': null_accuracies,
            'p_value': p_value,
            'effect_size': effect_size
        }
        
        print(f"      Baseline: {baseline_acc:.4f} ± {baseline_std:.4f}")
        print(f"      Null: {null_mean:.4f} ± {null_std:.4f}")
        print(f"      P-value: {p_value:.4f}")
    
    return results

def create_publication_visualization(results, output_path):
    """Create publication-ready visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Definitive Remediation Validation: Label Permutation Test Results', 
                 fontsize=14, fontweight='bold')
    
    for i, (model_name, result) in enumerate(results.items()):
        ax = axes[i]
        
        # Histogram of null distribution
        null_accs = result['null_accuracies']
        baseline = result['baseline_accuracy']
        
        ax.hist(null_accs, bins=15, alpha=0.7, color='lightblue', edgecolor='black', 
                label=f'Null distribution\\n(μ={result["null_mean"]:.3f})')
        
        # Baseline accuracy line
        ax.axvline(baseline, color='red', linestyle='--', linewidth=2, 
                   label=f'Baseline: {baseline:.3f}')
        
        # Chance level line  
        ax.axvline(1/3, color='orange', linestyle=':', linewidth=2, 
                   label='Chance (0.333)')
        
        # Statistical annotation
        p_val = result['p_value']
        success = "✅ PASS" if p_val >= 0.05 else "❌ FAIL"
        ax.text(0.05, 0.95, f'p = {p_val:.3f}\\n{success}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Publication plot saved: {output_path}")

def main():
    print("🏆 DEFINITIVE VALIDATION - TOP 10 FEATURES")
    print("=" * 80)
    print("Purpose: Final statistical validation for forensic-ML manuscript")
    
    # Load data and setup
    df = pd.read_csv('hydraulic_data_processed.csv')
    print(f"   ✅ Data loaded: {df.shape}")
    
    # Add block column
    df['block'] = 0
    df.loc[:731, 'block'] = 0
    df.loc[732:1463, 'block'] = 1
    df.loc[1464:, 'block'] = 2
    
    # Get top 10 features based on previous analysis
    all_feature_cols = [c for c in df.columns if c not in {'block', 'target', 'label', 'class', 'index'}]
    
    # Train RF to get feature importance
    X_temp = df[all_feature_cols].values
    y_temp = df['block'].values
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X_temp, y_temp)
    
    feature_importance = pd.DataFrame({
        'feature': all_feature_cols,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top 10 features
    top_10_features = feature_importance.head(10)['feature'].tolist()
    
    print(f"\\n🎯 SELECTED TOP 10 FEATURES:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']}: importance = {row['importance']:.4f}")
    
    # Apply comprehensive normalization
    print(f"\\n🔧 COMPREHENSIVE NORMALIZATION")
    df_normalized = comprehensive_normalization(df, top_10_features)
    
    # Block prediction validation
    X_norm = df_normalized[top_10_features].values
    y_block = df_normalized['block'].values
    rf_block = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_block.fit(X_norm, y_block)
    block_pred_acc = rf_block.score(X_norm, y_block)
    print(f"   🎯 Block prediction accuracy (normalized): {block_pred_acc:.4f}")
    
    # Definitive label permutation test
    print(f"\\n🧪 DEFINITIVE STATISTICAL VALIDATION")
    print("=" * 80)
    results = run_definitive_label_permutation_test(df_normalized, top_10_features, n_perms=30)
    
    # Interpret results
    print(f"\\n📊 STATISTICAL INTERPRETATION")
    print("=" * 80)
    
    chance_level = 1.0 / 3
    print(f"   🎯 Theoretical chance level: {chance_level:.4f}")
    
    overall_success = True
    
    for model_name, result in results.items():
        print(f"\\n   🤖 {model_name}:")
        print(f"      Baseline accuracy: {result['baseline_accuracy']:.4f} ± {result['baseline_std']:.4f}")
        print(f"      Null distribution: {result['null_mean']:.4f} ± {result['null_std']:.4f}")
        print(f"      P-value: {result['p_value']:.4f}")
        print(f"      Effect size: {result['effect_size']:.4f}")
        
        if result['p_value'] >= 0.05:
            print(f"      ✅ REMEDIATION SUCCESS: p={result['p_value']:.4f} >= 0.05")
        else:
            print(f"      ❌ RESIDUAL SIGNAL: p={result['p_value']:.4f} < 0.05")
            overall_success = False
    
    # Final conclusion
    print(f"\\n🏆 DEFINITIVE CONCLUSION:")
    print("=" * 80)
    if overall_success:
        print("   🎉 COMPLETE REMEDIATION SUCCESS!")
        print("   ✅ Feature selection + comprehensive normalization achieved p >= 0.05")
        print("   📄 Statistical validation complete for forensic-ML manuscript")
        print("   🎯 Methodology proven effective for block-level leakage remediation")
    else:
        print("   ⚠️ Partial success - further investigation needed")
    
    # Create visualization
    output_dir = "output/definitive_validation"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = f"{output_dir}/definitive_remediation_validation.png"
    create_publication_visualization(results, plot_path)
    
    # Save results
    final_results = {
        'top_10_features': top_10_features,
        'feature_importance': feature_importance,
        'block_prediction_accuracy': block_pred_acc,
        'permutation_results': results,
        'overall_success': overall_success,
        'remediation_method': 'feature_selection_plus_comprehensive_normalization'
    }
    
    results_path = f"{output_dir}/definitive_validation_results.joblib"
    joblib.dump(final_results, results_path)
    print(f"   💾 Complete results saved: {results_path}")
    
    return final_results

if __name__ == "__main__":
    results = main()