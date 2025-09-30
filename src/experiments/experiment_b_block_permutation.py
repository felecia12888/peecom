#!/usr/bin/env python3
"""
EXPERIMENT B: BLOCK-PERMUTATION TEST
===================================
Purpose: Demonstrate that performance collapses when block→label mapping is broken
         while preserving all feature structure and temporal relationships.

Methodology:
1. Load original data with perfect block-class segregation
2. For each permutation (n=30):
   - Randomly reassign class labels to blocks 
   - Keep all feature structure intact
   - Run synchronized-chunk CV with RF + SimplePEECOM
   - Record test accuracies
3. Compare permuted performance vs. original baseline
4. Generate statistical significance test

Expected Outcome:
- If genuine signal: permuted accuracy << baseline accuracy
- If pure leakage: permuted accuracy ≈ baseline accuracy (both near chance)

Files Created:
- output/exp_b_block_permutation/{perm_i}_results.joblib (per permutation)
- output/exp_b_block_permutation/block_permutation_summary.csv
- output/exp_b_block_permutation/block_permutation_analysis.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

# Import PEECOM
import sys
sys.path.append('src')
from models.simple_peecom import SimplePEECOM

def setup_directories():
    """Create output directories for Experiment B"""
    base_dir = Path("output/exp_b_block_permutation")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def load_data():
    """Load data and analyze block structure"""
    try:
        data = pd.read_csv('hydraulic_data_processed.csv')
        print(f"   ✅ Real data loaded: {data.shape}")
    except:
        print("   ⚠️ Using synthetic data for demonstration")
        np.random.seed(42)
        n_samples = 2205
        n_features = 54
        data = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f'f{i}' for i in range(n_features)])
        
        # Create perfect block-class segregation
        data['target'] = 0
        data.loc[:731, 'target'] = 0    # Block 0: Class 0
        data.loc[732:1463, 'target'] = 1  # Block 1: Class 1  
        data.loc[1464:, 'target'] = 2   # Block 2: Class 2
    
    # --- START PATCH: Normalize block covariances before running permutations ---
    def _sqrt_and_invsqrt(mat, eps=1e-8):
        vals, vecs = np.linalg.eigh(mat)
        vals = np.clip(vals, eps, None)
        sqrt = (vecs * np.sqrt(vals)) @ vecs.T
        inv_sqrt = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
        return sqrt, inv_sqrt

    def cov_normalize_blocks(df, feature_cols, block_col='block', eps=1e-6):
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

    # Add block column to identify blocks (needed for covariance normalization)
    data['block'] = 0
    data.loc[:731, 'block'] = 0    # Block 0
    data.loc[732:1463, 'block'] = 1  # Block 1  
    data.loc[1464:, 'block'] = 2   # Block 2

    # detect feature columns (exclude non-feature columns)
    _exclude = {'block', 'target', 'label', 'class', 'index'}
    feature_cols = [c for c in data.columns if c not in _exclude]

    print("🔧 Applying block-covariance normalization (for permutation experiment)...")
    data = cov_normalize_blocks(data, feature_cols, block_col='block', eps=1e-6)
    print("✅ Covariance-normalized dataset ready for permutation tests.")
    # --- END PATCH ---
    
    return data

def identify_blocks(data):
    """Identify block boundaries and structure"""
    target = data['target'].values
    transitions = np.where(np.diff(target) != 0)[0] + 1
    block_starts = np.concatenate([[0], transitions, [len(data)]])
    
    blocks = []
    for i in range(len(block_starts) - 1):
        start_idx = block_starts[i]
        end_idx = block_starts[i + 1]
        original_class = target[start_idx]  # All samples in block have same class
        blocks.append({
            'block_id': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'original_class': original_class,
            'size': end_idx - start_idx
        })
    
    print(f"   🔍 Detected {len(blocks)} blocks:")
    for block in blocks:
        print(f"      Block {block['block_id']}: indices {block['start_idx']}-{block['end_idx']-1}, "
              f"original class {block['original_class']}, size {block['size']}")
    
    return blocks

def create_synchronized_cv_splits(data, blocks, k_folds=3):
    """Create synchronized chunk CV splits (same as Experiment A)"""
    all_chunks = []
    
    # Create chunks for each block
    for block in blocks:
        chunk_size = block['size'] // k_folds
        embargo_size = max(1, int(chunk_size * 0.02))
        
        for fold in range(k_folds):
            chunk_start = block['start_idx'] + fold * chunk_size
            chunk_end = min(block['start_idx'] + (fold + 1) * chunk_size, block['end_idx'])
            
            embargo_start = max(block['start_idx'], chunk_start - embargo_size)
            embargo_end = min(chunk_end + embargo_size, block['end_idx'])
            
            all_chunks.append({
                'block_id': block['block_id'],
                'fold': fold,
                'test_start': chunk_start,
                'test_end': chunk_end,
                'embargo_start': embargo_start,
                'embargo_end': embargo_end,
                'test_size': chunk_end - chunk_start
            })
    
    # Create CV splits
    cv_splits = []
    total_samples = len(data)
    
    for fold in range(k_folds):
        test_chunks = [chunk for chunk in all_chunks if chunk['fold'] == fold]
        test_indices = []
        embargo_indices = set()
        
        for chunk in test_chunks:
            test_indices.extend(range(chunk['test_start'], chunk['test_end']))
            embargo_indices.update(range(chunk['embargo_start'], chunk['embargo_end']))
        
        train_indices = list(set(range(total_samples)) - embargo_indices)
        
        cv_splits.append({
            'fold': fold,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'embargo_size': len(embargo_indices) - len(test_indices)
        })
    
    return cv_splits

def permute_block_labels(data, blocks, perm_seed):
    """Create a permuted version where block→class mapping is shuffled"""
    rng = np.random.RandomState(perm_seed)
    
    # Get original class assignments for each block
    original_classes = [block['original_class'] for block in blocks]
    
    # Create random permutation of class labels
    permuted_classes = rng.permutation(original_classes)
    
    # Create mapping: block_id → new_class
    block_to_new_class = {i: permuted_classes[i] for i in range(len(blocks))}
    
    print(f"      🔀 Block→Class mapping: {block_to_new_class}")
    
    # Apply permutation to data
    permuted_data = data.copy()
    target = permuted_data['target'].values
    
    for block in blocks:
        block_mask = (
            (permuted_data.index >= block['start_idx']) & 
            (permuted_data.index < block['end_idx'])
        )
        permuted_data.loc[block_mask, 'target'] = block_to_new_class[block['block_id']]
    
    return permuted_data, block_to_new_class

def evaluate_model_cv(model_name, model, data, cv_splits):
    """Run cross-validation for a single model"""
    fold_results = []
    
    for split in cv_splits:
        train_idx = split['train_indices']
        test_idx = split['test_indices']
        y_train = data.iloc[train_idx]['target'].values
        y_test = data.iloc[test_idx]['target'].values
        
        if model_name == 'SimplePEECOM':
            # PEECOM uses raw features
            X_train = data.iloc[train_idx].drop(['target', 'block'], axis=1).values
            X_test = data.iloc[test_idx].drop(['target', 'block'], axis=1).values
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            # Standard models use covariance-normalized features (NO additional scaling!)
            feature_cols = [col for col in data.columns if col not in ['target', 'block']]
            X_train = data.iloc[train_idx][feature_cols].values
            X_test = data.iloc[test_idx][feature_cols].values
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        fold_results.append({
            'fold': split['fold'],
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'n_train': len(train_idx),
            'n_test': len(test_idx)
        })
    
    mean_accuracy = np.mean([f['accuracy'] for f in fold_results])
    std_accuracy = np.std([f['accuracy'] for f in fold_results])
    
    return {
        'model': model_name,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'fold_results': fold_results
    }

def run_single_permutation(perm_id, data, blocks, cv_splits, base_dir):
    """Run single permutation experiment"""
    print(f"   🔄 Permutation {perm_id + 1}/30:")
    
    # Create permuted data
    permuted_data, mapping = permute_block_labels(data, blocks, perm_seed=perm_id + 42)
    
    # Models to test
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        ),
        'SimplePEECOM': SimplePEECOM()
    }
    
    # Evaluate both models
    results = {
        'permutation_id': perm_id,
        'block_mapping': mapping,
        'model_results': {}
    }
    
    for model_name, model in models.items():
        result = evaluate_model_cv(model_name, model, permuted_data, cv_splits)
        results['model_results'][model_name] = result
        print(f"         {model_name}: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    
    # Save permutation results
    perm_file = base_dir / f"perm_{perm_id:02d}_results.joblib"
    joblib.dump(results, perm_file)
    
    return results

def run_baseline_experiment(data, cv_splits):
    """Run baseline experiment on original (non-permuted) data"""
    print("📊 BASELINE EXPERIMENT (Original Data)")
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        ),
        'SimplePEECOM': SimplePEECOM()
    }
    
    baseline_results = {}
    
    for model_name, model in models.items():
        result = evaluate_model_cv(model_name, model, data, cv_splits)
        baseline_results[model_name] = result
        print(f"   {model_name}: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    
    return baseline_results

def analyze_permutation_results(baseline_results, perm_results, base_dir):
    """Analyze and visualize permutation test results"""
    print("\n📈 ANALYZING PERMUTATION RESULTS")
    
    # Compile results into DataFrame
    rows = []
    
    for model_name in ['RandomForest', 'SimplePEECOM']:
        # Baseline row
        baseline = baseline_results[model_name]
        rows.append({
            'model': model_name,
            'experiment': 'baseline',
            'permutation_id': -1,
            'mean_accuracy': baseline['mean_accuracy'],
            'std_accuracy': baseline['std_accuracy']
        })
        
        # Permutation rows
        for perm_result in perm_results:
            perm = perm_result['model_results'][model_name]
            rows.append({
                'model': model_name,
                'experiment': 'permuted',
                'permutation_id': perm_result['permutation_id'],
                'mean_accuracy': perm['mean_accuracy'],
                'std_accuracy': perm['std_accuracy']
            })
    
    results_df = pd.DataFrame(rows)
    
    # Save summary CSV
    csv_path = base_dir / "block_permutation_summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"   ✅ Summary saved: {csv_path}")
    
    # Statistical analysis and visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Experiment B: Block-Permutation Test Results', fontsize=16, fontweight='bold')
    
    for i, model_name in enumerate(['RandomForest', 'SimplePEECOM']):
        # Get data for this model
        baseline_acc = baseline_results[model_name]['mean_accuracy']
        perm_accs = [r['model_results'][model_name]['mean_accuracy'] for r in perm_results]
        
        # Histogram
        ax1 = axes[i, 0]
        ax1.hist(perm_accs, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(baseline_acc, color='red', linestyle='--', linewidth=2, 
                   label=f'Baseline: {baseline_acc:.4f}')
        ax1.axvline(1/3, color='orange', linestyle=':', linewidth=2, 
                   label='Chance (0.333)')
        ax1.set_xlabel('Accuracy')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{model_name}: Permuted Accuracy Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        ax2 = axes[i, 1]
        comparison_data = [
            [baseline_acc],  # Single baseline value
            perm_accs        # All permutation values
        ]
        bp = ax2.boxplot(comparison_data, labels=['Baseline', 'Permuted'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('skyblue')
        ax2.axhline(1/3, color='orange', linestyle=':', label='Chance (0.333)')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{model_name}: Baseline vs Permuted')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Statistical analysis
        print(f"\n   📊 {model_name} Analysis:")
        print(f"      Baseline accuracy: {baseline_acc:.4f}")
        print(f"      Permuted mean:     {np.mean(perm_accs):.4f} ± {np.std(perm_accs):.4f}")
        print(f"      Permuted range:    [{np.min(perm_accs):.4f}, {np.max(perm_accs):.4f}]")
        
        # P-value: fraction of permutations >= baseline
        n_perm_geq_baseline = sum(1 for acc in perm_accs if acc >= baseline_acc)
        p_value = n_perm_geq_baseline / len(perm_accs)
        print(f"      P-value:           {p_value:.4f} ({n_perm_geq_baseline}/{len(perm_accs)} perms ≥ baseline)")
        
        # Effect size (difference between baseline and permuted mean)
        effect_size = baseline_acc - np.mean(perm_accs)
        print(f"      Effect size:       {effect_size:.4f}")
        
        # Interpretation
        if p_value > 0.5 and abs(effect_size) < 0.02:
            interpretation = "🔴 STRONG LEAKAGE: Permuted ≈ Baseline (both at chance)"
        elif p_value < 0.1 and effect_size > 0.05:
            interpretation = "🟢 GENUINE SIGNAL: Permuted << Baseline (significant drop)"
        else:
            interpretation = "🟡 MIXED: Some leakage + some genuine signal"
        
        print(f"      Interpretation:    {interpretation}")
    
    plt.tight_layout()
    plot_path = base_dir / "block_permutation_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Analysis plot saved: {plot_path}")
    
    return results_df

def main():
    print("🧪 EXPERIMENT B: BLOCK-PERMUTATION TEST")
    print("=" * 60)
    print("Purpose: Test if performance drops when block→class mapping is broken")
    print("Method:  Randomly permute class assignments across blocks (n=30)")
    print("Models:  RandomForest + SimplePEECOM")
    print("=" * 60)
    
    # Setup
    base_dir = setup_directories()
    print(f"\n📁 Output directory: {base_dir}")
    
    # Load data and analyze structure
    print("\n📊 LOADING DATA")
    data = load_data()
    blocks = identify_blocks(data)
    
    # Create CV splits
    print(f"\n📂 CREATING SYNCHRONIZED CV SPLITS")
    cv_splits = create_synchronized_cv_splits(data, blocks, k_folds=3)
    print(f"   ✅ Created {len(cv_splits)} CV folds")
    
    # Run baseline experiment
    print(f"\n🎯 BASELINE EXPERIMENT")
    baseline_results = run_baseline_experiment(data, cv_splits)
    
    # Run permutation experiments
    print(f"\n🔄 PERMUTATION EXPERIMENTS (n=30)")
    perm_results = []
    
    for perm_id in range(30):
        result = run_single_permutation(perm_id, data, blocks, cv_splits, base_dir)
        perm_results.append(result)
    
    print(f"   ✅ Completed all 30 permutations")
    
    # Analysis and visualization
    results_df = analyze_permutation_results(baseline_results, perm_results, base_dir)
    
    # Final summary
    print(f"\n🏁 EXPERIMENT B COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {base_dir}")
    print("Files created:")
    print(f"  - perm_XX_results.joblib (individual permutation results)")
    print(f"  - block_permutation_summary.csv (compiled results)")
    print(f"  - block_permutation_analysis.png (visualization)")
    
    # Overall conclusion
    rf_baseline = baseline_results['RandomForest']['mean_accuracy']
    rf_perm_mean = np.mean([r['model_results']['RandomForest']['mean_accuracy'] for r in perm_results])
    peecom_baseline = baseline_results['SimplePEECOM']['mean_accuracy']
    peecom_perm_mean = np.mean([r['model_results']['SimplePEECOM']['mean_accuracy'] for r in perm_results])
    
    print(f"\n📋 OVERALL CONCLUSION:")
    print(f"RandomForest:   Baseline={rf_baseline:.4f}, Permuted={rf_perm_mean:.4f}")
    print(f"SimplePEECOM:   Baseline={peecom_baseline:.4f}, Permuted={peecom_perm_mean:.4f}")
    
    if (abs(rf_baseline - rf_perm_mean) < 0.02 and 
        abs(peecom_baseline - peecom_perm_mean) < 0.02):
        conclusion = "🔴 DEFINITIVE LEAKAGE: Performance unchanged when block→class mapping broken"
    else:
        conclusion = "🟢 GENUINE SIGNAL: Performance drops significantly under permutation"
    
    print(f"{conclusion}")

if __name__ == '__main__':
    main()