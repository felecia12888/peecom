#!/usr/bin/env python3
"""
QUICK LEAKAGE VALIDATION
========================
Streamlined validation to confirm data leakage findings without extensive computation.
Runs only the most critical diagnostics:
1. Block-label permutation (5 permutations)
2. Label permutation null test (5 permutations)
3. Summary comparison with baseline

Goal: Confirm that PEECOM performance drops to chance under proper controls.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle as sk_shuffle

import sys
sys.path.append('src')
from models.simple_peecom import SimplePEECOM

def load_data():
    """Load data (same as full diagnostics)"""
    if Path('hydraulic_data_processed.csv').exists():
        data = pd.read_csv('hydraulic_data_processed.csv')
    else:
        # Synthetic placeholder
        n_samples = 2205
        n_features = 54
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(n_samples, n_features), columns=[f'f{i}' for i in range(n_features)])
        y = np.zeros(n_samples, dtype=int)
        y[732:1464] = 1
        y[1464:] = 2
        data = X.copy()
        data['target'] = y
    return data

def create_quick_cv_splits():
    """Create 3-fold synchronized CV splits for speed"""
    data = load_data()
    target = data['target'].values
    transitions = np.where(np.diff(target) != 0)[0] + 1
    block_starts = np.concatenate([[0], transitions, [len(data)]])
    blocks = []
    for i in range(len(block_starts)-1):
        blocks.append({'block_id': i,'start_idx': block_starts[i],'end_idx': block_starts[i+1],'size': block_starts[i+1]-block_starts[i]})
    
    k_folds = 3  # Reduced for speed
    all_chunks = []
    for b in blocks:
        size = b['size']
        chunk_size = size // k_folds
        embargo = max(1, int(chunk_size*0.02))
        for f in range(k_folds):
            start = b['start_idx'] + f*chunk_size
            end = min(b['start_idx'] + (f+1)*chunk_size, b['end_idx'])
            all_chunks.append({'block_id': b['block_id'], 'fold': f, 'test_start': start,'test_end': end,'embargo_start': max(b['start_idx'], start-embargo),'embargo_end': min(end+embargo, b['end_idx'])})
    
    cv_splits = []
    total = len(data)
    for f in range(k_folds):
        test_idx = []
        embargo = set()
        for c in all_chunks:
            if c['fold']==f:
                test_idx.extend(range(c['test_start'], c['test_end']))
                embargo.update(range(c['embargo_start'], c['embargo_end']))
        train_idx = list(set(range(total)) - embargo)
        cv_splits.append({'fold': f,'train_indices': train_idx,'test_indices': test_idx,'embargo_size': len(embargo)-len(test_idx)})
    return cv_splits

def evaluate_simple_peecom_cv(data, cv_splits):
    """Run SimplePEECOM on synchronized CV"""
    fold_results = []
    target = data['target'].values
    
    for split in cv_splits:
        train_idx = split['train_indices']
        test_idx = split['test_indices']
        y_train = target[train_idx]
        y_test = target[test_idx]
        
        X_train_raw = data.iloc[train_idx].drop('target', axis=1).values
        X_test_raw = data.iloc[test_idx].drop('target', axis=1).values
        
        model = SimplePEECOM()
        model.fit(X_train_raw, y_train)
        y_pred = model.predict(X_test_raw)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        fold_results.append({
            'fold': split['fold'],
            'accuracy': acc,
            'macro_f1': f1
        })
    return fold_results

def run_quick_block_permutation(data, cv_splits, n_perms=5):
    """Block-label permutation test (reduced permutations for speed)"""
    print("   🔄 Block permutation test...")
    target = data['target'].values
    transitions = np.where(np.diff(target)!=0)[0] + 1
    block_ids = np.zeros(len(target), dtype=int)
    for t in transitions:
        block_ids[t:] += 1
    unique_blocks = np.unique(block_ids)
    original_labels = {b: target[block_ids==b][0] for b in unique_blocks}
    
    rng = np.random.RandomState(42)
    perm_accuracies = []
    
    for p in range(n_perms):
        mapping_labels = rng.permutation(list(original_labels.values()))
        mapping = {b: mapping_labels[i] for i,b in enumerate(unique_blocks)}
        permuted_target = np.array([mapping[b] for b in block_ids])
        
        data_perm = data.copy()
        data_perm['target'] = permuted_target
        fold_results = evaluate_simple_peecom_cv(data_perm, cv_splits)
        mean_acc = float(np.mean([fr['accuracy'] for fr in fold_results]))
        perm_accuracies.append(mean_acc)
    
    return perm_accuracies

def run_quick_label_permutation(data, cv_splits, n_perms=5):
    """Label permutation null test (reduced permutations for speed)"""
    print("   🔄 Label permutation test...")
    rng = np.random.RandomState(42)
    perm_accs = []
    
    for p in range(n_perms):
        perm_data = data.copy()
        perm_data['target'] = rng.permutation(perm_data['target'].values)
        fold_results = evaluate_simple_peecom_cv(perm_data, cv_splits)
        perm_accs.append(float(np.mean([fr['accuracy'] for fr in fold_results])))
    
    return perm_accs

def main():
    print("🚀 QUICK LEAKAGE VALIDATION (SimplePEECOM)")
    print("=" * 50)
    
    # Setup
    data = load_data()
    cv_splits = create_quick_cv_splits()
    output_dir = Path('output/quick_leakage_validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   📊 Data: {data.shape[0]} samples, {data.shape[1]-1} features")
    print(f"   📂 CV: {len(cv_splits)} folds (synchronized chunks)")
    
    # Baseline performance
    print("\n1️⃣ BASELINE PERFORMANCE")
    baseline_folds = evaluate_simple_peecom_cv(data, cv_splits)
    baseline_acc = np.mean([f['accuracy'] for f in baseline_folds])
    baseline_f1 = np.mean([f['macro_f1'] for f in baseline_folds])
    print(f"   ✅ SimplePEECOM baseline: Acc={baseline_acc:.4f}, F1={baseline_f1:.4f}")
    
    # Block permutation test
    print("\n2️⃣ BLOCK PERMUTATION TEST")
    block_perm_accs = run_quick_block_permutation(data, cv_splits)
    block_perm_mean = np.mean(block_perm_accs)
    block_perm_std = np.std(block_perm_accs)
    print(f"   ✅ Block permutation: Acc={block_perm_mean:.4f} ± {block_perm_std:.4f}")
    
    # Label permutation test
    print("\n3️⃣ LABEL PERMUTATION TEST")
    label_perm_accs = run_quick_label_permutation(data, cv_splits)
    label_perm_mean = np.mean(label_perm_accs)
    label_perm_std = np.std(label_perm_accs)
    print(f"   ✅ Label permutation: Acc={label_perm_mean:.4f} ± {label_perm_std:.4f}")
    
    # Summary and interpretation
    print("\n📋 SUMMARY & INTERPRETATION")
    print("=" * 50)
    chance_level = 1.0 / len(np.unique(data['target']))
    
    print(f"   🎯 Theoretical chance level: {chance_level:.4f}")
    print(f"   📈 Baseline accuracy:        {baseline_acc:.4f}")
    print(f"   🔀 Block permutation:        {block_perm_mean:.4f} ± {block_perm_std:.4f}")
    print(f"   🎲 Label permutation:        {label_perm_mean:.4f} ± {label_perm_std:.4f}")
    
    # Decision logic
    if baseline_acc > (chance_level + 0.05):  # 5% margin above chance
        if abs(block_perm_mean - baseline_acc) < 0.03:  # Block permutation doesn't drop much
            conclusion = "❌ BLOCK-LEVEL ENCODING DETECTED (Strong leakage signal)"
        else:
            conclusion = "⚠️ MIXED SIGNALS (Some genuine + some leaked signal)"
    else:
        conclusion = "✅ NO SIGNIFICANT LEAKAGE (Performance at chance level)"
    
    print(f"\n🏁 CONCLUSION: {conclusion}")
    
    # Save results
    summary = {
        'baseline': {'accuracy': baseline_acc, 'macro_f1': baseline_f1, 'folds': baseline_folds},
        'block_permutation': {'accuracies': block_perm_accs, 'mean': block_perm_mean, 'std': block_perm_std},
        'label_permutation': {'accuracies': label_perm_accs, 'mean': label_perm_mean, 'std': label_perm_std},
        'chance_level': chance_level,
        'conclusion': conclusion
    }
    
    with open(output_dir / 'quick_validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir / 'quick_validation_summary.json'}")

if __name__ == '__main__':
    main()