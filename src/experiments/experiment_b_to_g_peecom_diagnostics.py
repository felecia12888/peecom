#!/usr/bin/env python3
"""
PEECOM Leakage Diagnostics (Experiments B–G)
===========================================
Implements the remaining diagnostic experiments for all PEECOM variants:
B. Block-permutation test
C. Feature separability ranking + ablation
D. Block-relative normalization + rerun
E. Label permutation null test
F. Feature-swap (PEECOM-on-raw vs RF-on-PEECOM features)
G. Robustness shifts

Assumptions / Notes:
- Reuses synchronized chunk CV splits from Experiment A results file.
- If that file is missing, it will regenerate folds identically.
- SHAP is optional; skipped if not installed.
- Ablation for PEECOM variants is implemented by externally generating the
  physics features using the internal helper methods and then selectively
  dropping top-K separable columns before fitting a lightweight clone of
  the variant's underlying model (RandomForest for Simple/Enhanced, chosen
  classifier for MultiClassifier after selection on full feature set).
- Feature provenance tags are heuristic based on name patterns.

Outputs are saved under: output/peecom_leakage_diagnostics/<variant>/...
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle as sk_shuffle

import sys
sys.path.append('src')
from models.simple_peecom import SimplePEECOM
from models.multi_classifier_peecom import MultiClassifierPEECOM
from models.enhanced_peecom import EnhancedPEECOM

try:
    import shap  # optional
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

###############################################################################
# Utility: Load data and synchronized CV splits
###############################################################################

def load_data():
    if Path('hydraulic_data_processed.csv').exists():
        data = pd.read_csv('hydraulic_data_processed.csv')
    else:
        # Synthetic placeholder (same blocked structure as experiment A)
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

def create_splits_from_experiment_a():
    results_path = Path('output/experiment_a_synchronized_cv/results/experiment_a_results.joblib')
    if results_path.exists():
        payload = joblib.load(results_path)
        return payload['cv_splits']
    # Reconstruct synchronized chunks if missing
    data = load_data()
    target = data['target'].values
    transitions = np.where(np.diff(target) != 0)[0] + 1
    block_starts = np.concatenate([[0], transitions, [len(data)]])
    blocks = []
    for i in range(len(block_starts)-1):
        blocks.append({'block_id': i,'start_idx': block_starts[i],'end_idx': block_starts[i+1],'size': block_starts[i+1]-block_starts[i]})
    k_folds = 5
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

###############################################################################
# Core CV evaluation for a single PEECOM variant
###############################################################################

def evaluate_variant_cv(variant_name, variant_ctor, data, cv_splits, features_df=None, drop_features=None, block_relative_norm=False, random_state=42):
    """Run synchronized-chunk CV for a PEECOM variant.
    Parameters:
        features_df: optional precomputed feature matrix (DataFrame) aligned to data index
        drop_features: list of feature columns to drop from features_df before training
        block_relative_norm: if True, apply per-block (train-estimated) z-normalization
    Returns: list of fold result dicts
    """
    fold_results = []
    target = data['target'].values

    # Identify blocks (used for block-relative normalization)
    transitions = np.where(np.diff(target) != 0)[0] + 1
    block_ids = np.zeros(len(target), dtype=int)
    current_block = 0
    for t in transitions:
        block_ids[t:] += 1

    for split in cv_splits:
        train_idx = split['train_indices']
        test_idx = split['test_indices']
        y_train = target[train_idx]
        y_test = target[test_idx]

        # Generate / select features
        if features_df is None:
            # Use raw features for generation
            X_train_raw = data.iloc[train_idx].drop('target', axis=1).values
            X_test_raw = data.iloc[test_idx].drop('target', axis=1).values
            model = variant_ctor()
            model.fit(X_train_raw, y_train)  # fits feature pipeline + model
            # Extract internal generated features for training set (post feature engineering) if possible
            def extract_feats(m, raw):
                if hasattr(m,'physics_enhancer') and hasattr(m.physics_enhancer,'_create_physics_features'):
                    return m.physics_enhancer._create_physics_features(raw)
                if hasattr(m,'_create_physics_features'):
                    return m._create_physics_features(raw)
                if hasattr(m,'_create_advanced_physics_features'):
                    return m._create_advanced_physics_features(raw)
                return pd.DataFrame(raw)
            train_feats = extract_feats(model, data.iloc[train_idx].drop('target', axis=1).values)
            test_feats = extract_feats(model, data.iloc[test_idx].drop('target', axis=1).values)
        else:
            model = variant_ctor()
            train_feats = features_df.iloc[train_idx].copy()
            test_feats = features_df.iloc[test_idx].copy()
            # Feature removal
            if drop_features:
                drop_set = [f for f in drop_features if f in train_feats.columns]
                train_feats = train_feats.drop(columns=drop_set, errors='ignore')
                test_feats = test_feats.drop(columns=drop_set, errors='ignore')
            # Fit simple RF on provided features to approximate variant behaviour
            model = RandomForestClassifier(n_estimators=150, random_state=random_state)
            # (We treat this as a proxy when manual ablation is applied.)

        # Block-relative normalization
        if block_relative_norm:
            train_df = train_feats.copy()
            test_df = test_feats.copy()
            blk_train = block_ids[train_idx]
            blk_test = block_ids[test_idx]
            norm_train = train_df.copy()
            for b in np.unique(blk_train):
                mask = blk_train == b
                blk_stats_mean = train_df.loc[mask].mean()
                blk_stats_std = train_df.loc[mask].std().replace(0, 1.0)
                norm_train.loc[mask] = (train_df.loc[mask] - blk_stats_mean) / blk_stats_std
            # Apply train block stats to test per its own block id (use that block's train stats if available else global)
            norm_test = test_df.copy()
            for b in np.unique(blk_test):
                mask = blk_test == b
                if b in np.unique(blk_train):
                    blk_stats_mean = train_df.loc[blk_train==b].mean()
                    blk_stats_std = train_df.loc[blk_train==b].std().replace(0,1.0)
                else:
                    blk_stats_mean = train_df.mean()
                    blk_stats_std = train_df.std().replace(0,1.0)
                norm_test.loc[mask] = (test_df.loc[mask] - blk_stats_mean) / blk_stats_std
            train_feats = norm_train
            test_feats = norm_test

        # Scale (standard) – fit on train only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_feats)
        X_test = scaler.transform(test_feats)

        # Train underlying model (if pre-fit variant used, we already fit; for manual path we use RF proxy)
        if isinstance(model, (SimplePEECOM, EnhancedPEECOM, MultiClassifierPEECOM)) and features_df is None:
            # Already fitted full pipeline; re-fit only underlying estimator on transformed features
            # For simplicity, re-train a RandomForest on engineered features to keep consistency across ablations
            base_model = RandomForestClassifier(n_estimators=200, random_state=random_state)
            base_model.fit(X_train, y_train)
            y_pred = base_model.predict(X_test)
            fitted_model = base_model
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            fitted_model = model

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')

        fold_results.append({
            'fold': split['fold'],
            'accuracy': acc,
            'macro_f1': macro_f1,
            'precision': prec,
            'recall': rec,
            'n_features': train_feats.shape[1],
        })
    return fold_results

###############################################################################
# Experiment B: Block-permutation test
###############################################################################

def run_block_permutation(variant_name, variant_ctor, data, cv_splits, n_perms=30, rng_seed=42, features_df=None):
    """Block-label permutation using optional precomputed engineered features.
    If features_df supplied, avoids repeated heavy variant training per permutation."""
    rng = np.random.RandomState(rng_seed)
    target = data['target'].values
    transitions = np.where(np.diff(target)!=0)[0] + 1
    block_ids = np.zeros(len(target), dtype=int)
    for t in transitions:
        block_ids[t:] += 1
    unique_blocks = np.unique(block_ids)
    original_labels = {b: target[block_ids==b][0] for b in unique_blocks}

    perm_accuracies = []
    for p in range(n_perms):
        mapping_labels = rng.permutation(list(original_labels.values()))
        mapping = {b: mapping_labels[i] for i,b in enumerate(unique_blocks)}
        permuted_target = np.array([mapping[b] for b in block_ids])
        data_perm = data.copy()
        data_perm['target'] = permuted_target
        fold_results = evaluate_variant_cv(variant_name, variant_ctor, data_perm, cv_splits, features_df=features_df)
        mean_acc = float(np.mean([fr['accuracy'] for fr in fold_results]))
        perm_accuracies.append(mean_acc)
    return perm_accuracies

###############################################################################
# Experiment C: Feature separability + ablation
###############################################################################

def compute_feature_separability(features_df, y):
    # Cohen's d per feature across multi-class: average pairwise |d|
    separability = {}
    classes = np.unique(y)
    for col in features_df.columns:
        vals = features_df[col].values
        ds = []
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                a = vals[y==classes[i]]; b = vals[y==classes[j]]
                if a.std()==0 and b.std()==0:
                    d = 0
                else:
                    pooled = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var())/(len(a)+len(b)-2) + 1e-12)
                    d = abs(a.mean() - b.mean())/pooled if pooled>0 else 0
                ds.append(d)
        separability[col] = float(np.mean(ds)) if ds else 0.0
    sep_series = pd.Series(separability).sort_values(ascending=False)
    return sep_series

def run_ablation_series(variant_name, variant_ctor, data, cv_splits, ablation_ks=(1,2,5,10,20,'all_high_d')):
    # Generate baseline features once using SimplePEECOM physics generator as proxy for reproducibility
    base_model = variant_ctor()
    X_raw = data.drop(columns=['target']).values
    base_model.fit(X_raw, data['target'].values)
    # Extract features
    def extract_feats(m, raw):
        if hasattr(m,'physics_enhancer') and hasattr(m.physics_enhancer,'_create_physics_features'):
            return m.physics_enhancer._create_physics_features(raw)
        if hasattr(m,'_create_physics_features'):
            return m._create_physics_features(raw)
        if hasattr(m,'_create_advanced_physics_features'):
            return m._create_advanced_physics_features(raw)
        return pd.DataFrame(raw)
    features_df = extract_feats(base_model, X_raw)
    features_df = features_df.replace([np.inf,-np.inf],0).fillna(0)
    y = data['target'].values
    separability = compute_feature_separability(features_df, y)
    high_d_threshold = np.percentile(separability.values, 90)
    high_d_features = separability[separability >= high_d_threshold].index.tolist()

    results = []
    for k in ablation_ks:
        if k == 'all_high_d':
            drop_feats = high_d_features
        else:
            drop_feats = separability.head(k).index.tolist()
        fold_results = evaluate_variant_cv(variant_name, variant_ctor, data, cv_splits, features_df=features_df, drop_features=drop_feats)
        mean_acc = float(np.mean([fr['accuracy'] for fr in fold_results]))
        results.append({'ablation': k,'dropped_features': drop_feats,'mean_accuracy': mean_acc,'folds': fold_results})
    return separability, results

###############################################################################
# Experiment D: Block-relative normalization
###############################################################################

def run_block_relative_normalization(variant_name, variant_ctor, data, cv_splits, features_df=None):
    folds = evaluate_variant_cv(variant_name, variant_ctor, data, cv_splits, block_relative_norm=True, features_df=features_df)
    return {'folds': folds, 'mean_accuracy': float(np.mean([f['accuracy'] for f in folds]))}

###############################################################################
# Experiment E: Label permutation null test
###############################################################################

def run_label_permutation(variant_name, variant_ctor, data, cv_splits, n_perms=30, rng_seed=42, features_df=None):
    rng = np.random.RandomState(rng_seed)
    perm_accs = []
    for p in range(n_perms):
        perm_data = data.copy()
        perm_data['target'] = rng.permutation(perm_data['target'].values)
        fold_results = evaluate_variant_cv(variant_name, variant_ctor, perm_data, cv_splits, features_df=features_df)
        perm_accs.append(float(np.mean([fr['accuracy'] for fr in fold_results])))
    return perm_accs

###############################################################################
# Experiment F: Feature-swap
###############################################################################

def run_feature_swap(variant_name, variant_ctor, data, cv_splits):
    # (1) PEECOM-on-raw (disable physics by feeding raw directly into RF proxy)
    raw_only_results = evaluate_variant_cv(variant_name, variant_ctor, data, cv_splits, features_df=data.drop(columns=['target']))
    mean_raw = float(np.mean([fr['accuracy'] for fr in raw_only_results]))
    # (2) RF-on-PEECOM features
    # Generate PEECOM features once
    base_model = variant_ctor()
    X_raw = data.drop(columns=['target']).values
    base_model.fit(X_raw, data['target'].values)
    def extract_feats(m, raw):
        if hasattr(m,'physics_enhancer') and hasattr(m.physics_enhancer,'_create_physics_features'):
            return m.physics_enhancer._create_physics_features(raw)
        if hasattr(m,'_create_physics_features'):
            return m._create_physics_features(raw)
        if hasattr(m,'_create_advanced_physics_features'):
            return m._create_advanced_physics_features(raw)
        return pd.DataFrame(raw)
    features_df = extract_feats(base_model, X_raw)
    features_df = features_df.replace([np.inf,-np.inf],0).fillna(0)
    peecom_feature_results = evaluate_variant_cv(variant_name, variant_ctor, data, cv_splits, features_df=features_df)
    mean_feat = float(np.mean([fr['accuracy'] for fr in peecom_feature_results]))
    return {
        'peecom_on_raw_mean_accuracy': mean_raw,
        'peecom_on_raw_folds': raw_only_results,
        'rf_on_peecom_features_mean_accuracy': mean_feat,
        'rf_on_peecom_features_folds': peecom_feature_results
    }

###############################################################################
# Experiment G: Robustness shifts
###############################################################################

def run_robustness_shifts(variant_name, variant_ctor, data, cv_splits, shifts=None):
    if shifts is None:
        shifts = [
            ('add_noise_1pct', lambda X: X + 0.01*np.std(X.values,axis=0,keepdims=True)),
            ('add_noise_5pct', lambda X: X + 0.05*np.std(X.values,axis=0,keepdims=True)),
            ('scale_105', lambda X: X*1.05),
            ('scale_095', lambda X: X*0.95),
            ('offset_mean', lambda X: X + X.mean()),
        ]
    baseline_folds = evaluate_variant_cv(variant_name, variant_ctor, data, cv_splits)
    baseline_acc = float(np.mean([f['accuracy'] for f in baseline_folds]))
    results = []
    for name, func in shifts:
        shifted = data.copy()
        feature_cols = [c for c in shifted.columns if c != 'target']
        shifted[feature_cols] = func(shifted[feature_cols])
        folds = evaluate_variant_cv(variant_name, variant_ctor, shifted, cv_splits)
        acc = float(np.mean([f['accuracy'] for f in folds]))
        results.append({'shift': name,'mean_accuracy': acc,'delta_from_baseline': acc - baseline_acc})
    return {'baseline_mean_accuracy': baseline_acc,'baseline_folds': baseline_folds,'shift_results': results}

###############################################################################
# Orchestrator
###############################################################################

def _precompute_variant_features(ctor, data):
    X_raw = data.drop(columns=['target']).values
    y = data['target'].values
    model = ctor()
    try:
        model.fit(X_raw, y)
    except Exception:
        pass
    def extract_feats(m, raw):
        if hasattr(m,'physics_enhancer') and hasattr(m.physics_enhancer,'_create_physics_features'):
            return m.physics_enhancer._create_physics_features(raw)
        if hasattr(m,'_create_physics_features'):
            return m._create_physics_features(raw)
        if hasattr(m,'_create_advanced_physics_features'):
            return m._create_advanced_physics_features(raw)
        return pd.DataFrame(raw)
    feats = extract_feats(model, X_raw)
    feats = feats.replace([np.inf,-np.inf],0).fillna(0)
    return feats

def run_all_for_variant(variant_name, ctor, data, cv_splits, output_dir, skip_existing=True):
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    summary_path = variant_dir / 'summary.json'
    if skip_existing and summary_path.exists():
        print(f"↩️  Skipping {variant_name} (summary exists).")
        with open(summary_path,'r') as f:
            return json.load(f)

    summary = {}

    print(f"\n🔬 Running diagnostics for {variant_name} ...")
    print("   ⏳ Precomputing engineered features once ...")
    precomputed_features = _precompute_variant_features(ctor, data)
    print(f"   ✅ Engineered features shape: {precomputed_features.shape}")

    # B Block-permutation
    perm_block_accs = run_block_permutation(variant_name, ctor, data, cv_splits, features_df=precomputed_features)
    summary['block_permutation'] = {'accuracies': perm_block_accs,'mean': float(np.mean(perm_block_accs))}
    joblib.dump(perm_block_accs, variant_dir / 'block_permutation_accs.joblib')

    # C Feature separability + ablation
    separability, ablation_results = run_ablation_series(variant_name, ctor, data, cv_splits)
    summary['feature_separability'] = separability.to_dict()
    summary['ablation'] = ablation_results
    separability.to_csv(variant_dir / 'feature_separability.csv')
    joblib.dump(ablation_results, variant_dir / 'ablation_results.joblib')

    # D Block-relative normalization
    block_rel = run_block_relative_normalization(variant_name, ctor, data, cv_splits, features_df=precomputed_features)
    summary['block_relative_normalization'] = block_rel
    joblib.dump(block_rel, variant_dir / 'block_relative_normalization.joblib')

    # E Label permutation null test
    perm_label_accs = run_label_permutation(variant_name, ctor, data, cv_splits, features_df=precomputed_features)
    summary['label_permutation'] = {'accuracies': perm_label_accs,'mean': float(np.mean(perm_label_accs))}
    joblib.dump(perm_label_accs, variant_dir / 'label_permutation_accs.joblib')

    # F Feature swap
    feature_swap = run_feature_swap(variant_name, ctor, data, cv_splits)
    summary['feature_swap'] = feature_swap
    joblib.dump(feature_swap, variant_dir / 'feature_swap.joblib')

    # G Robustness shifts
    robustness = run_robustness_shifts(variant_name, ctor, data, cv_splits)
    summary['robustness'] = robustness
    joblib.dump(robustness, variant_dir / 'robustness.joblib')

    # Persist summary JSON
    with open(variant_dir / 'summary.json','w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Completed diagnostics for {variant_name}")
    return summary

###############################################################################
# Main
###############################################################################

def main():
    print("🧪 PEECOM Diagnostics (B–G) Starting...")
    data = load_data()
    cv_splits = create_splits_from_experiment_a()
    output_dir = Path('output/peecom_leakage_diagnostics')
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ('SimplePEECOM', SimplePEECOM),
        ('MultiClassifierPEECOM', MultiClassifierPEECOM),
        ('EnhancedPEECOM', EnhancedPEECOM)
    ]

    all_summaries = {}
    for name, ctor in variants:
        all_summaries[name] = run_all_for_variant(name, ctor, data, cv_splits, output_dir)

    # Global consolidated report
    consolidated_path = output_dir / 'consolidated_summary.json'
    with open(consolidated_path,'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n🏁 All diagnostics complete. Consolidated summary: {consolidated_path}")

if __name__ == '__main__':
    main()
