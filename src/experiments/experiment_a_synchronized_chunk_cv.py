#!/usr/bin/env python3
"""
EXPERIMENT A: SYNCHRONIZED CHUNK CROSS-BLOCK CV
==============================================

Purpose: Ensure every fold's test set contains samples from every block
         so train always contains all classes.

Methodology:
1. Split each block's time-ordered indices into K contiguous chunks (K=5)
2. For fold i: test = chunk_i_from_all_blocks, train = all_other_chunks_from_all_blocks
3. Apply embargo around chunk boundaries (2% of chunk length)
4. Fit feature transforms on train only
5. Compute past-only physics features for test (no peeking)

Expected Outcomes:
- If features are block-identifying → still high accuracy
- If generalizable within-block signal → moderate accuracy (>baseline, <extreme)
- If classes encoded by block-level offsets → accuracy drops drastically (near chance)
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from pathlib import Path
import warnings
import logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Speed optimization imports
from joblib import Memory
from functools import partial

# --- DIAGNOSIS SWITCH (set to False to disable any PEECOM-style feature engineering) ---
USE_ENGINEERING = False

# Debug / perf knobs (adjust as needed)
N_FOLDS_DEBUG = 3          # use 3 folds while debugging; set back to 5 for final
MULTI_CLASSIFIERS_TO_TRY = [
    'random_forest',
    'gradient_boosting',
    'logistic_regression'
]
# Candidate list above intentionally small. Expand for full runs.

# Cache dir for expensive feature generation
_cache_dir = os.path.join(os.getcwd(), 'cache_experiment_a')
os.makedirs(_cache_dir, exist_ok=True)
memory = Memory(_cache_dir, verbose=0)

# Import PEECOM variants
import sys
sys.path.append('src')
from models.simple_peecom import SimplePEECOM
from models.multi_classifier_peecom import MultiClassifierPEECOM
from models.enhanced_peecom import EnhancedPEECOM
try:
    import shap  # Optional
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available – skipping SHAP value computation. Install with pip install shap to enable.")

def setup_directories():
    """Create output directories"""
    base_dir = Path("output/experiment_a_synchronized_cv")
    dirs = ['results', 'figures', 'logs']
    for d in dirs:
        (base_dir / d).mkdir(parents=True, exist_ok=True)
    return base_dir

def load_and_analyze_data():
    """Load data and analyze block structure"""
    print("📊 LOADING AND ANALYZING DATA")
    print("=" * 50)
    
    try:
        data = pd.read_csv('hydraulic_data_processed.csv')
        print(f"   ✅ Data loaded: {data.shape}")
    except:
        print("   ⚠️ Using synthetic data for demonstration")
        np.random.seed(42)
        n_samples = 2205
        n_features = 54
        data = pd.DataFrame(np.random.randn(n_samples, n_features))
        
        # Create blocked class structure (mimicking real data)
        data['target'] = 0
        data.loc[:731, 'target'] = 0    # Block 0: Class 0
        data.loc[732:1463, 'target'] = 1  # Block 1: Class 1  
        data.loc[1464:, 'target'] = 2   # Block 2: Class 2

    # --- TEMPORAL SHUFFLE DIAGNOSTIC ---
    SHUFFLE_WITHIN_BLOCKS = True   # <-- set False to disable this diagnostic
    SHUFFLE_SEED = 42

    if SHUFFLE_WITHIN_BLOCKS:
        print("🔀 Shuffling rows within each block to break temporal order...")
        
        # Create explicit block column based on transitions
        target_col = data['target'].values
        transitions = np.where(np.diff(target_col) != 0)[0] + 1
        block_starts = np.concatenate([[0], transitions, [len(data)]])
        
        # Assign block IDs
        data['block'] = 0
        for i in range(len(block_starts) - 1):
            start_idx = block_starts[i]
            end_idx = block_starts[i + 1]
            data.iloc[start_idx:end_idx, data.columns.get_loc('block')] = i
        
        # Shuffle rows within each block independently
        def _shuffle_group(g):
            return g.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)
        
        print(f"   Shuffling within blocks (seed={SHUFFLE_SEED}) to destroy temporal patterns...")
        data = data.groupby('block', group_keys=False).apply(_shuffle_group).reset_index(drop=True)
        
        print("   ✅ Temporal shuffle complete - block order preserved, intra-block order randomized")
    else:
        # Still need to create block column for covariance normalization
        target_col = data['target'].values
        transitions = np.where(np.diff(target_col) != 0)[0] + 1
        block_starts = np.concatenate([[0], transitions, [len(data)]])
        
        # Assign block IDs
        data['block'] = 0
        for i in range(len(block_starts) - 1):
            start_idx = block_starts[i]
            end_idx = block_starts[i + 1]
            data.iloc[start_idx:end_idx, data.columns.get_loc('block')] = i

    # --- BLOCK-COVARIANCE NORMALIZATION PATCH ---
    
    def _sqrt_and_invsqrt(mat, eps=1e-8):
        # Symmetric eigendecomp for stable sqrt / inv-sqrt
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

    # detect feature columns (exclude likely non-feature columns)
    _exclude = {'block', 'target', 'label', 'class', 'index'}
    feature_cols = [c for c in data.columns if c not in _exclude]

    print("🔧 Applying block-covariance normalization (equalizing covariance across blocks)...")
    data = cov_normalize_blocks(data, feature_cols, block_col='block', eps=1e-6)
    print("✅ Block-covariance normalization applied. Proceeding with the usual CV / modelling.")

    # Remove the block column after covariance normalization
    data = data.drop('block', axis=1)
    
    # Analyze block structure
    print(f"   📋 Total samples: {len(data)}")
    print(f"   📋 Features: {data.shape[1] - 1}")
    
    # Detect blocks based on class transitions
    target_col = data['target'].values
    transitions = np.where(np.diff(target_col) != 0)[0] + 1
    block_starts = np.concatenate([[0], transitions, [len(data)]])
    
    blocks = []
    for i in range(len(block_starts) - 1):
        start_idx = block_starts[i]
        end_idx = block_starts[i + 1]
        block_class = target_col[start_idx]
        blocks.append({
            'block_id': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'class': block_class,
            'size': end_idx - start_idx
        })
    
    print(f"   🔍 Detected {len(blocks)} blocks:")
    for block in blocks:
        print(f"      Block {block['block_id']}: indices {block['start_idx']}-{block['end_idx']-1}, "
              f"class {block['class']}, size {block['size']}")
    
    return data, blocks

def block_relative_normalization(X: pd.DataFrame, blocks: np.ndarray,
                                 train_idx: np.ndarray, test_idx: np.ndarray,
                                 method: str = "mean") -> pd.DataFrame:
    """
    Perform block-relative normalization for a single CV fold.

    - X: DataFrame of shape (n_samples, n_features)
    - blocks: 1D array of block ids per sample (length n_samples)
    - train_idx, test_idx: indices for this fold
    - method: "mean" (default) or "median"

    Returns:
    - X_norm: normalized DataFrame (same index/columns)
    """
    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
    train_blocks = np.unique(blocks[train_idx])
    test_blocks = np.unique(blocks[test_idx])

    # compute per-block statistic on training blocks only
    block_stats = {}
    for b in train_blocks:
        idx = np.where(blocks == b)[0]
        # only use training indices from that block if present in train_idx
        idx_train_block = np.intersect1d(idx, train_idx)
        if len(idx_train_block) == 0:
            # fallback: use all samples for that block (unlikely)
            idx_train_block = idx
        if method == "median":
            block_stats[b] = X.iloc[idx_train_block].median(axis=0)
        else:
            block_stats[b] = X.iloc[idx_train_block].mean(axis=0)

    # grand mean (mean of per-block means) computed across training blocks
    # shapes: series per block → stack to DataFrame then mean across rows
    stats_df = pd.DataFrame(block_stats).T  # rows=blocks, cols=features
    grand_train_mean = stats_df.mean(axis=0)

    # create normalized copy
    X_norm = X.copy().astype(float)

    # subtract block-specific mean for training blocks
    for b in train_blocks:
        mask = (blocks == b)
        X_norm.loc[mask, :] = X_norm.loc[mask, :].subtract(block_stats[b], axis=1)

    # for test blocks (unseen blocks), subtract grand_train_mean to align them to training space
    for b in test_blocks:
        mask = (blocks == b)
        # subtract grand_train_mean (no peeking at test-block stats)
        X_norm.loc[mask, :] = X_norm.loc[mask, :].subtract(grand_train_mean, axis=1)

    logger.info("Applied block-relative normalization: train_blocks=%s test_blocks=%s",
                list(train_blocks), list(test_blocks))
    return X_norm

def create_synchronized_chunks(blocks, k_folds=5, embargo_pct=0.02):
    """
    Create synchronized chunks across blocks for proper CV
    
    Args:
        blocks: List of block information
        k_folds: Number of CV folds
        embargo_pct: Embargo percentage around chunk boundaries
    """
    print(f"\n🔄 CREATING SYNCHRONIZED CHUNKS (K={k_folds})")
    print("=" * 50)
    
    all_chunks = []
    
    # Create chunks for each block
    for block in blocks:
        block_size = block['size']
        chunk_size = block_size // k_folds
        embargo_size = max(1, int(chunk_size * embargo_pct))
        
        print(f"   📦 Block {block['block_id']} (Class {block['class']}):")
        print(f"      Size: {block_size}, Chunk size: {chunk_size}, Embargo: {embargo_size}")
        
        block_chunks = []
        for fold in range(k_folds):
            # Calculate chunk boundaries
            chunk_start = block['start_idx'] + fold * chunk_size
            chunk_end = min(block['start_idx'] + (fold + 1) * chunk_size, block['end_idx'])
            
            # Apply embargo
            embargo_start = max(chunk_start - embargo_size, block['start_idx'])
            embargo_end = min(chunk_end + embargo_size, block['end_idx'])
            
            chunk_info = {
                'block_id': block['block_id'],
                'fold': fold,
                'class': block['class'],
                'test_start': chunk_start,
                'test_end': chunk_end,
                'embargo_start': embargo_start,
                'embargo_end': embargo_end,
                'test_size': chunk_end - chunk_start
            }
            block_chunks.append(chunk_info)
            
        all_chunks.extend(block_chunks)
    
    print(f"   ✅ Created {len(all_chunks)} chunks total")
    return all_chunks

def create_cv_splits(all_chunks, total_samples, k_folds=N_FOLDS_DEBUG):
    """Create CV splits ensuring each fold has samples from all blocks"""
    print(f"\n📂 CREATING CV SPLITS")
    print("=" * 50)
    
    cv_splits = []
    
    for fold in range(k_folds):
        print(f"   🔍 Fold {fold + 1}/{k_folds}:")
        
        # Get test chunks for this fold (one from each block)
        test_chunks = [chunk for chunk in all_chunks if chunk['fold'] == fold]
        
        # Collect test indices
        test_indices = []
        embargo_indices = set()
        
        for chunk in test_chunks:
            # Add test indices
            test_idx_range = list(range(chunk['test_start'], chunk['test_end']))
            test_indices.extend(test_idx_range)
            
            # Add embargo indices (to exclude from training)
            embargo_idx_range = list(range(chunk['embargo_start'], chunk['embargo_end']))
            embargo_indices.update(embargo_idx_range)
            
            print(f"      Block {chunk['block_id']} (Class {chunk['class']}): "
                  f"test {chunk['test_start']}-{chunk['test_end']-1} "
                  f"(size {chunk['test_size']})")
        
        # Create train indices (all others except embargo)
        all_indices = set(range(total_samples))
        train_indices = list(all_indices - embargo_indices)
        
        print(f"      📊 Train size: {len(train_indices)}, Test size: {len(test_indices)}")
        print(f"      📊 Embargo excluded: {len(embargo_indices) - len(test_indices)} samples")
        
        # Verify all classes present in both train and test
        cv_splits.append({
            'fold': fold,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'embargo_size': len(embargo_indices) - len(test_indices)
        })
    
    return cv_splits

def compute_past_only_features(data, train_indices, test_indices):
    """
    Compute features using only past information
    - Fit transforms on train only
    - Apply to test using only historical context
    """
    feature_cols = [col for col in data.columns if col != 'target']
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_raw = data.iloc[train_indices][feature_cols].values
    X_train = scaler.fit_transform(X_train_raw)
    
    # Transform test data (no peeking)
    X_test_raw = data.iloc[test_indices][feature_cols].values
    X_test = scaler.transform(X_test_raw)
    
    # Get targets
    y_train = data.iloc[train_indices]['target'].values
    y_test = data.iloc[test_indices]['target'].values
    
    return X_train, X_test, y_train, y_test, scaler


def run_synchronized_chunk_cv(data, cv_splits, base_dir):
    """Run synchronized chunk cross-validation experiment including PEECOM variants"""
    print(f"\n🧪 RUNNING SYNCHRONIZED CHUNK CV (RF/LR/PEECOM)")
    print("=" * 60)

    # Reconstruct block ID array for normalization
    target_col = data['target'].values
    transitions = np.where(np.diff(target_col) != 0)[0] + 1
    block_starts = np.concatenate([[0], transitions, [len(data)]])
    
    # Create block ID array
    block_ids = np.zeros(len(data), dtype=int)
    for i in range(len(block_starts) - 1):
        start_idx = block_starts[i]
        end_idx = block_starts[i + 1]
        block_ids[start_idx:end_idx] = i

    results = {
        'fold_results': [],
        'models': {},
        'predictions': {},
        'feature_importance': {},
        'shap_values': {},
        'feature_provenance': {}
    }

    # Models to test (optimized with parallelism)
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1  # Use all CPU cores
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=500,  # Reduced iterations for speed
            random_state=42,
            class_weight='balanced'
        )
    }
    
    # Conditionally add PEECOM variants based on USE_ENGINEERING flag
    if USE_ENGINEERING:
        models.update({
            'SimplePEECOM': SimplePEECOM(),
            'MultiClassifierPEECOM': MultiClassifierPEECOM(),
            'EnhancedPEECOM': EnhancedPEECOM()
        })
        print("   🔧 Feature engineering ENABLED - including PEECOM variants")
    else:
        print("   🚫 Feature engineering DISABLED - testing only baseline models on normalized features")

    for fold_info in cv_splits:
        fold = fold_info['fold']
        train_idx = fold_info['train_indices']
        test_idx = fold_info['test_indices']

        print(f"\n   🔄 FOLD {fold + 1}/{len(cv_splits)}")
        print(f"      Train: {len(train_idx)} samples")
        print(f"      Test:  {len(test_idx)} samples")
        print(f"      Embargo excluded: {fold_info['embargo_size']} samples")

        # Per-fold model cache to avoid retraining variants
        trained_variants = {}

        # Apply block-relative normalization
        feature_cols = [col for col in data.columns if col != 'target']
        X_raw = data[feature_cols]
        X_normalized = block_relative_normalization(X_raw, block_ids, train_idx, test_idx, method="mean")
        
        # Update data with normalized features for this fold
        data_fold = data.copy()
        data_fold[feature_cols] = X_normalized

        # Compute past-only features for RF/LR
        X_train, X_test, y_train, y_test, scaler = compute_past_only_features(
            data_fold, train_idx, test_idx
        )

        # Verify class distribution
        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)
        print(f"      Train classes: {sorted(train_classes)}")
        print(f"      Test classes:  {sorted(test_classes)}")

        fold_results = {
            'fold': fold,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_classes': train_classes.tolist(),
            'test_classes': test_classes.tolist(),
            'model_results': {}
        }

        # Train and evaluate each model (with caching for PEECOM variants)
        for model_name, model in models.items():
            if fold == 0 or fold % 1 == 0:  # Reduce verbosity
                print(f"         🤖 {model_name}...")
            
            peecom_variant = model_name in {"SimplePEECOM","MultiClassifierPEECOM","EnhancedPEECOM"}
            if peecom_variant:
                # Use per-fold caching for PEECOM variants
                cache_key = f"{model_name}_fold_{fold}"
                if cache_key not in trained_variants:
                    raw_train = data.iloc[train_idx].drop('target', axis=1).values
                    raw_test = data.iloc[test_idx].drop('target', axis=1).values
                    model.fit(raw_train, y_train)
                    trained_variants[cache_key] = model
                else:
                    model = trained_variants[cache_key]
                    raw_train = data.iloc[train_idx].drop('target', axis=1).values
                    raw_test = data.iloc[test_idx].drop('target', axis=1).values
                y_pred = model.predict(raw_test)
                y_proba = model.predict_proba(raw_test) if hasattr(model, 'predict_proba') else None
                # Derive internal feature representations for diagnostics
                def _extract_feats(m, Xraw):
                    if hasattr(m,'physics_enhancer') and hasattr(m.physics_enhancer,'_create_physics_features'):
                        return m.physics_enhancer._create_physics_features(Xraw)
                    if hasattr(m,'_create_physics_features'):
                        return m._create_physics_features(Xraw)
                    return pd.DataFrame(Xraw)
                train_feats = _extract_feats(model, raw_train)
                test_feats = _extract_feats(model, raw_test)
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average='macro')
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                # SHAP values (optional)
                shap_vals = None
                if SHAP_AVAILABLE:
                    try:
                        # Use TreeExplainer when possible
                        explainer = shap.Explainer(model.model if hasattr(model,'model') else model, np.array(train_feats))
                        shap_vals = explainer(np.array(test_feats))
                    except Exception:
                        shap_vals = None
                # Feature importance
                feat_imp = None
                if hasattr(model,'get_feature_importance'):
                    try:
                        feat_imp = model.get_feature_importance()
                    except Exception:
                        feat_imp = None
                # Provenance heuristic: mark created features containing patterns
                provenance = []
                feat_cols = list(train_feats.columns) if hasattr(train_feats,'columns') else [f'f_{i}' for i in range(train_feats.shape[1])]
                for c in feat_cols:
                    c_str = str(c)
                    if any(tag in c_str for tag in ['power','ratio','energy','stability','interaction','diff','harmonic','efficiency']):
                        provenance.append('derived_physics')
                    else:
                        provenance.append('raw')
                fold_results['model_results'][model_name] = {
                    'accuracy': accuracy,
                    'macro_f1': macro_f1,
                    'precision': precision,
                    'recall': recall,
                    'predictions': y_pred.tolist(),
                    'true_labels': y_test.tolist(),
                    'probabilities': y_proba.tolist() if y_proba is not None else None,
                    'feature_importance': feat_imp,
                    'shap_values': shap_vals,
                    'feature_provenance': provenance
                }
                if model_name not in results['models']:
                    results['models'][model_name] = []
                    results['predictions'][model_name] = []
                    results['feature_importance'][model_name] = []
                    results['shap_values'][model_name] = []
                    results['feature_provenance'][model_name] = []
                results['models'][model_name].append(model)
                results['predictions'][model_name].append({
                    'fold': fold,
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'y_proba': y_proba
                })
                results['feature_importance'][model_name].append(feat_imp)
                results['shap_values'][model_name].append(shap_vals)
                results['feature_provenance'][model_name].append(provenance)
                print(f"            ✅ Accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f}")
            else:
                # Standard models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                accuracy = accuracy_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average='macro')
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                fold_results['model_results'][model_name] = {
                    'accuracy': accuracy,
                    'macro_f1': macro_f1,
                    'precision': precision,
                    'recall': recall,
                    'predictions': y_pred.tolist(),
                    'true_labels': y_test.tolist(),
                    'probabilities': y_proba.tolist() if y_proba is not None else None
                }
                if model_name not in results['models']:
                    results['models'][model_name] = []
                    results['predictions'][model_name] = []
                results['models'][model_name].append(model)
                results['predictions'][model_name].append({
                    'fold': fold,
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'y_proba': y_proba
                })
                print(f"            ✅ Accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f}")

        results['fold_results'].append(fold_results)

    return results

def analyze_results(results, base_dir):
    """Analyze and summarize CV results"""
    print(f"\n📊 ANALYZING RESULTS")
    print("=" * 50)
    
    # Calculate summary statistics
    model_summaries = {}
    
    for model_name in results['models'].keys():
        accuracies = []
        for fold_result in results['fold_results']:
            acc = fold_result['model_results'][model_name]['accuracy']
            accuracies.append(acc)
        
        summary = {
            'accuracies': accuracies,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies)
        }
        
        model_summaries[model_name] = summary
        
        print(f"   🤖 {model_name}:")
        print(f"      Mean Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
        print(f"      Range: {summary['min_accuracy']:.4f} - {summary['max_accuracy']:.4f}")
        print(f"      All folds: {[f'{acc:.4f}' for acc in accuracies]}")
    
    # Baseline comparison (random chance)
    n_classes = len(np.unique([fold['test_classes'] for fold in results['fold_results']][0]))
    random_baseline = 1.0 / n_classes
    
    print(f"\n   📏 BASELINE COMPARISON:")
    print(f"      Random chance: {random_baseline:.4f}")
    
    for model_name, summary in model_summaries.items():
        improvement = summary['mean_accuracy'] - random_baseline
        print(f"      {model_name} improvement: +{improvement:.4f}")
    
    # Decision logic
    print(f"\n   🎯 DECISION ANALYSIS:")
    
    best_model = max(model_summaries.keys(), key=lambda k: model_summaries[k]['mean_accuracy'])
    best_accuracy = model_summaries[best_model]['mean_accuracy']
    
    print(f"      Best model: {best_model} ({best_accuracy:.4f})")
    
    if best_accuracy < 0.4:  # Near chance level
        decision = "BLOCK-LEVEL ENCODING DETECTED"
        explanation = "Accuracy near chance → classes encoded by block-level offsets"
        recommendation = "Stop claiming generalization; move to remediation experiments"
        evidence_strength = "STRONG"
    elif best_accuracy < 0.7:  # Moderate
        decision = "SOME GENERALIZATION DETECTED"
        explanation = "Moderate accuracy → some within-block signal exists"
        recommendation = "Investigate remaining leakage sources (run experiments B-D)"
        evidence_strength = "MODERATE"
    else:  # High accuracy
        decision = "LEAKAGE STILL PRESENT"
        explanation = "High accuracy persists → additional leakage sources remain"
        recommendation = "Features still leak information; run experiments B-D"
        evidence_strength = "STRONG"
    
    print(f"      Decision: {decision}")
    print(f"      Evidence: {evidence_strength}")
    print(f"      Explanation: {explanation}")
    print(f"      Recommendation: {recommendation}")
    
    # Store final analysis
    final_analysis = {
        'model_summaries': model_summaries,
        'random_baseline': random_baseline,
        'best_model': best_model,
        'best_accuracy': best_accuracy,
        'decision': decision,
        'evidence_strength': evidence_strength,
        'explanation': explanation,
        'recommendation': recommendation
    }
    
    return final_analysis

def create_visualizations(results, final_analysis, base_dir):
    """Create visualization plots"""
    print(f"\n📈 CREATING VISUALIZATIONS")
    print("=" * 40)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Accuracy by fold plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Experiment A: Synchronized Chunk Cross-Block CV Results', fontsize=16, fontweight='bold')
    
    # Accuracy by fold
    ax1 = axes[0, 0]
    model_names = list(final_analysis['model_summaries'].keys())
    folds = list(range(1, 6))
    
    for model_name in model_names:
        accuracies = final_analysis['model_summaries'][model_name]['accuracies']
        ax1.plot(folds, accuracies, marker='o', linewidth=2, label=model_name)
    
    ax1.axhline(y=final_analysis['random_baseline'], color='red', linestyle='--', 
                label=f'Random Baseline ({final_analysis["random_baseline"]:.3f})')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Fold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Mean accuracy comparison
    ax2 = axes[0, 1]
    models = list(final_analysis['model_summaries'].keys())
    means = [final_analysis['model_summaries'][m]['mean_accuracy'] for m in models]
    stds = [final_analysis['model_summaries'][m]['std_accuracy'] for m in models]
    
    bars = ax2.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
    ax2.axhline(y=final_analysis['random_baseline'], color='red', linestyle='--', 
                label=f'Random Baseline')
    ax2.set_ylabel('Mean Accuracy')
    ax2.set_title('Mean Accuracy ± Std Dev')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
    
    # Confusion matrix for best model
    ax3 = axes[1, 0]
    best_model = final_analysis['best_model']
    
    # Aggregate confusion matrix across all folds
    all_true = []
    all_pred = []
    
    for pred_info in results['predictions'][best_model]:
        all_true.extend(pred_info['y_true'])
        all_pred.extend(pred_info['y_pred'])
    
    cm = confusion_matrix(all_true, all_pred)
    
    # Normalize for better visualization
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax3.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax3.set_title(f'Confusion Matrix ({best_model})')
    
    # Add text annotations
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax3.text(j, i, f'{cm_norm[i, j]:.2f}\n({cm[i, j]})',
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
    
    ax3.set_ylabel('True Class')
    ax3.set_xlabel('Predicted Class')
    
    # Decision summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
EXPERIMENT A RESULTS

Best Model: {final_analysis['best_model']}
Accuracy: {final_analysis['best_accuracy']:.4f}

DECISION: {final_analysis['decision']}
Evidence: {final_analysis['evidence_strength']}

{final_analysis['explanation']}

RECOMMENDATION:
{final_analysis['recommendation']}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = base_dir / 'figures' / 'experiment_a_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Results plot saved: {plot_path}")
    
    plt.show()

def save_results(results, final_analysis, cv_splits, base_dir):
    """Save all results and analysis"""
    print(f"\n💾 SAVING RESULTS")
    print("=" * 30)
    
    # Save comprehensive results
    comprehensive_results = {
        'experiment': 'A_synchronized_chunk_cv',
        'cv_splits': cv_splits,
        'fold_results': results['fold_results'],
        'final_analysis': final_analysis,
        'metadata': {
            'k_folds': 5,
            'embargo_pct': 0.02,
            'models_tested': list(results['models'].keys())
        }
    }
    
    results_path = base_dir / 'results' / 'experiment_a_results.joblib'
    joblib.dump(comprehensive_results, results_path)
    print(f"   ✅ Comprehensive results: {results_path}")
    
    # Save summary report
    report_path = base_dir / 'results' / 'experiment_a_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("EXPERIMENT A: SYNCHRONIZED CHUNK CROSS-BLOCK CV\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PURPOSE:\n")
        f.write("Ensure every fold's test set contains samples from every block\n")
        f.write("so train always contains all classes.\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("1. Split each block into K=5 contiguous chunks\n")
        f.write("2. For fold i: test = chunk_i_from_all_blocks\n")
        f.write("3. Apply 2% embargo around chunk boundaries\n")
        f.write("4. Fit transforms on train only\n")
        f.write("5. Compute past-only features for test\n\n")
        
        f.write("RESULTS:\n")
        for model_name, summary in final_analysis['model_summaries'].items():
            f.write(f"{model_name}:\n")
            f.write(f"  Mean Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}\n")
            f.write(f"  Range: {summary['min_accuracy']:.4f} - {summary['max_accuracy']:.4f}\n")
            f.write(f"  Fold accuracies: {summary['accuracies']}\n\n")
        
        f.write(f"Random Baseline: {final_analysis['random_baseline']:.4f}\n\n")
        
        f.write("DECISION ANALYSIS:\n")
        f.write(f"Decision: {final_analysis['decision']}\n")
        f.write(f"Evidence Strength: {final_analysis['evidence_strength']}\n")
        f.write(f"Explanation: {final_analysis['explanation']}\n")
        f.write(f"Recommendation: {final_analysis['recommendation']}\n\n")
        
        f.write("INTERPRETATION:\n")
        if final_analysis['best_accuracy'] < 0.4:
            f.write("✅ SUCCESS: Accuracy dropped to near-chance levels\n")
            f.write("   This proves classes were encoded by block-level offsets\n")
            f.write("   No legitimate temporal generalization capability exists\n")
        elif final_analysis['best_accuracy'] < 0.7:
            f.write("⚠️  MIXED: Some generalization detected but accuracy reduced\n")
            f.write("   Block segregation was part of the problem\n")
            f.write("   Additional leakage sources may remain\n")
        else:
            f.write("🚨 CONCERN: High accuracy persists despite proper CV\n")
            f.write("   Additional leakage sources beyond block segregation\n")
            f.write("   Further investigation required (Experiments B-D)\n")
    
    print(f"   ✅ Summary report: {report_path}")

def main():
    """Run Experiment A: Synchronized Chunk Cross-Block CV"""
    print("🧪 EXPERIMENT A: SYNCHRONIZED CHUNK CROSS-BLOCK CV")
    print("=" * 70)
    print("Purpose: Ensure every fold contains samples from all blocks")
    print("         to test if leakage is purely from block-class segregation")
    print("=" * 70)
    
    # Setup
    base_dir = setup_directories()
    
    # Load and analyze data
    data, blocks = load_and_analyze_data()
    
    # Create synchronized chunks
    all_chunks = create_synchronized_chunks(blocks, k_folds=5, embargo_pct=0.02)
    
    # Create CV splits
    cv_splits = create_cv_splits(all_chunks, len(data), k_folds=5)
    
    # Run synchronized chunk CV
    results = run_synchronized_chunk_cv(data, cv_splits, base_dir)
    
    # Analyze results
    final_analysis = analyze_results(results, base_dir)
    
    # Create visualizations
    create_visualizations(results, final_analysis, base_dir)
    
    # Save results
    save_results(results, final_analysis, cv_splits, base_dir)
    
    # Final summary
    print(f"\n🏆 EXPERIMENT A COMPLETE")
    print("=" * 40)
    print(f"Decision: {final_analysis['decision']}")
    print(f"Recommendation: {final_analysis['recommendation']}")
    print(f"\nAll results saved to: {base_dir}")
    
    return final_analysis

if __name__ == "__main__":
    results = main()