# block_mean_normalization_remediation.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import time

print("🔧 BLOCK-MEAN NORMALIZATION REMEDIATION")
print("=" * 50)

class TrainOnlyBlockMeanNormalizer(BaseEstimator, TransformerMixin):
    """
    Removes per-block mean differences using ONLY training data statistics
    This is the proper way to handle systematic block offsets
    """
    def __init__(self):
        self.block_means_ = None
        self.global_mean_ = None
        
    def fit(self, X, y=None, blocks=None):
        """Fit block mean corrections using training data only"""
        if blocks is None:
            raise ValueError("blocks parameter required")
            
        X = np.asarray(X)
        blocks = np.asarray(blocks)
        
        # Compute per-block means from training data only
        unique_blocks = np.unique(blocks)
        self.block_means_ = {}
        
        for block in unique_blocks:
            mask = (blocks == block)
            self.block_means_[block] = X[mask].mean(axis=0)
        
        # Global mean across all training data
        self.global_mean_ = X.mean(axis=0)
        
        return self
    
    def transform(self, X, blocks=None):
        """Apply block mean correction"""
        if self.block_means_ is None:
            raise ValueError("Must fit before transform")
        if blocks is None:
            raise ValueError("blocks parameter required for transform")
            
        X = np.asarray(X)
        blocks = np.asarray(blocks)
        X_corrected = X.copy()
        
        # Apply block-specific mean correction
        for block in self.block_means_:
            mask = (blocks == block)
            if np.any(mask):
                # Remove block mean, add global mean
                block_offset = self.block_means_[block] - self.global_mean_
                X_corrected[mask] = X[mask] - block_offset
        
        return X_corrected

# Load data
df = pd.read_csv('hydraulic_data_processed.csv')
feature_cols = [col for col in df.columns if col.startswith('f') and col[1:].isdigit()]
X = df[feature_cols].values
y = df['target'].values

# Generate blocks
blocks = np.zeros(len(y), dtype=int)
current_block = 0
for i in range(1, len(y)):
    if y[i] != y[i-1]:
        current_block += 1
    blocks[i] = current_block

print(f"Data: {len(feature_cols)} features, {len(X)} samples, {len(np.unique(blocks))} blocks")

def test_remediation_pipeline(X, y, blocks, use_normalization=True):
    """Test pipeline with/without block mean normalization"""
    
    if use_normalization:
        # Pipeline WITH block mean normalization
        pipeline = Pipeline([
            ('block_norm', TrainOnlyBlockMeanNormalizer()),
            ('clf', RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42))
        ])
        label = "WITH Block-Mean Normalization"
    else:
        # Pipeline WITHOUT normalization  
        pipeline = Pipeline([
            ('clf', RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42))
        ])
        label = "WITHOUT Normalization (Baseline)"
    
    print(f"\n🧪 Testing Pipeline {label}")
    print("-" * 40)
    
    # Custom CV that passes blocks to the normalizer
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    target_accs = []
    block_accs = []
    
    for train_idx, test_idx in skf.split(X, y):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        blocks_train, blocks_test = blocks[train_idx], blocks[test_idx]
        
        if use_normalization:
            # Fit normalizer on training data only
            pipeline.named_steps['block_norm'].fit(X_train, blocks=blocks_train)
            
            # Transform both sets  
            X_train_norm = pipeline.named_steps['block_norm'].transform(X_train, blocks=blocks_train)
            X_test_norm = pipeline.named_steps['block_norm'].transform(X_test, blocks=blocks_test)
            
            # Train classifier on normalized training data
            pipeline.named_steps['clf'].fit(X_train_norm, y_train)
            
            # Evaluate on normalized test data
            y_pred = pipeline.named_steps['clf'].predict(X_test_norm)
            target_acc = accuracy_score(y_test, y_pred)
            
            # Block predictor test on normalized data
            pipeline.named_steps['clf'].fit(X_train_norm, blocks_train)
            blocks_pred = pipeline.named_steps['clf'].predict(X_test_norm)
            block_acc = accuracy_score(blocks_test, blocks_pred)
            
        else:
            # No normalization - direct training
            pipeline.named_steps['clf'].fit(X_train, y_train)
            y_pred = pipeline.named_steps['clf'].predict(X_test)
            target_acc = accuracy_score(y_test, y_pred)
            
            # Block predictor test
            pipeline.named_steps['clf'].fit(X_train, blocks_train)
            blocks_pred = pipeline.named_steps['clf'].predict(X_test)
            block_acc = accuracy_score(blocks_test, blocks_pred)
        
        target_accs.append(target_acc)
        block_accs.append(block_acc)
    
    target_mean = np.mean(target_accs)
    target_std = np.std(target_accs)
    block_mean = np.mean(block_accs)
    block_std = np.std(block_accs)
    
    print(f"   Target accuracy: {target_mean:.4f} ± {target_std:.4f}")
    print(f"   Block accuracy:  {block_mean:.4f} ± {block_std:.4f}")
    
    # Assessment
    block_pass = block_mean <= 0.36
    print(f"   Block ≤ 0.36: {block_pass} {'✅' if block_pass else '❌'}")
    
    return {
        'target_acc': target_mean,
        'target_std': target_std,
        'block_acc': block_mean,
        'block_std': block_std,
        'block_pass': block_pass
    }

# Test baseline (no normalization)
baseline_results = test_remediation_pipeline(X, y, blocks, use_normalization=False)

# Test with block mean normalization
norm_results = test_remediation_pipeline(X, y, blocks, use_normalization=True)

print(f"\n📊 REMEDIATION COMPARISON")
print("=" * 40)
print(f"                    | Target Acc | Block Acc | Block Pass")
print(f"Baseline (Raw)      | {baseline_results['target_acc']:10.4f} | {baseline_results['block_acc']:9.4f} | {'✅' if baseline_results['block_pass'] else '❌'}")
print(f"Block-Mean Norm     | {norm_results['target_acc']:10.4f} | {norm_results['block_acc']:9.4f} | {'✅' if norm_results['block_pass'] else '❌'}")

# Calculate improvement
block_reduction = (baseline_results['block_acc'] - norm_results['block_acc']) / baseline_results['block_acc'] * 100
target_change = (norm_results['target_acc'] - baseline_results['target_acc']) / baseline_results['target_acc'] * 100

print(f"\n🎯 IMPROVEMENTS:")
print(f"   Block accuracy reduced by: {block_reduction:.1f}%")
print(f"   Target accuracy changed by: {target_change:+.1f}%")

if norm_results['block_pass']:
    print(f"\n🎉 REMEDIATION SUCCESS!")
    print(f"   Block mean normalization achieved block prediction ≤ 0.36")
    print(f"   Next step: Run full permutation test validation")
else:
    print(f"\n⚠️  PARTIAL SUCCESS:")
    print(f"   Block accuracy reduced but still > 0.36")  
    print(f"   May need additional normalization (covariance, etc.)")

print(f"\n✅ Block mean normalization analysis complete!")