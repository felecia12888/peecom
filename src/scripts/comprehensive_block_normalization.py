# comprehensive_block_normalization.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import time

print("🔧 COMPREHENSIVE BLOCK NORMALIZATION (Mean + Covariance)")
print("=" * 60)

class ComprehensiveBlockNormalizer(BaseEstimator, TransformerMixin):
    """
    Removes per-block mean AND covariance differences using ONLY training data
    This addresses both mean shifts and covariance structure differences
    """
    def __init__(self, eps=1e-6):
        self.block_means_ = None
        self.block_covs_ = None
        self.global_mean_ = None
        self.global_cov_ = None
        self.eps = eps
        
    def _sqrt_and_invsqrt(self, mat):
        """Helper for covariance square root"""
        vals, vecs = np.linalg.eigh(mat)
        vals = np.clip(vals, self.eps, None)
        sqrt = (vecs * np.sqrt(vals)) @ vecs.T
        inv_sqrt = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
        return sqrt, inv_sqrt
        
    def fit(self, X, y=None, blocks=None):
        """Fit comprehensive block normalization using training data only"""
        if blocks is None:
            raise ValueError("blocks parameter required")
            
        X = np.asarray(X, dtype=float)
        blocks = np.asarray(blocks)
        
        # Compute global statistics from all training data
        self.global_mean_ = X.mean(axis=0)
        self.global_cov_ = np.cov(X.T) + np.eye(X.shape[1]) * self.eps
        
        # Compute per-block statistics from training data only
        unique_blocks = np.unique(blocks)
        self.block_means_ = {}
        self.block_covs_ = {}
        
        for block in unique_blocks:
            mask = (blocks == block)
            X_block = X[mask]
            
            self.block_means_[block] = X_block.mean(axis=0)
            self.block_covs_[block] = np.cov(X_block.T) + np.eye(X.shape[1]) * self.eps
        
        return self
    
    def transform(self, X, blocks=None):
        """Apply comprehensive block normalization"""
        if self.block_means_ is None:
            raise ValueError("Must fit before transform")
        if blocks is None:
            raise ValueError("blocks parameter required for transform")
            
        X = np.asarray(X, dtype=float)
        blocks = np.asarray(blocks)
        X_normalized = X.copy()
        
        # Get global covariance square roots
        _, global_inv_sqrt = self._sqrt_and_invsqrt(self.global_cov_)
        global_sqrt, _ = self._sqrt_and_invsqrt(self.global_cov_)
        
        # Apply block-specific normalization
        for block in self.block_means_:
            mask = (blocks == block)
            if np.any(mask):
                X_block = X_normalized[mask]
                
                # Step 1: Remove block mean, center on global mean
                X_block = X_block - self.block_means_[block] + self.global_mean_
                
                # Step 2: Covariance normalization
                # Transform to standard space using block covariance
                _, block_inv_sqrt = self._sqrt_and_invsqrt(self.block_covs_[block])
                X_block = (X_block - self.global_mean_) @ block_inv_sqrt.T
                
                # Transform back using global covariance
                X_block = X_block @ global_sqrt.T + self.global_mean_
                
                X_normalized[mask] = X_block
        
        return X_normalized

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

def test_comprehensive_normalization(X, y, blocks):
    """Test comprehensive block normalization pipeline"""
    
    print(f"\n🧪 Testing Comprehensive Block Normalization")
    print("-" * 50)
    
    # Pipeline with comprehensive normalization
    pipeline = Pipeline([
        ('comp_norm', ComprehensiveBlockNormalizer()),
        ('clf', RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42))
    ])
    
    # Custom CV that passes blocks to the normalizer
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    target_accs = []
    block_accs = []
    
    start_time = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"   Processing fold {fold + 1}/3...")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        blocks_train, blocks_test = blocks[train_idx], blocks[test_idx]
        
        # Fit normalizer on training data only
        pipeline.named_steps['comp_norm'].fit(X_train, blocks=blocks_train)
        
        # Transform both sets  
        X_train_norm = pipeline.named_steps['comp_norm'].transform(X_train, blocks=blocks_train)
        X_test_norm = pipeline.named_steps['comp_norm'].transform(X_test, blocks=blocks_test)
        
        # Train classifier on normalized training data
        pipeline.named_steps['clf'].fit(X_train_norm, y_train)
        
        # Evaluate target prediction on normalized test data
        y_pred = pipeline.named_steps['clf'].predict(X_test_norm)
        target_acc = accuracy_score(y_test, y_pred)
        
        # Block predictor test on normalized data
        pipeline.named_steps['clf'].fit(X_train_norm, blocks_train)
        blocks_pred = pipeline.named_steps['clf'].predict(X_test_norm)
        block_acc = accuracy_score(blocks_test, blocks_pred)
        
        target_accs.append(target_acc)
        block_accs.append(block_acc)
        
        print(f"      Target: {target_acc:.4f}, Block: {block_acc:.4f}")
    
    target_mean = np.mean(target_accs)
    target_std = np.std(target_accs)
    block_mean = np.mean(block_accs)
    block_std = np.std(block_accs)
    
    print(f"\n   📊 RESULTS:")
    print(f"   Target accuracy: {target_mean:.4f} ± {target_std:.4f}")
    print(f"   Block accuracy:  {block_mean:.4f} ± {block_std:.4f}")
    print(f"   Processing time: {time.time() - start_time:.1f} seconds")
    
    # Assessment
    block_pass = block_mean <= 0.36
    chance_level = 1.0 / len(np.unique(y))
    target_near_chance = abs(target_mean - chance_level) < 0.1
    
    print(f"\n   🎯 ASSESSMENT:")
    print(f"   Block ≤ 0.36: {block_pass} {'✅' if block_pass else '❌'}")
    print(f"   Target ≈ chance ({chance_level:.3f}): {target_near_chance} {'✅' if target_near_chance else '❌'}")
    
    overall_success = block_pass and target_near_chance
    print(f"   OVERALL: {'🎉 REMEDIATION SUCCESS!' if overall_success else '⚠️ PARTIAL SUCCESS'}")
    
    return {
        'target_acc': target_mean,
        'target_std': target_std,
        'block_acc': block_mean,
        'block_std': block_std,
        'block_pass': block_pass,
        'target_near_chance': target_near_chance,
        'overall_success': overall_success
    }

# Run comprehensive normalization test
results = test_comprehensive_normalization(X, y, blocks)

# Quick permutation test if remediation looks successful
if results['block_pass']:
    print(f"\n🧪 QUICK PERMUTATION TEST (20 permutations)")
    print("-" * 40)
    
    # Baseline from above
    baseline = results['target_acc']
    
    # Quick permutation test
    rng = np.random.RandomState(42)
    perm_accs = []
    
    for i in range(20):
        y_perm = rng.permutation(y)
        
        # Single fold test for speed
        train_idx = np.arange(1500)  # First 1500 samples
        test_idx = np.arange(1500, len(X))  # Last 705 samples
        
        X_train, X_test = X[train_idx], X[test_idx]
        blocks_train, blocks_test = blocks[train_idx], blocks[test_idx]
        
        # Normalize
        normalizer = ComprehensiveBlockNormalizer()
        normalizer.fit(X_train, blocks=blocks_train)
        X_train_norm = normalizer.transform(X_train, blocks=blocks_train)
        X_test_norm = normalizer.transform(X_test, blocks=blocks_test)
        
        # Train and test
        clf = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=42)
        clf.fit(X_train_norm, y_perm[train_idx])
        perm_acc = accuracy_score(y_perm[test_idx], clf.predict(X_test_norm))
        perm_accs.append(perm_acc)
    
    perm_mean = np.mean(perm_accs)
    perm_std = np.std(perm_accs)
    p_value = (np.sum(np.array(perm_accs) >= baseline) + 1) / 21
    effect_size = (baseline - perm_mean) / perm_std if perm_std > 0 else 0
    
    print(f"   Baseline: {baseline:.4f}")
    print(f"   Null mean: {perm_mean:.4f} ± {perm_std:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Effect size: {effect_size:.2f}")
    
    perm_pass = p_value >= 0.05
    effect_pass = abs(effect_size) < 2.0
    
    print(f"   P ≥ 0.05: {perm_pass} {'✅' if perm_pass else '❌'}")
    print(f"   |Effect| < 2.0: {effect_pass} {'✅' if effect_pass else '❌'}")
    
    if perm_pass and effect_pass:
        print(f"\n🎉 FULL REMEDIATION SUCCESS!")
        print(f"   Ready for manuscript evidence generation")
    else:
        print(f"\n⚠️  Block remediated but permutation test needs refinement")

print(f"\n✅ Comprehensive block normalization complete!")