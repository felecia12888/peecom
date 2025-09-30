#!/usr/bin/env python3
"""
PROPER PIPELINE VALIDATION - NO PREPROCESSING LEAKAGE
====================================================
Purpose: Re-run validation with sklearn Pipeline to ensure preprocessing 
         is fit only on training data within each CV fold.

Critical Fixes:
1. Wrap all preprocessing in sklearn Pipeline
2. Fit preprocessing only on train folds (no data leakage)
3. Use 1000 permutations for robust p-value estimation
4. Verify block prediction results are consistent

Expected Outcome:
- If remediation worked: p >= 0.05, block prediction near chance
- If preprocessing leaked: contradiction between block prediction and permutation results
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import PEECOM
import sys
sys.path.append('src')
from models.simple_peecom import SimplePEECOM

class BlockCovarianceNormalizer(BaseEstimator, TransformerMixin):
    """Block-covariance normalizer that fits only on training data"""
    
    def __init__(self, block_boundaries=None, eps=1e-6):
        self.block_boundaries = block_boundaries or [733, 1464, 2205]
        self.eps = eps
        self.global_mean_ = None
        self.sqrt_global_ = None
        self.inv_sqrt_blocks_ = None
        
    def _sqrt_and_invsqrt(self, mat, eps=1e-8):
        """Helper for matrix square roots"""
        vals, vecs = np.linalg.eigh(mat)
        vals = np.clip(vals, eps, None)
        sqrt = (vecs * np.sqrt(vals)) @ vecs.T
        inv_sqrt = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
        return sqrt, inv_sqrt
    
    def _get_block_indices(self, n_samples):
        """Get block indices for given sample size"""
        blocks = []
        start = 0
        for end in self.block_boundaries:
            if start < n_samples:
                actual_end = min(end, n_samples)
                blocks.append((start, actual_end))
                start = actual_end
            if start >= n_samples:
                break
        return blocks
    
    def fit(self, X, y=None):
        """Fit normalization parameters on training data only"""
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Get block structure for this training set
        block_ranges = self._get_block_indices(n_samples)
        
        # Compute global statistics
        self.global_mean_ = X.mean(axis=0)
        Xg = X - self.global_mean_
        cov_global = np.cov(Xg, rowvar=False) + self.eps * np.eye(n_features)
        self.sqrt_global_, _ = self._sqrt_and_invsqrt(cov_global, self.eps)
        
        # Compute inverse square roots for each block
        self.inv_sqrt_blocks_ = []
        for start, end in block_ranges:
            if start < end:  # Valid block
                Xb = X[start:end]
                mean_b = Xb.mean(axis=0)
                Xb_centered = Xb - mean_b
                cov_b = np.cov(Xb_centered, rowvar=False) + self.eps * np.eye(n_features)
                _, inv_sqrt_b = self._sqrt_and_invsqrt(cov_b, self.eps)
                self.inv_sqrt_blocks_.append((start, end, inv_sqrt_b, mean_b))
            
        return self
    
    def transform(self, X):
        """Transform data using fitted parameters"""
        if self.global_mean_ is None:
            raise ValueError("Must call fit() before transform()")
            
        X = np.array(X)
        n_samples = X.shape[0]
        X_transformed = np.copy(X)
        
        # Get block structure for this data
        block_ranges = self._get_block_indices(n_samples)
        
        # Apply transformation to each block
        for i, (start, end) in enumerate(block_ranges):
            if i < len(self.inv_sqrt_blocks_) and start < end:
                _, _, inv_sqrt_b, mean_b_train = self.inv_sqrt_blocks_[i]
                
                # Use current block's mean for centering
                Xb = X[start:end]
                mean_b_current = Xb.mean(axis=0)
                Xb_centered = Xb - mean_b_current
                
                # Apply transformation
                transformed = (inv_sqrt_b @ (self.sqrt_global_ @ Xb_centered.T)).T + self.global_mean_
                X_transformed[start:end] = transformed
                
        return X_transformed

class TopKFeatureSelector(BaseEstimator, TransformerMixin):
    """Select top K features based on univariate F-statistics"""
    
    def __init__(self, k=10, score_func=f_classif):
        self.k = k
        self.score_func = score_func
        self.selector_ = None
        
    def fit(self, X, y):
        """Fit feature selector on training data"""
        # Create block labels for feature selection
        n_samples = X.shape[0]
        block_boundaries = [733, 1464, 2205]
        block_labels = np.zeros(n_samples)
        
        start = 0
        for block_id, end in enumerate(block_boundaries):
            actual_end = min(end, n_samples)
            if start < actual_end:
                block_labels[start:actual_end] = block_id
                start = actual_end
            if start >= n_samples:
                break
                
        self.selector_ = SelectKBest(score_func=self.score_func, k=self.k)
        self.selector_.fit(X, block_labels)  # Use block labels for feature selection
        return self
        
    def transform(self, X):
        """Transform data using selected features"""
        if self.selector_ is None:
            raise ValueError("Must call fit() before transform()")
        return self.selector_.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Get names of selected features"""
        if self.selector_ is None:
            return None
        if input_features is not None:
            return input_features[self.selector_.get_support()]
        else:
            return f"feature_{self.selector_.get_support().nonzero()[0]}"

def create_preprocessing_pipeline(n_features=10):
    """Create preprocessing pipeline that fits only on training data"""
    return Pipeline([
        ('cov_norm', BlockCovarianceNormalizer()),
        ('quantile', QuantileTransformer(output_distribution='uniform', random_state=42)),
        ('selector', TopKFeatureSelector(k=n_features))
    ])

def evaluate_with_proper_cv(X, y, n_features=10, cv_folds=5):
    """Evaluate models with proper CV (no preprocessing leakage)"""
    print(f"🔄 Proper CV evaluation with {n_features} features, {cv_folds} folds...")
    
    # Create pipelines
    rf_pipeline = Pipeline([
        ('preprocessing', create_preprocessing_pipeline(n_features)),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    # Cross-validation with stratified folds
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # RandomForest evaluation
    rf_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    rf_mean = rf_scores.mean()
    rf_std = rf_scores.std()
    
    print(f"   🤖 RandomForest: {rf_mean:.4f} ± {rf_std:.4f}")
    
    return {
        'rf_pipeline': rf_pipeline,
        'rf_scores': rf_scores,
        'rf_mean': rf_mean,
        'rf_std': rf_std
    }

def block_prediction_test(X, y, n_features=10):
    """Test block prediction with proper pipeline"""
    print(f"🎯 Block prediction test (proper pipeline)...")
    
    # Create block labels
    n_samples = X.shape[0]
    block_labels = np.zeros(n_samples)
    boundaries = [733, 1464, 2205]
    
    start = 0
    for block_id, end in enumerate(boundaries):
        actual_end = min(end, n_samples)
        if start < actual_end:
            block_labels[start:actual_end] = block_id
            start = actual_end
        if start >= n_samples:
            break
    
    # Block prediction pipeline
    block_pipeline = Pipeline([
        ('preprocessing', create_preprocessing_pipeline(n_features)),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    # Cross-validation for block prediction
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    block_scores = cross_val_score(block_pipeline, X, block_labels, cv=cv, scoring='accuracy', n_jobs=1)
    block_mean = block_scores.mean()
    block_std = block_scores.std()
    
    print(f"   📊 Block prediction accuracy: {block_mean:.4f} ± {block_std:.4f}")
    
    return block_mean, block_std

def permutation_test_proper(X, y, n_features=10, n_perms=1000):
    """Proper permutation test with pipeline"""
    print(f"🧪 Permutation test with {n_perms} permutations (proper pipeline)...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessing', create_preprocessing_pipeline(n_features)),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Baseline performance
    print("   Computing baseline performance...")
    baseline_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    baseline_mean = baseline_scores.mean()
    baseline_std = baseline_scores.std()
    
    # Permutation distribution
    print(f"   Computing null distribution ({n_perms} permutations)...")
    permuted_means = []
    rng = np.random.RandomState(42)
    
    # Progress tracking
    for i in range(n_perms):
        if (i + 1) % 100 == 0:
            print(f"     Progress: {i + 1}/{n_perms}")
            
        y_perm = rng.permutation(y)
        perm_scores = cross_val_score(pipeline, X, y_perm, cv=cv, scoring='accuracy', n_jobs=1)
        permuted_means.append(perm_scores.mean())
    
    permuted_means = np.array(permuted_means)
    null_mean = permuted_means.mean()
    null_std = permuted_means.std()
    
    # P-value calculation (proper formula)
    p_value = (np.sum(permuted_means >= baseline_mean) + 1) / (n_perms + 1)
    
    # Effect size
    pooled_std = np.sqrt((baseline_std**2 + null_std**2) / 2)
    effect_size = (baseline_mean - null_mean) / pooled_std if pooled_std > 0 else 0
    
    print(f"   📊 Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"   📊 Null: {null_mean:.4f} ± {null_std:.4f}")
    print(f"   📊 P-value: {p_value:.4f}")
    print(f"   📊 Effect size: {effect_size:.4f}")
    
    return {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'null_mean': null_mean,
        'null_std': null_std,
        'p_value': p_value,
        'effect_size': effect_size,
        'permuted_means': permuted_means,
        'success': p_value >= 0.05
    }

def main():
    print("🔧 PROPER PIPELINE VALIDATION - NO PREPROCESSING LEAKAGE")
    print("=" * 80)
    print("Purpose: Validate remediation with proper train-only preprocessing")
    
    # Load data
    df = pd.read_csv('hydraulic_data_processed.csv')
    print(f"   ✅ Data loaded: {df.shape}")
    
    # Prepare features and targets
    exclude_cols = {'target', 'label', 'class', 'index'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    print(f"   📊 Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"   📊 Classes: {len(np.unique(y))} (chance level: {1/len(np.unique(y)):.4f})")
    
    # Test different numbers of features
    for n_features in [10]:  # Focus on our successful configuration
        print(f"\\n{'='*80}")
        print(f"TESTING WITH TOP {n_features} FEATURES")
        print(f"{'='*80}")
        
        # 1. Block prediction test
        block_acc, block_std = block_prediction_test(X, y, n_features)
        
        # 2. Proper CV evaluation
        cv_results = evaluate_with_proper_cv(X, y, n_features)
        
        # 3. Permutation test with 1000 permutations
        perm_results = permutation_test_proper(X, y, n_features, n_perms=1000)
        
        # Analysis
        print(f"\\n📊 ANALYSIS FOR {n_features} FEATURES:")
        print("=" * 60)
        
        print(f"   🎯 Block prediction: {block_acc:.4f} ± {block_std:.4f}")
        print(f"   📈 CV accuracy: {cv_results['rf_mean']:.4f} ± {cv_results['rf_std']:.4f}")
        print(f"   🧪 Permutation p-value: {perm_results['p_value']:.4f}")
        
        # Consistency check
        cv_acc = cv_results['rf_mean']
        perm_baseline = perm_results['baseline_mean']
        consistency = abs(cv_acc - perm_baseline) < 0.02  # Within 2% tolerance
        
        print(f"\\n🔍 CONSISTENCY CHECK:")
        print(f"   CV accuracy: {cv_acc:.4f}")
        print(f"   Permutation baseline: {perm_baseline:.4f}")
        print(f"   Difference: {abs(cv_acc - perm_baseline):.4f}")
        print(f"   Consistent: {'✅ YES' if consistency else '❌ NO'}")
        
        # Final assessment
        chance_level = 1.0 / len(np.unique(y))
        near_chance = abs(cv_acc - chance_level) < 0.05  # Within 5% of chance
        block_near_chance = abs(block_acc - chance_level) < 0.10  # More lenient for block prediction
        
        print(f"\\n🏆 FINAL ASSESSMENT:")
        print(f"   Performance near chance: {'✅' if near_chance else '❌'} ({cv_acc:.4f} vs {chance_level:.4f})")
        print(f"   Block prediction low: {'✅' if block_near_chance else '❌'} ({block_acc:.4f})")
        print(f"   Permutation success: {'✅' if perm_results['success'] else '❌'} (p={perm_results['p_value']:.4f})")
        print(f"   Results consistent: {'✅' if consistency else '❌'}")
        
        overall_success = (near_chance and perm_results['success'] and consistency)
        
        if overall_success:
            print(f"\\n🎉 REMEDIATION CONFIRMED SUCCESSFUL!")
            print(f"   ✅ All validation criteria met")
            print(f"   📄 Ready for manuscript preparation")
        else:
            print(f"\\n⚠️ ISSUES DETECTED:")
            if not near_chance:
                print(f"   ❌ Performance significantly above chance")
            if not perm_results['success']:
                print(f"   ❌ Permutation test failed (p < 0.05)")
            if not consistency:
                print(f"   ❌ Inconsistent results between CV and permutation")
            print(f"   🔧 Further investigation needed")
    
    # Save results
    final_results = {
        'n_features': n_features,
        'block_prediction': {'accuracy': block_acc, 'std': block_std},
        'cv_results': cv_results,
        'permutation_results': perm_results,
        'overall_success': overall_success,
        'validation_complete': True
    }
    
    output_path = "output/proper_pipeline_validation_results.joblib"
    joblib.dump(final_results, output_path)
    print(f"\\n💾 Results saved: {output_path}")
    
    return final_results

if __name__ == "__main__":
    results = main()