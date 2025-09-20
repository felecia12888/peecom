#!/usr/bin/env python3
"""
PEECOM Efficiency Validation Suite

This script implements rigorous statistical tests to validate the claim that 
"PEECOM is more efficient than Random Forest". We use multiple approaches:

A. Statistical significance testing of performance gaps
B. Fair permutation importance comparison
C. Ablation curve analysis (informativeness per unit importance)
D. Feature-count parity testing
E. Robustness validation under stress conditions

This will provide concrete evidence for or against the efficiency claim.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import ttest_rel, wilcoxon
import joblib

# Import our models
import sys
sys.path.append('src')
from models.simple_peecom import SimplePEECOM

# Configure matplotlib
plt.rcParams.update({
    'font.size': 6,
    'axes.titlesize': 7,
    'axes.labelsize': 6,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
    'figure.titlesize': 8,
})

class PEECOMEfficiencyValidator:
    """Rigorous validation of PEECOM efficiency claims"""
    
    def __init__(self, output_dir="output/figures/validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the CMOHS dataset (where PEECOM showed advantage)
        self.load_cmohs_data()
        
        # Results storage
        self.validation_results = {}
        
    def load_cmohs_data(self):
        """Load CMOHS hydraulic system data for validation"""
        print("🔄 Loading CMOHS dataset for rigorous validation...")
        
        # Load the processed data
        X_path = "output/processed_data/cmohs/X_full.csv"
        y_path = "output/processed_data/cmohs/y_full.csv"
        
        if not Path(X_path).exists() or not Path(y_path).exists():
            print("❌ Processed CMOHS data not found!")
            print("   Please run dataset preprocessing first.")
            raise FileNotFoundError("Processed CMOHS data not available")
        
        # Load processed data
        X_full = pd.read_csv(X_path)
        y_full = pd.read_csv(y_path)
        
        print(f"✅ Loaded CMOHS data: X={X_full.shape}, y={y_full.shape}")
        
        # We'll focus on accumulator_pressure target for detailed analysis
        self.target = 'accumulator_pressure'
        if self.target not in y_full.columns:
            print(f"❌ Target {self.target} not found in y_full columns: {y_full.columns.tolist()}")
            # Use the first available target
            self.target = y_full.columns[0]
            print(f"   Using {self.target} instead")
        
        self.X = X_full
        self.y = y_full[self.target]
        
        print(f"📊 Analysis target: {self.target}")
        print(f"📊 Features: {self.X.shape[1]}, Samples: {len(self.y)}")
        print(f"📊 Class distribution: {self.y.value_counts().to_dict()}")
        
    def preprocess_cmohs(self):
        """Placeholder - not needed as data is already processed"""
        pass
        
    def test_a_statistical_significance(self):
        """Test A: Statistical significance of performance gap using k-fold CV"""
        
        print("\n" + "="*60)
        print("🧪 TEST A: STATISTICAL SIGNIFICANCE OF PERFORMANCE GAP")
        print("="*60)
        
        # Initialize models
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        peecom_model = SimplePEECOM(n_estimators=100, max_depth=10, random_state=42)
        
        # Setup k-fold cross-validation (same folds for both models)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Collect scores from each fold
        rf_scores = []
        peecom_scores = []
        
        print("🔄 Running 10-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # Train and evaluate Random Forest
            rf_clone = clone(rf_model)
            rf_clone.fit(X_train, y_train)
            rf_pred = rf_clone.predict(X_val)
            rf_score = accuracy_score(y_val, rf_pred)
            rf_scores.append(rf_score)
            
            # Train and evaluate PEECOM
            peecom_clone = clone(peecom_model)
            peecom_clone.fit(X_train, y_train)
            peecom_pred = peecom_clone.predict(X_val)
            peecom_score = accuracy_score(y_val, peecom_pred)
            peecom_scores.append(peecom_score)
            
            print(f"   Fold {fold+1:2d}: RF={rf_score:.4f}, PEECOM={peecom_score:.4f}, Diff={peecom_score-rf_score:+.4f}")
        
        # Convert to numpy arrays
        rf_scores = np.array(rf_scores)
        peecom_scores = np.array(peecom_scores)
        
        # Statistical tests
        print(f"\n📊 STATISTICAL ANALYSIS:")
        print(f"   Random Forest:  {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
        print(f"   PEECOM:         {peecom_scores.mean():.4f} ± {peecom_scores.std():.4f}")
        print(f"   Mean difference: {(peecom_scores - rf_scores).mean():+.4f}")
        
        # Paired t-test
        tstat, pval_t = ttest_rel(peecom_scores, rf_scores)
        print(f"\n🧮 PAIRED T-TEST:")
        print(f"   t-statistic: {tstat:.3f}")
        print(f"   p-value: {pval_t:.4f}")
        
        # Wilcoxon signed-rank test (non-parametric fallback)
        stat, pval_w = wilcoxon(peecom_scores, rf_scores)
        print(f"\n📈 WILCOXON SIGNED-RANK TEST:")
        print(f"   statistic: {stat:.3f}")
        print(f"   p-value: {pval_w:.4f}")
        
        # Effect size (Cohen's d for paired samples)
        diff = peecom_scores - rf_scores
        cohens_d = diff.mean() / diff.std(ddof=1)
        print(f"\n📏 EFFECT SIZE:")
        print(f"   Paired mean difference: {diff.mean():.5f}")
        print(f"   Cohen's d: {cohens_d:.3f}")
        
        # Interpretation
        print(f"\n🔍 INTERPRETATION:")
        if pval_t < 0.05:
            print(f"   ✅ SIGNIFICANT: Performance difference is statistically significant (p < 0.05)")
        else:
            print(f"   ❌ NOT SIGNIFICANT: Performance difference is not statistically significant (p ≥ 0.05)")
            
        if abs(cohens_d) < 0.2:
            effect_magnitude = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_magnitude = "small"
        elif abs(cohens_d) < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"
            
        print(f"   📊 Effect size: {effect_magnitude} ({cohens_d:.3f})")
        
        # Store results
        self.validation_results['statistical_test'] = {
            'rf_scores': rf_scores,
            'peecom_scores': peecom_scores,
            'mean_difference': diff.mean(),
            'pvalue_ttest': pval_t,
            'pvalue_wilcoxon': pval_w,
            'cohens_d': cohens_d,
            'significant': pval_t < 0.05,
            'effect_magnitude': effect_magnitude
        }
        
        return rf_scores, peecom_scores
    
    def test_b_fair_importance_comparison(self):
        """Test B: Fair permutation importance comparison"""
        
        print("\n" + "="*60)
        print("🧪 TEST B: FAIR PERMUTATION IMPORTANCE COMPARISON")
        print("="*60)
        
        # Split data for permutation importance (need held-out test set)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Train both models
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        peecom_model = SimplePEECOM(n_estimators=100, max_depth=10, random_state=42)
        
        print("🔄 Training models on training set...")
        rf_model.fit(X_train, y_train)
        peecom_model.fit(X_train, y_train)
        
        # Baseline performance
        rf_baseline = rf_model.score(X_test, y_test)
        peecom_baseline = peecom_model.score(X_test, y_test)
        
        print(f"📊 Baseline performance on test set:")
        print(f"   Random Forest: {rf_baseline:.4f}")
        print(f"   PEECOM:        {peecom_baseline:.4f}")
        
        # Permutation importance (fair comparison)
        print("🔄 Computing permutation importance (50 repeats)...")
        perm_rf = permutation_importance(rf_model, X_test, y_test, n_repeats=50, random_state=42, n_jobs=-1)
        perm_peecom = permutation_importance(peecom_model, X_test, y_test, n_repeats=50, random_state=42, n_jobs=-1)
        
        # Get mean importances
        importances_rf = perm_rf.importances_mean
        importances_peecom = perm_peecom.importances_mean
        
        # Normalize by sum for comparison
        norm_rf = importances_rf / importances_rf.sum()
        norm_peecom = importances_peecom / importances_peecom.sum()
        
        print(f"\n📊 PERMUTATION IMPORTANCE ANALYSIS:")
        print(f"   Random Forest  - Mean: {importances_rf.mean():.6f}, Max: {importances_rf.max():.6f}")
        print(f"   PEECOM         - Mean: {importances_peecom.mean():.6f}, Max: {importances_peecom.max():.6f}")
        
        # Count significant features
        threshold = 0.001
        rf_significant = (importances_rf > threshold).sum()
        peecom_significant = (importances_peecom > threshold).sum()
        
        print(f"   Significant features (>{threshold}):")
        print(f"   Random Forest: {rf_significant}")
        print(f"   PEECOM:        {peecom_significant}")
        
        # Efficiency metric: performance per unit average importance
        rf_efficiency = rf_baseline / importances_rf.mean()
        peecom_efficiency = peecom_baseline / importances_peecom.mean()
        
        print(f"\n🎯 EFFICIENCY METRICS:")
        print(f"   Random Forest efficiency: {rf_efficiency:.2f} (performance/avg_importance)")
        print(f"   PEECOM efficiency:        {peecom_efficiency:.2f}")
        print(f"   PEECOM advantage:         {peecom_efficiency/rf_efficiency:.2f}x")
        
        # Store results
        self.validation_results['permutation_importance'] = {
            'rf_importance': importances_rf,
            'peecom_importance': importances_peecom,
            'rf_efficiency': rf_efficiency,
            'peecom_efficiency': peecom_efficiency,
            'efficiency_ratio': peecom_efficiency/rf_efficiency,
            'feature_names': X_test.columns.tolist()
        }
        
        return importances_rf, importances_peecom, X_test.columns
    
    def test_c_ablation_curve_analysis(self):
        """Test C: Ablation curve analysis for informativeness per unit importance"""
        
        print("\n" + "="*60)
        print("🧪 TEST C: ABLATION CURVE ANALYSIS")
        print("="*60)
        
        # Use results from permutation importance test
        if 'permutation_importance' not in self.validation_results:
            print("⚠️ Running permutation importance test first...")
            self.test_b_fair_importance_comparison()
        
        # Get importance data
        importances_rf = self.validation_results['permutation_importance']['rf_importance']
        importances_peecom = self.validation_results['permutation_importance']['peecom_importance']
        feature_names = self.validation_results['permutation_importance']['feature_names']
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        def ablation_curve(model_class, importances, model_name):
            """Compute ablation curve for a model"""
            print(f"🔄 Computing ablation curve for {model_name}...")
            
            # Sort features by importance (descending)
            feature_importance_pairs = list(zip(feature_names, importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Initialize
            cumulative_importance = []
            performance_scores = []
            current_features = feature_names.copy()
            total_importance = importances.sum()
            
            # Baseline (all features)
            model = model_class(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            baseline_score = model.score(X_test, y_test)
            
            cumulative_importance.append(0.0)
            performance_scores.append(baseline_score)
            
            # Remove features one by one (most important first)
            cum_imp = 0.0
            for feature_name, importance in feature_importance_pairs:
                if feature_name in current_features:
                    current_features.remove(feature_name)
                    cum_imp += importance
                    
                    # Train model without this feature
                    if len(current_features) > 5:  # Keep minimum features
                        model = model_class(n_estimators=100, max_depth=10, random_state=42)
                        X_train_subset = X_train[current_features]
                        X_test_subset = X_test[current_features]
                        model.fit(X_train_subset, y_train)
                        score = model.score(X_test_subset, y_test)
                        
                        cumulative_importance.append(cum_imp / total_importance)
                        performance_scores.append(score)
                    
                    if len(cumulative_importance) >= 20:  # Limit for visualization
                        break
            
            return np.array(cumulative_importance), np.array(performance_scores)
        
        # Compute ablation curves
        rf_cum_imp, rf_perf = ablation_curve(RandomForestClassifier, importances_rf, "Random Forest")
        peecom_cum_imp, peecom_perf = ablation_curve(SimplePEECOM, importances_peecom, "PEECOM")
        
        # Compute area under curves (higher = more efficient)
        from numpy import trapz
        rf_auc = trapz(rf_perf, rf_cum_imp)
        peecom_auc = trapz(peecom_perf, peecom_cum_imp)
        
        print(f"\n📊 ABLATION CURVE ANALYSIS:")
        print(f"   Random Forest AUC: {rf_auc:.4f}")
        print(f"   PEECOM AUC:        {peecom_auc:.4f}")
        print(f"   PEECOM advantage:  {peecom_auc/rf_auc:.3f}x")
        
        if peecom_auc > rf_auc:
            print(f"   ✅ PEECOM retains higher performance when removing important features")
            print(f"   ✅ Evidence supports 'more informative per unit importance'")
        else:
            print(f"   ❌ Random Forest retains higher performance when removing important features")
            print(f"   ❌ Evidence does NOT support PEECOM efficiency claim")
        
        # Store results
        self.validation_results['ablation_analysis'] = {
            'rf_cumulative_importance': rf_cum_imp,
            'rf_performance': rf_perf,
            'peecom_cumulative_importance': peecom_cum_imp,
            'peecom_performance': peecom_perf,
            'rf_auc': rf_auc,
            'peecom_auc': peecom_auc,
            'efficiency_ratio': peecom_auc/rf_auc
        }
        
        return rf_cum_imp, rf_perf, peecom_cum_imp, peecom_perf
    
    def test_d_feature_count_parity(self):
        """Test D: Feature-count parity test"""
        
        print("\n" + "="*60)
        print("🧪 TEST D: FEATURE-COUNT PARITY TEST")
        print("="*60)
        
        # Use permutation importance results
        if 'permutation_importance' not in self.validation_results:
            self.test_b_fair_importance_comparison()
        
        importances_peecom = self.validation_results['permutation_importance']['peecom_importance']
        feature_names = self.validation_results['permutation_importance']['feature_names']
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Sort features by PEECOM importance
        feature_importance_pairs = list(zip(feature_names, importances_peecom))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [pair[0] for pair in feature_importance_pairs]
        
        print("🔄 Testing performance vs feature count...")
        
        # Test different feature counts
        feature_counts = [5, 10, 15, 20, 25, 30, 40, 50, min(len(sorted_features), 60)]
        rf_scores = []
        peecom_scores = []
        
        for k in feature_counts:
            if k > len(sorted_features):
                continue
                
            selected_features = sorted_features[:k]
            
            # Train Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf_model.fit(X_train[selected_features], y_train)
            rf_score = rf_model.score(X_val[selected_features], y_val)
            rf_scores.append(rf_score)
            
            # Train PEECOM
            peecom_model = SimplePEECOM(n_estimators=100, max_depth=10, random_state=42)
            peecom_model.fit(X_train[selected_features], y_train)
            peecom_score = peecom_model.score(X_val[selected_features], y_val)
            peecom_scores.append(peecom_score)
            
            print(f"   Top {k:2d} features: RF={rf_score:.4f}, PEECOM={peecom_score:.4f}, Diff={peecom_score-rf_score:+.4f}")
        
        # Analyze results
        rf_scores = np.array(rf_scores)
        peecom_scores = np.array(peecom_scores)
        differences = peecom_scores - rf_scores
        
        print(f"\n📊 FEATURE-COUNT PARITY ANALYSIS:")
        print(f"   Average PEECOM advantage: {differences.mean():+.4f}")
        print(f"   PEECOM wins in {(differences > 0).sum()}/{len(differences)} cases")
        
        # Find minimum features needed for 95% of best performance
        rf_best = rf_scores.max()
        peecom_best = peecom_scores.max()
        
        rf_95_threshold = rf_best * 0.95
        peecom_95_threshold = peecom_best * 0.95
        
        rf_min_features = feature_counts[np.where(rf_scores >= rf_95_threshold)[0][0]] if np.any(rf_scores >= rf_95_threshold) else feature_counts[-1]
        peecom_min_features = feature_counts[np.where(peecom_scores >= peecom_95_threshold)[0][0]] if np.any(peecom_scores >= peecom_95_threshold) else feature_counts[-1]
        
        print(f"\n🎯 EFFICIENCY ANALYSIS:")
        print(f"   Features needed for 95% of best performance:")
        print(f"   Random Forest: {rf_min_features} features")
        print(f"   PEECOM:        {peecom_min_features} features")
        
        if peecom_min_features < rf_min_features:
            print(f"   ✅ PEECOM reaches top performance with fewer features")
        else:
            print(f"   ❌ PEECOM does NOT reach top performance with fewer features")
        
        # Store results
        self.validation_results['feature_parity'] = {
            'feature_counts': feature_counts[:len(rf_scores)],
            'rf_scores': rf_scores,
            'peecom_scores': peecom_scores,
            'differences': differences,
            'rf_min_features': rf_min_features,
            'peecom_min_features': peecom_min_features
        }
        
        return feature_counts[:len(rf_scores)], rf_scores, peecom_scores
    
    def test_e_robustness_validation(self):
        """Test E: Robustness validation under stress conditions"""
        
        print("\n" + "="*60)
        print("🧪 TEST E: ROBUSTNESS VALIDATION")
        print("="*60)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Train baseline models
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        peecom_model = SimplePEECOM(n_estimators=100, max_depth=10, random_state=42)
        
        rf_model.fit(X_train, y_train)
        peecom_model.fit(X_train, y_train)
        
        # Baseline performance
        rf_baseline = rf_model.score(X_test, y_test)
        peecom_baseline = peecom_model.score(X_test, y_test)
        
        print(f"📊 Baseline performance:")
        print(f"   Random Forest: {rf_baseline:.4f}")
        print(f"   PEECOM:        {peecom_baseline:.4f}")
        
        robustness_results = {}
        
        # Test 1: Gaussian noise
        print(f"\n🔄 Test 1: Gaussian noise robustness...")
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
        rf_noise_scores = []
        peecom_noise_scores = []
        
        for noise_std in noise_levels:
            # Add Gaussian noise
            X_test_noisy = X_test + np.random.normal(0, noise_std, X_test.shape)
            
            rf_score = rf_model.score(X_test_noisy, y_test)
            peecom_score = peecom_model.score(X_test_noisy, y_test)
            
            rf_noise_scores.append(rf_score)
            peecom_noise_scores.append(peecom_score)
            
            print(f"   Noise std={noise_std:.2f}: RF={rf_score:.4f}, PEECOM={peecom_score:.4f}")
        
        robustness_results['noise'] = {
            'noise_levels': noise_levels,
            'rf_scores': rf_noise_scores,
            'peecom_scores': peecom_noise_scores
        }
        
        # Test 2: Random feature dropout
        print(f"\n🔄 Test 2: Random feature dropout robustness...")
        dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        rf_dropout_scores = []
        peecom_dropout_scores = []
        
        for dropout_rate in dropout_rates:
            # Randomly drop features
            n_features = X_test.shape[1]
            n_keep = int(n_features * (1 - dropout_rate))
            keep_features = np.random.choice(n_features, n_keep, replace=False)
            
            X_test_dropout = X_test.iloc[:, keep_features]
            X_train_dropout = X_train.iloc[:, keep_features]
            
            # Retrain models with dropped features
            rf_dropout = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            peecom_dropout = SimplePEECOM(n_estimators=100, max_depth=10, random_state=42)
            
            rf_dropout.fit(X_train_dropout, y_train)
            peecom_dropout.fit(X_train_dropout, y_train)
            
            rf_score = rf_dropout.score(X_test_dropout, y_test)
            peecom_score = peecom_dropout.score(X_test_dropout, y_test)
            
            rf_dropout_scores.append(rf_score)
            peecom_dropout_scores.append(peecom_score)
            
            print(f"   Dropout={dropout_rate:.1f}: RF={rf_score:.4f}, PEECOM={peecom_score:.4f}")
        
        robustness_results['dropout'] = {
            'dropout_rates': dropout_rates,
            'rf_scores': rf_dropout_scores,
            'peecom_scores': peecom_dropout_scores
        }
        
        # Analyze robustness
        print(f"\n📊 ROBUSTNESS ANALYSIS:")
        
        # Noise robustness
        rf_noise_degradation = (rf_baseline - min(rf_noise_scores)) / rf_baseline
        peecom_noise_degradation = (peecom_baseline - min(peecom_noise_scores)) / peecom_baseline
        
        print(f"   Noise degradation:")
        print(f"   Random Forest: {rf_noise_degradation:.3f}")
        print(f"   PEECOM:        {peecom_noise_degradation:.3f}")
        
        # Dropout robustness
        rf_dropout_degradation = (rf_baseline - min(rf_dropout_scores)) / rf_baseline
        peecom_dropout_degradation = (peecom_baseline - min(peecom_dropout_scores)) / peecom_baseline
        
        print(f"   Dropout degradation:")
        print(f"   Random Forest: {rf_dropout_degradation:.3f}")
        print(f"   PEECOM:        {peecom_dropout_degradation:.3f}")
        
        # Overall robustness assessment
        if peecom_noise_degradation < rf_noise_degradation and peecom_dropout_degradation < rf_dropout_degradation:
            print(f"   ✅ PEECOM is more robust under stress conditions")
        else:
            print(f"   ❌ PEECOM is NOT more robust under stress conditions")
        
        self.validation_results['robustness'] = robustness_results
        
        return robustness_results
    
    def create_validation_visualization(self):
        """Create comprehensive validation visualization"""
        
        print("\n" + "="*60)
        print("📊 CREATING VALIDATION VISUALIZATION")
        print("="*60)
        
        fig = plt.figure(figsize=(8.27, 11.7))  # A4 size
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
        
        colors = {'rf': '#ff7f0e', 'peecom': '#1f77b4'}
        
        # 1. Statistical significance (CV scores)
        if 'statistical_test' in self.validation_results:
            ax1 = fig.add_subplot(gs[0, 0])
            results = self.validation_results['statistical_test']
            
            ax1.boxplot([results['rf_scores'], results['peecom_scores']], 
                       labels=['Random Forest', 'PEECOM'])
            ax1.set_title(f'K-Fold CV Scores\np-value: {results["pvalue_ttest"]:.4f}', fontsize=6)
            ax1.set_ylabel('Accuracy', fontsize=5)
            ax1.grid(True, alpha=0.3)
        
        # 2. Ablation curves
        if 'ablation_analysis' in self.validation_results:
            ax2 = fig.add_subplot(gs[0, 1])
            results = self.validation_results['ablation_analysis']
            
            ax2.plot(results['rf_cumulative_importance'], results['rf_performance'], 
                    'o-', color=colors['rf'], label=f'RF (AUC: {results["rf_auc"]:.3f})', markersize=3)
            ax2.plot(results['peecom_cumulative_importance'], results['peecom_performance'], 
                    's-', color=colors['peecom'], label=f'PEECOM (AUC: {results["peecom_auc"]:.3f})', markersize=3)
            
            ax2.set_xlabel('Cumulative Importance Removed', fontsize=5)
            ax2.set_ylabel('Performance', fontsize=5)
            ax2.set_title('Ablation Curves\n(Higher AUC = More Efficient)', fontsize=6)
            ax2.legend(fontsize=4)
            ax2.grid(True, alpha=0.3)
        
        # 3. Feature count parity
        if 'feature_parity' in self.validation_results:
            ax3 = fig.add_subplot(gs[1, 0])
            results = self.validation_results['feature_parity']
            
            ax3.plot(results['feature_counts'], results['rf_scores'], 
                    'o-', color=colors['rf'], label='Random Forest', markersize=3)
            ax3.plot(results['feature_counts'], results['peecom_scores'], 
                    's-', color=colors['peecom'], label='PEECOM', markersize=3)
            
            ax3.set_xlabel('Number of Features', fontsize=5)
            ax3.set_ylabel('Accuracy', fontsize=5)
            ax3.set_title('Performance vs Feature Count', fontsize=6)
            ax3.legend(fontsize=4)
            ax3.grid(True, alpha=0.3)
        
        # 4. Robustness - Noise
        if 'robustness' in self.validation_results:
            ax4 = fig.add_subplot(gs[1, 1])
            results = self.validation_results['robustness']['noise']
            
            ax4.plot(results['noise_levels'], results['rf_scores'], 
                    'o-', color=colors['rf'], label='Random Forest', markersize=3)
            ax4.plot(results['noise_levels'], results['peecom_scores'], 
                    's-', color=colors['peecom'], label='PEECOM', markersize=3)
            
            ax4.set_xlabel('Noise Level (std)', fontsize=5)
            ax4.set_ylabel('Accuracy', fontsize=5)
            ax4.set_title('Robustness to Gaussian Noise', fontsize=6)
            ax4.legend(fontsize=4)
            ax4.grid(True, alpha=0.3)
        
        # 5. Robustness - Dropout
        if 'robustness' in self.validation_results:
            ax5 = fig.add_subplot(gs[2, 0])
            results = self.validation_results['robustness']['dropout']
            
            ax5.plot(results['dropout_rates'], results['rf_scores'], 
                    'o-', color=colors['rf'], label='Random Forest', markersize=3)
            ax5.plot(results['dropout_rates'], results['peecom_scores'], 
                    's-', color=colors['peecom'], label='PEECOM', markersize=3)
            
            ax5.set_xlabel('Feature Dropout Rate', fontsize=5)
            ax5.set_ylabel('Accuracy', fontsize=5)
            ax5.set_title('Robustness to Feature Dropout', fontsize=6)
            ax5.legend(fontsize=4)
            ax5.grid(True, alpha=0.3)
        
        # 6. Summary panel
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        # Create summary text
        summary_text = "VALIDATION SUMMARY\n\n"
        
        if 'statistical_test' in self.validation_results:
            st = self.validation_results['statistical_test']
            summary_text += f"Statistical Test:\n"
            summary_text += f"{'✅' if st['significant'] else '❌'} p-value: {st['pvalue_ttest']:.4f}\n"
            summary_text += f"Effect size: {st['effect_magnitude']}\n\n"
        
        if 'ablation_analysis' in self.validation_results:
            ab = self.validation_results['ablation_analysis']
            summary_text += f"Ablation Analysis:\n"
            summary_text += f"{'✅' if ab['efficiency_ratio'] > 1 else '❌'} Efficiency ratio: {ab['efficiency_ratio']:.3f}\n\n"
        
        if 'feature_parity' in self.validation_results:
            fp = self.validation_results['feature_parity']
            peecom_wins = (fp['differences'] > 0).sum()
            total_tests = len(fp['differences'])
            summary_text += f"Feature Parity:\n"
            summary_text += f"{'✅' if peecom_wins > total_tests/2 else '❌'} PEECOM wins: {peecom_wins}/{total_tests}\n\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=5,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.3))
        
        # Overall title
        fig.suptitle('PEECOM Efficiency Validation: Rigorous Statistical Analysis\n' + 
                    'Comprehensive Testing of Performance and Efficiency Claims', 
                    fontsize=8, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save
        output_path = self.output_dir / "peecom_validation_comprehensive.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        
        plt.show()
        print(f"✅ Validation visualization saved: {output_path}")
    
    def run_complete_validation(self):
        """Run all validation tests"""
        
        print("🚀 PEECOM EFFICIENCY VALIDATION SUITE")
        print("="*60)
        print("Running rigorous statistical tests to validate efficiency claims...")
        
        # Run all tests
        self.test_a_statistical_significance()
        self.test_b_fair_importance_comparison()
        self.test_c_ablation_curve_analysis()
        self.test_d_feature_count_parity()
        self.test_e_robustness_validation()
        
        # Create visualization
        self.create_validation_visualization()
        
        # Generate final verdict
        self.generate_final_verdict()
        
    def generate_final_verdict(self):
        """Generate final verdict on PEECOM efficiency claims"""
        
        print("\n" + "="*80)
        print("🏛️ FINAL VERDICT: PEECOM EFFICIENCY VALIDATION")
        print("="*80)
        
        verdicts = []
        
        # Test A: Statistical significance
        if 'statistical_test' in self.validation_results:
            st = self.validation_results['statistical_test']
            if st['significant'] and st['cohens_d'] > 0:
                verdicts.append(('Statistical Significance', True, f"p={st['pvalue_ttest']:.4f}, d={st['cohens_d']:.3f}"))
            else:
                verdicts.append(('Statistical Significance', False, f"p={st['pvalue_ttest']:.4f}, d={st['cohens_d']:.3f}"))
        
        # Test B: Fair importance comparison
        if 'permutation_importance' in self.validation_results:
            pi = self.validation_results['permutation_importance']
            if pi['efficiency_ratio'] > 1.0:
                verdicts.append(('Permutation Importance', True, f"Efficiency ratio: {pi['efficiency_ratio']:.3f}x"))
            else:
                verdicts.append(('Permutation Importance', False, f"Efficiency ratio: {pi['efficiency_ratio']:.3f}x"))
        
        # Test C: Ablation analysis
        if 'ablation_analysis' in self.validation_results:
            ab = self.validation_results['ablation_analysis']
            if ab['efficiency_ratio'] > 1.0:
                verdicts.append(('Ablation Analysis', True, f"AUC ratio: {ab['efficiency_ratio']:.3f}x"))
            else:
                verdicts.append(('Ablation Analysis', False, f"AUC ratio: {ab['efficiency_ratio']:.3f}x"))
        
        # Test D: Feature parity
        if 'feature_parity' in self.validation_results:
            fp = self.validation_results['feature_parity']
            peecom_wins = (fp['differences'] > 0).sum()
            total_tests = len(fp['differences'])
            if peecom_wins > total_tests / 2:
                verdicts.append(('Feature Count Parity', True, f"Wins: {peecom_wins}/{total_tests}"))
            else:
                verdicts.append(('Feature Count Parity', False, f"Wins: {peecom_wins}/{total_tests}"))
        
        # Print verdicts
        passed_tests = 0
        total_tests = len(verdicts)
        
        for test_name, passed, details in verdicts:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status:8} {test_name:25} {details}")
            if passed:
                passed_tests += 1
        
        # Final conclusion
        print(f"\n📊 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests >= total_tests * 0.75:  # 75% threshold
            print("🏆 VERDICT: PEECOM efficiency claims are VALIDATED")
            print("   Evidence supports that PEECOM is more efficient than Random Forest")
        elif passed_tests >= total_tests * 0.5:  # 50% threshold
            print("⚖️ VERDICT: PEECOM efficiency claims are PARTIALLY SUPPORTED")
            print("   Mixed evidence - some tests support efficiency claims")
        else:
            print("❌ VERDICT: PEECOM efficiency claims are NOT SUPPORTED")
            print("   Insufficient evidence to support efficiency claims")
        
        # Save results summary
        summary_path = self.output_dir / "validation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("PEECOM Efficiency Validation Summary\n")
            f.write("="*50 + "\n\n")
            
            for test_name, passed, details in verdicts:
                status = "PASS" if passed else "FAIL"
                f.write(f"{status:4} {test_name:25} {details}\n")
            
            f.write(f"\nOverall: {passed_tests}/{total_tests} tests passed\n")
        
        print(f"\n✅ Validation complete. Results saved to: {self.output_dir}")


if __name__ == "__main__":
    validator = PEECOMEfficiencyValidator()
    validator.run_complete_validation()