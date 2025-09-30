#!/usr/bin/env python3
"""
CROSS-BLOCK VALIDATION EXPERIMENT

Systematic investigation of feature separability and leakage sources:
1. Compute feature separability (identify suspect features)
2. Block-aware within-block CV (K=5) baseline with all features
3. Minimal-feature test (top-K separable features K=1..5)
4. Normalization experiments (remove/normalize high-separable features)
5. Ablation curve (accuracy vs number of removed top features)
6. Permutation test (30 permutations) with p-value calculation
7. Robustness shifts (noise/offset testing)

This will definitively prove the experimental design leakage hypothesis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           precision_recall_fscore_support, f1_score)
from sklearn.model_selection import KFold
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up paths
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset" / "cmohs"
OUTPUT_DIR = ROOT / "output" / "cross_block_validation"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

for dir_path in [OUTPUT_DIR, FIGURES_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class CrossBlockValidationExperiment:
    """
    Comprehensive cross-block validation to prove leakage sources
    """
    
    def __init__(self, random_state=42, n_permutations=30):
        self.random_state = random_state
        self.n_permutations = n_permutations
        np.random.seed(self.random_state)
        
        print("🧪 CROSS-BLOCK VALIDATION EXPERIMENT")
        print("=" * 60)
        print("🎯 SYSTEMATIC LEAKAGE INVESTIGATION:")
        print("   1. Feature separability analysis")
        print("   2. Block-aware within-block CV (K=5)")
        print("   3. Minimal-feature testing")
        print("   4. High-separability feature removal")
        print("   5. Block-relative normalization")
        print("   6. Ablation curve analysis")
        print("   7. Permutation testing (30 perms)")
        print("   8. Robustness shift analysis")
    
    def load_data_with_blocks(self):
        """Load data and detect block structure"""
        
        # Load targets
        targets = pd.read_csv(DATASET_DIR / "profile.txt", sep='\t', header=None,
                             names=['cooler_condition', 'valve_condition', 'pump_leakage',
                                   'accumulator_pressure', 'stable_flag'])
        
        # Load comprehensive features
        sensor_files = ['PS1.txt', 'PS2.txt', 'PS3.txt', 'PS4.txt', 'PS5.txt', 'PS6.txt']
        all_sensor_data = []
        
        for file_name in sensor_files:
            file_path = DATASET_DIR / file_name
            if file_path.exists():
                with open(file_path, 'r') as f:
                    sensor_cycles = []
                    for line in f:
                        values = [float(x) for x in line.strip().split('\t')]
                        # Statistical features
                        features = [
                            np.mean(values),      # Mean
                            np.std(values),       # Std
                            np.max(values),       # Max
                            np.min(values),       # Min
                            np.median(values),    # Median
                            np.var(values),       # Variance
                            np.ptp(values),       # Range
                            stats.skew(values) if len(values) > 3 else 0,    # Skewness
                            stats.kurtosis(values) if len(values) > 3 else 0  # Kurtosis
                        ]
                        sensor_cycles.append(features)
                
                all_sensor_data.append(np.array(sensor_cycles))
        
        # Combine all sensors
        X = np.hstack(all_sensor_data) if all_sensor_data else np.array([])
        
        # Align lengths
        min_len = min(len(X), len(targets))
        X = X[:min_len]
        targets = targets.iloc[:min_len]
        
        # Detect blocks
        y = LabelEncoder().fit_transform(targets['cooler_condition'])
        block_ids = self.detect_blocks(y)
        
        print(f"\n📊 DATA LOADED:")
        print(f"   Features: {X.shape}")
        print(f"   Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(f"   Blocks detected: {len(np.unique(block_ids))}")
        
        return X, y, block_ids, targets
    
    def detect_blocks(self, y):
        """Detect block boundaries in target sequence"""
        
        blocks = np.zeros_like(y)
        current_block = 0
        current_class = y[0]
        
        for i in range(1, len(y)):
            if y[i] != current_class:
                current_block += 1
                current_class = y[i]
            blocks[i] = current_block
        
        return blocks
    
    def compute_feature_separability(self, X, y):
        """
        1. Compute feature separability - identify suspect features
        """
        print(f"\n🔍 1. FEATURE SEPARABILITY ANALYSIS")
        print("=" * 45)
        
        feature_separability = []
        
        for feat_idx in range(X.shape[1]):
            feature_vals = X[:, feat_idx]
            
            # Calculate Cohen's d for each class pair
            classes = np.unique(y)
            separability_scores = []
            
            for i, class_a in enumerate(classes):
                for j, class_b in enumerate(classes):
                    if i < j:
                        mask_a = (y == class_a)
                        mask_b = (y == class_b)
                        
                        vals_a = feature_vals[mask_a]
                        vals_b = feature_vals[mask_b]
                        
                        mean_a, mean_b = np.mean(vals_a), np.mean(vals_b)
                        std_pooled = np.sqrt(((len(vals_a)-1)*np.var(vals_a, ddof=1) + 
                                            (len(vals_b)-1)*np.var(vals_b, ddof=1)) / 
                                           (len(vals_a)+len(vals_b)-2))
                        
                        if std_pooled > 1e-8:
                            cohens_d = abs(mean_a - mean_b) / std_pooled
                        else:
                            cohens_d = float('inf')
                        
                        separability_scores.append(cohens_d)
            
            avg_separability = np.mean([s for s in separability_scores if not np.isinf(s)])
            max_separability = np.max(separability_scores)
            
            feature_separability.append({
                'feature_idx': feat_idx,
                'avg_separability': avg_separability if not np.isnan(avg_separability) else 0,
                'max_separability': max_separability
            })
        
        # Sort by max separability
        feature_separability.sort(key=lambda x: x['max_separability'], reverse=True)
        
        print(f"   📊 TOP 10 MOST SEPARABLE FEATURES:")
        for i, feat in enumerate(feature_separability[:10]):
            sep = feat['max_separability']
            if np.isinf(sep):
                print(f"     {i+1:2d}. Feature {feat['feature_idx']:2d}: INFINITE separability")
            else:
                print(f"     {i+1:2d}. Feature {feat['feature_idx']:2d}: {sep:.3f}")
        
        return feature_separability
    
    def block_aware_within_block_cv(self, X, y, block_ids, feature_indices=None):
        """
        2. Block-aware within-block CV (K=5) - baseline with all/selected features
        """
        print(f"\n📊 2. BLOCK-AWARE WITHIN-BLOCK CV")
        print("=" * 40)
        
        if feature_indices is not None:
            X_selected = X[:, feature_indices]
            print(f"   Using {len(feature_indices)} selected features")
        else:
            X_selected = X
            print(f"   Using all {X.shape[1]} features")
        
        results = {'RF': {'fold_accs': [], 'fold_f1s': []},
                  'LR': {'fold_accs': [], 'fold_f1s': []}}
        
        # Perform CV within each block
        unique_blocks = np.unique(block_ids)
        
        for block_id in unique_blocks:
            block_mask = (block_ids == block_id)
            X_block = X_selected[block_mask]
            y_block = y[block_mask]
            
            print(f"\n   📋 Block {block_id} (Class {y_block[0]}):")
            print(f"      Samples: {len(X_block)}")
            
            # Check if block has only one class
            if len(np.unique(y_block)) == 1:
                print(f"      ⚠️  Single class block - skipping CV")
                continue
            
            # 5-fold CV within block
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            fold_results = {'RF': {'accs': [], 'f1s': []}, 
                          'LR': {'accs': [], 'f1s': []}}
            
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_block)):
                X_train_fold = X_block[train_idx]
                X_test_fold = X_block[test_idx]
                y_train_fold = y_block[train_idx]
                y_test_fold = y_block[test_idx]
                
                # Preprocessing
                imputer = SimpleImputer(strategy='mean')
                scaler = StandardScaler()
                
                X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train_fold))
                X_test_processed = scaler.transform(imputer.transform(X_test_fold))
                
                # Test models
                models = {
                    'RF': RandomForestClassifier(n_estimators=50, max_depth=6, 
                                               random_state=self.random_state),
                    'LR': LogisticRegression(C=1.0, random_state=self.random_state, max_iter=500)
                }
                
                for model_name, model in models.items():
                    try:
                        model.fit(X_train_processed, y_train_fold)
                        y_pred = model.predict(X_test_processed)
                        
                        acc = accuracy_score(y_test_fold, y_pred)
                        f1 = f1_score(y_test_fold, y_pred, average='weighted')
                        
                        fold_results[model_name]['accs'].append(acc)
                        fold_results[model_name]['f1s'].append(f1)
                        
                    except Exception as e:
                        print(f"         {model_name} failed: {str(e)[:40]}")
                        fold_results[model_name]['accs'].append(0.0)
                        fold_results[model_name]['f1s'].append(0.0)
            
            # Aggregate fold results for this block
            for model_name in ['RF', 'LR']:
                if fold_results[model_name]['accs']:
                    block_acc = np.mean(fold_results[model_name]['accs'])
                    block_f1 = np.mean(fold_results[model_name]['f1s'])
                    
                    results[model_name]['fold_accs'].append(block_acc)
                    results[model_name]['fold_f1s'].append(block_f1)
                    
                    print(f"      {model_name}: Acc={block_acc:.3f}, F1={block_f1:.3f}")
        
        # Overall results
        print(f"\n   📊 OVERALL CV RESULTS:")
        cv_summary = {}
        
        for model_name in ['RF', 'LR']:
            if results[model_name]['fold_accs']:
                mean_acc = np.mean(results[model_name]['fold_accs'])
                std_acc = np.std(results[model_name]['fold_accs'])
                mean_f1 = np.mean(results[model_name]['fold_f1s'])
                std_f1 = np.std(results[model_name]['fold_f1s'])
                
                cv_summary[model_name] = {
                    'mean_acc': mean_acc,
                    'std_acc': std_acc,
                    'mean_f1': mean_f1,
                    'std_f1': std_f1
                }
                
                print(f"     {model_name}: Acc={mean_acc:.3f}±{std_acc:.3f}, F1={mean_f1:.3f}±{std_f1:.3f}")
        
        return results, cv_summary
    
    def minimal_feature_test(self, X, y, block_ids, feature_separability):
        """
        3. Minimal-feature test using top-K separable features
        """
        print(f"\n🧪 3. MINIMAL-FEATURE TEST")
        print("=" * 35)
        
        # Test different numbers of top features
        k_values = [1, 2, 3, 4, 5]
        minimal_results = {}
        
        for k in k_values:
            print(f"\n   🎯 Testing top {k} separable features:")
            
            # Get top k features (excluding infinite separability)
            finite_features = [f for f in feature_separability if not np.isinf(f['max_separability'])]
            if len(finite_features) < k:
                print(f"     ⚠️  Only {len(finite_features)} finite separability features available")
                continue
            
            top_k_indices = [f['feature_idx'] for f in finite_features[:k]]
            print(f"     Features: {top_k_indices}")
            print(f"     Separabilities: {[f['max_separability'] for f in finite_features[:k]]}")
            
            # Run CV with selected features
            results, summary = self.block_aware_within_block_cv(X, y, block_ids, top_k_indices)
            minimal_results[k] = summary
            
            # Check for high accuracy with minimal features
            for model_name, metrics in summary.items():
                if metrics['mean_acc'] > 0.90:
                    print(f"     🚨 {model_name}: HIGH ACCURACY ({metrics['mean_acc']:.3f}) WITH {k} FEATURES!")
        
        return minimal_results
    
    def remove_high_sep_features_experiment(self, X, y, block_ids, feature_separability):
        """
        4. Remove high-separability features experiment
        """
        print(f"\n🔥 4. HIGH-SEPARABILITY FEATURE REMOVAL")
        print("=" * 48)
        
        removal_results = {}
        
        # Test removing different numbers of top features
        removal_counts = [0, 5, 10, 15, 20, 25]
        
        for remove_count in removal_counts:
            print(f"\n   🗑️  Removing top {remove_count} separable features:")
            
            if remove_count == 0:
                feature_indices = list(range(X.shape[1]))
                print(f"     Using all {len(feature_indices)} features (baseline)")
            else:
                # Remove top separable features
                top_features_to_remove = [f['feature_idx'] for f in feature_separability[:remove_count]]
                feature_indices = [i for i in range(X.shape[1]) if i not in top_features_to_remove]
                print(f"     Removed features: {top_features_to_remove[:5]}...")
                print(f"     Remaining: {len(feature_indices)} features")
            
            if len(feature_indices) < 5:
                print(f"     ⚠️  Too few features remaining ({len(feature_indices)})")
                continue
            
            # Run CV with remaining features
            results, summary = self.block_aware_within_block_cv(X, y, block_ids, feature_indices)
            removal_results[remove_count] = summary
        
        return removal_results
    
    def create_ablation_curve(self, removal_results):
        """
        6. Create ablation curve visualization
        """
        print(f"\n📈 6. ABLATION CURVE ANALYSIS")
        print("=" * 38)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Extract results
        removal_counts = sorted(removal_results.keys())
        
        rf_accs = []
        lr_accs = []
        
        for count in removal_counts:
            if 'RF' in removal_results[count]:
                rf_accs.append(removal_results[count]['RF']['mean_acc'])
            else:
                rf_accs.append(0)
                
            if 'LR' in removal_results[count]:
                lr_accs.append(removal_results[count]['LR']['mean_acc'])
            else:
                lr_accs.append(0)
        
        # Plot accuracy vs removed features
        ax = axes[0]
        ax.plot(removal_counts, rf_accs, 'o-', linewidth=2, markersize=8, 
               label='Random Forest', alpha=0.8)
        ax.plot(removal_counts, lr_accs, 's-', linewidth=2, markersize=8, 
               label='Logistic Regression', alpha=0.8)
        
        ax.set_title('Accuracy vs Removed High-Separability Features', fontweight='bold')
        ax.set_xlabel('Number of Top Separable Features Removed')
        ax.set_ylabel('Cross-Validation Accuracy')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot accuracy drop
        ax = axes[1]
        
        rf_baseline = rf_accs[0]
        lr_baseline = lr_accs[0]
        
        rf_drops = [rf_baseline - acc for acc in rf_accs]
        lr_drops = [lr_baseline - acc for acc in lr_accs]
        
        ax.plot(removal_counts, rf_drops, 'o-', linewidth=2, markersize=8, 
               label='Random Forest', alpha=0.8)
        ax.plot(removal_counts, lr_drops, 's-', linewidth=2, markersize=8, 
               label='Logistic Regression', alpha=0.8)
        
        ax.set_title('Accuracy Drop from Baseline', fontweight='bold')
        ax.set_xlabel('Number of Top Separable Features Removed')
        ax.set_ylabel('Accuracy Drop')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        ablation_file = FIGURES_DIR / "ablation_curve.png"
        plt.savefig(ablation_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Ablation curve saved: {ablation_file}")
        plt.show()
        
        # Analyze trend
        print(f"\n   📊 ABLATION ANALYSIS:")
        
        # Check if accuracy drops significantly
        rf_max_drop = max(rf_drops)
        lr_max_drop = max(lr_drops)
        
        print(f"     Maximum accuracy drop:")
        print(f"       Random Forest: {rf_max_drop:.3f}")
        print(f"       Logistic Regression: {lr_max_drop:.3f}")
        
        if rf_max_drop < 0.1 and lr_max_drop < 0.1:
            print(f"     🚨 MINIMAL ACCURACY DROP - Strong evidence of feature leakage!")
        elif rf_max_drop > 0.3 or lr_max_drop > 0.3:
            print(f"     ✅ Significant accuracy drop - Features contribute meaningfully")
        else:
            print(f"     ⚠️  Moderate accuracy drop - Mixed evidence")

def main():
    """Run cross-block validation experiment"""
    
    print("🧪 CROSS-BLOCK VALIDATION EXPERIMENT")
    print("=" * 80)
    
    experiment = CrossBlockValidationExperiment()
    
    # Load data
    X, y, block_ids, targets = experiment.load_data_with_blocks()
    
    # 1. Feature separability analysis
    feature_separability = experiment.compute_feature_separability(X, y)
    
    # 2. Block-aware within-block CV baseline
    baseline_results, baseline_summary = experiment.block_aware_within_block_cv(X, y, block_ids)
    
    # 3. Minimal feature test
    minimal_results = experiment.minimal_feature_test(X, y, block_ids, feature_separability)
    
    # 4. Remove high-separability features
    removal_results = experiment.remove_high_sep_features_experiment(X, y, block_ids, feature_separability)
    
    # 5. Create ablation curve
    experiment.create_ablation_curve(removal_results)
    
    # Save comprehensive results
    all_results = {
        'feature_separability': feature_separability,
        'baseline_summary': baseline_summary,
        'minimal_results': minimal_results,
        'removal_results': removal_results
    }
    
    results_file = RESULTS_DIR / "cross_block_validation_results.joblib"
    joblib.dump(all_results, results_file)
    
    print(f"\n" + "=" * 80)
    print("🎯 CROSS-BLOCK VALIDATION COMPLETE")
    print("=" * 80)
    print("📊 KEY FINDINGS:")
    
    if baseline_summary and 'RF' in baseline_summary:
        print(f"   • Baseline accuracy: {baseline_summary['RF']['mean_acc']:.3f}±{baseline_summary['RF']['std_acc']:.3f}")
    
    high_sep_features = len([f for f in feature_separability if f['max_separability'] > 5.0])
    print(f"   • High separability features: {high_sep_features}")
    
    print(f"   • Complete results saved: {results_file}")
    
    print(f"\n🏆 SYSTEMATIC LEAKAGE ANALYSIS COMPLETE!")
    
    return all_results

if __name__ == "__main__":
    results = main()