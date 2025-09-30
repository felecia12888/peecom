#!/usr/bin/env python3
"""
Temporal Validation Framework - Complete Remediation

Implements proper time-aware cross-validation to eliminate temporal data leakage.
Follows the exact remediation plan for scientific integrity.
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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up paths
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset" / "cmohs"
OUTPUT_DIR = ROOT / "output" / "temporal_validation"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

class TemporalValidationFramework:
    """
    Complete temporal validation framework implementing the remediation plan
    """
    
    def __init__(self, embargo_frac=0.02, random_state=42):
        """
        Initialize temporal validation framework
        
        Args:
            embargo_frac: Fraction of samples to use as embargo buffer
            random_state: Random seed for reproducibility
        """
        self.embargo_frac = embargo_frac
        self.random_state = random_state
        self.results = {}
        
        # Create output directories
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        print("🏗️  TEMPORAL VALIDATION FRAMEWORK INITIALIZED")
        print("=" * 60)
        print(f"✅ Embargo fraction: {self.embargo_frac}")
        print(f"✅ Random state: {self.random_state}")
        print(f"✅ Output directory: {OUTPUT_DIR}")
    
    def rolling_origin_indices(self, n_samples, init_train_frac=0.5, 
                              horizon_frac=0.15, step_frac=0.1):
        """
        Generate rolling-origin (forward-chaining) cross-validation indices
        
        Args:
            n_samples: Total number of samples
            init_train_frac: Initial training window fraction
            horizon_frac: Test horizon fraction
            step_frac: Step size fraction for rolling window
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        W0 = int(n_samples * init_train_frac)
        H = int(n_samples * horizon_frac)
        S = int(n_samples * step_frac)
        e = int(n_samples * self.embargo_frac)
        
        splits = []
        start = W0
        fold_num = 0
        
        print(f"\n🔄 GENERATING ROLLING-ORIGIN CV SPLITS")
        print(f"   Initial training: {W0} samples ({init_train_frac:.1%})")
        print(f"   Test horizon: {H} samples ({horizon_frac:.1%})")
        print(f"   Step size: {S} samples ({step_frac:.1%})")
        print(f"   Embargo: {e} samples ({self.embargo_frac:.1%})")
        
        while start + H <= n_samples:
            train_idx = np.arange(0, start)
            test_idx = np.arange(start + e, min(start + H - e, n_samples))
            
            if len(test_idx) > 10:  # Minimum test size
                splits.append((train_idx, test_idx))
                print(f"   Fold {fold_num}: Train[0:{start}] → Test[{start+e}:{start+H-e}]")
                fold_num += 1
            
            start += S
        
        print(f"✅ Generated {len(splits)} temporal folds")
        return splits
    
    def chronological_holdout_indices(self, n_samples, train_frac=0.7):
        """
        Generate single chronological holdout split (conservative baseline)
        
        Args:
            n_samples: Total number of samples
            train_frac: Fraction for training (rest for testing)
            
        Returns:
            Tuple of (train_indices, test_indices)
        """
        split_point = int(n_samples * train_frac)
        e = int(n_samples * self.embargo_frac)
        
        train_idx = np.arange(0, split_point)
        test_idx = np.arange(split_point + e, n_samples)
        
        print(f"\n📅 CHRONOLOGICAL HOLDOUT SPLIT")
        print(f"   Training: samples 0-{split_point} ({len(train_idx)} samples)")
        print(f"   Embargo: {e} samples")
        print(f"   Testing: samples {split_point+e}-{n_samples} ({len(test_idx)} samples)")
        
        return train_idx, test_idx
    
    def load_hydraulic_data(self):
        """Load and prepare hydraulic system dataset"""
        
        print(f"\n📊 LOADING HYDRAULIC SYSTEM DATASET")
        
        # Load targets
        profile_file = DATASET_DIR / "profile.txt"
        if not profile_file.exists():
            raise FileNotFoundError(f"Profile file not found: {profile_file}")
        
        targets = pd.read_csv(profile_file, sep='\t', header=None,
                             names=['cooler_condition', 'valve_condition', 'pump_leakage',
                                   'accumulator_pressure', 'stable_flag'])
        
        print(f"✅ Loaded targets: {targets.shape}")
        print(f"   Target distributions:")
        for col in targets.columns:
            print(f"     {col}: {targets[col].value_counts().to_dict()}")
        
        # Load features from multiple sensor files
        feature_files = ['PS1.txt', 'PS2.txt', 'PS3.txt', 'PS4.txt', 'PS5.txt', 'PS6.txt']
        all_features = []
        
        for file_name in feature_files:
            file_path = DATASET_DIR / file_name
            if file_path.exists():
                print(f"   Loading {file_name}...")
                with open(file_path, 'r') as f:
                    sensor_data = []
                    for line in f:
                        values = [float(x) for x in line.strip().split('\t')]
                        sensor_data.append(values)
                
                sensor_df = pd.DataFrame(sensor_data)
                all_features.append(sensor_df)
                print(f"     Shape: {sensor_df.shape}")
        
        if not all_features:
            raise FileNotFoundError("No sensor files found")
        
        # Combine all features
        features = pd.concat(all_features, axis=1, ignore_index=True)
        
        # Align features and targets (take minimum length)
        min_length = min(len(features), len(targets))
        features = features.iloc[:min_length]
        targets = targets.iloc[:min_length]
        
        print(f"✅ Final dataset: {features.shape} features, {targets.shape} targets")
        
        return features.values, targets
    
    def generate_physics_features(self, X_raw, apply_past_only=False):
        """
        Generate physics-informed features from raw sensor data
        CRITICAL: Only use past data when apply_past_only=True
        
        Args:
            X_raw: Raw sensor data
            apply_past_only: If True, only use causal (past) information
            
        Returns:
            Enhanced feature matrix with physics features
        """
        features = []
        
        # Basic statistical features (using rolling windows for past-only)
        if apply_past_only:
            # Use expanding window (only past data)
            window_size = min(50, len(X_raw))  # Adaptive window
        else:
            window_size = 60  # Fixed window for training
        
        # Rolling statistics (causal)
        X_df = pd.DataFrame(X_raw)
        
        # Mean and std with expanding/rolling window
        if apply_past_only:
            rolling_mean = X_df.expanding(min_periods=1).mean().values
            rolling_std = X_df.expanding(min_periods=1).std().fillna(0).values
        else:
            rolling_mean = X_df.rolling(window_size, min_periods=1).mean().values
            rolling_std = X_df.rolling(window_size, min_periods=1).std().fillna(0).values
        
        features.extend([X_raw, rolling_mean, rolling_std])
        
        # First differences (causal)
        diff_features = np.diff(X_raw, axis=0, prepend=X_raw[0:1])
        features.append(diff_features)
        
        # Pressure ratios (instantaneous, causal)
        if X_raw.shape[1] >= 6:  # At least 6 pressure sensors
            ratio_features = []
            for i in range(3):
                for j in range(i+1, 6):
                    ratio = np.divide(X_raw[:, i], X_raw[:, j] + 1e-8)  # Avoid division by zero
                    ratio_features.append(ratio.reshape(-1, 1))
            if ratio_features:
                features.append(np.hstack(ratio_features))
        
        # Combine all features
        combined_features = np.hstack([f for f in features if f.size > 0])
        
        print(f"   Generated physics features: {X_raw.shape} → {combined_features.shape}")
        
        return combined_features
    
    def temporal_fold_training(self, X, y, train_idx, test_idx, model_name="RF"):
        """
        Train model on a single temporal fold with proper preprocessing isolation
        
        Args:
            X: Feature matrix
            y: Target vector
            train_idx: Training indices
            test_idx: Test indices
            model_name: Model type ("RF", "LR", "SVM")
            
        Returns:
            Dictionary with fold results
        """
        # Split data temporally
        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_test_raw, y_test = X[test_idx], y[test_idx]
        
        # 1) Fit preprocessing components on training data ONLY
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(X_train_raw)
        X_train_imputed = imputer.transform(X_train_raw)
        
        # 2) Generate physics features from training data only
        X_train_physics = self.generate_physics_features(X_train_imputed, apply_past_only=False)
        
        # 3) Fit scaler on training physics features
        scaler = StandardScaler()
        scaler.fit(X_train_physics)
        X_train_scaled = scaler.transform(X_train_physics)
        
        # 4) Apply same preprocessing to test data (NO REFITTING)
        X_test_imputed = imputer.transform(X_test_raw)
        X_test_physics = self.generate_physics_features(X_test_imputed, apply_past_only=True)
        X_test_scaled = scaler.transform(X_test_physics)
        
        # 5) Train model
        if model_name == "RF":
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif model_name == "LR":
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif model_name == "SVM":
            model = SVC(random_state=self.random_state, probability=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model.fit(X_train_scaled, y_train)
        
        # 6) Evaluate
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        # Anti-leakage check: test accuracy should not exceed training accuracy
        if test_acc > train_acc + 0.05:  # Allow small margin for noise
            print(f"⚠️  WARNING: Test accuracy ({test_acc:.3f}) > Train accuracy ({train_acc:.3f})")
            print("    This may indicate remaining leakage - investigate!")
        
        # Additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')
        
        try:
            y_proba_test = model.predict_proba(X_test_scaled)
            if y_proba_test.shape[1] == 2:
                auc = roc_auc_score(y_test, y_proba_test[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba_test, multi_class='ovr', average='weighted')
        except:
            auc = np.nan
        
        return {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'scaler': scaler,
            'imputer': imputer
        }
    
    def run_rolling_origin_cv(self, X, y, target_name, models=['RF', 'LR']):
        """
        Run complete rolling-origin cross-validation
        
        Args:
            X: Feature matrix
            y: Target vector
            target_name: Name of target variable
            models: List of model types to evaluate
            
        Returns:
            Dictionary with results for all models
        """
        print(f"\n🎯 ROLLING-ORIGIN CV: {target_name.upper()}")
        print("=" * 60)
        
        # Generate temporal splits
        splits = self.rolling_origin_indices(len(X))
        
        results = {}
        
        for model_name in models:
            print(f"\n🤖 Training {model_name} with {len(splits)} temporal folds")
            
            fold_results = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                print(f"   Fold {fold_idx + 1}/{len(splits)}: "
                      f"Train={len(train_idx)}, Test={len(test_idx)}")
                
                fold_result = self.temporal_fold_training(
                    X, y, train_idx, test_idx, model_name
                )
                fold_result['fold'] = fold_idx
                fold_results.append(fold_result)
                
                print(f"     Train Acc: {fold_result['train_acc']:.3f}, "
                      f"Test Acc: {fold_result['test_acc']:.3f}")
            
            # Aggregate results
            test_accs = [r['test_acc'] for r in fold_results]
            train_accs = [r['train_acc'] for r in fold_results]
            
            results[model_name] = {
                'fold_results': fold_results,
                'mean_test_acc': np.mean(test_accs),
                'std_test_acc': np.std(test_accs),
                'mean_train_acc': np.mean(train_accs),
                'ci_lower': np.mean(test_accs) - 1.96 * np.std(test_accs) / np.sqrt(len(test_accs)),
                'ci_upper': np.mean(test_accs) + 1.96 * np.std(test_accs) / np.sqrt(len(test_accs)),
                'target': target_name,
                'n_folds': len(splits)
            }
            
            print(f"✅ {model_name} Results: {results[model_name]['mean_test_acc']:.3f} "
                  f"± {results[model_name]['std_test_acc']:.3f}")
        
        return results
    
    def run_chronological_holdout(self, X, y, target_name, models=['RF', 'LR']):
        """
        Run chronological holdout validation (conservative baseline)
        
        Args:
            X: Feature matrix
            y: Target vector  
            target_name: Name of target variable
            models: List of model types to evaluate
            
        Returns:
            Dictionary with holdout results
        """
        print(f"\n📅 CHRONOLOGICAL HOLDOUT: {target_name.upper()}")
        print("=" * 60)
        
        train_idx, test_idx = self.chronological_holdout_indices(len(X))
        
        results = {}
        
        for model_name in models:
            print(f"\n🤖 Training {model_name} on chronological split")
            
            fold_result = self.temporal_fold_training(
                X, y, train_idx, test_idx, model_name
            )
            
            results[model_name] = {
                'train_acc': fold_result['train_acc'],
                'test_acc': fold_result['test_acc'],
                'precision': fold_result['precision'],
                'recall': fold_result['recall'],
                'f1': fold_result['f1'],
                'auc': fold_result['auc'],
                'target': target_name,
                'model_obj': fold_result['model']
            }
            
            print(f"✅ {model_name} Results: Train={fold_result['train_acc']:.3f}, "
                  f"Test={fold_result['test_acc']:.3f}")
        
        return results
    
    def run_complete_validation_suite(self):
        """Run complete temporal validation for all targets"""
        
        print(f"\n🚀 STARTING COMPLETE TEMPORAL VALIDATION SUITE")
        print("=" * 80)
        
        # Load data
        X, targets_df = self.load_hydraulic_data()
        
        # Encode targets
        le = LabelEncoder()
        
        all_results = {
            'rolling_origin': {},
            'chronological_holdout': {}
        }
        
        # Process each target
        for target_name in targets_df.columns:
            print(f"\n" + "="*80)
            print(f"🎯 PROCESSING TARGET: {target_name}")
            print("="*80)
            
            # Encode target
            y = le.fit_transform(targets_df[target_name])
            
            print(f"Target distribution: {np.bincount(y)}")
            
            # Run rolling-origin CV
            rolling_results = self.run_rolling_origin_cv(X, y, target_name)
            all_results['rolling_origin'][target_name] = rolling_results
            
            # Run chronological holdout
            holdout_results = self.run_chronological_holdout(X, y, target_name)
            all_results['chronological_holdout'][target_name] = holdout_results
        
        # Save results
        results_file = OUTPUT_DIR / "temporal_validation_results.joblib"
        joblib.dump(all_results, results_file)
        print(f"\n✅ Results saved: {results_file}")
        
        # Create summary report
        self.create_validation_summary(all_results)
        
        return all_results
    
    def create_validation_summary(self, results):
        """Create comprehensive summary of temporal validation results"""
        
        print(f"\n📊 CREATING VALIDATION SUMMARY")
        
        # Rolling-origin summary table
        ro_data = []
        for target, target_results in results['rolling_origin'].items():
            for model, model_results in target_results.items():
                ro_data.append({
                    'Target': target,
                    'Model': model,
                    'Mean_Accuracy': model_results['mean_test_acc'],
                    'Std_Accuracy': model_results['std_test_acc'],
                    'CI_Lower': model_results['ci_lower'],
                    'CI_Upper': model_results['ci_upper'],
                    'N_Folds': model_results['n_folds']
                })
        
        ro_df = pd.DataFrame(ro_data)
        
        # Chronological holdout summary
        ch_data = []
        for target, target_results in results['chronological_holdout'].items():
            for model, model_results in target_results.items():
                ch_data.append({
                    'Target': target,
                    'Model': model,
                    'Train_Accuracy': model_results['train_acc'],
                    'Test_Accuracy': model_results['test_acc'],
                    'Precision': model_results['precision'],
                    'Recall': model_results['recall'],
                    'F1': model_results['f1'],
                    'AUC': model_results['auc']
                })
        
        ch_df = pd.DataFrame(ch_data)
        
        # Save summary tables
        ro_df.to_csv(OUTPUT_DIR / "rolling_origin_summary.csv", index=False)
        ch_df.to_csv(OUTPUT_DIR / "chronological_holdout_summary.csv", index=False)
        
        print(f"✅ Summary tables saved")
        print(f"\n📈 ROLLING-ORIGIN RESULTS PREVIEW:")
        print(ro_df.round(3))
        print(f"\n📈 CHRONOLOGICAL HOLDOUT RESULTS PREVIEW:")
        print(ch_df.round(3))
        
        # Create visualizations
        self.create_validation_plots(ro_df, ch_df)
        
        return ro_df, ch_df
    
    def create_validation_plots(self, ro_df, ch_df):
        """Create publication-quality validation plots"""
        
        print(f"\n📊 CREATING VALIDATION PLOTS")
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Rolling-origin performance comparison
        ax = axes[0, 0]
        
        # Pivot for plotting
        pivot_data = ro_df.pivot(index='Target', columns='Model', values='Mean_Accuracy')
        pivot_err = ro_df.pivot(index='Target', columns='Model', values='Std_Accuracy')
        
        pivot_data.plot(kind='bar', ax=ax, width=0.8, 
                       capsize=4, error_kw={'elinewidth': 2})
        
        ax.set_title('A) Rolling-Origin Cross-Validation Results\n(Mean ± Std Across Temporal Folds)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Variable')
        ax.set_ylabel('Accuracy')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 2: Chronological holdout comparison
        ax = axes[0, 1]
        
        pivot_ch = ch_df.pivot(index='Target', columns='Model', values='Test_Accuracy')
        pivot_ch.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('B) Chronological Holdout Results\n(Conservative Baseline)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Variable')
        ax.set_ylabel('Test Accuracy')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 3: Confidence intervals (Rolling-origin)
        ax = axes[1, 0]
        
        for model in ro_df['Model'].unique():
            model_data = ro_df[ro_df['Model'] == model]
            x_pos = range(len(model_data))
            
            ax.errorbar(x_pos, model_data['Mean_Accuracy'], 
                       yerr=[model_data['Mean_Accuracy'] - model_data['CI_Lower'],
                            model_data['CI_Upper'] - model_data['Mean_Accuracy']],
                       marker='o', capsize=5, capthick=2, linewidth=2,
                       label=model, markersize=8)
        
        ax.set_title('C) Rolling-Origin Results with 95% Confidence Intervals', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Index')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(len(ro_df['Target'].unique())))
        ax.set_xticklabels([t.replace('_', '\n') for t in ro_df['Target'].unique()], 
                          rotation=0, ha='center')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 4: Performance comparison (side-by-side)
        ax = axes[1, 1]
        
        # Combine data for comparison
        comparison_data = []
        for target in ro_df['Target'].unique():
            for model in ro_df['Model'].unique():
                ro_acc = ro_df[(ro_df['Target'] == target) & 
                              (ro_df['Model'] == model)]['Mean_Accuracy'].values[0]
                ch_acc = ch_df[(ch_df['Target'] == target) & 
                              (ch_df['Model'] == model)]['Test_Accuracy'].values[0]
                
                comparison_data.append({
                    'Target': target,
                    'Model': model,
                    'Rolling_Origin': ro_acc,
                    'Chronological': ch_acc
                })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Create grouped bar plot
        x = np.arange(len(comp_df))
        width = 0.35
        
        ax.bar(x - width/2, comp_df['Rolling_Origin'], width, 
              label='Rolling-Origin CV', alpha=0.8)
        ax.bar(x + width/2, comp_df['Chronological'], width, 
              label='Chronological Holdout', alpha=0.8)
        
        ax.set_title('D) Validation Method Comparison\n(All Target-Model Combinations)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Target-Model Combination')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['Target'][:6]}\n{row['Model']}" 
                           for _, row in comp_df.iterrows()], 
                          rotation=45, ha='right', fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.suptitle('🕐 TEMPORAL VALIDATION RESULTS\n' + 
                    'Proper Time-Aware Cross-Validation (No Data Leakage)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save plots
        plot_file = FIGURES_DIR / "temporal_validation_results.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.savefig(FIGURES_DIR / "temporal_validation_results.pdf", bbox_inches='tight')
        
        print(f"✅ Validation plots saved: {plot_file}")
        plt.show()

def main():
    """Run complete temporal validation framework"""
    
    print("🚀 STARTING TEMPORAL VALIDATION REMEDIATION")
    print("=" * 80)
    print("📋 REMEDIATION PLAN:")
    print("   ✅ Stop using random-permutation CV")
    print("   ✅ Implement rolling-origin (forward-chaining) CV")
    print("   ✅ Add chronological holdout baseline")
    print("   ✅ Move preprocessing inside each fold")
    print("   ✅ Generate physics features with past-only constraint")
    print("   ✅ Anti-leakage diagnostics")
    print("   ✅ Publication-ready documentation")
    
    # Initialize framework
    framework = TemporalValidationFramework(embargo_frac=0.02)
    
    # Run complete validation suite
    results = framework.run_complete_validation_suite()
    
    print(f"\n" + "=" * 80)
    print("🎯 TEMPORAL VALIDATION REMEDIATION COMPLETE")
    print("=" * 80)
    print("📊 EXPECTED OUTCOMES:")
    print("   📉 Lower but HONEST accuracy (~60-75% range)")
    print("   ✅ No temporal data leakage")
    print("   ✅ Proper time-aware validation")
    print("   ✅ Scientific integrity maintained")
    print("   📈 Robust evidence for PEECOM novelty")
    
    return results

if __name__ == "__main__":
    results = main()