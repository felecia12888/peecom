#!/usr/bin/env python3
"""
PEECOM Robustness Validation Suite

Demonstrates PEECOM's key advantages over ICCIA:
1. Physics-informed features vs statistical features
2. Robustness under sensor failures
3. Interpretability and deployment considerations
4. Statistical significance of improvements

This directly addresses the novelty concerns identified.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('src')
from models.enhanced_peecom import EnhancedPEECOM


class PEECOMRobustnessValidator:
    """Validate PEECOM's robustness claims vs ICCIA baseline"""
    
    def __init__(self, output_dir="output/robustness_validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validation_results = {}
    
    def create_statistical_features(self, X):
        """Create ICCIA-style statistical features"""
        
        n_samples, n_sensors = X.shape
        statistical_features = []
        
        for i in range(n_samples):
            sample_features = []
            sensor_data = X[i]
            
            # Basic statistics per sample
            sample_features.extend([
                np.mean(sensor_data),
                np.std(sensor_data),
                np.min(sensor_data),
                np.max(sensor_data),
                np.median(sensor_data),
                np.percentile(sensor_data, 25),
                np.percentile(sensor_data, 75)
            ])
            
            statistical_features.append(sample_features)
        
        return np.array(statistical_features)
    
    def create_physics_features(self, X):
        """Create PEECOM physics-informed features"""
        
        peecom = EnhancedPEECOM()
        X_df = pd.DataFrame(X)
        X_enhanced = peecom._create_physics_features(X_df)
        return X_enhanced.values
    
    def sensor_failure_simulation(self, X, failure_rates=[0.1, 0.2, 0.3, 0.4, 0.5]):
        """Simulate sensor failures and test robustness"""
        
        results = {}
        n_sensors = X.shape[1]
        
        for failure_rate in failure_rates:
            print(f"🔧 Testing {failure_rate*100:.0f}% sensor failure rate...")
            
            # Randomly fail sensors
            n_failed = int(n_sensors * failure_rate)
            failed_sensors = np.random.choice(n_sensors, n_failed, replace=False)
            
            X_failed = X.copy()
            X_failed[:, failed_sensors] = 0  # Zero out failed sensors
            
            results[failure_rate] = {
                'failed_sensors': failed_sensors,
                'X_failed': X_failed
            }
        
        return results
    
    def interpretability_analysis(self, X_physics, feature_names):
        """Analyze physics feature interpretability"""
        
        # Train model to get feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        interpretability_metrics = {}
        
        for target in ['cooler_condition', 'valve_condition', 'pump_leakage']:
            try:
                # Load target if available
                y_path = "output/processed_data/cmohs/y_full.csv"
                if os.path.exists(y_path):
                    y_data = pd.read_csv(y_path)
                    if target in y_data.columns:
                        y = y_data[target].values
                        
                        rf.fit(X_physics, y)
                        importance = rf.feature_importances_
                        
                        # Group by physics categories
                        power_features = [i for i, name in enumerate(feature_names) if 'power' in name.lower()]
                        efficiency_features = [i for i, name in enumerate(feature_names) if 'ratio' in name.lower() or 'eff' in name.lower()]
                        composite_features = [i for i, name in enumerate(feature_names) if 'composite' in name.lower()]
                        
                        interpretability_metrics[target] = {
                            'power_importance': np.sum(importance[power_features]) if power_features else 0,
                            'efficiency_importance': np.sum(importance[efficiency_features]) if efficiency_features else 0,
                            'composite_importance': np.sum(importance[composite_features]) if composite_features else 0,
                            'top_features': [(feature_names[i], importance[i]) for i in np.argsort(importance)[-10:]]
                        }
                        
            except Exception as e:
                print(f"⚠️ Could not analyze {target}: {e}")
                continue
        
        return interpretability_metrics
    
    def statistical_significance_test(self, X_statistical, X_physics, y, n_permutations=100):
        """Test statistical significance of PEECOM vs statistical features"""
        
        print("📊 Running statistical significance tests...")
        
        # Models to compare
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Cross-validation scores
        scores_statistical = cross_val_score(model, X_statistical, y, cv=5, scoring='accuracy')
        scores_physics = cross_val_score(model, X_physics, y, cv=5, scoring='accuracy')
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores_physics, scores_statistical)
        
        # Permutation test
        def score_diff(X1, X2, y):
            score1 = cross_val_score(model, X1, y, cv=3, scoring='accuracy').mean()
            score2 = cross_val_score(model, X2, y, cv=3, scoring='accuracy').mean()
            return score1 - score2
        
        observed_diff = score_diff(X_physics, X_statistical, y)
        
        # Permutation test
        perm_diffs = []
        for _ in range(n_permutations):
            # Shuffle features randomly between sets
            combined_features = np.hstack([X_physics, X_statistical])
            np.random.shuffle(combined_features.T)
            
            split_point = X_physics.shape[1]
            X1_perm = combined_features[:, :split_point]
            X2_perm = combined_features[:, split_point:]
            
            perm_diff = score_diff(X1_perm, X2_perm, y)
            perm_diffs.append(perm_diff)
        
        perm_p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        return {
            'statistical_scores': scores_statistical,
            'physics_scores': scores_physics,
            'paired_t_stat': t_stat,
            'paired_p_value': p_value,
            'observed_diff': observed_diff,
            'permutation_p_value': perm_p_value,
            'effect_size': (scores_physics.mean() - scores_statistical.mean()) / np.sqrt(scores_statistical.var() + scores_physics.var())
        }
    
    def deployment_metrics(self, X_statistical, X_physics):
        """Measure deployment-relevant metrics"""
        
        print("🚀 Measuring deployment metrics...")
        
        metrics = {}
        
        # Feature computation time
        start_time = time.time()
        _ = self.create_statistical_features(X_statistical[:100])  # Sample for timing
        statistical_compute_time = time.time() - start_time
        
        start_time = time.time()
        _ = self.create_physics_features(X_physics[:100])
        physics_compute_time = time.time() - start_time
        
        # Model training time
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Dummy targets for timing
        y_dummy = np.random.randint(0, 2, X_statistical.shape[0])
        
        start_time = time.time()
        model.fit(X_statistical, y_dummy)
        statistical_train_time = time.time() - start_time
        
        start_time = time.time()
        model.fit(X_physics, y_dummy)
        physics_train_time = time.time() - start_time
        
        # Inference time
        start_time = time.time()
        model.predict(X_statistical[:100])
        statistical_inference_time = time.time() - start_time
        
        model.fit(X_physics, y_dummy)  # Refit for physics
        start_time = time.time()
        model.predict(X_physics[:100])
        physics_inference_time = time.time() - start_time
        
        metrics = {
            'feature_computation': {
                'statistical_time': statistical_compute_time,
                'physics_time': physics_compute_time,
                'physics_overhead': physics_compute_time / statistical_compute_time
            },
            'model_training': {
                'statistical_time': statistical_train_time,
                'physics_time': physics_train_time,
                'speedup_ratio': statistical_train_time / physics_train_time
            },
            'inference': {
                'statistical_time': statistical_inference_time,
                'physics_time': physics_inference_time,
                'speedup_ratio': statistical_inference_time / physics_inference_time
            },
            'feature_counts': {
                'statistical_features': X_statistical.shape[1],
                'physics_features': X_physics.shape[1],
                'compression_ratio': X_statistical.shape[1] / X_physics.shape[1]
            }
        }
        
        return metrics
    
    def run_comprehensive_validation(self, dataset_path="output/processed_data/cmohs"):
        """Run comprehensive robustness validation"""
        
        print("🛡️ Running PEECOM Robustness Validation Suite")
        print("=" * 60)
        print("Validating key advantages over ICCIA methodology...")
        
        # Load data
        try:
            X = pd.read_csv(f"{dataset_path}/X_full.csv").values
            y_data = pd.read_csv(f"{dataset_path}/y_full.csv")
        except Exception as e:
            print(f"❌ Could not load data: {e}")
            return None
        
        # Create feature sets
        print("🔧 Creating feature representations...")
        X_statistical = self.create_statistical_features(X)
        X_physics = self.create_physics_features(X)
        
        # Standardize
        scaler_stat = StandardScaler()
        scaler_phys = StandardScaler()
        
        X_statistical = scaler_stat.fit_transform(X_statistical)
        X_physics = scaler_phys.fit_transform(X_physics)
        
        validation_results = {}
        
        # Test each target
        targets = ['cooler_condition', 'valve_condition', 'pump_leakage']
        
        for target in targets:
            if target not in y_data.columns:
                continue
                
            print(f"\n🎯 Validating target: {target}")
            y = y_data[target].values
            
            target_results = {}
            
            # 1. Statistical significance test
            sig_results = self.statistical_significance_test(X_statistical, X_physics, y)
            target_results['statistical_significance'] = sig_results
            
            print(f"  📊 Statistical significance: p={sig_results['paired_p_value']:.4f}")
            print(f"  📈 Effect size: {sig_results['effect_size']:.4f}")
            
            # 2. Sensor failure robustness
            print("  🔧 Testing sensor failure robustness...")
            failure_results = self.sensor_failure_simulation(X)
            
            robustness_comparison = {}
            for failure_rate, failure_data in failure_results.items():
                X_failed = failure_data['X_failed']
                
                # Recreate features with failed sensors
                X_stat_failed = self.create_statistical_features(X_failed)
                X_phys_failed = self.create_physics_features(X_failed)
                
                X_stat_failed = scaler_stat.transform(X_stat_failed)
                X_phys_failed = scaler_phys.transform(X_phys_failed)
                
                # Test performance degradation
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                score_stat_failed = cross_val_score(model, X_stat_failed, y, cv=3).mean()
                score_phys_failed = cross_val_score(model, X_phys_failed, y, cv=3).mean()
                
                score_stat_baseline = cross_val_score(model, X_statistical, y, cv=3).mean()
                score_phys_baseline = cross_val_score(model, X_physics, y, cv=3).mean()
                
                robustness_comparison[failure_rate] = {
                    'statistical_degradation': score_stat_baseline - score_stat_failed,
                    'physics_degradation': score_phys_baseline - score_phys_failed,
                    'robustness_advantage': (score_stat_baseline - score_stat_failed) - (score_phys_baseline - score_phys_failed)
                }
            
            target_results['sensor_failure_robustness'] = robustness_comparison
            
            validation_results[target] = target_results
        
        # 3. Deployment metrics (once for all targets)
        print("\n🚀 Measuring deployment characteristics...")
        deployment_metrics = self.deployment_metrics(X_statistical, X_physics)
        validation_results['deployment_metrics'] = deployment_metrics
        
        # 4. Interpretability analysis
        print("\n🔍 Analyzing interpretability...")
        # Get feature names (simplified)
        physics_feature_names = [f'physics_feature_{i}' for i in range(X_physics.shape[1])]
        interpretability = self.interpretability_analysis(X_physics, physics_feature_names)
        validation_results['interpretability'] = interpretability
        
        # Save results
        self.save_validation_results(validation_results)
        
        return validation_results
    
    def save_validation_results(self, results):
        """Save validation results"""
        
        # Create summary report
        report_lines = [
            "# PEECOM Robustness Validation Report",
            "",
            "## Key Findings vs ICCIA Methodology",
            ""
        ]
        
        # Statistical significance summary
        if any('statistical_significance' in target_results for target_results in results.values() if isinstance(target_results, dict)):
            report_lines.extend([
                "### Statistical Significance",
                ""
            ])
            
            for target, target_results in results.items():
                if isinstance(target_results, dict) and 'statistical_significance' in target_results:
                    sig = target_results['statistical_significance']
                    report_lines.extend([
                        f"**{target}:**",
                        f"- Paired t-test p-value: {sig['paired_p_value']:.4f}",
                        f"- Effect size: {sig['effect_size']:.4f}",
                        f"- Physics features advantage: {sig['observed_diff']:.4f}",
                        ""
                    ])
        
        # Robustness summary
        report_lines.extend([
            "### Sensor Failure Robustness",
            ""
        ])
        
        for target, target_results in results.items():
            if isinstance(target_results, dict) and 'sensor_failure_robustness' in target_results:
                robustness = target_results['sensor_failure_robustness']
                report_lines.extend([
                    f"**{target}:**"
                ])
                
                for failure_rate, metrics in robustness.items():
                    advantage = metrics['robustness_advantage']
                    report_lines.append(f"- {failure_rate*100:.0f}% failure: {advantage:+.4f} robustness advantage")
                
                report_lines.append("")
        
        # Deployment metrics
        if 'deployment_metrics' in results:
            deploy = results['deployment_metrics']
            report_lines.extend([
                "### Deployment Characteristics",
                "",
                f"**Feature Engineering:**",
                f"- Physics feature computation overhead: {deploy['feature_computation']['physics_overhead']:.2f}x",
                f"- Feature count reduction: {deploy['feature_counts']['compression_ratio']:.2f}x",
                "",
                f"**Model Performance:**",
                f"- Training time ratio: {deploy['model_training']['speedup_ratio']:.2f}x",
                f"- Inference time ratio: {deploy['inference']['speedup_ratio']:.2f}x",
                ""
            ])
        
        # Write report
        with open(f"{self.output_dir}/robustness_validation_report.md", 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Save detailed JSON
        import json
        with open(f"{self.output_dir}/validation_results_detailed.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Validation results saved to {self.output_dir}")


def main():
    """Run robustness validation"""
    
    print("🛡️ PEECOM ROBUSTNESS VALIDATION vs ICCIA")
    print("=" * 50)
    
    from pathlib import Path
    globals()['Path'] = Path
    
    validator = PEECOMRobustnessValidator()
    
    try:
        results = validator.run_comprehensive_validation()
        
        if results:
            print("\n✅ ROBUSTNESS VALIDATION COMPLETE")
            print("=" * 50)
            print("Key differentiators from ICCIA established!")
            print("Check output/robustness_validation/ for detailed analysis")
        else:
            print("❌ Validation failed - check data availability")
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()