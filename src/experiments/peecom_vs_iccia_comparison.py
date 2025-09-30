#!/usr/bin/env python3
"""
PEECOM vs ICCIA Comparison Suite

Critical experiments to differentiate PEECOM from ICCIA 2023 paper:
1. Feature ablation + fusion cross-test
2. PEECOM + stacked ensemble baseline
3. Compute/latency comparison
4. Robustness under sensor failures

This addresses the novelty concerns identified in literature review.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import time
import os
import sys
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')
from loader.data_loader import load_data
from models.enhanced_peecom import EnhancedPEECOM


class PEECOMvsICCIAComparison:
    """Comprehensive comparison between PEECOM and ICCIA methodologies"""
    
    def __init__(self, output_dir="output/iccia_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ICCIA target performance for comparison
        self.iccia_performance = {
            'cooler_condition': {'best_acc': 0.998812, 'best_model': 'XGBOOST'},
            'valve_condition': {'best_acc': 0.8966, 'best_model': 'STACKING'},
            'pump_leakage': {'best_acc': 0.99093, 'best_model': 'XGBOOST/RF'},
            'accumulator_pressure': {'best_acc': 0.9682, 'best_model': 'STACKING'},
            'stable_flag': {'best_acc': 0.9659, 'best_model': 'STACKING/RF'}
        }
        
        self.results = {}
    
    def create_iccia_statistical_features(self, X):
        """Create ICCIA-style statistical features (mean, std, skew, kurtosis, etc.)"""
        
        statistical_features = []
        feature_names = []
        
        # Basic statistical measures per sensor
        for i in range(X.shape[1]):
            sensor_data = X[:, i]
            
            # ICCIA-style features
            features = [
                np.mean(sensor_data),           # Mean
                np.std(sensor_data),            # Standard deviation  
                np.min(sensor_data),            # Minimum
                np.max(sensor_data),            # Maximum
                np.percentile(sensor_data, 25), # Q1
                np.percentile(sensor_data, 75), # Q3
            ]
            
            names = [f'sensor_{i}_mean', f'sensor_{i}_std', f'sensor_{i}_min', 
                    f'sensor_{i}_max', f'sensor_{i}_q25', f'sensor_{i}_q75']
            
            statistical_features.extend(features)
            feature_names.extend(names)
        
        # Global statistical measures
        statistical_features.extend([
            np.mean(X),              # Overall mean
            np.std(X),               # Overall std
            np.min(X),               # Overall min
            np.max(X),               # Overall max
        ])
        
        feature_names.extend(['global_mean', 'global_std', 'global_min', 'global_max'])
        
        return np.array(statistical_features).reshape(1, -1), feature_names
    
    def create_iccia_features_batch(self, X):
        """Create ICCIA-style features for entire dataset"""
        
        all_features = []
        for i in range(X.shape[0]):
            features, _ = self.create_iccia_statistical_features(X[i:i+1])
            all_features.append(features[0])
        
        return np.array(all_features)
    
    def create_peecom_features(self, X):
        """Create PEECOM physics-informed features"""
        
        # Use existing PEECOM feature engineering
        peecom = EnhancedPEECOM()
        X_enhanced = peecom._create_physics_features(pd.DataFrame(X))
        return X_enhanced.values
    
    def create_iccia_fusion_models(self):
        """Create ICCIA-style fusion models"""
        
        # Base classifiers
        base_models = [
            ('svm', SVC(probability=True, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('xgboost', xgb.XGBClassifier(random_state=42, eval_metric='logloss')),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]
        
        # Meta learner for stacking
        meta_learner = LogisticRegression(random_state=42)
        
        # ICCIA fusion models
        fusion_models = {
            'stacking': StackingClassifier(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=5,
                n_jobs=-1
            ),
            'bagging_majority': VotingClassifier(
                estimators=base_models,
                voting='hard',
                n_jobs=-1
            ),
            'adaboost': AdaBoostClassifier(
                base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
                n_estimators=50,
                random_state=42
            )
        }
        
        return fusion_models
    
    def create_peecom_fusion_models(self):
        """Create PEECOM + fusion combinations"""
        
        # Enhanced base models with PEECOM features
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', xgb.XGBClassifier(random_state=42, eval_metric='logloss')),
            ('lgb', lgb.LGBMClassifier(random_state=42, verbose=-1))
        ]
        
        meta_learner = LogisticRegression(random_state=42)
        
        peecom_fusion = {
            'peecom_stacking': StackingClassifier(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=5,
                n_jobs=-1
            ),
            'peecom_voting': VotingClassifier(
                estimators=base_models,
                voting='soft',
                n_jobs=-1
            )
        }
        
        return peecom_fusion
    
    def measure_computation_time(self, model, X_train, y_train, X_test):
        """Measure training and inference time"""
        
        # Training time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Inference time (per sample)
        start_time = time.time()
        predictions = model.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test)
        
        return training_time, inference_time, predictions
    
    def feature_ablation_test(self, X_enhanced, y, target_name):
        """Test robustness through systematic feature removal"""
        
        # Get feature importance from Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_enhanced, y)
        
        feature_importance = rf.feature_importances_
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        # Progressive feature removal
        ablation_results = []
        
        for remove_pct in [0, 10, 20, 30, 40, 50]:
            n_remove = int(len(sorted_indices) * remove_pct / 100)
            
            if n_remove == 0:
                X_ablated = X_enhanced
            else:
                # Remove most important features (stress test)
                features_to_remove = sorted_indices[:n_remove]
                X_ablated = np.delete(X_enhanced, features_to_remove, axis=1)
            
            # Cross-validation with ablated features
            cv_scores = cross_val_score(
                RandomForestClassifier(n_estimators=100, random_state=42),
                X_ablated, y, cv=5, scoring='accuracy'
            )
            
            ablation_results.append({
                'removed_pct': remove_pct,
                'n_features': X_ablated.shape[1],
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            })
        
        return ablation_results
    
    def run_comprehensive_comparison(self, dataset_path="output/processed_data/cmohs"):
        """Run comprehensive PEECOM vs ICCIA comparison"""
        
        print("🔬 Running PEECOM vs ICCIA Comprehensive Comparison")
        print("=" * 60)
        
        # Load data
        X = pd.read_csv(f"{dataset_path}/X_full.csv").values
        y_data = pd.read_csv(f"{dataset_path}/y_full.csv")
        
        targets = ['cooler_condition', 'valve_condition', 'pump_leakage', 
                  'accumulator_pressure', 'stable_flag']
        
        comparison_results = {}
        
        for target in targets:
            if target not in y_data.columns:
                print(f"⚠️ Target {target} not found, skipping...")
                continue
                
            print(f"\n📊 Analyzing target: {target}")
            y = y_data[target].values
            
            # Create feature sets
            print("🔧 Creating feature sets...")
            X_iccia = self.create_iccia_features_batch(X)
            X_peecom = self.create_peecom_features(X)
            
            # Standardize features
            scaler_iccia = StandardScaler()
            scaler_peecom = StandardScaler()
            
            X_iccia_scaled = scaler_iccia.fit_transform(X_iccia)
            X_peecom_scaled = scaler_peecom.fit_transform(X_peecom)
            
            target_results = {}
            
            # 1. ICCIA methods on statistical features
            print("🏛️ Testing ICCIA fusion methods...")
            iccia_models = self.create_iccia_fusion_models()
            
            for model_name, model in iccia_models.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_iccia_scaled, y, cv=5, scoring='accuracy')
                    
                    # Ablation test
                    ablation_results = self.feature_ablation_test(X_iccia_scaled, y, target)
                    
                    target_results[f'iccia_{model_name}'] = {
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'feature_count': X_iccia_scaled.shape[1],
                        'ablation_results': ablation_results
                    }
                    
                except Exception as e:
                    print(f"❌ Error with {model_name}: {e}")
                    continue
            
            # 2. ICCIA methods on PEECOM features
            print("🔬 Testing ICCIA fusion on PEECOM features...")
            for model_name, model in iccia_models.items():
                try:
                    cv_scores = cross_val_score(model, X_peecom_scaled, y, cv=5, scoring='accuracy')
                    ablation_results = self.feature_ablation_test(X_peecom_scaled, y, target)
                    
                    target_results[f'iccia_{model_name}_peecom_features'] = {
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'feature_count': X_peecom_scaled.shape[1],
                        'ablation_results': ablation_results
                    }
                    
                except Exception as e:
                    print(f"❌ Error with {model_name} on PEECOM features: {e}")
                    continue
            
            # 3. PEECOM + fusion methods
            print("⚡ Testing PEECOM + fusion methods...")
            peecom_fusion = self.create_peecom_fusion_models()
            
            for model_name, model in peecom_fusion.items():
                try:
                    cv_scores = cross_val_score(model, X_peecom_scaled, y, cv=5, scoring='accuracy')
                    ablation_results = self.feature_ablation_test(X_peecom_scaled, y, target)
                    
                    target_results[model_name] = {
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'feature_count': X_peecom_scaled.shape[1],
                        'ablation_results': ablation_results
                    }
                    
                except Exception as e:
                    print(f"❌ Error with {model_name}: {e}")
                    continue
            
            # 4. Computational comparison
            print("⏱️ Measuring computational performance...")
            try:
                # Split data for timing
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_peecom_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Test key models
                timing_models = {
                    'iccia_stacking': self.create_iccia_fusion_models()['stacking'],
                    'peecom_stacking': self.create_peecom_fusion_models()['peecom_stacking'],
                    'peecom_rf': RandomForestClassifier(n_estimators=100, random_state=42)
                }
                
                for model_name, model in timing_models.items():
                    try:
                        train_time, inference_time, _ = self.measure_computation_time(
                            model, X_train, y_train, X_test
                        )
                        
                        target_results[f'{model_name}_timing'] = {
                            'training_time': train_time,
                            'inference_time_per_sample': inference_time,
                            'total_inference_time': inference_time * len(X_test)
                        }
                        
                    except Exception as e:
                        print(f"❌ Timing error with {model_name}: {e}")
                        
            except Exception as e:
                print(f"❌ Computational comparison failed: {e}")
            
            comparison_results[target] = target_results
            
            # Print summary for this target
            print(f"\n📈 {target} Results Summary:")
            for method, results in target_results.items():
                if 'cv_mean' in results:
                    iccia_best = self.iccia_performance.get(target, {}).get('best_acc', 0)
                    improvement = (results['cv_mean'] - iccia_best) * 100
                    print(f"  {method}: {results['cv_mean']:.4f} ± {results['cv_std']:.4f} "
                          f"(vs ICCIA: {improvement:+.2f}%)")
        
        # Save comprehensive results
        self.save_comparison_results(comparison_results)
        return comparison_results
    
    def save_comparison_results(self, results):
        """Save comparison results to files"""
        
        # Save detailed results
        import json
        with open(self.output_dir / "iccia_comparison_detailed.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary table
        summary_data = []
        
        for target, target_results in results.items():
            iccia_best = self.iccia_performance.get(target, {}).get('best_acc', 0)
            
            for method, method_results in target_results.items():
                if 'cv_mean' in method_results:
                    summary_data.append({
                        'target': target,
                        'method': method,
                        'cv_mean': method_results['cv_mean'],
                        'cv_std': method_results['cv_std'],
                        'feature_count': method_results.get('feature_count', 0),
                        'iccia_best': iccia_best,
                        'improvement': (method_results['cv_mean'] - iccia_best) * 100
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / "iccia_comparison_summary.csv", index=False)
        
        print(f"\n✅ Results saved to {self.output_dir}")
        return summary_df


def main():
    """Run PEECOM vs ICCIA comparison"""
    
    print("🚨 CRITICAL NOVELTY VALIDATION: PEECOM vs ICCIA 2023")
    print("=" * 60)
    print("Addressing concerns about overlapping contributions...")
    print("Running comprehensive comparison experiments...")
    
    # Initialize comparison
    comparison = PEECOMvsICCIAComparison()
    
    # Check if data exists
    data_path = "output/processed_data/cmohs"
    if not os.path.exists(data_path):
        print(f"❌ Data not found at {data_path}")
        print("Please run dataset preprocessing first.")
        return 1
    
    # Run comprehensive comparison
    try:
        results = comparison.run_comprehensive_comparison(data_path)
        
        print("\n" + "=" * 60)
        print("✅ PEECOM vs ICCIA COMPARISON COMPLETE")
        print("=" * 60)
        print("Key findings will help establish PEECOM's novelty...")
        print("Check output/iccia_comparison/ for detailed results")
        
        return 0
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())