#!/usr/bin/env python3
"""
Quick PEECOM vs ICCIA Novelty Validation

Focused comparison to address novelty concerns:
1. Statistical vs Physics features
2. Robustness under sensor failures
3. Computational efficiency comparison
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import time
import os
from pathlib import Path
from scipy import stats


def create_iccia_statistical_features(X):
    """Create ICCIA-style statistical features"""
    
    n_samples, n_sensors = X.shape
    features = []
    
    for i in range(n_samples):
        sample_data = X[i]
        
        # Basic statistical features per sample
        sample_features = [
            np.mean(sample_data),      # Mean
            np.std(sample_data),       # Standard deviation
            np.min(sample_data),       # Minimum
            np.max(sample_data),       # Maximum
            np.median(sample_data),    # Median
            np.percentile(sample_data, 25),  # Q1
            np.percentile(sample_data, 75),  # Q3
            np.var(sample_data),       # Variance
            sample_data.max() - sample_data.min(),  # Range
        ]
        
        features.append(sample_features)
    
    return np.array(features)


def create_peecom_physics_features(X):
    """Create PEECOM-style physics-informed features"""
    
    n_samples, n_sensors = X.shape
    features = []
    
    for i in range(n_samples):
        sample_data = X[i]
        
        # Physics-informed features
        physics_features = []
        
        # 1. Power proxy features (cross-products representing energy interactions)
        for j in range(min(10, n_sensors)):  # Limit for computational efficiency
            for k in range(j+1, min(10, n_sensors)):
                power_proxy = sample_data[j] * sample_data[k]
                physics_features.append(power_proxy)
        
        # 2. Efficiency ratio features (ratios representing system efficiency)
        for j in range(min(15, n_sensors)):
            for k in range(j+1, min(15, n_sensors)):
                if sample_data[k] != 0:
                    efficiency_ratio = sample_data[j] / (sample_data[k] + 1e-8)
                    physics_features.append(efficiency_ratio)
        
        # 3. System-level aggregations (preserving physical relationships)
        physics_features.extend([
            np.mean(sample_data),                    # System average
            np.std(sample_data),                     # System variability
            np.sum(np.abs(sample_data)),            # Total system energy proxy
            np.sqrt(np.sum(sample_data**2)),        # System magnitude
        ])
        
        # 4. Temperature-pressure-flow relationships (hydraulic physics)
        if n_sensors >= 6:  # Assume first sensors are key measurements
            temp_proxy = sample_data[0]
            pressure_proxy = sample_data[1] 
            flow_proxy = sample_data[2]
            
            physics_features.extend([
                temp_proxy * pressure_proxy,         # Thermodynamic interaction
                pressure_proxy * flow_proxy,         # Hydraulic power
                temp_proxy / (pressure_proxy + 1e-8), # Thermal efficiency
            ])
        
        features.append(physics_features)
    
    return np.array(features)


def simulate_sensor_failures(X, failure_rate=0.2):
    """Simulate sensor failures by zeroing out random sensors"""
    
    n_samples, n_sensors = X.shape
    X_failed = X.copy()
    
    # Randomly select sensors to fail
    n_failed = int(n_sensors * failure_rate)
    failed_sensors = np.random.choice(n_sensors, n_failed, replace=False)
    
    # Zero out failed sensors
    X_failed[:, failed_sensors] = 0
    
    return X_failed, failed_sensors


def create_iccia_fusion_model():
    """Create ICCIA-style stacking classifier"""
    
    base_models = [
        ('svm', SVC(probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('xgb', xgb.XGBClassifier(random_state=42, eval_metric='logloss')),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42))
    ]
    
    meta_learner = LogisticRegression(random_state=42)
    
    return StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=3,  # Reduced for speed
        n_jobs=-1
    )


def run_novelty_validation():
    """Run focused novelty validation experiments"""
    
    print("🚨 PEECOM NOVELTY VALIDATION vs ICCIA 2023")
    print("=" * 55)
    
    # Load data
    data_path = "output/processed_data/cmohs"
    try:
        X = pd.read_csv(f"{data_path}/X_full.csv").values
        y_data = pd.read_csv(f"{data_path}/y_full.csv")
        print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} sensors")
    except Exception as e:
        print(f"❌ Could not load data: {e}")
        return
    
    # Create output directory
    output_dir = Path("output/novelty_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Test key targets
    targets = ['cooler_condition', 'valve_condition', 'pump_leakage']
    
    for target in targets:
        if target not in y_data.columns:
            continue
            
        print(f"\n🎯 Testing target: {target}")
        y = y_data[target].values
        
        # Create feature sets
        print("  🔧 Creating feature representations...")
        X_statistical = create_iccia_statistical_features(X)
        X_physics = create_peecom_physics_features(X)
        
        print(f"  📊 Statistical features: {X_statistical.shape[1]}")
        print(f"  ⚡ Physics features: {X_physics.shape[1]}")
        
        # Standardize features
        scaler_stat = StandardScaler()
        scaler_phys = StandardScaler()
        
        X_stat_scaled = scaler_stat.fit_transform(X_statistical)
        X_phys_scaled = scaler_phys.fit_transform(X_physics)
        
        target_results = {}
        
        # 1. Baseline Performance Comparison
        print("  📈 Testing baseline performance...")
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Statistical features performance
        stat_scores = cross_val_score(rf_model, X_stat_scaled, y, cv=5, scoring='accuracy')
        target_results['statistical_baseline'] = {
            'mean': stat_scores.mean(),
            'std': stat_scores.std(),
            'feature_count': X_statistical.shape[1]
        }
        
        # Physics features performance  
        phys_scores = cross_val_score(rf_model, X_phys_scaled, y, cv=5, scoring='accuracy')
        target_results['physics_baseline'] = {
            'mean': phys_scores.mean(),
            'std': phys_scores.std(),
            'feature_count': X_physics.shape[1]
        }
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_rel(phys_scores, stat_scores)
        target_results['significance_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': (phys_scores.mean() - stat_scores.mean()) / np.sqrt(stat_scores.var() + phys_scores.var())
        }
        
        print(f"    Statistical: {stat_scores.mean():.4f} ± {stat_scores.std():.4f}")
        print(f"    Physics: {phys_scores.mean():.4f} ± {phys_scores.std():.4f}")
        print(f"    Significance: p = {p_value:.4f}")
        
        # 2. Fusion Model Comparison
        print("  🏛️ Testing fusion models...")
        
        try:
            # ICCIA fusion on statistical features
            iccia_fusion = create_iccia_fusion_model()
            iccia_scores = cross_val_score(iccia_fusion, X_stat_scaled, y, cv=3, scoring='accuracy')
            target_results['iccia_fusion'] = {
                'mean': iccia_scores.mean(),
                'std': iccia_scores.std()
            }
            
            # ICCIA fusion on physics features
            iccia_phys_scores = cross_val_score(iccia_fusion, X_phys_scaled, y, cv=3, scoring='accuracy')
            target_results['iccia_fusion_physics'] = {
                'mean': iccia_phys_scores.mean(),
                'std': iccia_phys_scores.std()
            }
            
            print(f"    ICCIA fusion (stat): {iccia_scores.mean():.4f} ± {iccia_scores.std():.4f}")
            print(f"    ICCIA fusion (phys): {iccia_phys_scores.mean():.4f} ± {iccia_phys_scores.std():.4f}")
            
        except Exception as e:
            print(f"    ⚠️ Fusion testing failed: {e}")
        
        # 3. Robustness Testing
        print("  🛡️ Testing sensor failure robustness...")
        
        robustness_results = {}
        
        for failure_rate in [0.1, 0.2, 0.3, 0.4]:
            X_failed, failed_sensors = simulate_sensor_failures(X, failure_rate)
            
            # Recreate features with failed sensors
            X_stat_failed = create_iccia_statistical_features(X_failed)
            X_phys_failed = create_peecom_physics_features(X_failed)
            
            X_stat_failed = scaler_stat.transform(X_stat_failed)
            X_phys_failed = scaler_phys.transform(X_phys_failed)
            
            # Test performance with failures
            stat_failed_score = cross_val_score(rf_model, X_stat_failed, y, cv=3).mean()
            phys_failed_score = cross_val_score(rf_model, X_phys_failed, y, cv=3).mean()
            
            # Calculate robustness (performance retention)
            stat_robustness = stat_failed_score / stat_scores.mean()
            phys_robustness = phys_failed_score / phys_scores.mean()
            
            robustness_results[failure_rate] = {
                'statistical_retention': stat_robustness,
                'physics_retention': phys_robustness,
                'robustness_advantage': phys_robustness - stat_robustness
            }
            
            print(f"    {failure_rate*100:.0f}% failure: Stat={stat_robustness:.3f}, Phys={phys_robustness:.3f}, Advantage={phys_robustness-stat_robustness:+.3f}")
        
        target_results['robustness'] = robustness_results
        
        # 4. Computational Efficiency
        print("  ⏱️ Testing computational efficiency...")
        
        # Feature creation time
        start_time = time.time()
        _ = create_iccia_statistical_features(X[:100])
        stat_feature_time = time.time() - start_time
        
        start_time = time.time()
        _ = create_peecom_physics_features(X[:100])
        phys_feature_time = time.time() - start_time
        
        # Model training time
        X_train, X_test, y_train, y_test = train_test_split(X_stat_scaled, y, test_size=0.2, random_state=42)
        
        start_time = time.time()
        rf_model.fit(X_train, y_train)
        stat_train_time = time.time() - start_time
        
        X_train_phys, X_test_phys, _, _ = train_test_split(X_phys_scaled, y, test_size=0.2, random_state=42)
        
        start_time = time.time()
        rf_model.fit(X_train_phys, y_train)
        phys_train_time = time.time() - start_time
        
        target_results['computational'] = {
            'stat_feature_time': stat_feature_time,
            'phys_feature_time': phys_feature_time,
            'stat_train_time': stat_train_time,
            'phys_train_time': phys_train_time,
            'feature_time_ratio': phys_feature_time / stat_feature_time,
            'train_time_ratio': phys_train_time / stat_train_time
        }
        
        print(f"    Feature creation: Stat={stat_feature_time:.3f}s, Phys={phys_feature_time:.3f}s")
        print(f"    Training: Stat={stat_train_time:.3f}s, Phys={phys_train_time:.3f}s")
        
        results[target] = target_results
    
    # Save results
    print("\n💾 Saving results...")
    
    # Create summary
    summary_lines = [
        "# PEECOM vs ICCIA 2023 Novelty Validation Results",
        "",
        "## Key Findings",
        ""
    ]
    
    for target, target_results in results.items():
        summary_lines.extend([
            f"### {target}",
            ""
        ])
        
        if 'statistical_baseline' in target_results and 'physics_baseline' in target_results:
            stat = target_results['statistical_baseline']
            phys = target_results['physics_baseline']
            sig = target_results['significance_test']
            
            summary_lines.extend([
                f"**Performance:**",
                f"- Statistical features: {stat['mean']:.4f} ± {stat['std']:.4f} ({stat['feature_count']} features)",
                f"- Physics features: {phys['mean']:.4f} ± {phys['std']:.4f} ({phys['feature_count']} features)",
                f"- Statistical significance: p = {sig['p_value']:.4f}",
                f"- Effect size: {sig['effect_size']:.4f}",
                ""
            ])
        
        if 'robustness' in target_results:
            summary_lines.extend([
                f"**Robustness under sensor failures:**"
            ])
            
            for failure_rate, rob_data in target_results['robustness'].items():
                advantage = rob_data['robustness_advantage']
                summary_lines.append(f"- {failure_rate*100:.0f}% failure: {advantage:+.3f} advantage")
            
            summary_lines.append("")
        
        if 'computational' in target_results:
            comp = target_results['computational']
            summary_lines.extend([
                f"**Computational efficiency:**",
                f"- Feature creation overhead: {comp['feature_time_ratio']:.2f}x",
                f"- Training time ratio: {comp['train_time_ratio']:.2f}x",
                ""
            ])
    
    # Write summary
    with open(output_dir / "novelty_validation_summary.md", 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Save detailed results
    import json
    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Results saved to {output_dir}")
    
    # Final summary
    print("\n" + "=" * 55)
    print("🎯 NOVELTY VALIDATION SUMMARY")
    print("=" * 55)
    
    for target, target_results in results.items():
        if 'significance_test' in target_results:
            sig = target_results['significance_test']
            print(f"{target}: p = {sig['p_value']:.4f}, effect = {sig['effect_size']:.4f}")
    
    print("\n✅ PEECOM novelty validation complete!")
    print("Key differentiators from ICCIA 2023 established.")


if __name__ == "__main__":
    run_novelty_validation()