#!/usr/bin/env python3
"""
EXPERIMENT C: FEATURE SEPARABILITY RANKING + ABLATION
=====================================================
Purpose: Identify which features contribute most to block-level separability
         and test how removing them affects performance.

Methodology:
1. Compute Cohen's d for each feature across blocks (higher = more block-separating)
2. Rank features by separability score
3. Run ablation series: remove top-K separable features (K ∈ {1,2,5,10,20})
4. Re-evaluate with RF + SimplePEECOM + EnhancedPEECOM under synchronized CV
5. Generate ablation curve showing accuracy vs. features removed

Expected Outcomes:
- If pure leakage: removing top separable features should improve generalization
- If mixed signal: removing features should hurt performance initially, then plateau
- If genuine signal: removing features should consistently hurt performance

Files Created:
- output/exp_c_ablation/feature_separability_ranking.csv
- output/exp_c_ablation/ablation_K_{K}_results.joblib
- output/exp_c_ablation/ablation_curve.png
- output/exp_c_ablation/ablation_summary_table.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

# Import PEECOM variants
import sys
sys.path.append('src')
from models.simple_peecom import SimplePEECOM
from models.enhanced_peecom import EnhancedPEECOM

def setup_directories():
    """Create output directories for Experiment C"""
    base_dir = Path("output/exp_c_ablation")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def load_data():
    """Load data and analyze block structure (same as previous experiments)"""
    try:
        data = pd.read_csv('hydraulic_data_processed.csv')
        print(f"   ✅ Real data loaded: {data.shape}")
    except:
        print("   ⚠️ Using synthetic data for demonstration")
        np.random.seed(42)
        n_samples = 2205
        n_features = 54
        data = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f'f{i}' for i in range(n_features)])
        
        # Create perfect block-class segregation
        data['target'] = 0
        data.loc[:731, 'target'] = 0    # Block 0: Class 0
        data.loc[732:1463, 'target'] = 1  # Block 1: Class 1  
        data.loc[1464:, 'target'] = 2   # Block 2: Class 2
    
    return data

def identify_blocks(data):
    """Identify block boundaries"""
    target = data['target'].values
    transitions = np.where(np.diff(target) != 0)[0] + 1
    block_starts = np.concatenate([[0], transitions, [len(data)]])
    
    blocks = []
    for i in range(len(block_starts) - 1):
        start_idx = block_starts[i]
        end_idx = block_starts[i + 1]
        original_class = target[start_idx]
        blocks.append({
            'block_id': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'class': original_class,
            'size': end_idx - start_idx
        })
    
    return blocks

def create_synchronized_cv_splits(data, blocks, k_folds=3):
    """Create synchronized chunk CV splits"""
    all_chunks = []
    
    for block in blocks:
        chunk_size = block['size'] // k_folds
        embargo_size = max(1, int(chunk_size * 0.02))
        
        for fold in range(k_folds):
            chunk_start = block['start_idx'] + fold * chunk_size
            chunk_end = min(block['start_idx'] + (fold + 1) * chunk_size, block['end_idx'])
            
            embargo_start = max(block['start_idx'], chunk_start - embargo_size)
            embargo_end = min(chunk_end + embargo_size, block['end_idx'])
            
            all_chunks.append({
                'block_id': block['block_id'],
                'fold': fold,
                'test_start': chunk_start,
                'test_end': chunk_end,
                'embargo_start': embargo_start,
                'embargo_end': embargo_end,
                'test_size': chunk_end - chunk_start
            })
    
    cv_splits = []
    total_samples = len(data)
    
    for fold in range(k_folds):
        test_chunks = [chunk for chunk in all_chunks if chunk['fold'] == fold]
        test_indices = []
        embargo_indices = set()
        
        for chunk in test_chunks:
            test_indices.extend(range(chunk['test_start'], chunk['test_end']))
            embargo_indices.update(range(chunk['embargo_start'], chunk['embargo_end']))
        
        train_indices = list(set(range(total_samples)) - embargo_indices)
        
        cv_splits.append({
            'fold': fold,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'embargo_size': len(embargo_indices) - len(test_indices)
        })
    
    return cv_splits

def compute_feature_separability(data, blocks):
    """Compute Cohen's d for each feature across blocks"""
    print("   🔍 Computing feature separability (Cohen's d across blocks)...")
    
    feature_cols = [col for col in data.columns if col != 'target']
    separability_scores = {}
    
    for feature in feature_cols:
        feature_values = data[feature].values
        
        # Compute pairwise Cohen's d between all block pairs
        cohens_d_values = []
        
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                block_i_mask = (
                    (data.index >= blocks[i]['start_idx']) & 
                    (data.index < blocks[i]['end_idx'])
                )
                block_j_mask = (
                    (data.index >= blocks[j]['start_idx']) & 
                    (data.index < blocks[j]['end_idx'])
                )
                
                values_i = feature_values[block_i_mask]
                values_j = feature_values[block_j_mask]
                
                # Cohen's d calculation
                mean_i = np.mean(values_i)
                mean_j = np.mean(values_j)
                std_i = np.std(values_i, ddof=1)
                std_j = np.std(values_j, ddof=1)
                
                # Pooled standard deviation
                n_i, n_j = len(values_i), len(values_j)
                pooled_std = np.sqrt(((n_i - 1) * std_i**2 + (n_j - 1) * std_j**2) / (n_i + n_j - 2))
                
                if pooled_std > 0:
                    cohens_d = abs(mean_i - mean_j) / pooled_std
                else:
                    cohens_d = 0.0
                
                cohens_d_values.append(cohens_d)
        
        # Average Cohen's d across all block pairs
        separability_scores[feature] = np.mean(cohens_d_values)
    
    # Create ranking DataFrame
    separability_df = pd.DataFrame([
        {'feature': feature, 'cohens_d': score}
        for feature, score in separability_scores.items()
    ]).sort_values('cohens_d', ascending=False).reset_index(drop=True)
    
    separability_df['rank'] = range(1, len(separability_df) + 1)
    
    print(f"   ✅ Computed separability for {len(separability_df)} features")
    print(f"   📊 Top 5 most separable features:")
    for _, row in separability_df.head().iterrows():
        print(f"      {row['rank']:2d}. {row['feature']}: Cohen's d = {row['cohens_d']:.4f}")
    
    return separability_df

def extract_peecom_features(model, X_raw):
    """Extract engineered features from PEECOM models"""
    if hasattr(model, 'physics_enhancer') and hasattr(model.physics_enhancer, '_create_physics_features'):
        return model.physics_enhancer._create_physics_features(X_raw)
    elif hasattr(model, '_create_physics_features'):
        return model._create_physics_features(X_raw)
    elif hasattr(model, '_create_advanced_physics_features'):
        return model._create_advanced_physics_features(X_raw)
    else:
        return pd.DataFrame(X_raw)

def evaluate_model_cv_with_ablation(model_name, model_constructor, data, cv_splits, features_to_remove=None):
    """Run cross-validation with optional feature ablation"""
    fold_results = []
    
    for split in cv_splits:
        train_idx = split['train_indices']
        test_idx = split['test_indices']
        y_train = data.iloc[train_idx]['target'].values
        y_test = data.iloc[test_idx]['target'].values
        
        if model_name in ['SimplePEECOM', 'EnhancedPEECOM']:
            # PEECOM models: use raw features, then extract engineered features
            X_train_raw = data.iloc[train_idx].drop('target', axis=1).values
            X_test_raw = data.iloc[test_idx].drop('target', axis=1).values
            
            # Fit model to extract features
            model = model_constructor()
            model.fit(X_train_raw, y_train)
            
            # Extract engineered features
            train_features_df = extract_peecom_features(model, X_train_raw)
            test_features_df = extract_peecom_features(model, X_test_raw)
            
            # Apply feature ablation if specified
            if features_to_remove:
                available_features = [f for f in features_to_remove if f in train_features_df.columns]
                if available_features:
                    train_features_df = train_features_df.drop(columns=available_features)
                    test_features_df = test_features_df.drop(columns=available_features)
            
            # Scale and train final model
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_features_df)
            X_test = scaler.transform(test_features_df)
            
            # Use RandomForest as proxy for final classification
            final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            final_model.fit(X_train, y_train)
            y_pred = final_model.predict(X_test)
            
        else:
            # Standard models: use raw features directly
            feature_cols = [col for col in data.columns if col != 'target']
            
            # Apply feature ablation if specified
            if features_to_remove:
                available_features = [f for f in features_to_remove if f in feature_cols]
                remaining_features = [f for f in feature_cols if f not in available_features]
            else:
                remaining_features = feature_cols
            
            # Scale and train
            scaler = StandardScaler()
            X_train = scaler.fit_transform(data.iloc[train_idx][remaining_features])
            X_test = scaler.transform(data.iloc[test_idx][remaining_features])
            
            model = model_constructor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        fold_results.append({
            'fold': split['fold'],
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'n_features_used': X_train.shape[1]
        })
    
    mean_accuracy = np.mean([f['accuracy'] for f in fold_results])
    std_accuracy = np.std([f['accuracy'] for f in fold_results])
    mean_features = np.mean([f['n_features_used'] for f in fold_results])
    
    return {
        'model': model_name,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_features_used': mean_features,
        'fold_results': fold_results
    }

def run_ablation_experiment(data, cv_splits, separability_df, k_values, base_dir):
    """Run ablation experiment for different values of K"""
    print(f"\n🔄 ABLATION EXPERIMENTS")
    
    # Models to test
    models = {
        'RandomForest': lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'SimplePEECOM': lambda: SimplePEECOM(),
        'EnhancedPEECOM': lambda: EnhancedPEECOM()
    }
    
    ablation_results = []
    
    # Baseline (no features removed)
    print("   📊 Baseline (K=0, no features removed):")
    baseline_results = {}
    for model_name, model_constructor in models.items():
        result = evaluate_model_cv_with_ablation(model_name, model_constructor, data, cv_splits)
        baseline_results[model_name] = result
        print(f"      {model_name}: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f} ({result['mean_features_used']:.0f} features)")
    
    # Save baseline
    joblib.dump(baseline_results, base_dir / 'ablation_K_0_baseline_results.joblib')
    
    # Ablation series
    for k in k_values:
        print(f"\n   🗑️ Ablation K={k} (remove {k} most separable features):")
        
        # Get top-K most separable features to remove
        features_to_remove = separability_df.head(k)['feature'].tolist()
        print(f"      Removing features: {features_to_remove}")
        
        k_results = {}
        for model_name, model_constructor in models.items():
            result = evaluate_model_cv_with_ablation(
                model_name, model_constructor, data, cv_splits, features_to_remove
            )
            k_results[model_name] = result
            
            baseline_acc = baseline_results[model_name]['mean_accuracy']
            delta = result['mean_accuracy'] - baseline_acc
            
            print(f"      {model_name}: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f} "
                  f"({result['mean_features_used']:.0f} features, Δ{delta:+.4f})")
        
        # Save K-specific results
        k_results['k'] = k
        k_results['features_removed'] = features_to_remove
        joblib.dump(k_results, base_dir / f'ablation_K_{k}_results.joblib')
        
        # Compile for summary
        for model_name in models.keys():
            ablation_results.append({
                'k': k,
                'model': model_name,
                'features_removed': k,
                'features_remaining': int(k_results[model_name]['mean_features_used']),
                'mean_accuracy': k_results[model_name]['mean_accuracy'],
                'std_accuracy': k_results[model_name]['std_accuracy'],
                'delta_from_baseline': k_results[model_name]['mean_accuracy'] - baseline_results[model_name]['mean_accuracy'],
                'removed_feature_list': features_to_remove
            })
    
    return baseline_results, ablation_results

def create_ablation_visualizations(baseline_results, ablation_results, separability_df, base_dir):
    """Create ablation curve and analysis plots"""
    print(f"\n📈 CREATING VISUALIZATIONS")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment C: Feature Separability & Ablation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Feature separability ranking (top 20)
    ax1 = axes[0, 0]
    top_features = separability_df.head(20)
    bars = ax1.barh(range(len(top_features)), top_features['cohens_d'], color='lightcoral', alpha=0.7)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels([f"{row['feature'][:8]}..." if len(row['feature']) > 8 else row['feature'] 
                        for _, row in top_features.iterrows()], fontsize=8)
    ax1.set_xlabel("Cohen's d (Block Separability)")
    ax1.set_title('Top 20 Most Block-Separable Features')
    ax1.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=7)
    
    # 2. Ablation curves
    ax2 = axes[0, 1]
    ablation_df = pd.DataFrame(ablation_results)
    
    for model_name in ['RandomForest', 'SimplePEECOM', 'EnhancedPEECOM']:
        model_data = ablation_df[ablation_df['model'] == model_name].sort_values('k')
        k_values = model_data['k'].values
        accuracies = model_data['mean_accuracy'].values
        std_errors = model_data['std_accuracy'].values
        
        # Add baseline point (K=0)
        baseline_acc = baseline_results[model_name]['mean_accuracy']
        baseline_std = baseline_results[model_name]['std_accuracy']
        
        k_plot = np.concatenate([[0], k_values])
        acc_plot = np.concatenate([[baseline_acc], accuracies])
        std_plot = np.concatenate([[baseline_std], std_errors])
        
        ax2.errorbar(k_plot, acc_plot, yerr=std_plot, marker='o', label=model_name, linewidth=2)
    
    ax2.axhline(1/3, color='red', linestyle='--', alpha=0.7, label='Chance Level (0.333)')
    ax2.set_xlabel('Number of Features Removed (K)')
    ax2.set_ylabel('Mean Accuracy')
    ax2.set_title('Ablation Curves: Accuracy vs Features Removed')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Delta from baseline
    ax3 = axes[1, 0]
    
    for model_name in ['RandomForest', 'SimplePEECOM', 'EnhancedPEECOM']:
        model_data = ablation_df[ablation_df['model'] == model_name].sort_values('k')
        k_values = model_data['k'].values
        deltas = model_data['delta_from_baseline'].values
        
        ax3.plot(k_values, deltas, marker='o', label=model_name, linewidth=2)
    
    ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Number of Features Removed (K)')
    ax3.set_ylabel('Δ Accuracy from Baseline')
    ax3.set_title('Performance Change from Baseline')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature importance heatmap (top 10 features across models)
    ax4 = axes[1, 1]
    top_10_features = separability_df.head(10)['feature'].tolist()
    
    # Create heatmap data
    heatmap_data = []
    for feature in top_10_features:
        heatmap_data.append([separability_df[separability_df['feature'] == feature]['cohens_d'].iloc[0]])
    
    im = ax4.imshow(heatmap_data, cmap='Reds', aspect='auto')
    ax4.set_yticks(range(len(top_10_features)))
    ax4.set_yticklabels([f[:10] + '...' if len(f) > 10 else f for f in top_10_features])
    ax4.set_xticks([0])
    ax4.set_xticklabels(['Cohen\'s d'])
    ax4.set_title('Top 10 Feature Separability Heatmap')
    
    # Add text annotations
    for i in range(len(top_10_features)):
        ax4.text(0, i, f'{heatmap_data[i][0]:.3f}', ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plot_path = base_dir / 'ablation_curve.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Visualization saved: {plot_path}")

def create_summary_table(baseline_results, ablation_results, base_dir):
    """Create comprehensive summary table"""
    print(f"   📊 Creating summary table...")
    
    # Compile all results into comprehensive table
    summary_rows = []
    
    # Add baseline rows
    for model_name in ['RandomForest', 'SimplePEECOM', 'EnhancedPEECOM']:
        baseline = baseline_results[model_name]
        summary_rows.append({
            'K': 0,
            'features_removed': 0,
            'features_remaining': int(baseline['mean_features_used']),
            'model': model_name,
            'mean_accuracy': baseline['mean_accuracy'],
            'std_accuracy': baseline['std_accuracy'],
            'delta_from_baseline': 0.0,
            'removed_features': 'None'
        })
    
    # Add ablation rows
    for result in ablation_results:
        summary_rows.append({
            'K': result['k'],
            'features_removed': result['features_removed'],
            'features_remaining': result['features_remaining'],
            'model': result['model'],
            'mean_accuracy': result['mean_accuracy'],
            'std_accuracy': result['std_accuracy'],
            'delta_from_baseline': result['delta_from_baseline'],
            'removed_features': ', '.join(result['removed_feature_list'][:3]) + '...' if len(result['removed_feature_list']) > 3 else ', '.join(result['removed_feature_list'])
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary table
    table_path = base_dir / 'ablation_summary_table.csv'
    summary_df.to_csv(table_path, index=False)
    print(f"   ✅ Summary table saved: {table_path}")
    
    return summary_df

def main():
    print("🧪 EXPERIMENT C: FEATURE SEPARABILITY RANKING + ABLATION")
    print("=" * 70)
    print("Purpose: Identify which features drive block separability")
    print("Method:  Cohen's d ranking + ablation series (K ∈ {1,2,5,10,20})")
    print("Models:  RandomForest + SimplePEECOM + EnhancedPEECOM")
    print("=" * 70)
    
    # Setup
    base_dir = setup_directories()
    print(f"\n📁 Output directory: {base_dir}")
    
    # Load data
    print("\n📊 LOADING DATA")
    data = load_data()
    blocks = identify_blocks(data)
    
    # Create CV splits
    print(f"\n📂 CREATING SYNCHRONIZED CV SPLITS")
    cv_splits = create_synchronized_cv_splits(data, blocks, k_folds=3)
    print(f"   ✅ Created {len(cv_splits)} CV folds")
    
    # Compute feature separability
    print(f"\n🔍 FEATURE SEPARABILITY ANALYSIS")
    separability_df = compute_feature_separability(data, blocks)
    
    # Save separability ranking
    ranking_path = base_dir / 'feature_separability_ranking.csv'
    separability_df.to_csv(ranking_path, index=False)
    print(f"   ✅ Feature ranking saved: {ranking_path}")
    
    # Run ablation experiments
    k_values = [1, 2, 5, 10, 20]
    baseline_results, ablation_results = run_ablation_experiment(
        data, cv_splits, separability_df, k_values, base_dir
    )
    
    # Create visualizations
    create_ablation_visualizations(baseline_results, ablation_results, separability_df, base_dir)
    
    # Create summary table
    summary_df = create_summary_table(baseline_results, ablation_results, base_dir)
    
    # Analysis and interpretation
    print(f"\n📋 ANALYSIS & INTERPRETATION")
    print("=" * 50)
    
    # Find most separable features
    top_5_features = separability_df.head()
    print("📊 Top 5 Most Block-Separable Features:")
    for _, row in top_5_features.iterrows():
        print(f"   {row['rank']:2d}. {row['feature']}: Cohen's d = {row['cohens_d']:.4f}")
    
    # Analyze ablation trends
    print(f"\n🔄 Ablation Trends:")
    for model_name in ['RandomForest', 'SimplePEECOM', 'EnhancedPEECOM']:
        model_results = [r for r in ablation_results if r['model'] == model_name]
        baseline_acc = baseline_results[model_name]['mean_accuracy']
        
        print(f"\n   {model_name}:")
        print(f"      Baseline: {baseline_acc:.4f}")
        
        trend_direction = []
        for result in sorted(model_results, key=lambda x: x['k']):
            delta = result['delta_from_baseline']
            trend_direction.append(delta)
            print(f"      K={result['k']:2d}: {result['mean_accuracy']:.4f} (Δ{delta:+.4f})")
        
        # Interpret trend
        if all(d >= -0.01 for d in trend_direction):  # No significant drops
            interpretation = "🔴 LEAKAGE: Performance maintained despite removing separable features"
        elif all(d <= -0.01 for d in trend_direction):  # Consistent drops
            interpretation = "🟢 GENUINE: Performance drops as expected when removing informative features"
        else:
            interpretation = "🟡 MIXED: Complex interaction between leakage and genuine signal"
        
        print(f"      Interpretation: {interpretation}")
    
    # Overall conclusion
    print(f"\n🏁 OVERALL CONCLUSION")
    print("=" * 30)
    
    # Check if removing most separable features improves or maintains performance
    rf_k20 = next(r for r in ablation_results if r['model'] == 'RandomForest' and r['k'] == 20)
    peecom_k20 = next(r for r in ablation_results if r['model'] == 'SimplePEECOM' and r['k'] == 20)
    
    if (rf_k20['delta_from_baseline'] >= -0.02 and peecom_k20['delta_from_baseline'] >= -0.02):
        conclusion = "🔴 STRONG LEAKAGE: Removing most block-separable features doesn't hurt performance"
    else:
        conclusion = "🟢 GENUINE SIGNAL: Performance degrades when removing informative features"
    
    print(conclusion)
    
    print(f"\n📁 FILES CREATED:")
    print(f"   - feature_separability_ranking.csv")
    print(f"   - ablation_K_{{K}}_results.joblib (K=0,1,2,5,10,20)")
    print(f"   - ablation_curve.png")
    print(f"   - ablation_summary_table.csv")
    
    print(f"\n✅ EXPERIMENT C COMPLETE")

if __name__ == '__main__':
    main()