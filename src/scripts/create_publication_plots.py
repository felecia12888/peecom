#!/usr/bin/env python3
"""
Publication-Quality Specialized Plots

Create specialized visualizations for manuscript:
1. Head-to-head PEECOM vs MCF comparison
2. Method evolution/progression plot
3. Performance distribution analysis
4. Target difficulty ranking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths and style
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "output" / "reports"
FIGURES_DIR = ROOT / "output" / "figures"

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_comprehensive_results():
    """Load the comprehensive comparison results"""
    
    stats_file = RESULTS_DIR / "comprehensive_model_statistics.csv"
    target_file = RESULTS_DIR / "target_specific_performance.csv"
    
    if not stats_file.exists() or not target_file.exists():
        print("❌ Comprehensive results not found. Run comprehensive_final_comparison.py first.")
        return None, None
        
    stats_df = pd.read_csv(stats_file)
    target_df = pd.read_csv(target_file)
    
    print(f"✅ Loaded comprehensive results")
    return stats_df, target_df

def create_head_to_head_mcf_plot(stats_df, target_df):
    """Create head-to-head PEECOM vs MCF comparison plot"""
    
    print("🎯 Creating head-to-head PEECOM vs MCF comparison...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel A: Best methods overall comparison
    best_methods = stats_df.nlargest(10, 'accuracy_mean')
    
    colors = []
    for model in best_methods['model']:
        if 'PEECOM' in model:
            colors.append('#F18F01')  # Orange for PEECOM
        elif 'MCF' in model:
            colors.append('#A23B72')  # Purple for MCF
        else:
            colors.append('#2E86AB')  # Blue for Baseline
    
    bars = ax1.barh(range(len(best_methods)), best_methods['accuracy_mean'], 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_yticks(range(len(best_methods)))
    ax1.set_yticklabels(best_methods['model'], fontsize=10)
    ax1.set_xlabel('Accuracy', fontsize=12)
    ax1.set_title('A) Top 10 Methods Overall Performance', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, best_methods['accuracy_mean'])):
        ax1.text(acc + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{acc:.3f}', va='center', fontsize=9)
    
    # Panel B: PEECOM vs MCF category comparison
    peecom_methods = stats_df[stats_df['model'].str.contains('PEECOM')]
    mcf_methods = stats_df[stats_df['model'].str.contains('MCF')]
    
    peecom_mean = peecom_methods['accuracy_mean'].mean()
    mcf_mean = mcf_methods['accuracy_mean'].mean()
    peecom_std = peecom_methods['accuracy_mean'].std()
    mcf_std = mcf_methods['accuracy_mean'].std()
    
    categories = ['MCF Methods', 'PEECOM Methods']
    means = [mcf_mean, peecom_mean]
    stds = [mcf_std, peecom_std]
    colors_cat = ['#A23B72', '#F18F01']
    
    bars2 = ax2.bar(categories, means, yerr=stds, color=colors_cat, alpha=0.8,
                    edgecolor='black', linewidth=1, capsize=5)
    
    ax2.set_ylabel('Average Accuracy', fontsize=12)
    ax2.set_title('B) Method Category Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 0.8)
    
    # Add value labels and improvement
    for bar, mean, std in zip(bars2, means, stds):
        ax2.text(bar.get_x() + bar.get_width()/2, mean + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Add improvement arrow and text
    improvement = ((peecom_mean - mcf_mean) / mcf_mean) * 100
    ax2.annotate(f'+{improvement:.1f}%', xy=(0.5, max(means) + max(stds) + 0.03),
                ha='center', fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Panel C: Target-specific best method comparison
    target_names = ['Cooler', 'Valve', 'Pump', 'Accumulator', 'Stable']
    targets_orig = ['cooler_condition', 'valve_condition', 'pump_leakage', 
                   'accumulator_pressure', 'stable_flag']
    
    peecom_best = []
    mcf_best = []
    
    for target in targets_orig:
        target_data = target_df[target_df['target'] == target]
        peecom_data = target_data[target_data['model'].str.contains('PEECOM')]
        mcf_data = target_data[target_data['model'].str.contains('MCF')]
        
        peecom_best.append(peecom_data['acc_mean'].max() if len(peecom_data) > 0 else 0)
        mcf_best.append(mcf_data['acc_mean'].max() if len(mcf_data) > 0 else 0)
    
    x = np.arange(len(target_names))
    width = 0.35
    
    bars3 = ax3.bar(x - width/2, mcf_best, width, label='Best MCF', 
                    color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars4 = ax3.bar(x + width/2, peecom_best, width, label='Best PEECOM',
                    color='#F18F01', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax3.set_xlabel('Diagnostic Targets', fontsize=12)
    ax3.set_ylabel('Best Accuracy', fontsize=12)
    ax3.set_title('C) Target-Specific Best Method Performance', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(target_names)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Panel D: Performance distribution comparison
    peecom_accs = stats_df[stats_df['model'].str.contains('PEECOM')]['accuracy_mean']
    mcf_accs = stats_df[stats_df['model'].str.contains('MCF')]['accuracy_mean']
    baseline_accs = stats_df[~stats_df['model'].str.contains('PEECOM|MCF')]['accuracy_mean']
    
    # Create violin plot
    data_to_plot = [baseline_accs, mcf_accs, peecom_accs]
    parts = ax4.violinplot(data_to_plot, positions=[1, 2, 3], widths=0.6)
    
    # Color the violins
    colors_violin = ['#2E86AB', '#A23B72', '#F18F01']
    for pc, color in zip(parts['bodies'], colors_violin):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['Baseline', 'MCF', 'PEECOM'])
    ax4.set_ylabel('Accuracy Distribution', fontsize=12)
    ax4.set_title('D) Performance Distribution by Method Type', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = FIGURES_DIR / "head_to_head_mcf_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "head_to_head_mcf_comparison.pdf", bbox_inches='tight')
    
    print(f"✅ Saved head-to-head comparison: {plot_file}")
    plt.show()

def create_method_evolution_plot(stats_df):
    """Create method evolution/progression visualization"""
    
    print("🚀 Creating method evolution plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel A: Method progression timeline
    method_groups = {
        'Traditional ML': ['svm', 'logistic_regression'],
        'Tree-Based': ['random_forest', 'gradient_boosting'], 
        'MCF Individual': ['MCF_KNN', 'MCF_SVM', 'MCF_XGBoost', 'MCF_RandomForest'],
        'MCF Fusion': ['MCF_Stacking', 'MCF_Bayesian', 'MCF_DempsterShafer'],
        'PEECOM': ['PEECOM_Base', 'PEECOM_Enhanced', 'PEECOM_Optimized', 'PEECOM_Full']
    }
    
    group_means = []
    group_stds = []
    group_names = []
    
    for group_name, models in method_groups.items():
        group_data = stats_df[stats_df['model'].isin(models)]
        if len(group_data) > 0:
            group_means.append(group_data['accuracy_mean'].mean())
            group_stds.append(group_data['accuracy_mean'].std())
            group_names.append(group_name)
    
    # Create progression plot with connecting line
    x_pos = range(len(group_names))
    ax1.plot(x_pos, group_means, 'o-', linewidth=3, markersize=10, 
             color='#2E86AB', markerfacecolor='#F18F01', markeredgecolor='black', markeredgewidth=2)
    ax1.errorbar(x_pos, group_means, yerr=group_stds, fmt='none', 
                color='black', capsize=5, capthick=2)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(group_names, rotation=45, ha='right')
    ax1.set_ylabel('Average Accuracy', fontsize=12)
    ax1.set_title('A) Method Evolution and Performance Progression', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(group_means, group_stds)):
        ax1.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    # Panel B: PEECOM variant progression
    peecom_variants = ['PEECOM_Base', 'PEECOM_Enhanced', 'PEECOM_Optimized', 'PEECOM_Full']
    peecom_data = stats_df[stats_df['model'].isin(peecom_variants)]
    peecom_data = peecom_data.sort_values('accuracy_mean')
    
    bars = ax2.bar(range(len(peecom_data)), peecom_data['accuracy_mean'],
                   color=['#FFE5B4', '#FFD700', '#FFA500', '#FF8C00'], 
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.errorbar(range(len(peecom_data)), peecom_data['accuracy_mean'],
                yerr=peecom_data['accuracy_std'], fmt='none',
                color='black', capsize=3)
    
    ax2.set_xticks(range(len(peecom_data)))
    ax2.set_xticklabels(peecom_data['model'], rotation=45, ha='right')
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('B) PEECOM Variant Performance Progression', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels and improvements
    for i, (bar, acc, std) in enumerate(zip(bars, peecom_data['accuracy_mean'], peecom_data['accuracy_std'])):
        ax2.text(bar.get_x() + bar.get_width()/2, acc + std + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add improvement percentage from base
        if i > 0:
            base_acc = peecom_data['accuracy_mean'].iloc[0]
            improvement = ((acc - base_acc) / base_acc) * 100
            ax2.text(bar.get_x() + bar.get_width()/2, acc/2,
                    f'+{improvement:.1f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='darkred')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = FIGURES_DIR / "method_evolution_progression.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "method_evolution_progression.pdf", bbox_inches='tight')
    
    print(f"✅ Saved evolution plot: {plot_file}")
    plt.show()

def create_target_difficulty_ranking(target_df):
    """Create target difficulty ranking visualization"""
    
    print("🎯 Creating target difficulty ranking...")
    
    # Calculate difficulty metrics
    target_difficulty = []
    
    for target in target_df['target'].unique():
        target_data = target_df[target_df['target'] == target]
        
        # Overall performance across all methods
        avg_acc = target_data['acc_mean'].mean()
        std_acc = target_data['acc_mean'].std()
        max_acc = target_data['acc_mean'].max()
        min_acc = target_data['acc_mean'].min()
        
        # Difficulty score (lower accuracy = higher difficulty)
        difficulty_score = 1 - avg_acc
        
        target_difficulty.append({
            'target': target,
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'max_accuracy': max_acc,
            'min_accuracy': min_acc,
            'difficulty_score': difficulty_score,
            'performance_range': max_acc - min_acc
        })
    
    diff_df = pd.DataFrame(target_difficulty)
    diff_df = diff_df.sort_values('difficulty_score')  # Easiest to hardest
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel A: Target difficulty ranking
    target_names = [t.replace('_', ' ').title() for t in diff_df['target']]
    colors = plt.cm.RdYlGn_r(diff_df['difficulty_score'] / diff_df['difficulty_score'].max())
    
    bars = ax1.barh(range(len(diff_df)), diff_df['avg_accuracy'], 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_yticks(range(len(diff_df)))
    ax1.set_yticklabels(target_names)
    ax1.set_xlabel('Average Accuracy Across All Methods', fontsize=12)
    ax1.set_title('A) Diagnostic Target Difficulty Ranking', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add difficulty labels
    difficulty_labels = ['EASY', 'MODERATE', 'HARD', 'VERY HARD', 'EXTREMELY HARD']
    for i, (bar, acc, diff_score) in enumerate(zip(bars, diff_df['avg_accuracy'], diff_df['difficulty_score'])):
        difficulty_idx = min(int(diff_score * len(difficulty_labels)), len(difficulty_labels) - 1)
        difficulty = difficulty_labels[difficulty_idx]
        
        ax1.text(acc + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.3f} ({difficulty})', va='center', fontsize=10, fontweight='bold')
    
    # Panel B: Performance variability analysis
    bars2 = ax2.bar(range(len(diff_df)), diff_df['performance_range'],
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_xticks(range(len(diff_df)))
    ax2.set_xticklabels(target_names, rotation=45, ha='right')
    ax2.set_ylabel('Performance Range (Max - Min Accuracy)', fontsize=12)
    ax2.set_title('B) Method Performance Variability by Target', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, range_val in zip(bars2, diff_df['performance_range']):
        ax2.text(bar.get_x() + bar.get_width()/2, range_val + 0.005,
                f'{range_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = FIGURES_DIR / "target_difficulty_ranking.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "target_difficulty_ranking.pdf", bbox_inches='tight')
    
    print(f"✅ Saved target difficulty plot: {plot_file}")
    
    # Save difficulty analysis
    diff_file = RESULTS_DIR / "target_difficulty_analysis.csv"
    diff_df.to_csv(diff_file, index=False)
    print(f"✅ Saved difficulty analysis: {diff_file}")
    
    plt.show()
    
    return diff_df

def main():
    """Main specialized plots workflow"""
    
    print("📊 PUBLICATION-QUALITY SPECIALIZED PLOTS")
    print("=" * 50)
    
    # Load data
    stats_df, target_df = load_comprehensive_results()
    if stats_df is None:
        return
    
    # Create specialized plots
    create_head_to_head_mcf_plot(stats_df, target_df)
    create_method_evolution_plot(stats_df)
    difficulty_df = create_target_difficulty_ranking(target_df)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 SPECIALIZED PLOTS COMPLETE")
    print("=" * 50)
    print("✅ Head-to-head PEECOM vs MCF comparison")
    print("✅ Method evolution and progression analysis")
    print("✅ Target difficulty ranking and variability")
    print("✅ All plots saved in PNG and PDF formats")
    
    print(f"\n📊 Key insights:")
    best_peecom = stats_df[stats_df['model'].str.contains('PEECOM')]['accuracy_mean'].max()
    best_mcf = stats_df[stats_df['model'].str.contains('MCF')]['accuracy_mean'].max()
    avg_improvement = ((stats_df[stats_df['model'].str.contains('PEECOM')]['accuracy_mean'].mean() - 
                       stats_df[stats_df['model'].str.contains('MCF')]['accuracy_mean'].mean()) / 
                      stats_df[stats_df['model'].str.contains('MCF')]['accuracy_mean'].mean()) * 100
    
    print(f"  PEECOM vs MCF improvement: {avg_improvement:.1f}% average")
    print(f"  Best PEECOM: {best_peecom:.3f}")
    print(f"  Best MCF: {best_mcf:.3f}")
    
    easiest_target = difficulty_df.iloc[0]['target'].replace('_', ' ').title()
    hardest_target = difficulty_df.iloc[-1]['target'].replace('_', ' ').title()
    print(f"  Easiest target: {easiest_target} ({difficulty_df.iloc[0]['avg_accuracy']:.3f})")
    print(f"  Hardest target: {hardest_target} ({difficulty_df.iloc[-1]['avg_accuracy']:.3f})")

if __name__ == "__main__":
    main()