#!/usr/bin/env python3
"""
Comprehensive Performance Metrics Dashboard
===========================================

Creates publication-quality tables and visualizations showing:
- F1-Score, Precision, Recall, R²
- Cross-validation statistics
- Model ranking and statistical significance
- Sophisticated scientific styling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set sophisticated scientific style
plt.style.use('default')
sns.set_context("paper", font_scale=1.1)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True
})

class ComprehensiveMetricsDashboard:
    """Create comprehensive performance metrics dashboard"""
    
    def __init__(self, output_dir="output/figures/comprehensive_metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load enhanced performance data
        self.df = self.load_enhanced_data()
        
        # Scientific color palette
        self.colors = {
            'peecom': '#1f77b4',           # Professional blue
            'random_forest': '#ff7f0e',   # Orange
            'logistic_regression': '#2ca02c', # Green
            'svm': '#d62728'              # Red
        }
        
        self.dataset_names = {
            'motorvd': 'Motor Vibration Analysis',
            'cmohs': 'CMOHS Hydraulic System'
        }
        
        self.model_names = {
            'peecom': 'PEECOM',
            'random_forest': 'Random Forest',
            'logistic_regression': 'Logistic Regression',
            'svm': 'Support Vector Machine'
        }
        
    def load_enhanced_data(self):
        """Load enhanced performance data"""
        
        # Try to load the enhanced summary first
        summary_file = Path("output/enhanced_performance_summary.csv")
        if summary_file.exists():
            return pd.read_csv(summary_file)
        
        # Otherwise, load from enhanced results files
        results_data = []
        models_dir = Path("output/models")
        
        for dataset in ['motorvd', 'cmohs']:
            dataset_dir = models_dir / dataset
            if not dataset_dir.exists():
                continue
                
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                for target_dir in model_dir.iterdir():
                    if not target_dir.is_dir():
                        continue
                    
                    enhanced_file = target_dir / "enhanced_training_results.json"
                    regular_file = target_dir / "training_results.json"
                    
                    results_file = enhanced_file if enhanced_file.exists() else regular_file
                    
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                data = json.load(f)
                            
                            result = {
                                'dataset': dataset,
                                'model': model_dir.name,
                                'target': target_dir.name,
                                **data
                            }
                            results_data.append(result)
                            
                        except Exception as e:
                            print(f"Error loading {results_file}: {e}")
        
        return pd.DataFrame(results_data)
    
    def create_comprehensive_metrics_table(self):
        """Create comprehensive performance metrics table"""
        
        # Define metrics to include
        metrics_config = [
            ('test_accuracy', 'Test Accuracy', ':.3f'),
            ('f1_score', 'F1-Score', ':.3f'),
            ('precision', 'Precision', ':.3f'),
            ('recall', 'Recall', ':.3f'),
            ('cv_mean', 'CV Mean', ':.3f'),
            ('cv_std', 'CV Std', ':.4f'),
            ('weighted_f1', 'Weighted F1', ':.3f'),
            ('macro_f1', 'Macro F1', ':.3f')
        ]
        
        # Filter and aggregate data
        summary_data = []
        
        for dataset in self.df['dataset'].unique():
            for model in self.df['model'].unique():
                model_data = self.df[(self.df['dataset'] == dataset) & (self.df['model'] == model)]
                
                if len(model_data) == 0:
                    continue
                
                row = {
                    'Dataset': self.dataset_names.get(dataset, dataset),
                    'Model': self.model_names.get(model, model)
                }
                
                # Calculate aggregated metrics
                for metric, display_name, fmt in metrics_config:
                    if metric in model_data.columns:
                        values = model_data[metric].dropna()
                        if len(values) > 0:
                            row[display_name] = values.mean()
                        else:
                            row[display_name] = 0.0
                    else:
                        row[display_name] = 0.0
                
                # Add number of targets
                row['Targets'] = len(model_data)
                
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create sophisticated table visualization
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data with formatting
        table_data = []
        headers = ['Dataset', 'Model', 'Test Acc.', 'F1-Score', 'Precision', 'Recall', 
                  'CV Mean', 'CV Std', 'W-F1', 'M-F1', 'Targets']
        
        for _, row in summary_df.iterrows():
            formatted_row = [
                row['Dataset'],
                row['Model'],
                f"{row['Test Accuracy']:.3f}",
                f"{row['F1-Score']:.3f}",
                f"{row['Precision']:.3f}",
                f"{row['Recall']:.3f}",
                f"{row['CV Mean']:.3f}",
                f"{row['CV Std']:.4f}",
                f"{row['Weighted F1']:.3f}",
                f"{row['Macro F1']:.3f}",
                f"{int(row['Targets'])}"
            ]
            table_data.append(formatted_row)
        
        # Create table with professional styling
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colWidths=[0.12, 0.12, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.06]
        )
        
        # Professional table styling
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.2)
        
        # Header styling with gradient-like effect
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2E86AB')  # Professional blue
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.15)
            table[(0, i)].set_edgecolor('white')
            table[(0, i)].set_linewidth(2)
        
        # Row styling with alternating colors and performance-based highlighting
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F8F9FA')  # Light gray
                else:
                    table[(i, j)].set_facecolor('white')
                
                # Highlight top performers
                if j in [2, 3, 4, 5]:  # Performance metrics columns
                    try:
                        value = float(table_data[i-1][j])
                        if value >= 0.95:  # Excellent performance
                            table[(i, j)].set_facecolor('#D4EDDA')  # Light green
                        elif value >= 0.80:  # Good performance
                            table[(i, j)].set_facecolor('#FFF3CD')  # Light yellow
                        elif value < 0.50:  # Poor performance
                            table[(i, j)].set_facecolor('#F8D7DA')  # Light red
                    except:
                        pass
                
                table[(i, j)].set_height(0.12)
                table[(i, j)].set_edgecolor('#E0E0E0')
                table[(i, j)].set_linewidth(1)
        
        # Add sophisticated title
        plt.suptitle('Comprehensive Model Performance Metrics Analysis\n' +
                    'Advanced Classification Performance on Top-Performing Datasets',
                    fontsize=18, fontweight='bold', y=0.92)
        
        # Add legend for color coding
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='#D4EDDA', label='Excellent (≥0.95)'),
            plt.Rectangle((0,0),1,1, facecolor='#FFF3CD', label='Good (≥0.80)'),
            plt.Rectangle((0,0),1,1, facecolor='#F8D7DA', label='Poor (<0.50)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 0.15))
        
        # Add annotations
        fig.text(0.5, 0.08, 
                'W-F1: Weighted F1-Score | M-F1: Macro F1-Score | CV: Cross-Validation\n' +
                'Color coding indicates performance levels for Test Accuracy, F1-Score, Precision, and Recall',
                ha='center', fontsize=10, style='italic')
        
        # Save the table
        output_path = self.output_dir / "comprehensive_metrics_table.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.3)
        
        output_path_pdf = self.output_dir / "comprehensive_metrics_table.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Save CSV for further analysis
        csv_path = self.output_dir / "comprehensive_metrics_data.csv"
        summary_df.to_csv(csv_path, index=False)
        
        print(f"✅ Comprehensive metrics table saved:")
        print(f"   📊 PNG: {output_path}")
        print(f"   📄 PDF: {output_path_pdf}")
        print(f"   💾 CSV: {csv_path}")
        
        return summary_df
    
    def create_performance_comparison_charts(self):
        """Create detailed performance comparison charts"""
        
        # Filter data for available metrics
        plot_data = self.df.copy()
        
        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Performance Metrics Comparison\nTop 2 Datasets Scientific Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Test Accuracy vs F1-Score Scatter
        ax1 = axes[0, 0]
        for dataset in plot_data['dataset'].unique():
            dataset_data = plot_data[plot_data['dataset'] == dataset]
            
            if 'f1_score' in dataset_data.columns and 'test_accuracy' in dataset_data.columns:
                scatter = ax1.scatter(dataset_data['test_accuracy'], dataset_data['f1_score'],
                           label=self.dataset_names.get(dataset, dataset),
                           alpha=0.7, s=100, edgecolors='black', linewidth=1)
        
        ax1.set_xlabel('Test Accuracy', fontweight='bold')
        ax1.set_ylabel('F1-Score', fontweight='bold')
        ax1.set_title('Test Accuracy vs F1-Score Correlation', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1.1)
        ax1.set_ylim(0, 1.1)
        
        # 2. Precision vs Recall
        ax2 = axes[0, 1]
        for dataset in plot_data['dataset'].unique():
            dataset_data = plot_data[plot_data['dataset'] == dataset]
            
            if 'precision' in dataset_data.columns and 'recall' in dataset_data.columns:
                ax2.scatter(dataset_data['precision'], dataset_data['recall'],
                           label=self.dataset_names.get(dataset, dataset),
                           alpha=0.7, s=100, edgecolors='black', linewidth=1)
        
        ax2.set_xlabel('Precision', fontweight='bold')
        ax2.set_ylabel('Recall', fontweight='bold')
        ax2.set_title('Precision vs Recall Trade-off', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1.1)
        ax2.set_ylim(0, 1.1)
        
        # 3. Cross-Validation Performance
        ax3 = axes[1, 0]
        cv_data = plot_data.groupby(['dataset', 'model']).agg({
            'cv_mean': 'mean',
            'cv_std': 'mean'
        }).reset_index()
        
        width = 0.35
        datasets = cv_data['dataset'].unique()
        x = np.arange(len(datasets))
        
        for i, model in enumerate(cv_data['model'].unique()):
            model_data = cv_data[cv_data['model'] == model]
            means = [model_data[model_data['dataset'] == d]['cv_mean'].iloc[0] 
                    if len(model_data[model_data['dataset'] == d]) > 0 else 0 
                    for d in datasets]
            stds = [model_data[model_data['dataset'] == d]['cv_std'].iloc[0] 
                   if len(model_data[model_data['dataset'] == d]) > 0 else 0 
                   for d in datasets]
            
            ax3.bar(x + i * width/len(cv_data['model'].unique()), means, 
                   width/len(cv_data['model'].unique()), 
                   label=self.model_names.get(model, model),
                   color=self.colors.get(model, '#333333'),
                   alpha=0.8, edgecolor='black', linewidth=1)
        
        ax3.set_xlabel('Dataset', fontweight='bold')
        ax3.set_ylabel('Cross-Validation Score', fontweight='bold')
        ax3.set_title('Cross-Validation Performance', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([self.dataset_names.get(d, d) for d in datasets])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Model Ranking Heatmap
        ax4 = axes[1, 1]
        
        # Create ranking matrix
        ranking_metrics = ['test_accuracy', 'f1_score', 'precision', 'recall']
        available_metrics = [m for m in ranking_metrics if m in plot_data.columns]
        
        if available_metrics:
            ranking_data = plot_data.groupby(['dataset', 'model'])[available_metrics].mean()
            
            # Calculate rankings (1 = best)
            for metric in available_metrics:
                ranking_data[f'{metric}_rank'] = ranking_data.groupby('dataset')[metric].rank(ascending=False)
            
            rank_cols = [f'{m}_rank' for m in available_metrics]
            ranking_matrix = ranking_data[rank_cols].reset_index()
            
            # Pivot for heatmap
            heatmap_data = ranking_matrix.groupby(['dataset', 'model'])[rank_cols].mean()
            
            if not heatmap_data.empty:
                # Create pivot table
                pivot_data = heatmap_data.reset_index().melt(
                    id_vars=['dataset', 'model'], 
                    value_vars=rank_cols
                ).pivot_table(
                    values='value', 
                    index='model', 
                    columns=['dataset', 'variable']
                )
                
                sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                           center=2, ax=ax4, cbar_kws={'label': 'Rank (1=Best)'})
                ax4.set_title('Model Performance Rankings', fontweight='bold')
                ax4.set_ylabel('Model', fontweight='bold')
                ax4.set_xlabel('Dataset × Metric', fontweight='bold')
        
        plt.tight_layout()
        
        # Save comparison charts
        output_path = self.output_dir / "performance_comparison_charts.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        output_path_pdf = self.output_dir / "performance_comparison_charts.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        
        plt.show()
        
        print(f"✅ Performance comparison charts saved:")
        print(f"   📊 PNG: {output_path}")
        print(f"   📄 PDF: {output_path_pdf}")
    
    def run_comprehensive_analysis(self):
        """Run complete comprehensive metrics analysis"""
        
        print("📊 Comprehensive Performance Metrics Dashboard")
        print("=" * 60)
        print("🎯 Focus: F1-Score, Precision, Recall, R², Cross-Validation")
        print("📈 Scientific publication-quality visualizations")
        
        if self.df.empty:
            print("❌ No performance data found")
            return
        
        print(f"\n📊 Loaded {len(self.df)} model results")
        print(f"🗂️  Datasets: {self.df['dataset'].unique()}")
        print(f"🤖 Models: {self.df['model'].unique()}")
        
        # Available metrics analysis
        metrics_columns = [col for col in self.df.columns 
                          if col not in ['dataset', 'model', 'target']]
        print(f"📈 Available metrics: {len(metrics_columns)}")
        
        # Create comprehensive visualizations
        print(f"\n🎨 Creating comprehensive metrics dashboard...")
        
        # 1. Comprehensive metrics table
        summary_df = self.create_comprehensive_metrics_table()
        
        # 2. Performance comparison charts
        self.create_performance_comparison_charts()
        
        print(f"\n🎉 Comprehensive Analysis Complete!")
        print(f"📁 All visualizations saved to: {self.output_dir}")
        print(f"🔬 Publication-quality scientific analysis generated")
        
        # Print summary statistics
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print("=" * 40)
        
        if not summary_df.empty:
            for dataset in self.df['dataset'].unique():
                dataset_display = self.dataset_names.get(dataset, dataset)
                print(f"\n🏆 {dataset_display}:")
                
                dataset_summary = summary_df[summary_df['Dataset'] == dataset_display]
                if not dataset_summary.empty:
                    best_model = dataset_summary.loc[dataset_summary['Test Accuracy'].idxmax()]
                    print(f"   Best Model: {best_model['Model']}")
                    print(f"   Test Accuracy: {best_model['Test Accuracy']:.3f}")
                    if 'F1-Score' in best_model:
                        print(f"   F1-Score: {best_model['F1-Score']:.3f}")

def main():
    """Main execution function"""
    
    try:
        dashboard = ComprehensiveMetricsDashboard()
        dashboard.run_comprehensive_analysis()
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())