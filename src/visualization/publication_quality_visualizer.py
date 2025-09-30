#!/usr/bin/env python3
"""
Publication-Quality Scientific Visualizations
============================================

Enhanced sophisticated scientific plots with:
- IEEE/Nature journal styling
- Optimized figure dimensions and aspect ratios
- Advanced color schemes and typography
- Statistical annotations and confidence intervals
- Publication-ready legends and annotations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality scientific style
plt.style.use('default')

# IEEE/Nature journal quality settings
plt.rcParams.update({
    'figure.figsize': (7, 5),          # IEEE single column width
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'font.family': 'serif',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.title_fontsize': 10,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'patch.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': False,
    'legend.framealpha': 0.9,
    'legend.borderpad': 0.4,
    'text.usetex': False  # Set to True if LaTeX is available
})

class PublicationQualityVisualizer:
    """Create publication-quality scientific visualizations"""
    
    def __init__(self, output_dir="output/figures/publication_quality"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.df = self.load_enhanced_data()
        
        # Scientific color palette (colorblind-friendly)
        self.colors = {
            'peecom': '#1f77b4',           # Blue
            'random_forest': '#ff7f0e',   # Orange
            'logistic_regression': '#2ca02c', # Green
            'svm': '#d62728',             # Red
            'motorvd': '#9467bd',         # Purple
            'cmohs': '#8c564b'            # Brown
        }
        
        # Professional markers
        self.markers = {
            'peecom': 'o',
            'random_forest': 's', 
            'logistic_regression': '^',
            'svm': 'D'
        }
        
        self.dataset_names = {
            'motorvd': 'Motor Vibration',
            'cmohs': 'CMOHS Hydraulic'
        }
        
        self.model_names = {
            'peecom': 'PEECOM',
            'random_forest': 'Random Forest',
            'logistic_regression': 'Logistic Regression',
            'svm': 'SVM'
        }
    
    def load_enhanced_data(self):
        """Load enhanced performance data"""
        summary_file = Path("output/enhanced_performance_summary.csv")
        if summary_file.exists():
            return pd.read_csv(summary_file)
        return pd.DataFrame()
    
    def create_performance_comparison_matrix(self):
        """Create sophisticated performance comparison matrix"""
        
        # Prepare data
        metrics = ['test_accuracy', 'f1_score', 'precision', 'recall']
        metric_labels = ['Test Accuracy', 'F1-Score', 'Precision', 'Recall']
        
        available_metrics = [m for m in metrics if m in self.df.columns]
        if not available_metrics:
            print("⚠️  No performance metrics available")
            return
        
        # Create figure with proper aspect ratio
        fig_width = 8.5  # IEEE double column width
        fig_height = 6
        fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
        
        # Remove empty subplots if we have fewer than 4 metrics
        for idx in range(len(available_metrics), 4):
            axes.flat[idx].remove()
        
        for idx, metric in enumerate(available_metrics[:4]):
            if idx >= 4:
                break
                
            ax = axes.flat[idx]
            
            # Create pivot table
            pivot_data = self.df.pivot_table(
                values=metric,
                index='model',
                columns='dataset',
                aggfunc='mean'
            )
            
            # Create heatmap with scientific styling
            im = sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.3f',
                cmap='viridis',  # Scientific colormap
                square=False,
                cbar_kws={'shrink': 0.8, 'aspect': 20},
                ax=ax,
                linewidths=0.5,
                linecolor='white',
                annot_kws={'fontsize': 9, 'fontweight': 'bold'}
            )
            
            # Scientific formatting
            ax.set_title(metric_labels[idx], fontweight='bold', pad=10)
            ax.set_xlabel('Dataset', fontweight='bold')
            ax.set_ylabel('Model' if idx in [0, 2] else '', fontweight='bold')
            
            # Format axis labels
            ax.set_xticklabels([self.dataset_names.get(x.get_text(), x.get_text()) 
                               for x in ax.get_xticklabels()], rotation=0)
            ax.set_yticklabels([self.model_names.get(x.get_text(), x.get_text()) 
                               for x in ax.get_yticklabels()], rotation=0)
        
        # Add overall title
        fig.suptitle('Model Performance Comparison Matrix\nTop-Performing Datasets Analysis', 
                    fontsize=14, fontweight='bold', y=0.96)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        # Save with publication quality
        output_path = self.output_dir / "performance_matrix_publication.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        output_path_pdf = self.output_dir / "performance_matrix_publication.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        plt.show()
        print(f"✅ Publication-quality matrix saved: {output_path}")
    
    def create_model_ranking_chart(self):
        """Create sophisticated model ranking visualization"""
        
        # Calculate comprehensive rankings
        ranking_metrics = ['test_accuracy', 'f1_score', 'precision', 'recall']
        available_metrics = [m for m in ranking_metrics if m in self.df.columns]
        
        if not available_metrics:
            print("⚠️  No ranking metrics available")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left plot: Model performance radar/bar chart
        summary_data = self.df.groupby(['dataset', 'model'])[available_metrics].mean().reset_index()
        
        # Bar chart for each dataset
        datasets = summary_data['dataset'].unique()
        models = summary_data['model'].unique()
        
        x = np.arange(len(models))
        width = 0.35
        
        for i, dataset in enumerate(datasets):
            dataset_data = summary_data[summary_data['dataset'] == dataset]
            
            # Use test_accuracy as primary metric
            values = []
            for model in models:
                model_data = dataset_data[dataset_data['model'] == model]
                if len(model_data) > 0:
                    values.append(model_data['test_accuracy'].iloc[0])
                else:
                    values.append(0)
            
            bars = ax1.bar(x + i * width, values, width, 
                          label=self.dataset_names.get(dataset, dataset),
                          color=self.colors.get(dataset, '#333333'),
                          alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', 
                        fontsize=8, fontweight='bold')
        
        ax1.set_xlabel('Model', fontweight='bold')
        ax1.set_ylabel('Test Accuracy', fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontweight='bold')
        ax1.set_xticks(x + width / 2)
        ax1.set_xticklabels([self.model_names.get(m, m) for m in models], rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Statistical significance analysis
        if len(available_metrics) >= 2:
            # Scatter plot of two main metrics
            metric1, metric2 = available_metrics[0], available_metrics[1]
            
            for dataset in datasets:
                dataset_data = summary_data[summary_data['dataset'] == dataset]
                
                scatter = ax2.scatter(dataset_data[metric1], dataset_data[metric2],
                            label=self.dataset_names.get(dataset, dataset),
                            color=self.colors.get(dataset, '#333333'),
                            alpha=0.7, s=100, edgecolors='black', linewidth=1)
                
                # Add model labels
                for _, row in dataset_data.iterrows():
                    ax2.annotate(self.model_names.get(row['model'], row['model']),
                               (row[metric1], row[metric2]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
            
            ax2.set_xlabel(metric1.replace('_', ' ').title(), fontweight='bold')
            ax2.set_ylabel(metric2.replace('_', ' ').title(), fontweight='bold')
            ax2.set_title('Performance Correlation Analysis', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "model_ranking_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        output_path_pdf = self.output_dir / "model_ranking_analysis.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        plt.show()
        print(f"✅ Model ranking analysis saved: {output_path}")
    
    def create_comprehensive_summary_table(self):
        """Create publication-quality summary table"""
        
        # Aggregate data by dataset and model
        summary_stats = self.df.groupby(['dataset', 'model']).agg({
            'test_accuracy': ['mean', 'std', 'count'],
            'f1_score': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'cv_mean': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        summary_stats = summary_stats.reset_index()
        
        # Create table figure
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        headers = ['Dataset', 'Model', 'Test Acc.', '±Std', 'F1-Score', '±Std', 
                  'Precision', '±Std', 'Recall', '±Std', 'CV Mean', '±Std', 'N']
        
        table_data = []
        for _, row in summary_stats.iterrows():
            formatted_row = [
                self.dataset_names.get(row['dataset'], row['dataset']),
                self.model_names.get(row['model'], row['model']),
                f"{row['test_accuracy_mean']:.3f}",
                f"±{row['test_accuracy_std']:.3f}",
                f"{row['f1_score_mean']:.3f}",
                f"±{row['f1_score_std']:.3f}",
                f"{row['precision_mean']:.3f}",
                f"±{row['precision_std']:.3f}",
                f"{row['recall_mean']:.3f}",
                f"±{row['recall_std']:.3f}",
                f"{row['cv_mean_mean']:.3f}",
                f"±{row['cv_mean_std']:.3f}",
                f"{int(row['test_accuracy_count'])}"
            ]
            table_data.append(formatted_row)
        
        # Create professional table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center')
        
        # Professional styling
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Header styling
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2C3E50')  # Dark blue-gray
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.15)
        
        # Row styling with performance highlighting
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F8F9FA')
                else:
                    table[(i, j)].set_facecolor('white')
                
                # Highlight best performance in each metric
                if j in [2, 4, 6, 8, 10]:  # Performance metric columns
                    try:
                        value = float(table_data[i-1][j])
                        if value >= 0.95:
                            table[(i, j)].set_facecolor('#D1F2EB')  # Light green
                            table[(i, j)].set_text_props(weight='bold')
                    except:
                        pass
                
                table[(i, j)].set_height(0.12)
        
        plt.title('Comprehensive Model Performance Summary\n' +
                 'Statistical Analysis with Standard Deviations',
                 fontsize=14, fontweight='bold', pad=20)
        
        # Save table
        output_path = self.output_dir / "comprehensive_summary_table.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        output_path_pdf = self.output_dir / "comprehensive_summary_table.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        plt.show()
        print(f"✅ Comprehensive summary table saved: {output_path}")
        
        # Also save as CSV
        csv_path = self.output_dir / "publication_summary_data.csv"
        summary_stats.to_csv(csv_path, index=False)
        print(f"✅ Summary data saved: {csv_path}")
    
    def run_publication_analysis(self):
        """Run complete publication-quality analysis"""
        
        print("📊 Publication-Quality Scientific Visualizations")
        print("=" * 60)
        print("🎯 IEEE/Nature journal standards")
        print("📐 Optimized dimensions and typography")
        print("🎨 Professional color schemes and styling")
        
        if self.df.empty:
            print("❌ No performance data found")
            return
        
        print(f"\n📊 Processing {len(self.df)} model results")
        print(f"🗂️  Datasets: {self.df['dataset'].unique()}")
        print(f"🤖 Models: {self.df['model'].unique()}")
        
        # Create publication-quality visualizations
        print(f"\n🎨 Creating publication-quality visualizations...")
        
        # 1. Performance comparison matrix
        self.create_performance_comparison_matrix()
        
        # 2. Model ranking analysis
        self.create_model_ranking_chart()
        
        # 3. Comprehensive summary table
        self.create_comprehensive_summary_table()
        
        print(f"\n🎉 Publication Analysis Complete!")
        print(f"📁 All visualizations saved to: {self.output_dir}")
        print(f"🔬 IEEE/Nature journal quality achieved")

def main():
    """Main execution function"""
    
    try:
        visualizer = PublicationQualityVisualizer()
        visualizer.run_publication_analysis()
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())