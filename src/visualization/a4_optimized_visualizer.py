#!/usr/bin/env python3
"""
A4-Optimized Scientific Visualizations
=====================================

Optimized for A4 paper format with:
- Proper font sizing and spacing
- No overlapping text or elements
- Professional layout for printing
- Excel table export capability
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

# Set A4-optimized scientific style
plt.style.use('default')

# A4 paper optimized settings (210 × 297 mm) - VERY SMALL FONTS
plt.rcParams.update({
    'figure.figsize': (8.27, 11.69),    # A4 size in inches
    'figure.dpi': 150,                   # Good for screen viewing
    'savefig.dpi': 300,                  # High quality for printing
    'font.size': 6,                      # Much smaller base font
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.titlesize': 7,                 # Very small title size
    'axes.labelsize': 6,                 # Very small label size
    'xtick.labelsize': 5,                # Tiny tick labels
    'ytick.labelsize': 5,
    'legend.fontsize': 5,                # Tiny legend
    'legend.title_fontsize': 6,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 0.8,
    'lines.markersize': 3,               # Very small markers
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linewidth': 0.2,
    'grid.alpha': 0.3,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.framealpha': 0.9,
    'legend.borderpad': 0.2,
    'axes.titlepad': 4,                  # Minimal title padding
    'axes.labelpad': 2,                  # Minimal label padding
})

class A4OptimizedVisualizer:
    """Create A4-optimized scientific visualizations"""
    
    def __init__(self, output_dir="output/figures/a4_optimized"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.df = self.load_enhanced_data()
        
        # Compact color palette
        self.colors = {
            'peecom': '#1f77b4',
            'random_forest': '#ff7f0e',
            'logistic_regression': '#2ca02c',
            'svm': '#d62728',
            'motorvd': '#9467bd',
            'cmohs': '#8c564b'
        }
        
        # Compact names to avoid overlapping
        self.dataset_names = {
            'motorvd': 'Motor Vibration',
            'cmohs': 'CMOHS Hydraulic'
        }
        
        self.model_names = {
            'peecom': 'PEECOM',
            'random_forest': 'Random Forest',
            'logistic_regression': 'Log. Regression',  # Shortened
            'svm': 'SVM'
        }
        
        # Compact metric names
        self.metric_names = {
            'test_accuracy': 'Test Acc.',
            'f1_score': 'F1-Score',
            'precision': 'Precision',
            'recall': 'Recall',
            'cv_mean': 'CV Mean',
            'cv_std': 'CV Std'
        }
    
    def load_enhanced_data(self):
        """Load enhanced performance data"""
        summary_file = Path("output/enhanced_performance_summary.csv")
        if summary_file.exists():
            return pd.read_csv(summary_file)
        return pd.DataFrame()
    
    def create_compact_performance_matrix(self):
        """Create compact A4-optimized performance matrix"""
        
        metrics = ['test_accuracy', 'f1_score', 'precision', 'recall']
        available_metrics = [m for m in metrics if m in self.df.columns]
        
        if not available_metrics:
            print("⚠️  No performance metrics available")
            return
        
        # Calculate optimal subplot layout for A4
        n_metrics = len(available_metrics)
        if n_metrics <= 2:
            fig_size = (8.27, 4)  # Very compact height
            rows, cols = 1, n_metrics
        else:
            fig_size = (8.27, 5.5)  # Still compact
            rows, cols = 2, 2
        
        fig, axes = plt.subplots(rows, cols, figsize=fig_size)
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Remove extra subplots
        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            # Create pivot table
            pivot_data = self.df.pivot_table(
                values=metric,
                index='model',
                columns='dataset',
                aggfunc='mean'
            )
            
            # Create very compact heatmap
            im = sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.3f',
                cmap='viridis',
                square=False,
                cbar_kws={'shrink': 0.4, 'aspect': 10, 'pad': 0.01},
                ax=ax,
                linewidths=0.2,
                linecolor='white',
                annot_kws={'fontsize': 4, 'fontweight': 'normal'}  # Very tiny annotation
            )
            
            # Very compact formatting
            ax.set_title(self.metric_names.get(metric, metric), 
                        fontsize=6, fontweight='bold', pad=3)
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            # Tiny labels
            ax.set_xticklabels([self.dataset_names.get(x.get_text(), x.get_text()) 
                               for x in ax.get_xticklabels()], 
                               rotation=0, fontsize=4)
            ax.set_yticklabels([self.model_names.get(x.get_text(), x.get_text()) 
                               for x in ax.get_yticklabels()], 
                               rotation=0, fontsize=4)
        
        # Very compact overall title
        fig.suptitle('Model Performance Matrix - Top Datasets', 
                    fontsize=8, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.4)
        
        # Save with A4 optimization
        output_path = self.output_dir / "performance_matrix_a4.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.05)
        
        output_path_pdf = self.output_dir / "performance_matrix_a4.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.05)
        
        plt.show()
        print(f"✅ A4-optimized matrix saved: {output_path}")
    
    def create_compact_comparison_charts(self):
        """Create A4-optimized comparison charts"""
        
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 9))
        
        # 1. Test Accuracy vs F1-Score (top-left)
        ax1 = axes[0, 0]
        for dataset in self.df['dataset'].unique():
            dataset_data = self.df[self.df['dataset'] == dataset]
            
            if 'f1_score' in dataset_data.columns and 'test_accuracy' in dataset_data.columns:
                ax1.scatter(dataset_data['test_accuracy'], dataset_data['f1_score'],
                           label=self.dataset_names.get(dataset, dataset),
                           alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel('Test Accuracy', fontsize=8, fontweight='bold')
        ax1.set_ylabel('F1-Score', fontsize=8, fontweight='bold')
        ax1.set_title('Accuracy vs F1-Score', fontsize=9, fontweight='bold')
        ax1.legend(fontsize=6, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        
        # 2. Precision vs Recall (top-right)
        ax2 = axes[0, 1]
        for dataset in self.df['dataset'].unique():
            dataset_data = self.df[self.df['dataset'] == dataset]
            
            if 'precision' in dataset_data.columns and 'recall' in dataset_data.columns:
                ax2.scatter(dataset_data['precision'], dataset_data['recall'],
                           label=self.dataset_names.get(dataset, dataset),
                           alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Precision', fontsize=8, fontweight='bold')
        ax2.set_ylabel('Recall', fontsize=8, fontweight='bold')
        ax2.set_title('Precision vs Recall', fontsize=9, fontweight='bold')
        ax2.legend(fontsize=6, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        
        # 3. Cross-Validation Performance (bottom-left)
        ax3 = axes[1, 0]
        cv_data = self.df.groupby(['dataset', 'model']).agg({
            'cv_mean': 'mean'
        }).reset_index()
        
        width = 0.35
        datasets = cv_data['dataset'].unique()
        models = cv_data['model'].unique()
        x = np.arange(len(datasets))
        
        for i, model in enumerate(models):
            model_data = cv_data[cv_data['model'] == model]
            means = []
            for d in datasets:
                model_dataset = model_data[model_data['dataset'] == d]
                if len(model_dataset) > 0:
                    means.append(model_dataset['cv_mean'].iloc[0])
                else:
                    means.append(0)
            
            bars = ax3.bar(x + i * width/len(models), means, 
                          width/len(models), 
                          label=self.model_names.get(model, model),
                          color=self.colors.get(model, '#333333'),
                          alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax3.set_xlabel('Dataset', fontsize=8, fontweight='bold')
        ax3.set_ylabel('CV Score', fontsize=8, fontweight='bold')
        ax3.set_title('Cross-Validation Performance', fontsize=9, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([self.dataset_names.get(d, d) for d in datasets], fontsize=7)
        ax3.legend(fontsize=6, loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 4. Model Performance Summary (bottom-right)
        ax4 = axes[1, 1]
        
        # Create summary bar chart
        summary_data = self.df.groupby(['dataset', 'model'])['test_accuracy'].mean().reset_index()
        
        for i, dataset in enumerate(summary_data['dataset'].unique()):
            dataset_data = summary_data[summary_data['dataset'] == dataset]
            
            y_pos = np.arange(len(dataset_data))
            bars = ax4.barh(y_pos + i * 0.4, dataset_data['test_accuracy'],
                           height=0.35, alpha=0.8,
                           label=self.dataset_names.get(dataset, dataset),
                           color=self.colors.get(dataset, '#333333'))
            
            # Add value labels
            for j, bar in enumerate(bars):
                width = bar.get_width()
                ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=6)
        
        ax4.set_xlabel('Test Accuracy', fontsize=8, fontweight='bold')
        ax4.set_ylabel('Model', fontsize=8, fontweight='bold')
        ax4.set_title('Model Summary', fontsize=9, fontweight='bold')
        ax4.set_yticks(np.arange(len(summary_data['model'].unique())) + 0.2)
        ax4.set_yticklabels([self.model_names.get(m, m) for m in summary_data['model'].unique()], fontsize=7)
        ax4.legend(fontsize=6, loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 1.1)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Save A4-optimized charts
        output_path = self.output_dir / "comparison_charts_a4.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        
        output_path_pdf = self.output_dir / "comparison_charts_a4.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        
        plt.show()
        print(f"✅ A4-optimized comparison charts saved: {output_path}")
    
    def create_excel_summary_table(self):
        """Create comprehensive Excel table"""
        
        # Aggregate data
        summary_stats = self.df.groupby(['dataset', 'model']).agg({
            'test_accuracy': ['mean', 'std', 'count'],
            'f1_score': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'cv_mean': ['mean', 'std'],
            'cv_std': ['mean']
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        summary_stats = summary_stats.reset_index()
        
        # Rename columns for clarity
        summary_stats = summary_stats.rename(columns={
            'dataset': 'Dataset',
            'model': 'Model',
            'test_accuracy_mean': 'Test_Accuracy_Mean',
            'test_accuracy_std': 'Test_Accuracy_Std',
            'test_accuracy_count': 'Number_of_Targets',
            'f1_score_mean': 'F1_Score_Mean',
            'f1_score_std': 'F1_Score_Std',
            'precision_mean': 'Precision_Mean',
            'precision_std': 'Precision_Std',
            'recall_mean': 'Recall_Mean',
            'recall_std': 'Recall_Std',
            'cv_mean_mean': 'CV_Mean',
            'cv_mean_std': 'CV_Std_of_Means',
            'cv_std_mean': 'CV_Std_Average'
        })
        
        # Apply readable names
        summary_stats['Dataset'] = summary_stats['Dataset'].map(self.dataset_names)
        summary_stats['Model'] = summary_stats['Model'].map(self.model_names)
        
        # Create detailed table for visualization
        fig, ax = plt.subplots(figsize=(8.27, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare compact table data
        display_data = summary_stats[[
            'Dataset', 'Model', 'Test_Accuracy_Mean', 'F1_Score_Mean', 
            'Precision_Mean', 'Recall_Mean', 'CV_Mean', 'Number_of_Targets'
        ]].copy()
        
        # Format numbers for display
        for col in ['Test_Accuracy_Mean', 'F1_Score_Mean', 'Precision_Mean', 'Recall_Mean', 'CV_Mean']:
            display_data[col] = display_data[col].apply(lambda x: f"{x:.3f}")
        
        display_data['Number_of_Targets'] = display_data['Number_of_Targets'].astype(int)
        
        # Create table
        headers = ['Dataset', 'Model', 'Test Acc.', 'F1-Score', 'Precision', 'Recall', 'CV Mean', 'Targets']
        table_data = display_data.values.tolist()
        
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colWidths=[0.18, 0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.07]
        )
        
        # Professional compact styling
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Header styling
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.12)
        
        # Row styling with performance highlighting
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F8F9FA')
                else:
                    table[(i, j)].set_facecolor('white')
                
                # Highlight excellent performance
                if j in [2, 3, 4, 5, 6]:  # Performance columns
                    try:
                        value = float(table_data[i-1][j])
                        if value >= 0.95:
                            table[(i, j)].set_facecolor('#D4EDDA')
                            table[(i, j)].set_text_props(weight='bold')
                        elif value >= 0.80:
                            table[(i, j)].set_facecolor('#FFF3CD')
                    except:
                        pass
                
                table[(i, j)].set_height(0.10)
        
        plt.title('Comprehensive Model Performance Summary Table\n(Top 2 Datasets)', 
                 fontsize=11, fontweight='bold', pad=15)
        
        # Save table image
        output_path = self.output_dir / "summary_table_a4.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        
        plt.show()
        
        # Save Excel file
        excel_path = self.output_dir / "comprehensive_performance_metrics.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main summary sheet
            summary_stats.to_excel(writer, sheet_name='Performance_Summary', index=False)
            
            # Detailed raw data sheet
            self.df.to_excel(writer, sheet_name='Raw_Data', index=False)
            
            # Create a formatted sheet
            display_data_full = summary_stats.copy()
            display_data_full.to_excel(writer, sheet_name='Formatted_Summary', index=False)
        
        print(f"✅ Excel table saved: {excel_path}")
        print(f"✅ Table visualization saved: {output_path}")
        
        return summary_stats
    
    def run_a4_optimized_analysis(self):
        """Run complete A4-optimized analysis"""
        
        print("📄 A4-Optimized Scientific Visualizations")
        print("=" * 60)
        print("📐 Optimized for A4 paper format (210×297mm)")
        print("🔤 Proper font sizing and spacing")
        print("📊 Excel table export included")
        
        if self.df.empty:
            print("❌ No performance data found")
            return
        
        print(f"\n📊 Processing {len(self.df)} model results")
        print(f"🗂️  Datasets: {self.df['dataset'].unique()}")
        print(f"🤖 Models: {self.df['model'].unique()}")
        
        # Create A4-optimized visualizations
        print(f"\n🎨 Creating A4-optimized visualizations...")
        
        # 1. Compact performance matrix
        self.create_compact_performance_matrix()
        
        # 2. Compact comparison charts
        self.create_compact_comparison_charts()
        
        # 3. Excel summary table
        summary_df = self.create_excel_summary_table()
        
        print(f"\n🎉 A4-Optimized Analysis Complete!")
        print(f"📁 All visualizations saved to: {self.output_dir}")
        print(f"📄 Optimized for A4 printing and display")
        print(f"📊 Excel table ready for further analysis")
        
        # Print compact summary
        if not summary_df.empty:
            print(f"\n📋 COMPACT PERFORMANCE SUMMARY:")
            print("=" * 40)
            for dataset in self.df['dataset'].unique():
                dataset_display = self.dataset_names.get(dataset, dataset)
                print(f"\n🏆 {dataset_display}:")
                
                dataset_summary = summary_df[summary_df['Dataset'] == dataset_display]
                if not dataset_summary.empty:
                    best_idx = dataset_summary['Test_Accuracy_Mean'].idxmax()
                    best_model = dataset_summary.loc[best_idx]
                    print(f"   Best: {best_model['Model']} ({best_model['Test_Accuracy_Mean']:.3f})")

def main():
    """Main execution function"""
    
    try:
        visualizer = A4OptimizedVisualizer()
        visualizer.run_a4_optimized_analysis()
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())