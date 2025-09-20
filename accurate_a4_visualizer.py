#!/usr/bin/env python3
"""
Accurate A4-Optimized Visualizations - Based on Real PEECOM Performance Data

This script creates A4 paper format optimized visualizations showing the REAL performance 
comparison between PEECOM and other models based on actual training results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for A4 format
plt.rcParams.update({
    'font.size': 6,
    'axes.titlesize': 7,
    'axes.labelsize': 6,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
    'figure.titlesize': 8,
    'lines.linewidth': 0.8,
    'lines.markersize': 4,
    'grid.linewidth': 0.2,
    'grid.alpha': 0.3,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.framealpha': 0.9,
    'legend.borderpad': 0.2,
    'axes.titlepad': 4,
    'axes.labelpad': 2,
})

class AccurateA4Visualizer:
    """Create accurate A4-optimized visualizations based on real performance data"""
    
    def __init__(self, output_dir="output/figures/accurate_a4"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load REAL performance data
        self.df = self.load_real_data()
        
        # Color scheme
        self.colors = {
            'peecom': '#1f77b4',
            'random_forest': '#ff7f0e',
            'cmohs': '#8c564b',
            'motorvd': '#9467bd'
        }
        
        # Display names
        self.dataset_names = {
            'motorvd': 'Motor Vibration',
            'cmohs': 'CMOHS Hydraulic'
        }
        
        self.model_names = {
            'peecom': 'PEECOM',
            'random_forest': 'Random Forest'
        }
        
    def load_real_data(self):
        """Load real performance data from comprehensive CSV"""
        data_file = "comprehensive_performance_data.csv"
        if Path(data_file).exists():
            return pd.read_csv(data_file)
        else:
            print(f"❌ {data_file} not found!")
            return pd.DataFrame()
    
    def create_performance_comparison_matrix(self):
        """Create comprehensive performance comparison matrix"""
        
        if self.df.empty:
            print("⚠️ No data available for visualization")
            return
        
        # Create figure with proper A4 dimensions
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.27, 5.8))
        
        # 1. Overall Performance Heatmap
        pivot_data = self.df.pivot_table(
            values='test_accuracy',
            index='model',
            columns='dataset',
            aggfunc='mean'
        )
        
        im1 = sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            square=False,
            cbar_kws={'shrink': 0.4, 'aspect': 10, 'pad': 0.01},
            ax=ax1,
            linewidths=0.2,
            linecolor='white',
            annot_kws={'fontsize': 4, 'fontweight': 'bold'}
        )
        
        ax1.set_title('Average Test Accuracy', fontsize=6, fontweight='bold', pad=3)
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_xticklabels([self.dataset_names.get(x.get_text(), x.get_text()) 
                           for x in ax1.get_xticklabels()], rotation=0, fontsize=4)
        ax1.set_yticklabels([self.model_names.get(x.get_text(), x.get_text()) 
                           for x in ax1.get_yticklabels()], rotation=0, fontsize=4)
        
        # 2. PEECOM vs Random Forest Direct Comparison
        comparison_data = []
        for dataset in self.df['dataset'].unique():
            dataset_df = self.df[self.df['dataset'] == dataset]
            peecom_avg = dataset_df[dataset_df['model'] == 'peecom']['test_accuracy'].mean()
            rf_avg = dataset_df[dataset_df['model'] == 'random_forest']['test_accuracy'].mean()
            
            comparison_data.append({
                'Dataset': self.dataset_names.get(dataset, dataset),
                'PEECOM': peecom_avg,
                'Random Forest': rf_avg,
                'Difference': peecom_avg - rf_avg
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        x = np.arange(len(comp_df))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, comp_df['PEECOM'], width, 
                       label='PEECOM', color=self.colors['peecom'], alpha=0.8)
        bars2 = ax2.bar(x + width/2, comp_df['Random Forest'], width,
                       label='Random Forest', color=self.colors['random_forest'], alpha=0.8)
        
        ax2.set_title('PEECOM vs Random Forest', fontsize=6, fontweight='bold', pad=3)
        ax2.set_ylabel('Test Accuracy', fontsize=5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(comp_df['Dataset'], fontsize=4)
        ax2.legend(fontsize=4, loc='lower right')
        ax2.grid(True, alpha=0.2, linewidth=0.3)
        ax2.set_ylim(0.95, 1.01)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=3)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=3)
        
        # 3. Performance by Target (CMOHS detailed)
        cmohs_data = self.df[self.df['dataset'] == 'cmohs']
        target_comparison = cmohs_data.pivot_table(
            values='test_accuracy',
            index='target',
            columns='model',
            aggfunc='mean'
        )
        
        target_comparison.plot(kind='bar', ax=ax3, width=0.7, 
                              color=[self.colors['peecom'], self.colors['random_forest']])
        ax3.set_title('CMOHS Targets Detailed', fontsize=6, fontweight='bold', pad=3)
        ax3.set_xlabel('')
        ax3.set_ylabel('Test Accuracy', fontsize=5)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=4)
        ax3.legend([self.model_names[col] for col in target_comparison.columns], 
                  fontsize=4, loc='lower right')
        ax3.grid(True, alpha=0.2, linewidth=0.3)
        
        # 4. Cross-validation stability
        cv_data = self.df.groupby(['dataset', 'model']).agg({
            'cv_mean': 'mean',
            'cv_std': 'mean'
        }).reset_index()
        
        for i, dataset in enumerate(cv_data['dataset'].unique()):
            dataset_cv = cv_data[cv_data['dataset'] == dataset]
            x_pos = np.arange(len(dataset_cv)) + i * 0.4
            
            for j, (_, row) in enumerate(dataset_cv.iterrows()):
                color = self.colors.get(row['model'], '#333333')
                ax4.bar(x_pos[j], row['cv_mean'], 0.35, 
                       yerr=row['cv_std'], capsize=2,
                       label=f"{row['model']} ({dataset})" if i == 0 else "",
                       color=color, alpha=0.8)
        
        ax4.set_title('Cross-Validation Stability', fontsize=6, fontweight='bold', pad=3)
        ax4.set_ylabel('CV Score', fontsize=5)
        ax4.set_xlabel('Model & Dataset', fontsize=5)
        ax4.legend(fontsize=3, loc='lower left')
        ax4.grid(True, alpha=0.2, linewidth=0.3)
        
        # Overall title
        fig.suptitle('REAL PEECOM Performance Analysis - Comprehensive Comparison', 
                    fontsize=8, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.87, hspace=0.4, wspace=0.4)
        
        # Save outputs
        output_path = self.output_dir / "real_performance_comparison_a4.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.05)
        
        output_path_pdf = self.output_dir / "real_performance_comparison_a4.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.05)
        
        plt.show()
        print(f"✅ Real performance comparison saved: {output_path}")
        
    def create_peecom_wins_analysis(self):
        """Create detailed analysis showing where PEECOM wins and loses"""
        
        if self.df.empty:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.27, 4))
        
        # Calculate wins/losses
        wins_data = []
        for dataset in self.df['dataset'].unique():
            dataset_df = self.df[self.df['dataset'] == dataset]
            
            for target in dataset_df['target'].unique():
                target_df = dataset_df[dataset_df['target'] == target]
                
                peecom_score = target_df[target_df['model'] == 'peecom']['test_accuracy'].iloc[0]
                rf_score = target_df[target_df['model'] == 'random_forest']['test_accuracy'].iloc[0]
                
                wins_data.append({
                    'dataset': dataset,
                    'target': target,
                    'peecom_score': peecom_score,
                    'rf_score': rf_score,
                    'difference': peecom_score - rf_score,
                    'winner': 'PEECOM' if peecom_score > rf_score else 'Random Forest' if rf_score > peecom_score else 'Tie'
                })
        
        wins_df = pd.DataFrame(wins_data)
        
        # 1. Win/Loss summary
        win_counts = wins_df['winner'].value_counts()
        colors = [self.colors['peecom'] if x == 'PEECOM' else self.colors['random_forest'] if x == 'Random Forest' else '#cccccc' 
                 for x in win_counts.index]
        
        bars = ax1.bar(win_counts.index, win_counts.values, color=colors, alpha=0.8)
        ax1.set_title('Head-to-Head Results', fontsize=6, fontweight='bold', pad=3)
        ax1.set_ylabel('Number of Targets Won', fontsize=5)
        ax1.tick_params(axis='x', labelsize=4)
        ax1.tick_params(axis='y', labelsize=4)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{int(height)}', ha='center', va='bottom', fontsize=4, fontweight='bold')
        
        # 2. Performance difference analysis
        ax2.scatter(wins_df['peecom_score'], wins_df['rf_score'], 
                   c=[self.colors['peecom'] if x > 0 else self.colors['random_forest'] if x < 0 else '#cccccc' 
                      for x in wins_df['difference']], 
                   alpha=0.7, s=15)
        
        # Add diagonal line (equal performance)
        min_score = min(wins_df['peecom_score'].min(), wins_df['rf_score'].min())
        max_score = max(wins_df['peecom_score'].max(), wins_df['rf_score'].max())
        ax2.plot([min_score, max_score], [min_score, max_score], 'k--', alpha=0.5, linewidth=1)
        
        ax2.set_xlabel('PEECOM Accuracy', fontsize=5)
        ax2.set_ylabel('Random Forest Accuracy', fontsize=5)
        ax2.set_title('Performance Correlation', fontsize=6, fontweight='bold', pad=3)
        ax2.tick_params(axis='both', labelsize=4)
        ax2.grid(True, alpha=0.2, linewidth=0.3)
        
        # Add text annotations for key points
        for _, row in wins_df.iterrows():
            if abs(row['difference']) > 0.01:  # Only annotate significant differences
                ax2.annotate(f"{row['target'][:8]}", 
                           (row['peecom_score'], row['rf_score']),
                           xytext=(2, 2), textcoords='offset points',
                           fontsize=3, alpha=0.8)
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        
        # Save outputs
        output_path = self.output_dir / "peecom_wins_analysis_a4.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.05)
        
        plt.show()
        print(f"✅ PEECOM wins analysis saved: {output_path}")
        
    def export_performance_table(self):
        """Export comprehensive performance table to Excel"""
        
        if self.df.empty:
            return
            
        # Create summary table
        summary_data = []
        
        for dataset in self.df['dataset'].unique():
            dataset_df = self.df[self.df['dataset'] == dataset]
            
            for model in ['peecom', 'random_forest']:
                model_data = dataset_df[dataset_df['model'] == model]
                
                summary_data.append({
                    'Dataset': self.dataset_names.get(dataset, dataset),
                    'Model': self.model_names.get(model, model),
                    'Avg_Test_Accuracy': model_data['test_accuracy'].mean(),
                    'Std_Test_Accuracy': model_data['test_accuracy'].std(),
                    'Avg_CV_Score': model_data['cv_mean'].mean(),
                    'Std_CV_Score': model_data['cv_std'].mean(),
                    'Targets_Count': len(model_data),
                    'Min_Accuracy': model_data['test_accuracy'].min(),
                    'Max_Accuracy': model_data['test_accuracy'].max()
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to Excel
        excel_path = self.output_dir / "real_performance_summary.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            self.df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        print(f"✅ Performance table exported: {excel_path}")
        
    def generate_all_visualizations(self):
        """Generate all accurate A4 visualizations"""
        
        print("🎨 Creating ACCURATE A4-Optimized Visualizations")
        print("="*50)
        print(f"📊 Based on REAL performance data: {len(self.df)} results")
        print(f"🗂️ Datasets: {sorted(self.df['dataset'].unique())}")
        print(f"🤖 Models: {sorted(self.df['model'].unique())}")
        
        # Create visualizations
        self.create_performance_comparison_matrix()
        self.create_peecom_wins_analysis()
        self.export_performance_table()
        
        print(f"\n🎉 Accurate A4 Analysis Complete!")
        print(f"📁 All outputs saved to: {self.output_dir}")

if __name__ == "__main__":
    visualizer = AccurateA4Visualizer()
    visualizer.generate_all_visualizations()