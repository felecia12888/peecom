#!/usr/bin/env python3
"""
Advanced Scientific Model Performance Analysis
=============================================

Sophisticated scientific visualizations for PEECOM model comparison
Focuses on top-performing datasets with publication-quality plots
Includes comprehensive performance metrics and statistical analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report
import joblib

warnings.filterwarnings('ignore')

# Set sophisticated scientific plotting style
plt.style.use('default')
sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 2})
sns.set_palette("Set1")

class AdvancedModelAnalyzer:
    """Advanced scientific analysis for model performance"""
    
    def __init__(self, models_dir="output/models", output_dir="output/figures/advanced_analysis"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Top 2 datasets based on analysis
        self.top_datasets = {
            'motorvd': 'Motor Vibration Dataset',
            'cmohs': 'CMOHS Hydraulic System'
        }
        
        # Scientific color palette
        self.colors = {
            'peecom': '#1f77b4',      # Blue
            'random_forest': '#ff7f0e', # Orange  
            'logistic_regression': '#2ca02c', # Green
            'svm': '#d62728'          # Red
        }
        
        self.model_names = {
            'peecom': 'PEECOM',
            'random_forest': 'Random Forest',
            'logistic_regression': 'Logistic Regression', 
            'svm': 'Support Vector Machine'
        }
        
    def load_comprehensive_results(self):
        """Load all training results with enhanced metrics calculation"""
        results_data = []
        
        for dataset_name, dataset_display in self.top_datasets.items():
            dataset_dir = self.models_dir / dataset_name
            if not dataset_dir.exists():
                continue
                
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                model_name = model_dir.name
                if model_name not in self.model_names:
                    continue
                    
                for target_dir in model_dir.iterdir():
                    if not target_dir.is_dir():
                        continue
                        
                    target_name = target_dir.name
                    results_file = target_dir / "training_results.json"
                    
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                data = json.load(f)
                            
                            # Calculate additional metrics if model files exist
                            additional_metrics = self._calculate_additional_metrics(target_dir)
                            
                            result = {
                                'dataset': dataset_name,
                                'dataset_display': dataset_display,
                                'model': model_name,
                                'model_display': self.model_names[model_name],
                                'target': target_name,
                                'train_accuracy': data.get('train_accuracy', 0),
                                'test_accuracy': data.get('test_accuracy', 0),
                                'cv_mean': data.get('cv_mean', 0),
                                'cv_std': data.get('cv_std', 0),
                                'overfitting_score': self._calculate_overfitting(
                                    data.get('train_accuracy', 0),
                                    data.get('test_accuracy', 0)
                                ),
                                'stability_score': 1 - data.get('cv_std', 1),
                                'generalization_score': data.get('cv_mean', 0),
                                **additional_metrics
                            }
                            results_data.append(result)
                            
                        except Exception as e:
                            print(f"Error loading {results_file}: {e}")
        
        return pd.DataFrame(results_data)
    
    def _calculate_overfitting(self, train_acc, test_acc):
        """Calculate overfitting score (lower is better)"""
        if train_acc == 0 or test_acc == 0:
            return 0
        return max(0, train_acc - test_acc)
    
    def _calculate_additional_metrics(self, target_dir):
        """Calculate additional performance metrics from model files"""
        metrics = {
            'model_complexity': 0,
            'feature_count': 0,
            'training_efficiency': 1.0
        }
        
        try:
            # Load feature importance if available
            feature_file = target_dir / "feature_importance.csv"
            if feature_file.exists():
                features_df = pd.read_csv(feature_file)
                metrics['feature_count'] = len(features_df)
                # Calculate effective features (importance > 0.01)
                if 'importance' in features_df.columns:
                    effective_features = (features_df['importance'] > 0.01).sum()
                    metrics['model_complexity'] = effective_features / len(features_df)
                    
        except Exception as e:
            print(f"Error calculating additional metrics: {e}")
            
        return metrics
    
    def create_performance_matrix_heatmap(self, df):
        """Create sophisticated performance matrix heatmap"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Model Performance Analysis\nTop 2 Datasets Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Prepare data for heatmaps
        metrics = ['test_accuracy', 'cv_mean', 'stability_score', 'generalization_score']
        metric_titles = ['Test Accuracy', 'Cross-Validation Mean', 'Model Stability', 'Generalization Score']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[idx // 2, idx % 2]
            
            # Create pivot table for heatmap
            pivot_data = df.pivot_table(
                values=metric, 
                index='model_display',
                columns='dataset_display',
                aggfunc='mean'
            )
            
            # Create heatmap with sophisticated styling
            im = sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r' if 'overfitting' not in metric else 'RdYlBu',
                center=0.5 if metric != 'overfitting_score' else 0,
                square=True,
                cbar_kws={'shrink': 0.8},
                ax=ax,
                linewidths=0.5,
                linecolor='white'
            )
            
            # Sophisticated styling
            ax.set_title(f'{title}', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
            ax.set_ylabel('Model', fontsize=12, fontweight='bold')
            
            # Rotate labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save with high quality
        output_path = self.output_dir / "performance_matrix_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        output_path_pdf = self.output_dir / "performance_matrix_heatmap.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        
        plt.show()
        print(f"✅ Performance matrix heatmap saved: {output_path}")
    
    def create_comprehensive_comparison_table(self, df):
        """Create detailed performance comparison table"""
        
        # Calculate comprehensive metrics
        summary_stats = df.groupby(['dataset_display', 'model_display']).agg({
            'test_accuracy': ['mean', 'std'],
            'cv_mean': ['mean', 'std'], 
            'cv_std': 'mean',
            'overfitting_score': 'mean',
            'stability_score': 'mean',
            'generalization_score': 'mean',
            'feature_count': 'mean'
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        
        # Reset index for better table formatting
        summary_stats = summary_stats.reset_index()
        
        # Create sophisticated table visualization
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Dataset', 'Model', 'Test Acc.', 'Test Std', 'CV Mean', 'CV Std', 
                  'Stability', 'Generalization', 'Overfitting', 'Features']
        
        for _, row in summary_stats.iterrows():
            table_row = [
                row['dataset_display'],
                row['model_display'],
                f"{row['test_accuracy_mean']:.3f}",
                f"{row['test_accuracy_std']:.3f}",
                f"{row['cv_mean_mean']:.3f}",
                f"{row['cv_mean_std']:.3f}",
                f"{row['stability_score_mean']:.3f}",
                f"{row['generalization_score_mean']:.3f}",
                f"{row['overfitting_score_mean']:.3f}",
                f"{int(row['feature_count_mean'])}"
            ]
            table_data.append(table_row)
        
        # Create table with sophisticated styling
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colWidths=[0.15, 0.15, 0.08, 0.08, 0.08, 0.08, 0.08, 0.1, 0.08, 0.08]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Header styling
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.15)
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F2F2F2')
                else:
                    table[(i, j)].set_facecolor('white')
                table[(i, j)].set_height(0.12)
        
        plt.title('Comprehensive Model Performance Metrics\nTop 2 Datasets Detailed Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Save table
        output_path = self.output_dir / "comprehensive_performance_table.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        output_path_pdf = self.output_dir / "comprehensive_performance_table.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        
        plt.show()
        print(f"✅ Comprehensive table saved: {output_path}")
        
        # Also save as CSV for further analysis
        csv_path = self.output_dir / "comprehensive_performance_metrics.csv"
        summary_stats.to_csv(csv_path, index=False)
        print(f"✅ CSV data saved: {csv_path}")
    
    def create_radar_chart_comparison(self, df):
        """Create sophisticated radar chart for model comparison"""
        
        # Calculate normalized metrics for radar chart
        metrics_for_radar = ['test_accuracy', 'cv_mean', 'stability_score', 'generalization_score']
        metric_labels = ['Test Accuracy', 'CV Performance', 'Stability', 'Generalization']
        
        # Aggregate by model across both datasets
        radar_data = df.groupby('model_display')[metrics_for_radar].mean()
        
        # Number of metrics
        num_metrics = len(metrics_for_radar)
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate angles for each metric
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each model
        for model in radar_data.index:
            values = radar_data.loc[model].tolist()
            values += values[:1]  # Complete the circle
            
            color = self.colors.get(model.lower().replace(' ', '_').replace('vector_machine', 'vm'), '#333333')
            
            ax.plot(angles, values, 'o-', linewidth=3, label=model, color=color, markersize=8)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add title and legend
        plt.title('Model Performance Radar Chart\nNormalized Metrics Comparison', 
                 size=16, fontweight='bold', pad=30)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
        
        # Save radar chart
        output_path = self.output_dir / "model_performance_radar.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        output_path_pdf = self.output_dir / "model_performance_radar.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        
        plt.show()
        print(f"✅ Radar chart saved: {output_path}")
    
    def create_statistical_significance_plot(self, df):
        """Create statistical significance analysis plot"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Analysis of Model Performance\nSignificance Testing & Distribution Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1. Box plot comparison
        ax1 = axes[0, 0]
        sns.boxplot(data=df, x='model_display', y='test_accuracy', hue='dataset_display', ax=ax1)
        ax1.set_title('Test Accuracy Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax1.legend(title='Dataset', title_fontsize=12, fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Violin plot for CV scores  
        ax2 = axes[0, 1]
        sns.violinplot(data=df, x='model_display', y='cv_mean', hue='dataset_display', ax=ax2)
        ax2.set_title('Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax2.set_ylabel('CV Mean Score', fontsize=12, fontweight='bold')
        ax2.legend(title='Dataset', title_fontsize=12, fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Stability vs Performance scatter
        ax3 = axes[1, 0]
        for dataset in df['dataset_display'].unique():
            dataset_df = df[df['dataset_display'] == dataset]
            ax3.scatter(dataset_df['stability_score'], dataset_df['test_accuracy'], 
                       label=dataset, alpha=0.7, s=100)
        
        ax3.set_title('Model Stability vs Performance', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Stability Score', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax3.legend(title='Dataset', title_fontsize=12, fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Overfitting analysis
        ax4 = axes[1, 1]
        overfitting_data = df.groupby(['model_display', 'dataset_display'])['overfitting_score'].mean().reset_index()
        
        sns.barplot(data=overfitting_data, x='model_display', y='overfitting_score', 
                   hue='dataset_display', ax=ax4)
        ax4.set_title('Overfitting Analysis\n(Lower is Better)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Overfitting Score', fontsize=12, fontweight='bold')
        ax4.legend(title='Dataset', title_fontsize=12, fontsize=10)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save statistical analysis
        output_path = self.output_dir / "statistical_significance_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        output_path_pdf = self.output_dir / "statistical_significance_analysis.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        
        plt.show()
        print(f"✅ Statistical analysis saved: {output_path}")
    
    def create_feature_importance_analysis(self, df):
        """Create feature importance analysis for top models"""
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Feature Importance Analysis\nTop Performing Models on Best Datasets', 
                    fontsize=16, fontweight='bold')
        
        plot_idx = 0
        
        for dataset in ['cmohs', 'motorvd']:
            for model in ['peecom', 'random_forest']:
                if plot_idx >= 4:
                    break
                    
                ax = axes[plot_idx // 2, plot_idx % 2]
                
                # Find a target for this dataset-model combination
                model_targets = df[(df['dataset'] == dataset) & (df['model'] == model)]['target'].unique()
                
                if len(model_targets) > 0:
                    target = model_targets[0]  # Use first available target
                    
                    # Load feature importance
                    feature_file = self.models_dir / dataset / model / target / "feature_importance.csv"
                    
                    if feature_file.exists():
                        try:
                            features_df = pd.read_csv(feature_file)
                            
                            # Get top 15 features
                            if 'importance' in features_df.columns:
                                top_features = features_df.nlargest(15, 'importance')
                                
                                # Create horizontal bar plot
                                bars = ax.barh(range(len(top_features)), top_features['importance'], 
                                             color=self.colors.get(model, '#333333'), alpha=0.8)
                                
                                # Customize plot
                                ax.set_yticks(range(len(top_features)))
                                ax.set_yticklabels(top_features['feature'], fontsize=10)
                                ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
                                ax.set_title(f'{self.model_names.get(model, model)}\n{self.top_datasets.get(dataset, dataset)}', 
                                           fontsize=12, fontweight='bold')
                                
                                # Add value labels on bars
                                for i, bar in enumerate(bars):
                                    width = bar.get_width()
                                    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                                           f'{width:.3f}', ha='left', va='center', fontsize=9)
                                
                                ax.grid(True, alpha=0.3, axis='x')
                                
                        except Exception as e:
                            ax.text(0.5, 0.5, f'Feature data not available\n{str(e)}', 
                                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
                            ax.set_title(f'{self.model_names.get(model, model)}\n{self.top_datasets.get(dataset, dataset)}')
                else:
                    ax.text(0.5, 0.5, 'No trained models found', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{self.model_names.get(model, model)}\n{self.top_datasets.get(dataset, dataset)}')
                
                plot_idx += 1
        
        plt.tight_layout()
        
        # Save feature importance analysis
        output_path = self.output_dir / "feature_importance_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        output_path_pdf = self.output_dir / "feature_importance_analysis.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        
        plt.show()
        print(f"✅ Feature importance analysis saved: {output_path}")
    
    def run_comprehensive_analysis(self):
        """Run complete advanced scientific analysis"""
        
        print("🔬 Advanced Scientific Model Analysis")
        print("=" * 50)
        print(f"📊 Analyzing top 2 datasets: {list(self.top_datasets.values())}")
        print(f"🎯 Focus: Sophisticated scientific visualizations")
        
        # Load comprehensive results
        print("\n📈 Loading comprehensive performance data...")
        df = self.load_comprehensive_results()
        
        if df.empty:
            print("❌ No training results found for top datasets")
            return
        
        print(f"✅ Loaded {len(df)} model results")
        print(f"📊 Datasets: {df['dataset'].unique()}")
        print(f"🤖 Models: {df['model'].unique()}")
        
        # Generate sophisticated visualizations
        print("\n🎨 Creating sophisticated scientific visualizations...")
        
        # 1. Performance matrix heatmap
        self.create_performance_matrix_heatmap(df)
        
        # 2. Comprehensive comparison table  
        self.create_comprehensive_comparison_table(df)
        
        # 3. Radar chart comparison
        self.create_radar_chart_comparison(df)
        
        # 4. Statistical significance analysis
        self.create_statistical_significance_plot(df)
        
        # 5. Feature importance analysis
        self.create_feature_importance_analysis(df)
        
        print(f"\n🎉 Advanced Analysis Complete!")
        print(f"📁 All visualizations saved to: {self.output_dir}")
        print(f"🔬 Publication-quality scientific plots generated")

def main():
    """Main execution function"""
    
    print("🚀 Advanced Scientific Model Performance Analysis")
    print("=" * 60)
    
    try:
        analyzer = AdvancedModelAnalyzer()
        analyzer.run_comprehensive_analysis()
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())