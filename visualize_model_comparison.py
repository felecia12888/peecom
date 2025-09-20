#!/usr/bin/env python3
"""
PEECOM Model Performance Visualization
=====================================

Comprehensive visualization system for comparing model performance across:
- All available datasets  
- All trained models
- All targets within each dataset

Features:
- Performance heatmaps
- Accuracy bar charts
- Model ranking plots
- Dataset-specific comparisons
- Cross-dataset analysis
- Publication-quality plots

Usage:
    python visualize_model_comparison.py                    # Generate all visualizations
    python visualize_model_comparison.py --dataset cmohs    # Dataset-specific plots
    python visualize_model_comparison.py --model peecom     # Model-specific plots
    python visualize_model_comparison.py --save-format pdf  # Save as PDF
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Dataset configuration
DATASET_CONFIG = {
    'cmohs': {
        'name': 'CMOHS Hydraulic System',
        'targets': ['cooler_condition', 'valve_condition', 'pump_leakage', 'accumulator_pressure', 'stable_flag']
    },
    'equipmentad': {
        'name': 'Equipment Anomaly Detection', 
        'targets': ['anomaly', 'equipment_type', 'location']
    },
    'mlclassem': {
        'name': 'ML Classification Energy Monthly',
        'targets': ['status', 'region', 'equipment_type']
    },
    'motorvd': {
        'name': 'Motor Vibration Dataset',
        'targets': ['condition', 'file_id']
    },
    'multivariatesd': {
        'name': 'Multivariate Time Series Dataset',
        'targets': ['RUL', 'engine_id', 'cycle']
    },
    'sensord': {
        'name': 'Sensor Monitoring Dataset', 
        'targets': ['machine_status', 'alert_level', 'maintenance_required']
    },
    'smartmd': {
        'name': 'Smart Maintenance Dataset',
        'targets': ['anomaly_flag', 'machine_status', 'maintenance_required']
    }
}

MODEL_DISPLAY_NAMES = {
    'peecom': 'PEECOM (Physics-Enhanced)',
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression', 
    'svm': 'Support Vector Machine'
}

class ModelPerformanceVisualizer:
    """Comprehensive model performance visualization system."""
    
    def __init__(self, models_dir='output/models', output_dir='output/figures/model_comparison'):
        """
        Initialize the visualizer.
        
        Args:
            models_dir: Directory containing trained models
            output_dir: Directory to save visualization plots
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all performance data
        self.performance_data = self._load_all_performance_data()
        
    def _load_all_performance_data(self):
        """Load performance data from all trained models."""
        performance_data = defaultdict(lambda: defaultdict(dict))
        
        if not self.models_dir.exists():
            print(f"Models directory {self.models_dir} does not exist!")
            return performance_data
            
        # Scan through all model directories
        for dataset_dir in self.models_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            dataset_name = dataset_dir.name
            
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                model_name = model_dir.name
                
                for target_dir in model_dir.iterdir():
                    if not target_dir.is_dir():
                        continue
                        
                    target_name = target_dir.name
                    
                    # Look for results file (try different possible names)
                    for results_file in ['training_results.json', 'results.json', 'metrics.json']:
                        results_path = target_dir / results_file
                        if results_path.exists():
                            try:
                                with open(results_path, 'r') as f:
                                    results = json.load(f)
                                performance_data[dataset_name][model_name][target_name] = results
                                break
                            except Exception as e:
                                print(f"Error loading {results_path}: {e}")
                                continue
                                
        return performance_data
    
    def create_overall_performance_heatmap(self, save_format='png'):
        """Create a comprehensive heatmap showing performance across all models and datasets."""
        print("Creating overall performance heatmap...")
        
        # Prepare data for heatmap
        heatmap_data = []
        
        for dataset_name, models in self.performance_data.items():
            if dataset_name not in DATASET_CONFIG:
                continue
                
            for model_name, targets in models.items():
                if model_name not in MODEL_DISPLAY_NAMES:
                    continue
                    
                # Calculate average accuracy across all targets for this model-dataset combo
                accuracies = []
                for target_name, results in targets.items():
                    if 'test_accuracy' in results:
                        accuracies.append(results['test_accuracy'])
                    elif 'accuracy' in results:
                        accuracies.append(results['accuracy'])
                        
                if accuracies:
                    avg_accuracy = np.mean(accuracies)
                    heatmap_data.append({
                        'Dataset': DATASET_CONFIG[dataset_name]['name'],
                        'Model': MODEL_DISPLAY_NAMES[model_name],
                        'Average_Accuracy': avg_accuracy
                    })
        
        if not heatmap_data:
            print("No performance data found for heatmap!")
            return
            
        # Create DataFrame and pivot for heatmap
        df = pd.DataFrame(heatmap_data)
        heatmap_matrix = df.pivot(index='Model', columns='Dataset', values='Average_Accuracy')
        
        # Create the heatmap
        plt.figure(figsize=(14, 8))
        
        # Create heatmap with custom colormap
        sns.heatmap(heatmap_matrix, 
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlGn',
                   vmin=0.7, 
                   vmax=1.0,
                   center=0.85,
                   cbar_kws={'label': 'Average Test Accuracy'},
                   linewidths=0.5)
        
        plt.title('Model Performance Comparison Across All Datasets\\n(Average Test Accuracy)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Dataset', fontsize=12, fontweight='bold')
        plt.ylabel('Model', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the plot
        filename = f'overall_performance_heatmap.{save_format}'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
    def create_model_ranking_plot(self, save_format='png'):
        """Create a bar plot ranking models by average performance."""
        print("Creating model ranking plot...")
        
        # Calculate average performance for each model across all datasets
        model_scores = defaultdict(list)
        
        for dataset_name, models in self.performance_data.items():
            if dataset_name not in DATASET_CONFIG:
                continue
                
            for model_name, targets in models.items():
                if model_name not in MODEL_DISPLAY_NAMES:
                    continue
                    
                # Get accuracies for this model on this dataset
                accuracies = []
                for target_name, results in targets.items():
                    if 'test_accuracy' in results:
                        accuracies.append(results['test_accuracy'])
                    elif 'accuracy' in results:
                        accuracies.append(results['accuracy'])
                        
                if accuracies:
                    model_scores[model_name].extend(accuracies)
        
        # Calculate statistics for each model
        model_stats = {}
        for model_name, scores in model_scores.items():
            if scores:
                model_stats[model_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'count': len(scores)
                }
        
        if not model_stats:
            print("No model statistics available!")
            return
            
        # Sort models by mean performance
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = [MODEL_DISPLAY_NAMES[model] for model, _ in sorted_models]
        means = [stats['mean'] for _, stats in sorted_models]
        stds = [stats['std'] for _, stats in sorted_models]
        counts = [stats['count'] for _, stats in sorted_models]
        
        # Create bar plot with error bars
        bars = ax.bar(models, means, yerr=stds, capsize=5, 
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Color bars by performance
        colors = plt.cm.RdYlGn([m/max(means) for m in means])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for i, (bar, mean, count) in enumerate(zip(bars, means, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.005,
                   f'{mean:.3f}\\n(n={count})',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Average Test Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Ranking\\n(Average Accuracy Across All Datasets & Targets)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Formatting
        ax.set_ylim(0, min(1.05, max(means) + max(stds) + 0.1))
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        filename = f'model_ranking.{save_format}'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
    def create_dataset_specific_comparison(self, dataset_name, save_format='png'):
        """Create detailed comparison for a specific dataset."""
        if dataset_name not in self.performance_data:
            print(f"No data found for dataset: {dataset_name}")
            return
            
        print(f"Creating comparison for {dataset_name}...")
        
        models = self.performance_data[dataset_name]
        targets = DATASET_CONFIG[dataset_name]['targets']
        
        # Prepare data
        comparison_data = []
        for model_name, model_targets in models.items():
            if model_name not in MODEL_DISPLAY_NAMES:
                continue
                
            for target_name in targets:
                if target_name in model_targets:
                    results = model_targets[target_name]
                    accuracy = results.get('test_accuracy', results.get('accuracy', 0))
                    comparison_data.append({
                        'Model': MODEL_DISPLAY_NAMES[model_name],
                        'Target': target_name.replace('_', ' ').title(),
                        'Accuracy': accuracy
                    })
        
        if not comparison_data:
            print(f"No comparison data for {dataset_name}")
            return
            
        df = pd.DataFrame(comparison_data)
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Pivot for grouped bar plot
        pivot_df = df.pivot(index='Target', columns='Model', values='Accuracy')
        
        # Create the plot
        pivot_df.plot(kind='bar', ax=ax, width=0.8, alpha=0.8)
        
        ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Target Variable', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Performance Comparison: {DATASET_CONFIG[dataset_name]["name"]}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Formatting
        ax.set_ylim(0, 1.05)
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        filename = f'{dataset_name}_model_comparison.{save_format}'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
    def create_target_difficulty_analysis(self, save_format='png'):
        """Analyze which targets are most difficult across datasets."""
        print("Creating target difficulty analysis...")
        
        target_scores = defaultdict(list)
        
        # Collect scores for each target type across all datasets
        for dataset_name, models in self.performance_data.items():
            if dataset_name not in DATASET_CONFIG:
                continue
                
            targets = DATASET_CONFIG[dataset_name]['targets']
            
            for target_name in targets:
                target_accuracies = []
                
                for model_name, model_targets in models.items():
                    if model_name not in MODEL_DISPLAY_NAMES:
                        continue
                        
                    if target_name in model_targets:
                        results = model_targets[target_name]
                        accuracy = results.get('test_accuracy', results.get('accuracy'))
                        if accuracy is not None:
                            target_accuracies.append(accuracy)
                
                if target_accuracies:
                    # Use the best performing model's score for this target
                    target_scores[target_name].append(max(target_accuracies))
        
        # Calculate statistics
        target_stats = {}
        for target_name, scores in target_scores.items():
            if scores:
                target_stats[target_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'count': len(scores)
                }
        
        if not target_stats:
            print("No target statistics available!")
            return
            
        # Sort by difficulty (lower score = more difficult)
        sorted_targets = sorted(target_stats.items(), key=lambda x: x[1]['mean'])
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        targets = [target.replace('_', ' ').title() for target, _ in sorted_targets]
        means = [stats['mean'] for _, stats in sorted_targets]
        stds = [stats['std'] for _, stats in sorted_targets]
        
        # Create horizontal bar plot
        bars = ax.barh(targets, means, xerr=stds, capsize=5, 
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # Color bars by difficulty (red = difficult, green = easy)
        colors = plt.cm.RdYlGn([m for m in means])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for i, (bar, mean) in enumerate(zip(bars, means)):
            width = bar.get_width()
            ax.text(width + stds[i] + 0.01, bar.get_y() + bar.get_height()/2.,
                   f'{mean:.3f}',
                   ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Best Model Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Target Variable', fontsize=12, fontweight='bold')
        ax.set_title('Target Difficulty Analysis\\n(Best Achievable Accuracy per Target)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xlim(0, min(1.05, max(means) + max(stds) + 0.1))
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        filename = f'target_difficulty_analysis.{save_format}'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
    def create_performance_distribution_plot(self, save_format='png'):
        """Create box plots showing performance distribution for each model."""
        print("Creating performance distribution plot...")
        
        # Collect all accuracy scores for each model
        model_scores = defaultdict(list)
        
        for dataset_name, models in self.performance_data.items():
            if dataset_name not in DATASET_CONFIG:
                continue
                
            for model_name, targets in models.items():
                if model_name not in MODEL_DISPLAY_NAMES:
                    continue
                    
                for target_name, results in targets.items():
                    accuracy = results.get('test_accuracy', results.get('accuracy'))
                    if accuracy is not None:
                        model_scores[model_name].append(accuracy)
        
        if not model_scores:
            print("No performance distribution data available!")
            return
            
        # Prepare data for box plot
        plot_data = []
        for model_name, scores in model_scores.items():
            for score in scores:
                plot_data.append({
                    'Model': MODEL_DISPLAY_NAMES[model_name],
                    'Accuracy': score
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create box plot
        sns.boxplot(data=df, x='Model', y='Accuracy', ax=ax, palette='Set2')
        
        # Add individual points
        sns.stripplot(data=df, x='Model', y='Accuracy', ax=ax, 
                     color='black', alpha=0.6, size=4)
        
        ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Performance Distribution Across All Targets\\n(Box Plot with Individual Data Points)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        filename = f'performance_distribution.{save_format}'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
    def generate_all_visualizations(self, save_format='png'):
        """Generate all available visualizations."""
        print("=== PEECOM Model Performance Visualization ===")
        print(f"Output directory: {self.output_dir}")
        print(f"Save format: {save_format}")
        print()
        
        # Overall comparisons
        self.create_overall_performance_heatmap(save_format)
        self.create_model_ranking_plot(save_format)
        self.create_target_difficulty_analysis(save_format)
        self.create_performance_distribution_plot(save_format)
        
        # Dataset-specific comparisons
        available_datasets = list(self.performance_data.keys())
        for dataset_name in available_datasets:
            if dataset_name in DATASET_CONFIG:
                self.create_dataset_specific_comparison(dataset_name, save_format)
        
        print()
        print(f"All visualizations saved to: {self.output_dir}")
        print("Generated plots:")
        for plot_file in sorted(self.output_dir.glob(f"*.{save_format}")):
            print(f"  - {plot_file.name}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive model performance visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_model_comparison.py                    # Generate all plots
  python visualize_model_comparison.py --dataset cmohs    # Only CMOHS plots  
  python visualize_model_comparison.py --save-format pdf  # Save as PDF
  python visualize_model_comparison.py --output-dir viz   # Custom output directory
        """
    )
    
    parser.add_argument('--models-dir', default='output/models',
                       help='Directory containing trained models (default: output/models)')
    parser.add_argument('--output-dir', default='output/figures/model_comparison',
                       help='Output directory for plots (default: output/figures/model_comparison)')
    parser.add_argument('--dataset', type=str, 
                       help='Generate plots for specific dataset only')
    parser.add_argument('--save-format', choices=['png', 'pdf', 'svg'], default='png',
                       help='Save format for plots (default: png)')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ModelPerformanceVisualizer(
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )
    
    if not visualizer.performance_data:
        print("No performance data found! Make sure you have trained models.")
        print(f"Expected directory structure: {args.models_dir}/[dataset]/[model]/[target]/")
        return
    
    if args.dataset:
        # Generate plots for specific dataset
        if args.dataset in visualizer.performance_data:
            visualizer.create_dataset_specific_comparison(args.dataset, args.save_format)
        else:
            print(f"Dataset '{args.dataset}' not found!")
            print(f"Available datasets: {list(visualizer.performance_data.keys())}")
    else:
        # Generate all visualizations
        visualizer.generate_all_visualizations(args.save_format)


if __name__ == '__main__':
    main()