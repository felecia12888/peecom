#!/usr/bin/env python3
"""
Model Performance Report Generator
=================================

Generates a comprehensive performance report with:
- Performance summary tables
- Best model recommendations  
- Dataset-specific insights
- Model comparison rankings

Usage:
    python generate_performance_report.py
    python generate_performance_report.py --format markdown
    python generate_performance_report.py --output custom_report.md
"""

import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

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

class PerformanceReportGenerator:
    """Generate comprehensive performance reports."""
    
    def __init__(self, models_dir='output/models'):
        """Initialize the report generator."""
        self.models_dir = Path(models_dir)
        self.performance_data = self._load_all_performance_data()
        
    def _load_all_performance_data(self):
        """Load performance data from all trained models."""
        performance_data = defaultdict(lambda: defaultdict(dict))
        
        if not self.models_dir.exists():
            return performance_data
            
        # Alternative directory structure: output/models/[dataset]/[model]/[target]/
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
                    
                    # Look for results file
                    for results_file in ['training_results.json', 'results.json', 'metrics.json']:
                        results_path = target_dir / results_file
                        if results_path.exists():
                            try:
                                with open(results_path, 'r') as f:
                                    results = json.load(f)
                                performance_data[dataset_name][model_name][target_name] = results
                                break
                            except Exception:
                                continue
                                
        return performance_data
    
    def generate_summary_table(self):
        """Generate overall performance summary table."""
        summary_data = []
        
        for dataset_name, models in self.performance_data.items():
            if dataset_name not in DATASET_CONFIG:
                continue
                
            for model_name, targets in models.items():
                if model_name not in MODEL_DISPLAY_NAMES:
                    continue
                    
                # Calculate statistics for this model-dataset combination
                accuracies = []
                target_count = 0
                
                for target_name, results in targets.items():
                    target_count += 1
                    accuracy = results.get('test_accuracy', results.get('accuracy'))
                    if accuracy is not None:
                        accuracies.append(accuracy)
                
                if accuracies:
                    summary_data.append({
                        'Dataset': DATASET_CONFIG[dataset_name]['name'],
                        'Model': MODEL_DISPLAY_NAMES[model_name],
                        'Targets': target_count,
                        'Avg_Accuracy': np.mean(accuracies),
                        'Min_Accuracy': np.min(accuracies),
                        'Max_Accuracy': np.max(accuracies),
                        'Std_Accuracy': np.std(accuracies)
                    })
        
        return pd.DataFrame(summary_data)
    
    def find_best_models(self):
        """Find best performing models for each dataset."""
        best_models = {}
        
        for dataset_name, models in self.performance_data.items():
            if dataset_name not in DATASET_CONFIG:
                continue
                
            model_scores = {}
            
            for model_name, targets in models.items():
                if model_name not in MODEL_DISPLAY_NAMES:
                    continue
                    
                # Calculate average accuracy for this model
                accuracies = []
                for target_name, results in targets.items():
                    accuracy = results.get('test_accuracy', results.get('accuracy'))
                    if accuracy is not None:
                        accuracies.append(accuracy)
                
                if accuracies:
                    model_scores[model_name] = np.mean(accuracies)
            
            if model_scores:
                best_model = max(model_scores.items(), key=lambda x: x[1])
                best_models[dataset_name] = {
                    'model': best_model[0],
                    'accuracy': best_model[1],
                    'display_name': MODEL_DISPLAY_NAMES[best_model[0]]
                }
        
        return best_models
    
    def generate_detailed_results(self):
        """Generate detailed results for each dataset."""
        detailed_results = {}
        
        for dataset_name, models in self.performance_data.items():
            if dataset_name not in DATASET_CONFIG:
                continue
                
            dataset_results = []
            targets = DATASET_CONFIG[dataset_name]['targets']
            
            for target_name in targets:
                target_results = {'Target': target_name.replace('_', ' ').title()}
                
                for model_name in MODEL_DISPLAY_NAMES.keys():
                    if model_name in models and target_name in models[model_name]:
                        results = models[model_name][target_name]
                        accuracy = results.get('test_accuracy', results.get('accuracy', 0))
                        target_results[MODEL_DISPLAY_NAMES[model_name]] = f"{accuracy:.3f}"
                    else:
                        target_results[MODEL_DISPLAY_NAMES[model_name]] = "N/A"
                
                dataset_results.append(target_results)
            
            detailed_results[dataset_name] = pd.DataFrame(dataset_results)
        
        return detailed_results
    
    def generate_markdown_report(self, output_file='model_performance_report.md'):
        """Generate a comprehensive Markdown report."""
        
        # Generate data
        summary_df = self.generate_summary_table()
        best_models = self.find_best_models()
        detailed_results = self.generate_detailed_results()
        
        # Start building the report
        report = []
        report.append("# PEECOM Model Performance Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## 📊 Executive Summary")
        report.append("")
        
        if not summary_df.empty:
            total_models = len(summary_df)
            avg_accuracy = summary_df['Avg_Accuracy'].mean()
            best_overall = summary_df.loc[summary_df['Avg_Accuracy'].idxmax()]
            
            report.append(f"- **Total Model-Dataset Combinations:** {total_models}")
            report.append(f"- **Overall Average Accuracy:** {avg_accuracy:.3f}")
            report.append(f"- **Best Performing Combination:** {best_overall['Model']} on {best_overall['Dataset']} ({best_overall['Avg_Accuracy']:.3f})")
            report.append("")
        
        # Overall Performance Summary
        report.append("## 🏆 Overall Performance Summary")
        report.append("")
        
        if not summary_df.empty:
            # Sort by average accuracy
            summary_sorted = summary_df.sort_values('Avg_Accuracy', ascending=False)
            report.append(summary_sorted.to_markdown(index=False, floatfmt='.3f'))
            report.append("")
        
        # Best Models by Dataset
        report.append("## 🥇 Best Models by Dataset")
        report.append("")
        
        for dataset_name, best_info in best_models.items():
            dataset_display = DATASET_CONFIG[dataset_name]['name']
            report.append(f"- **{dataset_display}:** {best_info['display_name']} ({best_info['accuracy']:.3f})")
        report.append("")
        
        # Model Rankings
        report.append("## 📈 Model Rankings (Overall)")
        report.append("")
        
        if not summary_df.empty:
            # Calculate average performance by model across all datasets
            model_avg = summary_df.groupby('Model')['Avg_Accuracy'].agg(['mean', 'std', 'count']).round(3)
            model_avg = model_avg.sort_values('mean', ascending=False)
            model_avg.columns = ['Average Accuracy', 'Standard Deviation', 'Dataset Count']
            
            report.append(model_avg.to_markdown(floatfmt='.3f'))
            report.append("")
        
        # Detailed Results by Dataset
        report.append("## 📋 Detailed Results by Dataset")
        report.append("")
        
        for dataset_name, results_df in detailed_results.items():
            dataset_display = DATASET_CONFIG[dataset_name]['name']
            report.append(f"### {dataset_display}")
            report.append("")
            
            if not results_df.empty:
                report.append(results_df.to_markdown(index=False))
                report.append("")
        
        # Performance Insights
        report.append("## 🔍 Performance Insights")
        report.append("")
        
        if not summary_df.empty:
            # Find most difficult datasets
            dataset_difficulty = summary_df.groupby('Dataset')['Avg_Accuracy'].mean().sort_values()
            most_difficult = dataset_difficulty.index[0]
            easiest = dataset_difficulty.index[-1]
            
            report.append(f"- **Most Challenging Dataset:** {most_difficult} (avg: {dataset_difficulty[most_difficult]:.3f})")
            report.append(f"- **Easiest Dataset:** {easiest} (avg: {dataset_difficulty[easiest]:.3f})")
            report.append("")
            
            # Model consistency analysis
            model_consistency = summary_df.groupby('Model')['Std_Accuracy'].mean().sort_values()
            most_consistent = model_consistency.index[0]
            least_consistent = model_consistency.index[-1]
            
            report.append(f"- **Most Consistent Model:** {most_consistent} (std: {model_consistency[most_consistent]:.3f})")
            report.append(f"- **Least Consistent Model:** {least_consistent} (std: {model_consistency[least_consistent]:.3f})")
            report.append("")
        
        # Recommendations
        report.append("## 💡 Recommendations")
        report.append("")
        
        if best_models:
            report.append("### Dataset-Specific Recommendations")
            for dataset_name, best_info in best_models.items():
                dataset_display = DATASET_CONFIG[dataset_name]['name']
                report.append(f"- **{dataset_display}:** Use {best_info['display_name']} for best performance")
            report.append("")
        
        if not summary_df.empty:
            # Overall best model
            overall_best = summary_df.groupby('Model')['Avg_Accuracy'].mean().sort_values(ascending=False)
            if len(overall_best) > 0:
                report.append("### General Recommendations")
                report.append(f"- **Best Overall Model:** {overall_best.index[0]} (average accuracy: {overall_best.iloc[0]:.3f})")
                report.append(f"- **Most Versatile Model:** Consider Random Forest for balanced performance across datasets")
                report.append(f"- **Physics-Aware Applications:** Use PEECOM for hydraulic and mechanical systems")
                report.append("")
        
        # Appendix
        report.append("## 📎 Appendix")
        report.append("")
        report.append("### Dataset Information")
        report.append("")
        report.append("| Dataset | Full Name | Targets |")
        report.append("|---------|-----------|---------|")
        
        for dataset_id, config in DATASET_CONFIG.items():
            targets_str = ", ".join(config['targets'])
            report.append(f"| {dataset_id} | {config['name']} | {targets_str} |")
        
        report.append("")
        report.append("### Model Information")
        report.append("")
        report.append("| Model ID | Full Name | Description |")
        report.append("|----------|-----------|-------------|")
        report.append("| peecom | PEECOM (Physics-Enhanced) | Custom model with domain-specific physics features |")
        report.append("| random_forest | Random Forest | Ensemble decision trees with feature importance |")
        report.append("| logistic_regression | Logistic Regression | Fast, interpretable linear classifier |")
        report.append("| svm | Support Vector Machine | Robust classifier for high-dimensional data |")
        report.append("")
        
        # Write the report
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\\n'.join(report))
        
        return output_path
    
    def generate_csv_summary(self, output_file='model_performance_summary.csv'):
        """Generate a CSV summary for further analysis."""
        summary_df = self.generate_summary_table()
        
        if not summary_df.empty:
            output_path = Path(output_file)
            summary_df.to_csv(output_path, index=False)
            return output_path
        
        return None

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive model performance report'
    )
    
    parser.add_argument('--models-dir', default='output/models',
                       help='Directory containing trained models')
    parser.add_argument('--output', default='model_performance_report.md',
                       help='Output file for the report')
    parser.add_argument('--format', choices=['markdown', 'csv'], default='markdown',
                       help='Output format')
    
    args = parser.parse_args()
    
    # Create report generator
    generator = PerformanceReportGenerator(models_dir=args.models_dir)
    
    if not generator.performance_data:
        print("No performance data found!")
        print(f"Expected directory: {args.models_dir}")
        return
    
    print("📊 Generating model performance report...")
    
    if args.format == 'markdown':
        output_path = generator.generate_markdown_report(args.output)
        print(f"✅ Markdown report saved to: {output_path}")
    elif args.format == 'csv':
        output_path = generator.generate_csv_summary(args.output)
        if output_path:
            print(f"✅ CSV summary saved to: {output_path}")
        else:
            print("❌ No data available for CSV export")

if __name__ == '__main__':
    main()