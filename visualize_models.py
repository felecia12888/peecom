#!/usr/bin/env python3
"""
PEECOM Model Visualization System
Advanced visualization for individual model analysis and physics-enhanced features
"""

import os
import sys
import argparse
import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class ModelVisualizer:
    """Comprehensive model visualization system"""
    
    def __init__(self, output_dir="output", figures_dir="output/figures"):
        self.output_dir = Path(output_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
    def find_trained_models(self, dataset=None, model=None, target=None):
        """Find all trained models matching criteria"""
        models_dir = self.output_dir / "models"
        if not models_dir.exists():
            print(f"❌ No models directory found at {models_dir}")
            return []
        
        found_models = []
        
        # Search through model structure: models/{dataset}/{model}/{target}/
        for dataset_dir in models_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            if dataset and dataset_dir.name != dataset:
                continue
                
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                if model and model_dir.name != model:
                    continue
                    
                for target_dir in model_dir.iterdir():
                    if not target_dir.is_dir():
                        continue
                    if target and target_dir.name != target:
                        continue
                    
                    # Check if this is a valid trained model
                    model_file = target_dir / f"{model_dir.name}_model.joblib"
                    results_file = target_dir / "training_results.json"
                    
                    if model_file.exists() and results_file.exists():
                        found_models.append({
                            'dataset': dataset_dir.name,
                            'model': model_dir.name,
                            'target': target_dir.name,
                            'path': target_dir,
                            'model_file': model_file,
                            'results_file': results_file
                        })
        
        return found_models
    
    def load_model_results(self, model_info):
        """Load model training results and metrics"""
        try:
            with open(model_info['results_file'], 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            print(f"❌ Error loading results for {model_info['model']}/{model_info['target']}: {e}")
            return None
    
    def create_feature_importance_plot(self, model_info, save_pdf=True, save_png=True):
        """Create feature importance visualization"""
        try:
            # Load results
            results = self.load_model_results(model_info)
            if not results or 'feature_importance' not in results:
                print(f"⚠️ No feature importance data for {model_info['model']}/{model_info['target']}")
                return None
            
            importance_data = results['feature_importance']
            if not importance_data:
                print(f"⚠️ Empty feature importance for {model_info['model']}/{model_info['target']}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(importance_data)
            
            # Sort by importance and take top 15
            df = df.sort_values('importance', ascending=True).tail(15)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            bars = ax.barh(range(len(df)), df['importance'], color='skyblue', alpha=0.7)
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df['feature'], fontsize=10)
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_title(f'Feature Importance - {model_info["model"].upper()} Model\n'
                        f'Dataset: {model_info["dataset"]} | Target: {model_info["target"]}', 
                        fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=9)
            
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save files
            output_dir = model_info['path'] / 'figures'
            output_dir.mkdir(exist_ok=True)
            
            filename = f"{model_info['model']}_{model_info['target']}_feature_importance"
            
            if save_pdf:
                pdf_path = output_dir / f"{filename}.pdf"
                plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
                print(f"✅ Saved: {pdf_path}")
            
            if save_png:
                png_path = output_dir / f"{filename}.png"
                plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
                print(f"✅ Saved: {png_path}")
            
            plt.close()
            return fig
            
        except Exception as e:
            print(f"❌ Error creating feature importance plot: {e}")
            return None
    
    def create_peecom_physics_analysis(self, model_info, save_pdf=True, save_png=True):
        """Create PEECOM-specific physics analysis plots"""
        if model_info['model'] != 'peecom':
            return None
        
        try:
            results = self.load_model_results(model_info)
            if not results or 'feature_importance' not in results:
                return None
            
            importance_data = results['feature_importance']
            df = pd.DataFrame(importance_data)
            
            # Identify physics features (created by PEECOM)
            physics_features = []
            standard_features = []
            
            for _, row in df.iterrows():
                feature_name = row['feature']
                # Physics features typically have combinations or calculations
                if any(keyword in feature_name.lower() for keyword in 
                      ['hydraulic', 'pressure_diff', 'thermal', 'ratio', 'efficiency', 'power']):
                    physics_features.append(row)
                else:
                    standard_features.append(row)
            
            if not physics_features:
                print(f"⚠️ No physics features identified for {model_info['target']}")
                return None
            
            output_dir = model_info['path'] / 'figures'
            output_dir.mkdir(exist_ok=True)
            
            # Plot 1: Physics vs Standard Features Comparison
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            physics_df = pd.DataFrame(physics_features)
            standard_df = pd.DataFrame(standard_features)
            
            avg_physics = physics_df['importance'].mean() if len(physics_df) > 0 else 0
            avg_standard = standard_df['importance'].mean() if len(standard_df) > 0 else 0
            
            categories = ['Physics\nFeatures', 'Standard\nFeatures']
            averages = [avg_physics, avg_standard]
            colors = ['#FF6B6B', '#4ECDC4']
            
            bars = ax1.bar(categories, averages, color=colors, alpha=0.7)
            ax1.set_ylabel('Average Feature Importance')
            ax1.set_title(f'Physics vs Standard Features - {model_info["target"]}', 
                         fontweight='bold', fontsize=14)
            
            # Add value labels
            for bar, avg in zip(bars, averages):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{avg:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax1.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            filename1 = f"peecom_{model_info['target']}_physics_vs_standard"
            if save_pdf:
                plt.savefig(output_dir / f"{filename1}.pdf", format='pdf', dpi=300, bbox_inches='tight')
            if save_png:
                plt.savefig(output_dir / f"{filename1}.png", format='png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filename1}")
            plt.close()
            
            # Plot 2: Top Physics Features
            if len(physics_df) > 0:
                fig2, ax2 = plt.subplots(figsize=(12, 8))
                
                top_physics = physics_df.sort_values('importance', ascending=True).tail(10)
                
                bars = ax2.barh(range(len(top_physics)), top_physics['importance'], 
                               color='#FF6B6B', alpha=0.7)
                ax2.set_yticks(range(len(top_physics)))
                ax2.set_yticklabels(top_physics['feature'], fontsize=10)
                ax2.set_xlabel('Feature Importance')
                ax2.set_title(f'Top Physics-Enhanced Features - {model_info["target"]}', 
                             fontweight='bold', fontsize=14)
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                           f'{width:.3f}', ha='left', va='center', fontsize=9)
                
                ax2.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                filename2 = f"peecom_{model_info['target']}_top_physics_features"
                if save_pdf:
                    plt.savefig(output_dir / f"{filename2}.pdf", format='pdf', dpi=300, bbox_inches='tight')
                if save_png:
                    plt.savefig(output_dir / f"{filename2}.png", format='png', dpi=300, bbox_inches='tight')
                print(f"✅ Saved: {filename2}")
                plt.close()
            
            # Plot 3: Feature Distribution Pie Chart
            fig3, ax3 = plt.subplots(figsize=(8, 8))
            
            counts = [len(physics_features), len(standard_features)]
            labels = [f'Physics Features\n({len(physics_features)})', 
                     f'Standard Features\n({len(standard_features)})']
            colors = ['#FF6B6B', '#4ECDC4']
            
            wedges, texts, autotexts = ax3.pie(counts, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            ax3.set_title(f'Feature Distribution - {model_info["target"]}', 
                         fontweight='bold', fontsize=14)
            
            filename3 = f"peecom_{model_info['target']}_feature_distribution"
            if save_pdf:
                plt.savefig(output_dir / f"{filename3}.pdf", format='pdf', dpi=300, bbox_inches='tight')
            if save_png:
                plt.savefig(output_dir / f"{filename3}.png", format='png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filename3}")
            plt.close()
            
            # Plot 4: Physics Impact Analysis
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            
            # Get performance metrics
            test_accuracy = results.get('test_accuracy', 0)
            
            # Estimate physics contribution (simplified)
            total_importance = df['importance'].sum()
            physics_contribution = physics_df['importance'].sum() if len(physics_df) > 0 else 0
            physics_percentage = (physics_contribution / total_importance * 100) if total_importance > 0 else 0
            
            metrics = ['Test Accuracy', 'Physics Feature\nContribution']
            values = [test_accuracy * 100, physics_percentage]
            colors = ['#4ECDC4', '#FF6B6B']
            
            bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
            ax4.set_ylabel('Percentage (%)')
            ax4.set_title(f'Physics Impact Analysis - {model_info["target"]}', 
                         fontweight='bold', fontsize=14)
            ax4.set_ylim(0, 100)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax4.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            filename4 = f"peecom_{model_info['target']}_physics_impact"
            if save_pdf:
                plt.savefig(output_dir / f"{filename4}.pdf", format='pdf', dpi=300, bbox_inches='tight')
            if save_png:
                plt.savefig(output_dir / f"{filename4}.png", format='png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filename4}")
            plt.close()
            
            print(f"🔬 Generated 4 PEECOM physics analysis plots for {model_info['target']}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating PEECOM physics analysis: {e}")
            return None
    
    def visualize_model(self, dataset, model, target=None, eval_all=False):
        """Visualize specific model or all targets"""
        print(f"\n🎨 Generating visualizations for {model} on {dataset}")
        
        if eval_all:
            # Find all targets for this model/dataset combination
            models = self.find_trained_models(dataset=dataset, model=model)
            if not models:
                print(f"❌ No trained models found for {model} on {dataset}")
                return
            
            print(f"📊 Found {len(models)} trained models to visualize")
            
            for model_info in models:
                print(f"\n--- Visualizing {model_info['target']} ---")
                
                # Create feature importance plot
                self.create_feature_importance_plot(model_info)
                
                # Create PEECOM-specific plots if applicable
                if model_info['model'] == 'peecom':
                    self.create_peecom_physics_analysis(model_info)
        else:
            # Single target visualization
            models = self.find_trained_models(dataset=dataset, model=model, target=target)
            if not models:
                print(f"❌ No trained model found for {model}/{target} on {dataset}")
                return
            
            model_info = models[0]
            print(f"📊 Visualizing {model_info['target']}")
            
            # Create feature importance plot
            self.create_feature_importance_plot(model_info)
            
            # Create PEECOM-specific plots if applicable
            if model_info['model'] == 'peecom':
                self.create_peecom_physics_analysis(model_info)
        
        print(f"✅ Visualization complete for {model} on {dataset}")
    
    def list_available_models(self):
        """List all available trained models"""
        models = self.find_trained_models()
        if not models:
            print("❌ No trained models found")
            return
        
        print(f"\n📋 Found {len(models)} trained models:")
        
        # Group by dataset and model
        grouped = {}
        for model in models:
            key = f"{model['dataset']}/{model['model']}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(model['target'])
        
        for key, targets in grouped.items():
            dataset, model_name = key.split('/')
            print(f"  📊 {dataset} / {model_name}")
            for target in sorted(targets):
                print(f"    - {target}")
    
    def list_available_targets(self, dataset=None):
        """List available targets for datasets"""
        models = self.find_trained_models(dataset=dataset)
        if not models:
            print(f"❌ No trained models found{' for ' + dataset if dataset else ''}")
            return
        
        # Group by dataset
        datasets = {}
        for model in models:
            if model['dataset'] not in datasets:
                datasets[model['dataset']] = set()
            datasets[model['dataset']].add(model['target'])
        
        print(f"\n📋 Available targets{' for ' + dataset if dataset else ''}:")
        for ds, targets in datasets.items():
            print(f"  📊 {ds}: {', '.join(sorted(targets))}")

def main():
    parser = argparse.ArgumentParser(description='PEECOM Model Visualization System')
    
    # Dataset and model selection
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g., cmohs, equipmentad)')
    parser.add_argument('--model', type=str, help='Model name (e.g., peecom, random_forest)')
    parser.add_argument('--target', type=str, help='Target variable name')
    
    # Evaluation options
    parser.add_argument('--eval-all', action='store_true', 
                       help='Evaluate all targets for specified model/dataset')
    
    # Information commands
    parser.add_argument('--list-models', action='store_true', 
                       help='List all available trained models')
    parser.add_argument('--list-targets', action='store_true', 
                       help='List available targets')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory (default: output)')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ModelVisualizer(output_dir=args.output_dir)
    
    # Handle information commands
    if args.list_models:
        visualizer.list_available_models()
        return
    
    if args.list_targets:
        visualizer.list_available_targets(args.dataset)
        return
    
    # Validate arguments for visualization
    if not args.dataset or not args.model:
        print("❌ Error: --dataset and --model are required for visualization")
        print("💡 Use --list-models to see available options")
        return
    
    if not args.eval_all and not args.target:
        print("❌ Error: Either --target or --eval-all is required")
        return
    
    # Run visualization
    try:
        visualizer.visualize_model(
            dataset=args.dataset,
            model=args.model,
            target=args.target,
            eval_all=args.eval_all
        )
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())