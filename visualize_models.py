#!/usr/bin/env python3
"""
Integrated Visualization System for PEECOM

Comprehensive visualization system that integrates with the model training pipeline.
Supports individual model-target visualization generation that mirrors the training structure.

Usage:
    python visualize_models.py --model peecom --target cooler_condition
    python visualize_models.py --model peecom --eval-all
    python visualize_models.py --generate-all-data-plots
    python visualize_models.py --generate-all
"""

from src.visualization.model_visualizer import ModelVisualizer
from src.visualization.data_visualizer import DataVisualizer
from src.visualization.performance_visualizer import PerformanceVisualizer
import argparse
import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))


class PeecomVisualizationSystem:
    """
    Integrated visualization system that mirrors the model training structure.

    Generates visualizations for specific model-target combinations or comprehensive
    analysis across all models and targets.
    """

    AVAILABLE_MODELS = ['peecom', 'random_forest',
                        'logistic_regression', 'svm', 'gradient_boosting']
    AVAILABLE_TARGETS = ['cooler_condition', 'valve_condition', 'pump_leakage',
                         'accumulator_pressure', 'stable_flag']

    def __init__(self, base_output_dir='output'):
        """
        Initialize the visualization system.

        Args:
            base_output_dir: Base directory containing models and data
        """
        self.base_output_dir = Path(base_output_dir)
        self.models_dir = self.base_output_dir / 'models'
        self.data_dir = Path('dataset/cmohs')

    def visualize_model_target(self, model_name, target_name, output_dir=None):
        """
        Generate visualizations for a specific model-target combination.

        Args:
            model_name: Name of the model
            target_name: Name of the target
            output_dir: Custom output directory (defaults to model dir)
        """
        if output_dir is None:
            output_dir = self.models_dir / model_name / target_name / 'figures'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📊 Generating visualizations for {model_name} → {target_name}")
        print(f"📁 Output: {output_dir}")

        plots_generated = {}

        # 1. Model-specific visualizations
        model_viz = ModelVisualizer(
            output_dir=output_dir, models_dir=self.models_dir)

        # Feature importance for this specific model-target
        print("  ├── Feature importance analysis...")
        try:
            importance_plots = model_viz.create_feature_importance_comparison(
                models=[model_name], target=target_name
            )
            if importance_plots:
                plots_generated['feature_importance'] = importance_plots
        except Exception as e:
            print(f"    ⚠️  Warning: {e}")

        # PEECOM physics analysis if applicable
        if model_name == 'peecom':
            print("  ├── PEECOM physics analysis...")
            try:
                physics_plots = model_viz.create_peecom_physics_analysis(
                    target=target_name)
                if physics_plots:
                    plots_generated['peecom_physics'] = physics_plots
            except Exception as e:
                print(f"    ⚠️  Warning: {e}")

        # 2. Performance visualization for this target
        perf_viz = PerformanceVisualizer(
            output_dir=output_dir, results_dir=self.models_dir)

        print("  ├── Performance comparison...")
        try:
            performance_data = perf_viz.load_performance_data()
            if performance_data:
                # Create target-specific performance plot
                target_plots = perf_viz.create_target_specific_comparison(
                    performance_data)
                if target_plots and target_name in target_plots:
                    plots_generated['performance'] = target_plots[target_name]
        except Exception as e:
            print(f"    ⚠️  Warning: {e}")

        print(
            f"  └── ✅ Generated {len(plots_generated)} visualization categories")

        # Save visualization summary
        summary_file = output_dir / 'visualization_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'model': model_name,
                'target': target_name,
                'plots_generated': list(plots_generated.keys()),
                'output_directory': str(output_dir)
            }, f, indent=2)

        return plots_generated

    def visualize_model_all_targets(self, model_name, output_dir=None):
        """
        Generate visualizations for a model across all targets.

        Args:
            model_name: Name of the model
            output_dir: Custom output directory
        """
        print(f"🎯 Generating comprehensive visualizations for {model_name}")

        all_plots = {}

        # Generate for each target
        for target in self.AVAILABLE_TARGETS:
            model_target_dir = self.models_dir / model_name / target
            if model_target_dir.exists():
                target_plots = self.visualize_model_target(
                    model_name, target, output_dir)
                all_plots[target] = target_plots
            else:
                print(
                    f"  ⚠️  No trained model found for {model_name} → {target}")

        # Generate model-wide comparisons
        if output_dir is None:
            output_dir = self.models_dir / model_name / 'comprehensive_figures'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Model complexity analysis
        model_viz = ModelVisualizer(
            output_dir=output_dir, models_dir=self.models_dir)
        print(f"  └── 📈 Model complexity analysis...")
        try:
            complexity_plots = model_viz.create_model_complexity_comparison()
            if complexity_plots:
                all_plots['complexity'] = complexity_plots
        except Exception as e:
            print(f"    ⚠️  Warning: {e}")

        return all_plots

    def generate_data_analysis(self, output_dir=None):
        """
        Generate comprehensive data analysis visualizations.

        Args:
            output_dir: Custom output directory
        """
        if output_dir is None:
            output_dir = self.base_output_dir / 'data_analysis_figures'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("📊 Generating data analysis visualizations...")
        print(f"📁 Output: {output_dir}")

        data_viz = DataVisualizer(
            output_dir=output_dir, data_dir=self.data_dir)

        plots_generated = {}

        # Generate all data plots
        try:
            all_data_plots = data_viz.generate_all_data_plots()
            if all_data_plots:
                plots_generated.update(all_data_plots)
        except Exception as e:
            print(f"⚠️  Warning: {e}")

        print(f"✅ Generated {len(plots_generated)} data analysis figures")
        return plots_generated

    def generate_comprehensive_analysis(self, output_dir=None):
        """
        Generate comprehensive analysis across all models and targets.

        Args:
            output_dir: Custom output directory
        """
        if output_dir is None:
            output_dir = self.base_output_dir / 'comprehensive_figures'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("🎯 COMPREHENSIVE PEECOM VISUALIZATION ANALYSIS")
        print("=" * 60)

        all_plots = {}

        # 1. Data analysis
        print("\n1. DATA ANALYSIS")
        print("-" * 30)
        data_plots = self.generate_data_analysis(output_dir / 'data_analysis')
        all_plots['data_analysis'] = data_plots

        # 2. Cross-model performance comparison
        print("\n2. PERFORMANCE ANALYSIS")
        print("-" * 30)
        perf_viz = PerformanceVisualizer(output_dir=output_dir / 'performance',
                                         results_dir=self.models_dir)
        try:
            performance_plots = perf_viz.generate_all_performance_plots()
            all_plots['performance_analysis'] = performance_plots
        except Exception as e:
            print(f"⚠️  Warning: {e}")

        # 3. Model analysis
        print("\n3. MODEL ANALYSIS")
        print("-" * 30)
        model_viz = ModelVisualizer(output_dir=output_dir / 'models',
                                    models_dir=self.models_dir)
        try:
            model_plots = model_viz.generate_all_model_plots()
            all_plots['model_analysis'] = model_plots
        except Exception as e:
            print(f"⚠️  Warning: {e}")

        # Generate summary
        summary_file = output_dir / 'comprehensive_summary.json'
        total_plots = sum(
            len(plots) if plots else 0 for plots in all_plots.values())

        with open(summary_file, 'w') as f:
            json.dump({
                'total_figures': total_plots,
                'categories': list(all_plots.keys()),
                'output_directory': str(output_dir),
                'generation_complete': True
            }, f, indent=2)

        print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"📊 Total figures generated: {total_plots}")
        print(f"📁 Output directory: {output_dir}")

        return all_plots


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='PEECOM Integrated Visualization System')

    # Model selection (mirrors main.py structure)
    model_choices = PeecomVisualizationSystem.AVAILABLE_MODELS + ['all']
    target_choices = PeecomVisualizationSystem.AVAILABLE_TARGETS + ['all']

    parser.add_argument('--model', type=str, choices=model_choices,
                        help='Model to visualize (use "all" for all models)')
    parser.add_argument('--target', type=str, choices=target_choices,
                        help='Target to visualize (use "all" for all targets)')

    # Evaluation options (mirrors main.py structure)
    parser.add_argument('--eval-all', action='store_true',
                        help='Generate visualizations for all targets of specified model')

    # Comprehensive options
    parser.add_argument('--generate-all-data-plots', action='store_true',
                        help='Generate all data analysis plots')
    parser.add_argument('--generate-all', action='store_true',
                        help='Generate comprehensive analysis across all models and targets')

    # Output options
    parser.add_argument('--output-dir', type=str,
                        help='Custom output directory')
    parser.add_argument('--base-dir', type=str, default='output',
                        help='Base directory containing models and data')

    # Information options
    parser.add_argument('--list-models', action='store_true',
                        help='List available models')
    parser.add_argument('--list-targets', action='store_true',
                        help='List available targets')

    args = parser.parse_args()

    # Information requests
    if args.list_models:
        print("Available models:")
        for model in PeecomVisualizationSystem.AVAILABLE_MODELS:
            print(f"  - {model}")
        return

    if args.list_targets:
        print("Available targets:")
        for target in PeecomVisualizationSystem.AVAILABLE_TARGETS:
            print(f"  - {target}")
        return

    # Initialize system
    viz_system = PeecomVisualizationSystem(base_output_dir=args.base_dir)

    # Generate visualizations based on arguments
    if args.generate_all or (args.model == 'all' and args.target == 'all'):
        viz_system.generate_comprehensive_analysis(args.output_dir)

    elif args.generate_all_data_plots:
        viz_system.generate_data_analysis(args.output_dir)

    elif args.model == 'all' and args.target and args.target != 'all':
        # Generate for all models on specific target
        for model in PeecomVisualizationSystem.AVAILABLE_MODELS:
            print(f"\n🎯 Processing {model} → {args.target}")
            viz_system.visualize_model_target(
                model, args.target, args.output_dir)

    elif args.model and args.model != 'all' and args.target == 'all':
        # Generate for specific model on all targets (same as --eval-all)
        viz_system.visualize_model_all_targets(args.model, args.output_dir)

    elif args.model and args.eval_all:
        viz_system.visualize_model_all_targets(args.model, args.output_dir)

    elif args.model and args.target and args.model != 'all' and args.target != 'all':
        viz_system.visualize_model_target(
            args.model, args.target, args.output_dir)

    elif args.model and args.model != 'all':
        print(
            f"⚠️  Please specify --target or --eval-all for model {args.model}")
        print("Available targets:", ', '.join(
            PeecomVisualizationSystem.AVAILABLE_TARGETS + ['all']))

    else:
        print("⚠️  Please specify visualization options. Use --help for details.")
        print("\nQuick examples:")
        print("  python visualize_models.py --model peecom --target cooler_condition")
        print("  python visualize_models.py --model peecom --eval-all")
        print("  python visualize_models.py --generate-all")


if __name__ == "__main__":
    main()
