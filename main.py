#!/usr/bin/env python3
"""
PEECOM Main Application

Main entry point for hydraulic system condition monitoring using processed data.
This script loads the processed CSV data and trains machine learning models
for PEECOM hydraulic system condition monitoring.

Usage:
    python main.py --dataset equipmentad --target stable_flag --model random_forest
    python main.py --dataset cmohs --target cooler_condition --model logistic_regression
    python main.py --dataset mlclassem --eval-all --model gradient_boosting  # Evaluate all targets
    python main.py --dataset motorvd --model peecom --eval-all  # Use PEECOM model on motor vibration data
    python main.py --list-models  # Show all available models
    python main.py --list-datasets  # Show all available processed datasets
    
    # Alternative: specify data path directly
    python main.py --data output/processed_data/cmohs --target stable_flag --model random_forest
"""

from src.utils.training_utils import load_processed_data, prepare_targets, train_model_with_loader, evaluate_all_targets
from src.models.model_loader import model_loader, get_model_choices
import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import utilities


def get_available_datasets(processed_data_root='output/processed_data'):
    """Get list of available processed datasets"""
    datasets = []
    if os.path.exists(processed_data_root):
        for item in os.listdir(processed_data_root):
            item_path = os.path.join(processed_data_root, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if it has the required CSV files
                x_file = os.path.join(item_path, 'X_full.csv')
                y_file = os.path.join(item_path, 'y_full.csv')
                if os.path.exists(x_file) and os.path.exists(y_file):
                    datasets.append(item)
    return sorted(datasets)


def main():
    """Main training function"""
    # Get available datasets for choices
    available_datasets = get_available_datasets()

    parser = argparse.ArgumentParser(
        description='Train model on processed PEECOM data')
    parser.add_argument('--dataset', type=str, default='cmohs',
                        choices=available_datasets if available_datasets else [
                            'cmohs'],
                        help='Dataset name to use for training (auto-detects processed datasets)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to processed data directory (overrides --dataset)')
    parser.add_argument('--target', type=str, default='stable_flag',
                        help='Target column name')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=get_model_choices(),
                        help='Model type to train')
    parser.add_argument('--output', type=str, default='output/models',
                        help='Output directory for model and results')
    parser.add_argument('--eval-all', action='store_true',
                        help='Evaluate all available targets')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations after training')
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and exit')
    parser.add_argument('--list-datasets', action='store_true',
                        help='List all available datasets and exit')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed model information when listing')

    args = parser.parse_args()

    # Handle list datasets request
    if args.list_datasets:
        print("PEECOM Available Processed Datasets")
        print("=" * 50)
        if available_datasets:
            for dataset in available_datasets:
                dataset_path = os.path.join('output/processed_data', dataset)
                print(f"- {dataset}: {dataset_path}")
                if args.verbose:
                    x_file = os.path.join(dataset_path, 'X_full.csv')
                    y_file = os.path.join(dataset_path, 'y_full.csv')
                    if os.path.exists(x_file) and os.path.exists(y_file):
                        import pandas as pd
                        X_info = pd.read_csv(x_file, nrows=1)
                        y_info = pd.read_csv(y_file, nrows=1)
                        print(
                            f"  Features: {len(X_info.columns)}, Samples: {len(pd.read_csv(x_file))}")
                        print(f"  Targets: {list(y_info.columns)}")
        else:
            print("No processed datasets found. Run dataset_preprocessing.py first.")
        return 0

    # Handle list models request
    if args.list_models:
        print("PEECOM Available Models")
        print("=" * 50)
        model_loader.list_models(verbose=args.verbose)
        return 0

    # Determine data path
    if args.data:
        data_path = args.data
        dataset_name = os.path.basename(data_path)
    else:
        dataset_name = args.dataset
        data_path = os.path.join('output/processed_data', dataset_name)

    print("="*60)
    print("PEECOM HYDRAULIC SYSTEM CONDITION MONITORING")
    print("="*60)
    print(f"Selected Model: {model_loader.get_model_display_name(args.model)}")
    print(f"Dataset: {dataset_name}")
    print(f"Data directory: {data_path}")
    print(f"Output directory: {args.output}")
    print(f"Target variable: {args.target}")

    # Check if processed data exists
    if not os.path.exists(data_path):
        print(f"Error: Data directory not found: {data_path}")
        print("Available datasets:")
        for dataset in available_datasets:
            print(f"  - {dataset}")
        print("Please run dataset_preprocessing.py first to generate processed data.")
        return 1

    try:
        if args.eval_all:
            print(
                f"Training {model_loader.get_model_display_name(args.model)} for all target variables...")
            results = evaluate_all_targets(
                data_path, args.output, args.model, dataset_name)
        else:
            # Load data
            X, y = load_processed_data(data_path)

            # Prepare target
            target_result = prepare_targets(y, args.target)
            if isinstance(target_result, tuple):
                target, mask = target_result
                if mask is not None:
                    # Filter X as well if we filtered the target
                    X = X[mask]
            else:
                target = target_result

            # Train model using model loader
            model, scaler, results, model_output_dir = train_model_with_loader(
                X, target, args.model, args.output, args.target, dataset_name
            )

            # Save results using results handler
            from src.utils.results_handler import save_training_results
            # Use the correct feature names from the results, not the original X.columns
            feature_names_for_saving = results.get(
                'feature_names', list(X.columns))
            save_training_results(
                results, feature_names_for_saving, model_output_dir)

        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        if not args.eval_all:
            print(f"Model: {results['model_display_name']}")
            print(f"Target: {results['target_name']}")
            print(f"Model accuracy: {results['test_accuracy']:.4f}")
            print(
                f"CV score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
        print(f"Results saved to: {args.output}")

        # Generate visualizations if requested
        if args.visualize:
            print("\n" + "="*60)
            print("GENERATING VISUALIZATIONS...")
            print("="*60)
            try:
                # Import visualization system
                from src.visualization.visualize_models import PeecomVisualizationSystem

                viz_system = PeecomVisualizationSystem(
                    base_output_dir='output')

                if args.eval_all:
                    print(
                        f"Generating comprehensive visualizations for {args.model}...")
                    viz_system.visualize_model_all_targets(args.model)
                else:
                    print(
                        f"Generating visualizations for {args.model} → {args.target}...")
                    viz_system.visualize_model_target(args.model, args.target)

                print("✅ VISUALIZATION GENERATION COMPLETED!")

            except Exception as viz_error:
                print(f"⚠️  Visualization generation failed: {viz_error}")
                print(
                    "Model training was successful. You can generate visualizations manually using:")
                if args.eval_all:
                    print(
                        f"python visualize_models.py --model {args.model} --eval-all")
                else:
                    print(
                        f"python visualize_models.py --model {args.model} --target {args.target}")

        return 0

    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
