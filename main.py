#!/usr/bin/env python3
"""
PEECOM Main Application

Main entry point for hydraulic system condition monitoring using processed data.
This script loads the processed CSV data and trains machine learning models
for PEECOM hydraulic system condition monitoring.

Usage:
    python main.py --data output/processed_data/cmohs --target stable_flag --model random_forest
    python main.py --data output/processed_data/cmohs --target cooler_condition --model logistic_regression
    python main.py --eval-all --model gradient_boosting  # Evaluate all targets with gradient boosting
    python main.py --list-models  # Show all available models
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


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description='Train model on processed PEECOM data')
    parser.add_argument('--data', type=str, default='output/processed_data/cmohs',
                        help='Path to processed data directory')
    parser.add_argument('--target', type=str, default='stable_flag',
                        help='Target column name')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=get_model_choices(),
                        help='Model type to train')
    parser.add_argument('--output', type=str, default='output/models',
                        help='Output directory for model and results')
    parser.add_argument('--eval-all', action='store_true',
                        help='Evaluate all available targets')
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and exit')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed model information when listing')

    args = parser.parse_args()

    # Handle list models request
    if args.list_models:
        print("PEECOM Available Models")
        print("=" * 50)
        model_loader.list_models(verbose=args.verbose)
        return 0

    print("="*60)
    print("PEECOM HYDRAULIC SYSTEM CONDITION MONITORING")
    print("="*60)
    print(f"Selected Model: {model_loader.get_model_display_name(args.model)}")
    print(f"Data directory: {args.data}")
    print(f"Output directory: {args.output}")
    print(f"Target variable: {args.target}")

    # Check if processed data exists
    if not os.path.exists(args.data):
        print(f"Error: Data directory not found: {args.data}")
        print("Please run dataset_preprocessing.py first to generate processed data.")
        return 1

    try:
        if args.eval_all:
            print(
                f"Training {model_loader.get_model_display_name(args.model)} for all target variables...")
            results = evaluate_all_targets(args.data, args.output, args.model)
        else:
            # Load data
            X, y = load_processed_data(args.data)

            # Prepare target
            target = prepare_targets(y, args.target)

            # Train model using model loader
            model, scaler, results, model_output_dir = train_model_with_loader(
                X, target, args.model, args.output, args.target
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

        return 0

    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
