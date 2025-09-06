#!/usr/bin/env python3
"""
Enhanced PEECOM Training Script

This script provides enhanced training with the new physics-aware PEECOM model,
proper data handling, cross-validation, and comprehensive evaluation.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import json
import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

try:
    from src.models.peecom_enhanced import EnhancedPEECOMClassifier, DataLeakageDetector
    from src.models.model_loader import ModelLoader
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_enhanced_training_parser():
    """Create argument parser for enhanced training"""
    parser = argparse.ArgumentParser(
        description='Enhanced PEECOM model training')

    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name to train on')
    parser.add_argument('--model', type=str, default='peecom',
                        choices=['peecom', 'peecom_unified',
                                 'random_forest', 'logistic_regression', 'svm'],
                        help='Model to train')
    parser.add_argument('--target', type=str, default=None,
                        help='Specific target to train on (if None, train on all targets)')
    parser.add_argument('--eval-all', action='store_true',
                        help='Evaluate on all available targets')
    parser.add_argument('--use-physics', action='store_true', default=True,
                        help='Use physics-enhanced features (for PEECOM)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')

    return parser


class EnhancedModelTrainer:
    """Enhanced model trainer with comprehensive evaluation"""

    def __init__(self, output_dir='output', verbose=True):
        self.output_dir = output_dir
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_loader = ModelLoader()

    def load_processed_data(self, dataset_name):
        """Load processed dataset"""
        dataset_dir = os.path.join(
            self.output_dir, 'processed_data', dataset_name)

        # Try to load split data first
        if os.path.exists(os.path.join(dataset_dir, 'X_train.csv')):
            self.logger.info("Loading split dataset...")
            X_train = pd.read_csv(os.path.join(dataset_dir, 'X_train.csv'))
            y_train = pd.read_csv(os.path.join(dataset_dir, 'y_train.csv'))
            X_test = pd.read_csv(os.path.join(dataset_dir, 'X_test.csv'))
            y_test = pd.read_csv(os.path.join(dataset_dir, 'y_test.csv'))

            # Load validation set if available
            X_val, y_val = None, None
            if os.path.exists(os.path.join(dataset_dir, 'X_val.csv')):
                X_val = pd.read_csv(os.path.join(dataset_dir, 'X_val.csv'))
                y_val = pd.read_csv(os.path.join(dataset_dir, 'y_val.csv'))

            return {
                'train': (X_train, y_train),
                'test': (X_test, y_test),
                'val': (X_val, y_val) if X_val is not None else None
            }

        # Fallback to full dataset
        elif os.path.exists(os.path.join(dataset_dir, 'X_full.csv')):
            self.logger.info(
                "Loading full dataset (will create splits during training)...")
            X_full = pd.read_csv(os.path.join(dataset_dir, 'X_full.csv'))
            y_full = pd.read_csv(os.path.join(dataset_dir, 'y_full.csv'))
            return {'full': (X_full, y_full)}

        else:
            raise FileNotFoundError(
                f"No processed data found in {dataset_dir}")

    def prepare_data_for_training(self, data_dict, target_name=None):
        """Prepare data for training"""
        # Handle full dataset case
        if 'full' in data_dict:
            X_full, y_full = data_dict['full']

            # Remove potential leakage features
            target_names = list(y_full.columns) if hasattr(
                y_full, 'columns') else []
            X_clean, removed_features = DataLeakageDetector.remove_leakage_features(
                X_full, y_full, target_names, self.logger
            )

            # Select target
            if target_name and target_name in y_full.columns:
                y_target = y_full[target_name]
            elif y_full.shape[1] == 1:
                y_target = y_full.iloc[:, 0]
            else:
                y_target = y_full.iloc[:, 0]  # Default to first target
                if self.verbose:
                    self.logger.warning(
                        f"Multiple targets available. Using first target: {y_full.columns[0]}")

            # Create train/test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_target, test_size=0.2, random_state=42,
                stratify=y_target if len(np.unique(y_target)) > 1 else None
            )

            return X_train, X_test, y_train, y_test, removed_features

        # Handle split dataset case
        else:
            X_train, y_train = data_dict['train']
            X_test, y_test = data_dict['test']

            # Remove potential leakage features
            target_names = list(y_train.columns) if hasattr(
                y_train, 'columns') else []
            X_train_clean, removed_features = DataLeakageDetector.remove_leakage_features(
                X_train, y_train, target_names, self.logger
            )
            X_test_clean = X_test.drop(
                columns=removed_features, errors='ignore')

            # Select target
            if target_name and target_name in y_train.columns:
                y_train_target = y_train[target_name]
                y_test_target = y_test[target_name]
            elif y_train.shape[1] == 1:
                y_train_target = y_train.iloc[:, 0]
                y_test_target = y_test.iloc[:, 0]
            else:
                y_train_target = y_train.iloc[:, 0]
                y_test_target = y_test.iloc[:, 0]

            return X_train_clean, X_test_clean, y_train_target, y_test_target, removed_features

    def train_model(self, model_name, X_train, X_test, y_train, y_test,
                    target_name='target', cv_folds=5, **model_params):
        """Train and evaluate a model"""
        self.logger.info(f"Training {model_name} on target: {target_name}")

        # Get model class and create instance
        model_info = self.model_loader.get_model_info(model_name)
        model_class = model_info['class']

        # Create model with parameters
        if model_name == 'peecom':
            model = model_class(
                use_physics_features=model_params.get('use_physics', True),
                cv_folds=cv_folds,
                random_state=model_params.get('random_state', 42),
                verbose=self.verbose
            )
        else:
            model = model_class(**model_params)

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring='accuracy')

        # Get feature importance if available
        feature_importance = {}
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
        elif hasattr(model, 'feature_importances_'):
            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_))

        # Physics analysis (for PEECOM)
        physics_analysis = {}
        if hasattr(model, 'get_physics_feature_analysis'):
            physics_analysis = model.get_physics_feature_analysis()

        results = {
            'model_name': model_name,
            'model_display_name': model_info.get('display_name', model_name),
            'target_name': target_name,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'feature_importance': feature_importance,
            'physics_analysis': physics_analysis,
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
            'model_params': model_params,
            'feature_names': list(X_train.columns)
        }

        if self.verbose:
            self.logger.info(f"Train Accuracy: {train_accuracy:.4f}")
            self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            self.logger.info(
                f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return model, results

    def save_model_results(self, model, results, dataset_name):
        """Save model and results"""
        model_name = results['model_name']
        target_name = results['target_name']

        # Create output directory
        model_dir = os.path.join(
            self.output_dir, 'models', model_name, target_name)
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_file = os.path.join(model_dir, f'{model_name}_model.joblib')
        joblib.dump(model, model_file)

        # Save scaler if available
        if hasattr(model, 'scaler'):
            scaler_file = os.path.join(
                model_dir, f'{model_name}_scaler.joblib')
            joblib.dump(model.scaler, scaler_file)
            results['scaler_file'] = scaler_file

        results['model_file'] = model_file

        # Save detailed results
        results_file = os.path.join(model_dir, 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save feature importance
        if results['feature_importance']:
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v}
                for k, v in results['feature_importance'].items()
            ]).sort_values('importance', ascending=False)
            importance_df.to_csv(os.path.join(
                model_dir, 'feature_importance.csv'), index=False)

        # Save training summary
        self._save_training_summary(results, model_dir, dataset_name)

        self.logger.info(f"Model saved to: {model_dir}")
        return model_dir

    def _save_training_summary(self, results, model_dir, dataset_name):
        """Save human-readable training summary"""
        summary_file = os.path.join(model_dir, 'training_summary.txt')

        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENHANCED PEECOM TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write("TRAINING INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model: {results['model_display_name']}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Target Variable: {results['target_name']}\n")
            f.write(f"Training Date: {datetime.now().isoformat()}\n")
            f.write(f"Number of Features: {len(results['feature_names'])}\n\n")

            f.write("MODEL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"Training Accuracy: {results['train_accuracy']:.4f} ({results['train_accuracy']*100:.2f}%)\n")
            f.write(
                f"Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)\n")
            f.write(
                f"Cross-Validation Mean: {results['cv_mean']:.4f} ({results['cv_mean']*100:.2f}%)\n")
            f.write(
                f"Cross-Validation Std: ±{results['cv_std']:.4f} (±{results['cv_std']*100:.2f}%)\n\n")

            # Performance assessment
            f.write("PERFORMANCE ASSESSMENT\n")
            f.write("-" * 40 + "\n")
            test_acc = results['test_accuracy']
            train_test_gap = abs(
                results['train_accuracy'] - results['test_accuracy'])
            cv_stability = results['cv_std']

            if test_acc >= 0.95:
                f.write("✓ EXCELLENT: Test accuracy ≥ 95%\n")
            elif test_acc >= 0.85:
                f.write("✓ GOOD: Test accuracy ≥ 85%\n")
            else:
                f.write("⚠ NEEDS IMPROVEMENT: Test accuracy < 85%\n")

            if train_test_gap <= 0.05:
                f.write("✓ LOW OVERFITTING: Train-Test gap ≤ 5%\n")
            else:
                f.write("⚠ POTENTIAL OVERFITTING: Train-Test gap > 5%\n")

            if cv_stability <= 0.05:
                f.write("✓ STABLE: CV std ≤ 5%\n")
            else:
                f.write("⚠ UNSTABLE: CV std > 5%\n")

            f.write("\n")

            # Physics analysis (if available)
            if results.get('physics_analysis'):
                physics = results['physics_analysis']
                f.write("PHYSICS FEATURE ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"Physics Features: {len(physics.get('physics_features', []))}\n")
                f.write(
                    f"Standard Features: {len(physics.get('standard_features', []))}\n")
                f.write(
                    f"Physics Avg Importance: {physics.get('physics_avg_importance', 0):.4f}\n")
                f.write(
                    f"Standard Avg Importance: {physics.get('standard_avg_importance', 0):.4f}\n")
                f.write(
                    f"Physics Advantage: {physics.get('physics_advantage', 1):.2f}x\n\n")

                if physics.get('top_physics_features'):
                    f.write("TOP PHYSICS FEATURES:\n")
                    for i, (feature, importance) in enumerate(physics['top_physics_features'][:5]):
                        f.write(f"  {i+1}. {feature}: {importance:.4f}\n")
                    f.write("\n")

            # Feature importance
            if results['feature_importance']:
                f.write("TOP 10 MOST IMPORTANT FEATURES\n")
                f.write("-" * 40 + "\n")
                sorted_features = sorted(results['feature_importance'].items(),
                                         key=lambda x: x[1], reverse=True)
                for i, (feature, importance) in enumerate(sorted_features[:10]):
                    f.write(
                        f"{i+1:2d}. {feature:<20} - {importance:.4f} ({importance*100:.2f}%)\n")
                f.write("\n")

            f.write("SAVED FILES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model: {results.get('model_file', 'N/A')}\n")
            f.write(f"Scaler: {results.get('scaler_file', 'N/A')}\n")
            f.write(
                f"Results: {os.path.join(model_dir, 'training_results.json')}\n")
            f.write(
                f"Feature Importance: {os.path.join(model_dir, 'feature_importance.csv')}\n")

            f.write("\n" + "="*80 + "\n")


def enhanced_training_main():
    """Enhanced training main function"""
    parser = create_enhanced_training_parser()
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("enhanced_training")

    logger.info(f"Enhanced training: {args.model} on {args.dataset}")

    # Initialize trainer
    trainer = EnhancedModelTrainer(args.output_dir, args.verbose)

    try:
        # Load data
        data_dict = trainer.load_processed_data(args.dataset)
        logger.info(f"Loaded data for dataset: {args.dataset}")

        # Determine targets to train on
        if 'full' in data_dict:
            _, y_sample = data_dict['full']
        else:
            _, y_sample = data_dict['train']

        available_targets = list(y_sample.columns) if hasattr(
            y_sample, 'columns') else ['target']

        if args.target and args.target in available_targets:
            targets_to_train = [args.target]
        elif args.eval_all:
            targets_to_train = available_targets
        else:
            # Default to first target
            targets_to_train = [available_targets[0]]

        logger.info(f"Training on targets: {targets_to_train}")

        # Train models
        all_results = {}
        for target_name in targets_to_train:
            logger.info(f"\nTraining on target: {target_name}")

            # Prepare data
            X_train, X_test, y_train, y_test, removed_features = trainer.prepare_data_for_training(
                data_dict, target_name
            )

            # Train model
            model_params = {
                'use_physics': args.use_physics,
                'random_state': args.random_state
            }

            model, results = trainer.train_model(
                args.model, X_train, X_test, y_train, y_test,
                target_name=target_name, cv_folds=args.cv_folds,
                **model_params
            )

            # Save results
            model_dir = trainer.save_model_results(
                model, results, args.dataset)
            all_results[target_name] = results

            logger.info(f"Completed training for target: {target_name}")

        # Save summary of all targets
        if len(all_results) > 1:
            summary_file = os.path.join(
                args.output_dir, 'models', args.model, 'all_targets_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"All targets summary saved to: {summary_file}")

        logger.info("Enhanced training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    enhanced_training_main()
