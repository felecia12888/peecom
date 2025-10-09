#!/usr/bin/env python3
"""
PEECOM Argument Parser

Comprehensive argument parser for PEECOM project that allows easy configuration
of all parameters through command line arguments.

Usage Examples:
    # Basic training
    python main.py --mode train --epochs 100 --batch_size 32

    # Data preprocessing only
    python main.py --mode preprocess --dataset_path ./dataset/dataset

    # Model evaluation
    python main.py --mode evaluate --model_path ./models/best_model.h5

    # Custom configuration
    python main.py --mode train --config custom_config.yaml --lr 0.001 --dropout 0.3

    # Dataset analysis
    python main.py --mode analyze --dataset_path ./dataset/dataset --output_dir ./analysis
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class PEECOMArgumentParser:
    """Comprehensive argument parser for PEECOM project"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="PEECOM: Physics-Enhanced Equipment Condition Monitoring",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_examples()
        )
        self._setup_arguments()

    def _get_examples(self) -> str:
        """Get usage examples"""
        return """
Examples:
  # Train a model with default settings
  python main.py --mode train
  
  # Train with custom parameters
  python main.py --mode train --epochs 200 --batch_size 64 --lr 0.001
  
  # Preprocess dataset
  python main.py --mode preprocess --dataset_path ./dataset/dataset
  
  # Analyze dataset
  python main.py --mode analyze --dataset_path ./dataset/dataset
  
  # Evaluate model
  python main.py --mode evaluate --model_path ./models/best_model.h5
  
  # Cross-validation training
  python main.py --mode train --cv_folds 5 --cv_strategy stratified
  
  # Hyperparameter tuning
  python main.py --mode tune --max_trials 50 --tuner_epochs 10
  
  # Custom configuration
  python main.py --config ./configs/custom.yaml --mode train
        """

    def _setup_arguments(self):
        """Setup all command line arguments"""

        # === Main Mode Arguments ===
        self.parser.add_argument(
            "--mode",
            type=str,
            required=True,
            choices=["train", "evaluate", "predict",
                     "preprocess", "analyze", "tune", "pipeline"],
            help="Execution mode"
        )

        self.parser.add_argument(
            "--config",
            type=str,
            default="src/loader/config.yaml",
            help="Path to configuration YAML file (default: src/loader/config.yaml)"
        )

        self.parser.add_argument(
            "--verbose", "-v",
            action="count",
            default=0,
            help="Increase verbosity level (use -v, -vv, -vvv)"
        )

        self.parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="Suppress all output except errors"
        )

        # === Data Arguments ===
        data_group = self.parser.add_argument_group("Data Configuration")
        data_group.add_argument(
            "--dataset_path",
            type=str,
            default="dataset/dataset",
            help="Path to dataset directory (default: dataset/dataset)"
        )

        data_group.add_argument(
            "--output_dir",
            type=str,
            default="output",
            help="Output directory for results (default: output)"
        )

        data_group.add_argument(
            "--test_size",
            type=float,
            default=0.2,
            help="Test set size as fraction (default: 0.2)"
        )

        data_group.add_argument(
            "--val_size",
            type=float,
            default=0.2,
            help="Validation set size as fraction (default: 0.2)"
        )

        data_group.add_argument(
            "--random_state",
            type=int,
            default=42,
            help="Random seed for reproducibility (default: 42)"
        )

        data_group.add_argument(
            "--timesteps",
            type=int,
            default=10,
            help="Number of timesteps for sequences (default: 10)"
        )

        # === Model Arguments ===
        model_group = self.parser.add_argument_group("Model Configuration")
        model_group.add_argument(
            "--model_type",
            type=str,
            default="peecom",
            choices=["peecom", "peecom_base", "peecom_physics", "peecom_adaptive",
                     "random_forest", "gradient_boosting", "logistic_regression",
                     "svm", "lstm", "cnn", "transformer", "hybrid"],
            help="Model architecture type (default: peecom)"
        )

        model_group.add_argument(
            "--model_path",
            type=str,
            help="Path to saved model for evaluation/prediction"
        )

        model_group.add_argument(
            "--hidden_units",
            type=int,
            nargs="+",
            default=[64, 32],
            help="Hidden layer units (default: 64 32)"
        )

        model_group.add_argument(
            "--dropout",
            type=float,
            default=0.3,
            help="Dropout rate (default: 0.3)"
        )

        model_group.add_argument(
            "--l2_reg",
            type=float,
            default=0.01,
            help="L2 regularization strength (default: 0.01)"
        )

        # === Training Arguments ===
        train_group = self.parser.add_argument_group("Training Configuration")
        train_group.add_argument(
            "--epochs",
            type=int,
            default=100,
            help="Number of training epochs (default: 100)"
        )

        train_group.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Batch size for training (default: 32)"
        )

        train_group.add_argument(
            "--lr", "--learning_rate",
            type=float,
            default=0.001,
            help="Learning rate (default: 0.001)"
        )

        train_group.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            choices=["adam", "sgd", "rmsprop", "adagrad"],
            help="Optimizer (default: adam)"
        )

        train_group.add_argument(
            "--early_stopping_patience",
            type=int,
            default=10,
            help="Early stopping patience (default: 10)"
        )

        train_group.add_argument(
            "--reduce_lr_patience",
            type=int,
            default=5,
            help="Reduce LR on plateau patience (default: 5)"
        )

        train_group.add_argument(
            "--class_weight",
            type=str,
            default="balanced",
            choices=["none", "balanced", "custom"],
            help="Class weighting strategy (default: balanced)"
        )

        # === Cross-Validation Arguments ===
        cv_group = self.parser.add_argument_group(
            "Cross-Validation Configuration")
        cv_group.add_argument(
            "--cv_folds",
            type=int,
            default=5,
            help="Number of CV folds (default: 5)"
        )

        cv_group.add_argument(
            "--cv_strategy",
            type=str,
            default="stratified",
            choices=["stratified", "kfold", "group"],
            help="Cross-validation strategy (default: stratified)"
        )

        cv_group.add_argument(
            "--cv_shuffle",
            action="store_true",
            help="Shuffle data in cross-validation"
        )

        # === Hyperparameter Tuning Arguments ===
        tune_group = self.parser.add_argument_group("Hyperparameter Tuning")
        tune_group.add_argument(
            "--tuner",
            type=str,
            default="random",
            choices=["random", "bayesian", "hyperband"],
            help="Hyperparameter tuning algorithm (default: random)"
        )

        tune_group.add_argument(
            "--max_trials",
            type=int,
            default=50,
            help="Maximum number of tuning trials (default: 50)"
        )

        tune_group.add_argument(
            "--tuner_epochs",
            type=int,
            default=10,
            help="Epochs per trial in tuning (default: 10)"
        )

        # === Preprocessing Arguments ===
        preprocess_group = self.parser.add_argument_group(
            "Preprocessing Configuration")
        preprocess_group.add_argument(
            "--scaler",
            type=str,
            default="robust",
            choices=["standard", "robust", "minmax", "none"],
            help="Data scaling method (default: robust)"
        )

        preprocess_group.add_argument(
            "--imputation",
            type=str,
            default="iterative",
            choices=["mean", "median", "iterative", "none"],
            help="Missing value imputation method (default: iterative)"
        )

        preprocess_group.add_argument(
            "--feature_selection",
            action="store_true",
            help="Enable feature selection"
        )

        preprocess_group.add_argument(
            "--augmentation",
            action="store_true",
            help="Enable data augmentation"
        )

        # BLAST preprocessing options
        preprocess_group.add_argument(
            "--use_blast",
            action="store_true",
            help="Enable BLAST preprocessing for batch effect removal"
        )

        preprocess_group.add_argument(
            "--blast_variance_retention",
            type=float,
            default=0.95,
            help="BLAST variance retention ratio (default: 0.95)"
        )

        # Outlier removal options
        preprocess_group.add_argument(
            "--remove_outliers",
            action="store_true",
            help="Enable outlier removal"
        )

        preprocess_group.add_argument(
            "--outlier_method",
            type=str,
            default="iqr",
            choices=["iqr", "zscore", "modified_zscore", "isolation_forest"],
            help="Outlier detection method (default: iqr)"
        )

        preprocess_group.add_argument(
            "--outlier_threshold",
            type=float,
            help="Threshold for outlier detection (method-specific)"
        )

        # Leakage detection options
        preprocess_group.add_argument(
            "--check_leakage",
            action="store_true",
            help="Enable data leakage detection"
        )

        preprocess_group.add_argument(
            "--leakage_correlation_threshold",
            type=float,
            default=0.95,
            help="Correlation threshold for leakage detection (default: 0.95)"
        )

        # === Analysis Arguments ===
        analysis_group = self.parser.add_argument_group(
            "Analysis Configuration")
        analysis_group.add_argument(
            "--analysis_type",
            type=str,
            nargs="+",
            default=["basic", "correlation", "health"],
            choices=["basic", "correlation", "health", "temporal", "anomaly"],
            help="Types of analysis to perform (default: basic correlation health)"
        )

        analysis_group.add_argument(
            "--generate_plots",
            action="store_true",
            default=True,
            help="Generate analysis plots (default: True)"
        )

        analysis_group.add_argument(
            "--save_results",
            action="store_true",
            default=True,
            help="Save analysis results (default: True)"
        )

        # === Hardware/Performance Arguments ===
        perf_group = self.parser.add_argument_group(
            "Performance Configuration")
        perf_group.add_argument(
            "--gpu",
            action="store_true",
            help="Use GPU if available"
        )

        perf_group.add_argument(
            "--mixed_precision",
            action="store_true",
            help="Enable mixed precision training"
        )

        perf_group.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of worker processes (default: 4)"
        )

        perf_group.add_argument(
            "--memory_limit",
            type=float,
            help="GPU memory limit in GB"
        )

        # === Experiment Tracking Arguments ===
        exp_group = self.parser.add_argument_group("Experiment Tracking")
        exp_group.add_argument(
            "--experiment_name",
            type=str,
            help="Name for experiment tracking"
        )

        exp_group.add_argument(
            "--tags",
            type=str,
            nargs="+",
            help="Tags for experiment"
        )

        exp_group.add_argument(
            "--log_level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging level (default: INFO)"
        )

    def parse_args(self, args=None) -> argparse.Namespace:
        """Parse command line arguments"""
        parsed_args = self.parser.parse_args(args)

        # Validate arguments
        self._validate_arguments(parsed_args)

        return parsed_args

    def _validate_arguments(self, args: argparse.Namespace):
        """Validate parsed arguments"""

        # Check if config file exists (optional - use defaults if not found)
        if args.config and not os.path.exists(args.config):
            print(
                f"Warning: Config file {args.config} not found. Using command-line args and defaults.")

        # Check if dataset path exists for modes that need it (optional for some modes)
        if args.mode in ["preprocess", "analyze"] and hasattr(args, 'dataset_path'):
            if not os.path.exists(args.dataset_path):
                print(
                    f"Warning: Dataset path {args.dataset_path} not found. Will be created if needed.")

        # Check if model path exists for evaluation/prediction (only if provided)
        if args.mode in ["evaluate", "predict"] and hasattr(args, 'model_path') and args.model_path:
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(
                    f"Model path {args.model_path} not found")

        # Validate numeric ranges (with safe defaults)
        if hasattr(args, 'test_size'):
            if args.test_size <= 0 or args.test_size >= 1:
                raise ValueError("test_size must be between 0 and 1")

        if hasattr(args, 'val_size'):
            if args.val_size <= 0 or args.val_size >= 1:
                raise ValueError("val_size must be between 0 and 1")

        if hasattr(args, 'dropout'):
            if args.dropout < 0 or args.dropout >= 1:
                raise ValueError("dropout must be between 0 and 1")

        if hasattr(args, 'lr'):
            if args.lr <= 0:
                raise ValueError("learning_rate must be positive")

        # Validate BLAST parameters
        if hasattr(args, 'blast_variance_retention'):
            if args.blast_variance_retention <= 0 or args.blast_variance_retention > 1:
                raise ValueError(
                    "blast_variance_retention must be between 0 and 1")

    def merge_with_config(self, args: argparse.Namespace, config_override: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Merge command line arguments with configuration file.
        Command line arguments take precedence over config file.
        """

        # Load config file if it exists
        config = {}
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f) or {}

        # Apply config override if provided
        if config_override:
            config.update(config_override)

        # Convert args to dict
        args_dict = vars(args)

        # Merge configurations - command line args override config file
        merged_config = {**config, **{k: v for k,
                                      v in args_dict.items() if v is not None}}

        return merged_config

    def get_help(self) -> str:
        """Get help text"""
        return self.parser.format_help()


def create_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Create a configuration dictionary from parsed arguments"""

    config = {
        'mode': args.mode,
        'data': {
            'dataset_path': args.dataset_path,
            'test_size': args.test_size,
            'val_size': args.val_size,
            'random_state': args.random_state,
            'timesteps': args.timesteps,
            'scaler': args.scaler,
            'imputation': args.imputation,
            'feature_selection': args.feature_selection,
            'augmentation': args.augmentation
        },
        'model': {
            'type': args.model_type,
            'hidden_units': args.hidden_units,
            'dropout': args.dropout,
            'l2_reg': args.l2_reg,
            'input_shape': [args.timesteps, 60]  # Default feature count
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'optimizer': args.optimizer,
            'early_stopping_patience': args.early_stopping_patience,
            'reduce_lr_patience': args.reduce_lr_patience,
            'class_weight': args.class_weight
        },
        'cross_validation': {
            'folds': args.cv_folds,
            'strategy': args.cv_strategy,
            'shuffle': args.cv_shuffle
        },
        'hyperparameter_tuning': {
            'tuner': args.tuner,
            'max_trials': args.max_trials,
            'epochs_per_trial': args.tuner_epochs
        },
        'analysis': {
            'types': args.analysis_type,
            'generate_plots': args.generate_plots,
            'save_results': args.save_results
        },
        'performance': {
            'use_gpu': args.gpu,
            'mixed_precision': args.mixed_precision,
            'num_workers': args.num_workers,
            'memory_limit': args.memory_limit
        },
        'output': {
            'output_dir': args.output_dir,
            'experiment_name': args.experiment_name,
            'tags': args.tags or [],
            'log_level': args.log_level
        }
    }

    return config


def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Example usage
    parser = PEECOMArgumentParser()
    args = parser.parse_args()

    # Create config from args
    config = create_config_from_args(args)

    # Print configuration
    print("Parsed Configuration:")
    print(yaml.dump(config, default_flow_style=False, indent=2))
