#!/usr/bin/env python3
"""
Enhanced Dataset Processing with Data Leakage Prevention

This script provides enhanced dataset processing with proper train/test splits,
data leakage detection and prevention, and improved feature engineering.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path

# Add src to path
sys.path.append('src')


def create_enhanced_preprocessing_parser():
    """Create argument parser for enhanced preprocessing"""
    parser = argparse.ArgumentParser(
        description='Enhanced PEECOM dataset preprocessing with leakage prevention')

    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name to process')
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                        help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for processed data')
    parser.add_argument('--enforce-split', action='store_true',
                        help='Create train/validation/test splits')
    parser.add_argument('--train-split', type=float, default=0.7,
                        help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.15,
                        help='Test split ratio')
    parser.add_argument('--remove-leakage', action='store_true', default=True,
                        help='Remove potential data leakage features')
    parser.add_argument('--stratify', action='store_true', default=True,
                        help='Use stratified sampling for splits')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducible splits')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    return parser


class DataLeakageDetector:
    """Enhanced data leakage detection and prevention"""

    LEAKAGE_KEYWORDS = [
        'file_id', 'filename', 'index', 'id', 'timestamp', 'date', 'time',
        'row', 'sample', 'cycle_id', 'record_id', 'seq_id', 'order', 'position'
    ]

    @classmethod
    def detect_leakage_features(cls, feature_names, target_names=None):
        """Detect potential data leakage features"""
        leakage_features = []

        for feature in feature_names:
            feature_lower = feature.lower()

            # Check for leakage keywords
            if any(keyword in feature_lower for keyword in cls.LEAKAGE_KEYWORDS):
                leakage_features.append(feature)
                continue

            # Check for target name similarity
            if target_names:
                for target in target_names:
                    target_lower = target.lower()
                    if (target_lower in feature_lower or
                        feature_lower in target_lower or
                            abs(len(target_lower) - len(feature_lower)) < 3):
                        if feature not in leakage_features:
                            leakage_features.append(feature)
                        break

        return leakage_features

    @classmethod
    def remove_leakage_features(cls, X, y, target_names=None, logger=None):
        """Remove data leakage features"""
        if target_names is None:
            target_names = list(y.columns) if hasattr(y, 'columns') else []

        leakage_features = cls.detect_leakage_features(X.columns, target_names)

        if leakage_features:
            if logger:
                logger.warning(
                    f"Removing {len(leakage_features)} potential data leakage features: {leakage_features}")
            X_clean = X.drop(columns=leakage_features)
            return X_clean, leakage_features

        return X, []


class EnhancedDataSplitter:
    """Enhanced data splitting with stratification and validation"""

    def __init__(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 random_state=42, stratify=True):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.stratify = stratify

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def create_splits(self, X, y, logger=None):
        """Create train/validation/test splits"""
        if logger:
            logger.info(f"Creating data splits: train={self.train_ratio}, "
                        f"val={self.val_ratio}, test={self.test_ratio}")

        # Prepare stratification targets
        stratify_targets = None
        if self.stratify and len(y.shape) > 1:
            # For multi-target, use first target for stratification
            stratify_targets = y.iloc[:, 0] if hasattr(y, 'iloc') else y[:, 0]
        elif self.stratify:
            stratify_targets = y

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_ratio,
            random_state=self.random_state,
            stratify=stratify_targets
        )

        # Second split: separate train and validation
        val_size_adjusted = self.val_ratio / \
            (self.train_ratio + self.val_ratio)

        # Update stratification for remaining data
        if self.stratify:
            if len(y_temp.shape) > 1:
                stratify_temp = y_temp.iloc[:, 0] if hasattr(
                    y_temp, 'iloc') else y_temp[:, 0]
            else:
                stratify_temp = y_temp
        else:
            stratify_temp = None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state + 1,
            stratify=stratify_temp
        )

        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }

        if logger:
            for split_name, (X_split, y_split) in splits.items():
                logger.info(
                    f"{split_name.capitalize()} set: {X_split.shape[0]} samples")

        return splits

    def save_splits(self, splits, output_dir, dataset_name, logger=None):
        """Save splits to files"""
        split_metadata = {
            'dataset_name': dataset_name,
            'split_timestamp': datetime.now().isoformat(),
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'random_state': self.random_state,
            'stratified': self.stratify,
            'splits': {}
        }

        for split_name, (X_split, y_split) in splits.items():
            # Save features
            X_file = os.path.join(output_dir, f'X_{split_name}.csv')
            X_split.to_csv(X_file, index=False)

            # Save targets
            y_file = os.path.join(output_dir, f'y_{split_name}.csv')
            y_split.to_csv(y_file, index=False)

            # Update metadata
            split_metadata['splits'][split_name] = {
                'samples': X_split.shape[0],
                'features': X_split.shape[1],
                'targets': y_split.shape[1] if len(y_split.shape) > 1 else 1,
                'X_file': X_file,
                'y_file': y_file
            }

            if logger:
                logger.info(f"Saved {split_name} split: {X_file}, {y_file}")

        # Save metadata
        metadata_file = os.path.join(output_dir, 'split_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(split_metadata, f, indent=2)

        return split_metadata


def enhanced_preprocessing_main():
    """Enhanced preprocessing main function"""
    parser = create_enhanced_preprocessing_parser()
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("enhanced_preprocessing")

    # Setup output directories
    base_output_dir = args.output_dir
    processed_data_dir = os.path.join(
        base_output_dir, 'processed_data', args.dataset)
    os.makedirs(processed_data_dir, exist_ok=True)

    logger.info(f"Enhanced preprocessing for dataset: {args.dataset}")
    logger.info(f"Output directory: {processed_data_dir}")

    # Load dataset using existing loader
    try:
        from src.loader.dataset_loader import dataset_loader

        # Get handler
        config = {}  # Load config if needed
        handler = dataset_loader.get_handler(args.dataset, config)
        logger.info(
            f"Using {type(handler).__name__} for dataset {args.dataset}")

        # Load data
        features, targets = handler.load_data()
        features, targets = handler.preprocess_data(features, targets)

        logger.info(
            f"Loaded dataset: Features {features.shape}, Targets {targets.shape}")

    except Exception as e:
        logger.error(f"Failed to load dataset {args.dataset}: {e}")
        return

    # Data leakage detection and removal
    if args.remove_leakage:
        target_names = list(targets.columns) if hasattr(
            targets, 'columns') else []
        features_clean, removed_features = DataLeakageDetector.remove_leakage_features(
            features, targets, target_names, logger
        )

        if removed_features:
            logger.info(
                f"Dataset shape after leakage removal: {features_clean.shape}")
            features = features_clean

    # Create data splits if requested
    if args.enforce_split:
        splitter = EnhancedDataSplitter(
            train_ratio=args.train_split,
            val_ratio=args.val_split,
            test_ratio=args.test_split,
            random_state=args.random_state,
            stratify=args.stratify
        )

        try:
            splits = splitter.create_splits(features, targets, logger)
            split_metadata = splitter.save_splits(
                splits, processed_data_dir, args.dataset, logger)

            logger.info("Data splits created successfully")

        except Exception as e:
            logger.error(f"Failed to create splits: {e}")
            # Fallback to single dataset
            logger.info("Saving as single dataset without splits")
            features.to_csv(os.path.join(
                processed_data_dir, 'X_full.csv'), index=False)
            targets.to_csv(os.path.join(
                processed_data_dir, 'y_full.csv'), index=False)
    else:
        # Save as single dataset
        features.to_csv(os.path.join(
            processed_data_dir, 'X_full.csv'), index=False)
        targets.to_csv(os.path.join(
            processed_data_dir, 'y_full.csv'), index=False)
        logger.info("Saved as single dataset (no splits)")

    # Create comprehensive metadata
    metadata = {
        'preprocessing_timestamp': datetime.now().isoformat(),
        'dataset_name': args.dataset,
        'processing_type': 'enhanced_preprocessing',
        'original_samples': features.shape[0],
        'final_samples': features.shape[0],
        'features': features.shape[1],
        'targets': targets.shape[1] if len(targets.shape) > 1 else 1,
        'feature_columns': list(features.columns) if hasattr(features, 'columns') else [],
        'target_columns': list(targets.columns) if hasattr(targets, 'columns') else [],
        'leakage_prevention': {
            'enabled': args.remove_leakage,
            'removed_features': removed_features if args.remove_leakage else []
        },
        'data_splits': {
            'enabled': args.enforce_split,
            'train_ratio': args.train_split if args.enforce_split else None,
            'val_ratio': args.val_split if args.enforce_split else None,
            'test_ratio': args.test_split if args.enforce_split else None,
            'stratified': args.stratify if args.enforce_split else None
        },
        'command_args': vars(args)
    }

    # Save metadata
    with open(os.path.join(processed_data_dir, 'enhanced_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Enhanced preprocessing completed successfully!")
    logger.info(f"All outputs saved to: {processed_data_dir}")


if __name__ == "__main__":
    enhanced_preprocessing_main()
