#!/usr/bin/env python3
"""
Training Utilities for PEECOM

This module contains utility functions for training machine learning models
including data loading, target preparation, and model training orchestration.
"""

from src.models.model_loader import model_loader
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import sys
import os

# Add src to path for imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)


def load_processed_data(data_dir):
    """Load processed features and targets from CSV files"""
    print(f"Loading data from: {data_dir}")

    # Load features and targets
    X = pd.read_csv(os.path.join(data_dir, 'X_full.csv'))
    y = pd.read_csv(os.path.join(data_dir, 'y_full.csv'))

    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Available targets: {list(y.columns)}")

    return X, y


def prepare_targets(y, target_column='stable_flag'):
    """Prepare target variable for classification"""
    if target_column in y.columns:
        target = y[target_column].values
        print(f"Using {target_column} as target")
        print(f"Target distribution: {np.bincount(target)}")
        return target
    else:
        print(f"Available target columns: {list(y.columns)}")
        # Default to first column
        target = y.iloc[:, 0].values
        print(f"Using {y.columns[0]} as target")
        return target


def train_model_with_loader(X, y, model_name='random_forest', output_dir='output/models', target_name='unknown'):
    """Train a model using the model loader system"""
    print(
        f"\nTraining {model_loader.get_model_display_name(model_name)} model...")

    # Create output directory structure: output/models/model_name/target_name/
    model_output_dir = Path(output_dir) / model_name / target_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load model from model loader
    model_instance = model_loader.load_model(model_name)

    # Handle PEECOM model differently (it has its own preprocessing)
    if model_name == 'peecom':
        print("Training PEECOM model with physics-enhanced features...")
        # PEECOM handles its own feature engineering and scaling
        model_instance.fit(X_train, y_train)

        # For evaluation, we need to use model_instance methods
        train_score = model_instance.model.score(
            model_instance.scaler.transform(
                model_instance._engineer_physics_features(X_train)
            ), y_train
        )
        test_score = model_instance.model.score(
            model_instance.scaler.transform(
                model_instance._engineer_physics_features(X_test)
            ), y_test
        )

        # Get predictions for detailed evaluation
        y_pred = model_instance.predict(X_test)

        # Cross-validation with PEECOM
        X_engineered = model_instance._engineer_physics_features(X)
        X_scaled_full = model_instance.scaler.transform(X_engineered)
        cv_scores = cross_val_score(
            model_instance.model, X_scaled_full, y, cv=5)

        model = model_instance.model  # For saving
        scaler = model_instance.scaler  # For saving

    else:
        # Standard model training
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = model_instance.get_model()

        # Train the model
        print("Training model...")
        model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"CV scores: {cv_scores}")
    print(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Prepare results
    results = {
        'model_name': model_name,
        'model_display_name': model_loader.get_model_display_name(model_name),
        'target_name': target_name,
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importance': model_instance.get_feature_importance(),
        'model_params': model_instance.get_params(),
        'feature_names': model_instance.feature_names_ if hasattr(model_instance, 'feature_names_') and model_instance.feature_names_ else list(X.columns),
        'model_file': str(model_output_dir / f'{model_name}_model.joblib'),
        'scaler_file': str(model_output_dir / f'{model_name}_scaler.joblib'),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'cv_scores': cv_scores.tolist()
    }

    # Add PEECOM-specific insights if applicable
    if model_name == 'peecom' and hasattr(model_instance, 'get_physics_insights'):
        results['physics_insights'] = model_instance.get_physics_insights()

    # Print training and test accuracy
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

    # Detailed evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation
    print("\nCross-validation scores:")
    print(f"CV scores: {cv_scores}")
    print(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Save model and scaler
    joblib.dump(model, results['model_file'])
    joblib.dump(scaler, results['scaler_file'])

    print(f"Model saved to: {results['model_file']}")

    return model, scaler, results, model_output_dir


def evaluate_all_targets(data_dir: str, output_dir: str, model_name: str = 'random_forest'):
    """Evaluate model performance on all available targets"""
    print("\n" + "="*60)
    print(
        f"EVALUATING ALL TARGETS WITH {model_loader.get_model_display_name(model_name).upper()}")
    print("="*60)

    # Load data
    X, y = load_processed_data(data_dir)

    results = {}

    # Evaluate each target
    for target_col in y.columns:
        print(f"\n--- Evaluating {target_col} ---")
        try:
            target = prepare_targets(y, target_col)
            model, scaler, result, model_output_dir = train_model_with_loader(
                X, target, model_name, output_dir, target_col
            )

            # Save results using results handler
            from src.utils.results_handler import save_training_results
            # Use the correct feature names from the results, not the original X.columns
            feature_names_for_saving = result.get(
                'feature_names', list(X.columns))
            save_training_results(
                result, feature_names_for_saving, model_output_dir)

            results[target_col] = result
            print(f"✓ {target_col}: {result['test_accuracy']:.4f} accuracy")
        except Exception as e:
            print(f"✗ {target_col}: Failed - {str(e)}")
            results[target_col] = {'error': str(e)}

    # Save summary results
    import json
    summary_file = Path(output_dir) / model_name / "all_targets_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSummary saved to: {summary_file}")
    return results
