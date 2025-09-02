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
import signal
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

        # Handle different data types
        if target.dtype == 'float64':
            # Check if float values are actually integers
            if np.all(np.equal(np.mod(target, 1), 0)):
                # Safe to convert to int
                target = target.astype(int)
                print(f"Converted float target to int")
            else:
                # Round to nearest integer for classification
                target = np.round(target).astype(int)
                print(f"Rounded float target to int for classification")

        # Ensure labels start from 0 for sklearn
        unique_vals = np.unique(target)
        if len(unique_vals) > 1 and np.min(unique_vals) != 0:
            # Create label mapping
            label_map = {val: idx for idx,
                         val in enumerate(sorted(unique_vals))}
            target = np.array([label_map[val] for val in target])
            print(
                f"Remapped target labels: {dict(zip(sorted(unique_vals), range(len(unique_vals))))}")

        print(f"Target distribution: {np.bincount(target)}")
        return target
    else:
        print(f"Available target columns: {list(y.columns)}")
        # Default to first column
        target = y.iloc[:, 0].values

        # Apply same preprocessing to default target
        if target.dtype == 'float64':
            if np.all(np.equal(np.mod(target, 1), 0)):
                target = target.astype(int)
            else:
                target = np.round(target).astype(int)

        # Ensure labels start from 0
        unique_vals = np.unique(target)
        if len(unique_vals) > 1 and np.min(unique_vals) != 0:
            label_map = {val: idx for idx,
                         val in enumerate(sorted(unique_vals))}
            target = np.array([label_map[val] for val in target])

        print(f"Using {y.columns[0]} as target")
        print(f"Target distribution: {np.bincount(target)}")
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

        # For evaluation, use the model's predict methods since ensemble doesn't have single .score()
        try:
            # Get predictions using the ensemble
            train_pred = model_instance.predict(X_train)
            test_pred = model_instance.predict(X_test)

            # Calculate accuracy scores manually
            from sklearn.metrics import accuracy_score
            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
        except Exception as e:
            print(f"Error in PEECOM evaluation: {e}")
            # Fallback to using the primary model if available
            if hasattr(model_instance, 'model') and model_instance.model is not None:
                try:
                    if hasattr(model_instance, '_engineer_fallback_features'):
                        X_train_eng = model_instance._engineer_fallback_features(
                            X_train)
                        X_test_eng = model_instance._engineer_fallback_features(
                            X_test)
                        X_train_scaled = model_instance.scaler.transform(
                            X_train_eng)
                        X_test_scaled = model_instance.scaler.transform(
                            X_test_eng)
                    else:
                        # Use scaled original features
                        temp_scaler = StandardScaler()
                        X_train_scaled = temp_scaler.fit_transform(X_train)
                        X_test_scaled = temp_scaler.transform(X_test)

                    train_score = model_instance.model.score(
                        X_train_scaled, y_train)
                    test_score = model_instance.model.score(
                        X_test_scaled, y_test)
                except Exception as fallback_error:
                    print(f"Fallback evaluation failed: {fallback_error}")
                    train_score, test_score = 0.0, 0.0
            else:
                # Ultimate fallback
                train_score, test_score = 0.0, 0.0

        # Get predictions for detailed evaluation
        y_pred = model_instance.predict(X_test)

        # Cross-validation with PEECOM
        # For CV, we need to engineer features consistently
        try:
            if hasattr(model_instance, '_engineer_fallback_features'):
                X_engineered = model_instance._engineer_fallback_features(X)
                X_scaled_full = model_instance.scaler.transform(X_engineered)
            else:
                # Fallback: just scale the original features
                temp_scaler = StandardScaler()
                X_scaled_full = temp_scaler.fit_transform(X)
        except Exception as e:
            print(f"Warning: Feature engineering failed for CV: {e}")
            # Ultimate fallback: just scale the original features
            temp_scaler = StandardScaler()
            X_scaled_full = temp_scaler.fit_transform(X)
        try:
            if hasattr(model_instance, '_engineer_fallback_features'):
                X_engineered = model_instance._engineer_fallback_features(X)
                X_scaled_full = model_instance.scaler.transform(X_engineered)
            else:
                # Fallback: just scale the original features
                temp_scaler = StandardScaler()
                X_scaled_full = temp_scaler.fit_transform(X)
        except Exception as e:
            print(f"Warning: Feature engineering failed for CV: {e}")
            # Ultimate fallback: just scale the original features
            temp_scaler = StandardScaler()
            X_scaled_full = temp_scaler.fit_transform(X)

        cv_scores = cross_val_score(
            model_instance, X, y, cv=5)

        model = model_instance  # For saving (the entire PEECOM model)
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

    # Check dataset size and warn about complex models
    n_samples, n_features = X.shape
    print(f"Dataset size: {n_samples} samples, {n_features} features")

    if model_name == 'peecom' and n_samples > 5000:
        print(f"⚠️  Warning: Large dataset ({n_samples} samples) detected.")
        print(
            f"⚠️  PEECOM model may take a long time. Consider using random_forest instead.")

        # Ask for confirmation or provide timeout
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(
                "Training timeout - dataset too large for PEECOM model")

        # Set a 5-minute timeout for PEECOM on large datasets
        if n_samples > 5000:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5 minutes

    results = {}

    # Evaluate each target
    for target_col in y.columns:
        print(f"\n--- Evaluating {target_col} ---")
        try:
            target = prepare_targets(y, target_col)

            # Skip if target has only one class
            if len(np.unique(target)) < 2:
                print(f"⚠️  Skipping {target_col}: Only one class found")
                results[target_col] = {
                    'error': 'Single class target - no classification possible'}
                continue

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

        except TimeoutError as e:
            print(f"✗ {target_col}: Timeout - {str(e)}")
            print(f"💡 Suggestion: Try with --model random_forest for faster training")
            results[target_col] = {'error': f'Timeout: {str(e)}'}
            break  # Don't try remaining targets if we're timing out

        except Exception as e:
            print(f"✗ {target_col}: Failed - {str(e)}")
            results[target_col] = {'error': str(e)}

    # Clear any remaining timeout
    if model_name == 'peecom' and n_samples > 5000:
        signal.alarm(0)

    # Save summary results
    import json
    summary_file = Path(output_dir) / model_name / "all_targets_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSummary saved to: {summary_file}")
    return results
