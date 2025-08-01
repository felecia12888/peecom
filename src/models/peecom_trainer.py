#!/usr/bin/env python3
"""
PEECOM Training Script

Robust training pipeline for hydraulic system condition monitoring.
Supports multiple models and target variables with comprehensive evaluation.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any
import logging

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import joblib


class PEECOMTrainer:
    """
    Comprehensive training pipeline for PEECOM hydraulic system monitoring.
    """

    def __init__(self, data_dir: str, output_dir: str = "output", logger: Optional[logging.Logger] = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or logging.getLogger(__name__)

        # Data containers
        self.features = None
        self.targets = None
        self.metadata = None

        # Available models
        self.available_models = {
            'random_forest': self._create_random_forest,
            'logistic': self._create_logistic_regression
        }

        # Target column information
        self.target_columns = [
            'cooler_condition', 'valve_condition', 'pump_leakage',
            'accumulator_pressure', 'stable_flag'
        ]

    def load_data(self) -> bool:
        """Load processed features and targets"""
        try:
            # Load features
            features_path = self.data_dir / "X_full.csv"
            if not features_path.exists():
                self.logger.error(f"Features file not found: {features_path}")
                return False

            self.features = pd.read_csv(features_path)
            self.logger.info(f"Loaded features: {self.features.shape}")

            # Load targets
            targets_path = self.data_dir / "y_full.csv"
            if not targets_path.exists():
                self.logger.error(f"Targets file not found: {targets_path}")
                return False

            self.targets = pd.read_csv(targets_path)
            self.logger.info(f"Loaded targets: {self.targets.shape}")

            # Load metadata if available
            metadata_path = self.data_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.logger.info("Loaded metadata")

            # Validate data consistency
            if len(self.features) != len(self.targets):
                self.logger.error(
                    "Features and targets have different lengths!")
                return False

            self.logger.info(
                f"Data validation successful: {len(self.features)} samples")
            return True

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False

    def _create_random_forest(self, **kwargs) -> RandomForestClassifier:
        """Create Random Forest classifier"""
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=kwargs.get('random_state', 42),
            n_jobs=kwargs.get('n_jobs', -1)
        )

    def _create_logistic_regression(self, **kwargs) -> LogisticRegression:
        """Create Logistic Regression classifier"""
        return LogisticRegression(
            random_state=kwargs.get('random_state', 42),
            max_iter=kwargs.get('max_iter', 1000),
            n_jobs=kwargs.get('n_jobs', -1)
        )

    def train_single_target(self, target: str, model_type: str = 'random_forest',
                            test_size: float = 0.2, save_model: bool = True, **kwargs) -> Optional[Dict]:
        """
        Train a model for a single target variable.

        Args:
            target: Target column name
            model_type: Type of model to train
            test_size: Proportion of data for testing
            save_model: Whether to save the trained model
            **kwargs: Additional model parameters

        Returns:
            Dictionary with training results
        """
        if target not in self.target_columns:
            self.logger.error(
                f"Invalid target: {target}. Available: {self.target_columns}")
            return None

        if model_type not in self.available_models:
            self.logger.error(
                f"Invalid model: {model_type}. Available: {list(self.available_models.keys())}")
            return None

        try:
            self.logger.info(f"Training {model_type} for target: {target}")

            # Prepare data
            X = self.features.copy()
            y = self.targets[target].copy()

            # Handle class labels for classification
            if y.dtype == 'object' or len(y.unique()) < 10:
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                classes = label_encoder.classes_
            else:
                y_encoded = y
                label_encoder = None
                classes = None

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )

            # Feature scaling for logistic regression
            scaler = None
            if model_type == 'logistic':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test

            # Create and train model
            model = self.available_models[model_type](**kwargs)
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring='accuracy'
            )

            # Detailed metrics
            if label_encoder:
                y_test_labels = label_encoder.inverse_transform(y_test)
                y_test_pred_labels = label_encoder.inverse_transform(
                    y_test_pred)
                target_names = [str(cls) for cls in classes]
            else:
                y_test_labels = y_test
                y_test_pred_labels = y_test_pred
                target_names = None

            # Classification report
            class_report = classification_report(
                y_test_labels, y_test_pred_labels,
                target_names=target_names, output_dict=True
            )

            # Confusion matrix
            conf_matrix = confusion_matrix(y_test_labels, y_test_pred_labels)

            # Compile results
            results = {
                'target': target,
                'model_type': model_type,
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'feature_names': list(X.columns),
                'n_samples': len(X),
                'n_features': len(X.columns),
                'class_distribution': y.value_counts().to_dict(),
                'timestamp': datetime.now().isoformat()
            }

            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                results['feature_importance'] = feature_importance.to_dict(
                    'records')

                # Save feature importance
                importance_path = self.models_dir / \
                    f"feature_importance_{target}_{model_type}.csv"
                feature_importance.to_csv(importance_path, index=False)
                self.logger.info(
                    f"Feature importance saved to: {importance_path}")

            # Save model if requested
            if save_model:
                model_data = {
                    'model': model,
                    'scaler': scaler,
                    'label_encoder': label_encoder,
                    'feature_names': list(X.columns),
                    'target': target,
                    'model_type': model_type
                }

                model_path = self.models_dir / \
                    f"model_{target}_{model_type}.joblib"
                joblib.dump(model_data, model_path)
                self.logger.info(f"Model saved to: {model_path}")
                results['model_path'] = str(model_path)

            # Save results
            results_path = self.models_dir / \
                f"results_{target}_{model_type}.json"
            with open(results_path, 'w') as f:
                # Create a copy without numpy arrays for JSON serialization
                json_results = {k: v for k, v in results.items()
                                if k not in ['confusion_matrix']}
                json_results['confusion_matrix'] = conf_matrix.tolist()
                json.dump(json_results, f, indent=2)

            # Log results
            self.logger.info(f"Training accuracy: {train_accuracy:.4f}")
            self.logger.info(f"Test accuracy: {test_accuracy:.4f}")
            self.logger.info(
                f"CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            return results

        except Exception as e:
            self.logger.error(f"Error training model for {target}: {e}")
            return None

    def train_all_targets(self, model_type: str = 'random_forest', **kwargs) -> Dict[str, Dict]:
        """
        Train models for all target variables.

        Args:
            model_type: Type of model to train
            **kwargs: Additional model parameters

        Returns:
            Dictionary with results for each target
        """
        results = {}

        for target in self.target_columns:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Training {model_type} for {target}")
            self.logger.info(f"{'='*50}")

            result = self.train_single_target(
                target=target,
                model_type=model_type,
                save_model=True,
                **kwargs
            )

            if result:
                results[target] = result
                self.logger.info(
                    f"✅ {target}: Accuracy {result['test_accuracy']:.4f}")
            else:
                self.logger.error(f"❌ Failed to train model for {target}")

        # Save combined results
        combined_results_path = self.models_dir / \
            f"all_targets_{model_type}_results.json"
        with open(combined_results_path, 'w') as f:
            # Prepare for JSON serialization
            json_results = {}
            for target, result in results.items():
                json_results[target] = {k: v for k, v in result.items()
                                        if k not in ['confusion_matrix']}
                json_results[target]['confusion_matrix'] = result['confusion_matrix']

            json.dump(json_results, f, indent=2)

        self.logger.info(f"Combined results saved to: {combined_results_path}")

        return results

    def load_trained_model(self, target: str, model_type: str = 'random_forest'):
        """Load a previously trained model"""
        model_path = self.models_dir / f"model_{target}_{model_type}.joblib"

        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return None

        try:
            model_data = joblib.load(model_path)
            self.logger.info(f"Loaded model for {target} from {model_path}")
            return model_data
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None

    def predict(self, X, target: str, model_type: str = 'random_forest'):
        """Make predictions using a trained model"""
        model_data = self.load_trained_model(target, model_type)
        if not model_data:
            return None

        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']

        # Ensure feature order matches training
        X_ordered = X[model_data['feature_names']]

        # Apply scaling if used during training
        if scaler:
            X_scaled = scaler.transform(X_ordered)
        else:
            X_scaled = X_ordered

        # Make predictions
        predictions = model.predict(X_scaled)

        # Convert back to original labels if needed
        if label_encoder:
            predictions = label_encoder.inverse_transform(predictions)

        return predictions
