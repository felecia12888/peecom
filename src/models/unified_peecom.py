"""
Unified PEECOM Model Interface
=============================

This module provides a unified interface to the PEECOM model system that
integrates seamlessly with the existing training pipeline while providing
access to the new modular architecture.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import time
import warnings
from typing import Dict, Any, Optional, Union

try:
    # Try to import the modular PEECOM system
    from src.models.peecom.peecom_factory import PEECOMFactory, create_high_performance_peecom
    MODULAR_PEECOM_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        from models.peecom.peecom_factory import PEECOMFactory, create_high_performance_peecom
        MODULAR_PEECOM_AVAILABLE = True
    except ImportError:
        MODULAR_PEECOM_AVAILABLE = False
        warnings.warn(
            "Modular PEECOM system not available, using fallback implementation")


class PEECOMModel(BaseEstimator, ClassifierMixin):
    """
    Unified PEECOM Model Interface

    This class provides backward compatibility with existing code while
    integrating the new high-performance modular PEECOM system.

    When the modular system is available, it automatically uses the
    high-performance variant for optimal results.
    """

    def __init__(self,
                 ensemble_method='voting',
                 calibration_method=None,
                 random_state=42,
                 verbose=True,
                 use_modular=True,
                 **kwargs):
        """
        Initialize unified PEECOM model.

        Args:
            ensemble_method: Ensemble strategy ('voting', 'weighted')
            calibration_method: Calibration method (None, 'isotonic', 'sigmoid')
            random_state: Random seed for reproducibility
            verbose: Whether to print progress messages
            use_modular: Whether to use the new modular system (if available)
            **kwargs: Additional model parameters
        """
        self.ensemble_method = ensemble_method
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.verbose = verbose
        self.use_modular = use_modular and MODULAR_PEECOM_AVAILABLE

        # Store parameters for model creation
        self.model_params = kwargs

        # Initialize components
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names_ = None
        self.is_fitted_ = False

        # Initialize the appropriate implementation
        if self.use_modular:
            self._init_modular_peecom()
        else:
            self._init_fallback_peecom()

    def _init_modular_peecom(self):
        """Initialize the modular PEECOM system."""
        try:
            self.peecom_model = create_high_performance_peecom(
                random_state=self.random_state,
                verbose=self.verbose,
                **self.model_params
            )
            if self.verbose:
                print("Using High-Performance Modular PEECOM system")
        except Exception as e:
            if self.verbose:
                print(f"Failed to initialize modular PEECOM: {e}")
                print("Falling back to standard implementation")
            self.use_modular = False
            self._init_fallback_peecom()

    def _init_fallback_peecom(self):
        """Initialize fallback PEECOM implementation."""
        if self.verbose:
            print("Using Fallback PEECOM implementation")

        # Initialize high-performance ensemble models
        self.models = {
            'random_forest_primary': RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'random_forest_aggressive': RandomForestClassifier(
                n_estimators=600,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                max_features='sqrt',
                random_state=self.random_state
            )
        }
        self.ensemble_weights = {}

    def _engineer_fallback_features(self, X):
        """Engineer features using fallback implementation."""
        X_engineered = X.copy()

        # Get sensor mappings
        sensors = self._get_sensor_mappings(X)

        # Core high-performance features
        flow_primary = sensors['FS1'] + 1e-6
        flow_secondary = sensors['FS2'] + 1e-6

        # 1. Hydraulic Power Analysis
        X_engineered['hydraulic_power_primary'] = sensors['PS1'] * \
            flow_primary / 1000.0
        X_engineered['power_efficiency'] = X_engineered['hydraulic_power_primary'] / \
            (sensors['EPS1'] + 1e-6)

        # 2. Critical Pressure Features
        X_engineered['pressure_ratio_ps1_ps2'] = (
            sensors['PS1'] + 1) / (sensors['PS2'] + 1)
        X_engineered['pressure_diff_ps1_ps2'] = sensors['PS1'] - sensors['PS2']

        # 3. Flow Balance
        X_engineered['flow_balance'] = (flow_primary + flow_secondary) / 2
        X_engineered['flow_conservation_error'] = np.abs(
            flow_primary - flow_secondary) / (flow_primary + flow_secondary + 1e-6)

        # 4. Thermal Features
        temp_sensors = [sensors['TS1'], sensors['TS2'],
                        sensors['TS3'], sensors['TS4']]
        X_engineered['thermal_efficiency'] = (
            sensors['TS1'] - sensors['TS3']) / (sensors['TS1'] + 273.15 + 1e-6)
        # Fix for pandas Series - use element-wise operations
        temp_df = pd.DataFrame(temp_sensors).T
        X_engineered['temp_range'] = temp_df.max(axis=1) - temp_df.min(axis=1)

        # 5. System Health
        pressure_health = 1 / \
            (1 + np.abs(X_engineered['pressure_diff_ps1_ps2']
                        ) / (sensors['PS1'] + 1))
        flow_health = 1 / (1 + X_engineered['flow_conservation_error'])
        thermal_health = 1 / (1 + X_engineered['temp_range'] / 100)

        X_engineered['system_health_score'] = (
            pressure_health + flow_health + thermal_health) / 3

        # Clean up
        X_engineered = X_engineered.replace([np.inf, -np.inf], np.nan)
        X_engineered = X_engineered.fillna(X_engineered.median())

        return X_engineered

    def _get_sensor_mappings(self, X):
        """Get sensor mappings with fallbacks."""
        sensors = {}
        sensor_names = ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2',
                        'TS1', 'TS2', 'TS3', 'TS4', 'VS1', 'CE', 'CP', 'SE']

        for i, sensor in enumerate(sensor_names):
            if hasattr(X, 'columns') and sensor in X.columns:
                sensors[sensor] = X[sensor]
            elif hasattr(X, 'iloc') and i < len(X.columns):
                sensors[sensor] = X.iloc[:, i]
            else:
                # Safe fallbacks with proper index
                n_rows = len(X) if hasattr(X, '__len__') else 100
                index = X.index if hasattr(X, 'index') else range(n_rows)

                if 'PS' in sensor:
                    sensors[sensor] = pd.Series([100.0] * n_rows, index=index)
                elif 'TS' in sensor:
                    sensors[sensor] = pd.Series([40.0] * n_rows, index=index)
                elif 'FS' in sensor:
                    sensors[sensor] = pd.Series([10.0] * n_rows, index=index)
                else:
                    sensors[sensor] = pd.Series([1.0] * n_rows, index=index)

        return sensors

    def fit(self, X, y):
        """
        Fit the PEECOM model.

        Args:
            X: Input features
            y: Target labels

        Returns:
            Self (fitted model)
        """
        start_time = time.time()

        if self.verbose:
            print("Training Unified PEECOM Model...")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(
                X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        if self.use_modular:
            # Use modular PEECOM system
            try:
                self.peecom_model.fit(X, y)
                self.feature_names_ = self.peecom_model.get_feature_names()
                self.is_fitted_ = True

                if self.verbose:
                    training_time = time.time() - start_time
                    print(
                        f"Modular PEECOM training completed in {training_time:.2f} seconds")

                return self

            except Exception as e:
                if self.verbose:
                    print(f"Modular PEECOM training failed: {e}")
                    print("Falling back to standard implementation")
                self.use_modular = False

        # Fallback implementation
        if self.verbose:
            print("Using fallback PEECOM implementation...")

        # Engineer features
        X_engineered = self._engineer_fallback_features(X)
        self.feature_names_ = X_engineered.columns.tolist()

        # Scale features
        X_scaled = self.scaler.fit_transform(X_engineered)

        # Train ensemble models with cross-validation for weighting
        cv_scores = {}

        for name, model in self.models.items():
            if self.verbose:
                print(f"Training {name}...")

            # Handle class imbalance
            try:
                classes = np.unique(y)
                if len(classes) > 1:
                    class_weights = compute_class_weight(
                        'balanced', classes=classes, y=y)
                    weight_dict = dict(zip(classes, class_weights))

                    if hasattr(model, 'set_params') and 'class_weight' in model.get_params():
                        model.set_params(class_weight=weight_dict)
            except Exception:
                pass

            # Fit model
            model.fit(X_scaled, y)

            # Compute CV score for ensemble weighting
            try:
                scores = cross_val_score(
                    model, X_scaled, y, cv=3, scoring='accuracy', n_jobs=-1)
                cv_scores[name] = scores.mean()
            except:
                cv_scores[name] = 0.5

        # Calculate ensemble weights
        scores_array = np.array(list(cv_scores.values()))
        exp_scores = np.exp(scores_array * 5.0)  # Temperature scaling
        weights = exp_scores / exp_scores.sum()
        self.ensemble_weights = dict(zip(cv_scores.keys(), weights))

        self.is_fitted_ = True

        if self.verbose:
            training_time = time.time() - start_time
            print(
                f"Fallback PEECOM training completed in {training_time:.2f} seconds")

        return self

    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(
                X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        if self.use_modular:
            return self.peecom_model.predict(X)

        # Fallback implementation
        X_engineered = self._engineer_fallback_features(X)
        X_scaled = self.scaler.transform(X_engineered)

        # Ensemble prediction
        predictions = []
        weights = []

        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions.append(pred)
            weights.append(self.ensemble_weights.get(
                name, 1.0 / len(self.models)))

        # Weighted majority voting
        predictions_array = np.array(predictions)
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()

        weighted_preds = np.average(
            predictions_array, axis=0, weights=weights_array)
        return np.round(weighted_preds).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(
                X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        if self.use_modular:
            return self.peecom_model.predict_proba(X)

        # Fallback implementation
        X_engineered = self._engineer_fallback_features(X)
        X_scaled = self.scaler.transform(X_engineered)

        # Ensemble probability prediction
        probabilities = []
        weights = []

        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)
                probabilities.append(proba)
                weights.append(self.ensemble_weights.get(
                    name, 1.0 / len(self.models)))

        if probabilities:
            weights_array = np.array(weights)
            weights_array = weights_array / weights_array.sum()
            return np.average(probabilities, axis=0, weights=weights_array)
        else:
            # Fallback to hard predictions
            predictions = self.predict(X)
            n_classes = len(np.unique(predictions))
            n_samples = len(predictions)

            probas = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(predictions):
                probas[i, pred] = 1.0

            return probas

    def get_feature_names(self):
        """Get feature names."""
        return self.feature_names_

    def get_feature_importance(self):
        """Get feature importance scores."""
        if self.use_modular and hasattr(self.peecom_model, 'get_feature_importance'):
            return self.peecom_model.get_feature_importance()

        # Fallback implementation
        if not self.is_fitted_ or not self.feature_names_:
            return {}

        importance_scores = {}
        total_weight = 0

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                weight = self.ensemble_weights.get(name, 1.0)
                importances = model.feature_importances_

                for i, importance in enumerate(importances):
                    if i < len(self.feature_names_):
                        feature_name = self.feature_names_[i]
                        if feature_name not in importance_scores:
                            importance_scores[feature_name] = 0
                        importance_scores[feature_name] += importance * weight

                total_weight += weight

        # Normalize and sort
        if total_weight > 0:
            for feature in importance_scores:
                importance_scores[feature] /= total_weight

        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))

    def get_model_info(self):
        """Get model information."""
        info = {
            'model_type': 'Unified PEECOM',
            'using_modular': self.use_modular,
            'is_fitted': self.is_fitted_,
            'ensemble_method': self.ensemble_method,
            'calibration_method': self.calibration_method
        }

        if self.use_modular and hasattr(self.peecom_model, 'get_model_info'):
            info.update(self.peecom_model.get_model_info())

        return info
