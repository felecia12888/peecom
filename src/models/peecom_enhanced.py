#!/usr/bin/env python3
"""
Enhanced PEECOM Model with Advanced Physics Features

This module implements an enhanced version of the PEECOM model specifically
designed for hydraulic system condition monitoring with physics-informed
feature engineering and proper data leakage prevention.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PhysicsFeatureEngineer:
    """
    Advanced physics-informed feature engineering for hydraulic systems
    """

    def __init__(self):
        self.feature_groups = {
            'pressure': ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6'],
            'flow': ['FS1', 'FS2'],
            'temperature': ['TS1', 'TS2', 'TS3', 'TS4'],
            'power': ['EPS1'],
            'vibration': ['VS1'],
            'efficiency': ['CE', 'CP', 'SE']
        }

    def create_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced physics-informed features for hydraulic systems
        """
        physics_features = pd.DataFrame(index=df.index)

        # 1. Hydraulic Power Features (P = Flow × Pressure)
        for flow_sensor in self.feature_groups['flow']:
            for pressure_sensor in self.feature_groups['pressure']:
                flow_cols = [
                    col for col in df.columns if col.startswith(f'{flow_sensor}_')]
                pressure_cols = [
                    col for col in df.columns if col.startswith(f'{pressure_sensor}_')]

                for flow_col in flow_cols:
                    for pressure_col in pressure_cols:
                        if flow_col in df.columns and pressure_col in df.columns:
                            feature_name = f'hydraulic_power_{flow_col}_{pressure_col}'
                            physics_features[feature_name] = df[flow_col] * \
                                df[pressure_col]

        # 2. Pressure Differential Features (System Health Indicators)
        pressure_sensors = self.feature_groups['pressure']
        for i, sensor1 in enumerate(pressure_sensors):
            for sensor2 in pressure_sensors[i+1:]:
                for stat in ['mean', 'std', 'min', 'max']:
                    col1 = f'{sensor1}_{stat}'
                    col2 = f'{sensor2}_{stat}'
                    if col1 in df.columns and col2 in df.columns:
                        # Absolute difference
                        physics_features[f'pressure_diff_{col1}_{col2}'] = abs(
                            df[col1] - df[col2])
                        # Ratio (avoid division by zero)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            # Add small epsilon
                            ratio = df[col1] / (df[col2] + 1e-6)
                            ratio = np.where(np.isfinite(ratio), ratio, 0)
                            physics_features[f'pressure_ratio_{col1}_{col2}'] = ratio

        # 3. Flow Conservation Features (Mass Balance)
        flow_features = []
        for sensor in self.feature_groups['flow']:
            for stat in ['mean', 'std']:
                col = f'{sensor}_{stat}'
                if col in df.columns:
                    flow_features.append(df[col])

        if len(flow_features) >= 2:
            # Flow balance (should be close to zero in ideal system)
            physics_features['flow_balance'] = abs(
                flow_features[0] - flow_features[1])
            # Flow conservation error
            physics_features['flow_conservation_error'] = abs(
                sum(flow_features) - np.mean(flow_features) * len(flow_features))

        # 4. Thermal Efficiency Features
        temp_features = []
        for sensor in self.feature_groups['temperature']:
            col = f'{sensor}_mean'
            if col in df.columns:
                temp_features.append(df[col])

        if temp_features and 'EPS1_energy' in df.columns:
            avg_temp = np.mean(temp_features, axis=0)
            # Thermal efficiency (inverse relationship with temperature rise)
            # Kelvin
            physics_features['thermal_efficiency'] = df['EPS1_energy'] / \
                (avg_temp + 273.15)
            # Temperature spread (system thermal balance)
            if len(temp_features) > 1:
                physics_features['temp_spread'] = np.max(
                    temp_features, axis=0) - np.min(temp_features, axis=0)

        # 5. System Efficiency Features
        if 'EPS1_energy' in df.columns:
            efficiency_cols = [col for col in df.columns if col.startswith(
                'CE_') or col.startswith('CP_') or col.startswith('SE_')]
            if efficiency_cols:
                avg_efficiency = df[efficiency_cols].mean(axis=1)
                # Power efficiency ratio
                physics_features['power_efficiency'] = avg_efficiency * \
                    df['EPS1_energy']
                # System efficiency score
                physics_features['system_efficiency'] = avg_efficiency / \
                    (df['EPS1_energy'] + 1e-6)

        # 6. Vibration Analysis Features
        if 'VS1_rms' in df.columns and 'EPS1_mean' in df.columns:
            # Vibration-to-power ratio (indicator of mechanical issues)
            physics_features['vibration_power_ratio'] = df['VS1_rms'] / \
                (df['EPS1_mean'] + 1e-6)

            # Mechanical health score (lower vibration per unit power is better)
            physics_features['mechanical_health'] = 1.0 / \
                (df['VS1_rms'] + 1e-3)

        # 7. Pressure System Health Features
        pressure_means = [df[f'{sensor}_mean'] for sensor in self.feature_groups['pressure']
                          if f'{sensor}_mean' in df.columns]
        if len(pressure_means) >= 3:
            pressure_array = np.array(pressure_means).T
            # Pressure system variance (lower is better)
            physics_features['pressure_system_variance'] = np.var(
                pressure_array, axis=1)
            # Pressure system stability
            physics_features['pressure_stability'] = 1.0 / \
                (np.std(pressure_array, axis=1) + 1e-3)

        # 8. Advanced Hydraulic Features
        # Cavitation risk indicator (based on pressure drops)
        pressure_mins = [df[f'{sensor}_min'] for sensor in self.feature_groups['pressure']
                         if f'{sensor}_min' in df.columns]
        if pressure_mins:
            min_pressure = np.min(pressure_mins, axis=0)
            physics_features['cavitation_risk'] = np.where(
                min_pressure < 50, 1.0, 0.0)  # 50 bar threshold

        # System load indicator
        if 'EPS1_mean' in df.columns and pressure_means:
            avg_pressure = np.mean(pressure_means, axis=0)
            physics_features['system_load'] = df['EPS1_mean'] * avg_pressure

        # Remove any infinite or NaN values
        physics_features = physics_features.replace([np.inf, -np.inf], 0)
        physics_features = physics_features.fillna(0)

        logger.info(
            f"Created {physics_features.shape[1]} physics-informed features")
        return physics_features

    def get_feature_categories(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Categorize features into physics-enhanced and standard features
        """
        physics_keywords = [
            'hydraulic_power', 'pressure_diff', 'pressure_ratio', 'flow_balance',
            'flow_conservation', 'thermal_efficiency', 'power_efficiency',
            'system_efficiency', 'vibration_power', 'mechanical_health',
            'pressure_system', 'pressure_stability', 'cavitation_risk',
            'system_load', 'temp_spread'
        ]

        physics_features = []
        standard_features = []

        for feature in feature_names:
            is_physics = any(
                keyword in feature for keyword in physics_keywords)
            if is_physics:
                physics_features.append(feature)
            else:
                standard_features.append(feature)

        return {
            'physics': physics_features,
            'standard': standard_features
        }


class EnhancedPEECOMClassifier(BaseEstimator, ClassifierMixin):
    """
    Enhanced PEECOM (Physics-Enhanced Equipment Condition Monitoring) Classifier

    Combines physics-informed feature engineering with ensemble learning
    for superior hydraulic system condition monitoring.
    """

    def __init__(self,
                 use_physics_features: bool = True,
                 ensemble_method: str = 'voting',
                 random_state: int = 42,
                 cv_folds: int = 5,
                 verbose: bool = True):
        self.use_physics_features = use_physics_features
        self.ensemble_method = ensemble_method
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.verbose = verbose

        # Initialize components
        self.physics_engineer = PhysicsFeatureEngineer()
        self.scaler = StandardScaler()
        self.ensemble = None
        self.feature_names_ = None
        self.classes_ = None
        self.is_fitted_ = False

    def _create_ensemble(self, n_features: int) -> VotingClassifier:
        """Create the ensemble classifier"""

        # Base classifiers optimized for different aspects
        classifiers = [
            ('rf_physics', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced'
            )),
            ('rf_standard', RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=3,
                random_state=self.random_state + 1,
                class_weight='balanced'
            )),
            ('logistic', LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced',
                C=1.0
            ))
        ]

        return VotingClassifier(
            estimators=classifiers,
            voting='soft',
            n_jobs=-1
        )

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering pipeline"""
        if not self.use_physics_features:
            return X

        # Create physics features
        physics_features = self.physics_engineer.create_physics_features(X)

        # Combine with original features
        enhanced_features = pd.concat([X, physics_features], axis=1)

        if self.verbose:
            logger.info(f"Enhanced features: {X.shape[1]} → {enhanced_features.shape[1]} "
                        f"(+{physics_features.shape[1]} physics features)")

        return enhanced_features

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'EnhancedPEECOMClassifier':
        """
        Fit the Enhanced PEECOM classifier
        """
        if self.verbose:
            logger.info("Training Enhanced PEECOM classifier...")

        # Store classes
        self.classes_ = np.unique(y)

        # Feature engineering
        X_enhanced = self._engineer_features(X)
        self.feature_names_ = list(X_enhanced.columns)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_enhanced)

        # Create and train ensemble
        self.ensemble = self._create_ensemble(X_scaled.shape[1])
        self.ensemble.fit(X_scaled, y)

        # Perform cross-validation for model assessment
        if self.verbose:
            cv = StratifiedKFold(n_splits=self.cv_folds,
                                 shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                self.ensemble, X_scaled, y, cv=cv, scoring='accuracy')
            logger.info(
                f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        # Feature engineering
        X_enhanced = self._engineer_features(X)

        # Ensure same features as training
        for feature in self.feature_names_:
            if feature not in X_enhanced.columns:
                X_enhanced[feature] = 0
        X_enhanced = X_enhanced[self.feature_names_]

        # Scale and predict
        X_scaled = self.scaler.transform(X_enhanced)
        return self.ensemble.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        # Feature engineering
        X_enhanced = self._engineer_features(X)

        # Ensure same features as training
        for feature in self.feature_names_:
            if feature not in X_enhanced.columns:
                X_enhanced[feature] = 0
        X_enhanced = X_enhanced[self.feature_names_]

        # Scale and predict
        X_scaled = self.scaler.transform(X_enhanced)
        return self.ensemble.predict_proba(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the ensemble"""
        if not self.is_fitted_:
            raise ValueError(
                "Model must be fitted before getting feature importance")

        # Get importance from Random Forest estimators
        rf_estimators = [est for name, est in self.ensemble.named_estimators_.items()
                         if 'rf' in name]

        if not rf_estimators:
            return {}

        # Average importance across RF estimators
        importance_sum = np.zeros(len(self.feature_names_))
        for rf in rf_estimators:
            importance_sum += rf.feature_importances_

        avg_importance = importance_sum / len(rf_estimators)

        return dict(zip(self.feature_names_, avg_importance))

    def get_physics_feature_analysis(self) -> Dict[str, any]:
        """Analyze the contribution of physics features"""
        if not self.is_fitted_ or not self.use_physics_features:
            return {}

        feature_importance = self.get_feature_importance()
        feature_categories = self.physics_engineer.get_feature_categories(
            self.feature_names_)

        physics_importance = np.mean([feature_importance.get(
            f, 0) for f in feature_categories['physics']])
        standard_importance = np.mean([feature_importance.get(
            f, 0) for f in feature_categories['standard']])

        return {
            'physics_features': feature_categories['physics'],
            'standard_features': feature_categories['standard'],
            'physics_avg_importance': physics_importance,
            'standard_avg_importance': standard_importance,
            'physics_advantage': physics_importance / (standard_importance + 1e-6),
            'top_physics_features': sorted(
                [(f, feature_importance.get(f, 0))
                 for f in feature_categories['physics']],
                key=lambda x: x[1], reverse=True
            )[:10]
        }


class DataLeakageDetector:
    """
    Detect and prevent common data leakage issues
    """

    @staticmethod
    def detect_leakage_features(feature_names: List[str], target_names: List[str]) -> List[str]:
        """Detect potential data leakage features"""
        leakage_keywords = [
            'file_id', 'filename', 'index', 'id', 'timestamp',
            'date', 'time', 'row', 'sample', 'cycle_id'
        ]

        leakage_features = []
        for feature in feature_names:
            feature_lower = feature.lower()
            if any(keyword in feature_lower for keyword in leakage_keywords):
                leakage_features.append(feature)

            # Check if feature name is similar to target names
            for target in target_names:
                if target.lower() in feature_lower or feature_lower in target.lower():
                    if feature not in leakage_features:
                        leakage_features.append(feature)

        return leakage_features

    @staticmethod
    def remove_leakage_features(df: pd.DataFrame, target_names: List[str]) -> pd.DataFrame:
        """Remove potential data leakage features"""
        leakage_features = DataLeakageDetector.detect_leakage_features(
            list(df.columns), target_names
        )

        if leakage_features:
            logger.warning(
                f"Removing potential data leakage features: {leakage_features}")
            df = df.drop(columns=leakage_features)

        return df


def create_enhanced_peecom_model(**kwargs) -> EnhancedPEECOMClassifier:
    """
    Factory function to create an Enhanced PEECOM model
    """
    return EnhancedPEECOMClassifier(**kwargs)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Load some example data (replace with actual data loading)
    print("Enhanced PEECOM Classifier - Physics-Informed Hydraulic Condition Monitoring")
    print("This module provides advanced physics features and data leakage prevention.")
