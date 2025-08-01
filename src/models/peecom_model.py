#!/usr/bin/env python3
"""
PEECOM Model for Hydraulic System Condition Monitoring

A simplified PEECOM (Physics-Enhanced Equipment Condition Monitoring) model
that integrates physics-inspired features with machine learning for hydraulic
system anomaly detection and condition monitoring.

This implementation uses scikit-learn for compatibility with the existing workflow.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd


class PEECOMModel:
    """
    PEECOM: Physics-Enhanced Equipment Condition Monitoring Model

    A hybrid model that combines:
    1. Physics-inspired feature engineering
    2. Class-balanced learning for anomaly detection
    3. Multi-target prediction (anomaly + condition states)
    4. Ensemble methods for robust predictions
    """

    def __init__(self, **kwargs):
        """Initialize PEECOM model with physics-enhanced features"""
        default_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'  # Handle imbalanced data
        }

        # Update defaults with any provided kwargs
        default_params.update(kwargs)
        self.params = default_params
        self.model = RandomForestClassifier(**self.params)
        self.name = "peecom"
        self.display_name = "PEECOM (Physics-Enhanced)"

        # Physics-enhanced feature engineering
        self.scaler = StandardScaler()
        self.feature_engineered = False

    def _engineer_physics_features(self, X):
        """
        Engineer physics-inspired features for hydraulic systems

        Based on hydraulic system physics:
        - Energy relationships between pressure, flow, and temperature
        - Efficiency indicators
        - Anomaly detection features
        """
        if isinstance(X, pd.DataFrame):
            features = X.copy()
        else:
            # If numpy array, create DataFrame with generic column names
            features = pd.DataFrame(
                X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        # Identify sensor groups based on column names
        pressure_cols = [
            col for col in features.columns if col.startswith('PS')]
        flow_cols = [col for col in features.columns if col.startswith('FS')]
        temp_cols = [col for col in features.columns if col.startswith('TS')]
        motor_cols = [col for col in features.columns if col.startswith('EPS')]
        efficiency_cols = [col for col in features.columns if any(
            col.startswith(x) for x in ['CE', 'CP', 'SE'])]

        # Physics-inspired feature engineering
        # Collect all new features first to avoid DataFrame fragmentation
        new_features = {}

        try:
            # 1. Energy-based features
            if pressure_cols and flow_cols:
                # Hydraulic power approximation: P = pressure * flow_rate
                for p_col in pressure_cols[:3]:  # Use first 3 pressure sensors
                    for f_col in flow_cols:
                        if p_col in features.columns and f_col in features.columns:
                            power_col = f"hydraulic_power_{p_col}_{f_col}"
                            new_features[power_col] = features[p_col] * \
                                features[f_col]

            # 2. Pressure system health indicators
            if len(pressure_cols) >= 2:
                # Pressure differentials (indicator of system health)
                for i in range(len(pressure_cols)-1):
                    p1, p2 = pressure_cols[i], pressure_cols[i+1]
                    if p1 in features.columns and p2 in features.columns:
                        diff_col = f"pressure_diff_{p1}_{p2}"
                        new_features[diff_col] = features[p1] - \
                            features[p2]

                # Pressure ratios
                if 'PS1_mean' in features.columns and 'PS2_mean' in features.columns:
                    new_features['pressure_ratio_PS1_PS2'] = (
                        features['PS1_mean'] / (features['PS2_mean'] + 1e-6)
                    )

            # 3. Temperature-based efficiency indicators
            if temp_cols and motor_cols:
                for t_col in temp_cols:
                    for m_col in motor_cols:
                        if t_col in features.columns and m_col in features.columns:
                            # Thermal efficiency approximation
                            eff_col = f"thermal_efficiency_{t_col}_{m_col}"
                            new_features[eff_col] = (
                                features[m_col] / (features[t_col] + 1e-6)
                            )

            # 4. System stability indicators
            if pressure_cols:
                # Coefficient of variation for pressure sensors (stability measure)
                pressure_values = features[pressure_cols].values
                if pressure_values.shape[1] > 1:
                    pressure_mean = np.mean(pressure_values, axis=1)
                    pressure_std = np.std(pressure_values, axis=1)
                    new_features['pressure_stability'] = (
                        pressure_std / (pressure_mean + 1e-6)
                    )

            # 5. Anomaly detection features
            if efficiency_cols:
                # Overall system efficiency
                efficiency_values = features[efficiency_cols].values
                new_features['system_efficiency'] = np.mean(
                    efficiency_values, axis=1)

                # Efficiency variance (anomaly indicator)
                if efficiency_values.shape[1] > 1:
                    new_features['efficiency_variance'] = np.var(
                        efficiency_values, axis=1)

            # 6. Flow-based features
            if flow_cols and len(flow_cols) >= 2:
                # Flow balance (should be conserved in healthy systems)
                flow_values = features[flow_cols].values
                new_features['flow_imbalance'] = np.std(
                    flow_values, axis=1)

            # 7. Motor load indicators
            if motor_cols:
                for m_col in motor_cols:
                    if m_col in features.columns:
                        # Motor efficiency indicators
                        base_name = m_col.replace(
                            '_mean', '').replace('_std', '')

                        # Peak-to-average ratio (load variation indicator)
                        peak_col = f"{base_name}_peak"
                        if peak_col in features.columns:
                            ratio_col = f"motor_load_ratio_{base_name}"
                            new_features[ratio_col] = (
                                features[peak_col] / (features[m_col] + 1e-6)
                            )

            # Combine original features with new features efficiently
            if new_features:
                new_features_df = pd.DataFrame(
                    new_features, index=features.index)
                engineered_features = pd.concat(
                    [features, new_features_df], axis=1)
            else:
                engineered_features = features.copy()

        except Exception as e:
            print(
                f"Warning: Some physics features could not be engineered: {e}")
            engineered_features = features.copy()

        # Remove any infinite or NaN values
        engineered_features = engineered_features.replace(
            [np.inf, -np.inf], np.nan)
        engineered_features = engineered_features.fillna(0)

        return engineered_features

    def _compute_class_weights(self, y):
        """Compute class weights for imbalanced data"""
        try:
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            return dict(zip(classes, weights))
        except:
            return None

    def get_model(self):
        """Return the sklearn model instance"""
        return self.model

    def get_params(self):
        """Return model parameters"""
        return self.params

    def get_param_grid(self):
        """Return parameter grid for hyperparameter tuning"""
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }

    def fit(self, X, y):
        """Fit the PEECOM model with physics-enhanced features"""
        # Engineer physics-inspired features
        X_engineered = self._engineer_physics_features(X)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_engineered)

        # Compute class weights for imbalanced data
        class_weights = self._compute_class_weights(y)
        if class_weights:
            self.model.set_params(class_weight=class_weights)

        # Train the model
        self.model.fit(X_scaled, y)
        self.feature_engineered = True
        self.feature_names_ = X_engineered.columns.tolist(
        ) if hasattr(X_engineered, 'columns') else None

        return self

    def predict(self, X):
        """Make predictions using physics-enhanced features"""
        if not self.feature_engineered:
            raise ValueError("Model must be fitted before making predictions")

        # Engineer same physics features
        X_engineered = self._engineer_physics_features(X)

        # Scale features using fitted scaler
        X_scaled = self.scaler.transform(X_engineered)

        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.feature_engineered:
            raise ValueError("Model must be fitted before making predictions")

        # Engineer same physics features
        X_engineered = self._engineer_physics_features(X)

        # Scale features using fitted scaler
        X_scaled = self.scaler.transform(X_engineered)

        return self.model.predict_proba(X_scaled)

    def tune_hyperparameters(self, X_train, y_train, cv=5):
        """Perform hyperparameter tuning using GridSearchCV"""
        # Engineer features first
        X_engineered = self._engineer_physics_features(X_train)
        X_scaled = self.scaler.fit_transform(X_engineered)

        param_grid = self.get_param_grid()
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='f1_weighted',  # Good for imbalanced data
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_scaled, y_train)

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.params.update(grid_search.best_params_)
        self.feature_engineered = True

        return grid_search.best_params_, grid_search.best_score_

    def get_feature_importance(self):
        """Return feature importance including engineered features"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def get_physics_insights(self):
        """Return insights about physics-based features"""
        if not self.feature_engineered or not self.feature_names_:
            return "Model not fitted with feature engineering"

        importance = self.get_feature_importance()
        if importance is None:
            return "Feature importance not available"

        # Identify physics-engineered features
        physics_features = [
            name for name in self.feature_names_
            if any(keyword in name for keyword in [
                'hydraulic_power', 'pressure_diff', 'pressure_ratio',
                'thermal_efficiency', 'pressure_stability', 'system_efficiency',
                'efficiency_variance', 'flow_imbalance', 'motor_load_ratio'
            ])
        ]

        insights = {
            'total_features': len(self.feature_names_),
            'physics_features': len(physics_features),
            'physics_feature_names': physics_features
        }

        if len(physics_features) > 0:
            # Get importance of physics features
            physics_indices = [self.feature_names_.index(
                name) for name in physics_features]
            physics_importance = importance[physics_indices]

            insights['physics_importance_sum'] = np.sum(physics_importance)
            insights['physics_importance_mean'] = np.mean(physics_importance)
            insights['top_physics_features'] = [
                (name, importance[self.feature_names_.index(name)])
                for name in physics_features
            ]
            insights['top_physics_features'].sort(
                key=lambda x: x[1], reverse=True)
            insights['top_physics_features'] = insights['top_physics_features'][:5]

        return insights

    def __str__(self):
        return f"{self.display_name} ({self.name})"
