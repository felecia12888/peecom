#!/usr/bin/env python3
"""
Simple Fast PEECOM - Lightweight and Fast Physics-Enhanced Model

This is a streamlined version focused on:
1. Fast training (< 30 seconds)
2. Simple physics features
3. Reliable performance
4. No complexity overhead
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class SimplePEECOM(BaseEstimator, ClassifierMixin):
    """
    Simple PEECOM - Fast and reliable physics-enhanced classifier
    """

    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.original_features = None

    def _create_physics_features(self, X):
        """Create simple physics-inspired features quickly"""

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(
                X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Start with original features
        features = X.copy()

        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return features

        # Simple physics features (only the most effective ones)
        try:
            # Energy-related features (simple combinations)
            # Limit to first 5 features
            for i, col1 in enumerate(numeric_cols[:5]):
                # Limit combinations
                for j, col2 in enumerate(numeric_cols[i+1:6], i+1):
                    if col1 != col2:
                        # Power feature (multiplication)
                        power_name = f'power_{col1}_{col2}'
                        features[power_name] = X[col1] * X[col2]

                        # Ratio feature (safer division)
                        ratio_name = f'ratio_{col1}_{col2}'
                        features[ratio_name] = X[col1] / (X[col2] + 1e-8)

            # Statistical features (simple ones)
            features['mean_all'] = X[numeric_cols].mean(axis=1)
            features['std_all'] = X[numeric_cols].std(axis=1)
            features['max_all'] = X[numeric_cols].max(axis=1)
            features['min_all'] = X[numeric_cols].min(axis=1)

            # Simple physics ratios
            features['max_min_ratio'] = features['max_all'] / \
                (features['min_all'] + 1e-8)
            features['std_mean_ratio'] = features['std_all'] / \
                (features['mean_all'] + 1e-8)

        except Exception as e:
            print(f"Warning: Physics feature creation simplified due to: {e}")
            # If anything fails, just use original features
            pass

        # Clean up any infinite or NaN values
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)

        return features

    def fit(self, X, y):
        """Fit the Simple PEECOM model"""

        print("Training Simple PEECOM model (fast version)...")
        # Create physics features
        X_enhanced = self._create_physics_features(X)
        # Store feature names for provenance
        self.feature_names_ = list(X_enhanced.columns) if hasattr(X_enhanced, 'columns') else [f'f_{i}' for i in range(X_enhanced.shape[1])]

        # Store original feature info
        self.original_features = X.shape[1] if hasattr(
            X, 'shape') else len(X[0])

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_enhanced)

        # Use simple Random Forest (fast and reliable)
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1  # Use all cores for speed
        )

        # Fit the model
        self.model.fit(X_scaled, y)

        print(f"✅ Simple PEECOM trained successfully!")
        print(
            f"   Features: {self.original_features} → {X_enhanced.shape[1]} (with physics)")

        return self

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")

        # Create physics features
        X_enhanced = self._create_physics_features(X)

        # Scale features
        X_scaled = self.scaler.transform(X_enhanced)

        # Predict
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")

        # Create physics features
        X_enhanced = self._create_physics_features(X)

        # Scale features
        X_scaled = self.scaler.transform(X_enhanced)

        # Predict probabilities
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self):
        """Return feature importances if available"""
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def get_feature_names(self):
        return getattr(self, 'feature_names_', None)

    def get_feature_importance(self):
        """Get feature importance from the underlying Random Forest model"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")

        # Random Forest has feature_importances_ attribute
        return self.model.feature_importances_


# For backward compatibility
class WinningPEECOM(SimplePEECOM):
    """Alias for SimplePEECOM to maintain compatibility"""
    pass


# Export the class
__all__ = ['SimplePEECOM', 'WinningPEECOM']
