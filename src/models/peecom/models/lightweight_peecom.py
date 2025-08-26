"""
Lightweight PEECOM Model
========================

This is a simpler, faster PEECOM variant that focuses on core physics
principles with minimal computational overhead.
"""

from features.physics_features import LightweightPhysicsFeatures
from base.base_peecom import BasePEECOM
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any

# Import base class and features
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class LightweightPEECOM(BasePEECOM):
    """
    Lightweight PEECOM Model for fast inference and training.

    Uses core physics principles with a single optimized Random Forest.
    """

    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: int = 10,
                 random_state: int = 42,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize Lightweight PEECOM.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
            verbose: Whether to print progress
            **kwargs: Additional parameters
        """
        super().__init__(random_state=random_state, verbose=verbose, **kwargs)

        self.rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': 'balanced',
            'random_state': random_state,
            'n_jobs': kwargs.get('n_jobs', -1)
        }

        # Initialize lightweight feature engineer
        self.feature_engineer = LightweightPhysicsFeatures()

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer lightweight physics features."""
        return self.feature_engineer.engineer_features(X)

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize single Random Forest model."""
        models = {
            'random_forest': RandomForestClassifier(**self.rf_params)
        }
        return models

    def _fit_ensemble(self, X: np.ndarray, y: np.ndarray) -> 'LightweightPEECOM':
        """Fit single Random Forest model."""
        if self.verbose:
            print("Training Random Forest...")

        # Handle class imbalance
        try:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y)
            if len(classes) > 1:
                class_weights = compute_class_weight(
                    'balanced', classes=classes, y=y)
                weight_dict = dict(zip(classes, class_weights))
                self.models['random_forest'].set_params(
                    class_weight=weight_dict)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not set class weights: {e}")

        # Fit model
        self.models['random_forest'].fit(X, y)

        return self

    def _predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using single Random Forest."""
        return self.models['random_forest'].predict(X)

    def _predict_proba_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using single Random Forest."""
        return self.models['random_forest'].predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest."""
        if not hasattr(self, 'models') or not self.feature_names_:
            return {}

        model = self.models['random_forest']
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(
                zip(self.feature_names_, model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return {}
