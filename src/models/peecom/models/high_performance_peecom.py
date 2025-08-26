"""
High-Performance PEECOM Model
=============================

This is the flagship PEECOM model optimized for maximum performance.
It uses advanced Random Forest ensembles with optimized hyperparameters
to outperform baseline models.
"""

from features.physics_features import HighPerformancePhysicsFeatures
from base.base_peecom import BasePEECOM
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Any, List, Optional

# Import base class and features
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class HighPerformancePEECOM(BasePEECOM):
    """
    High-Performance PEECOM Model designed to outperform Random Forest baseline.

    This model uses:
    1. Optimized physics feature engineering
    2. Multiple Random Forest variants with different hyperparameters
    3. Intelligent ensemble weighting based on cross-validation performance
    4. Advanced preprocessing and calibration
    """

    def __init__(self,
                 n_estimators: int = 500,
                 max_depth: Optional[int] = 15,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 class_weight: str = 'balanced',
                 random_state: int = 42,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize High-Performance PEECOM.

        Args:
            n_estimators: Number of trees in Random Forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            max_features: Features to consider for splits
            bootstrap: Whether to use bootstrap sampling
            class_weight: Class weighting strategy
            random_state: Random seed
            verbose: Whether to print progress
            **kwargs: Additional parameters
        """
        super().__init__(random_state=random_state, verbose=verbose, **kwargs)

        # Store Random Forest parameters
        self.rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'class_weight': class_weight,
            'random_state': random_state,
            'n_jobs': kwargs.get('n_jobs', -1)
        }

        # Initialize feature engineer
        self.feature_engineer = HighPerformancePhysicsFeatures()

        # Ensemble weights (to be optimized during training)
        self.ensemble_weights = {}

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer high-performance physics features.

        Args:
            X: Raw sensor data

        Returns:
            Engineered feature matrix
        """
        return self.feature_engineer.engineer_features(X)

    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize multiple Random Forest variants for optimal performance.

        Returns:
            Dictionary of initialized models
        """
        models = {}

        # Primary Random Forest (optimized for overall performance)
        models['rf_primary'] = RandomForestClassifier(**self.rf_params)

        # Aggressive Random Forest (more trees, deeper)
        aggressive_params = self.rf_params.copy()
        aggressive_params.update({
            'n_estimators': 600,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        })
        models['rf_aggressive'] = RandomForestClassifier(**aggressive_params)

        # Conservative Random Forest (fewer trees, shallower, more regularization)
        conservative_params = self.rf_params.copy()
        conservative_params.update({
            'n_estimators': 400,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        })
        models['rf_conservative'] = RandomForestClassifier(
            **conservative_params)

        # Diverse Random Forest (different feature sampling)
        diverse_params = self.rf_params.copy()
        diverse_params.update({
            'max_features': 'log2',
            'bootstrap': True,
            'oob_score': True
        })
        models['rf_diverse'] = RandomForestClassifier(**diverse_params)

        # Gradient Boosting for complementary learning
        gb_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'max_features': 'sqrt',
            'random_state': self.random_state
        }
        models['gradient_boosting'] = GradientBoostingClassifier(**gb_params)

        if self.verbose:
            print(
                f"Initialized {len(models)} high-performance models: {list(models.keys())}")

        return models

    def _fit_ensemble(self, X: np.ndarray, y: np.ndarray) -> 'HighPerformancePEECOM':
        """
        Fit ensemble with optimal weighting.

        Args:
            X: Scaled feature matrix
            y: Target labels

        Returns:
            Self (fitted model)
        """
        # Handle class imbalance for all models
        try:
            classes = np.unique(y)
            if len(classes) > 1:
                class_weights = compute_class_weight(
                    'balanced', classes=classes, y=y)
                weight_dict = dict(zip(classes, class_weights))

                # Apply class weights to Random Forest models
                for name, model in self.models.items():
                    if hasattr(model, 'set_params') and 'class_weight' in model.get_params():
                        model.set_params(class_weight=weight_dict)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not set class weights: {e}")

        # Train all models and compute cross-validation scores for weighting
        cv_scores = {}

        for name, model in self.models.items():
            if self.verbose:
                print(f"Training {name}...")

            # Fit model
            model.fit(X, y)

            # Compute cross-validation score for ensemble weighting
            try:
                scores = cross_val_score(
                    model, X, y, cv=3, scoring='accuracy', n_jobs=-1)
                cv_score = scores.mean()
                cv_scores[name] = cv_score

                if self.verbose:
                    print(f"  {name} CV score: {cv_score:.4f}")

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: CV scoring failed for {name}: {e}")
                cv_scores[name] = 0.5  # Default score

        # Calculate ensemble weights based on performance
        self._calculate_ensemble_weights(cv_scores)

        return self

    def _calculate_ensemble_weights(self, cv_scores: Dict[str, float]) -> None:
        """
        Calculate ensemble weights based on cross-validation performance.

        Args:
            cv_scores: Cross-validation scores for each model
        """
        # Convert scores to weights using softmax with temperature scaling
        scores_array = np.array(list(cv_scores.values()))

        # Temperature scaling to emphasize performance differences
        temperature = 5.0
        exp_scores = np.exp(scores_array * temperature)
        weights = exp_scores / exp_scores.sum()

        # Store weights
        self.ensemble_weights = dict(zip(cv_scores.keys(), weights))

        if self.verbose:
            print("Optimized ensemble weights:")
            for name, weight in self.ensemble_weights.items():
                print(f"  {name}: {weight:.3f}")

    def _predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted ensemble.

        Args:
            X: Scaled feature matrix

        Returns:
            Ensemble predictions
        """
        predictions = []
        weights = []

        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.ensemble_weights.get(
                    name, 1.0 / len(self.models)))
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Prediction failed for {name}: {e}")
                continue

        if not predictions:
            raise RuntimeError("No models were able to make predictions")

        # Weighted majority voting
        predictions_array = np.array(predictions)
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()  # Normalize

        # Calculate weighted predictions
        weighted_preds = np.average(
            predictions_array, axis=0, weights=weights_array)
        final_predictions = np.round(weighted_preds).astype(int)

        return final_predictions

    def _predict_proba_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using weighted ensemble.

        Args:
            X: Scaled feature matrix

        Returns:
            Ensemble probability predictions
        """
        probabilities = []
        weights = []

        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    probabilities.append(proba)
                    weights.append(self.ensemble_weights.get(
                        name, 1.0 / len(self.models)))
            except Exception as e:
                if self.verbose:
                    print(
                        f"Warning: Probability prediction failed for {name}: {e}")
                continue

        if not probabilities:
            # Fallback to hard predictions
            predictions = self._predict_ensemble(X)
            n_classes = len(np.unique(predictions))
            n_samples = len(predictions)

            probas = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(predictions):
                probas[i, pred] = 1.0

            return probas

        # Weighted average of probabilities
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()  # Normalize

        ensemble_proba = np.average(
            probabilities, axis=0, weights=weights_array)

        return ensemble_proba

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get aggregated feature importance from ensemble.

        Returns:
            Feature importance dictionary
        """
        if not hasattr(self, 'models') or not self.feature_names_:
            return None

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

        # Normalize by total weight
        if total_weight > 0:
            for feature in importance_scores:
                importance_scores[feature] /= total_weight

        # Sort by importance
        sorted_importance = dict(sorted(importance_scores.items(),
                                        key=lambda x: x[1], reverse=True))

        return sorted_importance

    def get_model_performance(self) -> Dict[str, float]:
        """Get individual model performance scores."""
        return getattr(self, 'ensemble_weights', {})
