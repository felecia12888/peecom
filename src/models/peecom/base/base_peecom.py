"""
Base PEECOM Model Class
=======================

This module provides the abstract base class for all PEECOM model variants.
It defines the common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union


class BasePEECOM(BaseEstimator, ClassifierMixin, ABC):
    """
    Abstract base class for all PEECOM model variants.

    This class defines the common interface and shared functionality
    for all Physics-Enhanced Equipment Condition Monitoring models.
    """

    def __init__(self,
                 random_state: int = 42,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize base PEECOM model.

        Args:
            random_state: Random seed for reproducibility
            verbose: Whether to print progress messages
            **kwargs: Additional model-specific parameters
        """
        self.random_state = random_state
        self.verbose = verbose

        # Common components
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.is_fitted_ = False

        # Model-specific parameters
        self.model_params = kwargs

    @abstractmethod
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer physics-based features from raw sensor data.

        Args:
            X: Raw sensor data

        Returns:
            Engineered feature matrix
        """
        pass

    @abstractmethod
    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize the base models for the ensemble.

        Returns:
            Dictionary of initialized models
        """
        pass

    @abstractmethod
    def _fit_ensemble(self, X: np.ndarray, y: np.ndarray) -> 'BasePEECOM':
        """
        Fit the ensemble of models.

        Args:
            X: Scaled feature matrix
            y: Target labels

        Returns:
            Self (fitted model)
        """
        pass

    @abstractmethod
    def _predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble.

        Args:
            X: Scaled feature matrix

        Returns:
            Predictions
        """
        pass

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> 'BasePEECOM':
        """
        Fit the PEECOM model.

        Args:
            X: Input features
            y: Target labels

        Returns:
            Self (fitted model)
        """
        if self.verbose:
            print(f"Training {self.__class__.__name__}...")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(
                X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        # Engineer features
        if self.verbose:
            print("Engineering physics-based features...")
        X_engineered = self._engineer_features(X)

        # Store feature names
        self.feature_names_ = X_engineered.columns.tolist(
        ) if hasattr(X_engineered, 'columns') else None

        # Scale features
        if self.verbose:
            print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X_engineered)

        # Initialize models
        if self.verbose:
            print("Initializing ensemble models...")
        self.models = self._initialize_models()

        # Fit ensemble
        if self.verbose:
            print("Training ensemble...")
        self._fit_ensemble(X_scaled, y)

        self.is_fitted_ = True

        if self.verbose:
            print(f"{self.__class__.__name__} training completed!")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(
                X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        # Apply same preprocessing
        X_engineered = self._engineer_features(X)
        X_scaled = self.scaler.transform(X_engineered)

        # Make predictions
        return self._predict_ensemble(X_scaled)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        if not hasattr(self, '_predict_proba_ensemble'):
            # Fallback for models without probability prediction
            predictions = self.predict(X)
            n_classes = len(np.unique(predictions))
            n_samples = len(predictions)

            # Create dummy probabilities (1.0 for predicted class, 0.0 for others)
            probas = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(predictions):
                probas[i, pred] = 1.0

            return probas

        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(
                X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        # Apply same preprocessing
        X_engineered = self._engineer_features(X)
        X_scaled = self.scaler.transform(X_engineered)

        # Make probability predictions
        return self._predict_proba_ensemble(X_scaled)

    def get_feature_names(self) -> Optional[list]:
        """Get feature names after engineering."""
        return self.feature_names_

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters."""
        return {
            'model_type': self.__class__.__name__,
            'is_fitted': self.is_fitted_,
            'n_features': len(self.feature_names_) if self.feature_names_ else None,
            'random_state': self.random_state,
            'model_params': self.model_params
        }
