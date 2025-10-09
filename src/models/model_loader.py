#!/usr/bin/env python3
"""
Model Loader for PEECOM Hydraulic System Condition Monitoring

This module provides a centralized way to load and manage different machine learning models
for hydraulic system condition monitoring. All available models are registered here and
can be accessed by name.
"""

from .svm import SVM
from .linear import LogisticRegression
from .boosting import GradientBoosting
from .forest import RandomForest
from .peecom import PEECOM, PhysicsEnhancedPEECOM, AdaptivePEECOM
import os
import sys
from typing import Dict, List, Optional, Any

# Add src to path for imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import all available models from new structure


class ModelLoader:
    """
    Centralized model loader and manager for PEECOM models.

    This class provides a unified interface to access all available models,
    their parameters, and functionality for model selection and instantiation.
    """

    def __init__(self):
        """Initialize the model loader with all available models"""
        self._models = self._register_models()

    def _register_models(self) -> Dict[str, Any]:
        """Register all available models"""
        models = {
            'random_forest': {
                'class': RandomForest,
                'display_name': 'Random Forest',
                'description': 'Ensemble method using multiple decision trees',
                'suitable_for': ['multi-class', 'feature_importance', 'robust'],
                'pros': ['High accuracy', 'Feature importance', 'Handles overfitting well'],
                'cons': ['Can be slow on large datasets', 'Less interpretable']
            },
            'logistic_regression': {
                'class': LogisticRegression,
                'display_name': 'Logistic Regression',
                'description': 'Linear model for classification problems',
                'suitable_for': ['multi-class', 'interpretable', 'fast'],
                'pros': ['Fast training', 'Interpretable', 'Good baseline'],
                'cons': ['Assumes linear relationships', 'May need feature engineering']
            },
            'svm': {
                'class': SVM,
                'display_name': 'Support Vector Machine',
                'description': 'Finds optimal hyperplane for classification',
                'suitable_for': ['multi-class', 'high_dimensional', 'robust'],
                'pros': ['Effective in high dimensions', 'Memory efficient', 'Versatile'],
                'cons': ['Slow on large datasets', 'Sensitive to feature scaling']
            },
            'gradient_boosting': {
                'class': GradientBoosting,
                'display_name': 'Gradient Boosting',
                'description': 'Sequential ensemble of weak learners',
                'suitable_for': ['multi-class', 'feature_importance', 'high_accuracy'],
                'pros': ['High accuracy', 'Feature importance', 'Handles missing data'],
                'cons': ['Can overfit', 'Sensitive to hyperparameters', 'Longer training time']
            },
            'peecom': {
                'class': PEECOM,
                'display_name': 'PEECOM',
                'description': 'Physics-Enhanced Equipment Condition Monitoring with physics-informed features',
                'suitable_for': ['hydraulic_systems', 'physics_aware', 'condition_monitoring', 'fast'],
                'pros': ['Physics-informed features', 'Fast training', 'Reliable performance', 'Feature selection'],
                'cons': ['Requires domain knowledge for feature engineering']
            },
            'physics_enhanced_peecom': {
                'class': PhysicsEnhancedPEECOM,
                'display_name': 'Physics-Enhanced PEECOM',
                'description': 'Advanced PEECOM with sophisticated feature engineering and optimization',
                'suitable_for': ['hydraulic_systems', 'high_performance', 'physics_aware', 'feature_engineering'],
                'pros': ['Advanced physics features', 'Feature selection', 'Polynomial features', 'Best performance'],
                'cons': ['Longer training time', 'More complex pipeline', 'Higher computational cost']
            },
            'adaptive_peecom': {
                'class': AdaptivePEECOM,
                'display_name': 'Adaptive PEECOM',
                'description': 'PEECOM with automatic classifier selection based on physics feature benefits',
                'suitable_for': ['hydraulic_systems', 'adaptive_selection', 'physics_aware', 'optimal_performance'],
                'pros': ['Automatic classifier optimization', 'Physics-informed features', 'Adaptive selection'],
                'cons': ['Longer training time', 'More complex than base PEECOM']
            }
        }
        return models

    def get_available_models(self) -> List[str]:
        """Get list of all available model names"""
        return list(self._models.keys())

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        if model_name not in self._models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {self.get_available_models()}")
        return self._models[model_name]

    def load_model(self, model_name: str, **kwargs) -> Any:
        """
        Load and instantiate a model by name

        Args:
            model_name: Name of the model to load
            **kwargs: Additional parameters to pass to the model constructor

        Returns:
            Instantiated model object
        """
        if model_name not in self._models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {self.get_available_models()}")

        model_class = self._models[model_name]['class']
        return model_class(**kwargs)

    def get_model_display_name(self, model_name: str) -> str:
        """Get the display name for a model"""
        return self._models[model_name]['display_name']

    def get_model_description(self, model_name: str) -> str:
        """Get the description for a model"""
        return self._models[model_name]['description']

    def list_models(self, verbose: bool = False) -> None:
        """
        Print a list of all available models

        Args:
            verbose: If True, show detailed information for each model
        """
        print("Available Models:")
        print("=" * 50)

        for name, info in self._models.items():
            print(f"\n{name}:")
            print(f"  Display Name: {info['display_name']}")
            print(f"  Description: {info['description']}")

            if verbose:
                print(f"  Suitable for: {', '.join(info['suitable_for'])}")
                print(f"  Pros: {', '.join(info['pros'])}")
                print(f"  Cons: {', '.join(info['cons'])}")

    def recommend_model(self, criteria: List[str]) -> List[str]:
        """
        Recommend models based on given criteria

        Args:
            criteria: List of criteria (e.g., ['interpretable', 'fast'])

        Returns:
            List of recommended model names
        """
        recommendations = []

        for name, info in self._models.items():
            if any(criterion in info['suitable_for'] for criterion in criteria):
                recommendations.append(name)

        return recommendations

    def get_all_model_names_for_cli(self) -> List[str]:
        """Get all model names formatted for CLI argument choices"""
        return list(self._models.keys())


# Global instance for easy access
model_loader = ModelLoader()


def get_model(model_name: str, **kwargs):
    """Convenience function to get a model instance"""
    return model_loader.load_model(model_name, **kwargs)


def list_available_models(verbose: bool = False):
    """Convenience function to list all available models"""
    model_loader.list_models(verbose)


def get_model_choices():
    """Get model choices for argument parser"""
    return model_loader.get_all_model_names_for_cli()


if __name__ == "__main__":
    # Demo the model loader
    print("PEECOM Model Loader Demo")
    print("=" * 40)

    # List all models
    list_available_models(verbose=True)

    # Load a specific model
    print(f"\nLoading Random Forest model...")
    rf_model = get_model('random_forest', n_estimators=50)
    print(f"Loaded: {rf_model}")

    # Get recommendations
    print(f"\nModels suitable for interpretability:")
    recommendations = model_loader.recommend_model(['interpretable'])
    for rec in recommendations:
        print(f"  - {rec}: {model_loader.get_model_display_name(rec)}")
