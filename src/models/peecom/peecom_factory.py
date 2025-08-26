"""
PEECOM Factory - Model Selection and Instantiation
==================================================

This module provides a factory interface for creating different PEECOM
model variants based on requirements and performance needs.
"""

from typing import Dict, Any, Optional, Union
import warnings

# Import model variants
try:
    from .models.high_performance_peecom import HighPerformancePEECOM
    from .models.lightweight_peecom import LightweightPEECOM
    from .base.base_peecom import BasePEECOM
except ImportError as e:
    warnings.warn(f"Import error in PEECOM factory: {e}")
    # Fallback imports (will be created)
    HighPerformancePEECOM = None
    LightweightPEECOM = None
    BasePEECOM = None


class PEECOMFactory:
    """
    Factory class for creating PEECOM model variants.

    This provides a unified interface for selecting and instantiating
    the appropriate PEECOM model based on performance requirements.
    """

    # Available model variants
    MODELS = {
        'high_performance': {
            'class': HighPerformancePEECOM,
            'display_name': 'High-Performance PEECOM',
            'description': 'Maximum accuracy with advanced ensemble techniques',
            'pros': ['Highest accuracy', 'Advanced ensemble', 'Optimized hyperparameters', 'Physics-informed features'],
            'cons': ['Higher computational cost', 'Longer training time'],
            'recommended_for': ['Production deployment', 'Competitive benchmarking', 'Research papers']
        },
        'lightweight': {
            'class': LightweightPEECOM,
            'display_name': 'Lightweight PEECOM',
            'description': 'Fast and efficient with core physics principles',
            'pros': ['Fast training', 'Low memory usage', 'Simple deployment', 'Good baseline performance'],
            'cons': ['Lower accuracy than high-performance variant'],
            'recommended_for': ['Real-time applications', 'Edge deployment', 'Prototyping']
        }
    }

    @classmethod
    def create_model(cls,
                     model_type: str = 'high_performance',
                     **kwargs) -> Optional[BasePEECOM]:
        """
        Create a PEECOM model instance.

        Args:
            model_type: Type of PEECOM model ('high_performance', 'lightweight')
            **kwargs: Parameters to pass to the model constructor

        Returns:
            PEECOM model instance

        Raises:
            ValueError: If model_type is not recognized
        """
        if model_type not in cls.MODELS:
            available_models = list(cls.MODELS.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. Available models: {available_models}")

        model_config = cls.MODELS[model_type]
        model_class = model_config['class']

        if model_class is None:
            raise RuntimeError(
                f"Model class for '{model_type}' is not available. Check imports.")

        try:
            return model_class(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create {model_type} model: {e}")

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available PEECOM models.

        Returns:
            Dictionary with model information
        """
        return cls.MODELS.copy()

    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model type.

        Args:
            model_type: Type of PEECOM model

        Returns:
            Model information dictionary

        Raises:
            ValueError: If model_type is not recognized
        """
        if model_type not in cls.MODELS:
            available_models = list(cls.MODELS.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. Available models: {available_models}")

        return cls.MODELS[model_type].copy()

    @classmethod
    def recommend_model(cls,
                        priority: str = 'accuracy',
                        constraints: Optional[Dict[str, Any]] = None) -> str:
        """
        Recommend a PEECOM model based on requirements.

        Args:
            priority: Main priority ('accuracy', 'speed', 'memory')
            constraints: Additional constraints (e.g., {'max_training_time': 300})

        Returns:
            Recommended model type
        """
        constraints = constraints or {}

        if priority == 'accuracy':
            return 'high_performance'
        elif priority in ['speed', 'memory', 'efficiency']:
            return 'lightweight'
        else:
            # Default recommendation based on constraints
            if constraints.get('max_training_time', float('inf')) < 60:
                return 'lightweight'
            elif constraints.get('max_memory_mb', float('inf')) < 500:
                return 'lightweight'
            else:
                return 'high_performance'

    @classmethod
    def create_optimized_model(cls,
                               X_shape: tuple,
                               target_type: str = 'multiclass',
                               performance_target: float = 0.95,
                               **kwargs) -> BasePEECOM:
        """
        Create an optimized PEECOM model based on data characteristics.

        Args:
            X_shape: Shape of training data (n_samples, n_features)
            target_type: Type of target ('binary', 'multiclass')
            performance_target: Target accuracy (0.0 to 1.0)
            **kwargs: Additional model parameters

        Returns:
            Optimized PEECOM model instance
        """
        n_samples, n_features = X_shape

        # Automatic model selection based on data characteristics
        if performance_target >= 0.98:
            model_type = 'high_performance'
            # Increase ensemble size for very high accuracy targets
            kwargs.setdefault('n_estimators', 600)
        elif n_samples < 1000 or n_features > 100:
            # Use lightweight for small datasets or high-dimensional data
            model_type = 'lightweight'
        else:
            model_type = 'high_performance'

        # Adjust parameters based on target type
        if target_type == 'binary':
            kwargs.setdefault('class_weight', 'balanced')

        return cls.create_model(model_type, **kwargs)


# Convenience functions for direct model creation
def create_high_performance_peecom(**kwargs) -> Optional[BasePEECOM]:
    """Create a high-performance PEECOM model."""
    return PEECOMFactory.create_model('high_performance', **kwargs)


def create_lightweight_peecom(**kwargs) -> Optional[BasePEECOM]:
    """Create a lightweight PEECOM model."""
    return PEECOMFactory.create_model('lightweight', **kwargs)


def create_auto_peecom(X_shape: tuple, **kwargs) -> BasePEECOM:
    """Create an automatically optimized PEECOM model."""
    return PEECOMFactory.create_optimized_model(X_shape, **kwargs)
