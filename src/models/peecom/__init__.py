"""
PEECOM Models Package

This package contains all PEECOM model variants:
- base: Core PEECOM with physics features and optimization
- physics_enhanced: Advanced physics feature engineering
- adaptive: Automatic classifier selection

All models follow scikit-learn API conventions.
"""

from .base import PEECOM
from .physics_enhanced import PhysicsEnhancedPEECOM
from .adaptive import AdaptivePEECOM

__all__ = ['PEECOM', 'PhysicsEnhancedPEECOM', 'AdaptivePEECOM']
