"""
PEECOM Package - Physics-Enhanced Equipment Condition Monitoring
================================================================

This package contains all PEECOM model variants and components in a modular,
scalable architecture for hydraulic system condition monitoring.

Structure:
- base/: Base classes and interfaces
- features/: Feature engineering modules
- models/: Different PEECOM model implementations
- utils/: Utility functions
"""

__version__ = "2.0.0"
__author__ = "PEECOM Research Team"

# Try to import main classes for easy access
try:
    from .peecom_factory import PEECOMFactory
    from .base.base_peecom import BasePEECOM

    # Export main interface
    __all__ = ['PEECOMFactory', 'BasePEECOM']
except ImportError as e:
    # Handle import errors gracefully
    print(f"Warning: Some PEECOM components may not be available: {e}")
    __all__ = []
