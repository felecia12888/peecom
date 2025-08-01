"""
PEECOM Source Package

Main package containing all PEECOM functionality organized into modules:
- loader: Data loading and preprocessing
- models: Model definitions and training
- utils: Utility functions and helpers
- config: Configuration files and templates
"""

from . import loader
from . import models
from . import utils

__all__ = ['loader', 'models', 'utils']
