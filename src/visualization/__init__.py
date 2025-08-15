#!/usr/bin/env python3
"""
Visualization Package

Publication-quality visualization tools for the PEECOM project.
Provides comprehensive plotting capabilities for data analysis, model performance,
and physics-enhanced insights following DRY principles.

Classes:
    BaseVisualizer: Common functionality and styling for all visualizers
    PerformanceVisualizer: Model performance comparisons and metrics
    DataVisualizer: Dataset analysis and sensor data exploration
    ModelVisualizer: Model-specific insights and feature analysis

Usage:
    from src.visualization import PerformanceVisualizer
    
    viz = PerformanceVisualizer(output_dir='figures')
    plots = viz.generate_all_performance_plots()
"""

from .base_visualizer import BaseVisualizer
from .performance_visualizer import PerformanceVisualizer
from .data_visualizer import DataVisualizer
from .model_visualizer import ModelVisualizer

__version__ = "1.0.0"
__author__ = "PEECOM Team"

__all__ = [
    'BaseVisualizer',
    'PerformanceVisualizer',
    'DataVisualizer',
    'ModelVisualizer'
]
