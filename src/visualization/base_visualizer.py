#!/usr/bin/env python3
"""
Base Visualizer Class

Provides common functionality for all visualizers following DRY principles.
Contains shared plotting functions, styling, and configuration.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BaseVisualizer:
    """
    Base class for all visualizers with common functionality.

    Follows DRY principles by centralizing:
    - Plot styling and themes
    - Color palettes
    - Figure saving and formatting
    - Common plot elements
    """

    # Publication-quality style configuration
    STYLE_CONFIG = {
        'figure.figsize': (12, 8),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif'
    }

    # Professional color palettes
    COLOR_PALETTES = {
        'models': {
            'peecom': '#2E86AB',           # Professional blue
            'random_forest': '#A23B72',    # Deep magenta
            'logistic_regression': '#F18F01',  # Orange
            'svm': '#C73E1D',              # Red
            'gradient_boosting': '#592E83'  # Purple
        },
        'targets': {
            'cooler_condition': '#1f77b4',
            'valve_condition': '#ff7f0e',
            'pump_leakage': '#2ca02c',
            'accumulator_pressure': '#d62728',
            'stable_flag': '#9467bd'
        },
        'conditions': {
            'good': '#2ca02c',      # Green
            'medium': '#ff7f0e',    # Orange
            'critical': '#d62728'   # Red
        },
        'sequential': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
        'diverging': ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
    }

    def __init__(self, output_dir='output/figures', theme='publication'):
        """
        Initialize base visualizer.

        Args:
            output_dir: Directory to save figures
            theme: Visualization theme ('publication', 'presentation', 'paper')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.theme = theme

        # Apply styling
        self._setup_style()

    def _setup_style(self):
        """Setup matplotlib and seaborn styling for publication quality."""
        # Apply custom rcParams
        plt.rcParams.update(self.STYLE_CONFIG)

        # Set seaborn style
        sns.set_style("whitegrid", {
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.alpha': 0.3
        })

        # Set color palette
        sns.set_palette("husl")

    def get_model_color(self, model_name):
        """Get consistent color for model across all plots."""
        return self.COLOR_PALETTES['models'].get(model_name, '#333333')

    def get_target_color(self, target_name):
        """Get consistent color for target across all plots."""
        return self.COLOR_PALETTES['targets'].get(target_name, '#333333')

    def save_figure(self, fig, filename, formats=['png', 'pdf'], close=True):
        """
        Save figure in multiple formats with consistent naming.

        Args:
            fig: Matplotlib figure object
            filename: Base filename (without extension)
            formats: List of formats to save
            close: Whether to close figure after saving
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_files = []
        for fmt in formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(filepath, format=fmt, bbox_inches='tight',
                        dpi=300, facecolor='white', edgecolor='none')
            saved_files.append(filepath)

        if close:
            plt.close(fig)

        return saved_files

    def create_figure(self, nrows=1, ncols=1, figsize=None, **kwargs):
        """Create figure with consistent styling."""
        if figsize is None:
            width = self.STYLE_CONFIG['figure.figsize'][0] * ncols
            height = self.STYLE_CONFIG['figure.figsize'][1] * nrows * 0.8
            figsize = (width, height)

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
        return fig, axes

    def add_significance_bar(self, ax, x1, x2, y, text, height=0.02):
        """Add significance comparison bar between two points."""
        ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y], 'k-', linewidth=1)
        ax.text((x1+x2)*0.5, y+height, text,
                ha='center', va='bottom', fontsize=9)

    def format_percentage(self, value):
        """Format value as percentage with appropriate precision."""
        if value >= 0.99:
            return f"{value:.1%}"
        elif value >= 0.9:
            return f"{value:.2%}"
        else:
            return f"{value:.1%}"

    def add_grid(self, ax, alpha=0.3):
        """Add subtle grid to axes."""
        ax.grid(True, alpha=alpha, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

    def set_spine_style(self, ax, show_top=False, show_right=False):
        """Set consistent spine styling."""
        ax.spines['top'].set_visible(show_top)
        ax.spines['right'].set_visible(show_right)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

    def create_legend_elements(self, items, palette_key='models'):
        """Create legend elements with consistent colors."""
        elements = []
        palette = self.COLOR_PALETTES[palette_key]

        for item in items:
            color = palette.get(item, '#333333')
            patch = mpatches.Patch(
                color=color, label=item.replace('_', ' ').title())
            elements.append(patch)

        return elements

    def annotate_best_performer(self, ax, x, y, text, color='green'):
        """Annotate the best performing point."""
        ax.annotate(text, xy=(x, y), xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor=color, alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    def add_model_comparison_table(self, fig, data, position=(0.02, 0.02, 0.25, 0.2)):
        """Add a summary table to the figure."""
        table_ax = fig.add_axes(position)
        table_ax.axis('off')

        # Create table
        table = table_ax.table(cellText=data['values'],
                               colLabels=data['columns'],
                               rowLabels=data['rows'],
                               cellLoc='center',
                               loc='center')

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        return table

    def create_publication_subplot_labels(self, axes, labels=None):
        """Add (a), (b), (c) labels to subplots for publication."""
        if labels is None:
            labels = [f"({chr(97+i)})" for i in range(len(axes.flat))]

        for ax, label in zip(axes.flat, labels):
            ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='bottom', ha='right')
