#!/usr/bin/env python3
"""
Performance Visualizer

Specialized visualizer for model performance comparisons, metrics, and evaluation plots.
Inherits from BaseVisualizer for consistent styling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import json

from .base_visualizer import BaseVisualizer


class PerformanceVisualizer(BaseVisualizer):
    """
    Visualizer for model performance analysis and comparison.

    Creates publication-quality plots for:
    - Model accuracy comparisons
    - Confusion matrices
    - Performance heatmaps
    - Target-specific analysis
    """

    def __init__(self, results_dir='output/models', **kwargs):
        """
        Initialize performance visualizer.

        Args:
            results_dir: Directory containing model results
            **kwargs: Arguments passed to BaseVisualizer
        """
        super().__init__(**kwargs)
        self.results_dir = Path(results_dir)

    def load_performance_data(self):
        """Load all model performance data from results directory."""
        performance_data = {}

        # Scan for all model results
        for model_dir in self.results_dir.glob('*'):
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            performance_data[model_name] = {}

            # Load results for each target
            for target_dir in model_dir.glob('*'):
                if not target_dir.is_dir():
                    continue

                target_name = target_dir.name
                results_file = target_dir / 'results.json'

                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    performance_data[model_name][target_name] = results

        return performance_data

    def create_accuracy_comparison(self, performance_data, figsize=(14, 8)):
        """
        Create comprehensive accuracy comparison across all models and targets.

        Args:
            performance_data: Dictionary of model performance results
            figsize: Figure size tuple
        """
        # Extract accuracy data
        models = list(performance_data.keys())
        targets = list(next(iter(performance_data.values())).keys())

        accuracy_matrix = np.zeros((len(models), len(targets)))

        for i, model in enumerate(models):
            for j, target in enumerate(targets):
                if target in performance_data[model]:
                    accuracy_matrix[i,
                                    j] = performance_data[model][target]['test_accuracy']

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Subplot 1: Accuracy heatmap
        im = ax1.imshow(accuracy_matrix, cmap='RdYlGn',
                        aspect='auto', vmin=0.8, vmax=1.0)

        # Set ticks and labels
        ax1.set_xticks(range(len(targets)))
        ax1.set_yticks(range(len(models)))
        ax1.set_xticklabels([t.replace('_', ' ').title()
                            for t in targets], rotation=45, ha='right')
        ax1.set_yticklabels([m.replace('_', ' ').title() for m in models])

        # Add accuracy values as text
        for i in range(len(models)):
            for j in range(len(targets)):
                text = ax1.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                                ha="center", va="center", color="black", fontweight='bold')

        ax1.set_title('Model Accuracy Heatmap', fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Accuracy Score', rotation=270, labelpad=20)

        # Subplot 2: Average accuracy bar plot
        avg_accuracies = accuracy_matrix.mean(axis=1)
        colors = [self.get_model_color(model) for model in models]

        bars = ax2.bar(range(len(models)), avg_accuracies,
                       color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, avg_accuracies)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                     f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace('_', ' ').title()
                            for m in models], rotation=45, ha='right')
        ax2.set_ylabel('Average Accuracy')
        ax2.set_title('Average Model Performance',
                      fontsize=14, fontweight='bold')
        ax2.set_ylim(0.8, 1.0)

        # Add grid and styling
        self.add_grid(ax2)
        self.set_spine_style(ax1)
        self.set_spine_style(ax2)

        # Add subplot labels
        ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes,
                 fontsize=14, fontweight='bold')
        ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes,
                 fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def create_target_specific_comparison(self, performance_data, figsize=(10, 6)):
        """
        Create individual detailed comparison plots for each target showing all metrics.

        Args:
            performance_data: Dictionary of model performance results
            figsize: Figure size tuple for each individual plot
        """
        models = list(performance_data.keys())
        targets = list(next(iter(performance_data.values())).keys())

        saved_plots = {}
        metrics = ['test_accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        for target in targets:
            # Create individual figure for each target
            fig, ax = self.create_figure(figsize=figsize)

            # Collect data for this target
            target_data = {metric: [] for metric in metrics}

            for model in models:
                if target in performance_data[model]:
                    results = performance_data[model][target]
                    for metric in metrics:
                        target_data[metric].append(results.get(metric, 0))
                else:
                    for metric in metrics:
                        target_data[metric].append(0)

            # Create grouped bar chart
            x = np.arange(len(models))
            width = 0.2

            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                offset = (i - len(metrics)/2) * width + width/2
                bars = ax.bar(x + offset, target_data[metric], width,
                              label=label, alpha=0.8)

            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title(f'{target.replace("_", " ").title()} Performance Metrics',
                         fontweight='bold', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('_', ' ').title()
                               for m in models], rotation=45, ha='right')
            ax.legend(loc='lower right', fontsize=10)
            ax.set_ylim(0.8, 1.05)

            self.add_grid(ax)
            self.set_spine_style(ax)

            plt.tight_layout()

            # Save individual plot
            filename = f'{target}_performance_comparison'
            saved_files = self.save_figure(fig, filename)
            saved_plots[target] = saved_files

        return saved_plots

    def create_peecom_physics_insight(self, performance_data, figsize=(12, 8)):
        """
        Create visualization showing PEECOM's physics-enhanced performance.

        Args:
            performance_data: Dictionary of model performance results
            figsize: Figure size tuple
        """
        if 'peecom' not in performance_data:
            print("PEECOM results not found in performance data")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Extract PEECOM vs other models comparison
        models = [m for m in performance_data.keys() if m != 'peecom']
        targets = list(performance_data['peecom'].keys())

        peecom_scores = []
        best_other_scores = []
        improvements = []

        for target in targets:
            peecom_acc = performance_data['peecom'][target]['test_accuracy']
            peecom_scores.append(peecom_acc)

            # Find best other model for this target
            other_scores = []
            for model in models:
                if target in performance_data[model]:
                    other_scores.append(
                        performance_data[model][target]['test_accuracy'])

            best_other = max(other_scores) if other_scores else 0
            best_other_scores.append(best_other)
            improvements.append(peecom_acc - best_other)

        # Subplot 1: PEECOM vs Best Other Model
        x = np.arange(len(targets))
        width = 0.35

        bars1 = ax1.bar(x - width/2, peecom_scores, width, label='PEECOM',
                        color=self.get_model_color('peecom'), alpha=0.8)
        bars2 = ax1.bar(x + width/2, best_other_scores, width, label='Best Other Model',
                        color='#888888', alpha=0.8)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        ax1.set_xlabel('Target Variables')
        ax1.set_ylabel('Accuracy Score')
        ax1.set_title('PEECOM vs Best Competing Model', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([t.replace('_', ' ').title()
                            for t in targets], rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0.85, 1.02)

        # Subplot 2: Performance Improvement
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax2.bar(x, improvements, color=colors,
                       alpha=0.7, edgecolor='black')

        # Add value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            y_pos = height + 0.0005 if height >= 0 else height - 0.0005
            va = 'bottom' if height >= 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                     f'{imp:+.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Target Variables')
        ax2.set_ylabel('Accuracy Improvement')
        ax2.set_title('PEECOM Physics Enhancement Benefit', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([t.replace('_', ' ').title()
                            for t in targets], rotation=45, ha='right')

        # Add styling
        self.add_grid(ax1)
        self.add_grid(ax2)
        self.set_spine_style(ax1)
        self.set_spine_style(ax2)

        # Add subplot labels
        ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes,
                 fontsize=14, fontweight='bold')
        ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes,
                 fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def generate_all_performance_plots(self):
        """Generate all performance visualization plots."""
        print("Loading performance data...")
        performance_data = self.load_performance_data()

        if not performance_data:
            print("No performance data found!")
            return

        print(f"Found data for {len(performance_data)} models")

        # Generate plots
        plots = {}

        # 1. Overall accuracy comparison
        print("Creating accuracy comparison plot...")
        fig1 = self.create_accuracy_comparison(performance_data)
        plots['accuracy_comparison'] = self.save_figure(
            fig1, 'model_accuracy_comparison')

        # 2. Target-specific comparison (individual plots)
        print("Creating individual target-specific comparison plots...")
        target_plots = self.create_target_specific_comparison(performance_data)
        if target_plots:
            plots.update(target_plots)

        # 3. PEECOM physics insight
        print("Creating PEECOM physics insight plot...")
        fig3 = self.create_peecom_physics_insight(performance_data)
        if fig3:
            plots['peecom_insight'] = self.save_figure(
                fig3, 'peecom_physics_insight')

        print(f"Performance plots saved to: {self.output_dir}")
        return plots
