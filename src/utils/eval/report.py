"""
PEECOM Performance Report Generation

Tools for generating comprehensive performance reports, summaries, and comparisons.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
from .metrics import compute_all_metrics, compare_model_metrics


class PerformanceReport:
    """Generate comprehensive performance reports for model evaluation."""

    def __init__(self, output_dir: str = 'output/reports'):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_model_report(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive report for a single model.

        Args:
            model_name: Name of the model
            metrics: Dictionary of computed metrics
            dataset_info: Information about the dataset
            save_path: Custom save path (optional)

        Returns:
            Path to the generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if save_path is None:
            save_path = self.output_dir / \
                f"{model_name}_report_{timestamp}.txt"

        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"PEECOM Model Performance Report\n")
            f.write("="*80 + "\n\n")

            f.write(f"Model: {model_name}\n")
            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Dataset information
            if dataset_info:
                f.write("-"*80 + "\n")
                f.write("Dataset Information\n")
                f.write("-"*80 + "\n")
                for key, value in dataset_info.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

            # Performance metrics
            f.write("-"*80 + "\n")
            f.write("Performance Metrics\n")
            f.write("-"*80 + "\n")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    f.write(f"{metric_name:30s}: {metric_value:.4f}\n")
                elif isinstance(metric_value, np.ndarray):
                    f.write(f"{metric_name}:\n{metric_value}\n\n")

            f.write("\n")
            f.write("="*80 + "\n")

        print(f"Report saved to: {save_path}")
        return str(save_path)

    def generate_comparison_report(
        self,
        models_metrics: Dict[str, Dict[str, Any]],
        primary_metric: str = 'accuracy',
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a comparison report for multiple models.

        Args:
            models_metrics: Dictionary mapping model names to their metrics
            primary_metric: Primary metric for ranking
            save_path: Custom save path (optional)

        Returns:
            Path to the generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if save_path is None:
            save_path = self.output_dir / f"model_comparison_{timestamp}.txt"

        # Create comparison DataFrame
        df = compare_model_metrics(models_metrics, primary_metric)

        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PEECOM Models Comparison Report\n")
            f.write("="*80 + "\n\n")

            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of models compared: {len(models_metrics)}\n")
            f.write(f"Primary metric: {primary_metric}\n\n")

            f.write("-"*80 + "\n")
            f.write("Model Rankings\n")
            f.write("-"*80 + "\n\n")

            f.write(df.to_string())
            f.write("\n\n")

            # Best model summary
            best_model = df.index[0]
            f.write("-"*80 + "\n")
            f.write(f"Best Model: {best_model}\n")
            f.write("-"*80 + "\n")
            f.write(
                f"{primary_metric}: {df.loc[best_model, primary_metric]:.4f}\n\n")

            f.write("="*80 + "\n")

        print(f"Comparison report saved to: {save_path}")
        return str(save_path)

    def save_metrics_json(
        self,
        metrics: Dict[str, Any],
        model_name: str,
        save_path: Optional[str] = None
    ) -> str:
        """
        Save metrics as JSON for programmatic access.

        Args:
            metrics: Metrics dictionary
            model_name: Name of the model
            save_path: Custom save path (optional)

        Returns:
            Path to saved JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if save_path is None:
            save_path = self.output_dir / \
                f"{model_name}_metrics_{timestamp}.json"

        # Convert numpy types to native Python types for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, (np.int64, np.int32)):
                serializable_metrics[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value

        with open(save_path, 'w') as f:
            json.dump({
                'model': model_name,
                'timestamp': timestamp,
                'metrics': serializable_metrics
            }, f, indent=2)

        print(f"Metrics JSON saved to: {save_path}")
        return str(save_path)

    def save_comparison_csv(
        self,
        models_metrics: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> str:
        """
        Save model comparison as CSV.

        Args:
            models_metrics: Dictionary mapping model names to metrics
            save_path: Custom save path (optional)

        Returns:
            Path to saved CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if save_path is None:
            save_path = self.output_dir / f"model_comparison_{timestamp}.csv"

        df = compare_model_metrics(models_metrics)
        df.to_csv(save_path)

        print(f"Comparison CSV saved to: {save_path}")
        return str(save_path)


def generate_summary_statistics(
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate summary statistics from results.

    Args:
        results: Results dictionary

    Returns:
        Summary statistics
    """
    summary = {
        'total_samples': results.get('total_samples', 0),
        'train_samples': results.get('train_samples', 0),
        'test_samples': results.get('test_samples', 0),
        'n_features': results.get('n_features', 0),
        'n_classes': results.get('n_classes', 0),
    }

    return summary


def print_performance_summary(
    model_name: str,
    metrics: Dict[str, float],
    detailed: bool = False
):
    """
    Print a formatted performance summary to console.

    Args:
        model_name: Name of the model
        metrics: Metrics dictionary
        detailed: Whether to print detailed metrics
    """
    print("\n" + "="*60)
    print(f"Performance Summary: {model_name}")
    print("="*60)

    # Key metrics
    key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    for metric in key_metrics:
        if metric in metrics:
            value = metrics[metric]
            if value is not None:
                print(f"{metric:20s}: {value:.4f}")

    if detailed:
        print("\nDetailed Metrics:")
        print("-"*60)
        for key, value in metrics.items():
            if key not in key_metrics and isinstance(value, (int, float)):
                print(f"{key:30s}: {value:.4f}")

    print("="*60 + "\n")


__all__ = [
    'PerformanceReport',
    'generate_summary_statistics',
    'print_performance_summary',
]
