"""
PEECOM Evaluation Metrics

Comprehensive metrics computation for model evaluation including
classification metrics, regression metrics, and custom domain-specific metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, Any, Optional, Union
import pandas as pd


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for ROC-AUC)
        average: Averaging strategy for multi-class ('weighted', 'macro', 'micro')

    Returns:
        Dictionary containing all computed metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # Add ROC-AUC if probabilities provided
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multi-class
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_prob,
                    multi_class='ovr',
                    average=average
                )
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")
            metrics['roc_auc'] = None

    return metrics


def compute_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None
) -> Dict[str, Any]:
    """
    Compute confusion matrix and derived metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names (optional)

    Returns:
        Dictionary with confusion matrix and per-class metrics
    """
    cm = confusion_matrix(y_true, y_pred)

    result = {
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    }

    # Per-class metrics
    if labels is not None:
        result['labels'] = labels

    return result


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary containing regression metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }

    # Add MAPE if no zeros in y_true
    if not np.any(y_true == 0):
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return metrics


def compute_cross_validation_metrics(
    cv_scores: np.ndarray,
    metric_name: str = 'accuracy'
) -> Dict[str, float]:
    """
    Compute statistics from cross-validation scores.

    Args:
        cv_scores: Array of cross-validation scores
        metric_name: Name of the metric

    Returns:
        Dictionary with CV statistics
    """
    return {
        f'cv_{metric_name}_mean': np.mean(cv_scores),
        f'cv_{metric_name}_std': np.std(cv_scores),
        f'cv_{metric_name}_min': np.min(cv_scores),
        f'cv_{metric_name}_max': np.max(cv_scores),
        f'cv_{metric_name}_median': np.median(cv_scores),
    }


def compute_condition_monitoring_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    condition_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Compute domain-specific metrics for condition monitoring.

    Args:
        y_true: True condition labels
        y_pred: Predicted condition labels
        condition_names: Names of conditions (optional)

    Returns:
        Dictionary with condition monitoring metrics
    """
    # Standard classification metrics
    metrics = compute_classification_metrics(y_true, y_pred)

    # Add confusion matrix
    cm_metrics = compute_confusion_matrix_metrics(
        y_true, y_pred, condition_names)
    metrics.update(cm_metrics)

    # Critical failure detection rate (if condition 0 is critical)
    if 0 in y_true:
        critical_mask = y_true == 0
        metrics['critical_detection_rate'] = accuracy_score(
            y_true[critical_mask],
            y_pred[critical_mask]
        )

    return metrics


def compare_model_metrics(
    models_results: Dict[str, Dict[str, float]],
    primary_metric: str = 'accuracy'
) -> pd.DataFrame:
    """
    Compare metrics across multiple models.

    Args:
        models_results: Dictionary mapping model names to their metrics
        primary_metric: Primary metric for ranking

    Returns:
        DataFrame with model comparison
    """
    df = pd.DataFrame(models_results).T

    # Sort by primary metric
    if primary_metric in df.columns:
        df = df.sort_values(primary_metric, ascending=False)

    return df


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    cv_scores: Optional[np.ndarray] = None,
    task_type: str = 'classification'
) -> Dict[str, Any]:
    """
    Compute all relevant metrics based on task type.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        y_prob: Predicted probabilities (for classification)
        cv_scores: Cross-validation scores (optional)
        task_type: 'classification' or 'regression'

    Returns:
        Comprehensive metrics dictionary
    """
    metrics = {}

    if task_type == 'classification':
        metrics.update(compute_classification_metrics(y_true, y_pred, y_prob))
        metrics.update(compute_confusion_matrix_metrics(y_true, y_pred))
    else:
        metrics.update(compute_regression_metrics(y_true, y_pred))

    # Add CV metrics if provided
    if cv_scores is not None:
        metrics.update(compute_cross_validation_metrics(cv_scores))

    return metrics


__all__ = [
    'compute_classification_metrics',
    'compute_confusion_matrix_metrics',
    'compute_regression_metrics',
    'compute_cross_validation_metrics',
    'compute_condition_monitoring_metrics',
    'compare_model_metrics',
    'compute_all_metrics',
]
