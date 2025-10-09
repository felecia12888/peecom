#!/usr/bin/env python3
"""
Outlier Detection and Removal

This module provides statistical methods for detecting and removing outliers
from sensor data and feature matrices. Multiple detection methods are supported
with configurable thresholds.

Methods Supported:
- IQR (Interquartile Range): Robust to non-normal distributions
- Z-Score: Assumes normal distribution
- Isolation Forest: Machine learning-based detection
- Modified Z-Score (MAD): Median-based robust method

Usage:
    from src.loader.outlier_remover import OutlierRemover
    
    remover = OutlierRemover(method='iqr', threshold=1.5)
    X_clean, mask = remover.fit_transform(X, return_mask=True)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Detect and remove outliers from feature matrices.

    Parameters
    ----------
    method : str, default='iqr'
        Outlier detection method:
        - 'iqr': Interquartile Range (recommended for skewed data)
        - 'zscore': Standard Z-score (assumes normality)
        - 'modified_zscore': MAD-based Z-score (robust)
        - 'isolation_forest': ML-based anomaly detection

    threshold : float, default=1.5
        Threshold for outlier detection:
        - IQR: multiplier for IQR (1.5 = mild, 3.0 = extreme)
        - Z-score: number of standard deviations (3.0 typical)
        - Modified Z-score: number of MADs (3.5 typical)
        - Isolation Forest: contamination fraction (0.1 = 10%)

    column_wise : bool, default=True
        Apply detection column-wise (per feature) vs row-wise.

    removal_strategy : str, default='mask'
        How to handle detected outliers:
        - 'mask': Return mask of clean samples (removes rows)
        - 'clip': Clip outlier values to threshold bounds
        - 'nan': Replace outliers with NaN

    verbose : bool, default=False
        Print diagnostic information.

    Attributes
    ----------
    outlier_mask_ : ndarray of shape (n_samples,)
        Boolean mask indicating outlier samples (True = outlier).

    bounds_ : dict
        Lower and upper bounds for each feature (if column_wise).

    n_outliers_ : int
        Total number of outlier samples detected.

    Examples
    --------
    >>> from src.loader.outlier_remover import OutlierRemover
    >>> import numpy as np
    >>> 
    >>> X = np.random.randn(100, 10)
    >>> X[0, :] = 100  # Add outlier
    >>> 
    >>> remover = OutlierRemover(method='iqr', threshold=1.5)
    >>> X_clean = remover.fit_transform(X)
    >>> print(f"Removed {remover.n_outliers_} outliers")
    """

    def __init__(
        self,
        method='iqr',
        threshold=1.5,
        column_wise=True,
        removal_strategy='mask',
        verbose=False
    ):
        self.method = method
        self.threshold = threshold
        self.column_wise = column_wise
        self.removal_strategy = removal_strategy
        self.verbose = verbose

        # Validate parameters
        valid_methods = ['iqr', 'zscore',
                         'modified_zscore', 'isolation_forest']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        valid_strategies = ['mask', 'clip', 'nan']
        if removal_strategy not in valid_strategies:
            raise ValueError(
                f"removal_strategy must be one of {valid_strategies}")

    def fit(self, X, y=None):
        """
        Fit outlier detector on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = self._validate_data(X, dtype=np.float64, ensure_2d=True)

        if self.method == 'iqr':
            self.bounds_ = self._compute_iqr_bounds(X)
        elif self.method == 'zscore':
            self.bounds_ = self._compute_zscore_bounds(X)
        elif self.method == 'modified_zscore':
            self.bounds_ = self._compute_modified_zscore_bounds(X)
        elif self.method == 'isolation_forest':
            self._fit_isolation_forest(X)

        return self

    def transform(self, X):
        """
        Transform data by removing or clipping outliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : ndarray
            Transformed data with outliers handled.
        """
        X = self._validate_data(X, dtype=np.float64,
                                ensure_2d=True, reset=False)

        # Detect outliers
        if self.method == 'isolation_forest':
            outlier_mask = self._detect_with_isolation_forest(X)
        else:
            outlier_mask = self._detect_with_bounds(X)

        self.outlier_mask_ = outlier_mask
        self.n_outliers_ = np.sum(outlier_mask)

        if self.verbose:
            pct = 100 * self.n_outliers_ / len(X)
            print(
                f"OutlierRemover: Detected {self.n_outliers_} outliers ({pct:.1f}%)")

        # Apply removal strategy
        if self.removal_strategy == 'mask':
            X_transformed = X[~outlier_mask]
        elif self.removal_strategy == 'clip':
            X_transformed = self._clip_outliers(X)
        elif self.removal_strategy == 'nan':
            X_transformed = X.copy()
            X_transformed[outlier_mask] = np.nan

        return X_transformed

    def fit_transform(self, X, y=None, return_mask=False):
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to fit and transform.

        y : Ignored
            Not used, present for API consistency.

        return_mask : bool, default=False
            If True, return (X_transformed, outlier_mask).

        Returns
        -------
        X_transformed : ndarray
            Transformed data.

        outlier_mask : ndarray, optional
            Boolean mask of outliers (if return_mask=True).
        """
        self.fit(X, y)
        X_transformed = self.transform(X)

        if return_mask:
            return X_transformed, self.outlier_mask_
        return X_transformed

    def _compute_iqr_bounds(self, X):
        """Compute IQR-based bounds for each feature."""
        bounds = {}

        if self.column_wise:
            for col_idx in range(X.shape[1]):
                col_data = X[:, col_idx]
                q25, q75 = np.percentile(col_data, [25, 75])
                iqr = q75 - q25

                lower = q25 - self.threshold * iqr
                upper = q75 + self.threshold * iqr

                bounds[col_idx] = (lower, upper)
        else:
            # Global bounds across all features
            q25, q75 = np.percentile(X.flatten(), [25, 75])
            iqr = q75 - q25
            lower = q25 - self.threshold * iqr
            upper = q75 + self.threshold * iqr
            bounds['global'] = (lower, upper)

        return bounds

    def _compute_zscore_bounds(self, X):
        """Compute Z-score based bounds for each feature."""
        bounds = {}

        if self.column_wise:
            for col_idx in range(X.shape[1]):
                col_data = X[:, col_idx]
                mean = np.mean(col_data)
                std = np.std(col_data, ddof=1)

                lower = mean - self.threshold * std
                upper = mean + self.threshold * std

                bounds[col_idx] = (lower, upper)
        else:
            mean = np.mean(X)
            std = np.std(X, ddof=1)
            lower = mean - self.threshold * std
            upper = mean + self.threshold * std
            bounds['global'] = (lower, upper)

        return bounds

    def _compute_modified_zscore_bounds(self, X):
        """Compute Modified Z-score (MAD-based) bounds."""
        bounds = {}

        if self.column_wise:
            for col_idx in range(X.shape[1]):
                col_data = X[:, col_idx]
                median = np.median(col_data)
                mad = np.median(np.abs(col_data - median))

                # Modified Z-score
                if mad == 0:
                    warnings.warn(
                        f"MAD is zero for feature {col_idx}, using fallback")
                    mad = 1e-10

                # Typical threshold for modified Z-score is 3.5
                lower = median - self.threshold * mad
                upper = median + self.threshold * mad

                bounds[col_idx] = (lower, upper)
        else:
            median = np.median(X)
            mad = np.median(np.abs(X - median))
            if mad == 0:
                mad = 1e-10
            lower = median - self.threshold * mad
            upper = median + self.threshold * mad
            bounds['global'] = (lower, upper)

        return bounds

    def _fit_isolation_forest(self, X):
        """Fit Isolation Forest for outlier detection."""
        self.isolation_forest_ = IsolationForest(
            contamination=self.threshold,
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest_.fit(X)

    def _detect_with_bounds(self, X):
        """Detect outliers using precomputed bounds."""
        outlier_mask = np.zeros(len(X), dtype=bool)

        if self.column_wise:
            for col_idx, (lower, upper) in self.bounds_.items():
                col_data = X[:, col_idx]
                col_outliers = (col_data < lower) | (col_data > upper)
                outlier_mask |= col_outliers
        else:
            lower, upper = self.bounds_['global']
            outlier_mask = np.any((X < lower) | (X > upper), axis=1)

        return outlier_mask

    def _detect_with_isolation_forest(self, X):
        """Detect outliers using Isolation Forest."""
        predictions = self.isolation_forest_.predict(X)
        outlier_mask = predictions == -1  # -1 indicates outlier
        return outlier_mask

    def _clip_outliers(self, X):
        """Clip outlier values to bounds."""
        X_clipped = X.copy()

        if self.column_wise:
            for col_idx, (lower, upper) in self.bounds_.items():
                X_clipped[:, col_idx] = np.clip(
                    X_clipped[:, col_idx], lower, upper
                )
        else:
            lower, upper = self.bounds_['global']
            X_clipped = np.clip(X_clipped, lower, upper)

        return X_clipped


def detect_outliers_summary(X, methods=None, thresholds=None):
    """
    Compare multiple outlier detection methods on dataset.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to analyze.

    methods : list of str, optional
        Methods to compare. Default: ['iqr', 'zscore', 'modified_zscore']

    thresholds : dict, optional
        Custom thresholds per method. Default uses recommended values.

    Returns
    -------
    summary : pd.DataFrame
        Summary of outliers detected by each method.

    Examples
    --------
    >>> summary = detect_outliers_summary(X)
    >>> print(summary)
    """
    if methods is None:
        methods = ['iqr', 'zscore', 'modified_zscore']

    if thresholds is None:
        thresholds = {
            'iqr': 1.5,
            'zscore': 3.0,
            'modified_zscore': 3.5,
            'isolation_forest': 0.1
        }

    results = []

    for method in methods:
        threshold = thresholds.get(method, 1.5)

        remover = OutlierRemover(
            method=method,
            threshold=threshold,
            removal_strategy='mask'
        )

        try:
            remover.fit(X)
            remover.transform(X)

            n_outliers = remover.n_outliers_
            pct_outliers = 100 * n_outliers / len(X)

            results.append({
                'Method': method,
                'Threshold': threshold,
                'Outliers_Detected': n_outliers,
                'Percentage': f"{pct_outliers:.2f}%",
                'Samples_Remaining': len(X) - n_outliers
            })
        except Exception as e:
            results.append({
                'Method': method,
                'Threshold': threshold,
                'Outliers_Detected': 'Error',
                'Percentage': str(e),
                'Samples_Remaining': '-'
            })

    return pd.DataFrame(results)


__all__ = ['OutlierRemover', 'detect_outliers_summary']
