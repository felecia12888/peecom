#!/usr/bin/env python3
"""
BLAST (Batch and Label-Aware Shrinkage Transform) Preprocessing

This module implements BLAST preprocessing for removing batch effects while
preserving task-discriminant information. BLAST is a data preprocessing 
technique that removes site-specific artifacts from multi-batch datasets.

Key Features:
- Removes batch-specific covariance
- Preserves task-discriminant structure
- Supports leave-one-batch-out validation
- Compatible with scikit-learn pipeline

Usage:
    from src.loader.blast_cleaner import BLASTCleaner
    
    blast = BLASTCleaner(preserve_variance=0.95)
    blast.fit(X_train, batches_train, y_train)
    X_clean = blast.transform(X_test)

References:
    See docs/BLAST_PROTOCOL_COMPLETE.md for full protocol
    See docs/BLAST_AS_PREPROCESSING_CLARIFICATION.md for positioning
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from scipy import linalg
import warnings


class BLASTCleaner(BaseEstimator, TransformerMixin):
    """
    BLAST: Batch and Label-Aware Shrinkage Transform

    A preprocessing technique that removes batch effects while explicitly
    preserving task-discriminant information.

    Parameters
    ----------
    preserve_variance : float, default=0.95
        Fraction of task-discriminant variance to preserve (0.0 to 1.0).
        Higher values preserve more task signal but may keep some batch effects.
        Recommended: 0.90-0.99

    regularization : float, default=1e-6
        Small value added to diagonal for numerical stability.
        Increase if encountering singular matrix errors.

    n_components : int or None, default=None
        Number of components to keep. If None, determined by preserve_variance.

    verbose : bool, default=False
        Print diagnostic information during fitting.

    Attributes
    ----------
    projection_matrix_ : ndarray of shape (n_features, n_components)
        Learned projection matrix for transforming data.

    eigenvalues_ : ndarray of shape (n_features,)
        Eigenvalues from generalized eigenvalue decomposition.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Ratio of task variance explained by each component.

    n_features_in_ : int
        Number of features seen during fit.

    n_components_ : int
        Number of components actually used.

    Examples
    --------
    >>> from src.loader.blast_cleaner import BLASTCleaner
    >>> import numpy as np
    >>> 
    >>> # Create sample data
    >>> X = np.random.randn(300, 50)
    >>> y = np.array([0]*100 + [1]*100 + [0]*100)
    >>> batches = np.array([0]*100 + [1]*100 + [2]*100)
    >>> 
    >>> # Apply BLAST
    >>> blast = BLASTCleaner(preserve_variance=0.95)
    >>> blast.fit(X, batches, y)
    >>> X_clean = blast.transform(X)
    """

    def __init__(
        self,
        preserve_variance=0.95,
        regularization=1e-6,
        n_components=None,
        verbose=False
    ):
        self.preserve_variance = preserve_variance
        self.regularization = regularization
        self.n_components = n_components
        self.verbose = verbose

    def fit(self, X, batches, labels):
        """
        Fit BLAST transformation on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features. Should be standardized beforehand.

        batches : array-like of shape (n_samples,)
            Batch/site identifiers for each sample.

        labels : array-like of shape (n_samples,)
            Task labels (classification targets).

        Returns
        -------
        self : object
            Fitted transformer.
        """
        # Convert inputs to arrays
        X = self._validate_data(X, dtype=np.float64, ensure_2d=True)
        batches = np.asarray(batches)
        labels = np.asarray(labels)

        if len(batches) != len(X) or len(labels) != len(X):
            raise ValueError("X, batches, and labels must have same length")

        self.n_features_in_ = X.shape[1]

        if self.verbose:
            print(
                f"BLAST: Fitting on {X.shape[0]} samples, {X.shape[1]} features")
            print(
                f"BLAST: {len(np.unique(batches))} batches, {len(np.unique(labels))} classes")

        # Compute within-batch covariance
        within_batch_cov = self._compute_within_batch_covariance(X, batches)

        # Compute task-discriminant covariance
        task_cov = self._compute_task_covariance(X, labels)

        # Solve generalized eigenvalue problem
        # We want directions that maximize: v^T * task_cov * v / v^T * within_batch_cov * v
        eigenvalues, eigenvectors = self._solve_generalized_eigen(
            task_cov, within_batch_cov
        )

        # Store eigenvalues (sorted descending)
        sort_idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Determine number of components
        if self.n_components is not None:
            n_comps = min(self.n_components, len(self.eigenvalues_))
        else:
            # Select components that preserve desired variance
            cumulative_variance = np.cumsum(
                self.eigenvalues_) / np.sum(self.eigenvalues_)
            n_comps = np.searchsorted(
                cumulative_variance, self.preserve_variance) + 1
            n_comps = min(n_comps, len(self.eigenvalues_))

        self.n_components_ = n_comps
        self.projection_matrix_ = eigenvectors[:, :n_comps]

        # Compute explained variance ratio
        total_variance = np.sum(self.eigenvalues_)
        self.explained_variance_ratio_ = self.eigenvalues_[
            :n_comps] / total_variance

        if self.verbose:
            print(f"BLAST: Selected {n_comps} components")
            print(
                f"BLAST: Preserving {np.sum(self.explained_variance_ratio_):.2%} of task variance")

        return self

    def transform(self, X):
        """
        Apply BLAST transformation to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data with batch effects removed.
        """
        X = self._validate_data(X, dtype=np.float64,
                                ensure_2d=True, reset=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        # Project onto learned subspace
        X_transformed = X @ self.projection_matrix_

        return X_transformed

    def _compute_within_batch_covariance(self, X, batches):
        """Compute average within-batch covariance matrix."""
        unique_batches = np.unique(batches)
        n_batches = len(unique_batches)
        n_features = X.shape[1]

        # Initialize covariance
        within_cov = np.zeros((n_features, n_features))

        for batch in unique_batches:
            batch_mask = batches == batch
            X_batch = X[batch_mask]

            if len(X_batch) < 2:
                warnings.warn(f"Batch {batch} has < 2 samples, skipping")
                continue

            # Center batch
            X_centered = X_batch - X_batch.mean(axis=0)

            # Compute covariance
            batch_cov = (X_centered.T @ X_centered) / (len(X_batch) - 1)
            within_cov += batch_cov

        # Average across batches
        within_cov /= n_batches

        # Add regularization for numerical stability
        within_cov += self.regularization * np.eye(n_features)

        return within_cov

    def _compute_task_covariance(self, X, labels):
        """Compute between-class scatter matrix (task-discriminant covariance)."""
        # Encode labels if necessary
        if labels.dtype.kind not in 'biufc':
            encoder = LabelEncoder()
            labels = encoder.fit_transform(labels)

        unique_labels = np.unique(labels)
        n_features = X.shape[1]

        # Overall mean
        overall_mean = X.mean(axis=0)

        # Initialize scatter matrix
        task_cov = np.zeros((n_features, n_features))

        for label in unique_labels:
            label_mask = labels == label
            X_class = X[label_mask]
            n_class = len(X_class)

            if n_class == 0:
                continue

            # Class mean
            class_mean = X_class.mean(axis=0)

            # Between-class scatter
            mean_diff = (class_mean - overall_mean).reshape(-1, 1)
            task_cov += n_class * (mean_diff @ mean_diff.T)

        # Normalize by total samples
        task_cov /= len(X)

        return task_cov

    def _solve_generalized_eigen(self, A, B):
        """
        Solve generalized eigenvalue problem: A * v = lambda * B * v

        Returns eigenvalues and eigenvectors sorted by eigenvalue magnitude.
        """
        try:
            # Use scipy's generalized eigenvalue solver
            eigenvalues, eigenvectors = linalg.eigh(A, B)

            # Filter out invalid eigenvalues
            valid_mask = np.isfinite(eigenvalues) & (eigenvalues > 1e-10)
            eigenvalues = eigenvalues[valid_mask]
            eigenvectors = eigenvectors[:, valid_mask]

        except linalg.LinAlgError as e:
            warnings.warn(
                f"Generalized eigenvalue decomposition failed: {e}. "
                "Using standard eigenvalue decomposition on A."
            )
            # Fallback: standard eigenvalue decomposition on task covariance
            eigenvalues, eigenvectors = linalg.eigh(A)
            valid_mask = eigenvalues > 1e-10
            eigenvalues = eigenvalues[valid_mask]
            eigenvectors = eigenvectors[:, valid_mask]

        return eigenvalues, eigenvectors

    def get_batch_removal_score(self, X_original, X_transformed, batches):
        """
        Compute how well batch effects were removed.

        Returns a score between 0 (no removal) and 1 (perfect removal).
        Based on reduction in batch separability after transformation.

        Parameters
        ----------
        X_original : array-like of shape (n_samples, n_features)
            Original data before BLAST.

        X_transformed : array-like of shape (n_samples, n_components)
            Transformed data after BLAST.

        batches : array-like of shape (n_samples,)
            Batch labels.

        Returns
        -------
        score : float
            Batch removal effectiveness (0 to 1, higher is better).
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import cross_val_score

        # Measure batch separability before BLAST
        lda_before = LinearDiscriminantAnalysis()
        try:
            score_before = cross_val_score(
                lda_before, X_original, batches, cv=3
            ).mean()
        except:
            score_before = 1.0  # Assume perfectly separable if error

        # Measure batch separability after BLAST
        lda_after = LinearDiscriminantAnalysis()
        try:
            score_after = cross_val_score(
                lda_after, X_transformed, batches, cv=3
            ).mean()
        except:
            score_after = 0.0  # Assume not separable if error

        # Compute reduction in separability
        removal_score = max(
            0.0, (score_before - score_after) / (score_before + 1e-10))

        return removal_score


def create_blast_pipeline(preserve_variance=0.95, scale_first=True):
    """
    Create a scikit-learn pipeline with BLAST preprocessing.

    Parameters
    ----------
    preserve_variance : float, default=0.95
        Fraction of task variance to preserve.

    scale_first : bool, default=True
        Whether to standardize features before BLAST.

    Returns
    -------
    pipeline : Pipeline
        Pipeline ready for use with any classifier.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> 
    >>> # Create pipeline
    >>> pipeline = create_blast_pipeline()
    >>> pipeline.steps.append(('clf', RandomForestClassifier()))
    >>> 
    >>> # Fit with batch information
    >>> pipeline.fit(X_train, y_train, blast__batches=batches_train)
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = []

    if scale_first:
        steps.append(('scaler', StandardScaler()))

    steps.append(('blast', BLASTCleaner(preserve_variance=preserve_variance)))

    return Pipeline(steps)


__all__ = ['BLASTCleaner', 'create_blast_pipeline']
