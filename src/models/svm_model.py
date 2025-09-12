#!/usr/bin/env python3
"""
Support Vector Machine Model for PEECOM Hydraulic System Condition Monitoring
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np


class SVMModel:
    """Support Vector Machine Classifier for hydraulic system condition monitoring"""

    def __init__(self, **kwargs):
        """Initialize SVM model with default or custom parameters"""
        # Extract dataset size hint if provided
        dataset_size = kwargs.pop('dataset_size', None)

        # Optimize parameters based on dataset size
        default_params = self._get_optimized_params(dataset_size)

        # Update defaults with any provided kwargs
        default_params.update(kwargs)
        self.params = default_params
        self.model = SVC(**self.params)
        self.name = "svm"
        self.display_name = "Support Vector Machine"
        self.dataset_size = dataset_size

    def _get_optimized_params(self, dataset_size):
        """Get optimized parameters based on dataset size"""
        if dataset_size is None:
            dataset_size = 1000  # Default assumption

        if dataset_size > 10000:
            # Large dataset: Use linear kernel for speed
            return {
                'kernel': 'linear',
                'C': 1.0,
                'random_state': 42,
                'probability': True,
                'max_iter': 1000  # Limit iterations
            }
        elif dataset_size > 5000:
            # Medium dataset: Use RBF with limited iterations
            return {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'random_state': 42,
                'probability': True,
                'max_iter': 2000
            }
        else:
            # Small dataset: Full RBF kernel
            return {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'random_state': 42,
                'probability': True
            }

    def get_model(self):
        """Return the sklearn model instance"""
        return self.model

    def get_params(self):
        """Return model parameters"""
        return self.params

    def get_param_grid(self):
        """Return parameter grid for hyperparameter tuning (optimized for dataset size)"""
        if self.dataset_size and self.dataset_size > 10000:
            # Large dataset: Only linear kernel
            return {
                'C': [0.1, 1, 10],
                'kernel': ['linear']
            }
        elif self.dataset_size and self.dataset_size > 5000:
            # Medium dataset: Limited grid
            return {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        else:
            # Small dataset: Full grid
            return {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }

    def tune_hyperparameters(self, X_train, y_train, cv=5):
        """Perform hyperparameter tuning using GridSearchCV (with timeout protection)"""
        param_grid = self.get_param_grid()

        # Reduce CV folds for large datasets
        if len(X_train) > 10000:
            cv = min(cv, 3)

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        try:
            grid_search.fit(X_train, y_train)
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            self.params.update(grid_search.best_params_)
            return grid_search.best_params_, grid_search.best_score_
        except Exception as e:
            print(f"⚠️  Hyperparameter tuning failed: {e}")
            print("⚠️  Using default parameters")
            return self.params, None

    def get_feature_importance(self):
        """SVM doesn't have built-in feature importance, return None"""
        return None

    def __str__(self):
        return f"{self.display_name} ({self.name})"
