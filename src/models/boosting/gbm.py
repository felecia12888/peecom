#!/usr/bin/env python3
"""
Gradient Boosting Model for Condition Monitoring

Sequential ensemble method that builds trees iteratively to correct
errors from previous iterations.
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


class GradientBoosting:
    """
    Gradient Boosting Classifier wrapper for hydraulic system monitoring

    Provides a clean interface with hyperparameter tuning and feature
    importance capabilities.
    """

    def __init__(self, **kwargs):
        """
        Initialize Gradient Boosting model

        Parameters
        ----------
        **kwargs : dict
            Parameters to pass to GradientBoostingClassifier
        """
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }

        # Update defaults with any provided kwargs
        default_params.update(kwargs)
        self.params = default_params
        self.model = GradientBoostingClassifier(**self.params)
        self.name = "gradient_boosting"
        self.display_name = "Gradient Boosting"

    def get_model(self):
        """Return the sklearn model instance"""
        return self.model

    def get_params(self):
        """Return model parameters"""
        return self.params

    def get_param_grid(self):
        """Return parameter grid for hyperparameter tuning"""
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }

    def tune_hyperparameters(self, X_train, y_train, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
        cv : int, default=5
            Number of cross-validation folds

        Returns
        -------
        best_params : dict
            Best parameters found
        best_score : float
            Best cross-validation score
        """
        param_grid = self.get_param_grid()
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.params.update(grid_search.best_params_)

        return grid_search.best_params_, grid_search.best_score_

    def get_feature_importance(self):
        """Return feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def __str__(self):
        return f"{self.display_name} ({self.name})"


__all__ = ['GradientBoosting']
