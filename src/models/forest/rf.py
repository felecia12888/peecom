#!/usr/bin/env python3
"""
Random Forest Model for Equipment Condition Monitoring

Wraps scikit-learn's RandomForestClassifier with sensible defaults
for condition monitoring tasks.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForest:
    """Random Forest Classifier for condition monitoring"""

    def __init__(self, **kwargs):
        """
        Initialize Random Forest model.

        Parameters
        ----------
        **kwargs : dict
            Parameters passed to RandomForestClassifier.
            Default parameters optimized for condition monitoring.
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }

        # Update defaults with any provided kwargs
        default_params.update(kwargs)
        self.params = default_params
        self.model = RandomForestClassifier(**self.params)
        self.name = "random_forest"
        self.display_name = "Random Forest"
        self.feature_names_ = None

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
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    def tune_hyperparameters(self, X_train, y_train, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV.

        Parameters
        ----------
        X_train : array-like
            Training features.

        y_train : array-like
            Training labels.

        cv : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        best_params : dict
            Best hyperparameters found.

        best_score : float
            Best cross-validation score.
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

    def get_feature_names(self):
        """Return feature names if available"""
        return self.feature_names_

    def __str__(self):
        return f"{self.display_name} ({self.name})"


__all__ = ['RandomForest']
