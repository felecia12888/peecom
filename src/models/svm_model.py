#!/usr/bin/env python3
"""
Support Vector Machine Model for PEECOM Hydraulic System Condition Monitoring
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class SVMModel:
    """Support Vector Machine Classifier for hydraulic system condition monitoring"""

    def __init__(self, **kwargs):
        """Initialize SVM model with default or custom parameters"""
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'random_state': 42,
            'probability': True  # Enable probability estimates
        }

        # Update defaults with any provided kwargs
        default_params.update(kwargs)
        self.params = default_params
        self.model = SVC(**self.params)
        self.name = "svm"
        self.display_name = "Support Vector Machine"

    def get_model(self):
        """Return the sklearn model instance"""
        return self.model

    def get_params(self):
        """Return model parameters"""
        return self.params

    def get_param_grid(self):
        """Return parameter grid for hyperparameter tuning"""
        return {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }

    def tune_hyperparameters(self, X_train, y_train, cv=5):
        """Perform hyperparameter tuning using GridSearchCV"""
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
        """SVM doesn't have built-in feature importance, return None"""
        return None

    def __str__(self):
        return f"{self.display_name} ({self.name})"
