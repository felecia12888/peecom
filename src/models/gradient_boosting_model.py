#!/usr/bin/env python3
"""
Gradient Boosting Model for PEECOM Hydraulic System Condition Monitoring
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


class GradientBoostingModel:
    """Gradient Boosting Classifier for hydraulic system condition monitoring"""

    def __init__(self, **kwargs):
        """Initialize Gradient Boosting model with default or custom parameters"""
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
        """Return feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def __str__(self):
        return f"{self.display_name} ({self.name})"
