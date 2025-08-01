#!/usr/bin/env python3
"""
Logistic Regression Model for PEECOM Hydraulic System Condition Monitoring
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class LogisticRegressionModel:
    """Logistic Regression Classifier for hydraulic system condition monitoring"""

    def __init__(self, **kwargs):
        """Initialize Logistic Regression model with default or custom parameters"""
        default_params = {
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'liblinear',
            'multi_class': 'ovr'
        }

        # Update defaults with any provided kwargs
        default_params.update(kwargs)
        self.params = default_params
        self.model = LogisticRegression(**self.params)
        self.name = "logistic_regression"
        self.display_name = "Logistic Regression"

    def get_model(self):
        """Return the sklearn model instance"""
        return self.model

    def get_params(self):
        """Return model parameters"""
        return self.params

    def get_param_grid(self):
        """Return parameter grid for hyperparameter tuning"""
        return {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs', 'newton-cg'],
            'penalty': ['l1', 'l2', 'elasticnet', 'none']
        }

    def tune_hyperparameters(self, X_train, y_train, cv=5):
        """Perform hyperparameter tuning using GridSearchCV"""
        param_grid = self.get_param_grid()

        # Adjust param grid based on solver compatibility
        adjusted_grid = []
        for C in param_grid['C']:
            for solver in param_grid['solver']:
                for penalty in param_grid['penalty']:
                    # Check solver-penalty compatibility
                    if self._is_valid_combination(solver, penalty):
                        adjusted_grid.append({
                            'C': C,
                            'solver': solver,
                            'penalty': penalty
                        })

        grid_search = GridSearchCV(
            self.model,
            adjusted_grid,
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

    def _is_valid_combination(self, solver, penalty):
        """Check if solver-penalty combination is valid"""
        valid_combinations = {
            'liblinear': ['l1', 'l2'],
            'lbfgs': ['l2', 'none'],
            'newton-cg': ['l2', 'none'],
            'sag': ['l2', 'none'],
            'saga': ['l1', 'l2', 'elasticnet', 'none']
        }
        return penalty in valid_combinations.get(solver, [])

    def get_feature_importance(self):
        """Return feature coefficients as importance"""
        if hasattr(self.model, 'coef_'):
            # For binary classification, coef_ shape is (1, n_features)
            # For multiclass, coef_ shape is (n_classes, n_features)
            coef = self.model.coef_
            if coef.ndim == 2 and coef.shape[0] == 1:
                return abs(coef[0])  # Binary classification
            elif coef.ndim == 2:
                # Multiclass - average absolute coefficients
                return abs(coef).mean(axis=0)
            else:
                return abs(coef)  # Should not happen, but just in case
        return None

    def __str__(self):
        return f"{self.display_name} ({self.name})"
