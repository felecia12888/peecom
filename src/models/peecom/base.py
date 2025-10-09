#!/usr/bin/env python3
"""
PEECOM Base Model: Physics-Enhanced Equipment Condition Monitoring

This module implements the core PEECOM model with:
- Physics-inspired feature engineering
- Hyperparameter optimization
- Feature selection
- Random Forest classifier with balanced classes

PEECOM enhances raw sensor features with physics-based derived features
such as power, efficiency ratios, and statistical aggregations to improve
condition monitoring performance.

Usage:
    from src.models.peecom.base import PEECOM
    
    model = PEECOM()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class PEECOM(BaseEstimator, ClassifierMixin):
    """
    PEECOM: Physics-Enhanced Equipment Condition Monitoring

    A machine learning model that enhances sensor features with physics-inspired
    derived features for improved condition monitoring performance.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees in the random forest.

    max_depth : int or None, default=None
        Maximum depth of trees. None means unlimited.

    random_state : int, default=42
        Random seed for reproducibility.

    max_physics_features : int, default=120
        Maximum number of physics features to keep after selection.
        Set to None to keep all generated features.

    hyperparameter_search : bool, default=True
        Whether to perform randomized hyperparameter search.

    search_iterations : int, default=10
        Number of hyperparameter combinations to try.

    cv_folds : int, default=3
        Number of cross-validation folds for hyperparameter search.

    Attributes
    ----------
    model : RandomForestClassifier
        The underlying trained classifier.

    scaler : StandardScaler
        Fitted scaler for feature normalization.

    feature_selector : SelectKBest or None
        Fitted feature selector (if feature selection used).

    feature_names_ : list
        Names of selected features after engineering and selection.

    best_params_ : dict or None
        Best hyperparameters found (if hyperparameter_search=True).

    original_features : int
        Number of original input features.

    Examples
    --------
    >>> from src.models.peecom.base import PEECOM
    >>> import numpy as np
    >>> 
    >>> X = np.random.randn(100, 20)
    >>> y = np.random.randint(0, 3, 100)
    >>> 
    >>> model = PEECOM(n_estimators=100)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    """

    def __init__(
        self,
        n_estimators=200,
        max_depth=None,
        random_state=42,
        max_physics_features=120,
        hyperparameter_search=True,
        search_iterations=10,
        cv_folds=3,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_physics_features = max_physics_features
        self.hyperparameter_search = hyperparameter_search
        self.search_iterations = search_iterations
        self.cv_folds = cv_folds

        # Initialize attributes
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.original_features = None
        self.selected_feature_names_ = None
        self.best_params_ = None

    def _create_physics_features(self, X):
        """
        Create physics-inspired features from raw sensor data.

        Features include:
        - Power features (multiplication of sensor pairs)
        - Efficiency ratios (division of sensor pairs)
        - Statistical aggregations (mean, std, max, min)
        - Derived ratios (max/min, std/mean)

        Parameters
        ----------
        X : array-like
            Raw sensor features.

        Returns
        -------
        features : pd.DataFrame
            Enhanced feature set with physics-derived features.
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(
                X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Start with original features
        features = X.copy()

        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return features

        try:
            # Power and efficiency features (limited to avoid combinatorial explosion)
            for i, col1 in enumerate(numeric_cols[:5]):
                for j, col2 in enumerate(numeric_cols[i+1:6], i+1):
                    if col1 != col2:
                        # Power feature (multiplication - represents energy)
                        power_name = f'power_{col1}_{col2}'
                        features[power_name] = X[col1] * X[col2]

                        # Efficiency ratio (division - represents relative performance)
                        ratio_name = f'ratio_{col1}_{col2}'
                        features[ratio_name] = X[col1] / (X[col2] + 1e-8)

            # Statistical aggregations across all sensors
            features['mean_all'] = X[numeric_cols].mean(axis=1)
            features['std_all'] = X[numeric_cols].std(axis=1)
            features['max_all'] = X[numeric_cols].max(axis=1)
            features['min_all'] = X[numeric_cols].min(axis=1)

            # Physics-inspired ratios
            features['max_min_ratio'] = features['max_all'] / \
                (features['min_all'] + 1e-8)
            features['std_mean_ratio'] = features['std_all'] / \
                (features['mean_all'] + 1e-8)

        except Exception as e:
            warnings.warn(f"Physics feature creation simplified due to: {e}")

        # Clean up invalid values
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)

        return features

    def fit(self, X, y):
        """
        Fit the PEECOM model.

        Steps:
        1. Engineer physics features
        2. Select top features (if max_physics_features specified)
        3. Scale features
        4. Optimize hyperparameters (if hyperparameter_search=True)
        5. Train final model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted model.
        """
        print("Training PEECOM model...")

        # Store original feature count
        self.original_features = X.shape[1] if hasattr(
            X, 'shape') else len(X[0])

        # Create physics features
        X_enhanced = self._create_physics_features(X)

        # Feature selection (if specified)
        if (
            self.max_physics_features is not None
            and hasattr(X_enhanced, 'shape')
            and X_enhanced.shape[1] > self.max_physics_features
        ):
            selector = SelectKBest(
                score_func=mutual_info_classif,
                k=self.max_physics_features
            )
            X_selected = selector.fit_transform(X_enhanced.values, y)
            feature_indices = selector.get_support(indices=True)
            self.selected_feature_names_ = [
                X_enhanced.columns[i] for i in feature_indices
            ]
            self.feature_selector = selector
        else:
            X_selected = X_enhanced.values if hasattr(
                X_enhanced, 'values') else X_enhanced
            self.selected_feature_names_ = list(X_enhanced.columns) if hasattr(
                X_enhanced, 'columns') else [f'f_{i}' for i in range(X_enhanced.shape[1])]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_selected)

        # Base Random Forest configuration
        base_rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced_subsample'
        )

        # Hyperparameter optimization
        if self.hyperparameter_search:
            param_distributions = {
                'n_estimators': [200, 300, 400],
                'max_depth': [12, 15, 18, None],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['sqrt', 0.8],
                'class_weight': ['balanced', 'balanced_subsample', None],
            }

            search = RandomizedSearchCV(
                estimator=base_rf,
                param_distributions=param_distributions,
                n_iter=self.search_iterations,
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0,
            )

            search.fit(X_scaled, y)
            self.model = search.best_estimator_
            self.best_params_ = search.best_params_

            print("   Hyperparameter optimization complete:")
            for key, value in sorted(self.best_params_.items()):
                print(f"      {key}: {value}")
        else:
            self.model = base_rf
            self.model.fit(X_scaled, y)

        # Ensure model is fitted on full data
        if self.model is not base_rf:
            self.model.fit(X_scaled, y)

        # Store final feature names
        self.feature_names_ = self.selected_feature_names_

        print(f"✅ PEECOM trained successfully!")
        print(
            f"   Features: {self.original_features} → {len(self.feature_names_)} (physics-enhanced)")

        return self

    def predict(self, X):
        """
        Make predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet!")

        # Apply same transformation pipeline
        X_enhanced = self._create_physics_features(X)

        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_enhanced.values)
        else:
            X_selected = X_enhanced.values if hasattr(
                X_enhanced, 'values') else X_enhanced

        X_scaled = self.scaler.transform(X_selected)

        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """
        Get prediction probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet!")

        # Apply same transformation pipeline
        X_enhanced = self._create_physics_features(X)

        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_enhanced.values)
        else:
            X_selected = X_enhanced.values if hasattr(
                X_enhanced, 'values') else X_enhanced

        X_scaled = self.scaler.transform(X_selected)

        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self):
        """
        Get feature importance from the underlying Random Forest.

        Returns
        -------
        importances : ndarray
            Feature importances.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        return self.model.feature_importances_

    def get_feature_names(self):
        """
        Get names of features used by the model.

        Returns
        -------
        feature_names : list
            Feature names.
        """
        return getattr(self, 'feature_names_', None)

    def get_params(self):
        """
        Get model parameters.

        Returns
        -------
        params : dict
            Model parameters.
        """
        if self.model is not None:
            return self.model.get_params()
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }


__all__ = ['PEECOM']
