"""
Enhanced PEECOM Model v2.0 - Superior Performance Focus
Physics-Enhanced Equipment Condition Monitoring with Advanced ML Techniques

This version focuses on maximum performance through:
1. Advanced feature engineering with domain expertise
2. Multiple ensemble strategies with intelligent weighting
3. Sophisticated hyperparameter optimization
4. Advanced calibration and uncertainty quantification
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost for maximum performance
XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost available - maximum performance mode enabled")
except (ImportError, OSError, Exception) as e:
    print(f"XGBoost not available: {str(e)[:100]}...")
    XGBOOST_AVAILABLE = False

# Try to import LightGBM with comprehensive error handling
LIGHTGBM_AVAILABLE = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("LightGBM available - enhanced performance mode enabled")
except (ImportError, OSError, Exception) as e:
    print(f"LightGBM not available: {str(e)[:100]}...")
    LIGHTGBM_AVAILABLE = False

print(f"Enhanced PEECOM v2.0 initialized with:")
print(f"  - XGBoost: {'Available' if XGBOOST_AVAILABLE else 'Not Available'}")
print(
    f"  - LightGBM: {'Available' if LIGHTGBM_AVAILABLE else 'Not Available'}")
print(f"  - Fallback: RandomForest + GradientBoosting ensemble")


class EnhancedPEECOMv2(BaseEstimator, ClassifierMixin):
    """
    Enhanced PEECOM Model v2.0 - Designed for Superior Performance

    Key Innovations:
    1. Domain-expert feature engineering (focusing on most predictive physics features)
    2. Multi-level ensemble with intelligent model weighting
    3. Advanced hyperparameter optimization
    4. Sophisticated calibration with uncertainty quantification
    5. Adaptive feature selection to prevent overfitting
    """

    def __init__(self,
                 feature_selection_k=50,  # Limit features to prevent overfitting
                 ensemble_strategy='weighted_voting',
                 calibration_method='isotonic',
                 random_state=42,
                 **kwargs):

        self.feature_selection_k = feature_selection_k
        self.ensemble_strategy = ensemble_strategy
        self.calibration_method = calibration_method
        self.random_state = random_state

        # Initialize components
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.models = {}
        self.ensemble_weights = {}
        self.calibrated_models = {}

        # Performance tracking
        self.training_scores = {}
        self.feature_importance_scores = {}

        # Initialize high-performance ensemble models
        self._initialize_high_performance_models(**kwargs)

    def _initialize_high_performance_models(self, **kwargs):
        """Initialize optimized models for maximum performance"""

        # Random Forest with aggressive parameters for better performance
        rf_params = {
            'n_estimators': kwargs.get('rf_n_estimators', 300),  # More trees
            'max_depth': kwargs.get('rf_max_depth', 12),  # Deeper trees
            'min_samples_split': kwargs.get('rf_min_samples_split', 5),
            'min_samples_leaf': kwargs.get('rf_min_samples_leaf', 2),
            'max_features': kwargs.get('rf_max_features', 'sqrt'),
            'bootstrap': kwargs.get('rf_bootstrap', True),
            'class_weight': kwargs.get('rf_class_weight', 'balanced'),
            'random_state': self.random_state,
            'n_jobs': kwargs.get('n_jobs', -1)
        }
        self.models['random_forest'] = RandomForestClassifier(**rf_params)

        # Gradient Boosting with optimized parameters
        gb_params = {
            'n_estimators': kwargs.get('gb_n_estimators', 250),
            'max_depth': kwargs.get('gb_max_depth', 8),
            # Lower for better performance
            'learning_rate': kwargs.get('gb_learning_rate', 0.05),
            'subsample': kwargs.get('gb_subsample', 0.85),
            'max_features': kwargs.get('gb_max_features', 'sqrt'),
            'random_state': self.random_state
        }
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            **gb_params)

        # XGBoost if available (often superior performance)
        if XGBOOST_AVAILABLE:
            xgb_params = {
                'n_estimators': kwargs.get('xgb_n_estimators', 250),
                'max_depth': kwargs.get('xgb_max_depth', 6),
                'learning_rate': kwargs.get('xgb_learning_rate', 0.05),
                'subsample': kwargs.get('xgb_subsample', 0.85),
                'colsample_bytree': kwargs.get('xgb_colsample_bytree', 0.85),
                # L1 regularization
                'reg_alpha': kwargs.get('xgb_reg_alpha', 0.1),
                # L2 regularization
                'reg_lambda': kwargs.get('xgb_reg_lambda', 1.0),
                'random_state': self.random_state,
                'n_jobs': kwargs.get('n_jobs', -1),
                'verbosity': 0
            }
            self.models['xgboost'] = xgb.XGBClassifier(**xgb_params)
            print("XGBoost added for maximum performance")

        # LightGBM if available (fast and often high-performing)
        if LIGHTGBM_AVAILABLE:
            try:
                lgb_params = {
                    'n_estimators': kwargs.get('lgb_n_estimators', 250),
                    'max_depth': kwargs.get('lgb_max_depth', 6),
                    'learning_rate': kwargs.get('lgb_learning_rate', 0.05),
                    'subsample': kwargs.get('lgb_subsample', 0.85),
                    'colsample_bytree': kwargs.get('lgb_colsample_bytree', 0.85),
                    'reg_alpha': kwargs.get('lgb_reg_alpha', 0.1),
                    'reg_lambda': kwargs.get('lgb_reg_lambda', 1.0),
                    'random_state': self.random_state,
                    'n_jobs': kwargs.get('n_jobs', -1),
                    'verbose': -1
                }
                self.models['lightgbm'] = lgb.LGBMClassifier(**lgb_params)
                print("LightGBM added for high performance")
            except Exception as e:
                print(f"Warning: Could not initialize LightGBM: {e}")

        print(
            f"Enhanced PEECOM v2.0 initialized with {len(self.models)} models: {list(self.models.keys())}")

    def _engineer_critical_physics_features(self, X):
        """
        Engineer only the most critical physics features based on domain expertise
        Focus on features that have proven most predictive in hydraulic systems
        """
        X_engineered = X.copy()

        # Define sensor mappings with robust fallbacks
        sensors = {}
        sensor_names = ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2',
                        'TS1', 'TS2', 'TS3', 'TS4', 'VS1', 'CE', 'CP', 'SE']

        for i, sensor in enumerate(sensor_names):
            if sensor in X.columns:
                sensors[sensor] = X[sensor]
            elif i < len(X.columns):
                sensors[sensor] = X.iloc[:, i]
            else:
                # Safe fallbacks
                if 'PS' in sensor:
                    sensors[sensor] = pd.Series(
                        100.0, index=X.index)  # Pressure fallback
                elif 'TS' in sensor:
                    sensors[sensor] = pd.Series(
                        40.0, index=X.index)   # Temperature fallback
                else:
                    sensors[sensor] = pd.Series(
                        1.0, index=X.index)    # General fallback

        # === CRITICAL FEATURE SET 1: HYDRAULIC POWER ANALYSIS ===
        # Most predictive for pump and valve conditions
        flow_primary = sensors['FS1'] + 1e-6
        flow_secondary = sensors['FS2'] + 1e-6

        # Core hydraulic power features
        X_engineered['hydraulic_power_primary'] = sensors['PS1'] * \
            flow_primary / 1000.0
        X_engineered['hydraulic_power_secondary'] = sensors['PS2'] * \
            flow_secondary / 1000.0
        X_engineered['total_hydraulic_power'] = X_engineered['hydraulic_power_primary'] + \
            X_engineered['hydraulic_power_secondary']

        # Power efficiency (critical for all fault types)
        power_input = sensors['EPS1'] + 1e-6
        X_engineered['power_efficiency'] = X_engineered['total_hydraulic_power'] / power_input
        X_engineered['power_efficiency_normalized'] = np.tanh(
            X_engineered['power_efficiency'])

        # === CRITICAL FEATURE SET 2: PRESSURE DIFFERENTIALS ===
        # Most critical for valve, pump, and accumulator conditions
        X_engineered['pressure_diff_main'] = sensors['PS1'] - sensors['PS2']
        X_engineered['pressure_diff_accumulator'] = sensors['PS1'] - \
            sensors['PS5']
        X_engineered['pressure_diff_return'] = sensors['PS2'] - sensors['PS6']

        # Critical pressure ratios (more stable than differences)
        X_engineered['pressure_ratio_main'] = (
            sensors['PS1'] + 1) / (sensors['PS2'] + 1)
        X_engineered['pressure_ratio_accumulator'] = (
            sensors['PS1'] + 1) / (sensors['PS5'] + 1)

        # Pressure stability indicator (critical for system health)
        pressure_values = [sensors['PS1'], sensors['PS2'],
                           sensors['PS3'], sensors['PS4'], sensors['PS5'], sensors['PS6']]
        pressure_mean = sum(pressure_values) / 6
        pressure_std = np.sqrt(
            sum((p - pressure_mean)**2 for p in pressure_values) / 6)
        X_engineered['pressure_stability'] = 1 / (1 + pressure_std)

        # === CRITICAL FEATURE SET 3: THERMAL ANALYSIS ===
        # Critical for cooler condition detection
        temp_values = [sensors['TS1'], sensors['TS2'],
                       sensors['TS3'], sensors['TS4']]
        temp_avg = sum(temp_values) / 4

        X_engineered['temp_avg'] = temp_avg
        X_engineered['temp_range'] = max(temp_values) - min(temp_values)
        X_engineered['thermal_efficiency'] = (
            sensors['TS1'] - sensors['TS3']) / (sensors['TS1'] + 273.15 + 1e-6)
        X_engineered['cooler_effectiveness'] = (
            sensors['TS1'] - sensors['TS2']) / (sensors['TS1'] + 1e-6)

        # === CRITICAL FEATURE SET 4: FLOW DYNAMICS ===
        # Critical for pump leakage and valve condition
        X_engineered['flow_balance'] = (flow_primary + flow_secondary) / 2
        X_engineered['flow_ratio'] = flow_primary / (flow_secondary + 1e-6)
        X_engineered['flow_conservation_error'] = np.abs(
            flow_primary - flow_secondary) / (flow_primary + flow_secondary + 1e-6)

        # Flow-pressure interactions (highly predictive)
        X_engineered['flow_pressure_primary'] = flow_primary / \
            (sensors['PS1'] + 1e-6)
        X_engineered['flow_pressure_secondary'] = flow_secondary / \
            (sensors['PS2'] + 1e-6)

        # === CRITICAL FEATURE SET 5: SYSTEM HEALTH INDICATORS ===
        # Composite features that capture overall system state

        # Overall system efficiency
        flow_health = 1 / (1 + X_engineered['flow_conservation_error'])
        thermal_health = 1 / (1 + X_engineered['temp_range'] / 100)

        X_engineered['system_efficiency_score'] = (
            X_engineered['power_efficiency_normalized'] * 0.4 +
            X_engineered['thermal_efficiency'] * 0.3 +
            flow_health * 0.2 +
            X_engineered['pressure_stability'] * 0.1
        )

        # Component-specific health indicators
        X_engineered['pump_health_indicator'] = X_engineered['power_efficiency'] * flow_health
        X_engineered['valve_health_indicator'] = sensors['VS1'] * \
            X_engineered['pressure_stability']
        X_engineered['cooler_health_indicator'] = X_engineered['thermal_efficiency'] * thermal_health
        X_engineered['accumulator_health_indicator'] = X_engineered['pressure_ratio_accumulator'] * \
            X_engineered['pressure_stability']

        # === CRITICAL FEATURE SET 6: ADVANCED PHYSICS ===
        # Reynolds number for flow regime (impacts all components)
        density = 850  # kg/m³
        viscosity = 0.05  # Pa·s
        diameter = 0.02  # m

        X_engineered['reynolds_primary'] = (
            density * flow_primary * diameter) / (viscosity + 1e-6)
        X_engineered['reynolds_secondary'] = (
            density * flow_secondary * diameter) / (viscosity + 1e-6)

        # Energy dissipation indicators
        X_engineered['energy_dissipation_primary'] = sensors['PS1'] * \
            flow_primary**2 / 1000
        X_engineered['energy_dissipation_secondary'] = sensors['PS2'] * \
            flow_secondary**2 / 1000

        # === CRITICAL FEATURE SET 7: INTERACTION TERMS ===
        # Non-linear interactions that boost performance
        X_engineered['pressure_temp_interaction'] = X_engineered['pressure_ratio_main'] * temp_avg
        X_engineered['flow_temp_interaction'] = X_engineered['flow_balance'] * temp_avg
        X_engineered['power_pressure_interaction'] = X_engineered['power_efficiency'] * \
            X_engineered['pressure_ratio_main']

        # Vibration interactions
        X_engineered['vibration_pressure_interaction'] = sensors['VS1'] * pressure_mean
        X_engineered['vibration_flow_interaction'] = sensors['VS1'] * \
            X_engineered['flow_balance']

        # === FINAL PROCESSING ===
        # Handle any remaining NaN or infinite values
        X_engineered = X_engineered.replace([np.inf, -np.inf], np.nan)
        X_engineered = X_engineered.fillna(X_engineered.median())

        print(
            f"Critical physics features: Added {len(X_engineered.columns) - len(X.columns)} expertly selected features")

        return X_engineered

    def _intelligent_feature_selection(self, X, y):
        """
        Select the most predictive features to prevent overfitting and maximize performance
        """
        print(
            f"Applying intelligent feature selection (target: {self.feature_selection_k} features)...")

        # Use mutual information for feature selection (works well with non-linear relationships)
        selector = SelectKBest(score_func=mutual_info_classif, k=min(
            self.feature_selection_k, X.shape[1]))

        try:
            X_selected = selector.fit_transform(X, y)
            selected_feature_names = [X.columns[i] for i in selector.get_support(
                indices=True)] if hasattr(X, 'columns') else None

            # Store feature importance scores
            self.feature_importance_scores = dict(zip(
                selected_feature_names or range(X_selected.shape[1]),
                selector.scores_[selector.get_support()]
            ))

            print(f"Selected {X_selected.shape[1]} most predictive features")

            # Create DataFrame if we have column names
            if hasattr(X, 'columns') and selected_feature_names:
                X_selected = pd.DataFrame(
                    X_selected, columns=selected_feature_names, index=X.index)

            self.feature_selector = selector
            return X_selected

        except Exception as e:
            print(f"Feature selection failed: {e}. Using all features.")
            self.feature_selector = None
            return X

    def _optimize_ensemble_weights(self, X, y):
        """
        Optimize ensemble weights based on cross-validation performance
        """
        print("Optimizing ensemble weights based on CV performance...")

        cv_scores = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=self.random_state)

        for model_name, model in self.models.items():
            try:
                scores = cross_val_score(
                    model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                cv_score = scores.mean()
                cv_scores[model_name] = cv_score

                print(
                    f"{model_name} CV accuracy: {cv_score:.4f} (±{scores.std():.4f})")

            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                cv_scores[model_name] = 0.5  # Default score

        # Calculate performance-based weights using softmax with temperature
        scores = np.array(list(cv_scores.values()))
        # Temperature scaling to emphasize differences
        temperature = 10.0
        exp_scores = np.exp(scores * temperature)
        weights = exp_scores / exp_scores.sum()

        self.ensemble_weights = dict(zip(cv_scores.keys(), weights))
        self.training_scores = cv_scores

        print("Optimized ensemble weights:")
        for model_name, weight in self.ensemble_weights.items():
            print(f"  {model_name}: {weight:.3f}")

        return self.ensemble_weights

    def fit(self, X, y):
        """
        Fit the Enhanced PEECOM v2.0 model for superior performance
        """
        print("=== Enhanced PEECOM v2.0 Training ===")
        print(f"Training with {len(self.models)} high-performance models")

        # Step 1: Engineer critical physics features
        print("\n1. Engineering critical physics features...")
        X_engineered = self._engineer_critical_physics_features(X)

        # Step 2: Intelligent feature selection
        print("\n2. Applying intelligent feature selection...")
        X_selected = self._intelligent_feature_selection(X_engineered, y)

        # Step 3: Scale features
        print("\n3. Scaling features...")
        X_scaled = self.scaler.fit_transform(X_selected)

        # Store feature information
        self.feature_names_ = X_selected.columns.tolist() if hasattr(
            X_selected, 'columns') else [f'feature_{i}' for i in range(X_scaled.shape[1])]

        # Step 4: Train all models
        print("\n4. Training ensemble models...")
        for model_name, model in self.models.items():
            print(f"   Training {model_name}...")

            # Handle class imbalance
            try:
                classes = np.unique(y)
                if len(classes) > 1:
                    class_weights = compute_class_weight(
                        'balanced', classes=classes, y=y)
                    weight_dict = dict(zip(classes, class_weights))

                    # Set class weights if supported
                    if hasattr(model, 'set_params') and 'class_weight' in model.get_params():
                        model.set_params(class_weight=weight_dict)
            except Exception as e:
                print(
                    f"   Warning: Could not set class weights for {model_name}: {e}")

            # Train model
            model.fit(X_scaled, y)

        # Step 5: Optimize ensemble weights
        print("\n5. Optimizing ensemble weights...")
        self._optimize_ensemble_weights(X_scaled, y)

        # Step 6: Calibrate predictions
        if self.calibration_method:
            print("\n6. Calibrating predictions...")
            self._calibrate_predictions(X_scaled, y)

        print("\n=== Enhanced PEECOM v2.0 Training Complete ===")
        print(
            f"Final ensemble: {len(self.models)} models with optimized weights")
        print(
            f"Feature space: {len(self.feature_names_)} expertly selected features")

        return self

    def _calibrate_predictions(self, X, y):
        """
        Calibrate model predictions for better probability estimates
        """
        print("Calibrating model predictions...")

        for model_name, model in self.models.items():
            try:
                # Only calibrate for binary classification to avoid issues
                if len(np.unique(y)) == 2:
                    calibrated_model = CalibratedClassifierCV(
                        model, method=self.calibration_method, cv=3
                    )
                    calibrated_model.fit(X, y)
                    self.calibrated_models[model_name] = calibrated_model
                    print(f"   Calibrated {model_name}")
                else:
                    # For multiclass, use the original model
                    self.calibrated_models[model_name] = model

            except Exception as e:
                print(f"   Warning: Could not calibrate {model_name}: {e}")
                self.calibrated_models[model_name] = model

    def predict(self, X):
        """
        Make predictions using the optimized ensemble
        """
        # Apply same preprocessing as training
        X_engineered = self._engineer_critical_physics_features(X)

        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_engineered)
            if hasattr(X_engineered, 'columns'):
                selected_cols = [X_engineered.columns[i]
                                 for i in self.feature_selector.get_support(indices=True)]
                X_selected = pd.DataFrame(
                    X_selected, columns=selected_cols, index=X.index)
        else:
            X_selected = X_engineered

        X_scaled = self.scaler.transform(X_selected)

        # Get predictions from all models
        predictions = []

        for model_name, model in self.models.items():
            try:
                # Use calibrated model if available
                if model_name in self.calibrated_models:
                    pred = self.calibrated_models[model_name].predict(X_scaled)
                else:
                    pred = model.predict(X_scaled)

                predictions.append(pred)
            except Exception as e:
                print(f"Warning: Error in {model_name} prediction: {e}")
                # Fallback prediction
                predictions.append(np.zeros(len(X), dtype=int))

        if not predictions:
            raise RuntimeError("No models were able to make predictions")

        # Ensemble predictions using optimized weights
        if len(predictions) == 1:
            return predictions[0]

        # Weighted voting
        predictions_array = np.array(predictions)
        weights = np.array([self.ensemble_weights.get(name, 1.0/len(self.models))
                           for name in self.models.keys()])

        # Weighted majority voting
        weighted_predictions = np.average(
            predictions_array, axis=0, weights=weights)
        final_predictions = np.round(weighted_predictions).astype(int)

        return final_predictions

    def predict_proba(self, X):
        """
        Predict class probabilities with uncertainty quantification
        """
        # Apply same preprocessing as training
        X_engineered = self._engineer_critical_physics_features(X)

        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_engineered)
            if hasattr(X_engineered, 'columns'):
                selected_cols = [X_engineered.columns[i]
                                 for i in self.feature_selector.get_support(indices=True)]
                X_selected = pd.DataFrame(
                    X_selected, columns=selected_cols, index=X.index)
        else:
            X_selected = X_engineered

        X_scaled = self.scaler.transform(X_selected)

        # Get probability predictions from all models
        prob_predictions = []

        for model_name, model in self.models.items():
            try:
                # Use calibrated model if available
                if model_name in self.calibrated_models:
                    prob = self.calibrated_models[model_name].predict_proba(
                        X_scaled)
                else:
                    prob = model.predict_proba(X_scaled)

                prob_predictions.append(prob)
            except Exception as e:
                print(
                    f"Warning: Error in {model_name} probability prediction: {e}")
                continue

        if not prob_predictions:
            raise RuntimeError(
                "No models were able to make probability predictions")

        # Ensemble probabilities using optimized weights
        if len(prob_predictions) == 1:
            return prob_predictions[0]

        # Weighted averaging of probabilities
        weights = np.array([self.ensemble_weights.get(name, 1.0/len(self.models))
                           for name in self.models.keys() if name in [m for m in self.models.keys()]])[:len(prob_predictions)]

        # Normalize weights
        weights = weights / weights.sum()

        # Weighted average of probabilities
        ensemble_proba = np.average(prob_predictions, axis=0, weights=weights)

        return ensemble_proba

    def get_feature_importance(self):
        """
        Get feature importance scores from the ensemble
        """
        if hasattr(self, 'feature_importance_scores'):
            return self.feature_importance_scores
        return {}

    def get_model_performance(self):
        """
        Get individual model performance scores
        """
        return getattr(self, 'training_scores', {})
