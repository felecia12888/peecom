"""
Physics-Enhanced Feature Engineering for PEECOM Models

This module contains optimized physics-based feature engineering
specifically designed for hydraulic system condition monitoring.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings


class PhysicsFeatureEngineer:
    """
    Advanced physics-based feature engineering for hydraulic systems.

    Incorporates domain knowledge about hydraulic system behavior,
    thermodynamics, fluid mechanics, and system dynamics.
    """

    def __init__(self):
        self.feature_scaler = RobustScaler()
        self.feature_selector = None
        self.fitted = False

    def engineer_physics_features(self, X, y=None):
        """
        Create comprehensive physics-based features for hydraulic system monitoring.

        Args:
            X: Input features DataFrame or array
            y: Target labels (optional, for feature selection)

        Returns:
            Enhanced feature set with physics-based features
        """
        if isinstance(X, np.ndarray):
            # Convert to DataFrame for easier manipulation
            X = pd.DataFrame(
                X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        X_enhanced = X.copy()

        # 1. THERMODYNAMIC FEATURES
        X_enhanced = self._add_thermodynamic_features(X_enhanced)

        # 2. HYDRAULIC EFFICIENCY FEATURES
        X_enhanced = self._add_hydraulic_efficiency_features(X_enhanced)

        # 3. SYSTEM DYNAMICS FEATURES
        X_enhanced = self._add_system_dynamics_features(X_enhanced)

        # 4. STATISTICAL AGGREGATIONS
        X_enhanced = self._add_statistical_features(X_enhanced)

        # 5. INTERACTION FEATURES
        X_enhanced = self._add_interaction_features(X_enhanced)

        # Handle NaN values
        X_enhanced = X_enhanced.fillna(X_enhanced.median())

        # Feature scaling
        if not self.fitted:
            X_scaled = self.feature_scaler.fit_transform(X_enhanced)
            self.fitted = True
        else:
            X_scaled = self.feature_scaler.transform(X_enhanced)

        # Feature selection (optional)
        if y is not None and not self.fitted:
            self.feature_selector = SelectKBest(
                f_classif, k=min(150, X_scaled.shape[1]))
            X_scaled = self.feature_selector.fit_transform(X_scaled, y)
        elif self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)

        return X_scaled

    def _add_thermodynamic_features(self, X):
        """Add thermodynamic and temperature-based features"""
        # Temperature columns (assuming they exist)
        temp_cols = [col for col in X.columns if any(temp_indicator in col.lower()
                                                     for temp_indicator in ['temp', 'ts', 'cooling', 'thermal'])]

        if temp_cols:
            # Temperature differentials
            if len(temp_cols) >= 2:
                X['temp_differential_max'] = X[temp_cols].max(
                    axis=1) - X[temp_cols].min(axis=1)
                X['temp_differential_std'] = X[temp_cols].std(axis=1)
                X['temp_range'] = X[temp_cols].max(
                    axis=1) - X[temp_cols].min(axis=1)

            # Heat generation indicators
            X['total_thermal_load'] = X[temp_cols].sum(axis=1)
            X['avg_thermal_load'] = X[temp_cols].mean(axis=1)

            # Thermal efficiency ratios
            for i, col in enumerate(temp_cols):
                X[f'{col}_thermal_ratio'] = X[col] / \
                    (X[temp_cols].mean(axis=1) + 1e-8)

        return X

    def _add_hydraulic_efficiency_features(self, X):
        """Add hydraulic system efficiency and performance features"""
        # Pressure columns
        press_cols = [col for col in X.columns if any(press_indicator in col.lower()
                                                      for press_indicator in ['ps', 'pressure', 'press'])]

        if press_cols:
            # Pressure efficiency ratios
            X['pressure_efficiency'] = X[press_cols].min(
                axis=1) / (X[press_cols].max(axis=1) + 1e-8)
            X['pressure_variance'] = X[press_cols].var(axis=1)
            X['pressure_stability'] = 1 / (X[press_cols].std(axis=1) + 1e-8)

            # System load indicators
            X['hydraulic_load'] = X[press_cols].sum(axis=1)
            X['pressure_imbalance'] = X[press_cols].std(
                axis=1) / (X[press_cols].mean(axis=1) + 1e-8)

        # Flow-related features (if available)
        flow_cols = [col for col in X.columns if any(flow_indicator in col.lower()
                                                     for flow_indicator in ['fs', 'flow', 'volume'])]

        if flow_cols:
            X['flow_consistency'] = 1 / (X[flow_cols].std(axis=1) + 1e-8)
            X['total_flow'] = X[flow_cols].sum(axis=1)

        return X

    def _add_system_dynamics_features(self, X):
        """Add system dynamics and vibration-based features"""
        # Vibration columns
        vib_cols = [col for col in X.columns if any(vib_indicator in col.lower()
                                                    for vib_indicator in ['vs', 'vibr', 'vibration'])]

        if vib_cols:
            # Vibration analysis
            X['vibration_intensity'] = X[vib_cols].sum(axis=1)
            X['vibration_imbalance'] = X[vib_cols].std(axis=1)
            X['max_vibration'] = X[vib_cols].max(axis=1)

            # System stability indicators
            X['system_stability'] = 1 / (X[vib_cols].std(axis=1) + 1e-8)

        # Efficiency columns
        eff_cols = [col for col in X.columns if any(eff_indicator in col.lower()
                                                    for eff_indicator in ['eff', 'se', 'efficiency'])]

        if eff_cols:
            X['overall_efficiency'] = X[eff_cols].mean(axis=1)
            X['efficiency_variance'] = X[eff_cols].var(axis=1)

        return X

    def _add_statistical_features(self, X):
        """Add statistical aggregation features"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            # Cross-feature statistics
            X['feature_mean'] = X[numeric_cols].mean(axis=1)
            X['feature_std'] = X[numeric_cols].std(axis=1)
            X['feature_max'] = X[numeric_cols].max(axis=1)
            X['feature_min'] = X[numeric_cols].min(axis=1)
            X['feature_range'] = X['feature_max'] - X['feature_min']
            X['feature_skew'] = X[numeric_cols].skew(axis=1)
            X['feature_kurt'] = X[numeric_cols].kurtosis(axis=1)

            # Coefficient of variation
            X['feature_cv'] = X['feature_std'] / (X['feature_mean'] + 1e-8)

        return X

    def _add_interaction_features(self, X):
        """Add carefully selected interaction features"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # Focus on most important interactions to avoid feature explosion
        important_pairs = []

        # Temperature-Pressure interactions
        temp_cols = [col for col in numeric_cols if any(temp_indicator in col.lower()
                                                        for temp_indicator in ['temp', 'ts'])]
        press_cols = [col for col in numeric_cols if any(press_indicator in col.lower()
                                                         for press_indicator in ['ps', 'pressure'])]

        for temp_col in temp_cols[:3]:  # Limit to top 3
            for press_col in press_cols[:3]:  # Limit to top 3
                if temp_col in X.columns and press_col in X.columns:
                    X[f'{temp_col}_x_{press_col}'] = X[temp_col] * X[press_col]
                    X[f'{temp_col}_div_{press_col}'] = X[temp_col] / \
                        (X[press_col] + 1e-8)

        # Efficiency-Load interactions
        eff_cols = [col for col in numeric_cols if 'eff' in col.lower()][:2]
        load_cols = press_cols[:2] + temp_cols[:2]

        for eff_col in eff_cols:
            for load_col in load_cols[:2]:  # Limit interactions
                if eff_col in X.columns and load_col in X.columns:
                    X[f'{eff_col}_under_{load_col}'] = X[eff_col] / \
                        (X[load_col] + 1e-8)

        return X


class HighPerformancePhysicsFeatures:
    """
    Optimized physics feature engineering for maximum model performance.

    Based on analysis of feature importance from Random Forest baseline,
    this focuses on the most predictive physics-based features.
    """

    def __init__(self, target_features: int = 40):
        """
        Initialize feature engineer.

        Args:
            target_features: Target number of engineered features
        """
        self.target_features = target_features
        self.feature_names_ = None

    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer high-performance physics features.

        Args:
            X: Raw sensor data

        Returns:
            Engineered feature matrix
        """
        X_engineered = X.copy()

        # Get sensor mappings with robust fallbacks
        sensors = self._get_sensor_mappings(X)

        # === TIER 1: MOST PREDICTIVE FEATURES ===
        # Based on Random Forest feature importance analysis

        # 1. Flow Rate Features (Highest importance for pump_leakage)
        flow_primary = sensors['FS1'] + 1e-6
        flow_secondary = sensors['FS2'] + 1e-6

        X_engineered['flow_primary_mean'] = flow_primary.rolling(
            window=3, min_periods=1).mean() if hasattr(flow_primary, 'rolling') else flow_primary
        X_engineered['flow_primary_std'] = flow_primary.rolling(
            window=3, min_periods=1).std().fillna(0) if hasattr(flow_primary, 'rolling') else 0
        X_engineered['flow_balance'] = (flow_primary + flow_secondary) / 2
        X_engineered['flow_ratio'] = flow_primary / (flow_secondary + 1e-6)

        # 2. Pressure Analysis (Critical for valve_condition)
        X_engineered['pressure_ps2_skew'] = self._calculate_skewness(
            sensors['PS2'])
        X_engineered['pressure_ps2_kurtosis'] = self._calculate_kurtosis(
            sensors['PS2'])
        X_engineered['pressure_ratio_ps1_ps2'] = (
            sensors['PS1'] + 1) / (sensors['PS2'] + 1)
        X_engineered['pressure_diff_ps1_ps2'] = sensors['PS1'] - sensors['PS2']

        # 3. Hydraulic Power (Core physics principle)
        X_engineered['hydraulic_power_primary'] = sensors['PS1'] * \
            flow_primary / 1000.0
        X_engineered['hydraulic_power_secondary'] = sensors['PS2'] * \
            flow_secondary / 1000.0
        X_engineered['total_hydraulic_power'] = X_engineered['hydraulic_power_primary'] + \
            X_engineered['hydraulic_power_secondary']

        # Power efficiency (critical for system health)
        power_input = sensors['EPS1'] + 1e-6
        X_engineered['power_efficiency'] = X_engineered['total_hydraulic_power'] / power_input

        # 4. Temperature Analysis (Critical for cooler_condition)
        temp_sensors = [sensors['TS1'], sensors['TS2'],
                        sensors['TS3'], sensors['TS4']]
        temp_mean = sum(temp_sensors) / 4
        X_engineered['temp_mean'] = temp_mean
        X_engineered['temp_range'] = max(temp_sensors) - min(temp_sensors)
        X_engineered['thermal_efficiency'] = (
            sensors['TS1'] - sensors['TS3']) / (sensors['TS1'] + 273.15 + 1e-6)
        X_engineered['cooler_effectiveness'] = (
            sensors['TS1'] - sensors['TS2']) / (sensors['TS1'] + 1e-6)

        # 5. Control System Features (Important for accumulator_pressure)
        X_engineered['control_error'] = sensors['CE'] - sensors['CP']
        X_engineered['control_efficiency'] = sensors['CP'] / \
            (sensors['CE'] + 1e-6)
        X_engineered['se_indicator'] = sensors['SE']

        # === TIER 2: SYSTEM INTERACTION FEATURES ===

        # 6. Flow-Pressure Interactions
        X_engineered['flow_pressure_primary'] = flow_primary / \
            (sensors['PS1'] + 1e-6)
        X_engineered['flow_pressure_secondary'] = flow_secondary / \
            (sensors['PS2'] + 1e-6)
        X_engineered['pressure_flow_product'] = X_engineered['pressure_ratio_ps1_ps2'] * \
            X_engineered['flow_ratio']

        # 7. Multi-Physics Health Indicators
        pressure_stability = 1 / \
            (1 + np.abs(X_engineered['pressure_diff_ps1_ps2']
                        ) / (sensors['PS1'] + 1))
        flow_health = 1 / (1 + np.abs(flow_primary - flow_secondary) /
                           (flow_primary + flow_secondary + 1e-6))
        thermal_health = 1 / (1 + X_engineered['temp_range'] / 100)

        X_engineered['system_health_score'] = (
            pressure_stability + flow_health + thermal_health) / 3

        # 8. Advanced Physics Features
        # Reynolds number approximation
        density = 850  # kg/m³
        viscosity = 0.05  # Pa·s
        diameter = 0.02  # m

        X_engineered['reynolds_primary'] = (
            density * flow_primary * diameter) / (viscosity + 1e-6)
        X_engineered['reynolds_secondary'] = (
            density * flow_secondary * diameter) / (viscosity + 1e-6)

        # Energy dissipation
        X_engineered['energy_dissipation'] = sensors['PS1'] * \
            flow_primary**2 / 1000

        # 9. Vibration and Dynamic Features
        X_engineered['vibration_indicator'] = sensors['VS1']
        X_engineered['vibration_pressure_interaction'] = sensors['VS1'] * \
            X_engineered['pressure_ratio_ps1_ps2']
        X_engineered['dynamic_stability'] = 1 / (1 + sensors['VS1'])

        # 10. Composite Performance Indicators
        X_engineered['pump_performance'] = X_engineered['power_efficiency'] * flow_health
        X_engineered['valve_performance'] = X_engineered['pressure_ratio_ps1_ps2'] * \
            pressure_stability
        X_engineered['cooler_performance'] = X_engineered['thermal_efficiency'] * thermal_health
        X_engineered['accumulator_performance'] = (
            sensors['PS5'] / (sensors['PS1'] + 1)) * pressure_stability
        X_engineered['overall_system_efficiency'] = X_engineered['system_health_score'] * \
            X_engineered['power_efficiency']

        # Clean up any NaN or infinite values
        X_engineered = X_engineered.replace([np.inf, -np.inf], np.nan)
        X_engineered = X_engineered.fillna(X_engineered.median())

        # Store feature names
        self.feature_names_ = X_engineered.columns.tolist()

        print(
            f"High-performance physics features: {len(X_engineered.columns)} total features ({len(X_engineered.columns) - len(X.columns)} engineered)")

        return X_engineered

    def _get_sensor_mappings(self, X: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get robust sensor mappings with fallbacks."""
        sensors = {}
        sensor_names = ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2',
                        'TS1', 'TS2', 'TS3', 'TS4', 'VS1', 'CE', 'CP', 'SE']

        for i, sensor in enumerate(sensor_names):
            if sensor in X.columns:
                sensors[sensor] = X[sensor]
            elif i < len(X.columns):
                sensors[sensor] = X.iloc[:, i]
            else:
                # Safe fallbacks based on sensor type
                if 'PS' in sensor:
                    sensors[sensor] = pd.Series(
                        100.0, index=X.index)  # Pressure
                elif 'TS' in sensor:
                    sensors[sensor] = pd.Series(
                        40.0, index=X.index)   # Temperature
                elif 'FS' in sensor:
                    sensors[sensor] = pd.Series(10.0, index=X.index)   # Flow
                else:
                    sensors[sensor] = pd.Series(
                        1.0, index=X.index)    # General

        return sensors

    def _calculate_skewness(self, data: pd.Series) -> pd.Series:
        """Calculate skewness robustly."""
        try:
            if hasattr(data, 'skew'):
                return pd.Series(data.skew(), index=data.index)
            else:
                # Simple skewness approximation
                mean_val = data.mean()
                std_val = data.std()
                if std_val > 1e-6:
                    return (data - mean_val)**3 / (std_val**3)
                else:
                    return pd.Series(0.0, index=data.index)
        except:
            return pd.Series(0.0, index=data.index)

    def _calculate_kurtosis(self, data: pd.Series) -> pd.Series:
        """Calculate kurtosis robustly."""
        try:
            if hasattr(data, 'kurtosis'):
                return pd.Series(data.kurtosis(), index=data.index)
            else:
                # Simple kurtosis approximation
                mean_val = data.mean()
                std_val = data.std()
                if std_val > 1e-6:
                    return (data - mean_val)**4 / (std_val**4) - 3
                else:
                    return pd.Series(0.0, index=data.index)
        except:
            return pd.Series(0.0, index=data.index)

    def get_feature_names(self) -> Optional[List[str]]:
        """Get engineered feature names."""
        return self.feature_names_


class LightweightPhysicsFeatures:
    """
    Lightweight feature engineering focused on core physics principles.

    This variant uses fewer features but focuses on the most predictive
    physics relationships for maximum efficiency.
    """

    def __init__(self):
        self.feature_names_ = None

    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer lightweight physics features.

        Args:
            X: Raw sensor data

        Returns:
            Engineered feature matrix with ~20 key features
        """
        X_engineered = X.copy()

        # Get sensor mappings
        sensors = self._get_sensor_mappings(X)

        # Core physics features only
        flow_primary = sensors['FS1'] + 1e-6
        flow_secondary = sensors['FS2'] + 1e-6

        # 1. Hydraulic Power (most fundamental)
        X_engineered['hydraulic_power'] = sensors['PS1'] * \
            flow_primary / 1000.0
        X_engineered['power_efficiency'] = X_engineered['hydraulic_power'] / \
            (sensors['EPS1'] + 1e-6)

        # 2. Critical Pressure Ratios
        X_engineered['pressure_ratio_main'] = (
            sensors['PS1'] + 1) / (sensors['PS2'] + 1)
        X_engineered['pressure_ratio_accumulator'] = (
            sensors['PS1'] + 1) / (sensors['PS5'] + 1)

        # 3. Flow Balance
        X_engineered['flow_balance'] = (flow_primary + flow_secondary) / 2
        X_engineered['flow_conservation_error'] = np.abs(
            flow_primary - flow_secondary) / (flow_primary + flow_secondary + 1e-6)

        # 4. Thermal Efficiency
        X_engineered['thermal_efficiency'] = (
            sensors['TS1'] - sensors['TS3']) / (sensors['TS1'] + 273.15 + 1e-6)
        X_engineered['temp_range'] = sensors['TS1'] - \
            sensors['TS2']  # Simplified

        # 5. System Health Indicators
        pressure_health = 1 / \
            (1 + np.abs(sensors['PS1'] -
             sensors['PS2']) / (sensors['PS1'] + 1))
        flow_health = 1 / (1 + X_engineered['flow_conservation_error'])
        thermal_health = 1 / (1 + np.abs(X_engineered['temp_range']) / 50)

        X_engineered['system_health'] = (
            pressure_health + flow_health + thermal_health) / 3

        # 6. Control System
        X_engineered['control_efficiency'] = sensors['CP'] / \
            (sensors['CE'] + 1e-6)

        # Clean up
        X_engineered = X_engineered.replace([np.inf, -np.inf], np.nan)
        X_engineered = X_engineered.fillna(X_engineered.median())

        self.feature_names_ = X_engineered.columns.tolist()

        print(
            f"Lightweight physics features: {len(X_engineered.columns)} total features ({len(X_engineered.columns) - len(X.columns)} engineered)")

        return X_engineered

    def _get_sensor_mappings(self, X: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get robust sensor mappings with fallbacks."""
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
                    sensors[sensor] = pd.Series(100.0, index=X.index)
                elif 'TS' in sensor:
                    sensors[sensor] = pd.Series(40.0, index=X.index)
                elif 'FS' in sensor:
                    sensors[sensor] = pd.Series(10.0, index=X.index)
                else:
                    sensors[sensor] = pd.Series(1.0, index=X.index)

        return sensors

    def get_feature_names(self) -> Optional[List[str]]:
        """Get engineered feature names."""
        return self.feature_names_
