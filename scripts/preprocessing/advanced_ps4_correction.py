#!/usr/bin/env python3
"""
Advanced PS4 Correction Algorithms

This script implements advanced algorithms specifically for PS4 sensor correction,
building on the analysis that showed PS4 has 66.68% zero readings (critical issue).

Algorithms implemented:
1. Multi-sensor correlation correction
2. Machine learning-based imputation  
3. Physical constraint modeling
4. Temporal pattern restoration
5. Ensemble correction approach

Author: PEECOM Project Team
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import signal, interpolate
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class AdvancedPS4Corrector:
    """Advanced PS4 sensor correction using multiple algorithms"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.correction_history = []
        self.confidence_scores = {}

    def method1_correlation_correction(self, ps4_data: np.ndarray,
                                       reference_sensors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Method 1: Multi-sensor correlation correction
        Uses PS1, PS3, PS5, PS6 to estimate PS4 values
        """
        self.logger.info(
            "Applying Method 1: Multi-sensor correlation correction")

        corrected_data = ps4_data.copy()
        zero_mask = (ps4_data == 0.0) | (np.abs(ps4_data) < 1e-6)

        if not np.any(zero_mask):
            return corrected_data, 1.0

        # Use multiple reference sensors for robust estimation
        ref_sensors = ['PS1', 'PS3', 'PS5', 'PS6']
        available_refs = {k: v for k,
                          v in reference_sensors.items() if k in ref_sensors}

        if len(available_refs) < 2:
            self.logger.warning(
                "Insufficient reference sensors for correlation correction")
            return corrected_data, 0.0

        # Calculate correlations and weights
        correlations = {}
        weights = {}

        for ref_name, ref_data in available_refs.items():
            if ref_data.shape == ps4_data.shape:
                # Calculate correlation on non-zero values
                valid_mask = ~zero_mask & (ref_data != 0)
                if np.any(valid_mask):
                    corr = np.corrcoef(ps4_data[valid_mask].flatten(),
                                       ref_data[valid_mask].flatten())[0, 1]
                    if not np.isnan(corr):
                        correlations[ref_name] = abs(corr)

        if not correlations:
            return corrected_data, 0.0

        # Normalize correlations to weights
        total_corr = sum(correlations.values())
        weights = {k: v / total_corr for k, v in correlations.items()}

        # Estimate PS4 values using weighted combination
        estimated_values = np.zeros_like(ps4_data)
        for ref_name, weight in weights.items():
            ref_data = available_refs[ref_name]
            estimated_values += weight * ref_data

        # Apply physical constraints (0-200 bar for pressure)
        estimated_values = np.clip(estimated_values, 0, 200)

        # Replace zero values
        corrected_data[zero_mask] = estimated_values[zero_mask]

        # Calculate confidence score
        confidence = np.mean(list(correlations.values())
                             ) if correlations else 0.0

        self.logger.info(
            f"Method 1: Used sensors {list(weights.keys())} with confidence {confidence:.3f}")
        return corrected_data, confidence

    def method2_ml_imputation(self, ps4_data: np.ndarray,
                              reference_sensors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Method 2: Machine Learning-based imputation
        Uses Random Forest to learn patterns from reference sensors
        """
        self.logger.info("Applying Method 2: ML-based imputation")

        corrected_data = ps4_data.copy()
        zero_mask = (ps4_data == 0.0) | (np.abs(ps4_data) < 1e-6)

        if not np.any(zero_mask) or len(reference_sensors) < 2:
            return corrected_data, 0.0

        try:
            # Prepare training data from non-zero values
            valid_mask = ~zero_mask

            # Create feature matrix from reference sensors
            features = []
            for ref_name, ref_data in reference_sensors.items():
                if ref_data.shape == ps4_data.shape:
                    features.append(ref_data.flatten())

            if len(features) < 2:
                return corrected_data, 0.0

            X = np.column_stack(features)
            y = ps4_data.flatten()

            # Use only valid (non-zero) samples for training
            valid_indices = valid_mask.flatten()
            X_train = X[valid_indices]
            y_train = y[valid_indices]

            if len(X_train) < 10:  # Need minimum samples
                return corrected_data, 0.0

            # Train Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=50, random_state=42, max_depth=10)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            rf_model.fit(X_train_scaled, y_train)

            # Predict missing values
            missing_indices = zero_mask.flatten()
            X_missing = X[missing_indices]
            X_missing_scaled = scaler.transform(X_missing)

            predicted_values = rf_model.predict(X_missing_scaled)

            # Apply physical constraints
            predicted_values = np.clip(predicted_values, 0, 200)

            # Replace zero values
            corrected_flat = corrected_data.flatten()
            corrected_flat[missing_indices] = predicted_values
            corrected_data = corrected_flat.reshape(ps4_data.shape)

            # Calculate confidence using model score
            confidence = rf_model.score(X_train_scaled, y_train)
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]

            self.logger.info(f"Method 2: ML model confidence {confidence:.3f}")
            return corrected_data, confidence

        except Exception as e:
            self.logger.error(f"Method 2 failed: {e}")
            return corrected_data, 0.0

    def method3_temporal_restoration(self, ps4_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Method 3: Temporal pattern restoration
        Uses time series analysis to restore missing values
        """
        from tqdm import tqdm

        self.logger.info("Applying Method 3: Temporal pattern restoration")

        corrected_data = ps4_data.copy()
        zero_mask = (ps4_data == 0.0) | (np.abs(ps4_data) < 1e-6)

        if not np.any(zero_mask):
            return corrected_data, 1.0

        try:
            # Process each cycle (row) separately
            confidence_scores = []

            for i in tqdm(range(ps4_data.shape[0]), desc="Temporal restoration", unit="cycle"):
                cycle_data = corrected_data[i, :].copy()
                cycle_zero_mask = zero_mask[i, :]

                if not np.any(cycle_zero_mask):
                    confidence_scores.append(1.0)
                    continue

                # Find valid (non-zero) segments
                valid_indices = np.where(~cycle_zero_mask)[0]

                if len(valid_indices) < 5:  # Need minimum valid points
                    confidence_scores.append(0.0)
                    continue

                # Interpolate using cubic spline
                try:
                    # Create interpolation function
                    interp_func = interpolate.interp1d(
                        valid_indices, cycle_data[valid_indices],
                        kind='cubic', bounds_error=False, fill_value='extrapolate'
                    )

                    # Fill missing values
                    missing_indices = np.where(cycle_zero_mask)[0]
                    interpolated_values = interp_func(missing_indices)

                    # Apply physical constraints
                    interpolated_values = np.clip(interpolated_values, 0, 200)

                    # Update cycle
                    cycle_data[missing_indices] = interpolated_values
                    corrected_data[i, :] = cycle_data

                    # Calculate confidence based on interpolation quality
                    confidence = max(
                        0.0, 1.0 - (len(missing_indices) / len(cycle_data)))
                    confidence_scores.append(confidence)

                except Exception:
                    # Fallback to linear interpolation
                    missing_indices = np.where(cycle_zero_mask)[0]
                    interpolated_values = np.interp(
                        missing_indices, valid_indices, cycle_data[valid_indices])
                    interpolated_values = np.clip(interpolated_values, 0, 200)
                    cycle_data[missing_indices] = interpolated_values
                    corrected_data[i, :] = cycle_data
                    confidence_scores.append(0.5)

            overall_confidence = np.mean(
                confidence_scores) if confidence_scores else 0.0

            self.logger.info(
                f"Method 3: Temporal restoration confidence {overall_confidence:.3f}")
            return corrected_data, overall_confidence

        except Exception as e:
            self.logger.error(f"Method 3 failed: {e}")
            return corrected_data, 0.0

    def method4_physical_modeling(self, ps4_data: np.ndarray,
                                  reference_sensors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Method 4: Physical constraint modeling
        Uses hydraulic system knowledge to estimate PS4
        """
        self.logger.info("Applying Method 4: Physical constraint modeling")

        corrected_data = ps4_data.copy()
        zero_mask = (ps4_data == 0.0) | (np.abs(ps4_data) < 1e-6)

        if not np.any(zero_mask):
            return corrected_data, 1.0

        # PS4 is typically related to pump outlet pressure
        # Use hydraulic relationships with other pressure sensors

        # Simple physical model: PS4 ≈ f(PS1, PS5, PS6)
        # Based on hydraulic circuit analysis

        confidence = 0.0

        if 'PS1' in reference_sensors and 'PS5' in reference_sensors:
            ps1_data = reference_sensors['PS1']
            ps5_data = reference_sensors['PS5']

            if ps1_data.shape == ps4_data.shape and ps5_data.shape == ps4_data.shape:
                # Physical relationship: PS4 should be between PS5 and a fraction of PS1
                # This is a simplified hydraulic model
                estimated_values = 0.1 * ps1_data + 0.9 * ps5_data

                # Apply constraints
                estimated_values = np.clip(estimated_values, 0, 200)

                # Replace zero values
                corrected_data[zero_mask] = estimated_values[zero_mask]
                confidence = 0.7  # Moderate confidence in physical model

        if confidence == 0.0 and 'PS3' in reference_sensors:
            # Fallback: use PS3 relationship
            ps3_data = reference_sensors['PS3']
            if ps3_data.shape == ps4_data.shape:
                # PS4 might be proportional to PS3 in some configurations
                estimated_values = 1.2 * ps3_data
                estimated_values = np.clip(estimated_values, 0, 200)
                corrected_data[zero_mask] = estimated_values[zero_mask]
                confidence = 0.5

        self.logger.info(
            f"Method 4: Physical modeling confidence {confidence:.3f}")
        return corrected_data, confidence

    def ensemble_correction(self, ps4_data: np.ndarray,
                            reference_sensors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        Ensemble approach: Combine all methods with weighted averaging
        """
        from tqdm import tqdm

        self.logger.info("Applying Ensemble PS4 correction")

        zero_mask = (ps4_data == 0.0) | (np.abs(ps4_data) < 1e-6)
        zero_percentage_before = np.sum(zero_mask) / ps4_data.size * 100

        if not np.any(zero_mask):
            return ps4_data, {'improvement': 0.0, 'methods_used': [], 'confidence': 1.0}

        # Apply all methods
        methods = {
            'correlation': self.method1_correlation_correction,
            'ml_imputation': self.method2_ml_imputation,
            'temporal': self.method3_temporal_restoration,
            'physical': self.method4_physical_modeling
        }

        results = {}
        confidences = {}

        for method_name, method_func in tqdm(methods.items(), desc="Applying correction methods", unit="method"):
            try:
                if method_name in ['correlation', 'ml_imputation', 'physical']:
                    corrected, confidence = method_func(
                        ps4_data, reference_sensors)
                else:
                    corrected, confidence = method_func(ps4_data)

                results[method_name] = corrected
                confidences[method_name] = confidence

            except Exception as e:
                self.logger.error(f"Method {method_name} failed: {e}")
                results[method_name] = ps4_data.copy()
                confidences[method_name] = 0.0

        # Weighted ensemble based on confidence scores
        total_confidence = sum(confidences.values())
        if total_confidence == 0:
            self.logger.warning("All methods failed, returning original data")
            return ps4_data, {'improvement': 0.0, 'methods_used': [], 'confidence': 0.0}

        # Normalize confidences to weights
        weights = {k: v / total_confidence for k, v in confidences.items()}

        # Weighted average of corrections
        ensemble_result = np.zeros_like(ps4_data)
        for method_name, weight in weights.items():
            ensemble_result += weight * results[method_name]

        # Apply final constraints
        ensemble_result = np.clip(ensemble_result, 0, 200)

        # Calculate improvement
        zero_mask_after = (ensemble_result == 0.0) | (
            np.abs(ensemble_result) < 1e-6)
        zero_percentage_after = np.sum(
            zero_mask_after) / ensemble_result.size * 100
        improvement = zero_percentage_before - zero_percentage_after

        # Prepare results summary
        methods_used = [k for k, v in confidences.items() if v > 0.1]
        overall_confidence = np.mean(
            [v for v in confidences.values() if v > 0])

        results_summary = {
            'improvement': improvement,
            'zero_percentage_before': zero_percentage_before,
            'zero_percentage_after': zero_percentage_after,
            'methods_used': methods_used,
            'method_confidences': confidences,
            'method_weights': weights,
            'overall_confidence': overall_confidence
        }

        self.logger.info(
            f"Ensemble correction: {zero_percentage_before:.2f}% → {zero_percentage_after:.2f}% zeros")
        self.logger.info(
            f"Improvement: {improvement:.2f}%, Confidence: {overall_confidence:.3f}")
        self.logger.info(f"Methods used: {methods_used}")

        return ensemble_result, results_summary


def apply_advanced_ps4_correction(sensor_data: Dict[str, np.ndarray],
                                  logger: Optional[logging.Logger] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Apply advanced PS4 correction to sensor data

    Args:
        sensor_data: Dictionary containing all sensor data
        logger: Optional logger instance

    Returns:
        Tuple of (corrected_sensor_data, correction_summary)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Starting advanced PS4 correction")

    if 'PS4' not in sensor_data:
        logger.warning("PS4 data not found in sensor data")
        return sensor_data, {'error': 'PS4 data not found'}

    # Create corrector instance
    corrector = AdvancedPS4Corrector(logger)

    # Extract PS4 data and reference sensors
    ps4_data = sensor_data['PS4']
    reference_sensors = {k: v for k, v in sensor_data.items()
                         if k.startswith('PS') and k != 'PS4'}

    logger.info(f"PS4 data shape: {ps4_data.shape}")
    logger.info(
        f"Reference sensors available: {list(reference_sensors.keys())}")

    # Apply ensemble correction
    corrected_ps4, correction_summary = corrector.ensemble_correction(
        ps4_data, reference_sensors)

    # Update sensor data
    corrected_sensor_data = sensor_data.copy()
    corrected_sensor_data['PS4'] = corrected_ps4

    logger.info("Advanced PS4 correction completed")

    return corrected_sensor_data, correction_summary


if __name__ == "__main__":
    # Test the advanced PS4 correction
    print("Advanced PS4 Correction Test")
    print("=" * 50)

    # This would be called from the main preprocessing workflow
    # For testing, you could load real data here
    pass
