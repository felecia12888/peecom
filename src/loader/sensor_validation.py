import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import RobustScaler


class AdvancedSensorValidator:
    """Enhanced sensor validator based on analysis results"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()

        self.sensor_specs = self.config.get('sensor_specs', {})
        self.preprocessing_config = self.config.get('preprocessing', {})
        self.alerts: List[str] = []

    def _get_default_config(self) -> Dict:
        """Default configuration based on analysis findings"""
        return {
            'preprocessing': {
                'zero_handling': {
                    'PS4': {'method': 'correlation_interpolation', 'threshold': 0.1},
                    'PS2': {'method': 'calibration_correction', 'threshold': 0.1},
                    'PS3': {'method': 'calibration_correction', 'threshold': 0.1},
                    'FS1': {'method': 'flow_validation', 'threshold': 0.2},
                    'SE1': {'method': 'efficiency_interpolation', 'threshold': 0.1}
                }
            }
        }

    def detect_zero_sequences(self, data: np.ndarray, threshold: int = 10) -> Dict:
        """Enhanced zero sequence detection"""
        zero_mask = (data == 0.0) | (np.abs(data) < 1e-6)

        # Find start and end of zero sequences
        diff = np.diff(np.concatenate(
            ([False], zero_mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        sequences = [(start, end) for start, end in zip(starts, ends)
                     if end - start >= threshold]

        return {
            'sequences': sequences,
            'count': len(sequences),
            'max_length': max([end - start for start, end in sequences]) if sequences else 0,
            'total_zeros': np.sum(zero_mask),
            'zero_percentage': np.sum(zero_mask) / len(data) * 100
        }

    def validate_sensor_health(self, sensor_id: str, data: np.ndarray) -> Dict:
        """Comprehensive sensor health validation"""
        self.alerts = []

        # Get sensor specifications
        sensor_spec = self.sensor_specs.get(sensor_id, {})
        expected_range = sensor_spec.get('expected_range', [0, 1000])

        # Zero analysis
        zero_analysis = self.detect_zero_sequences(data)

        # Statistical analysis
        stats_analysis = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'range_violations': np.sum((data < expected_range[0]) | (data > expected_range[1]))
        }

        # Determine health status based on analysis findings
        health_score = 100
        status = "Good"
        issues = []
        recommendations = []

        # Check zero percentage (based on analysis findings)
        if zero_analysis['zero_percentage'] > 50:
            health_score = 0
            status = "Critical"
            issues.append(
                f"Critical zero readings: {zero_analysis['zero_percentage']:.1f}%")
            recommendations.append(
                "IMMEDIATE ACTION: Replace or recalibrate sensor")
        elif zero_analysis['zero_percentage'] > 10:
            health_score = max(0, 100 - zero_analysis['zero_percentage'] * 3)
            status = "Warning"
            issues.append(
                f"High zero readings: {zero_analysis['zero_percentage']:.1f}%")
            recommendations.append(
                "Schedule calibration within next maintenance window")
        elif zero_analysis['zero_percentage'] > 5:
            health_score = max(50, 100 - zero_analysis['zero_percentage'] * 5)
            status = "Warning"
            issues.append(
                f"Moderate zero readings: {zero_analysis['zero_percentage']:.1f}%")
            recommendations.append("Monitor and validate readings")

        # Check for extended zero sequences
        if zero_analysis['max_length'] > 100:
            health_score = min(health_score, 50)
            status = "Critical" if status != "Critical" else status
            issues.append("Extended zero sequences detected")

        return {
            'sensor_id': sensor_id,
            'health_score': health_score,
            'status': status,
            'issues': issues,
            'recommendations': recommendations,
            'zero_analysis': zero_analysis,
            'statistics': stats_analysis,
            'alerts': self.alerts.copy()
        }


class PS4Corrector:
    """Enhanced PS4 correction based on analysis results"""

    def __init__(self):
        # Correlation coefficients from good pressure sensors
        self.reference_sensors = ['PS1', 'PS5', 'PS6']

    def estimate_ps4_multi_sensor(self, ps1: np.ndarray, ps5: np.ndarray,
                                  ps6: np.ndarray) -> np.ndarray:
        """Estimate PS4 using multiple reference sensors"""
        # Weighted average based on sensor reliability
        # PS1, PS5, PS6 all have 100% health scores
        weights = [0.33, 0.33, 0.34]  # Equal weights for good sensors

        # Simple linear combination (can be enhanced with learned coefficients)
        estimated = (weights[0] * ps1 + weights[1]
                     * ps5 + weights[2] * ps6) / 3

        # Apply reasonable bounds for pressure sensor
        estimated = np.clip(estimated, 0, 200)
        return estimated

    def interpolate_zeros(self, data: np.ndarray, method: str = 'linear') -> np.ndarray:
        """Interpolate zero values"""
        data_copy = data.copy()
        zero_mask = (data == 0.0) | (np.abs(data) < 1e-6)

        if method == 'linear':
            # Linear interpolation
            valid_indices = np.where(~zero_mask)[0]
            if len(valid_indices) > 1:
                data_copy[zero_mask] = np.interp(
                    np.where(zero_mask)[0], valid_indices, data[valid_indices]
                )
        elif method == 'median_filter':
            # Apply median filter for outliers
            from scipy.signal import medfilt
            data_copy = medfilt(data_copy, kernel_size=5)

        return data_copy

    def correct_ps4_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced PS4 correction using multiple methods"""
        df = df.copy()

        if 'PS4' not in df.columns:
            return df

        ps4_data = df['PS4'].values
        zero_mask = (ps4_data == 0.0) | (np.abs(ps4_data) < 1e-6)

        # If too many zeros (>50%), use multi-sensor estimation
        zero_percentage = np.sum(zero_mask) / len(ps4_data) * 100

        if zero_percentage > 50:
            # Use correlation with other sensors
            if all(col in df.columns for col in ['PS1', 'PS5', 'PS6']):
                estimated = self.estimate_ps4_multi_sensor(
                    df['PS1'].values, df['PS5'].values, df['PS6'].values
                )
                df['PS4_corrected'] = np.where(zero_mask, estimated, ps4_data)
            else:
                # Fallback to interpolation
                df['PS4_corrected'] = self.interpolate_zeros(ps4_data)
        else:
            # For smaller zero percentages, use interpolation
            df['PS4_corrected'] = self.interpolate_zeros(ps4_data)

        return df


class CalibrationCorrector:
    """Corrector for sensors needing calibration (PS2, PS3)"""

    def __init__(self):
        self.drift_window = 100  # Use first 100 samples as reference

    def detect_drift(self, data: np.ndarray) -> Dict:
        """Detect calibration drift"""
        reference_mean = np.mean(data[:self.drift_window])
        current_mean = np.mean(data[-self.drift_window:])
        drift = current_mean - reference_mean

        return {
            'drift_detected': abs(drift) > 0.1 * reference_mean,
            'drift_amount': drift,
            'drift_percentage': (drift / reference_mean) * 100 if reference_mean != 0 else 0
        }

    def apply_calibration_correction(self, data: np.ndarray) -> np.ndarray:
        """Apply calibration correction for drift"""
        data_copy = data.copy()

        # Replace zeros with interpolated values
        zero_mask = (data == 0.0) | (np.abs(data) < 1e-6)
        if np.any(zero_mask):
            valid_indices = np.where(~zero_mask)[0]
            if len(valid_indices) > 1:
                data_copy[zero_mask] = np.interp(
                    np.where(zero_mask)[0], valid_indices, data[valid_indices]
                )

        # Apply drift correction
        drift_info = self.detect_drift(data_copy)
        if drift_info['drift_detected']:
            # Apply linear drift correction
            correction_factor = 1 - (drift_info['drift_percentage'] / 100)
            data_copy = data_copy * correction_factor

        return data_copy


class FlowValidator:
    """Validator and corrector for flow sensors (FS1)"""

    def __init__(self):
        self.reference_sensor = 'FS2'  # FS2 has good health score

    def cross_validate_flow(self, fs1: np.ndarray, fs2: np.ndarray) -> Dict:
        """Cross-validate flow readings between FS1 and FS2"""
        # Calculate correlation
        correlation = np.corrcoef(fs1, fs2)[0, 1] if len(fs1) > 1 else 0

        # Check for anomalies
        ratio = fs1 / (fs2 + 1e-6)  # Avoid division by zero
        ratio_mean = np.mean(ratio[ratio > 0])
        ratio_std = np.std(ratio[ratio > 0])

        return {
            'correlation': correlation,
            'ratio_mean': ratio_mean,
            'ratio_std': ratio_std,
            'anomalies': np.sum(np.abs(ratio - ratio_mean) > 3 * ratio_std)
        }

    def correct_flow_readings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Correct FS1 readings using FS2 as reference"""
        df = df.copy()

        if 'FS1' not in df.columns or 'FS2' not in df.columns:
            return df

        fs1_data = df['FS1'].values
        fs2_data = df['FS2'].values

        # Replace zeros in FS1 with values derived from FS2
        zero_mask = (fs1_data == 0.0) | (np.abs(fs1_data) < 1e-6)

        if np.any(zero_mask):
            # Calculate typical ratio between FS1 and FS2
            valid_mask = ~zero_mask & (fs2_data > 0)
            if np.any(valid_mask):
                typical_ratio = np.median(
                    fs1_data[valid_mask] / fs2_data[valid_mask])
                df.loc[zero_mask, 'FS1'] = fs2_data[zero_mask] * typical_ratio

        return df


class EfficiencyCorrector:
    """Corrector for efficiency sensor (SE1)"""

    def __init__(self):
        self.pattern_length = 8  # Based on analysis findings

    def detect_periodic_patterns(self, data: np.ndarray) -> Dict:
        """Detect periodic patterns in efficiency data"""
        # Simple pattern detection
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]

        # Find peaks that might indicate periodicity
        peaks = []
        for i in range(1, min(len(autocorr) - 1, 50)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(i)

        return {
            'periodic_detected': len(peaks) > 0,
            'potential_periods': peaks[:5],  # Top 5 potential periods
            'pattern_strength': np.max(autocorr[1:min(50, len(autocorr))]) if len(autocorr) > 1 else 0
        }

    def correct_efficiency_readings(self, data: np.ndarray) -> np.ndarray:
        """Correct efficiency readings with periodic pattern consideration"""
        data_copy = data.copy()

        # Replace zeros with interpolated values
        zero_mask = (data == 0.0) | (np.abs(data) < 1e-6)

        if np.any(zero_mask):
            # Use pattern-aware interpolation
            pattern_info = self.detect_periodic_patterns(data)

            if pattern_info['periodic_detected'] and len(pattern_info['potential_periods']) > 0:
                # Use periodic pattern for interpolation
                period = pattern_info['potential_periods'][0]
                for i in np.where(zero_mask)[0]:
                    # Look for similar position in previous periods
                    prev_positions = [
                        i - k*period for k in range(1, 5) if i - k*period >= 0]
                    valid_positions = [
                        pos for pos in prev_positions if not zero_mask[pos]]

                    if valid_positions:
                        data_copy[i] = np.mean([data[pos]
                                               for pos in valid_positions])
                    else:
                        # Fallback to linear interpolation
                        valid_indices = np.where(~zero_mask)[0]
                        if len(valid_indices) > 1:
                            data_copy[i] = np.interp(
                                i, valid_indices, data[valid_indices])
            else:
                # Standard interpolation
                valid_indices = np.where(~zero_mask)[0]
                if len(valid_indices) > 1:
                    data_copy[zero_mask] = np.interp(
                        np.where(zero_mask)[
                            0], valid_indices, data[valid_indices]
                    )

        return data_copy


def apply_sensor_corrections(df: pd.DataFrame, config_path: Optional[str] = None) -> pd.DataFrame:
    """Apply all sensor corrections based on analysis findings"""
    df_corrected = df.copy()

    # Initialize correctors
    ps4_corrector = PS4Corrector()
    calibration_corrector = CalibrationCorrector()
    flow_validator = FlowValidator()
    efficiency_corrector = EfficiencyCorrector()

    # Apply corrections based on sensor health status

    # PS4 (Critical - 66.68% zeros)
    if 'PS4' in df.columns:
        df_corrected = ps4_corrector.correct_ps4_advanced(df_corrected)

    # PS2 and PS3 (Warning - 13-14% zeros)
    for sensor in ['PS2', 'PS3']:
        if sensor in df.columns:
            corrected_data = calibration_corrector.apply_calibration_correction(
                df[sensor].values)
            df_corrected[f'{sensor}_corrected'] = corrected_data

    # FS1 (Warning - 5.65% zeros)
    if 'FS1' in df.columns and 'FS2' in df.columns:
        df_corrected = flow_validator.correct_flow_readings(df_corrected)

    # SE1 (Warning - 13.33% zeros with patterns)
    if 'SE' in df.columns:
        corrected_data = efficiency_corrector.correct_efficiency_readings(
            df['SE'].values)
        df_corrected['SE_corrected'] = corrected_data

    return df_corrected


def monitor_sensor_health(df: pd.DataFrame, config_path: Optional[str] = None) -> Dict:
    """Enhanced sensor health monitoring"""
    validator = AdvancedSensorValidator(config_path)

    health_report = {}
    sensor_columns = [col for col in df.columns if any(
        col.startswith(sensor) for sensor in ['PS', 'FS', 'TS', 'VS', 'CE', 'CP', 'SE', 'EPS']
    )]

    for sensor_col in sensor_columns:
        if sensor_col in df.columns:
            health_report[sensor_col] = validator.validate_sensor_health(
                sensor_col, df[sensor_col].values
            )

    # Apply corrections and add corrected data to report
    df_corrected = apply_sensor_corrections(df, config_path)
    health_report['corrected_data'] = df_corrected

    return health_report
