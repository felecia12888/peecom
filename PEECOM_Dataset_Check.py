import logging
import os
import gc
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Tuple, Dict

# Directory to save analysis figures
FIGURES_DIR = "analysis/analysis_figures"
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

# Define filenames for results
TXT_RESULTS_FILE = "analysis/dataset_analysis_results.txt"
CSV_RESULTS_FILE = "analysis/dataset_analysis_results.csv"

# Sensor configurations
SENSOR_CONFIGS = {
    'PS': {'count': 6, 'hz': 100, 'unit': 'bar', 'samples': 6000},  # PS1-PS6
    'EPS': {'count': 1, 'hz': 100, 'unit': 'W', 'samples': 6000},   # EPS1
    'FS': {'count': 2, 'hz': 10, 'unit': 'l/min', 'samples': 600},  # FS1-FS2
    'TS': {'count': 4, 'hz': 1, 'unit': '°C', 'samples': 60},       # TS1-TS4
    'VS': {'count': 1, 'hz': 1, 'unit': 'mm/s', 'samples': 60},     # VS1
    'CE': {'count': 1, 'hz': 1, 'unit': '%', 'samples': 60},        # CE
    'CP': {'count': 1, 'hz': 1, 'unit': 'kW', 'samples': 60},       # CP
    'SE': {'count': 1, 'hz': 1, 'unit': '%', 'samples': 60}         # SE
}

def load_sensor_data(data_dir, sensor_type, number):
    """Load individual sensor data files with better error handling"""
    # Handle sensors that don't use numbering
    if sensor_type in ['CE', 'CP', 'SE']:
        filename = f"{sensor_type}.txt"
    # Handle EPS and VS which use numbering but have special cases
    elif sensor_type in ['EPS', 'VS']:
        filename = f"{sensor_type}{number}.txt"
    # Handle all other numbered sensors (PS, FS, TS)
    else:
        filename = f"{sensor_type}{number}.txt"
    
    filepath = os.path.join(data_dir, filename)
    
    try:
        if os.path.exists(filepath):
            return pd.read_csv(filepath, delimiter='\t', header=None)
        else:
            print(f"Warning: Sensor file not found - {filepath}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error reading {filepath}: {str(e)}")
        return pd.DataFrame()

def load_profile(data_dir):
    """Load condition profile data"""
    filepath = os.path.join(data_dir, "profile.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Profile file not found: {filepath}")
    return pd.read_csv(filepath, delimiter='\t', header=None)

def analyze_zero_patterns(data):
    """Analyze patterns in zero value occurrences"""
    zero_mask = (data == 0)
    zero_indices = np.where(zero_mask)[0]
    
    if len(zero_indices) == 0:
        return {'periodic': False, 'frequency': 0, 'pattern_length': 0}
        
    # Check for periodicity
    if len(zero_indices) > 1:
        differences = np.diff(zero_indices)
        unique_diffs, counts = np.unique(differences, return_counts=True)
        
        if len(unique_diffs) == 1:
            return {
                'periodic': True,
                'frequency': unique_diffs[0],
                'pattern_length': len(zero_indices)
            }
    
    return {'periodic': False, 'frequency': 0, 'pattern_length': len(zero_indices)}

def analyze_sensor_statistics(data, sensor_config):
    """Compute basic statistics for sensor data including zero value analysis"""
    values = data.values
    
    # Optimize computation using vectorized operations
    zero_mask = (values == 0)
    zero_count = np.sum(zero_mask)
    zero_percentage = (zero_count / values.size) * 100
    
    # Process zero patterns in chunks for better performance
    chunk_size = min(1000, len(values))  # Adjust chunk size based on data size
    zero_patterns = []
    
    for i in range(0, len(values), chunk_size):
        chunk = values[i:i + chunk_size]
        patterns = [analyze_zero_patterns(row) for row in chunk]
        zero_patterns.extend(patterns)
    
    periodic_count = sum(1 for p in zero_patterns if p['periodic'])
    
    stats = {
        'mean': np.mean(values, axis=0).mean(),
        'std': np.std(values, axis=0).mean(),
        'min': np.min(values, axis=None),
        'max': np.max(values, axis=None),
        'missing': np.sum(pd.isna(data).values),
        'expected_samples': sensor_config['samples'],
        'zero_analysis': {
            'zero_count': int(zero_count),
            'zero_percentage': float(zero_percentage),
            'max_zero_sequence': max(p['pattern_length'] for p in zero_patterns),
            'avg_zero_sequence': np.mean([p['pattern_length'] for p in zero_patterns]),
            'periodic_patterns': periodic_count,
            'pattern_details': zero_patterns[0] if periodic_count > 0 else None
        }
    }
    
    # Add visualization for PS sensors
    if sensor_config.get('unit') == 'bar':  # Only for pressure sensors
        plt.figure(figsize=(10, 6))
        plt.hist(values.flatten(), bins=50, range=(0, np.percentile(values, 99)))
        plt.title(f'Distribution of Readings (including zeros)')
        plt.xlabel('Pressure (bar)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(FIGURES_DIR, f'zero_analysis_{data.name if hasattr(data, "name") else "pressure"}.png'))
        plt.close()
    return stats

def generate_sensor_health_report(analysis_results):
    """Generate enhanced health scores and maintenance recommendations"""
    health_scores = {}
    maintenance_alerts = []
    
    # Critical thresholds
    CRITICAL_ZERO_PERCENT = 15
    WARNING_ZERO_PERCENT = 10
    CRITICAL_SEQUENCE_LENGTH = 100
    WARNING_SEQUENCE_LENGTH = 50
    
    for sensor, data in analysis_results.items():
        if 'statistics' in data and 'zero_analysis' in data['statistics']:
            zero_analysis = data['statistics']['zero_analysis']
            
            # Calculate health score with enhanced criteria
            score = 100
            if zero_analysis['zero_percentage'] > 0:
                score -= min(50, zero_analysis['zero_percentage'])
            if zero_analysis['max_zero_sequence'] > 10:
                score -= min(30, zero_analysis['max_zero_sequence']/100)
            if zero_analysis.get('periodic_patterns', 0) > 0:
                score -= 20  # Increased penalty for periodic patterns
                
            status = 'Critical' if score < 50 else 'Warning' if score < 80 else 'Good'
            
            # Generate specific maintenance recommendations
            recommendations = []
            if sensor == 'PS4' and status == 'Critical':
                recommendations.append("IMMEDIATE ACTION: Calibrate and verify sensor functionality")
                recommendations.append("Check for physical blockages or electrical issues")
            elif sensor in ['PS2', 'PS3'] and status == 'Warning':
                recommendations.append("Schedule calibration within next maintenance window")
            elif sensor == 'SE1':
                recommendations.append("Monitor system efficiency trends")
            elif sensor.startswith('FS'):
                recommendations.append("Regular flow rate validation recommended")
            elif sensor.startswith('TS'):
                recommendations.append("Verify temperature correlation patterns")
            
            health_scores[sensor] = {
                'score': score,
                'status': status,
                'issues': [],
                'recommendations': recommendations,
                'maintenance_priority': 1 if status == 'Critical' else 2 if status == 'Warning' else 3
            }
            
            # Enhanced issue reporting
            if zero_analysis['zero_percentage'] > CRITICAL_ZERO_PERCENT:
                health_scores[sensor]['issues'].append(
                    f"CRITICAL: Zero readings {zero_analysis['zero_percentage']:.2f}%"
                )
            elif zero_analysis['zero_percentage'] > WARNING_ZERO_PERCENT:
                health_scores[sensor]['issues'].append(
                    f"WARNING: High zero readings {zero_analysis['zero_percentage']:.2f}%"
                )
            
            if zero_analysis['max_zero_sequence'] > CRITICAL_SEQUENCE_LENGTH:
                health_scores[sensor]['issues'].append(
                    f"CRITICAL: Extended zero sequence detected"
                )
            elif zero_analysis['max_zero_sequence'] > WARNING_SEQUENCE_LENGTH:
                health_scores[sensor]['issues'].append(
                    f"WARNING: Long zero sequence detected"
                )
                
    return health_scores

def check_sampling_consistency(data, sensor_config):
    """Verify sampling rate consistency"""
    expected_cols = sensor_config['samples']
    actual_cols = data.shape[1] if isinstance(data, pd.DataFrame) else 0
    return {
        'expected': expected_cols,
        'actual': actual_cols,
        'consistent': expected_cols == actual_cols
    }

def analyze_temporal_patterns(data, sensor_type):
    """Analyze temporal characteristics of sensor data"""
    # Compute cycle-to-cycle variations with explicit axis
    cycle_means = data.mean(axis=1)
    cycle_stds = data.std(axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(cycle_means)
    plt.title(f'{sensor_type} Cycle Mean Values')
    plt.subplot(212)
    plt.plot(cycle_stds)
    plt.title(f'{sensor_type} Cycle Standard Deviations')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'{sensor_type}_temporal_patterns.png'))
    plt.close()

def calculate_trend_direction(data: pd.DataFrame) -> str:
    """Calculate time series trend direction with frequency awareness"""
    try:
        # Use robust trend calculation for time series
        values = data.values.mean(axis=1)
        detrended = pd.DataFrame(values).diff().dropna()
        
        # Calculate trend significance using Mann-Kendall test
        trend, p_value = stats.kendalltau(np.arange(len(detrended)), detrended)
        
        if p_value > 0.05:  # Not significant
            return "stable"
        return "increasing" if trend > 0 else "decreasing"
    except Exception:
        return "unknown"

def calculate_variability_score(data: pd.DataFrame) -> float:
    """Calculate variability score with optimized computation"""
    try:
        # Vectorized computation
        values = data.values
        return float(np.std(values) / (np.mean(values) + 1e-8))  # Add small epsilon to prevent division by zero
    except Exception:
        return 0.0

def check_correlation_anomalies(data: pd.DataFrame) -> bool:
    """Optimized correlation anomaly detection"""
    try:
        # Use numpy for faster correlation calculation
        values = data.values
        corr = np.corrcoef(values.T)
        return bool(np.any(np.abs(corr) > 0.9))
    except Exception:
        return False

def analyze_sensor_trends(sensor_data):
    """Analyze time series trends with optimized computation"""
    trends = {}
    total_sensors = len(sensor_data)
    
    with tqdm(total=total_sensors, desc="Analyzing time series trends", ncols=100) as pbar:
        for sensor_name, data in sensor_data.items():
            try:
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Convert to time series format for better trend analysis
                    values = data.values
                    
                    # Calculate trends using rolling statistics
                    window_size = min(100, values.shape[0])  # Adjust based on data frequency
                    rolled_mean = pd.DataFrame(values).rolling(window=window_size, min_periods=1).mean()
                    
                    trends[sensor_name] = {
                        'trend': calculate_trend_direction(rolled_mean),
                        'variability': calculate_variability_score(rolled_mean),
                        'correlation_issues': check_correlation_anomalies(rolled_mean),
                        'frequency': SENSOR_CONFIGS[sensor_name[:2]]['hz'] if sensor_name[:2] in SENSOR_CONFIGS else 1
                    }
                else:
                    trends[sensor_name] = {
                        'trend': 'unknown',
                        'variability': 0.0,
                        'correlation_issues': False,
                        'frequency': 0
                    }
            except Exception as e:
                print(f"\nWarning: Error analyzing trends for {sensor_name}: {str(e)}")
                trends[sensor_name] = {
                    'trend': 'error',
                    'variability': 0.0,
                    'correlation_issues': False,
                    'frequency': 0
                }
            finally:
                pbar.update(1)
                pbar.set_postfix({'Current': sensor_name})
                if len(trends) % 5 == 0:
                    gc.collect()
    
    return trends

def check_sensor_correlations(all_sensor_data):
    """Analyze correlations between different sensors with robust error handling"""
    try:
        sensor_means = {}
        total_sensors = len(all_sensor_data)
        
        window_size = 10
        
        with tqdm(total=total_sensors, desc="Computing time series correlations", ncols=100) as pbar:
            for name, data in all_sensor_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    values = data.values
                    # Replace inf/nan with zeros
                    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                    rolled = pd.DataFrame(values).rolling(window=window_size, min_periods=1).mean()
                    # Ensure no constant values that would cause division by zero
                    if np.std(rolled) > 1e-10:
                        sensor_means[name] = rolled.mean(axis=1).values
                    else:
                        sensor_means[name] = np.zeros_like(rolled.mean(axis=1).values)
                else:
                    sensor_means[name] = np.array([])
                pbar.update(1)
                pbar.set_postfix({'Current': name})
        
        with tqdm(total=2, desc="Calculating temporal correlations", ncols=100) as pbar:
            # Handle empty or constant value cases
            if not sensor_means or all(len(v) == 0 for v in sensor_means.values()):
                return pd.DataFrame()
            
            means_array = np.array(list(sensor_means.values())).T
            means_array = np.nan_to_num(means_array, nan=0.0, posinf=0.0, neginf=0.0)
            pbar.update(1)
            
            # Calculate correlation with error handling
            sensor_keys = list(sensor_means.keys())
            try:
                # Add small noise to prevent division by zero
                eps = 1e-8 * np.random.randn(*means_array.shape)
                means_array += eps
                
                corr = np.corrcoef(means_array.T)
                # Replace nan/inf values with 0
                corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                
                corr_df = pd.DataFrame(
                    corr,
                    index=sensor_keys,
                    columns=sensor_keys
                )
                
                # Generate correlation heatmap
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
                plt.title('Sensor Time Series Correlations')
                plt.tight_layout()
                plt.savefig(os.path.join(FIGURES_DIR, 'sensor_correlations.png'))
                plt.close()
                pbar.update(1)
                
                return corr_df
                
            except Exception as e:
                logging.error(f"Error in correlation calculation: {str(e)}")
                return pd.DataFrame()
                
    except Exception as e:
        logging.error(f"Error in sensor correlation analysis: {str(e)}")
        return pd.DataFrame()

def analyze_condition_distribution(profile):
    """Analyze the distribution of component conditions"""
    conditions = {
        'Cooler': profile[0],
        'Valve': profile[1],
        'Pump': profile[2],
        'Accumulator': profile[3]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for (name, values), ax in zip(conditions.items(), axes.ravel()):
        sns.histplot(values, ax=ax)
        ax.set_title(f'{name} Condition Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'condition_distributions.png'))
    plt.close()
    
    return conditions

def save_analysis_results(results, dataset_name):
    """Save analysis results to files"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save to text file
    with open(TXT_RESULTS_FILE, "a") as f:
        f.write(f"\n=== Analysis Results for {dataset_name} ({timestamp}) ===\n")
        for category, data in results.items():
            f.write(f"\n{category}:\n")
            f.write(str(data))
            f.write("\n" + "-"*50 + "\n")

    # Save to CSV
    results_flat = flatten_dict(results)
    results_flat['Dataset'] = dataset_name
    results_flat['Timestamp'] = timestamp
    
    pd.DataFrame([results_flat]).to_csv(
        CSV_RESULTS_FILE, 
        mode='a', 
        header=not os.path.exists(CSV_RESULTS_FILE),
        index=False
    )

def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dictionary for CSV storage"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class SensorValidator:
    def __init__(self, zero_sequence_threshold: int = 100, 
                 zero_percentage_threshold: float = 15.0,
                 correlation_threshold: float = 0.3):
        self.zero_sequence_threshold = zero_sequence_threshold
        self.zero_percentage_threshold = zero_percentage_threshold
        self.correlation_threshold = correlation_threshold
        self.alerts = []
        self.validation_log = []

    def validate_physical_readings(self, data: np.ndarray, sensor_type: str, expected_range: tuple) -> Dict:
        """Validate sensor readings against physical constraints"""
        min_val, max_val = expected_range
        violations = np.logical_or(data < min_val, data > max_val)
        return {
            'out_of_range': np.any(violations),
            'violation_count': np.sum(violations),
            'violation_indices': np.where(violations)[0]
        }

    def check_electrical_issues(self, data: np.ndarray) -> Dict:
        """Detect potential electrical connection issues"""
        sudden_drops = np.where(np.abs(np.diff(data)) > np.std(data) * 3)[0]
        flatline_sequences = self.detect_flatline_sequences(data)
        return {
            'sudden_drops': sudden_drops,
            'flatline_sequences': flatline_sequences,
            'possible_connection_issue': len(sudden_drops) > 0 or len(flatline_sequences) > 0
        }

    def detect_flatline_sequences(self, data: np.ndarray, min_length: int = 50) -> List[Tuple[int, int]]:
        """Detect sequences where values remain constant"""
        diff = np.diff(data)
        constant_mask = np.abs(diff) < 1e-10
        starts = np.where(np.diff(np.concatenate(([False], constant_mask))))[0]
        ends = np.where(np.diff(np.concatenate((constant_mask, [False]))))[0]
        return [(start, end) for start, end in zip(starts, ends) if end - start >= min_length]

    def validate_sensor(self, sensor_id: str, data: np.ndarray, related_sensors: Dict = None) -> Dict:
        """Enhanced sensor validation with physical and electrical checks"""
        validation_result = {
            'sensor_id': sensor_id,
            'status': 'Good',
            'alerts': [],
            'corrections_needed': False,
            'validation_details': {}
        }

        # Physical range checks based on sensor type
        sensor_type = sensor_id[:2]
        if sensor_type == 'PS':
            physical_check = self.validate_physical_readings(data, sensor_type, (0, 200))  # bar
            if physical_check['out_of_range']:
                validation_result['alerts'].append(f"Physical range violations detected: {physical_check['violation_count']} instances")
                validation_result['status'] = 'Warning'

        # Electrical issues check
        electrical_check = self.check_electrical_issues(data)
        if electrical_check['possible_connection_issue']:
            validation_result['alerts'].append("Potential electrical connection issues detected")
            validation_result['status'] = 'Warning'

        # Zero sequence and percentage checks
        zero_sequences = self.detect_zero_sequence(data)
        zero_percentage = self.calculate_zero_percentage(data)
        
        if zero_sequences or zero_percentage > self.zero_percentage_threshold:
            validation_result['corrections_needed'] = True
            validation_result['status'] = 'Critical' if zero_percentage > 50 else 'Warning'

        # Correlation check with related sensors
        if related_sensors and sensor_id == 'PS4':
            correlation_check = self.check_sensor_correlation(data, related_sensors)
            if not correlation_check['correlation_valid']:
                validation_result['alerts'].append("Correlation with related sensors below threshold")
                validation_result['status'] = 'Warning'

        validation_result['validation_details'] = {
            'zero_sequences': zero_sequences,
            'zero_percentage': zero_percentage,
            'electrical_issues': electrical_check,
            'physical_checks': physical_check if sensor_type == 'PS' else None
        }

        return validation_result

    def check_sensor_correlation(self, data: np.ndarray, related_sensors: Dict) -> Dict:
        """Check correlation with related sensors"""
        correlations = {}
        correlation_valid = True
        
        for sensor_id, sensor_data in related_sensors.items():
            if len(data) == len(sensor_data):
                correlation = np.corrcoef(data, sensor_data)[0, 1]
                correlations[sensor_id] = correlation
                if abs(correlation) < self.correlation_threshold:
                    correlation_valid = False

        return {
            'correlation_valid': correlation_valid,
            'correlations': correlations
        }

class DataCorrector:
    def __init__(self):
        self.correction_log = []
        self.confidence_threshold = 0.8

    @staticmethod
    def estimate_ps4(ps3: np.ndarray, ps5: np.ndarray, weights: tuple = (0.48, 0.74)) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate PS4 values with confidence scores"""
        w1, w2 = weights
        estimated = w1 * ps3 + w2 * ps5
        
        # Calculate confidence scores based on agreement between PS3 and PS5
        confidence = 1 - np.abs(np.corrcoef(ps3, ps5)[0, 1])
        confidence_scores = np.full_like(estimated, confidence)
        
        return estimated, confidence_scores

    def apply_corrections(self, data: pd.DataFrame, validation_results: Dict) -> pd.DataFrame:
        """Apply corrections with confidence tracking"""
        corrected_data = data.copy()
        
        for sensor_id, validation in validation_results.items():
            if validation['corrections_needed']:
                if sensor_id == 'PS4':
                    if 'PS3' in data.columns and 'PS5' in data.columns:
                        estimated_values, confidence = self.estimate_ps4(
                            data['PS3'].values, 
                            data['PS5'].values
                        )
                        
                        # Apply corrections only where confidence is high
                        mask = (data[sensor_id] == 0) & (confidence >= self.confidence_threshold)
                        corrected_data.loc[mask, sensor_id] = estimated_values[mask]
                        
                        # Log corrections
                        self.correction_log.append({
                            'sensor': sensor_id,
                            'corrections_applied': np.sum(mask),
                            'average_confidence': np.mean(confidence[mask])
                        })

        return corrected_data

    def get_correction_summary(self) -> Dict:
        """Summarize all corrections applied"""
        return {
            'total_corrections': len(self.correction_log),
            'corrections_by_sensor': pd.DataFrame(self.correction_log)
        }

def analyze_sensor_health(df: pd.DataFrame) -> Dict:
    """Analyze sensor health and correct data if needed"""
    validator = SensorValidator()
    corrector = DataCorrector()
    
    # Analyze each sensor
    health_report = {}
    for column in df.columns:
        if column.startswith(('PS', 'TS', 'FS', 'VS', 'CE', 'CP', 'SE')):
            health_report[column] = validator.validate_sensor(column, df[column].values)
    
    # Correct data if needed
    if any(report['status'] != 'Good' for report in health_report.values()):
        df = corrector.correct_sensor_data(df)
        
    return {
        'health_report': health_report,
        'corrected_data': df
    }

def generate_maintenance_plan(health_scores: Dict, trends: Dict) -> Dict:
    """Generate prioritized maintenance recommendations based on health scores and time series trends"""
    maintenance_plan = {
        'immediate_actions': [],
        'scheduled_maintenance': [],
        'monitoring_required': [],
        'optimization_suggestions': []
    }

    # Process health scores and trends for each sensor
    for sensor, health in health_scores.items():
        sensor_type = sensor[:2]  # Get sensor type (PS, TS, etc.)
        sensor_trend = trends.get(sensor, {})
        
        # Immediate actions for critical issues
        if health['status'] == 'Critical':
            action = {
                'sensor': sensor,
                'priority': 'High',
                'action': f"Repair/Replace {sensor}",
                'recommendations': health.get('recommendations', []),
                'trend_info': f"Trend: {sensor_trend.get('trend', 'unknown')}"
            }
            if sensor_type == 'PS':  # Pressure sensors
                action['recommendations'].append("Check hydraulic pressure system")
            maintenance_plan['immediate_actions'].append(action)
            
        # Scheduled maintenance for warnings
        elif health['status'] == 'Warning':
            action = {
                'sensor': sensor,
                'priority': 'Medium',
                'action': f"Inspect/Calibrate {sensor}",
                'recommendations': health.get('recommendations', [])
            }
            maintenance_plan['scheduled_maintenance'].append(action)
            
        # Add monitoring requirements based on trends
        if sensor_trend.get('variability', 0) > 0.5 or sensor_trend.get('correlation_issues', False):
            maintenance_plan['monitoring_required'].append({
                'sensor': sensor,
                'action': "Increased monitoring frequency",
                'reason': "High variability or correlation anomalies detected",
                'frequency': SENSOR_CONFIGS.get(sensor_type, {}).get('hz', 1)
            })
            
        # System optimization suggestions
        if sensor_type in ['SE', 'CE', 'CP']:  # Efficiency-related sensors
            maintenance_plan['optimization_suggestions'].append({
                'component': sensor,
                'suggestion': "Review system efficiency parameters",
                'trend': sensor_trend.get('trend', 'unknown')
            })
    
    return maintenance_plan

class PS4Monitor:
    def __init__(self):
        self.metadata = {
            'corrections': [],
            'alerts': [],
            'reliability_score': 0,
            'estimation_confidence': 0
        }

    def estimate_ps4_values(self, data, ps3, ps5):
        """Estimate PS4 values using neighboring sensors"""
        zero_mask = data == 0
        estimated_values = 0.48 * ps3 + 0.74 * ps5
        confidence_threshold = 0.8
        correlation = np.corrcoef([ps3, ps5])[0,1]
        
        if correlation > confidence_threshold:
            data[zero_mask] = estimated_values[zero_mask]
            self.metadata['corrections'].append({
                'timestamp': datetime.now(),
                'points_corrected': np.sum(zero_mask),
                'confidence': correlation
            })
        
        self.metadata['estimation_confidence'] = correlation
        return data

    def validate_ps4_readings(self, data):
        """Enhanced PS4 validation with physical constraints"""
        rules = {
            'min_pressure': 0.1,
            'max_pressure': 200,
            'max_rate_change': 5.0,
            'min_non_zero_sequence': 10
        }
        
        valid_range = (data >= rules['min_pressure']) & (data <= rules['max_pressure'])
        rate_change = np.abs(np.diff(data))
        valid_rates = rate_change <= rules['max_rate_change']
        
        reliability_score = np.mean(valid_range & np.pad(valid_rates, (0,1), 'edge'))
        self.metadata['reliability_score'] = reliability_score
        
        return {
            'valid_readings': np.sum(valid_range),
            'invalid_rates': np.sum(~valid_rates),
            'reliability_score': reliability_score
        }

    def detect_ps4_anomalies(self, data, related_sensors):
        """Advanced anomaly detection for PS4"""
        anomalies = {
            'sudden_drops': [],
            'stuck_values': [],
            'correlation_breaks': []
        }
        
        window = 100
        rolling_mean = pd.Series(data).rolling(window=window).mean()
        rolling_std = pd.Series(data).rolling(window=window).std()
        std_threshold = 3
        correlation_threshold = 0.3
        
        anomalies['statistical'] = np.where(
            np.abs(data - rolling_mean) > std_threshold * rolling_std
        )[0]
        
        for sensor, values in related_sensors.items():
            correlation = np.corrcoef(data, values)[0,1]
            if abs(correlation) < correlation_threshold:
                anomalies['correlation_breaks'].append(sensor)
                self.metadata['alerts'].append({
                    'timestamp': datetime.now(),
                    'type': 'correlation_break',
                    'sensor': sensor,
                    'correlation': correlation
                })
        
        return anomalies

    def calculate_quality_metrics(self, data):
        """Calculate quality metrics for PS4 data"""
        return {
            'completeness': 1 - (np.sum(data == 0) / len(data)),
            'consistency': np.std(np.diff(data[data > 0])),
            'reliability': self.metadata['reliability_score'],
            'accuracy': self.metadata['estimation_confidence']
        }

    def generate_report(self, original_data, corrected_data, related_sensors):
        """Generate comprehensive PS4 report"""
        validation_results = self.validate_ps4_readings(corrected_data)
        anomalies = self.detect_ps4_anomalies(corrected_data, related_sensors)
        quality_metrics = self.calculate_quality_metrics(corrected_data)
        
        report = {
            'validation': validation_results,
            'anomalies': anomalies,
            'quality_metrics': quality_metrics,
            'corrections': self.metadata['corrections'],
            'alerts': self.metadata['alerts']
        }
        
        # Generate visualization
        plt.figure(figsize=(12, 6))
        plt.plot(original_data, label='Original', alpha=0.5)
        plt.plot(corrected_data, label='Corrected', alpha=0.5)
        plt.title('PS4 Data Comparison')
        plt.legend()
        plt.savefig(os.path.join(FIGURES_DIR, 'ps4_corrections.png'))
        plt.close()
        
        return report

def analyze_dataset(data_dir: str) -> Dict:
    """Enhanced main analysis function with PS4 monitoring"""
    print(f"\nAnalyzing PEECOM dataset in {data_dir}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    results = {}
    all_sensor_data = {}
    
    # Calculate total iterations for progress bar
    total_sensors = sum(config['count'] for config in SENSOR_CONFIGS.values())
    
    # Create progress bar for sensor analysis
    with tqdm(total=total_sensors, desc="Analyzing sensors", ncols=100) as pbar:
        for sensor_type, config in SENSOR_CONFIGS.items():
            for i in range(1, config['count'] + 1):
                try:
                    data = load_sensor_data(data_dir, sensor_type, i)
                    if not data.empty:
                        sensor_name = f"{sensor_type}{i}"
                        all_sensor_data[sensor_name] = data
                        results[sensor_name] = {
                            'statistics': analyze_sensor_statistics(data, config),
                            'sampling': check_sampling_consistency(data, config)
                        }
                        analyze_temporal_patterns(data, sensor_name)
                except Exception as e:
                    print(f"Warning: Error analyzing {sensor_type}{i}: {str(e)}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix({'Current': f"{sensor_type}{i}"})
    
    print("\nPerforming additional analyses...")
    
    # Profile analysis
    try:
        with tqdm(total=1, desc="Loading profile data", ncols=100) as pbar:
            profile = load_profile(data_dir)
            results['condition_analysis'] = analyze_condition_distribution(profile)
            pbar.update(1)
    except Exception as e:
        print(f"Error analyzing profile data: {str(e)}")

    # Break down correlation analysis into smaller steps with progress tracking
    print("\nStarting correlation and health analysis...")
    
    with tqdm(total=4, desc="Performing analysis steps", ncols=100) as pbar:
        results['sensor_correlations'] = check_sensor_correlations(all_sensor_data)
        pbar.update(1)
        
        results['sensor_health'] = generate_sensor_health_report(results)
        pbar.update(1)
        
        sensor_trends = analyze_sensor_trends(all_sensor_data)
        pbar.update(1)
        
        results['maintenance_recommendations'] = generate_maintenance_plan(
            results['sensor_health'],
            sensor_trends
        )
        pbar.update(1)
    # Save results with progress tracking
    with tqdm(total=1, desc="Saving results", ncols=100) as pbar:
        save_analysis_results(results, os.path.basename(data_dir))
        pbar.update(1)
    
    # Initialize PS4 monitor
    ps4_monitor = PS4Monitor()
    
    # Process PS4 specifically
    if 'PS4' in all_sensor_data:
        ps4_data = all_sensor_data['PS4'].values.flatten()
        related_sensors = {
            'PS3': all_sensor_data['PS3'].values.flatten(),
            'PS5': all_sensor_data['PS5'].values.flatten()
        }
        
        # Correct PS4 data
        corrected_ps4 = ps4_monitor.estimate_ps4_values(
            ps4_data.copy(),
            related_sensors['PS3'],
            related_sensors['PS5']
        )
        
        # Generate PS4 report
        results['ps4_analysis'] = ps4_monitor.generate_report(
            ps4_data,
            corrected_ps4,
            related_sensors
        )
        
        # Update the sensor data with corrected values
        all_sensor_data['PS4'] = pd.DataFrame(corrected_ps4)
    
    return results

# Main execution block - keep only one instance at the end of file
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        filename='dataset_check.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Use absolute path to dataset
    data_dir = r"C:\Users\28151\Desktop\Updated code files\dataset"
    try:
        results = analyze_dataset(data_dir)
        print("\nAnalysis completed successfully. Check the analysis/ directory for results.")
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        print(f"Analysis failed. Check dataset_check.log for details.")
