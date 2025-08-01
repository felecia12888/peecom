import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class SensorValidator:
    def __init__(self, zero_sequence_threshold: int = 100, 
                 zero_percentage_threshold: float = 15.0):
        self.zero_sequence_threshold = zero_sequence_threshold
        self.zero_percentage_threshold = zero_percentage_threshold
        self.alerts: List[str] = []

    def detect_zero_sequence(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """Detect sequences of zeros in the data"""
        zero_mask = data == 0
        zero_starts = np.where(np.diff(np.concatenate(([0], zero_mask))))[0]
        zero_ends = np.where(np.diff(np.concatenate((zero_mask, [0]))))[0]
        return [(start, end) for start, end in zip(zero_starts, zero_ends) 
                if end - start > self.zero_sequence_threshold]

    def calculate_zero_percentage(self, data: np.ndarray) -> float:
        """Calculate percentage of zero values"""
        return (data == 0).sum() / len(data) * 100

    def validate_sensor(self, sensor_id: str, data: np.ndarray) -> Dict:
        """Validate sensor data and generate alerts"""
        self.alerts = []
        zero_sequences = self.detect_zero_sequence(data)
        zero_percentage = self.calculate_zero_percentage(data)
        
        status = "Good"
        if zero_sequences:
            self.alerts.append(f"Long zero sequences detected in {sensor_id}")
            status = "Warning"
        
        if zero_percentage > self.zero_percentage_threshold:
            self.alerts.append(
                f"High zero percentage ({zero_percentage:.1f}%) in {sensor_id}")
            status = "Critical" if zero_percentage > 50 else "Warning"
        
        return {
            "status": status,
            "zero_sequences": zero_sequences,
            "zero_percentage": zero_percentage,
            "alerts": self.alerts.copy()
        }

class PS4Corrector:
    @staticmethod
    def estimate_ps4(ps3: np.ndarray, ps5: np.ndarray) -> np.ndarray:
        """Estimate PS4 values using correlation-based formula"""
        return 0.48 * ps3 + 0.74 * ps5

    @staticmethod
    def validate_estimate(actual: float, estimate: float, 
                         tolerance: float = 2.0) -> bool:
        """Validate if estimated value is within tolerance"""
        return abs(actual - estimate) <= tolerance

    def correct_ps4_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Correct PS4 values using correlation and last valid value"""
        df = df.copy()
        
        # Use last valid value
        df['PS4_fixed'] = df['PS4'].replace(0, np.nan).ffill()
        
        # Generate estimates using correlation
        estimated = self.estimate_ps4(df['PS3'], df['PS5'])
        
        # Where PS4 is zero, use estimate if within tolerance, else use last valid
        zero_mask = df['PS4'] == 0
        df.loc[zero_mask, 'PS4_fixed'] = np.where(
            self.validate_estimate(df['PS4_fixed'], estimated),
            estimated,
            df['PS4_fixed']
        )
        
        return df

def monitor_sensor_health(df: pd.DataFrame) -> Dict:
    """Monitor overall sensor health and generate reports"""
    validator = SensorValidator()
    corrector = PS4Corrector()
    
    health_report = {}
    for sensor in ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6']:
        health_report[sensor] = validator.validate_sensor(sensor, df[sensor].values)
    
    # Special handling for PS4
    if health_report['PS4']['status'] in ['Warning', 'Critical']:
        df = corrector.correct_ps4_values(df)
        health_report['PS4']['corrected_data'] = df['PS4_fixed']
    
    return health_report
