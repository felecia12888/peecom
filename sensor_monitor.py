import pandas as pd
import numpy as np
from datetime import datetime
from sensor_validation import SensorValidator, PS4Corrector, monitor_sensor_health

class SensorMonitor:
    def __init__(self):
        self.validator = SensorValidator()
        self.corrector = PS4Corrector()
        self.alerts = []
        self.daily_stats = {}

    def log_alert(self, message: str):
        """Log alerts with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.alerts.append(f"{timestamp}: {message}")

    def analyze_sensor_data(self, df: pd.DataFrame) -> dict:
        """Analyze sensor data and generate health report"""
        health_report = monitor_sensor_health(df)
        
        # Process alerts
        for sensor, report in health_report.items():
            for alert in report.get('alerts', []):
                self.log_alert(alert)
        
        # Calculate daily statistics
        self.daily_stats = {
            'zero_percentages': {
                sensor: report['zero_percentage']
                for sensor, report in health_report.items()
            },
            'alerts_count': len(self.alerts),
            'sensors_needing_attention': [
                sensor for sensor, report in health_report.items()
                if report['status'] != 'Good'
            ]
        }
        
        return {
            'health_report': health_report,
            'daily_stats': self.daily_stats,
            'alerts': self.alerts
        }

    def validate_ps4_readings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and correct PS4 readings"""
        df = df.copy()
        
        # Calculate expected PS4 values
        expected_ps4 = self.corrector.estimate_ps4(df['PS3'], df['PS5'])
        
        # Detect anomalies
        anomalies = abs(df['PS4'] - expected_ps4) > 2.0
        if anomalies.any():
            self.log_alert(f"PS4 anomalies detected at indices: {np.where(anomalies)[0]}")
            
        # Correct anomalous readings
        df.loc[anomalies, 'PS4'] = expected_ps4[anomalies]
        
        return df
