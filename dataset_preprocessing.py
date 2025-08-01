# clean_dataset.py → peecoom_data.py
import pandas as pd
import numpy as np
from pathlib import Path

import yaml


class PEECOMDataProcessor:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.sensor_config = self._define_sensors()

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _define_sensors(self):
        return [
            # 1Hz Sensors
            {'name': 'TS1', 'hz': 1, 'cols': 60}, {'name': 'TS2', 'hz': 1, 'cols': 60},
            # ... (full sensor config from your existing code)
        ]

    def _inject_faults(self, data):
        """PEECOM-compliant fault injection with time-step labels"""
        fault_labels = np.zeros((len(data), self.config['max_time_steps']))
        for idx in data.index:
            if np.random.rand() < self.config['fault_probability']:
                # Fault injection logic with precise timestamps
                # Update fault_labels with 1s at faulty time steps
             return data, fault_labels

    def _extract_peecoom_features(self, data):
        """Extract 61 features (mean/std/FFT + energy metrics)"""
        features = pd.DataFrame()
        # Feature extraction logic from your existing code
        features['energy_usage'] = data['EPS1_mean'] + data['CP_mean']
        return features

    def process(self):
        """End-to-end PEECOM data pipeline"""
        # Load raw sensor data
        raw_data = pd.concat([pd.read_csv(f) for f in Path(self.config['data_path']).glob('cycle_*.txt')])
        
        # Inject faults & create labels
        data, temporal_labels = self._inject_faults(raw_data)
        
        # Feature engineering
        features = self._extract_peecoom_features(data)
        
        # Temporal validation split
        split_idx = int(len(features) * (1 - self.config['test_size']))
        return (
            features[:split_idx], features[split_idx:],
            temporal_labels[:split_idx], temporal_labels[split_idx:]
        )

if __name__ == '__main__':
    # Use absolute path to match your dataset
    dataset_dir = r"C:\Users\28151\Desktop\Updated code files\dataset"