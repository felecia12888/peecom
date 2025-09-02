"""
Dataset Handlers for PEECOM

Provides pluggable handlers for different dataset formats, mirroring the ModelLoader pattern.
Each handler implements a standard interface for loading, preprocessing, and feature extraction.
"""

import os
import glob
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import logging


class BaseDatasetHandler(ABC):
    """Base class for all dataset handlers"""

    def __init__(self, dataset_dir: str, config: Dict[str, Any]):
        self.dataset_dir = dataset_dir
        self.config = config
        self.logger = logging.getLogger(
            f"peecom_preprocessing.{self.__class__.__name__}")

    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw data and return (features, targets)"""
        pass

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset metadata and information"""
        pass

    def preprocess_data(self, features: pd.DataFrame, targets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply preprocessing to features and targets. Default implementation returns as-is."""
        return features, targets

    def extract_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Extract engineered features. Default implementation returns as-is."""
        return features


class TextSensorHandler(BaseDatasetHandler):
    """Handler for cmohs-style datasets with multiple .txt sensor files + profile.txt"""

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load sensor data from multiple .txt files and profile.txt"""
        # Load sensor data
        sensor_data = self._load_sensor_data_by_type()

        if not sensor_data:
            raise ValueError(f"No sensor data loaded from {self.dataset_dir}")

        # Apply sensor corrections if configured
        sensor_data = self._apply_sensor_corrections(sensor_data)

        # Extract features
        features = self._extract_features_from_sensors(sensor_data)

        # Load targets
        targets = self._load_profile()

        self.logger.info(
            f"Loaded {len(features)} samples with {len(features.columns)} features")

        return features, targets

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return sensor-based dataset information"""
        sensor_files = glob.glob(os.path.join(self.dataset_dir, "*.txt"))
        profile_exists = os.path.exists(
            os.path.join(self.dataset_dir, "profile.txt"))

        return {
            'type': 'text_sensors',
            'sensor_files': len([f for f in sensor_files if 'profile.txt' not in f]),
            'has_profile': profile_exists,
            'format': 'multiple_txt_files'
        }

    def _load_sensor_data_by_type(self) -> Dict[str, np.ndarray]:
        """Load sensor data organized by sensor type"""
        from tqdm import tqdm

        sensor_data = {}

        # Define sensor files (from cmohs config)
        sensor_files = {
            'PS1': 'PS1.txt', 'PS2': 'PS2.txt', 'PS3': 'PS3.txt',
            'PS4': 'PS4.txt', 'PS5': 'PS5.txt', 'PS6': 'PS6.txt',
            'EPS1': 'EPS1.txt',
            'FS1': 'FS1.txt', 'FS2': 'FS2.txt',
            'TS1': 'TS1.txt', 'TS2': 'TS2.txt', 'TS3': 'TS3.txt', 'TS4': 'TS4.txt',
            'VS1': 'VS1.txt',
            'CE': 'CE.txt', 'CP': 'CP.txt', 'SE': 'SE.txt'
        }

        for sensor_name, filename in tqdm(sensor_files.items(), desc="Loading sensors", unit="sensor"):
            filepath = os.path.join(self.dataset_dir, filename)
            if os.path.exists(filepath):
                try:
                    data = pd.read_csv(
                        filepath, delimiter='\t', header=None).values
                    sensor_data[sensor_name] = data
                    self.logger.debug(f"Loaded {sensor_name}: {data.shape}")
                except Exception as e:
                    self.logger.error(f"Error loading {sensor_name}: {e}")
            else:
                self.logger.warning(f"{filepath} not found")

        return sensor_data

    def _apply_sensor_corrections(self, sensor_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply sensor corrections based on config"""
        # Import PS4 correction if available
        correction_config = self.config.get(
            'preprocessing', {}).get('sensor_correction', {})

        if correction_config.get('PS4', {}).get('enabled', False) and 'PS4' in sensor_data:
            self.logger.info("Applying PS4 correction...")
            # Use simple correction as fallback
            if all(sensor in sensor_data for sensor in ['PS3', 'PS4', 'PS5']):
                ps3_data = sensor_data['PS3']
                ps4_data = sensor_data['PS4']
                ps5_data = sensor_data['PS5']

                zero_mask = ps4_data == 0
                if np.any(zero_mask):
                    estimated_values = 0.48 * ps3_data + 0.74 * ps5_data
                    corrected_ps4 = ps4_data.copy()
                    corrected_ps4[zero_mask] = estimated_values[zero_mask]
                    sensor_data['PS4'] = corrected_ps4
                    self.logger.info(
                        f"Applied PS4 correction to {np.sum(zero_mask)} zero values")

        return sensor_data

    def _extract_features_from_sensors(self, sensor_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Extract features from sensor data"""
        from tqdm import tqdm

        features = {}
        feature_config = self.config.get(
            'preprocessing', {}).get('feature_extraction', {})

        for sensor_name, sensor_values in tqdm(sensor_data.items(), desc="Extracting features", unit="sensor"):
            sensor_type = sensor_name[:2]  # PS, TS, FS, etc.

            if sensor_type == 'PS':  # Pressure sensors
                pressure_features = feature_config.get(
                    'pressure_sensors', ['mean', 'std'])
                for feature in pressure_features:
                    if feature == 'mean':
                        features[f'{sensor_name}_mean'] = np.mean(
                            sensor_values, axis=1)
                    elif feature == 'std':
                        features[f'{sensor_name}_std'] = np.std(
                            sensor_values, axis=1)
                    elif feature == 'min':
                        features[f'{sensor_name}_min'] = np.min(
                            sensor_values, axis=1)
                    elif feature == 'max':
                        features[f'{sensor_name}_max'] = np.max(
                            sensor_values, axis=1)

            elif sensor_type == 'TS':  # Temperature sensors
                temp_features = feature_config.get(
                    'temperature_sensors', ['mean', 'std'])
                for feature in temp_features:
                    if feature == 'mean':
                        features[f'{sensor_name}_mean'] = np.mean(
                            sensor_values, axis=1)
                    elif feature == 'std':
                        features[f'{sensor_name}_std'] = np.std(
                            sensor_values, axis=1)

            elif sensor_type == 'FS':  # Flow sensors
                flow_features = feature_config.get(
                    'flow_sensors', ['mean', 'std'])
                for feature in flow_features:
                    if feature == 'mean':
                        features[f'{sensor_name}_mean'] = np.mean(
                            sensor_values, axis=1)
                    elif feature == 'std':
                        features[f'{sensor_name}_std'] = np.std(
                            sensor_values, axis=1)

            elif sensor_type in ['EP']:  # Motor power
                motor_features = feature_config.get(
                    'motor_power', ['mean', 'std'])
                for feature in motor_features:
                    if feature == 'mean':
                        features[f'{sensor_name}_mean'] = np.mean(
                            sensor_values, axis=1)
                    elif feature == 'std':
                        features[f'{sensor_name}_std'] = np.std(
                            sensor_values, axis=1)

            else:  # Other sensors (VS, CE, CP, SE)
                # Default features
                features[f'{sensor_name}_mean'] = np.mean(
                    sensor_values, axis=1)
                features[f'{sensor_name}_std'] = np.std(sensor_values, axis=1)

        return pd.DataFrame(features)

    def _load_profile(self) -> pd.DataFrame:
        """Load profile.txt as targets"""
        profile_path = os.path.join(self.dataset_dir, 'profile.txt')
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Profile file not found: {profile_path}")

        targets = pd.read_csv(profile_path, delimiter='\t', header=None)
        targets.columns = ['cooler_condition', 'valve_condition', 'pump_leakage',
                           'accumulator_pressure', 'stable_flag']

        return targets


class CSVHandler(BaseDatasetHandler):
    """Handler for single CSV file datasets"""

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from a single CSV file"""
        csv_files = glob.glob(os.path.join(self.dataset_dir, "*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {self.dataset_dir}")

        # Use the first CSV file found
        csv_path = csv_files[0]
        self.logger.info(f"Loading CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        self.logger.info(f"Loaded CSV with shape: {df.shape}")

        # Split features and targets based on dataset type
        features, targets = self._split_features_targets(df)

        return features, targets

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return CSV dataset information"""
        csv_files = glob.glob(os.path.join(self.dataset_dir, "*.csv"))

        info = {
            'type': 'csv',
            'csv_files': len(csv_files),
            'format': 'single_csv'
        }

        if csv_files:
            # Get basic info about the first CSV
            df = pd.read_csv(csv_files[0], nrows=5)
            info.update({
                'columns': len(df.columns),
                'sample_columns': list(df.columns)[:10]  # First 10 columns
            })

        return info

    def _split_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into features and targets based on dataset configuration"""
        dataset_name = os.path.basename(self.dataset_dir)

        # Get dataset-specific config
        dataset_config = self.config.get('datasets', {}).get(dataset_name, {})
        data_config = dataset_config.get('data', {})
        targets_config = dataset_config.get('targets', {})

        # Use configuration to identify columns
        feature_columns = data_config.get('feature_columns', [])
        target_column = data_config.get('target_column', None)
        categorical_columns = data_config.get('categorical_columns', [])

        self.logger.info(
            f"Dataset config - Features: {feature_columns}, Target: {target_column}, Categorical: {categorical_columns}")

        if dataset_name == 'equipmentad':
            # Equipment anomaly: use configuration-based split
            features = df[feature_columns] if feature_columns else df[[
                col for col in df.columns if col not in ['faulty', 'equipment', 'location']]]

            targets = pd.DataFrame({
                'anomaly': df['faulty'],
                'equipment_type': pd.Categorical(df['equipment']).codes,
                'location': pd.Categorical(df['location']).codes
            })

        elif dataset_name == 'mlclassem':
            # ML classification energy: use config or fallback
            features = df[feature_columns] if feature_columns else df[[
                col for col in df.columns if col not in ['Equipment_Status', 'Date', 'Region', 'Equipment_Type']]]

            targets = pd.DataFrame({
                'status': pd.Categorical(df['Equipment_Status']).codes,
                'region': pd.Categorical(df['Region']).codes,
                'equipment_type': pd.Categorical(df['Equipment_Type']).codes
            })

        elif dataset_name == 'smartmd':
            # Smart maintenance: use comprehensive target columns
            features = df[feature_columns] if feature_columns else df[[
                'temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption']]

            targets = pd.DataFrame({
                'anomaly_flag': df['anomaly_flag'],
                'machine_status': df['machine_status'],
                'maintenance_required': df['maintenance_required']
            })

        elif dataset_name == 'sensord':
            # Sensor data: all sensor columns as features
            # sensor_1 through sensor_20
            sensor_cols = [f'sensor_{i}' for i in range(1, 21)]
            features = df[sensor_cols] if all(
                col in df.columns for col in sensor_cols) else df[feature_columns]

            targets = pd.DataFrame({
                'machine_status': pd.Categorical(df['machine_status']).codes
            })

        elif dataset_name == 'tsts':
            # Time series: use config-defined columns
            feature_cols = feature_columns if feature_columns else [
                'Temperature (°C)', 'Vibration (mm/s)', 'Pressure (Pa)', 'RPM']
            features = df[feature_cols] if all(
                col in df.columns for col in feature_cols) else df.select_dtypes(include=[np.number])

            targets = pd.DataFrame({
                'maintenance_required': df['Maintenance Required']
            }) if 'Maintenance Required' in df.columns else pd.DataFrame({'target': df.iloc[:, -1]})

        else:
            # Generic split: use config or assume last column is target
            if feature_columns and target_column:
                features = df[feature_columns]
                targets = pd.DataFrame({target_column: df[target_column]})
            else:
                # Fallback: numeric columns as features, last column as target
                feature_cols = df.select_dtypes(
                    include=[np.number]).columns.tolist()
                if feature_cols:
                    features = df[feature_cols[:-1]
                                  ] if len(feature_cols) > 1 else df[feature_cols]
                    target_col = feature_cols[-1] if len(
                        feature_cols) > 1 else df.columns[-1]
                else:
                    # No numeric columns, use all but last as features
                    features = df.iloc[:, :-1]
                    target_col = df.columns[-1]

                targets = pd.DataFrame({target_col: df[target_col]})

        self.logger.info(
            f"Final split - Features: {features.shape}, Targets: {targets.shape}")
        self.logger.info(f"Feature columns: {list(features.columns)}")
        self.logger.info(f"Target columns: {list(targets.columns)}")

        return features, targets


class MultiCSVHandler(BaseDatasetHandler):
    """Handler for datasets with multiple CSV files (like motorvd)"""

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and combine data from multiple CSV files"""
        csv_files = glob.glob(os.path.join(self.dataset_dir, "*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {self.dataset_dir}")

        self.logger.info(f"Found {len(csv_files)} CSV files")

        all_features = []
        all_targets = []

        # Limit to first 5 files for demo
        for i, csv_path in enumerate(csv_files[:5]):
            self.logger.info(f"Loading: {os.path.basename(csv_path)}")

            df = pd.read_csv(csv_path)

            # Extract condition from filename for motorvd
            condition = self._extract_condition_from_filename(csv_path)

            # Create features (statistical summary of time series)
            features = self._extract_time_series_features(df)
            features['file_id'] = i

            # Create targets
            targets = pd.DataFrame({
                'condition': [condition] * len(features),
                'file_id': [i] * len(features)
            })

            all_features.append(features)
            all_targets.append(targets)

        combined_features = pd.concat(all_features, ignore_index=True)
        combined_targets = pd.concat(all_targets, ignore_index=True)

        self.logger.info(
            f"Combined: Features {combined_features.shape}, Targets {combined_targets.shape}")

        return combined_features, combined_targets

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return multi-CSV dataset information"""
        csv_files = glob.glob(os.path.join(self.dataset_dir, "*.csv"))

        return {
            'type': 'multi_csv',
            'csv_files': len(csv_files),
            'format': 'multiple_csv_files'
        }

    def _extract_condition_from_filename(self, filepath: str) -> str:
        """Extract condition/label from filename"""
        filename = os.path.basename(filepath).lower()

        if 'normal' in filename or 'no_load' in filename:
            return 'normal'
        elif 'imbalanced' in filename:
            return 'imbalanced'
        elif 'fault' in filename:
            return 'fault'
        else:
            return 'unknown'

    def _extract_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features from time series data"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Create chunks (e.g., every 1000 rows)
        chunk_size = 1000
        features_list = []

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]

            if len(chunk) < chunk_size // 2:  # Skip small chunks
                continue

            chunk_features = {}

            for col in numeric_cols:
                if col != 'Timestamp':
                    chunk_features[f'{col}_mean'] = chunk[col].mean()
                    chunk_features[f'{col}_std'] = chunk[col].std()
                    chunk_features[f'{col}_min'] = chunk[col].min()
                    chunk_features[f'{col}_max'] = chunk[col].max()

            features_list.append(chunk_features)

        return pd.DataFrame(features_list)


class TextDataHandler(BaseDatasetHandler):
    """Handler for space-separated text files (like multivariatetsd)"""

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from text files"""
        train_files = glob.glob(os.path.join(self.dataset_dir, "train_*.txt"))
        test_files = glob.glob(os.path.join(self.dataset_dir, "test_*.txt"))

        if not train_files:
            raise ValueError(f"No train files found in {self.dataset_dir}")

        self.logger.info(f"Found {len(train_files)} train files")

        # Load first train file as example
        train_path = train_files[0]
        df = pd.read_csv(train_path, delimiter=' ', header=None)

        # Remove empty columns
        df = df.dropna(axis=1, how='all')

        self.logger.info(f"Loaded text data with shape: {df.shape}")

        # Split into features and targets
        # Typically first column is ID, second is cycle, rest are features
        features = df.iloc[:, 2:]  # Skip ID and cycle columns
        targets = pd.DataFrame({
            'engine_id': df.iloc[:, 0],
            'cycle': df.iloc[:, 1]
        })

        return features, targets

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return text data dataset information"""
        train_files = glob.glob(os.path.join(self.dataset_dir, "train_*.txt"))
        test_files = glob.glob(os.path.join(self.dataset_dir, "test_*.txt"))

        return {
            'type': 'text_data',
            'train_files': len(train_files),
            'test_files': len(test_files),
            'format': 'space_separated_text'
        }


# Handler Registry
DATASET_HANDLERS = {
    'text_sensors': TextSensorHandler,
    'csv': CSVHandler,
    'multi_csv': MultiCSVHandler,
    'text_data': TextDataHandler
}


def get_handler_for_dataset(dataset_dir: str, config: Dict[str, Any]) -> BaseDatasetHandler:
    """Auto-detect and return appropriate handler for a dataset"""

    # Check for profile.txt (text sensor format like cmohs)
    if os.path.exists(os.path.join(dataset_dir, 'profile.txt')):
        return TextSensorHandler(dataset_dir, config)

    # Check for multiple CSV files
    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    if len(csv_files) > 1:
        return MultiCSVHandler(dataset_dir, config)

    # Check for single CSV file
    if len(csv_files) == 1:
        return CSVHandler(dataset_dir, config)

    # Check for train/test text files
    train_files = glob.glob(os.path.join(dataset_dir, "train_*.txt"))
    if train_files:
        return TextDataHandler(dataset_dir, config)

    # Default to CSV handler
    return CSVHandler(dataset_dir, config)
