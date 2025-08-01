"""
PEECOM Data Loader Module

Contains functions for loading and basic processing of PEECOM sensor data.
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import yaml


class PEECOMDataLoader:
    """Main data loader class for PEECOM dataset"""

    def __init__(self, dataset_dir: str, config_path: Optional[str] = None):
        self.dataset_dir = dataset_dir
        self.config = self._load_config(config_path) if config_path else {}

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_sensor_data(self, delimiter: str = '\t') -> np.ndarray:
        """Load all sensor data from dataset directory"""
        return load_all_sensor_data(self.dataset_dir, delimiter)

    def load_profile(self, delimiter: str = '\t') -> np.ndarray:
        """Load profile data (labels)"""
        return load_profile(self.dataset_dir, delimiter)

    def get_train_test_split(self, X: np.ndarray, y: np.ndarray,
                             test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """Create train/test split"""
        return create_train_test_split(X, y, test_size, random_state)


def load_all_sensor_data(dataset_dir: str, delimiter: str = '\t') -> np.ndarray:
    """
    Load all sensor data files from dataset directory.

    Args:
        dataset_dir: Path to dataset directory
        delimiter: File delimiter (default: tab)

    Returns:
        Combined sensor data as numpy array
    """
    files = glob.glob(os.path.join(dataset_dir, "*.txt"))
    sensor_files = [
        f for f in files
        if os.path.basename(f).lower() not in ['profile.txt', 'description.txt', 'documentation.txt']
        and os.path.getsize(f) > 0
    ]

    if not sensor_files:
        raise ValueError(
            f"No non-empty sensor data files found in directory: {dataset_dir}")

    dfs = []
    common_length = 1000

    for f in sensor_files:
        df = pd.read_csv(f, delimiter=delimiter, header=None, dtype=np.float32)
        if not df.empty:
            resampled = resample(df.values, common_length, axis=0)
            dfs.append(pd.DataFrame(resampled))

    if not dfs:
        raise ValueError(
            "Sensor files found, but no data could be read from any file.")

    try:
        sensor_data = pd.concat(dfs, axis=1)
    except ValueError as e:
        raise ValueError(
            "No objects to concatenate. Check sensor files for valid data.") from e

    return sensor_data.values


def load_profile(dataset_dir: str, delimiter: str = '\t') -> np.ndarray:
    """
    Load profile data (labels) from dataset directory.

    Args:
        dataset_dir: Path to dataset directory
        delimiter: File delimiter (default: tab)

    Returns:
        Profile data as numpy array
    """
    profile_path = os.path.join(dataset_dir, "profile.txt")
    if not os.path.isfile(profile_path):
        raise FileNotFoundError(f"Expected profile.txt in {dataset_dir}")

    y = pd.read_csv(profile_path, delimiter=delimiter, header=None,
                    dtype={0: np.int32, 1: np.float32, 2: np.float32, 3: np.float32, 4: np.int32})
    return y.values


def preprocess_data(X: np.ndarray) -> np.ndarray:
    """
    Apply robust scaling to sensor data.

    Args:
        X: Input sensor data

    Returns:
        Scaled sensor data
    """
    scaler = RobustScaler()
    return scaler.fit_transform(X)


def create_train_test_split(X: np.ndarray, y: np.ndarray,
                            test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Create train/test split for data.

    Args:
        X: Feature data
        y: Target data
        test_size: Proportion of test data
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
