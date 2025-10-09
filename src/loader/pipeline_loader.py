"""
PEECOM Pipeline Loader

Unified data loading and preprocessing pipeline that integrates:
- Dataset loading
- BLAST preprocessing
- Outlier removal
- Leakage detection
- Sensor validation
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from .dataset_loader import dataset_loader
from .blast_cleaner import BLASTCleaner
from .outlier_remover import OutlierRemover
from .leakage_filter import LeakageDetector


class PEECOMPipeline:
    """
    Complete data loading and preprocessing pipeline for PEECOM.

    Handles:
    1. Dataset loading from various sources
    2. Optional BLAST preprocessing for batch effect removal
    3. Optional outlier detection and removal
    4. Optional data leakage detection
    5. Scaling and normalization
    6. Train/test splitting
    """

    def __init__(
        self,
        dataset_name: str = 'cmohs',
        config_path: Optional[str] = None,
        output_dir: str = 'output',
        use_blast: bool = False,
        remove_outliers: bool = False,
        check_leakage: bool = False,
        scaler_type: str = 'robust',
        random_state: int = 42
    ):
        """
        Initialize PEECOM pipeline.

        Args:
            dataset_name: Name of dataset to load
            config_path: Path to configuration YAML
            output_dir: Output directory for results
            use_blast: Whether to apply BLAST preprocessing
            remove_outliers: Whether to remove outliers
            check_leakage: Whether to check for data leakage
            scaler_type: Type of scaler ('robust', 'standard', 'none')
            random_state: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self.use_blast = use_blast or self.config.get(
            'preprocessing', {}).get('use_blast', False)
        self.remove_outliers = remove_outliers or self.config.get(
            'preprocessing', {}).get('remove_outliers', False)
        self.check_leakage = check_leakage or self.config.get(
            'preprocessing', {}).get('check_leakage', False)

        # Initialize preprocessing tools
        if self.use_blast:
            blast_config = self.config.get(
                'preprocessing', {}).get('blast', {})
            self.blast_cleaner = BLASTCleaner(
                variance_retention=blast_config.get('variance_retention', 0.95)
            )

        if self.remove_outliers:
            outlier_config = self.config.get(
                'preprocessing', {}).get('outlier', {})
            self.outlier_remover = OutlierRemover(
                method=outlier_config.get('method', 'iqr'),
                threshold=outlier_config.get('threshold', None)
            )

        if self.check_leakage:
            self.leakage_detector = LeakageDetector()

        # Initialize scaler
        self.scaler_type = scaler_type or self.config.get(
            'preprocessing', {}).get('scaler', 'robust')
        self.scaler = self._get_scaler(self.scaler_type)

        # Setup logging
        self._setup_logging()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _get_scaler(self, scaler_type: str):
        """Get scaler based on type."""
        if scaler_type == 'robust':
            return RobustScaler()
        elif scaler_type == 'standard':
            return StandardScaler()
        else:
            return None

    def _setup_logging(self):
        """Setup logging for pipeline."""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('PEECOMPipeline')

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load dataset using dataset_loader.

        Returns:
            Tuple of (features, targets)
        """
        self.logger.info(f"Loading dataset: {self.dataset_name}")

        # Get dataset directory
        dataset_dir = dataset_loader.get_dataset_dir(self.dataset_name)

        # Get appropriate handler
        handler = dataset_loader.get_handler(self.dataset_name, self.config)

        # Load data
        X, y = handler.load_data()

        self.logger.info(f"Loaded data: X shape={X.shape}, y shape={y.shape}")

        return X, y

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        batch_labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply preprocessing pipeline.

        Args:
            X: Feature DataFrame
            y: Target DataFrame
            batch_labels: Optional batch labels for BLAST

        Returns:
            Tuple of (X_processed, y_processed, preprocessing_info)
        """
        preprocessing_info = {}

        # Convert to numpy if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.DataFrame) else y

        # 1. Check for data leakage
        if self.check_leakage:
            self.logger.info("Checking for data leakage...")
            leakage_results = self.leakage_detector.detect_leakage(
                X_array, y_array)
            preprocessing_info['leakage'] = leakage_results

            if leakage_results.get('has_leakage', False):
                self.logger.warning(
                    f"Potential data leakage detected: {leakage_results}")

        # 2. Remove outliers
        if self.remove_outliers:
            self.logger.info("Removing outliers...")
            X_array = self.outlier_remover.fit_transform(X_array)
            preprocessing_info['outliers_removed'] = True
            self.logger.info(f"After outlier removal: X shape={X_array.shape}")

        # 3. Apply BLAST preprocessing
        if self.use_blast and batch_labels is not None:
            self.logger.info("Applying BLAST preprocessing...")
            X_array = self.blast_cleaner.fit_transform(
                X_array, batch_labels, y_array)
            preprocessing_info['blast_applied'] = True
            self.logger.info(f"After BLAST: X shape={X_array.shape}")

        # 4. Scale features
        if self.scaler is not None:
            self.logger.info(
                f"Scaling features using {self.scaler_type} scaler...")
            X_array = self.scaler.fit_transform(X_array)
            preprocessing_info['scaler'] = self.scaler_type

        return X_array, y_array, preprocessing_info

    def create_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Dict[str, np.ndarray]:
        """
        Create train/val/test splits.

        Args:
            X: Feature array
            y: Target array
            test_size: Fraction for test set
            val_size: Fraction for validation set (from training data)

        Returns:
            Dictionary with train/val/test splits
        """
        self.logger.info(
            f"Creating splits: test_size={test_size}, val_size={val_size}")

        # First split: train+val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y if len(
                np.unique(y)) > 1 else None
        )

        # Second split: train / val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.random_state, stratify=y_temp if len(
                np.unique(y_temp)) > 1 else None
        )

        self.logger.info(
            f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

    def run_full_pipeline(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        batch_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run complete pipeline from loading to splits.

        Args:
            test_size: Test set fraction
            val_size: Validation set fraction
            batch_labels: Optional batch labels for BLAST

        Returns:
            Dictionary with splits and metadata
        """
        self.logger.info("="*60)
        self.logger.info("Running PEECOM Full Pipeline")
        self.logger.info("="*60)

        # Load data
        X, y = self.load_data()

        # Preprocess
        X_processed, y_processed, preprocessing_info = self.preprocess(
            X, y, batch_labels)

        # Create splits
        splits = self.create_splits(
            X_processed, y_processed, test_size, val_size)

        # Add metadata
        splits['preprocessing_info'] = preprocessing_info
        splits['dataset_name'] = self.dataset_name
        splits['feature_names'] = X.columns.tolist(
        ) if isinstance(X, pd.DataFrame) else None

        self.logger.info("Pipeline completed successfully!")
        self.logger.info("="*60)

        return splits


def load_processed_data(processed_data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load previously processed data from CSV files.

    Args:
        processed_data_dir: Directory containing X_full.csv and y_full.csv

    Returns:
        Tuple of (X, y) DataFrames
    """
    X = pd.read_csv(os.path.join(processed_data_dir, 'X_full.csv'))
    y = pd.read_csv(os.path.join(processed_data_dir, 'y_full.csv'))
    return X, y


__all__ = [
    'PEECOMPipeline',
    'load_processed_data',
]
