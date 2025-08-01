#!/usr/bin/env python3
"""
Enhanced PEECOM Dataset Preprocessing Script

This script processes the ZeMA hydraulic systems dataset with sensor validation and correction:
1. Load raw sensor data from dataset/cmohs/
2. Apply sensor health analysis and corrections based on dataset analysis results
3. Apply data quality filters and feature extraction
4. Split data into train/val/test sets
5. Save processed data to output/processed_data/ directory

Updated based on analysis findings:
- PS4: Critical sensor (66.68% zeros) - requires correlation-based correction
- PS2/PS3: Warning sensors (~13-14% zeros) - require calibration correction  
- FS1: Warning sensor (5.65% zeros) - requires flow validation
- SE1: Warning sensor (13.33% zeros) - requires efficiency monitoring

Usage:
    python dataset_preprocessing.py --dataset cmohs --config src/config/config.yaml
"""

from src.loader.data_loader import PEECOMDataLoader, load_all_sensor_data, load_profile
from src.loader.sensor_validation import (
    apply_sensor_corrections,
    monitor_sensor_health,
    AdvancedSensorValidator
)
from src.loader.dataset_checker import analyze_dataset
from src.loader.data_pipeline import EnhancedDataPipelineProcessor
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional, Any
import yaml

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def setup_output_directories(base_output_dir: str) -> Dict[str, str]:
    """
    Setup output directory structure.

    Args:
        base_output_dir: Base output directory

    Returns:
        Dictionary of output directories
    """
    output_dirs = {
        'base': base_output_dir,
        'logs': os.path.join(base_output_dir, 'logs'),
        'analysis': os.path.join(base_output_dir, 'analysis'),
        'figures': os.path.join(base_output_dir, 'figures'),
        'reports': os.path.join(base_output_dir, 'reports'),
        'processed_data': os.path.join(base_output_dir, 'processed_data')
    }

    # Create all directories
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return output_dirs


def setup_logging(level: str = "INFO", output_dirs: Dict[str, str] = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("peecom_preprocessing")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if output directory is provided
    if output_dirs and 'logs' in output_dirs:
        log_file = os.path.join(output_dirs['logs'], 'preprocessing.log')
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def extract_features_from_sensors(data: Dict[str, np.ndarray], config: Dict) -> pd.DataFrame:
    """
    Extract meaningful features from high-frequency sensor data.

    Args:
        data: Dictionary of sensor data {sensor_name: data_array}
        config: Configuration dictionary

    Returns:
        DataFrame with extracted features
    """
    features = {}
    feature_config = config.get('preprocessing', {}).get(
        'feature_extraction', {})

    for sensor_name, sensor_data in data.items():
        sensor_type = sensor_name[:2]  # PS, TS, FS, etc.

        if sensor_type == 'PS':  # Pressure sensors (100Hz)
            pressure_features = feature_config.get(
                'pressure_sensors', ['mean', 'std'])
            for feature in pressure_features:
                if feature == 'mean':
                    features[f'{sensor_name}_mean'] = np.mean(
                        sensor_data, axis=1)
                elif feature == 'std':
                    features[f'{sensor_name}_std'] = np.std(
                        sensor_data, axis=1)
                elif feature == 'min':
                    features[f'{sensor_name}_min'] = np.min(
                        sensor_data, axis=1)
                elif feature == 'max':
                    features[f'{sensor_name}_max'] = np.max(
                        sensor_data, axis=1)
                elif feature == 'skew':
                    from scipy.stats import skew
                    features[f'{sensor_name}_skew'] = skew(sensor_data, axis=1)
                elif feature == 'kurtosis':
                    from scipy.stats import kurtosis
                    features[f'{sensor_name}_kurtosis'] = kurtosis(
                        sensor_data, axis=1)

        elif sensor_type == 'EP':  # Motor power (100Hz)
            motor_features = feature_config.get('motor_power', ['mean', 'std'])
            for feature in motor_features:
                if feature == 'mean':
                    features[f'{sensor_name}_mean'] = np.mean(
                        sensor_data, axis=1)
                elif feature == 'std':
                    features[f'{sensor_name}_std'] = np.std(
                        sensor_data, axis=1)
                elif feature == 'peak_power':
                    features[f'{sensor_name}_peak'] = np.max(
                        sensor_data, axis=1)
                elif feature == 'energy':
                    features[f'{sensor_name}_energy'] = np.sum(
                        sensor_data, axis=1)

        elif sensor_type == 'FS':  # Flow sensors (10Hz)
            flow_features = feature_config.get('flow_sensors', ['mean', 'std'])
            for feature in flow_features:
                if feature == 'mean':
                    features[f'{sensor_name}_mean'] = np.mean(
                        sensor_data, axis=1)
                elif feature == 'std':
                    features[f'{sensor_name}_std'] = np.std(
                        sensor_data, axis=1)
                elif feature == 'flow_rate_change':
                    features[f'{sensor_name}_rate_change'] = np.std(
                        np.diff(sensor_data, axis=1), axis=1)

        elif sensor_type == 'TS':  # Temperature sensors (1Hz)
            temp_features = feature_config.get(
                'temperature_sensors', ['mean', 'std'])
            for feature in temp_features:
                if feature == 'mean':
                    features[f'{sensor_name}_mean'] = np.mean(
                        sensor_data, axis=1)
                elif feature == 'std':
                    features[f'{sensor_name}_std'] = np.std(
                        sensor_data, axis=1)
                elif feature == 'trend':
                    # Linear trend over the cycle
                    trends = []
                    for cycle in sensor_data:
                        x = np.arange(len(cycle))
                        trend = np.polyfit(x, cycle, 1)[0]
                        trends.append(trend)
                    features[f'{sensor_name}_trend'] = np.array(trends)

        elif sensor_type == 'VS':  # Vibration sensor (1Hz)
            vib_features = feature_config.get('vibration', ['rms', 'peak'])
            for feature in vib_features:
                if feature == 'rms':
                    features[f'{sensor_name}_rms'] = np.sqrt(
                        np.mean(sensor_data**2, axis=1))
                elif feature == 'peak':
                    features[f'{sensor_name}_peak'] = np.max(
                        np.abs(sensor_data), axis=1)
                elif feature == 'crest_factor':
                    rms_vals = np.sqrt(np.mean(sensor_data**2, axis=1))
                    peak_vals = np.max(np.abs(sensor_data), axis=1)
                    features[f'{sensor_name}_crest'] = peak_vals / \
                        (rms_vals + 1e-8)

        elif sensor_type in ['CE', 'CP', 'SE']:  # Efficiency sensors (1Hz)
            eff_features = feature_config.get('efficiency', ['mean', 'trend'])
            for feature in eff_features:
                if feature == 'mean':
                    features[f'{sensor_name}_mean'] = np.mean(
                        sensor_data, axis=1)
                elif feature == 'trend':
                    trends = []
                    for cycle in sensor_data:
                        x = np.arange(len(cycle))
                        trend = np.polyfit(x, cycle, 1)[0]
                        trends.append(trend)
                    features[f'{sensor_name}_trend'] = np.array(trends)

    return pd.DataFrame(features)


def load_sensor_data_by_type(dataset_dir: str) -> Dict[str, np.ndarray]:
    """
    Load sensor data organized by sensor type.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        Dictionary of sensor data arrays
    """
    sensor_data = {}

    # Define sensor files
    sensor_files = {
        'PS1': 'PS1.txt', 'PS2': 'PS2.txt', 'PS3': 'PS3.txt',
        'PS4': 'PS4.txt', 'PS5': 'PS5.txt', 'PS6': 'PS6.txt',
        'EPS1': 'EPS1.txt',
        'FS1': 'FS1.txt', 'FS2': 'FS2.txt',
        'TS1': 'TS1.txt', 'TS2': 'TS2.txt', 'TS3': 'TS3.txt', 'TS4': 'TS4.txt',
        'VS1': 'VS1.txt',
        'CE': 'CE.txt', 'CP': 'CP.txt', 'SE': 'SE.txt'
    }

    for sensor_name, filename in sensor_files.items():
        filepath = os.path.join(dataset_dir, filename)
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath, delimiter='\t',
                                   header=None).values
                sensor_data[sensor_name] = data
                print(f"Loaded {sensor_name}: {data.shape}")
            except Exception as e:
                print(f"Error loading {sensor_name}: {e}")
        else:
            print(f"Warning: {filepath} not found")

    return sensor_data


def apply_sensor_corrections(sensor_data: Dict[str, np.ndarray], config: Dict) -> Dict[str, np.ndarray]:
    """
    Apply sensor corrections, especially for PS4.

    Args:
        sensor_data: Dictionary of sensor data
        config: Configuration dictionary

    Returns:
        Corrected sensor data
    """
    corrected_data = sensor_data.copy()

    # PS4 correction using PS3 and PS5
    if config.get('preprocessing', {}).get('sensor_correction', {}).get('ps4_correction', False):
        if all(sensor in sensor_data for sensor in ['PS3', 'PS4', 'PS5']):
            ps3_data = sensor_data['PS3']
            ps4_data = sensor_data['PS4']
            ps5_data = sensor_data['PS5']

            # Find zero values in PS4
            zero_mask = ps4_data == 0

            if np.any(zero_mask):
                # Estimate PS4 values using PS3 and PS5
                estimated_values = 0.48 * ps3_data + 0.74 * ps5_data
                corrected_ps4 = ps4_data.copy()
                corrected_ps4[zero_mask] = estimated_values[zero_mask]

                corrected_data['PS4'] = corrected_ps4
                print(
                    f"Applied PS4 correction to {np.sum(zero_mask)} zero values")

    return corrected_data


def create_data_splits(features: pd.DataFrame, targets: pd.DataFrame,
                       train_split: float, val_split: float, test_split: float,
                       random_state: int = 42) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Create train/validation/test splits.

    Args:
        features: Feature DataFrame
        targets: Target DataFrame
        train_split: Training split ratio
        val_split: Validation split ratio
        test_split: Test split ratio
        random_state: Random seed

    Returns:
        Tuple of (features_splits, targets_splits)
    """
    from sklearn.model_selection import train_test_split

    # Ensure splits sum to 1
    total = train_split + val_split + test_split
    train_split /= total
    val_split /= total
    test_split /= total

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, targets, test_size=(val_split + test_split),
        # stratify on stable_flag
        random_state=random_state, stratify=targets.iloc[:, -1]
    )

    # Second split: val vs test
    val_ratio = val_split / (val_split + test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio),
        random_state=random_state, stratify=y_temp.iloc[:, -1]
    )

    features_splits = {
        'train': X_train,
        'val': X_val,
        'test': X_test
    }

    targets_splits = {
        'train': y_train,
        'val': y_val,
        'test': y_test
    }

    return features_splits, targets_splits


def save_processed_data(features_splits: Dict[str, pd.DataFrame],
                        targets_splits: Dict[str, pd.DataFrame],
                        output_dir: str, metadata: Dict) -> None:
    """
    Save processed data to files.

    Args:
        features_splits: Dictionary of feature DataFrames
        targets_splits: Dictionary of target DataFrames
        output_dir: Output directory
        metadata: Processing metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save feature data
    for split_name, data in features_splits.items():
        filepath = os.path.join(output_dir, f'X_{split_name}.csv')
        data.to_csv(filepath, index=False)

        # Also save as numpy arrays for compatibility
        np_filepath = os.path.join(output_dir, f'X_{split_name}.npy')
        np.save(np_filepath, data.values)

        print(f"Saved {split_name} features: {data.shape} -> {filepath}")

    # Save target data
    for split_name, data in targets_splits.items():
        filepath = os.path.join(output_dir, f'y_{split_name}.csv')
        data.to_csv(filepath, index=False)

        # Also save as numpy arrays for compatibility
        np_filepath = os.path.join(output_dir, f'y_{split_name}.npy')
        np.save(np_filepath, data.values)

        print(f"Saved {split_name} targets: {data.shape} -> {filepath}")

    # Save metadata
    metadata_filepath = os.path.join(output_dir, 'metadata.json')
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Saved metadata -> {metadata_filepath}")


def main():
    """Main preprocessing function"""
    parser = create_preprocessing_parser()
    args = parser.parse_args()

    # Setup output directories first
    base_output_dir = args.output_dir or "output"
    output_dirs = setup_output_directories(base_output_dir)

    # Setup logging with file output
    logger = setup_logging(args.log_level, output_dirs)
    logger.info("Starting PEECOM dataset preprocessing")

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        config = {}

    # Override config with command line arguments
    if args.dataset:
        dataset_dir = f"dataset/{args.dataset}"
    else:
        dataset_dir = config.get('data', {}).get(
            'dataset_dir', 'dataset/cmohs')

    # Use processed_data subdirectory in output
    processed_data_dir = output_dirs['processed_data']

    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Output directory: {base_output_dir}")
    logger.info(f"Processed data directory: {processed_data_dir}")

    # Validate dataset directory
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return

    # Run dataset analysis first if requested
    if args.run_analysis:
        logger.info("Running dataset analysis...")
        try:
            # Update analysis output to use output directory
            analysis_results = analyze_dataset(dataset_dir)
            # Move analysis results to output directory
            analysis_output_dir = output_dirs['analysis']
            if os.path.exists('analysis'):
                import shutil
                if os.path.exists(os.path.join('analysis', 'dataset_analysis_results.txt')):
                    shutil.copy2('analysis/dataset_analysis_results.txt',
                                 os.path.join(analysis_output_dir, 'dataset_analysis_results.txt'))
                if os.path.exists(os.path.join('analysis', 'dataset_analysis_results.csv')):
                    shutil.copy2('analysis/dataset_analysis_results.csv',
                                 os.path.join(analysis_output_dir, 'dataset_analysis_results.csv'))
                if os.path.exists(os.path.join('analysis', 'analysis_figures')):
                    shutil.copytree('analysis/analysis_figures',
                                    os.path.join(
                                        output_dirs['figures'], 'analysis_figures'),
                                    dirs_exist_ok=True)
            logger.info(
                f"Dataset analysis completed. Results saved to {analysis_output_dir}")
        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")

    # Load sensor data
    logger.info("Loading sensor data...")
    sensor_data = load_sensor_data_by_type(dataset_dir)

    if not sensor_data:
        logger.error("No sensor data loaded. Check dataset directory.")
        return

    # Apply sensor corrections
    logger.info("Applying sensor corrections...")
    sensor_data = apply_sensor_corrections(sensor_data, config)

    # Extract features
    logger.info("Extracting features from sensor data...")
    features = extract_features_from_sensors(sensor_data, config)
    logger.info(f"Extracted features shape: {features.shape}")

    # Load targets (profile.txt)
    logger.info("Loading target data...")
    profile_path = os.path.join(dataset_dir, 'profile.txt')
    if os.path.exists(profile_path):
        targets = pd.read_csv(profile_path, delimiter='\t', header=None)
        targets.columns = ['cooler_condition', 'valve_condition', 'pump_leakage',
                           'accumulator_pressure', 'stable_flag']
        logger.info(f"Target data shape: {targets.shape}")
    else:
        logger.error(f"Profile file not found: {profile_path}")
        return

    # Ensure same number of samples
    min_samples = min(len(features), len(targets))
    features = features.iloc[:min_samples]
    targets = targets.iloc[:min_samples]

    # Create data splits if requested
    if args.enforce_split:
        logger.info("Creating data splits...")
        features_splits, targets_splits = create_data_splits(
            features, targets, args.train_split, args.val_split, args.test_split
        )
    else:
        # Save as single dataset
        features_splits = {'full': features}
        targets_splits = {'full': targets}

    # Prepare metadata
    metadata = {
        'preprocessing_timestamp': datetime.now().isoformat(),
        'dataset_dir': dataset_dir,
        'output_dir': output_dir,
        'original_samples': len(features),
        'feature_columns': list(features.columns),
        'target_columns': list(targets.columns),
        'splits': {
            split: len(data) for split, data in features_splits.items()
        },
        'sensor_corrections_applied': config.get('preprocessing', {}).get('sensor_correction', {}),
        'feature_extraction_config': config.get('preprocessing', {}).get('feature_extraction', {}),
        'command_line_args': vars(args)
    }    # Save processed data
    logger.info("Saving processed data...")
    save_processed_data(features_splits, targets_splits,
                        processed_data_dir, metadata)

    # Create processing report
    report_path = os.path.join(
        output_dirs['reports'], 'preprocessing_report.txt')
    with open(report_path, 'w') as f:
        f.write("PEECOM DATASET PREPROCESSING REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Dataset Directory: {dataset_dir}\n")
        f.write(f"Output Directory: {base_output_dir}\n\n")
        f.write(f"Original samples: {len(features)}\n")
        f.write(f"Features extracted: {len(features.columns)}\n")
        f.write(f"Target variables: {len(targets.columns)}\n\n")

        if args.enforce_split:
            f.write("Data Splits:\n")
            for split_name, data in features_splits.items():
                f.write(f"  {split_name.capitalize()}: {len(data)} samples\n")

        f.write(
            f"\nSensor corrections applied: {config.get('preprocessing', {}).get('sensor_correction', {})}\n")
        f.write(
            f"Feature extraction config: {config.get('preprocessing', {}).get('feature_extraction', {})}\n")

    logger.info(f"Processing report saved to: {report_path}")
    logger.info("Preprocessing completed successfully!")
    logger.info(f"All outputs saved to: {base_output_dir}")

    # Print summary
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Original samples: {len(features)}")
    print(f"Features extracted: {len(features.columns)}")
    print(f"Target variables: {len(targets.columns)}")

    if args.enforce_split:
        for split_name, data in features_splits.items():
            print(f"{split_name.capitalize()} set: {len(data)} samples")

    print(f"Output directory: {base_output_dir}")
    print(f"Processed data: {processed_data_dir}")
    print(f"Logs: {output_dirs['logs']}")
    print(f"Reports: {output_dirs['reports']}")
    print("="*50)


if __name__ == "__main__":
    main()
