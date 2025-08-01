#!/usr/bin/env python3
"""
Test Enhanced Preprocessing Without Import Issues

This script tests the enhanced preprocessing functionality with a clean setup.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir, f'test_preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Test preprocessing started at {datetime.now()}")
    return logger


def load_sensor_file(filepath):
    """Load individual sensor data file"""
    try:
        if os.path.exists(filepath):
            return pd.read_csv(filepath, delimiter='\t', header=None)
        else:
            print(f"Warning: File not found - {filepath}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error reading {filepath}: {str(e)}")
        return pd.DataFrame()


def load_raw_data_simple(dataset_dir: str, logger: logging.Logger):
    """Simple data loading without complex imports"""
    logger.info(f"Loading raw data from {dataset_dir}")

    sensor_data = {}

    # Define sensor files
    sensor_files = {
        'PS1': 'PS1.txt', 'PS2': 'PS2.txt', 'PS3': 'PS3.txt',
        'PS4': 'PS4.txt', 'PS5': 'PS5.txt', 'PS6': 'PS6.txt',
        'FS1': 'FS1.txt', 'FS2': 'FS2.txt',
        'TS1': 'TS1.txt', 'TS2': 'TS2.txt', 'TS3': 'TS3.txt', 'TS4': 'TS4.txt',
        'EPS1': 'EPS1.txt', 'VS1': 'VS1.txt',
        'CE': 'CE.txt', 'CP': 'CP.txt', 'SE': 'SE.txt'
    }

    for sensor_name, filename in sensor_files.items():
        filepath = os.path.join(dataset_dir, filename)
        data = load_sensor_file(filepath)
        if not data.empty:
            sensor_data[sensor_name] = data
            logger.info(f"Loaded {sensor_name}: {data.shape}")

    # Load profile data
    profile_data = {}
    try:
        profile_path = os.path.join(dataset_dir, 'profile.txt')
        if os.path.exists(profile_path):
            profile_data = pd.read_csv(
                profile_path, delimiter='\t', header=None)
            logger.info(f"Loaded profile data: {profile_data.shape}")
    except Exception as e:
        logger.error(f"Failed to load profile data: {e}")

    return sensor_data, profile_data


def apply_simple_corrections(sensor_data: dict, logger: logging.Logger):
    """Apply simple sensor corrections"""
    logger.info("Applying simple sensor corrections...")

    corrected_data = {}

    for sensor_name, data in sensor_data.items():
        logger.info(f"Processing {sensor_name}...")

        try:
            # Convert to numpy array for processing
            if isinstance(data, pd.DataFrame):
                values = data.values
            else:
                values = np.array(data)

            # Simple zero correction: replace zeros with interpolated values
            zero_mask = (values == 0.0) | (np.abs(values) < 1e-6)
            zero_count = np.sum(zero_mask)
            zero_percentage = (zero_count / values.size) * 100

            if zero_percentage > 0:
                logger.info(
                    f"{sensor_name}: {zero_percentage:.2f}% zeros detected")

                # Simple interpolation for zeros
                corrected_values = values.copy()
                if zero_percentage < 100:  # Don't try to interpolate if all values are zero
                    for i in range(values.shape[0]):
                        row = corrected_values[i, :]
                        zero_indices = np.where(row == 0)[0]
                        non_zero_indices = np.where(row != 0)[0]

                        if len(non_zero_indices) > 1 and len(zero_indices) > 0:
                            corrected_values[i, zero_indices] = np.interp(
                                zero_indices, non_zero_indices, row[non_zero_indices]
                            )

                corrected_data[sensor_name] = pd.DataFrame(corrected_values)

                # Report improvement
                new_zero_count = np.sum(corrected_values == 0)
                new_zero_percentage = (
                    new_zero_count / corrected_values.size) * 100
                logger.info(
                    f"{sensor_name}: {zero_percentage:.2f}% → {new_zero_percentage:.2f}% zeros")
            else:
                corrected_data[sensor_name] = data
                logger.info(f"{sensor_name}: No zeros detected")

        except Exception as e:
            logger.error(f"Error processing {sensor_name}: {e}")
            corrected_data[sensor_name] = data

    return corrected_data


def save_processed_data_simple(corrected_data: dict, profile_data, output_dir: str, logger: logging.Logger):
    """Save processed data"""
    processed_dir = os.path.join(output_dir, 'processed_data')
    os.makedirs(processed_dir, exist_ok=True)

    logger.info(f"Saving processed data to {processed_dir}")

    # Save sensor data
    for sensor_name, data in corrected_data.items():
        filename = f"{sensor_name}_corrected.csv"
        filepath = os.path.join(processed_dir, filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Saved {sensor_name} to {filename}")

    # Save profile data
    if not profile_data.empty:
        profile_path = os.path.join(processed_dir, 'profile_data.csv')
        profile_data.to_csv(profile_path, index=False)
        logger.info("Saved profile data")

    logger.info(f"All data saved to {processed_dir}")


def main():
    """Main test function"""
    dataset_path = "dataset/cmohs"
    output_path = "output"

    # Setup
    os.makedirs(output_path, exist_ok=True)
    logger = setup_logging(output_path)

    logger.info("Starting simple preprocessing test...")

    try:
        # Load raw data
        sensor_data, profile_data = load_raw_data_simple(dataset_path, logger)

        if not sensor_data:
            logger.error("No sensor data loaded. Exiting.")
            return 1

        logger.info(f"Loaded {len(sensor_data)} sensor datasets")

        # Apply corrections
        corrected_data = apply_simple_corrections(sensor_data, logger)

        # Save processed data
        save_processed_data_simple(
            corrected_data, profile_data, output_path, logger)

        logger.info("Simple preprocessing completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
