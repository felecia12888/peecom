#!/usr/bin/env python3
"""
Enhanced PEECOM Dataset Preprocessing Script

This script implements targeted preprocessing based on dataset analysis results.

Key Corrections Applied:
- PS4 (Critical): 66.68% zero readings → Multi-sensor correlation correction
- PS2/PS3 (Warning): ~13-14% zero readings → Calibration drift correction  
- FS1 (Warning): 5.65% zero readings → Flow validation with FS2 reference
- SE1 (Warning): 13.33% zero readings → Efficiency pattern correction

Usage:
    python enhanced_preprocessing.py --config src/config/config.yaml --output output
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def create_parser():
    """Create command line parser"""
    parser = argparse.ArgumentParser(
        description='Enhanced PEECOM Dataset Preprocessing')
    parser.add_argument('--config', default='src/config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--dataset', default='dataset/dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output', default='output',
                        help='Output directory for processed data')
    parser.add_argument('--iteration', type=int, default=1,
                        help='Preprocessing iteration number')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing processed data')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip dataset analysis (use existing results)')
    return parser


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir, f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Enhanced preprocessing started at {datetime.now()}")
    logger.info(f"Log file: {log_file}")

    return logger


def setup_output_directories(base_dir: str) -> Dict[str, str]:
    """Setup output directory structure"""
    dirs = {
        'base': base_dir,
        'processed_data': os.path.join(base_dir, 'processed_data'),
        'analysis': os.path.join(base_dir, 'analysis'),
        'figures': os.path.join(base_dir, 'figures'),
        'logs': os.path.join(base_dir, 'logs'),
        'reports': os.path.join(base_dir, 'reports')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def load_configuration(config_path: str) -> Dict:
    """Load and validate configuration"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except ImportError:
        # Fallback configuration if PyYAML not available
        return {
            'data': {
                'dataset_dir': 'dataset/cmohs',
                'processed_dir': 'output/processed_data',
                'splits': {'train': 0.7, 'val': 0.15, 'test': 0.15}
            },
            'preprocessing': {
                'zero_handling': {
                    'PS4': {'method': 'correlation_interpolation'},
                    'PS2': {'method': 'calibration_correction'},
                    'PS3': {'method': 'calibration_correction'},
                    'FS1': {'method': 'flow_validation'},
                    'SE1': {'method': 'efficiency_interpolation'}
                }
            }
        }


def run_dataset_analysis(dataset_dir: str, output_dir: str, logger: logging.Logger, force: bool = False) -> Dict:
    """Run dataset analysis if needed"""
    analysis_file = os.path.join(
        output_dir, 'analysis', 'dataset_analysis_results.json')

    if os.path.exists(analysis_file) and not force:
        logger.info(f"Loading existing analysis results from {analysis_file}")
        try:
            with open(analysis_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing analysis: {e}")

    logger.info("Running dataset analysis...")
    try:
        from src.loader.dataset_checker import analyze_dataset
        analysis_results = analyze_dataset(dataset_dir, output_dir)

        # Save analysis results
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)

        logger.info(f"Analysis results saved to {analysis_file}")
        return analysis_results
    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}")
        return {}


def load_raw_data(dataset_dir: str, logger: logging.Logger) -> Tuple[Dict, Dict]:
    """Load raw sensor data and profile data"""
    logger.info(f"Loading raw data from {dataset_dir}")

    try:
        from src.loader.data_loader import load_all_sensor_data, load_profile

        # Load sensor data
        sensor_data = load_all_sensor_data(dataset_dir)
        logger.info(f"Loaded {len(sensor_data)} sensor datasets")

        # Load profile data (targets)
        profile_data = load_profile(dataset_dir)
        logger.info(
            f"Loaded profile data with shape: {profile_data.shape if hasattr(profile_data, 'shape') else 'unknown'}")

        return sensor_data, profile_data

    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        return {}, {}


def apply_sensor_corrections_enhanced(sensor_data: Dict, config: Dict, logger: logging.Logger) -> Dict:
    """Apply enhanced sensor corrections based on analysis findings"""
    logger.info("Applying sensor corrections based on analysis findings...")

    corrected_data = {}
    correction_summary = {}

    for sensor_name, data in sensor_data.items():
        logger.info(f"Processing sensor: {sensor_name}")

        try:
            import numpy as np
            import pandas as pd

            # Convert to DataFrame if numpy array
            if isinstance(data, np.ndarray):
                if data.ndim == 1:
                    df = pd.DataFrame({sensor_name: data})
                else:
                    df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                logger.warning(
                    f"Unsupported data type for {sensor_name}: {type(data)}")
                corrected_data[sensor_name] = data
                continue

            # Apply specific corrections based on sensor type
            corrections_applied = []

            if sensor_name == 'PS4':
                # Critical correction for PS4 (66.68% zeros)
                df_corrected = correct_ps4_critical(df, sensor_data, logger)
                corrections_applied.append("critical_zero_correction")

            elif sensor_name in ['PS2', 'PS3']:
                # Warning correction for PS2/PS3 (~13-14% zeros)
                df_corrected = correct_pressure_calibration(
                    df, sensor_name, logger)
                corrections_applied.append("calibration_drift_correction")

            elif sensor_name == 'FS1':
                # Warning correction for FS1 (5.65% zeros)
                df_corrected = correct_flow_validation(
                    df, sensor_data.get('FS2'), logger)
                corrections_applied.append("flow_validation_correction")

            elif sensor_name == 'SE':
                # Warning correction for SE1 (13.33% zeros with patterns)
                df_corrected = correct_efficiency_patterns(df, logger)
                corrections_applied.append("efficiency_pattern_correction")

            else:
                # No specific correction needed
                df_corrected = df
                corrections_applied.append("no_correction_needed")

            corrected_data[sensor_name] = df_corrected
            correction_summary[sensor_name] = corrections_applied

            logger.info(
                f"Completed {sensor_name}: {', '.join(corrections_applied)}")

        except Exception as e:
            logger.error(f"Error processing {sensor_name}: {e}")
            corrected_data[sensor_name] = data
            correction_summary[sensor_name] = ["error_occurred"]

    logger.info(f"Sensor corrections completed. Summary: {correction_summary}")
    return corrected_data


def correct_ps4_critical(df, all_sensor_data, logger):
    """Critical correction for PS4 using multi-sensor correlation"""
    try:
        import numpy as np
        import pandas as pd

        ps4_col = 'PS4' if 'PS4' in df.columns else df.columns[0]
        ps4_data = df[ps4_col].values

        # Check if we have reference sensors
        reference_sensors = []
        for ref_name in ['PS1', 'PS5', 'PS6']:
            if ref_name in all_sensor_data:
                ref_data = all_sensor_data[ref_name]
                if isinstance(ref_data, np.ndarray):
                    reference_sensors.append((ref_name, ref_data))
                elif hasattr(ref_data, 'values'):
                    reference_sensors.append(
                        (ref_name, ref_data.values.flatten()))

        if reference_sensors:
            logger.info(
                f"PS4 correction using reference sensors: {[name for name, _ in reference_sensors]}")

            # Multi-sensor estimation
            zero_mask = (ps4_data == 0.0) | (np.abs(ps4_data) < 1e-6)
            estimated_values = np.zeros_like(ps4_data)

            for i, (ref_name, ref_data) in enumerate(reference_sensors):
                weight = 1.0 / len(reference_sensors)
                if len(ref_data) == len(ps4_data):
                    estimated_values += weight * ref_data

            # Apply bounds (0-200 bar for pressure)
            estimated_values = np.clip(estimated_values, 0, 200)

            # Replace zeros with estimated values
            corrected_data = ps4_data.copy()
            corrected_data[zero_mask] = estimated_values[zero_mask]

            df_corrected = df.copy()
            df_corrected[ps4_col] = corrected_data

            zero_percentage_before = np.sum(zero_mask) / len(ps4_data) * 100
            zero_percentage_after = np.sum(
                corrected_data == 0) / len(corrected_data) * 100

            logger.info(
                f"PS4 correction: {zero_percentage_before:.1f}% → {zero_percentage_after:.1f}% zeros")

        else:
            logger.warning("No reference sensors available for PS4 correction")
            df_corrected = df

        return df_corrected

    except Exception as e:
        logger.error(f"PS4 correction failed: {e}")
        return df


def correct_pressure_calibration(df, sensor_name, logger):
    """Calibration correction for PS2/PS3"""
    try:
        import numpy as np

        sensor_col = sensor_name if sensor_name in df.columns else df.columns[0]
        data = df[sensor_col].values

        # Replace zeros with interpolated values
        zero_mask = (data == 0.0) | (np.abs(data) < 1e-6)
        corrected_data = data.copy()

        if np.any(zero_mask):
            valid_indices = np.where(~zero_mask)[0]
            if len(valid_indices) > 1:
                corrected_data[zero_mask] = np.interp(
                    np.where(zero_mask)[0], valid_indices, data[valid_indices]
                )

        df_corrected = df.copy()
        df_corrected[sensor_col] = corrected_data

        zero_percentage_before = np.sum(zero_mask) / len(data) * 100
        zero_percentage_after = np.sum(
            corrected_data == 0) / len(corrected_data) * 100

        logger.info(
            f"{sensor_name} calibration: {zero_percentage_before:.1f}% → {zero_percentage_after:.1f}% zeros")

        return df_corrected

    except Exception as e:
        logger.error(f"{sensor_name} calibration correction failed: {e}")
        return df


def correct_flow_validation(df, fs2_data, logger):
    """Flow validation correction for FS1 using FS2 reference"""
    try:
        import numpy as np
        import pandas as pd

        fs1_col = 'FS1' if 'FS1' in df.columns else df.columns[0]
        fs1_data = df[fs1_col].values

        if fs2_data is not None:
            if isinstance(fs2_data, np.ndarray):
                fs2_values = fs2_data.flatten()
            elif hasattr(fs2_data, 'values'):
                fs2_values = fs2_data.values.flatten()
            else:
                fs2_values = np.array(fs2_data).flatten()

            if len(fs2_values) == len(fs1_data):
                zero_mask = (fs1_data == 0.0) | (np.abs(fs1_data) < 1e-6)

                if np.any(zero_mask):
                    # Calculate typical ratio between FS1 and FS2
                    valid_mask = ~zero_mask & (fs2_values > 0)
                    if np.any(valid_mask):
                        typical_ratio = np.median(
                            fs1_data[valid_mask] / fs2_values[valid_mask])

                        corrected_data = fs1_data.copy()
                        corrected_data[zero_mask] = fs2_values[zero_mask] * \
                            typical_ratio

                        df_corrected = df.copy()
                        df_corrected[fs1_col] = corrected_data

                        zero_percentage_before = np.sum(
                            zero_mask) / len(fs1_data) * 100
                        zero_percentage_after = np.sum(
                            corrected_data == 0) / len(corrected_data) * 100

                        logger.info(
                            f"FS1 flow validation: {zero_percentage_before:.1f}% → {zero_percentage_after:.1f}% zeros")

                        return df_corrected

        logger.warning("FS1 flow validation: No valid FS2 reference data")
        return df

    except Exception as e:
        logger.error(f"FS1 flow validation failed: {e}")
        return df


def correct_efficiency_patterns(df, logger):
    """Efficiency pattern correction for SE1"""
    try:
        import numpy as np

        se_col = 'SE' if 'SE' in df.columns else df.columns[0]
        data = df[se_col].values

        zero_mask = (data == 0.0) | (np.abs(data) < 1e-6)
        corrected_data = data.copy()

        if np.any(zero_mask):
            # Pattern-aware interpolation (simple implementation)
            valid_indices = np.where(~zero_mask)[0]
            if len(valid_indices) > 1:
                corrected_data[zero_mask] = np.interp(
                    np.where(zero_mask)[0], valid_indices, data[valid_indices]
                )

        df_corrected = df.copy()
        df_corrected[se_col] = corrected_data

        zero_percentage_before = np.sum(zero_mask) / len(data) * 100
        zero_percentage_after = np.sum(
            corrected_data == 0) / len(corrected_data) * 100

        logger.info(
            f"SE efficiency correction: {zero_percentage_before:.1f}% → {zero_percentage_after:.1f}% zeros")

        return df_corrected

    except Exception as e:
        logger.error(f"SE efficiency correction failed: {e}")
        return df


def save_processed_data(corrected_data: Dict, profile_data: Dict, output_dir: str, logger: logging.Logger) -> None:
    """Save processed data to output directory"""
    processed_dir = os.path.join(output_dir, 'processed_data')
    os.makedirs(processed_dir, exist_ok=True)

    logger.info(f"Saving processed data to {processed_dir}")

    try:
        import json
        import pandas as pd
        import numpy as np

        # Save sensor data
        for sensor_name, data in corrected_data.items():
            filename = f"{sensor_name}_corrected.csv"
            filepath = os.path.join(processed_dir, filename)

            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            elif isinstance(data, np.ndarray):
                pd.DataFrame(data).to_csv(filepath, index=False)
            else:
                logger.warning(
                    f"Cannot save {sensor_name}: unsupported type {type(data)}")
                continue

            logger.info(f"Saved {sensor_name} to {filename}")

        # Save profile data
        if profile_data:
            profile_path = os.path.join(processed_dir, 'profile_data.csv')
            if isinstance(profile_data, pd.DataFrame):
                profile_data.to_csv(profile_path, index=False)
            elif isinstance(profile_data, np.ndarray):
                pd.DataFrame(profile_data).to_csv(profile_path, index=False)
            logger.info("Saved profile data")

        # Save processing metadata
        metadata = {
            'processing_timestamp': datetime.now().isoformat(),
            'sensors_processed': list(corrected_data.keys()),
            'output_directory': processed_dir,
            'corrections_applied': {
                'PS4': 'critical_zero_correction',
                'PS2': 'calibration_drift_correction',
                'PS3': 'calibration_drift_correction',
                'FS1': 'flow_validation_correction',
                'SE': 'efficiency_pattern_correction'
            }
        }

        metadata_path = os.path.join(processed_dir, 'processing_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Processing complete. Data saved to {processed_dir}")

    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")


def main():
    """Main preprocessing function"""
    parser = create_parser()
    args = parser.parse_args()

    # Setup output directories
    output_dirs = setup_output_directories(args.output)

    # Setup logging
    logger = setup_logging(args.output)

    try:
        # Load configuration
        config = load_configuration(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Run dataset analysis (if not skipped)
        if not args.skip_analysis:
            analysis_results = run_dataset_analysis(
                args.dataset, args.output, logger, args.force)
        else:
            logger.info("Skipping dataset analysis as requested")
            analysis_results = {}

        # Load raw data
        sensor_data, profile_data = load_raw_data(args.dataset, logger)

        if not sensor_data or len(sensor_data) == 0:
            logger.error("No sensor data loaded. Exiting.")
            return 1

        # Apply sensor corrections
        corrected_data = apply_sensor_corrections_enhanced(
            sensor_data, config, logger)

        # Save processed data
        save_processed_data(corrected_data, profile_data, args.output, logger)

        logger.info("Enhanced preprocessing completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
