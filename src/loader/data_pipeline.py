"""
Enhanced Data Pipeline Processor Module

Contains the main data pipeline processing with sensor validation and correction.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import os
import yaml
import logging

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


class EnhancedDataPipelineProcessor:
    """
    Enhanced wrapper for PEECOM data processing with sensor validation and correction.
    """

    def __init__(self, config_path: str, output_dir: str = "output"):
        # Import here to avoid circular imports
        from .preprocessor import PEECOMDataProcessor
        from .sensor_validation import apply_sensor_corrections, monitor_sensor_health

        self.processor = PEECOMDataProcessor(config_path)
        self.config_path = config_path
        self.output_dir = output_dir

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self.setup_logging()

        # Import sensor validation functions
        self.apply_corrections = apply_sensor_corrections
        self.monitor_health = monitor_sensor_health

    def setup_logging(self):
        """Setup logging for pipeline processing"""
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(
                    log_dir, 'data_pipeline.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_and_correct_sensors(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Apply sensor validation and corrections based on analysis findings"""
        self.logger.info("Starting sensor validation and correction...")

        # Monitor sensor health first
        health_report = self.monitor_health(df, self.config_path)

        # Log health status
        for sensor, health in health_report.items():
            if isinstance(health, dict) and 'health_score' in health:
                self.logger.info(
                    f"Sensor {sensor}: Health Score = {health['health_score']:.1f}, Status = {health['status']}")
                if health['issues']:
                    self.logger.warning(
                        f"Sensor {sensor} issues: {', '.join(health['issues'])}")

        # Apply corrections
        df_corrected = self.apply_corrections(df, self.config_path)

        # Log correction results
        corrections_applied = []
        for col in df_corrected.columns:
            if col.endswith('_corrected'):
                corrections_applied.append(col)

        if corrections_applied:
            self.logger.info(
                f"Applied corrections to: {', '.join(corrections_applied)}")
        else:
            self.logger.info("No corrections were applied")

        return df_corrected, health_report

    def apply_data_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality filters based on configuration"""
        self.logger.info("Applying data quality filters...")

        # Get quality configuration
        quality_config = self.config.get(
            'preprocessing', {}).get('data_quality', {})
        max_zero_pct = quality_config.get('max_zero_percentage', 50)
        min_valid_sensors = quality_config.get('min_valid_sensors', 12)

        initial_rows = len(df)

        # Filter out rows with excessive zeros
        zero_percentages = (df == 0).sum(axis=1) / len(df.columns) * 100
        valid_rows = zero_percentages <= max_zero_pct
        df_filtered = df[valid_rows].copy()

        # Filter out rows with too few valid sensors
        valid_sensor_counts = (df_filtered != 0).sum(axis=1)
        sufficient_sensors = valid_sensor_counts >= min_valid_sensors
        df_filtered = df_filtered[sufficient_sensors].copy()

        final_rows = len(df_filtered)
        removed_rows = initial_rows - final_rows

        self.logger.info(
            f"Data quality filtering: Removed {removed_rows} rows ({removed_rows/initial_rows*100:.1f}%)")
        self.logger.info(f"Remaining samples: {final_rows}")

        return df_filtered

    def save_processed_data(self, X: pd.DataFrame, y: pd.DataFrame, health_report: Dict):
        """Save processed data and health report"""
        processed_dir = os.path.join(self.output_dir, 'processed_data')
        os.makedirs(processed_dir, exist_ok=True)

        # Save processed features and targets
        X.to_csv(os.path.join(processed_dir, 'features.csv'), index=False)
        y.to_csv(os.path.join(processed_dir, 'targets.csv'), index=False)

        # Save health report
        import json
        health_report_serializable = {}
        for sensor, health in health_report.items():
            if isinstance(health, dict):
                # Convert numpy arrays to lists for JSON serialization
                health_serializable = {}
                for key, value in health.items():
                    if isinstance(value, np.ndarray):
                        health_serializable[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        health_serializable[key] = float(value)
                    elif key != 'corrected_data':  # Skip large dataframes
                        health_serializable[key] = value
                health_report_serializable[sensor] = health_serializable

        with open(os.path.join(processed_dir, 'sensor_health_report.json'), 'w') as f:
            json.dump(health_report_serializable, f, indent=2)

        self.logger.info(f"Processed data saved to {processed_dir}")

    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Enhanced processing with sensor validation and correction"""
        self.logger.info("Starting enhanced data pipeline processing...")

        # Get original data from processor
        X_train, X_test, y_train, y_test = self.processor.process_all()

        # Concatenate to full datasets
        X = pd.concat([X_train, X_test], ignore_index=True)
        y = pd.concat([y_train, y_test], ignore_index=True)

        self.logger.info(f"Original data shape: X={X.shape}, y={y.shape}")

        # Apply sensor validation and corrections
        X_corrected, health_report = self.validate_and_correct_sensors(X)

        # Apply data quality filters
        X_filtered = self.apply_data_quality_filters(X_corrected)

        # Ensure y matches the filtered X
        if len(X_filtered) != len(X):
            # Keep only the rows that survived filtering
            surviving_indices = X_filtered.index
            y_filtered = y.loc[surviving_indices].reset_index(drop=True)
            X_filtered = X_filtered.reset_index(drop=True)
        else:
            y_filtered = y

        self.logger.info(
            f"Final processed data shape: X={X_filtered.shape}, y={y_filtered.shape}")

        # Save processed data
        self.save_processed_data(X_filtered, y_filtered, health_report)

        return X_filtered, y_filtered


# Backward compatibility - keep original class name as alias
class DataPipelineProcessor(EnhancedDataPipelineProcessor):
    """Backward compatibility alias"""
    pass


class ModelPredictiveController:
    """
    MPC aligned to your `mpc:` YAML section, using CVXPY.
    """

    def __init__(self, system_model, config: Dict):
        if not HAS_CVXPY:
            raise ImportError(
                "cvxpy is required for ModelPredictiveController. Install with: pip install cvxpy")

        self.model = system_model
        mpc_cfg = config['mpc']
        self.horizon = mpc_cfg['horizon']
        self.min_u, self.max_u = mpc_cfg['min_action'], mpc_cfg['max_action']
        self.input_shape = tuple(config['model']['input_shape'])

    def optimize(
        self,
        state_history: np.ndarray,
        attention_label: float = 0.0
    ) -> float:
        """
        Solve for a control sequence U[0..H-1], return U[0].

        Args:
            state_history: Historical state data
            attention_label: Attention weight (currently unused)

        Returns:
            Optimal control action for current timestep
        """
        x0 = state_history[-1].flatten()
        n = x0.size

        U = cp.Variable(self.horizon)
        X = cp.Variable((self.horizon + 1, n))

        # Cost: heavy penalty on last state, moderate on others, plus effort & smoothness
        cost = (
            5.0 * cp.sum_squares(X[:, -1]) +
            2.0 * cp.sum_squares(X[:, :n-1]) +
            0.1 * cp.sum_squares(U) +
            0.5 * cp.sum_squares(cp.diff(U))
        )

        cons = [
            X[0] == x0,
            U >= self.min_u,
            U <= self.max_u,
            cp.abs(cp.diff(U)) <= 0.2
        ]
        for t in range(self.horizon):
            cons.append(X[t+1] == X[t] + 0.1 * U[t])

        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return float(U.value[0])
        return 0.0
