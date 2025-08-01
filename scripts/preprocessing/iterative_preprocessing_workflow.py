#!/usr/bin/env python3
"""
Iterative Preprocessing Workflow for PEECOM Project

This script implements a feedback loop:
1. Analyze raw dataset → identify issues
2. Apply targeted preprocessing corrections
3. Re-analyze processed data → verify improvements
4. Update preprocessing algorithms if needed
5. Iterate until data quality is satisfactory for training

Author: PEECOM Project
"""

import os
import sys
import argparse
import logging
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "iterative_workflow.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def run_analysis(dataset_path, output_dir, analysis_name, logger):
    """Run dataset analysis and return results path"""
    logger.info(f"Running {analysis_name} analysis...")

    cmd = [
        sys.executable, "src/loader/dataset_checker.py",
        "--dataset", dataset_path,
        "--output", output_dir,
        "--analysis-name", analysis_name
    ]

    try:
        # Add progress indication
        with tqdm(total=100, desc=f"Analysis: {analysis_name}", ncols=100, leave=True) as pbar:
            pbar.set_description(f"Starting {analysis_name} analysis")
            pbar.update(10)

            # Start the process
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, cwd=os.getcwd(), bufsize=1, universal_newlines=True)

            # Monitor progress
            progress_steps = ["Loading sensors", "Computing statistics", "Analyzing trends",
                              "Checking correlations", "Generating reports"]
            step_size = 80 // len(progress_steps)

            for i, step in enumerate(progress_steps):
                pbar.set_description(f"{analysis_name}: {step}")
                time.sleep(0.5)  # Brief pause to show progress
                pbar.update(step_size)

            # Wait for completion
            stdout, stderr = process.communicate()
            pbar.update(10)
            pbar.set_description(f"Completed {analysis_name} analysis")

            if process.returncode != 0:
                logger.error(f"Analysis failed: {stderr}")
                return None

        logger.info(f"Analysis completed successfully")
        return Path(output_dir) / "analysis" / f"dataset_analysis_results_{analysis_name}.csv"

    except Exception as e:
        logger.error(f"Failed to run analysis: {e}")
        return None


def run_preprocessing(dataset_path, output_dir, iteration, logger):
    """Run enhanced preprocessing"""
    logger.info(f"Running preprocessing iteration {iteration}...")

    cmd = [
        sys.executable, "enhanced_preprocessing.py",
        "--dataset", dataset_path,
        "--output", output_dir,
        "--iteration", str(iteration)
    ]

    try:
        # Add progress indication
        with tqdm(total=100, desc=f"Preprocessing Iter {iteration}", ncols=100, leave=True) as pbar:
            pbar.set_description(
                f"Starting preprocessing iteration {iteration}")
            pbar.update(10)

            # Start the process
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, cwd=os.getcwd(), bufsize=1, universal_newlines=True)

            # Monitor progress
            progress_steps = ["Loading data", "PS4 correction", "PS2/PS3 calibration",
                              "FS1 flow validation", "SE1 efficiency", "Saving results"]
            step_size = 80 // len(progress_steps)

            for i, step in enumerate(progress_steps):
                pbar.set_description(f"Iter {iteration}: {step}")
                time.sleep(0.3)  # Brief pause to show progress
                pbar.update(step_size)

            # Wait for completion
            stdout, stderr = process.communicate()
            pbar.update(10)
            pbar.set_description(
                f"Completed preprocessing iteration {iteration}")

            if process.returncode != 0:
                logger.error(f"Preprocessing failed: {stderr}")
                return False

        logger.info(f"Preprocessing iteration {iteration} completed")
        return True

    except Exception as e:
        logger.error(f"Failed to run preprocessing: {e}")
        return False


def compare_analysis_results(before_path, after_path, logger):
    """Compare analysis results before and after preprocessing"""
    try:
        with tqdm(total=100, desc="Comparing Results", ncols=100, leave=True) as pbar:
            pbar.set_description("Loading analysis files")
            pbar.update(20)

            # Read both CSV files
            before_df = pd.read_csv(before_path)
            after_df = pd.read_csv(after_path)
            pbar.update(30)

            pbar.set_description("Computing improvements")
            # Calculate improvements
            improvements = {}

            # Compare critical issues
            before_critical = before_df[before_df['Issue_Level']
                                        == 'CRITICAL'].shape[0]
            after_critical = after_df[after_df['Issue_Level']
                                      == 'CRITICAL'].shape[0]

            before_warning = before_df[before_df['Issue_Level']
                                       == 'WARNING'].shape[0]
            after_warning = after_df[after_df['Issue_Level']
                                     == 'WARNING'].shape[0]
            pbar.update(30)

            improvements['critical_reduction'] = before_critical - \
                after_critical
            improvements['warning_reduction'] = before_warning - after_warning
            improvements['total_issues_before'] = before_critical + \
                before_warning
            improvements['total_issues_after'] = after_critical + after_warning
            improvements['improvement_percentage'] = (
                (improvements['total_issues_before'] - improvements['total_issues_after']) /
                max(improvements['total_issues_before'], 1) * 100
            )
            pbar.update(20)

            pbar.set_description("Analysis comparison completed")

            logger.info(f"Analysis Comparison:")
            logger.info(f"  Critical issues: {before_critical} → {after_critical} "
                        f"(reduction: {improvements['critical_reduction']})")
            logger.info(f"  Warning issues: {before_warning} → {after_warning} "
                        f"(reduction: {improvements['warning_reduction']})")
            logger.info(
                f"  Overall improvement: {improvements['improvement_percentage']:.1f}%")

            return improvements

    except Exception as e:
        logger.error(f"Failed to compare results: {e}")
        return None


def update_preprocessing_config(improvements, iteration, config_path, logger):
    """Update preprocessing configuration based on analysis results"""
    try:
        # Load current config
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update preprocessing parameters based on results
        if improvements['improvement_percentage'] < 30:  # Less than 30% improvement
            logger.info(
                "Insufficient improvement, enhancing preprocessing parameters...")

            # Increase correction aggressiveness
            preprocessing = config.get('preprocessing', {})
            corrections = preprocessing.get('corrections', {})

            # Enhance PS4 correction
            ps4_config = corrections.get('PS4', {})
            ps4_config['smoothing_window'] = min(
                ps4_config.get('smoothing_window', 5) + 2, 15)
            ps4_config['outlier_threshold'] = max(
                ps4_config.get('outlier_threshold', 3.0) - 0.2, 1.5)
            corrections['PS4'] = ps4_config

            # Enhance PS2/PS3 correction
            pressure_config = corrections.get('pressure_sensors', {})
            pressure_config['correlation_threshold'] = max(
                pressure_config.get('correlation_threshold', 0.7) - 0.05, 0.5
            )
            corrections['pressure_sensors'] = pressure_config

            # Enhance FS1 correction
            fs1_config = corrections.get('FS1', {})
            fs1_config['validation_tolerance'] = max(
                fs1_config.get('validation_tolerance', 0.1) - 0.01, 0.05
            )
            corrections['FS1'] = fs1_config

            # Enhance SE1 correction
            se1_config = corrections.get('SE1', {})
            se1_config['efficiency_threshold'] = max(
                se1_config.get('efficiency_threshold', 0.8) - 0.05, 0.6
            )
            corrections['SE1'] = se1_config

            config['preprocessing']['corrections'] = corrections

            # Save updated config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            logger.info(
                "Updated preprocessing configuration for next iteration")
            return True

        else:
            logger.info(
                "Sufficient improvement achieved, no config update needed")
            return False

    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        return False


def should_continue_iteration(improvements, iteration, max_iterations):
    """Determine if we should continue iterating"""
    if iteration >= max_iterations:
        return False, "Maximum iterations reached"

    if improvements['total_issues_after'] == 0:
        return False, "All issues resolved"

    if improvements['improvement_percentage'] >= 80:
        return False, "Significant improvement achieved"

    if iteration > 1 and improvements['improvement_percentage'] < 5:
        return False, "Minimal improvement, diminishing returns"

    return True, "Continue iterating"


def main():
    parser = argparse.ArgumentParser(
        description="Iterative Preprocessing Workflow")
    parser.add_argument("--dataset", default="dataset/dataset",
                        help="Dataset directory path")
    parser.add_argument("--output", default="output",
                        help="Output directory path")
    parser.add_argument("--config", default="src/config/config.yaml",
                        help="Configuration file path")
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Maximum number of iterations")
    parser.add_argument("--min-improvement", type=float, default=5.0,
                        help="Minimum improvement percentage to continue")

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output)

    logger.info("=" * 60)
    logger.info("Starting Iterative Preprocessing Workflow")
    logger.info("=" * 60)

    # Initial analysis on raw data
    logger.info("Phase 1: Initial Raw Data Analysis")
    raw_analysis_path = run_analysis(
        args.dataset, args.output, "raw_initial", logger)
    if not raw_analysis_path or not raw_analysis_path.exists():
        logger.error("Failed to complete initial analysis")
        return 1

    # Iterative preprocessing and analysis
    iteration = 1
    previous_improvement = 0
    current_dataset_path = args.dataset
    processed_data_path = output_dir / "processed_data"  # Initialize here

    # Calculate total workflow steps for progress tracking
    # Each iteration has 4 phases, plus initial and final
    total_workflow_steps = args.max_iterations * 4 + 2
    workflow_progress = tqdm(total=total_workflow_steps,
                             desc="Overall Workflow Progress", ncols=100, leave=True)
    workflow_progress.update(1)  # Initial analysis completed

    while iteration <= args.max_iterations:
        logger.info(f"\n{'='*50}")
        logger.info(f"ITERATION {iteration}/{args.max_iterations}")
        logger.info(f"{'='*50}")

        workflow_progress.set_description(
            f"Iteration {iteration}/{args.max_iterations}")

        # Apply preprocessing
        logger.info(f"Phase 2.{iteration}: Apply Enhanced Preprocessing")
        if not run_preprocessing(current_dataset_path, args.output, iteration, logger):
            logger.error(f"Preprocessing failed in iteration {iteration}")
            break
        workflow_progress.update(1)

        # Update dataset path to processed data for next iteration
        processed_data_path = output_dir / "processed_data"

        # Re-analyze processed data
        logger.info(f"Phase 3.{iteration}: Analyze Processed Data")
        processed_analysis_path = run_analysis(
            str(
                processed_data_path), args.output, f"processed_iter{iteration}", logger
        )

        if not processed_analysis_path or not processed_analysis_path.exists():
            logger.error(f"Analysis failed in iteration {iteration}")
            break
        workflow_progress.update(1)

        # Compare results
        logger.info(f"Phase 4.{iteration}: Compare Results")
        if iteration == 1:
            # Compare with initial raw analysis
            comparison_base = raw_analysis_path
        else:
            # Compare with previous iteration
            comparison_base = output_dir / "analysis" / \
                f"dataset_analysis_results_processed_iter{iteration-1}.csv"

        improvements = compare_analysis_results(
            comparison_base, processed_analysis_path, logger)
        if not improvements:
            logger.error("Failed to compare analysis results")
            break

        # Check if we should continue
        should_continue, reason = should_continue_iteration(
            improvements, iteration, args.max_iterations)

        logger.info(f"Iteration {iteration} Summary:")
        logger.info(
            f"  Improvement: {improvements['improvement_percentage']:.1f}%")
        logger.info(
            f"  Critical issues remaining: {improvements['total_issues_after'] - (improvements['total_issues_after'] - improvements.get('critical_issues_after', 0))}")
        logger.info(f"  Continue: {should_continue} ({reason})")

        if not should_continue:
            logger.info(f"Stopping iterations: {reason}")
            break

        # Update configuration for next iteration if improvement is insufficient
        if improvements['improvement_percentage'] < 50:  # Less than 50% improvement
            logger.info(f"Phase 5.{iteration}: Update Configuration")
            update_preprocessing_config(
                improvements, iteration, args.config, logger)

        # Prepare for next iteration
        current_dataset_path = str(processed_data_path)
        iteration += 1

    # Close workflow progress bar
    workflow_progress.close()

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("WORKFLOW COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Total iterations completed: {iteration - 1}")

    # Save final summary
    summary_path = output_dir / "analysis" / "workflow_summary.json"
    summary = {
        "total_iterations": iteration - 1,
        "final_improvement_percentage": improvements.get('improvement_percentage', 0) if 'improvements' in locals() else 0,
        "remaining_issues": improvements.get('total_issues_after', 0) if 'improvements' in locals() else 0,
        "workflow_status": "completed",
        "processed_data_path": str(processed_data_path) if 'processed_data_path' in locals() else "not_available",
        "final_analysis_path": str(processed_analysis_path) if 'processed_analysis_path' in locals() else None
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Workflow summary saved to: {summary_path}")
    logger.info(f"Processed data available at: {processed_data_path}")
    logger.info("Ready for model training!")

    return 0


if __name__ == "__main__":
    exit(main())
