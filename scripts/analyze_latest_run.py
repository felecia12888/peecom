#!/usr/bin/env python3
"""
Analyze Latest Preprocessing Run

This script analyzes the most recent preprocessing run results,
focusing on PS4 correction effectiveness and data quality.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse


def find_latest_run(processed_data_dir: str) -> str:
    """Find the most recent preprocessing run directory"""
    subdirs = []
    for item in os.listdir(processed_data_dir):
        item_path = os.path.join(processed_data_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Extract timestamp from directory name
            try:
                parts = item.split('_')
                if len(parts) >= 3:
                    timestamp_str = parts[-1]  # Last part should be timestamp
                    timestamp = datetime.strptime(
                        timestamp_str, "%Y%m%d_%H%M%S")
                    subdirs.append((timestamp, item_path, item))
            except:
                continue

    if not subdirs:
        return None

    # Sort by timestamp and return the latest
    subdirs.sort(key=lambda x: x[0], reverse=True)
    return subdirs[0][1], subdirs[0][2]


def analyze_ps4_data(data_dir: str) -> dict:
    """Analyze PS4 sensor data quality"""
    # Check if we have raw sensor files or processed features
    ps4_files = []

    # Look for PS4 data files
    for filename in os.listdir(data_dir):
        if 'PS4' in filename and filename.endswith('.csv'):
            ps4_files.append(filename)

    if not ps4_files:
        # Try to extract PS4 features from X_full.csv
        x_file = os.path.join(data_dir, 'X_full.csv')
        if os.path.exists(x_file):
            df = pd.read_csv(x_file)
            ps4_columns = [col for col in df.columns if 'PS4' in col]
            if ps4_columns:
                ps4_features = df[ps4_columns]
                zero_count = (ps4_features == 0).sum().sum()
                total_count = ps4_features.size
                zero_percentage = (zero_count / total_count) * 100

                return {
                    'type': 'features',
                    'ps4_columns': ps4_columns,
                    'total_values': total_count,
                    'zero_values': zero_count,
                    'zero_percentage': zero_percentage,
                    'min_value': ps4_features.min().min(),
                    'max_value': ps4_features.max().max(),
                    'mean_value': ps4_features.mean().mean()
                }

    return {'type': 'none', 'error': 'No PS4 data found'}


def generate_analysis_report(run_dir: str, run_name: str) -> str:
    """Generate analysis report for the run"""

    report_lines = []
    report_lines.append("="*60)
    report_lines.append("PEECOM PREPROCESSING RUN ANALYSIS")
    report_lines.append("="*60)
    report_lines.append(f"Run: {run_name}")
    report_lines.append(f"Location: {run_dir}")
    report_lines.append(f"Analysis Time: {datetime.now().isoformat()}")
    report_lines.append("")

    # Check metadata
    metadata_file = os.path.join(run_dir, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        report_lines.append("RUN METADATA:")
        report_lines.append("-" * 40)
        report_lines.append(
            f"Timestamp: {metadata.get('preprocessing_timestamp', 'Unknown')}")
        report_lines.append(
            f"Dataset: {metadata.get('dataset_dir', 'Unknown')}")
        report_lines.append(
            f"PS4 Method: {metadata.get('ps4_correction_method', 'Unknown')}")
        report_lines.append(
            f"Original Samples: {metadata.get('original_samples', 'Unknown')}")
        report_lines.append(
            f"Features Generated: {len(metadata.get('feature_columns', []))}")
        report_lines.append(
            f"Target Variables: {len(metadata.get('target_columns', []))}")
        report_lines.append("")

    # Analyze PS4 data
    ps4_analysis = analyze_ps4_data(run_dir)
    report_lines.append("PS4 SENSOR ANALYSIS:")
    report_lines.append("-" * 40)

    if ps4_analysis['type'] == 'features':
        report_lines.append(
            f"PS4 Features Found: {len(ps4_analysis['ps4_columns'])}")
        report_lines.append(f"Total Values: {ps4_analysis['total_values']:,}")
        report_lines.append(f"Zero Values: {ps4_analysis['zero_values']:,}")
        report_lines.append(
            f"Zero Percentage: {ps4_analysis['zero_percentage']:.2f}%")
        report_lines.append(
            f"Value Range: {ps4_analysis['min_value']:.3f} - {ps4_analysis['max_value']:.3f}")
        report_lines.append(f"Mean Value: {ps4_analysis['mean_value']:.3f}")

        if ps4_analysis['zero_percentage'] < 1.0:
            report_lines.append("✅ PS4 CORRECTION: EXCELLENT (< 1% zeros)")
        elif ps4_analysis['zero_percentage'] < 5.0:
            report_lines.append("✅ PS4 CORRECTION: GOOD (< 5% zeros)")
        elif ps4_analysis['zero_percentage'] < 20.0:
            report_lines.append("⚠️ PS4 CORRECTION: MODERATE (< 20% zeros)")
        else:
            report_lines.append(
                "❌ PS4 CORRECTION: NEEDS IMPROVEMENT (> 20% zeros)")
    else:
        report_lines.append("❌ PS4 data analysis failed")
        report_lines.append(
            f"Error: {ps4_analysis.get('error', 'Unknown error')}")

    report_lines.append("")

    # Check file sizes and data integrity
    report_lines.append("DATA FILES:")
    report_lines.append("-" * 40)

    files_to_check = ['X_full.csv', 'y_full.csv', 'metadata.json']
    for filename in files_to_check:
        filepath = os.path.join(run_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            report_lines.append(f"✅ {filename}: {size_mb:.2f} MB")
        else:
            report_lines.append(f"❌ {filename}: MISSING")

    report_lines.append("")
    report_lines.append("READY FOR:")
    report_lines.append("-" * 40)
    report_lines.append("• Model training and validation")
    report_lines.append("• Performance comparison with other methods")
    report_lines.append("• Integration into PEECOM pipeline")

    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze latest preprocessing run')
    parser.add_argument('--processed-data-dir', type=str, default='output/processed_data',
                        help='Processed data directory')
    args = parser.parse_args()

    # Find latest run
    result = find_latest_run(args.processed_data_dir)
    if result is None:
        print("No preprocessing runs found in", args.processed_data_dir)
        return

    latest_run_dir, run_name = result
    print(f"Analyzing latest run: {run_name}")
    print(f"Location: {latest_run_dir}")
    print()

    # Generate analysis report
    report = generate_analysis_report(latest_run_dir, run_name)
    print(report)

    # Save report
    report_file = os.path.join(latest_run_dir, 'analysis_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\nAnalysis report saved to: {report_file}")


if __name__ == "__main__":
    main()
