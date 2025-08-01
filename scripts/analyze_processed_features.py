#!/usr/bin/env python3
"""
Analyze Processed Features Results

This script analyzes the processed features (X_full.csv, y_full.csv) and metadata
to provide comprehensive analysis of the preprocessing results.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Analyze processed features data')
    parser.add_argument('data_dir', type=str,
                        help='Directory containing processed data (e.g., output/processed_data/cmohs)')
    parser.add_argument('--compare-original', action='store_true',
                        help='Compare with original analysis results')
    return parser


def load_metadata(data_dir):
    """Load metadata.json if available"""
    metadata_path = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def analyze_features(features_df, metadata=None):
    """Analyze processed features"""
    print("="*60)
    print("PROCESSED FEATURES ANALYSIS")
    print("="*60)

    print(f"Features shape: {features_df.shape}")
    print(f"Total samples: {features_df.shape[0]}")
    print(f"Total features: {features_df.shape[1]}")

    # Analyze each feature column
    feature_stats = {}
    sensor_groups = {}

    for col in features_df.columns:
        # Extract sensor name (e.g., PS4_mean -> PS4)
        if '_' in col:
            sensor_name = col.split('_')[0]
            feature_type = col.split('_')[1]
        else:
            sensor_name = col
            feature_type = 'raw'

        if sensor_name not in sensor_groups:
            sensor_groups[sensor_name] = []
        sensor_groups[sensor_name].append(col)

        # Calculate statistics
        values = features_df[col].values
        zero_mask = (values == 0.0) | (np.abs(values) < 1e-6)
        zero_percentage = (np.sum(zero_mask) / len(values)) * 100

        feature_stats[col] = {
            'zero_percentage': zero_percentage,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'sensor': sensor_name,
            'feature_type': feature_type
        }

    # Print sensor group analysis
    print("\n" + "="*60)
    print("SENSOR GROUP ANALYSIS")
    print("="*60)

    for sensor_name, features in sorted(sensor_groups.items()):
        sensor_type = get_sensor_type(sensor_name)
        avg_zero_pct = np.mean(
            [feature_stats[f]['zero_percentage'] for f in features])

        status = get_status_icon(avg_zero_pct)
        print(f"\n{sensor_name} ({sensor_type}):")
        print(f"  Features: {len(features)}")
        print(f"  Avg Zero%: {avg_zero_pct:.2f}% {status}")

        for feature in features:
            stats = feature_stats[feature]
            zero_pct = stats['zero_percentage']
            feature_status = get_status_icon(zero_pct)
            print(
                f"    {feature:20} | Zero%: {zero_pct:6.2f}% | Range: [{stats['min']:8.3f}, {stats['max']:8.3f}] {feature_status}")

    return feature_stats, sensor_groups


def get_sensor_type(sensor_name):
    """Get sensor type from sensor name"""
    if sensor_name.startswith('PS'):
        return 'Pressure'
    elif sensor_name.startswith('TS'):
        return 'Temperature'
    elif sensor_name.startswith('FS'):
        return 'Flow'
    elif sensor_name.startswith('EPS'):
        return 'Motor Power'
    elif sensor_name.startswith('VS'):
        return 'Vibration'
    elif sensor_name in ['CE', 'CP', 'SE']:
        return 'Efficiency'
    else:
        return 'Other'


def get_status_icon(zero_percentage):
    """Get status icon based on zero percentage"""
    if zero_percentage < 1.0:
        return "✅ EXCELLENT"
    elif zero_percentage < 5.0:
        return "⚠️  WARNING"
    elif zero_percentage < 15.0:
        return "⚠️  WARNING"
    else:
        return "❌ CRITICAL"


def analyze_targets(targets_df):
    """Analyze target variables"""
    print("\n" + "="*60)
    print("TARGET VARIABLES ANALYSIS")
    print("="*60)

    print(f"Targets shape: {targets_df.shape}")
    print(f"Target columns: {list(targets_df.columns)}")

    for col in targets_df.columns:
        values = targets_df[col].values
        unique_vals = np.unique(values)
        print(f"\n{col}:")
        print(f"  Unique values: {unique_vals}")
        print(f"  Value counts:")
        for val in unique_vals:
            count = np.sum(values == val)
            percentage = (count / len(values)) * 100
            print(f"    {val}: {count} ({percentage:.1f}%)")


def load_original_analysis(analysis_dir):
    """Load original dataset analysis results for comparison"""
    results_file = os.path.join(analysis_dir, 'dataset_analysis_results.csv')
    if os.path.exists(results_file):
        return pd.read_csv(results_file)
    return None


def compare_with_original(feature_stats, sensor_groups, original_analysis=None):
    """Compare current results with original analysis"""
    print("\n" + "="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60)

    if original_analysis is not None:
        print("Comparison with original raw data analysis:")
        print("-" * 50)

        # Map original sensor names to processed features
        sensor_improvements = {}

        for _, row in original_analysis.iterrows():
            sensor_name = row['sensor_name']
            original_zero_pct = row['zero_percentage']

            if sensor_name in sensor_groups:
                # Calculate average zero percentage for this sensor's features
                current_zero_pct = np.mean([
                    feature_stats[f]['zero_percentage']
                    for f in sensor_groups[sensor_name]
                ])

                improvement = original_zero_pct - current_zero_pct
                sensor_improvements[sensor_name] = {
                    'original': original_zero_pct,
                    'current': current_zero_pct,
                    'improvement': improvement
                }

                status = get_status_icon(current_zero_pct)
                print(f"{sensor_name:4} | Original: {original_zero_pct:6.2f}% | Current: {current_zero_pct:6.2f}% | Improvement: {improvement:+6.2f}% {status}")

        # Summary
        total_improvement = np.mean([v['improvement']
                                    for v in sensor_improvements.values()])
        print(f"\nOverall average improvement: {total_improvement:+.2f}%")

    else:
        print("No original analysis found for comparison.")
        print("Run dataset analysis first to enable comparison.")


def main():
    parser = create_parser()
    args = parser.parse_args()

    data_dir = args.data_dir

    print(f"Analyzing processed data in: {data_dir}")

    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        return

    # Load processed features
    features_path = os.path.join(data_dir, 'X_full.csv')
    targets_path = os.path.join(data_dir, 'y_full.csv')

    if not os.path.exists(features_path):
        print(f"Error: Features file {features_path} not found")
        return

    if not os.path.exists(targets_path):
        print(f"Error: Targets file {targets_path} not found")
        return

    # Load data
    print("Loading processed data...")
    features_df = pd.read_csv(features_path)
    targets_df = pd.read_csv(targets_path)
    metadata = load_metadata(data_dir)

    # Print metadata info
    if metadata:
        print(
            f"\nProcessing timestamp: {metadata.get('preprocessing_timestamp', 'Unknown')}")
        print(f"Dataset: {metadata.get('dataset_dir', 'Unknown')}")
        if 'command_line_args' in metadata:
            ps4_method = metadata['command_line_args'].get(
                'ps4_correction_method', 'Unknown')
            print(f"PS4 correction method: {ps4_method}")

    # Analyze features
    feature_stats, sensor_groups = analyze_features(features_df, metadata)

    # Analyze targets
    analyze_targets(targets_df)

    # Compare with original if requested
    if args.compare_original:
        # Look for original analysis in output/analysis
        analysis_dir = os.path.join('output', 'analysis')
        original_analysis = load_original_analysis(analysis_dir)
        compare_with_original(feature_stats, sensor_groups, original_analysis)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"✅ Processed data analyzed successfully!")
    print(f"📊 Ready for model training!")
    print(f"📁 Data location: {data_dir}")


if __name__ == "__main__":
    main()
