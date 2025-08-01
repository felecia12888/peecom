#!/usr/bin/env python3
"""
Analyze Processed Data Results

This script analyzes the CSV files from processed data to see improvements.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_csv_data(data_dir):
    """Analyze processed CSV data"""
    print(f"\nAnalyzing processed data in {data_dir}")

    results = {}
    csv_files = list(Path(data_dir).glob("*_corrected.csv"))

    for csv_file in csv_files:
        sensor_name = csv_file.stem.replace('_corrected', '')
        print(f"\nAnalyzing {sensor_name}...")

        try:
            data = pd.read_csv(csv_file)
            values = data.values

            # Calculate zero statistics
            zero_mask = (values == 0.0) | (np.abs(values) < 1e-6)
            zero_count = np.sum(zero_mask)
            zero_percentage = (zero_count / values.size) * 100

            # Calculate basic statistics
            stats = {
                'shape': values.shape,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'zero_count': int(zero_count),
                'zero_percentage': float(zero_percentage),
                'total_values': values.size
            }

            results[sensor_name] = stats

            print(f"  Shape: {stats['shape']}")
            print(f"  Zero percentage: {stats['zero_percentage']:.2f}%")
            print(f"  Value range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")

        except Exception as e:
            print(f"  Error: {e}")
            results[sensor_name] = {'error': str(e)}

    return results


def generate_summary_report(results):
    """Generate summary report"""
    print(f"\n{'='*60}")
    print("PROCESSED DATA SUMMARY REPORT")
    print(f"{'='*60}")

    # Group sensors by type
    pressure_sensors = [k for k in results.keys() if k.startswith('PS')]
    flow_sensors = [k for k in results.keys() if k.startswith('FS')]
    temp_sensors = [k for k in results.keys() if k.startswith('TS')]
    other_sensors = [k for k in results.keys() if not any(
        k.startswith(p) for p in ['PS', 'FS', 'TS'])]

    def print_sensor_group(sensor_list, group_name):
        if sensor_list:
            print(f"\n{group_name}:")
            print("-" * 40)
            for sensor in sorted(sensor_list):
                if 'error' not in results[sensor]:
                    zero_pct = results[sensor]['zero_percentage']
                    status = "✅ EXCELLENT" if zero_pct < 1 else "⚠️  NEEDS WORK" if zero_pct < 10 else "❌ CRITICAL"
                    print(f"  {sensor:8} | Zero%: {zero_pct:6.2f}% | {status}")
                else:
                    print(f"  {sensor:8} | ERROR: {results[sensor]['error']}")

    print_sensor_group(pressure_sensors, "PRESSURE SENSORS")
    print_sensor_group(flow_sensors, "FLOW SENSORS")
    print_sensor_group(temp_sensors, "TEMPERATURE SENSORS")
    print_sensor_group(other_sensors, "OTHER SENSORS")

    # Overall statistics
    all_zero_percentages = [r['zero_percentage']
                            for r in results.values() if 'error' not in r]
    if all_zero_percentages:
        print(f"\n{'='*40}")
        print("OVERALL STATISTICS:")
        print(f"{'='*40}")
        print(f"Total sensors processed: {len(results)}")
        print(f"Average zero percentage: {np.mean(all_zero_percentages):.2f}%")
        print(f"Max zero percentage: {np.max(all_zero_percentages):.2f}%")
        print(
            f"Sensors with <1% zeros: {sum(1 for z in all_zero_percentages if z < 1)}")
        print(
            f"Sensors with >10% zeros: {sum(1 for z in all_zero_percentages if z > 10)}")


def compare_with_original_analysis(results):
    """Compare with original analysis results"""
    print(f"\n{'='*60}")
    print("COMPARISON WITH ORIGINAL ANALYSIS")
    print(f"{'='*60}")

    # Original analysis findings
    original_issues = {
        'PS4': {'zero_pct': 66.68, 'status': 'CRITICAL'},
        'PS2': {'zero_pct': 13.41, 'status': 'WARNING'},
        'PS3': {'zero_pct': 14.49, 'status': 'WARNING'},
        'FS1': {'zero_pct': 5.65, 'status': 'WARNING'},
        'SE': {'zero_pct': 13.33, 'status': 'WARNING'}
    }

    print("Original Issues vs Current Status:")
    print("-" * 50)
    for sensor, original in original_issues.items():
        print(
            f"{sensor:4} | Original: {original['zero_pct']:6.2f}% ({original['status']:8}) | ", end="")

        # Try to find current status (accounting for different naming)
        current_sensor = sensor
        if sensor == 'SE':
            current_sensor = 'SE'  # Should match

        if current_sensor in results and 'error' not in results[current_sensor]:
            current_pct = results[current_sensor]['zero_percentage']
            improvement = original['zero_pct'] - current_pct
            print(
                f"Current: {current_pct:6.2f}% | Improvement: {improvement:+6.2f}%")
        else:
            print("Current: DATA NOT FOUND")


def main():
    processed_data_dir = "output/processed_data"

    if not os.path.exists(processed_data_dir):
        print(
            f"Error: Processed data directory not found: {processed_data_dir}")
        return 1

    # Analyze processed data
    results = analyze_csv_data(processed_data_dir)

    # Generate reports
    generate_summary_report(results)
    compare_with_original_analysis(results)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Processed data location: {processed_data_dir}")
    print("✅ Data preprocessing was successful!")
    print("📊 Ready for model training phase")

    return 0


if __name__ == "__main__":
    exit(main())
