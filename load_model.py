#!/usr/bin/env python3
"""
Model Loader and Prediction Script

This script demonstrates how to load trained PEECOM models and make predictions
on new data using the saved .joblib files.

Usage:
    python load_model.py --model-dir output/models/random_forest/stable_flag
    python load_model.py --model-dir output/models/gradient_boosting/cooler_condition --data new_data.csv
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import os


def load_trained_model(model_dir):
    """Load a trained model and scaler from directory"""
    model_dir = Path(model_dir)

    # Find model and scaler files
    model_files = list(model_dir.glob('*_model.joblib'))
    scaler_files = list(model_dir.glob('*_scaler.joblib'))

    if not model_files:
        raise FileNotFoundError(f"No model file found in {model_dir}")
    if not scaler_files:
        raise FileNotFoundError(f"No scaler file found in {model_dir}")

    # Load model and scaler
    model = joblib.load(model_files[0])
    scaler = joblib.load(scaler_files[0])

    # Load metadata if available
    metadata_file = model_dir / 'training_results.json'
    metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    print(f"✓ Loaded model from: {model_files[0]}")
    print(f"✓ Loaded scaler from: {scaler_files[0]}")

    if metadata:
        print(f"✓ Model: {metadata.get('model_display_name', 'Unknown')}")
        print(f"✓ Target: {metadata.get('target_name', 'Unknown')}")
        print(
            f"✓ Training Accuracy: {metadata.get('test_accuracy', 'Unknown')}")

    return model, scaler, metadata


def make_predictions(model, scaler, data, metadata=None):
    """Make predictions on new data"""
    print(f"\nMaking predictions on {len(data)} samples...")

    # Scale the data
    scaled_data = scaler.transform(data)

    # Make predictions
    predictions = model.predict(scaled_data)

    # Get prediction probabilities if available
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(scaled_data)

    return predictions, probabilities


def create_prediction_report(predictions, probabilities, output_file=None):
    """Create a detailed prediction report"""
    report = []
    report.append("PEECOM PREDICTION REPORT")
    report.append("=" * 50)
    report.append(f"Total Predictions: {len(predictions)}")
    report.append(f"Unique Classes: {len(np.unique(predictions))}")
    report.append("")

    # Class distribution
    unique, counts = np.unique(predictions, return_counts=True)
    report.append("PREDICTION DISTRIBUTION")
    report.append("-" * 30)
    for class_val, count in zip(unique, counts):
        percentage = (count / len(predictions)) * 100
        report.append(
            f"Class {class_val}: {count} samples ({percentage:.1f}%)")

    if probabilities is not None:
        report.append("")
        report.append("CONFIDENCE ANALYSIS")
        report.append("-" * 30)

        # Calculate confidence (max probability for each prediction)
        max_probs = np.max(probabilities, axis=1)
        report.append(f"Average Confidence: {np.mean(max_probs):.3f}")
        report.append(f"Min Confidence: {np.min(max_probs):.3f}")
        report.append(f"Max Confidence: {np.max(max_probs):.3f}")

        # High/Low confidence predictions
        high_conf = np.sum(max_probs > 0.9)
        low_conf = np.sum(max_probs < 0.7)
        report.append(f"High Confidence (>90%): {high_conf} samples")
        report.append(f"Low Confidence (<70%): {low_conf} samples")

    report_text = "\n".join(report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Prediction report saved to: {output_file}")

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description='Load trained PEECOM model and make predictions')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing the trained model files')
    parser.add_argument('--data', type=str,
                        help='CSV file with new data to predict (optional)')
    parser.add_argument('--output', type=str,
                        help='Output file for predictions (optional)')

    args = parser.parse_args()

    try:
        # Load the trained model
        print("Loading trained model...")
        model, scaler, metadata = load_trained_model(args.model_dir)

        if args.data:
            # Load new data and make predictions
            print(f"\nLoading data from: {args.data}")
            new_data = pd.read_csv(args.data)
            print(f"Data shape: {new_data.shape}")

            # Make predictions
            predictions, probabilities = make_predictions(
                model, scaler, new_data, metadata)

            # Create results DataFrame
            results_df = new_data.copy()
            results_df['prediction'] = predictions

            if probabilities is not None:
                for i in range(probabilities.shape[1]):
                    results_df[f'probability_class_{i}'] = probabilities[:, i]
                results_df['confidence'] = np.max(probabilities, axis=1)

            # Save results
            if args.output:
                results_df.to_csv(args.output, index=False)
                print(f"Predictions saved to: {args.output}")

                # Create prediction report
                report_file = Path(args.output).with_suffix('.txt')
                create_prediction_report(
                    predictions, probabilities, report_file)
            else:
                print("\nPredictions:")
                print(results_df[['prediction'] + [
                      col for col in results_df.columns if 'probability' in col or col == 'confidence']].head(10))

            # Display summary
            print("\n" + create_prediction_report(predictions, probabilities))

        else:
            print("\n✓ Model loaded successfully!")
            print("Use --data argument to provide data for predictions.")

            if metadata:
                print(f"\nModel Information:")
                print(
                    f"  Expected features: {len(metadata.get('feature_names', []))}")
                print(
                    f"  Feature names: {metadata.get('feature_names', [])[:5]}... (showing first 5)")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
