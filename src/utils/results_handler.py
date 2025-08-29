#!/usr/bin/env python3
"""
Results Handler for PEECOM

This module handles saving and formatting training results including
JSON files, CSV files, text summaries, and other output formats.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def save_training_results(results, feature_names, output_dir):
    """Save comprehensive training results in multiple formats"""
    output_dir = Path(output_dir)

    # Save JSON results
    _save_json_results(results, output_dir)

    # Save feature importance CSV
    _save_feature_importance(results, feature_names, output_dir)

    # Save comprehensive text summary
    _save_text_summary(results, feature_names, output_dir)


def _save_json_results(results, output_dir):
    """Save results as JSON file"""
    results_file = output_dir / 'training_results.json'

    # Convert numpy arrays and other non-serializable objects to JSON-compatible types
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, dict):
            # Handle nested dictionaries (like classification_report)
            json_results[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    json_results[key][sub_key] = sub_value.tolist()
                elif isinstance(sub_value, (np.floating, np.integer)):
                    json_results[key][sub_key] = float(sub_value)
                elif hasattr(sub_value, 'item'):  # numpy scalar
                    json_results[key][sub_key] = sub_value.item()
                else:
                    json_results[key][sub_key] = sub_value
        elif isinstance(value, (np.floating, np.integer)):
            json_results[key] = float(value)
        elif hasattr(value, 'item'):  # numpy scalar
            json_results[key] = value.item()
        else:
            json_results[key] = value

    json_results['timestamp'] = datetime.now().isoformat()

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to: {results_file}")


def _save_feature_importance(results, feature_names, output_dir):
    """Save feature importance as CSV file"""
    if results['feature_importance'] is not None:
        # Ensure feature_importance is a proper array/list
        importance_values = results['feature_importance']

        # Handle different data types that might come from the model
        if isinstance(importance_values, dict):
            # If it's a dict, extract values in order
            importance_values = list(importance_values.values())
        elif hasattr(importance_values, 'tolist'):
            # Convert numpy arrays to list
            importance_values = importance_values.tolist()
        elif not isinstance(importance_values, (list, np.ndarray)):
            # Convert to list if it's some other iterable
            importance_values = list(importance_values)

        # Ensure we have the right number of importance values
        if len(importance_values) != len(feature_names):
            print(
                f"Warning: Feature importance length ({len(importance_values)}) doesn't match feature names length ({len(feature_names)})")
            # Pad or truncate to match
            if len(importance_values) < len(feature_names):
                importance_values.extend(
                    [0.0] * (len(feature_names) - len(importance_values)))
            else:
                importance_values = importance_values[:len(feature_names)]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)

        importance_file = output_dir / 'feature_importance.csv'
        importance_df.to_csv(importance_file, index=False)

        print(f"Feature importance saved to: {importance_file}")
        print("\nTop 10 most important features:")
        print(importance_df.head(10))

        return importance_df
    return None


def _save_text_summary(results, feature_names, output_dir):
    """Save comprehensive text summary"""
    summary_file = output_dir / 'training_summary.txt'

    with open(summary_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("PEECOM HYDRAULIC SYSTEM CONDITION MONITORING - TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Training Information
        f.write("TRAINING INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model: {results['model_display_name']}\n")
        f.write(f"Target Variable: {results['target_name']}\n")
        f.write(f"Training Date: {datetime.now().isoformat()}\n")
        f.write(f"Number of Features: {len(feature_names)}\n\n")

        # Model Performance
        f.write("MODEL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"Training Accuracy: {results['train_accuracy']:.4f} ({results['train_accuracy']*100:.2f}%)\n")
        f.write(
            f"Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)\n")
        f.write(
            f"Cross-Validation Mean: {results['cv_mean']:.4f} ({results['cv_mean']*100:.2f}%)\n")
        f.write(
            f"Cross-Validation Std: ±{results['cv_std']:.4f} (±{results['cv_std']*100:.2f}%)\n\n")

        # Performance Assessment
        f.write("PERFORMANCE ASSESSMENT\n")
        f.write("-" * 40 + "\n")

        # Accuracy assessment
        test_acc = results['test_accuracy']
        if test_acc >= 0.95:
            f.write("✓ EXCELLENT: Test accuracy ≥ 95%\n")
        elif test_acc >= 0.90:
            f.write("⚠ GOOD: Test accuracy ≥ 90%\n")
        elif test_acc >= 0.80:
            f.write("⚠ MODERATE: Test accuracy ≥ 80%\n")
        else:
            f.write("✗ POOR: Test accuracy < 80%\n")

        # Overfitting assessment
        train_test_gap = abs(
            results['train_accuracy'] - results['test_accuracy'])
        if train_test_gap <= 0.05:
            f.write("✓ LOW OVERFITTING: Train-Test gap ≤ 5%\n")
        elif train_test_gap <= 0.10:
            f.write("⚠ MODERATE OVERFITTING: Train-Test gap ≤ 10%\n")
        else:
            f.write("✗ HIGH OVERFITTING: Train-Test gap > 10%\n")

        # Stability assessment
        cv_std = results['cv_std']
        if cv_std <= 0.02:
            f.write("✓ STABLE MODEL: CV std ≤ 2%\n")
        elif cv_std <= 0.05:
            f.write("⚠ MODERATE STABILITY: CV std ≤ 5%\n")
        else:
            f.write("✗ UNSTABLE MODEL: CV std > 5%\n")

        f.write("\n")

        # Key Findings
        f.write("KEY FINDINGS\n")
        f.write("-" * 40 + "\n")

        if test_acc >= 0.95 and train_test_gap <= 0.05:
            f.write("🎯 EXCELLENT MODEL: High accuracy with minimal overfitting\n")
            f.write("   → Ready for production deployment\n")
        elif test_acc >= 0.90:
            f.write("👍 GOOD MODEL: Solid performance\n")
            if train_test_gap > 0.05:
                f.write("   → Consider regularization to reduce overfitting\n")
        else:
            f.write("⚠️ MODEL NEEDS IMPROVEMENT\n")
            f.write("   → Consider feature engineering or model tuning\n")

        # Feature importance insights
        if results['feature_importance'] is not None:
            # Ensure feature_importance is properly formatted
            importance_values = results['feature_importance']
            if isinstance(importance_values, dict):
                importance_values = list(importance_values.values())
            elif hasattr(importance_values, 'tolist'):
                importance_values = importance_values.tolist()
            elif not isinstance(importance_values, (list, np.ndarray)):
                importance_values = list(importance_values)

            # Ensure correct length
            if len(importance_values) != len(feature_names):
                if len(importance_values) < len(feature_names):
                    importance_values.extend(
                        [0.0] * (len(feature_names) - len(importance_values)))
                else:
                    importance_values = importance_values[:len(feature_names)]

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)

            top_feature = importance_df.iloc[0]
            f.write(
                f"📊 MOST IMPORTANT SENSOR: {top_feature['feature']} ({top_feature['importance']:.3f})\n")

            # Group by sensor type
            pressure_features = importance_df[importance_df['feature'].str.contains(
                'PS')]
            if not pressure_features.empty:
                top_pressure = pressure_features.head(3)['feature'].tolist()
                f.write(f"🔧 KEY PRESSURE SENSORS: {', '.join(top_pressure)}\n")

        f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")

        if test_acc >= 0.95 and train_test_gap <= 0.05:
            f.write(
                "• Model performance is excellent - no immediate changes needed\n")
            f.write(
                "• Consider testing on additional datasets to validate robustness\n")
        elif test_acc >= 0.90:
            f.write("• Good model performance - minor optimizations possible\n")
            if train_test_gap > 0.05:
                f.write(
                    "• Consider regularization techniques to reduce overfitting\n")
        else:
            f.write("• Model needs improvement - consider:\n")
            f.write("  - Feature engineering and selection\n")
            f.write("  - Hyperparameter tuning\n")
            f.write("  - Alternative algorithms\n")
            f.write("  - Data quality assessment\n")

        f.write("\n")

        # Top Features
        if results['feature_importance'] is not None:
            # Ensure feature_importance is properly formatted
            importance_values = results['feature_importance']
            if isinstance(importance_values, dict):
                importance_values = list(importance_values.values())
            elif hasattr(importance_values, 'tolist'):
                importance_values = importance_values.tolist()
            elif not isinstance(importance_values, (list, np.ndarray)):
                importance_values = list(importance_values)

            # Ensure correct length
            if len(importance_values) != len(feature_names):
                if len(importance_values) < len(feature_names):
                    importance_values.extend(
                        [0.0] * (len(feature_names) - len(importance_values)))
                else:
                    importance_values = importance_values[:len(feature_names)]

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)

            f.write("TOP 10 MOST IMPORTANT FEATURES\n")
            f.write("-" * 40 + "\n")

            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                f.write(
                    f"{i:2d}. {row['feature']:<15} - {row['importance']:.4f} ({row['importance']*100:.2f}%)\n")

            f.write("\n")

            # Feature group analysis
            f.write("FEATURE IMPORTANCE INSIGHTS\n")
            f.write("-" * 40 + "\n")
            f.write("Average Importance by Sensor Group:\n")

            # Group features by prefix
            groups = {}
            for _, row in importance_df.iterrows():
                prefix = row['feature'][:3]  # First 3 characters
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(row['importance'])

            # Calculate averages and sort
            group_avgs = {k: np.mean(v) for k, v in groups.items()}
            sorted_groups = sorted(
                group_avgs.items(), key=lambda x: x[1], reverse=True)

            for prefix, avg_importance in sorted_groups[:5]:  # Top 5 groups
                f.write(
                    f"  {prefix}: {avg_importance:.4f} ({avg_importance*100:.2f}%)\n")

        f.write("\n")

        # File Information
        f.write("SAVED MODEL FILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model File: {results['model_file']}\n")
        f.write(f"Scaler File: {results['scaler_file']}\n")
        f.write("Usage Example:\n")
        f.write("  import joblib\n")
        f.write(f"  model = joblib.load('{results['model_file']}')\n")
        f.write(f"  scaler = joblib.load('{results['scaler_file']}')\n")
        f.write("  # predictions = model.predict(scaler.transform(new_data))\n\n")

        # Model Parameters
        f.write("MODEL PARAMETERS\n")
        f.write("-" * 40 + "\n")
        for param, value in results['model_params'].items():
            f.write(f"{param}: {value}\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n")

    print(f"Training summary saved to: {summary_file}")


def load_model_for_prediction(model_dir):
    """Load a saved model and scaler for making predictions"""
    import joblib

    model_dir = Path(model_dir)

    # Find model and scaler files
    model_files = list(model_dir.glob("*_model.joblib"))
    scaler_files = list(model_dir.glob("*_scaler.joblib"))

    if not model_files:
        raise FileNotFoundError(f"No model file found in {model_dir}")
    if not scaler_files:
        raise FileNotFoundError(f"No scaler file found in {model_dir}")

    model = joblib.load(model_files[0])
    scaler = joblib.load(scaler_files[0])

    return model, scaler


def create_prediction_example(model_dir):
    """Create a Python script example for using the saved model"""
    model_dir = Path(model_dir)
    example_file = model_dir / "prediction_example.py"

    # Get model info from results
    results_file = model_dir / "training_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)

        model_name = results.get('model_name', 'model')
        target_name = results.get('target_name', 'target')
        feature_names = results.get('feature_names', [])
    else:
        model_name = "model"
        target_name = "target"
        feature_names = []

    with open(example_file, 'w') as f:
        f.write("#!/usr/bin/env python3\n")
        f.write('"""\n')
        f.write(f"Example script for using the trained {model_name} model\n")
        f.write(f"to predict {target_name} on new data.\n")
        f.write('"""\n\n')

        f.write("import joblib\n")
        f.write("import pandas as pd\n")
        f.write("import numpy as np\n\n")

        f.write("# Load the trained model and scaler\n")
        f.write(f"model = joblib.load('{model_name}_model.joblib')\n")
        f.write(f"scaler = joblib.load('{model_name}_scaler.joblib')\n\n")

        f.write("# Example: Load new data (replace with your data loading logic)\n")
        f.write("# new_data = pd.read_csv('new_sensor_data.csv')\n")
        f.write("# Make sure the data has the same features as training data:\n")
        if feature_names:
            f.write(f"required_features = {feature_names}\n\n")

        f.write("# Preprocess the data (same as training)\n")
        f.write(
            "# scaled_data = scaler.transform(new_data[required_features])\n\n")

        f.write("# Make predictions\n")
        f.write("# predictions = model.predict(scaled_data)\n")
        f.write("# probabilities = model.predict_proba(scaled_data)  # If needed\n\n")

        f.write("# Example with dummy data\n")
        f.write("if __name__ == '__main__':\n")
        f.write("    print('Model loaded successfully!')\n")
        f.write("    print(f'Model type: {type(model).__name__}')\n")
        if feature_names:
            f.write(f"    print(f'Expected features: {len(feature_names)}')\n")
        f.write("    print('Ready for predictions!')\n")

    print(f"Prediction example saved to: {example_file}")
    return example_file
