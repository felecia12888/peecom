#!/usr/bin/env python3
"""
PEECOM Main Application

Main entry point for hydraulic system condition monitoring using processed data.
This script loads the processed CSV data and trains machine learning models
for PEECOM hydraulic system condition monitoring.

Usage:
    python main.py --data output/processed_data/cmohs --target stable_flag --model random_forest
    python main.py --data output/processed_data/cmohs --target cooler_condition --model logistic_regression
    python main.py --eval-all --model gradient_boosting  # Evaluate all targets with gradient boosting
    python main.py --list-models  # Show all available models
"""

from src.models.model_loader import model_loader, get_model_choices
import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import joblib
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import model loader


def load_processed_data(data_dir):
    """Load processed features and targets from CSV files"""
    print(f"Loading data from: {data_dir}")

    # Load features and targets
    X = pd.read_csv(os.path.join(data_dir, 'X_full.csv'))
    y = pd.read_csv(os.path.join(data_dir, 'y_full.csv'))

    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Available targets: {list(y.columns)}")

    return X, y


def prepare_targets(y, target_column='stable_flag'):
    """Prepare target variable for classification"""
    if target_column in y.columns:
        target = y[target_column].values
        print(f"Using {target_column} as target")
        print(f"Target distribution: {np.bincount(target)}")
        return target
    else:
        print(f"Available target columns: {list(y.columns)}")
        # Default to first column
        target = y.iloc[:, 0].values
        print(f"Using {y.columns[0]} as target")
        return target


def train_model_with_loader(X, y, model_name='random_forest', output_dir='output/models', target_name='unknown'):
    """Train a model using the model loader system"""
    print(
        f"\nTraining {model_loader.get_model_display_name(model_name)} model...")

    # Create output directory structure: output/models/model_name/target_name/
    model_output_dir = Path(output_dir) / model_name / target_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Load model from model loader
    model_instance = model_loader.load_model(model_name)
    model = model_instance.get_model()

    # Train the model
    print("Training model...")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Detailed evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation
    print("\nCross-validation scores:")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"CV scores: {cv_scores}")
    print(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Save model and scaler
    model_file = model_output_dir / f'{model_name}_model.joblib'
    scaler_file = model_output_dir / f'{model_name}_scaler.joblib'
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    # Prepare results
    results = {
        'model_name': model_name,
        'model_display_name': model_loader.get_model_display_name(model_name),
        'target_name': target_name,
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importance': model_instance.get_feature_importance(),
        'model_params': model_instance.get_params(),
        'timestamp': datetime.now().isoformat(),
        'feature_names': list(X.columns),
        'model_file': str(model_file),
        'scaler_file': str(scaler_file)
    }

    # Save results
    save_model_results(results, X.columns, model_output_dir)

    print(f"Model saved to: {model_file}")
    print(f"Results saved to: {model_output_dir}")

    return model, scaler, results


def save_model_results(results, feature_names, output_dir):
    """Save model results and feature importance"""
    output_dir = Path(output_dir)

    # Save results JSON
    results_file = output_dir / 'training_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)

    print(f"Results saved to: {results_file}")

    # Feature importance analysis
    if results['feature_importance'] is not None:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': results['feature_importance']
        }).sort_values('importance', ascending=False)

        importance_file = output_dir / 'feature_importance.csv'
        importance_df.to_csv(importance_file, index=False)

        print(f"Feature importance saved to: {importance_file}")
        print("\nTop 10 most important features:")
        print(importance_df.head(10))

    # Create comprehensive text summary
    summary_file = output_dir / 'training_summary.txt'
    create_training_summary(results, importance_df if results['feature_importance'] is not None else None, summary_file)
    print(f"Training summary saved to: {summary_file}")


def create_training_summary(results, importance_df, summary_file):
    """Create a comprehensive text summary of training results"""
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PEECOM HYDRAULIC SYSTEM CONDITION MONITORING - TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Basic Information
        f.write("TRAINING INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model: {results['model_display_name']}\n")
        f.write(f"Target Variable: {results['target_name']}\n")
        f.write(f"Training Date: {results['timestamp']}\n")
        f.write(f"Number of Features: {len(results['feature_names'])}\n\n")
        
        # Model Performance
        f.write("MODEL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training Accuracy: {results['train_accuracy']:.4f} ({results['train_accuracy']*100:.2f}%)\n")
        f.write(f"Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)\n")
        f.write(f"Cross-Validation Mean: {results['cv_mean']:.4f} ({results['cv_mean']*100:.2f}%)\n")
        f.write(f"Cross-Validation Std: ±{results['cv_std']:.4f} (±{results['cv_std']*100:.2f}%)\n\n")
        
        # Performance Assessment
        f.write("PERFORMANCE ASSESSMENT\n")
        f.write("-" * 40 + "\n")
        test_acc = results['test_accuracy']
        cv_mean = results['cv_mean']
        overfitting = results['train_accuracy'] - results['test_accuracy']
        
        if test_acc >= 0.95:
            f.write("✓ EXCELLENT: Test accuracy ≥ 95%\n")
        elif test_acc >= 0.90:
            f.write("✓ VERY GOOD: Test accuracy ≥ 90%\n")
        elif test_acc >= 0.85:
            f.write("⚠ GOOD: Test accuracy ≥ 85%\n")
        elif test_acc >= 0.80:
            f.write("⚠ FAIR: Test accuracy ≥ 80%\n")
        else:
            f.write("✗ POOR: Test accuracy < 80%\n")
        
        if overfitting <= 0.05:
            f.write("✓ LOW OVERFITTING: Train-Test gap ≤ 5%\n")
        elif overfitting <= 0.10:
            f.write("⚠ MODERATE OVERFITTING: Train-Test gap ≤ 10%\n")
        else:
            f.write("✗ HIGH OVERFITTING: Train-Test gap > 10%\n")
        
        cv_stability = results['cv_std']
        if cv_stability <= 0.02:
            f.write("✓ STABLE MODEL: CV std ≤ 2%\n")
        elif cv_stability <= 0.05:
            f.write("⚠ MODERATELY STABLE: CV std ≤ 5%\n")
        else:
            f.write("✗ UNSTABLE MODEL: CV std > 5%\n")
        f.write("\n")
        
        # Key Findings
        f.write("KEY FINDINGS\n")
        f.write("-" * 40 + "\n")
        
        if test_acc >= 0.95 and overfitting <= 0.05:
            f.write("🎯 EXCELLENT MODEL: High accuracy with minimal overfitting\n")
            f.write("   → Ready for production deployment\n")
        elif test_acc >= 0.90:
            f.write("✅ GOOD MODEL: Strong performance for condition monitoring\n")
            f.write("   → Suitable for most applications\n")
        else:
            f.write("🔧 NEEDS IMPROVEMENT: Consider feature engineering or model tuning\n")
        
        if importance_df is not None:
            top_feature = importance_df.iloc[0]
            f.write(f"📊 MOST IMPORTANT SENSOR: {top_feature['feature']} ({top_feature['importance']:.3f})\n")
            
            # Identify sensor groups
            pressure_features = importance_df[importance_df['feature'].str.contains('PS')].head(3)
            if not pressure_features.empty:
                f.write(f"🔧 KEY PRESSURE SENSORS: {', '.join(pressure_features['feature'].tolist())}\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        if test_acc >= 0.98:
            f.write("• Model performance is excellent - no immediate changes needed\n")
            f.write("• Consider testing on additional datasets to validate robustness\n")
        elif test_acc >= 0.90:
            f.write("• Good performance - monitor in production environment\n")
            if overfitting > 0.05:
                f.write("• Consider regularization to reduce overfitting\n")
        else:
            f.write("• Performance needs improvement - consider:\n")
            f.write("  - Additional feature engineering\n")
            f.write("  - Hyperparameter tuning\n")
            f.write("  - Different model architectures\n")
            f.write("  - More training data if available\n")
        
        if cv_stability > 0.03:
            f.write("• Model shows instability - consider cross-validation tuning\n")
        
        f.write("\n")
        
        # Feature Importance Analysis
        if importance_df is not None:
            f.write("TOP 10 MOST IMPORTANT FEATURES\n")
            f.write("-" * 40 + "\n")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                f.write(f"{i:2d}. {row['feature']:<15} - {row['importance']:.4f} ({row['importance']*100:.2f}%)\n")
            
            f.write("\nFEATURE IMPORTANCE INSIGHTS\n")
            f.write("-" * 40 + "\n")
            
            # Analyze by sensor type
            sensor_groups = {}
            for _, row in importance_df.iterrows():
                sensor_type = row['feature'][:3] if len(row['feature']) >= 3 else row['feature'][:2]
                if sensor_type not in sensor_groups:
                    sensor_groups[sensor_type] = []
                sensor_groups[sensor_type].append(row['importance'])
            
            # Calculate average importance by sensor group
            sensor_avg = {k: np.mean(v) for k, v in sensor_groups.items()}
            sorted_sensors = sorted(sensor_avg.items(), key=lambda x: x[1], reverse=True)
            
            f.write("Average Importance by Sensor Group:\n")
            for sensor, avg_imp in sorted_sensors[:5]:
                f.write(f"  {sensor}: {avg_imp:.4f} ({avg_imp*100:.2f}%)\n")
        
        f.write("\n")
        
        # Model Files Information
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
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF TRAINING SUMMARY\n")
        f.write("="*80 + "\n")


def evaluate_all_targets(data_dir: str, output_dir: str, model_type: str = 'random_forest'):
    """Evaluate model performance on all available targets"""
    print("\n" + "="*60)
    print("EVALUATING ALL TARGETS")
    print("="*60)

    # Load data
    X, y = load_processed_data(data_dir)

    results = {}

    # Evaluate each target
    for target_col in y.columns:
        print(f"\n--- Evaluating {target_col} ---")
        try:
            target = prepare_targets(y, target_col)
            model, scaler, result = train_model_with_loader(
                X, target, model_type, output_dir, target_col
            )

            results[target_col] = result
            print(f"✓ {target_col}: {result['test_accuracy']:.4f} accuracy")
        except Exception as e:
            print(f"✗ {target_col}: Failed - {str(e)}")
            results[target_col] = {'error': str(e)}

    # Save summary results
    summary_file = os.path.join(
        output_dir, "models", "all_targets_summary.json")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSummary saved to: {summary_file}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train model on processed PEECOM data')
    parser.add_argument('--data', type=str, default='output/processed_data/cmohs',
                        help='Path to processed data directory')
    parser.add_argument('--target', type=str, default='stable_flag',
                        help='Target column name')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=get_model_choices(),
                        help='Model type to train')
    parser.add_argument('--output', type=str, default='output/models',
                        help='Output directory for model and results')
    parser.add_argument('--eval-all', action='store_true',
                        help='Evaluate all available targets')
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and exit')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed model information when listing')

    args = parser.parse_args()

    # Handle list models request
    if args.list_models:
        print("PEECOM Available Models")
        print("=" * 50)
        model_loader.list_models(verbose=args.verbose)
        return 0

    print("="*60)
    print("PEECOM HYDRAULIC SYSTEM CONDITION MONITORING")
    print("="*60)
    print(f"Selected Model: {model_loader.get_model_display_name(args.model)}")
    print(f"Data directory: {args.data}")
    print(f"Output directory: {args.output}")
    print(f"Target variable: {args.target}")

    # Check if processed data exists
    if not os.path.exists(args.data):
        print(f"Error: Data directory not found: {args.data}")
        print("Please run dataset_preprocessing.py first to generate processed data.")
        return 1

    try:
        if args.eval_all:
            print(
                f"Training {model_loader.get_model_display_name(args.model)} for all target variables...")
            results = evaluate_all_targets(args.data, args.output, args.model)
        else:
            # Load data
            X, y = load_processed_data(args.data)

            # Prepare target
            target = prepare_targets(y, args.target)

            # Train model using model loader
            model, scaler, results = train_model_with_loader(
                X, target, args.model, args.output, args.target
            )

        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        if not args.eval_all:
            print(f"Model: {results['model_display_name']}")
            print(f"Target: {results['target_name']}")
            print(f"Model accuracy: {results['test_accuracy']:.4f}")
            print(
                f"CV score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
        print(f"Results saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
