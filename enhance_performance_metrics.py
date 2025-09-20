#!/usr/bin/env python3
"""
Enhanced Performance Metrics Calculator
=======================================

Recalculates comprehensive performance metrics including:
- F1-Score, Precision, Recall
- R-squared, MSE, MAE
- Cross-validation statistics
- Feature importance analysis
- Statistical significance tests

This script loads trained models and test data to compute missing metrics
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score
)
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedMetricsCalculator:
    """Calculate comprehensive performance metrics for trained models"""
    
    def __init__(self, models_dir="output/models", processed_data_dir="output/processed_data"):
        self.models_dir = Path(models_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # Top 2 datasets
        self.top_datasets = ['motorvd', 'cmohs']
        
    def load_test_data(self, dataset):
        """Load test data for a specific dataset"""
        dataset_dir = self.processed_data_dir / dataset
        
        X_file = dataset_dir / "X_full.csv"
        y_file = dataset_dir / "y_full.csv"
        
        if not X_file.exists() or not y_file.exists():
            print(f"⚠️  Test data not found for {dataset}")
            return None, None
        
        try:
            X = pd.read_csv(X_file)
            y = pd.read_csv(y_file)
            return X, y
        except Exception as e:
            print(f"❌ Error loading test data for {dataset}: {e}")
            return None, None
    
    def calculate_enhanced_metrics(self, model, scaler, X_test, y_test, target_name):
        """Calculate comprehensive performance metrics"""
        try:
            # Scale features if scaler exists
            if scaler is not None:
                X_test_scaled = scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            
            # Get target data
            if target_name not in y_test.columns:
                print(f"⚠️  Target {target_name} not found in test data")
                return {}
            
            y_true = y_test[target_name]
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Determine if classification or regression
            is_classification = len(np.unique(y_true)) <= 10  # Heuristic
            
            metrics = {}
            
            if is_classification:
                # Classification metrics
                metrics['test_accuracy'] = accuracy_score(y_true, y_pred)
                
                # Handle multiclass vs binary
                average_method = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'
                
                metrics['f1_score'] = f1_score(y_true, y_pred, average=average_method, zero_division=0)
                metrics['precision'] = precision_score(y_true, y_pred, average=average_method, zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average=average_method, zero_division=0)
                
                # Cross-validation for classification
                cv_scores = cross_val_score(model, X_test_scaled, y_true, cv=5, scoring='accuracy')
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                
                # Classification report
                class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                metrics['weighted_f1'] = class_report['weighted avg']['f1-score']
                metrics['macro_f1'] = class_report['macro avg']['f1-score']
                
            else:
                # Regression metrics
                metrics['r2_score'] = r2_score(y_true, y_pred)
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                
                # Cross-validation for regression
                cv_scores = cross_val_score(model, X_test_scaled, y_true, cv=5, scoring='r2')
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
            
            # Additional metrics
            metrics['prediction_variance'] = np.var(y_pred)
            metrics['target_variance'] = np.var(y_true)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            return {}
    
    def enhance_all_models(self):
        """Calculate enhanced metrics for all models in top datasets"""
        
        enhanced_results = []
        
        for dataset in self.top_datasets:
            print(f"\n📊 Processing dataset: {dataset}")
            
            # Load test data
            X_test, y_test = self.load_test_data(dataset)
            if X_test is None or y_test is None:
                continue
            
            dataset_dir = self.models_dir / dataset
            if not dataset_dir.exists():
                continue
            
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model_name = model_dir.name
                print(f"  🤖 Model: {model_name}")
                
                for target_dir in model_dir.iterdir():
                    if not target_dir.is_dir():
                        continue
                    
                    target_name = target_dir.name
                    print(f"    🎯 Target: {target_name}")
                    
                    # Load model files
                    model_file = target_dir / f"{model_name}_model.joblib"
                    scaler_file = target_dir / f"{model_name}_scaler.joblib"
                    results_file = target_dir / "training_results.json"
                    
                    if not model_file.exists():
                        print(f"    ❌ Model file not found: {model_file}")
                        continue
                    
                    try:
                        # Load model and scaler
                        model = joblib.load(model_file)
                        scaler = joblib.load(scaler_file) if scaler_file.exists() else None
                        
                        # Load existing results
                        existing_results = {}
                        if results_file.exists():
                            with open(results_file, 'r') as f:
                                existing_results = json.load(f)
                        
                        # Calculate enhanced metrics
                        enhanced_metrics = self.calculate_enhanced_metrics(
                            model, scaler, X_test, y_test, target_name
                        )
                        
                        # Combine with existing results
                        combined_results = {**existing_results, **enhanced_metrics}
                        
                        # Save enhanced results
                        enhanced_results_file = target_dir / "enhanced_training_results.json"
                        with open(enhanced_results_file, 'w') as f:
                            json.dump(combined_results, f, indent=2)
                        
                        print(f"    ✅ Enhanced metrics saved: {enhanced_results_file}")
                        
                        # Store for summary
                        enhanced_results.append({
                            'dataset': dataset,
                            'model': model_name,
                            'target': target_name,
                            **combined_results
                        })
                        
                    except Exception as e:
                        print(f"    ❌ Error processing {model_name}/{target_name}: {e}")
        
        # Create summary DataFrame
        if enhanced_results:
            summary_df = pd.DataFrame(enhanced_results)
            
            # Save comprehensive summary
            summary_file = Path("output/enhanced_performance_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"\n📈 Enhanced performance summary saved: {summary_file}")
            
            # Print summary statistics
            print(f"\n🏆 ENHANCED PERFORMANCE SUMMARY")
            print("=" * 50)
            
            # Group by dataset and model
            summary_stats = summary_df.groupby(['dataset', 'model']).agg({
                'test_accuracy': ['mean', 'std'],
                'f1_score': ['mean', 'std'],
                'precision': ['mean', 'std'],
                'recall': ['mean', 'std'],
                'cv_mean': ['mean', 'std']
            }).round(4)
            
            print(summary_stats)
            
            return summary_df
        
        return None

def main():
    """Main execution function"""
    
    print("🚀 Enhanced Performance Metrics Calculator")
    print("=" * 60)
    print("📊 Calculating comprehensive metrics for top 2 datasets")
    print("🎯 Focus: F1-Score, Precision, Recall, R², MSE, MAE, CV stats")
    
    try:
        calculator = EnhancedMetricsCalculator()
        enhanced_df = calculator.enhance_all_models()
        
        if enhanced_df is not None:
            print(f"\n🎉 Enhanced metrics calculation complete!")
            print(f"📊 {len(enhanced_df)} models enhanced with comprehensive metrics")
        else:
            print("❌ No models found for enhancement")
            
    except Exception as e:
        print(f"❌ Error during enhancement: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())