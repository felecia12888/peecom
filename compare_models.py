#!/usr/bin/env python3
"""
PEECOM Multi-Dataset Model Comparison
====================================

Comprehensive comparison of model performance across:
- All available datasets
- All trained models  
- All targets within each dataset

Features:
- Dataset-specific filtering
- Model-specific filtering
- PEECOM performance analysis
- Comprehensive summary export
- Cross-dataset performance ranking

Usage:
    python compare_models.py                     # Compare all models on all datasets
    python compare_models.py --dataset cmohs     # Compare models on CMOHS only
    python compare_models.py --model peecom      # Compare PEECOM across datasets
    python compare_models.py --summary           # Show summary only
    python compare_models.py --datasets cmohs,equipmentad  # Multiple datasets
"""

import json
import pandas as pd
import os
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# Dataset configuration with their display names and targets
DATASET_CONFIG = {
    'cmohs': {
        'name': 'CMOHS Hydraulic System',
        'targets': ['cooler_condition', 'valve_condition', 'pump_leakage', 'accumulator_pressure', 'stable_flag']
    },
    'equipmentad': {
        'name': 'Equipment Anomaly Detection',
        'targets': ['anomaly', 'equipment_type', 'location']
    },
    'mlclassem': {
        'name': 'ML Classification Energy Monthly',
        'targets': ['status', 'region', 'equipment_type']
    },
    'motorvd': {
        'name': 'Motor Vibration Dataset',
        'targets': ['condition', 'file_id']
    },
    'multivariatetsd': {
        'name': 'Multivariate Time Series',
        'targets': ['engine_id', 'cycle']
    },
    'sensord': {
        'name': 'Sensor Data',
        'targets': ['condition', 'file_id']
    },
    'smartmd': {
        'name': 'Smart Maintenance Dataset',
        'targets': ['anomaly_flag', 'machine_status', 'maintenance_required']
    }
}

MODELS = ['random_forest', 'logistic_regression', 'svm', 'peecom']


def get_available_datasets():
    """Get all datasets that have trained models"""
    datasets = set()
    models_dir = Path('output/models')

    if models_dir.exists():
        for dataset_dir in models_dir.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name not in MODELS:
                datasets.add(dataset_dir.name)

    return sorted(list(datasets))


def discover_dataset_structure():
    """Discover the actual structure of datasets and targets"""
    structure = {}
    models_dir = Path('output/models')

    if not models_dir.exists():
        return structure

    for dataset_dir in models_dir.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name not in MODELS:
            dataset_name = dataset_dir.name
            structure[dataset_name] = {
                'name': DATASET_CONFIG.get(dataset_name, {}).get('name', dataset_name.title()),
                'targets': set(),
                'models': set()
            }

            for model_dir in dataset_dir.iterdir():
                if model_dir.is_dir():
                    structure[dataset_name]['models'].add(model_dir.name)

                    for target_dir in model_dir.iterdir():
                        if target_dir.is_dir():
                            structure[dataset_name]['targets'].add(
                                target_dir.name)

            # Convert sets to sorted lists
            structure[dataset_name]['targets'] = sorted(
                list(structure[dataset_name]['targets']))
            structure[dataset_name]['models'] = sorted(
                list(structure[dataset_name]['models']))

    return structure


def load_model_results(datasets_filter=None):
    """Load results from all trained models"""
    results = []
    structure = discover_dataset_structure()

    for dataset_key, dataset_info in structure.items():
        if datasets_filter and dataset_key not in datasets_filter:
            continue

        for model in dataset_info['models']:
            for target in dataset_info['targets']:
                result_file = Path(
                    f'output/models/{dataset_key}/{model}/{target}/training_results.json')

                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)

                        # Extract key metrics
                        result_entry = {
                            'Dataset_Key': dataset_key,
                            'Dataset_Name': dataset_info['name'],
                            'Model_Key': model,
                            'Model_Name': data.get('model_display_name', model.replace('_', ' ').title()),
                            'Target_Key': target,
                            'Target_Name': target.replace('_', ' ').title(),
                            'Test_Accuracy': data.get('test_accuracy', 0.0),
                            'CV_Mean': data.get('cv_mean', 0.0),
                            'CV_Std': data.get('cv_std', 0.0),
                            'Train_Accuracy': data.get('train_accuracy', 0.0)
                        }

                        results.append(result_entry)

                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"📄 Error loading {result_file}: {e}")
                else:
                    print(f"📄 Result file not found: {result_file}")

    return pd.DataFrame(results)


def show_dataset_overview(df):
    """Show available datasets and models"""
    print("\n🗂️  AVAILABLE DATASETS AND MODELS")
    print("=" * 60)

    for dataset in sorted(df['Dataset_Key'].unique()):
        dataset_data = df[df['Dataset_Key'] == dataset]
        dataset_name = dataset_data['Dataset_Name'].iloc[0]
        models = sorted(dataset_data['Model_Name'].unique())
        targets = sorted(dataset_data['Target_Name'].unique())

        print(f"\n📊 {dataset_name.upper()}")
        print(f"   Models: {', '.join(models)}")
        print(f"   Targets: {', '.join(targets)}")
        print(f"   Total Results: {len(dataset_data)}")


def show_performance_by_dataset_model(df):
    """Show performance in pivot table format"""
    print("\n🏆 PERFORMANCE BY DATASET AND MODEL")
    print("=" * 80)

    # Create pivot table
    pivot = df.pivot_table(
        values='Test_Accuracy',
        index=['Dataset_Key', 'Target_Key'],
        columns='Model_Name',
        aggfunc='mean'
    )

    print(pivot.round(4))


def show_model_rankings(df):
    """Show overall model performance rankings"""
    print("\n📊 AVERAGE PERFORMANCE BY MODEL (All Datasets)")
    print("=" * 60)

    model_stats = df.groupby('Model_Name').agg({
        'Test_Accuracy': ['mean', 'std', 'count'],
        'CV_Mean': 'mean'
    }).round(4)

    model_stats.columns = ['Test_Accuracy',
                           'Std_Test_Accuracy', 'Total_Tasks', 'CV_Mean']
    model_stats = model_stats.sort_values('Test_Accuracy', ascending=False)

    print(model_stats)


def show_best_models_per_task(df):
    """Show best performing model for each dataset-target combination"""
    print("\n🥇 BEST PERFORMING MODEL PER DATASET-TARGET")
    print("=" * 70)

    best_models = []

    for dataset in sorted(df['Dataset_Key'].unique()):
        for target in sorted(df[df['Dataset_Key'] == dataset]['Target_Key'].unique()):
            subset = df[(df['Dataset_Key'] == dataset)
                        & (df['Target_Key'] == target)]
            if not subset.empty:
                best_idx = subset['Test_Accuracy'].idxmax()
                best_row = subset.loc[best_idx]
                best_models.append({
                    'Dataset': dataset,
                    'Target': best_row['Target_Name'],
                    'Model': best_row['Model_Name'],
                    'Test_Accuracy': best_row['Test_Accuracy']
                })

    best_df = pd.DataFrame(best_models)
    print(best_df.to_string(index=False, float_format='%.4f'))

    return best_df


def show_model_wins_summary(best_df):
    """Show summary of model wins"""
    print("\n🏆 MODEL WINS SUMMARY")
    print("=" * 50)

    wins = best_df['Model'].value_counts()
    total_tasks = len(best_df)

    for model, count in wins.items():
        percentage = (count / total_tasks) * 100
        print(f"{model}: {count}/{total_tasks} wins ({percentage:.1f}%)")


def analyze_peecom_performance(df):
    """Special analysis for PEECOM model"""
    print('\n' + '='*80)
    print('🚀 PEECOM MODEL ANALYSIS')
    print('='*80)

    peecom_data = df[df['Model_Key'] == 'peecom']

    if peecom_data.empty:
        print("❌ No PEECOM results found!")
        return

    # PEECOM vs others comparison
    comparison_results = []

    for dataset in df['Dataset_Key'].unique():
        for target in df[df['Dataset_Key'] == dataset]['Target_Key'].unique():
            subset = df[(df['Dataset_Key'] == dataset)
                        & (df['Target_Key'] == target)]

            if len(subset) > 1:  # At least 2 models to compare
                peecom_score = subset[subset['Model_Key']
                                      == 'peecom']['Test_Accuracy']
                other_scores = subset[subset['Model_Key']
                                      != 'peecom']['Test_Accuracy']

                if not peecom_score.empty and not other_scores.empty:
                    peecom_val = peecom_score.iloc[0]
                    best_other = other_scores.max()
                    improvement = peecom_val - best_other

                    comparison_results.append({
                        'Dataset': dataset,
                        'Target': target,
                        'PEECOM_Score': peecom_val,
                        'Best_Other': best_other,
                        'Improvement': improvement,
                        'Better_Than_Best': improvement > 0
                    })

    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)

        print(f"📊 PEECOM vs Best Traditional Model:")
        print(
            f"  Tasks where PEECOM wins: {sum(comparison_df['Better_Than_Best'])}/{len(comparison_df)}")
        print(
            f"  Average PEECOM score: {comparison_df['PEECOM_Score'].mean():.4f}")
        print(
            f"  Average best traditional: {comparison_df['Best_Other'].mean():.4f}")
        print(
            f"  Average improvement: {comparison_df['Improvement'].mean():.4f}")

        # Show biggest wins and losses
        print(f"\n🏆 Biggest PEECOM Wins:")
        wins = comparison_df[comparison_df['Better_Than_Best']].nlargest(
            3, 'Improvement')
        for _, row in wins.iterrows():
            print(
                f"  {row['Dataset']}/{row['Target']}: +{row['Improvement']:.4f}")

        if len(comparison_df[~comparison_df['Better_Than_Best']]) > 0:
            print(f"\n📉 Areas for PEECOM Improvement:")
            losses = comparison_df[~comparison_df['Better_Than_Best']].nsmallest(
                3, 'Improvement')
            for _, row in losses.iterrows():
                print(
                    f"  {row['Dataset']}/{row['Target']}: {row['Improvement']:.4f}")


def generate_comprehensive_summary(df):
    """Generate a comprehensive summary table"""
    print('\n' + '='*80)
    print('📋 COMPREHENSIVE SUMMARY TABLE')
    print('='*80)

    # Create comprehensive pivot table
    summary_data = []

    for dataset in sorted(df['Dataset_Key'].unique()):
        dataset_name = df[df['Dataset_Key'] == dataset]['Dataset_Name'].iloc[0]
        for target in sorted(df[df['Dataset_Key'] == dataset]['Target_Key'].unique()):
            target_name = df[(df['Dataset_Key'] == dataset) & (
                df['Target_Key'] == target)]['Target_Name'].iloc[0]

            row = {'Dataset': dataset_name, 'Target': target_name}

            # Get scores for each model
            subset = df[(df['Dataset_Key'] == dataset)
                        & (df['Target_Key'] == target)]
            for model in MODELS:
                model_data = subset[subset['Model_Key'] == model]
                if not model_data.empty:
                    row[model_data['Model_Name'].iloc[0]
                        ] = model_data['Test_Accuracy'].iloc[0]
                else:
                    row[f'{model.replace("_", " ").title()}'] = np.nan

            # Find best model and score
            scores = {k: v for k, v in row.items() if k not in [
                'Dataset', 'Target'] and not pd.isna(v)}
            if scores:
                best_model = max(scores, key=scores.get)
                best_score = scores[best_model]
                row['Best_Model'] = best_model
                row['Best_Score'] = best_score

            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Set proper index for display
    summary_df = summary_df.set_index(['Dataset', 'Target'])

    print(summary_df.round(4))

    # Save to CSV
    output_file = 'output/comprehensive_model_comparison.csv'
    summary_df.to_csv(output_file)
    print(f"\n💾 Detailed results saved to: {output_file}")

    return summary_df


def show_detailed_performance_table(df):
    """Show detailed performance table with CV scores"""
    print("\n📋 DETAILED PERFORMANCE TABLE")
    print("=" * 90)

    # Group and format the data
    detailed_data = []

    for _, row in df.iterrows():
        cv_score_str = f"{row['CV_Mean']:.3f} ± {row['CV_Std']:.3f}" if not pd.isna(
            row['CV_Std']) else f"{row['CV_Mean']:.3f} ± nan"

        detailed_data.append({
            'Dataset': row['Dataset_Key'],
            'Model': row['Model_Name'],
            'Target': row['Target_Name'],
            'Performance': f"{row['Test_Accuracy']:.1%}",
            'CV_Score': cv_score_str
        })

    detailed_df = pd.DataFrame(detailed_data)
    detailed_df = detailed_df.sort_values(['Dataset', 'Target', 'Model'])

    print(detailed_df.to_string(index=False))


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description='Compare model performance across datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_models.py                     # Compare all models on all datasets
  python compare_models.py --dataset cmohs     # Compare models on CMOHS only
  python compare_models.py --model peecom      # Compare PEECOM across datasets
  python compare_models.py --summary           # Show summary only
  python compare_models.py --datasets cmohs,equipmentad  # Multiple datasets
        """
    )

    parser.add_argument('--dataset', type=str,
                        help='Filter results for specific dataset')
    parser.add_argument('--datasets', type=str,
                        help='Filter results for multiple datasets (comma-separated)')
    parser.add_argument('--model', type=str,
                        choices=['random_forest',
                                 'logistic_regression', 'svm', 'peecom'],
                        help='Filter results for specific model')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary statistics only')

    args = parser.parse_args()

    print("🔬 PEECOM MULTI-DATASET MODEL COMPARISON")
    print("=" * 80)

    # Determine datasets to analyze
    datasets_filter = None
    if args.datasets:
        datasets_filter = [d.strip() for d in args.datasets.split(',')]
    elif args.dataset:
        datasets_filter = [args.dataset]

    # Load results
    df = load_model_results(datasets_filter)

    if df.empty:
        print("❌ No results found! Please train some models first.")
        return

    # Apply model filter if specified
    if args.model:
        df = df[df['Model_Key'] == args.model]
        if df.empty:
            print(f"❌ No results found for model: {args.model}")
            return

    print(f"📊 Loaded {len(df)} result entries")
    print(f"📦 Datasets: {len(df['Dataset_Key'].unique())}")
    print(f"🤖 Models: {len(df['Model_Key'].unique())}")
    print(f"🎯 Unique tasks: {len(df.groupby(['Dataset_Key', 'Target_Key']))}")

    # Show different views based on arguments
    if args.summary:
        show_model_rankings(df)
        show_best_models_per_task(df)
    else:
        # Full analysis
        show_dataset_overview(df)
        show_performance_by_dataset_model(df)
        show_model_rankings(df)

        best_df = show_best_models_per_task(df)
        show_model_wins_summary(best_df)

        # PEECOM-specific analysis
        if 'peecom' in df['Model_Key'].values:
            analyze_peecom_performance(df)

        # Comprehensive summary
        generate_comprehensive_summary(df)

        # Detailed table
        show_detailed_performance_table(df)

    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
