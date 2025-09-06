#!/usr/bin/env python3
"""
Enhanced Model Performance Comparison
Compares all models across all datasets with comprehensive analysis
"""
import json
import pandas as pd
import os
import numpy as np
from collections import defaultdict

# Dataset configuration with their targets
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
        'targets': ['condition']
    },
    'multivariatetsd': {
        'name': 'Multivariate Time Series',
        'targets': ['engine_id', 'cycle']
    },
    'sensord': {
        'name': 'Sensor Data',
        'targets': ['condition']
    },
    'smartmd': {
        'name': 'Smart Maintenance Dataset',
        'targets': ['anomaly_flag', 'machine_status', 'maintenance_required']
    }
}

MODELS = ['random_forest', 'logistic_regression', 'svm', 'peecom']


def load_model_results():
    """Load results from all trained models"""
    results = []

    for dataset_key, dataset_info in DATASET_CONFIG.items():
        for model in MODELS:
            summary_file = f'output/models/{model}/all_targets_summary.json'

            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        data = json.load(f)

                        for target in dataset_info['targets']:
                            if target in data and 'test_accuracy' in data[target]:
                                results.append({
                                    'Dataset': dataset_info['name'],
                                    'Dataset_Key': dataset_key,
                                    'Model': model.replace('_', ' ').title(),
                                    'Model_Key': model,
                                    'Target': target.replace('_', ' ').title(),
                                    'Target_Key': target,
                                    'Test_Accuracy': data[target]['test_accuracy'],
                                    'CV_Mean': data[target]['cv_mean'],
                                    'CV_Std': data[target]['cv_std'],
                                    'Train_Accuracy': data[target].get('train_accuracy', None),
                                    'F1_Score': data[target].get('f1_score', None),
                                    'Precision': data[target].get('precision', None),
                                    'Recall': data[target].get('recall', None)
                                })
                except Exception as e:
                    print(f'⚠️  Error reading {summary_file}: {e}')
            else:
                print(f'📄 Summary file not found: {summary_file}')

    return pd.DataFrame(results)


def analyze_overall_performance(df):
    """Overall performance analysis"""
    print('\n' + '='*80)
    print('🏆 OVERALL MODEL PERFORMANCE RANKING')
    print('='*80)

    if df.empty:
        print("❌ No results found!")
        return

    # Overall model ranking
    overall_ranking = df.groupby('Model').agg({
        'Test_Accuracy': ['mean', 'std', 'count'],
        'CV_Mean': 'mean'
    }).round(4)

    overall_ranking.columns = ['Avg_Test_Accuracy',
                               'Std_Test_Accuracy', 'Total_Tasks', 'Avg_CV_Score']
    overall_ranking = overall_ranking.sort_values(
        'Avg_Test_Accuracy', ascending=False)

    print(overall_ranking.to_string())

    # Performance by dataset
    print('\n' + '='*80)
    print('📊 PERFORMANCE BY DATASET')
    print('='*80)

    dataset_performance = df.groupby(['Dataset', 'Model'])[
        'Test_Accuracy'].mean().unstack()
    print(dataset_performance.round(4).to_string())

    return overall_ranking, dataset_performance


def analyze_by_dataset(df):
    """Detailed analysis by dataset"""
    print('\n' + '='*80)
    print('🎯 DETAILED ANALYSIS BY DATASET')
    print('='*80)

    for dataset in df['Dataset'].unique():
        dataset_data = df[df['Dataset'] == dataset]

        print(f'\n--- {dataset.upper()} ---')

        # Best model per target
        best_per_target = dataset_data.loc[dataset_data.groupby(
            'Target')['Test_Accuracy'].idxmax()]

        if not best_per_target.empty:
            print(f"🥇 Best Models per Target:")
            for _, row in best_per_target.iterrows():
                print(
                    f"  {row['Target']}: {row['Model']} ({row['Test_Accuracy']:.4f})")

            # Average performance by model for this dataset
            avg_performance = dataset_data.groupby(
                'Model')['Test_Accuracy'].agg(['mean', 'count']).round(4)
            avg_performance.columns = ['Avg_Accuracy', 'Targets_Count']
            avg_performance = avg_performance.sort_values(
                'Avg_Accuracy', ascending=False)

            print(f"\n📈 Average Performance by Model:")
            print(avg_performance.to_string())


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
        top_wins = comparison_df.nlargest(3, 'Improvement')
        for _, row in top_wins.iterrows():
            print(
                f"  {row['Dataset']}/{row['Target']}: +{row['Improvement']:.4f}")

        if any(comparison_df['Improvement'] < 0):
            print(f"\n📉 Areas for PEECOM Improvement:")
            losses = comparison_df[comparison_df['Improvement'] < 0].nsmallest(
                3, 'Improvement')
            for _, row in losses.iterrows():
                print(
                    f"  {row['Dataset']}/{row['Target']}: {row['Improvement']:.4f}")


def generate_summary_table(df):
    """Generate a comprehensive summary table"""
    print('\n' + '='*80)
    print('📋 COMPREHENSIVE SUMMARY TABLE')
    print('='*80)

    # Create pivot table
    summary_pivot = df.pivot_table(
        index=['Dataset', 'Target'],
        columns='Model',
        values='Test_Accuracy',
        aggfunc='mean'
    ).round(4)

    # Add best model column
    summary_pivot['Best_Model'] = summary_pivot.idxmax(axis=1)
    summary_pivot['Best_Score'] = summary_pivot.max(axis=1, numeric_only=True)

    print(summary_pivot.to_string())

    # Save to CSV
    output_file = 'output/comprehensive_model_comparison.csv'
    summary_pivot.to_csv(output_file)
    print(f"\n💾 Detailed results saved to: {output_file}")


def main():
    print("🔍 LOADING COMPREHENSIVE MODEL COMPARISON")
    print("=" * 50)

    # Load all results
    df = load_model_results()

    if df.empty:
        print("❌ No training results found!")
        print("🚀 Run: python batch_train_all.py first!")
        return

    print(f"📊 Loaded {len(df)} result entries")
    print(f"📦 Datasets: {len(df['Dataset'].unique())}")
    print(f"🤖 Models: {len(df['Model'].unique())}")
    print(f"🎯 Unique tasks: {len(df.groupby(['Dataset', 'Target']))}")

    # Run all analyses
    analyze_overall_performance(df)
    analyze_by_dataset(df)
    analyze_peecom_performance(df)
    generate_summary_table(df)

    print('\n' + '='*80)
    print('✅ COMPREHENSIVE ANALYSIS COMPLETE!')
    print('='*80)


if __name__ == "__main__":
    main()
