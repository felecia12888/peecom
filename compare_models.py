#!/usr/bin/env python3
import json
import pandas as pd
import os


def compare_model_performance():
    models = ['random_forest', 'logistic_regression', 'svm', 'peecom']
    targets = ['cooler_condition', 'valve_condition',
               'pump_leakage', 'accumulator_pressure', 'stable_flag']

    results = []
    for model in models:
        summary_file = f'output/models/{model}/all_targets_summary.json'
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    for target in targets:
                        if target in data and 'test_accuracy' in data[target]:
                            results.append({
                                'Model': model.replace('_', ' ').title(),
                                'Target': target.replace('_', ' ').title(),
                                'Test_Accuracy': round(data[target]['test_accuracy'], 4),
                                'CV_Mean': round(data[target]['cv_mean'], 4),
                                'CV_Std': round(data[target]['cv_std'], 4)
                            })
            except Exception as e:
                print(f'Error reading {summary_file}: {e}')
        else:
            print(f'Summary file not found: {summary_file}')

    if not results:
        print("No results found!")
        return

    df = pd.DataFrame(results)

    print('\n=== MODEL PERFORMANCE COMPARISON ===\n')
    pivot_table = df.pivot(
        index='Target', columns='Model', values='Test_Accuracy')
    print(pivot_table.to_string())

    print('\n=== AVERAGE PERFORMANCE BY MODEL ===\n')
    avg_by_model = df.groupby(
        'Model')[['Test_Accuracy', 'CV_Mean']].mean().round(4)
    avg_by_model = avg_by_model.sort_values('Test_Accuracy', ascending=False)
    print(avg_by_model.to_string())

    print('\n=== BEST PERFORMING MODEL PER TARGET ===\n')
    best_per_target = df.loc[df.groupby('Target')['Test_Accuracy'].idxmax()]
    print(best_per_target[['Target', 'Model',
          'Test_Accuracy']].to_string(index=False))

    print('\n=== DETAILED PERFORMANCE TABLE ===\n')
    detailed = df.copy()
    detailed['Performance'] = detailed['Test_Accuracy'].apply(
        lambda x: f'{x:.1%}')
    detailed['CV_Score'] = detailed.apply(
        lambda x: f"{x['CV_Mean']:.3f} ± {x['CV_Std']:.3f}", axis=1)
    detailed_display = detailed[[
        'Model', 'Target', 'Performance', 'CV_Score']].sort_values(['Model', 'Target'])
    print(detailed_display.to_string(index=False))


if __name__ == "__main__":
    compare_model_performance()
