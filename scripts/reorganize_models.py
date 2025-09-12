#!/usr/bin/env python3
"""
Reorganize PEECOM Model Structure
================================

This script reorganizes the model output structure from:
  output/models/{model}/{target}/*
to:
  output/models/{dataset}/{model}/{target}/*

This provides better organization and allows proper dataset-based comparisons.
"""

import os
import json
import shutil
from pathlib import Path


# Mapping targets to their datasets based on what we know
TARGET_TO_DATASET = {
    # CMOHS targets (hydraulic system)
    'cooler_condition': 'cmohs',
    'valve_condition': 'cmohs',
    'pump_leakage': 'cmohs',
    'accumulator_pressure': 'cmohs',
    'stable_flag': 'cmohs',

    # Equipment Anomaly Detection targets
    'anomaly': 'equipmentad',
    'equipment_type': 'equipmentad',
    'location': 'equipmentad',

    # Energy Classification targets
    'status': 'mlclassem',
    'region': 'mlclassem',

    # Motor Vibration targets
    'condition': 'motorvd',

    # Multivariate Time Series targets
    'engine_id': 'multivariatetsd',
    'cycle': 'multivariatetsd',

    # Sensor Data targets
    'file_id': 'sensord',

    # Smart Maintenance targets
    'anomaly_flag': 'smartmd',
    'machine_status': 'smartmd',
    'maintenance_required': 'smartmd',
}


def reorganize_models():
    """Reorganize existing model structure"""

    models_dir = Path('output/models')
    if not models_dir.exists():
        print("❌ No models directory found!")
        return

    backup_dir = Path('output/models_backup_old_structure')

    # Create backup of current structure
    print("📦 Creating backup of current structure...")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(models_dir, backup_dir)
    print(f"✅ Backup created at: {backup_dir}")

    # Get all models
    models = ['random_forest', 'logistic_regression', 'svm', 'peecom']

    # Create new directory structure
    new_models_dir = Path('output/models_new')
    if new_models_dir.exists():
        shutil.rmtree(new_models_dir)
    new_models_dir.mkdir(parents=True)

    print("\n🔄 Reorganizing model structure...")

    for model in models:
        old_model_dir = models_dir / model
        if not old_model_dir.exists():
            print(f"⚠️  Skipping {model} - directory not found")
            continue

        print(f"\n📂 Processing {model}...")

        # Get all target directories for this model
        for item in old_model_dir.iterdir():
            if item.is_dir() and item.name in TARGET_TO_DATASET:
                target = item.name
                dataset = TARGET_TO_DATASET[target]

                # Create new structure: dataset/model/target
                new_target_dir = new_models_dir / dataset / model / target
                new_target_dir.mkdir(parents=True, exist_ok=True)

                # Copy all files from old target dir to new target dir
                for file_item in item.iterdir():
                    if file_item.is_file():
                        shutil.copy2(
                            file_item, new_target_dir / file_item.name)
                    elif file_item.is_dir():
                        shutil.copytree(
                            file_item, new_target_dir / file_item.name, dirs_exist_ok=True)

                print(f"   ✅ {target} → {dataset}/{model}/{target}")

            elif item.name not in ['all_targets_summary.json', 'comprehensive_figures'] and item.is_dir():
                print(f"   ⚠️  Unknown target: {item.name} (skipped)")

    # Replace old structure with new structure
    print(f"\n🔄 Replacing old structure...")
    shutil.rmtree(models_dir)
    shutil.move(new_models_dir, models_dir)

    print(f"\n✅ Reorganization complete!")
    print(f"📁 New structure: output/models/{{dataset}}/{{model}}/{{target}}/")
    print(f"📦 Backup available at: {backup_dir}")


def create_dataset_summaries():
    """Create dataset-specific summary files"""

    models_dir = Path('output/models')

    print("\n📊 Creating dataset-specific summary files...")

    for dataset_dir in models_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        print(f"\n📈 Processing dataset: {dataset}")

        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model = model_dir.name
            summary_data = {}

            # Collect results from all targets for this model/dataset
            for target_dir in model_dir.iterdir():
                if not target_dir.is_dir():
                    continue

                target = target_dir.name
                results_file = target_dir / 'training_results.json'

                if results_file.exists():
                    try:
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                            summary_data[target] = results
                    except Exception as e:
                        print(f"   ⚠️  Error reading {results_file}: {e}")

            # Save dataset-specific summary
            if summary_data:
                summary_file = model_dir / 'dataset_summary.json'
                with open(summary_file, 'w') as f:
                    json.dump(summary_data, f, indent=2)
                print(f"   ✅ {model}: {len(summary_data)} targets saved")


def show_new_structure():
    """Display the new structure"""

    models_dir = Path('output/models')

    print(f"\n📁 NEW ORGANIZED STRUCTURE:")
    print("=" * 50)

    for dataset_dir in sorted(models_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        print(f"\n📊 {dataset.upper()}")

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            model = model_dir.name
            targets = [t.name for t in model_dir.iterdir() if t.is_dir()]

            print(f"   🤖 {model}: {len(targets)} targets")
            for target in sorted(targets)[:5]:  # Show first 5
                print(f"      📌 {target}")
            if len(targets) > 5:
                print(f"      📌 ... and {len(targets)-5} more")


def main():
    """Execute reorganization"""

    print("🏗️  PEECOM MODEL STRUCTURE REORGANIZATION")
    print("=" * 60)
    print("Reorganizing from: output/models/{model}/{target}/*")
    print("             to: output/models/{dataset}/{model}/{target}/*")
    print()

    # Check if we need to reorganize
    models_dir = Path('output/models')
    needs_reorganization = False

    if models_dir.exists():
        # Check if current structure is old (models directly under output/models)
        old_structure_models = ['random_forest',
                                'logistic_regression', 'svm', 'peecom']
        for model in old_structure_models:
            if (models_dir / model).exists():
                needs_reorganization = True
                break

    if needs_reorganization:
        reorganize_models()
        create_dataset_summaries()
        show_new_structure()
    else:
        print("✅ Structure already organized or no models found!")
        if models_dir.exists():
            show_new_structure()


if __name__ == "__main__":
    main()
