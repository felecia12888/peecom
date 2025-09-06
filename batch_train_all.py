#!/usr/bin/env python3
"""
Comprehensive Batch Training Script
Trains all models on all datasets automatically
"""
import os
import subprocess
import sys
import time
import threading
from datetime import datetime

# Define all available datasets and their respective models
DATASET_CONFIG = {
    'cmohs': {
        'targets': ['cooler_condition', 'valve_condition', 'pump_leakage', 'accumulator_pressure', 'stable_flag'],
        'models': ['random_forest', 'logistic_regression', 'svm', 'peecom']
    },
    'equipmentad': {
        'targets': ['anomaly', 'equipment_type', 'location'],
        'models': ['random_forest', 'logistic_regression', 'svm', 'peecom']
    },
    'mlclassem': {
        'targets': ['status', 'region', 'equipment_type'],
        'models': ['random_forest', 'logistic_regression', 'svm', 'peecom']
    },
    'motorvd': {
        # Excluding file_id as it was cleaned out as leakage
        'targets': ['condition'],
        'models': ['random_forest', 'logistic_regression', 'svm', 'peecom']
    },
    'multivariatetsd': {
        'targets': ['engine_id', 'cycle'],
        'models': ['random_forest', 'logistic_regression', 'svm', 'peecom']
    },
    'sensord': {
        # Excluding file_id as it was cleaned out as leakage
        'targets': ['condition'],
        'models': ['random_forest', 'logistic_regression', 'svm', 'peecom']
    },
    'smartmd': {
        'targets': ['anomaly_flag', 'machine_status', 'maintenance_required'],
        'models': ['random_forest', 'logistic_regression', 'svm', 'peecom']
    }
}


def check_dataset_exists(dataset):
    """Check if processed dataset exists"""
    return os.path.exists(f'output/processed_data/{dataset}/X_full.csv')


def print_progress_dots(stop_event):
    """Print progress dots while training is running"""
    count = 0
    while not stop_event.is_set():
        if count % 10 == 0:
            print(f"\n⏳ Training in progress", end="", flush=True)
        print(".", end="", flush=True)
        count += 1
        time.sleep(2)  # Print dot every 2 seconds


def run_training(dataset, model):
    """Run training for a specific dataset-model combination with real-time progress"""
    print(f"\n{'='*60}")
    print(f"🚀 Training {model.upper()} on {dataset.upper()}")
    print(f"{'='*60}")

    cmd = [sys.executable, 'main.py', '--dataset',
           dataset, '--model', model, '--eval-all']

    # Set different timeouts for different models
    timeout_minutes = {
        'peecom': 45,      # PEECOM gets more time
        'svm': 20,         # SVM can be slow
        'random_forest': 15,
        'logistic_regression': 10
    }
    timeout = timeout_minutes.get(model, 15) * 60  # Convert to seconds

    start_time = time.time()

    try:
        # Start progress indicator thread
        stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=print_progress_dots, args=(stop_event,))
        progress_thread.daemon = True
        progress_thread.start()

        # Run the training with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Monitor the process
        output_lines = []
        while True:
            line = process.stdout.readline()
            if line:
                output_lines.append(line.strip())
                # Show important progress lines
                if any(keyword in line.lower() for keyword in ['training', 'accuracy', 'completed', 'error', 'evaluating']):
                    stop_event.set()  # Stop dots temporarily
                    print(f"\n📊 {line.strip()}")
                    stop_event.clear()  # Resume dots
                    progress_thread = threading.Thread(
                        target=print_progress_dots, args=(stop_event,))
                    progress_thread.daemon = True
                    progress_thread.start()

            elif process.poll() is not None:
                break

            # Check timeout
            if time.time() - start_time > timeout:
                print(
                    f"\n⏰ Timeout reached ({timeout_minutes.get(model, 15)} minutes)")
                process.terminate()
                stop_event.set()
                return False, 0

        stop_event.set()  # Stop progress thread

        elapsed = time.time() - start_time

        if process.returncode == 0:
            print(
                f"\n✅ SUCCESS: {dataset}/{model} completed in {elapsed:.1f}s")
            return True, elapsed
        else:
            print(f"\n❌ FAILED: {dataset}/{model}")
            # Show last few lines of output for debugging
            if output_lines:
                print("Last output lines:")
                for line in output_lines[-5:]:
                    print(f"  {line}")
            return False, 0

    except Exception as e:
        stop_event.set()
        print(f"\n💥 ERROR: {dataset}/{model} - {e}")
        return False, 0


def main():
    print("🎯 COMPREHENSIVE BATCH TRAINING")
    print("=" * 50)

    # Check available datasets
    available_datasets = {name: config for name, config in DATASET_CONFIG.items()
                          if check_dataset_exists(name)}

    if not available_datasets:
        print("❌ No processed datasets found!")
        return

    print(f"📊 Found {len(available_datasets)} datasets:")
    for dataset in available_datasets:
        print(f"  ✅ {dataset}")

    # Training summary
    total_combinations = sum(len(config['models'])
                             for config in available_datasets.values())
    print(f"\n🎯 Total training combinations: {total_combinations}")

    # Ask user if they want to skip PEECOM for faster training
    print(f"\n⚠️  NOTE: PEECOM training can take 45+ minutes per dataset")
    skip_peecom = input(
        "Skip PEECOM for faster training? (y/N): ").lower().startswith('y')

    if skip_peecom:
        print("⏩ Skipping PEECOM models")
        for config in available_datasets.values():
            if 'peecom' in config['models']:
                config['models'].remove('peecom')
        total_combinations = sum(len(config['models'])
                                 for config in available_datasets.values())
        print(f"🎯 Reduced to {total_combinations} combinations")

    # Start batch training
    start_time = datetime.now()
    results = {'success': [], 'failed': []}
    total_time = 0
    completed = 0

    for dataset, config in available_datasets.items():
        for model in config['models']:
            completed += 1
            print(f"\n🎯 Progress: {completed}/{total_combinations}")

            success, elapsed = run_training(dataset, model)

            if success:
                results['success'].append(f"{dataset}/{model}")
                total_time += elapsed
            else:
                results['failed'].append(f"{dataset}/{model}")

                # Ask if user wants to continue after failure
                if model == 'peecom':
                    continue_after_fail = input(
                        f"\n❓ PEECOM failed on {dataset}. Continue with other models? (Y/n): ")
                    if continue_after_fail.lower().startswith('n'):
                        print("🛑 Stopping batch training")
                        break

    # Final summary
    end_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"🏁 BATCH TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"⏱️  Total Duration: {end_time - start_time}")
    print(f"✅ Successful: {len(results['success'])}/{total_combinations}")
    print(f"❌ Failed: {len(results['failed'])}/{total_combinations}")

    if results['success']:
        print(f"\n🎉 SUCCESSFUL TRAINING RUNS:")
        for item in results['success']:
            print(f"  ✅ {item}")

    if results['failed']:
        print(f"\n💥 FAILED TRAINING RUNS:")
        for item in results['failed']:
            print(f"  ❌ {item}")

    print(f"\n🎯 Now run: python compare_models_enhanced.py")
    print(f"📊 To see comprehensive performance comparison!")


if __name__ == "__main__":
    main()
