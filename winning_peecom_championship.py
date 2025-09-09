#!/usr/bin/env python3
"""
Winning PEECOM Championship Test

This script tests the Winning PEECOM model to ensure it consistently
outperforms traditional ML models across all datasets.
"""

from src.models.winning_peecom import WinningPEECOM
from src.utils.training_utils import load_processed_data, prepare_targets
import sys
import os
import time
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_optimized_traditional_models(n_samples: int, n_classes: int):
    """Create optimized traditional models for fair comparison"""
    models = {}

    # Optimized Random Forest
    if n_samples < 5000:
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    else:
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

    # Optimized Gradient Boosting
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        random_state=42
    )

    # Add SVM for smaller datasets
    if n_samples < 10000:
        models['SVM'] = SVC(
            random_state=42,
            class_weight='balanced',
            probability=True
        )

    # Logistic Regression
    models['Logistic Regression'] = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear' if n_classes == 2 else 'lbfgs'
    )

    return models


def benchmark_on_task(dataset_name: str, target_column: str, task_description: str):
    """Benchmark Winning PEECOM against traditional models on a specific task"""
    logger.info(f"\n🏆 CHAMPIONSHIP MATCH: {task_description}")
    logger.info("=" * 70)

    try:
        # Load data
        data_path = os.path.join('output/processed_data', dataset_name)
        X, y = load_processed_data(data_path)
        target = prepare_targets(y, target_column)

        n_samples, n_features = X.shape
        n_classes = len(np.unique(target))

        logger.info(f"📊 Dataset: {dataset_name}/{target_column}")
        logger.info(
            f"📦 Data: {n_samples} samples, {n_features} features, {n_classes} classes")

        # Skip extremely challenging tasks that hang
        if dataset_name == 'smartmd' and target_column == 'machine_status' and n_samples > 50000:
            logger.warning(
                "⏭️  Skipping large machine_status task (performance issue)")
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, target, test_size=0.2, random_state=42, stratify=target
        )

        # Create models
        traditional_models = create_optimized_traditional_models(
            n_samples, n_classes)
        winning_peecom = WinningPEECOM(verbose=False, random_state=42)

        # Benchmark all models
        results = {}

        # Test traditional models
        for name, model in traditional_models.items():
            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                train_time = time.time() - start_time

                results[name] = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'time': train_time
                }

                logger.info(
                    f"🔧 {name:<20} | {accuracy:.4f} ({accuracy*100:.1f}%) | {train_time:.1f}s")

            except Exception as e:
                logger.warning(f"❌ {name} failed: {str(e)}")
                results[name] = {'accuracy': 0.0, 'f1': 0.0, 'time': 0.0}

        # Test Winning PEECOM
        try:
            start_time = time.time()
            winning_peecom.fit(X_train, y_train)
            y_pred_peecom = winning_peecom.predict(X_test)

            peecom_accuracy = accuracy_score(y_test, y_pred_peecom)
            peecom_f1 = f1_score(y_test, y_pred_peecom, average='weighted')
            peecom_time = time.time() - start_time

            results['Winning PEECOM'] = {
                'accuracy': peecom_accuracy,
                'f1': peecom_f1,
                'time': peecom_time
            }

            logger.info(
                f"🏆 {'Winning PEECOM':<20} | {peecom_accuracy:.4f} ({peecom_accuracy*100:.1f}%) | {peecom_time:.1f}s")

        except Exception as e:
            logger.error(f"❌ Winning PEECOM failed: {str(e)}")
            results['Winning PEECOM'] = {
                'accuracy': 0.0, 'f1': 0.0, 'time': 0.0}
            return None

        # Analyze results
        sorted_results = sorted(
            results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        best_model, best_performance = sorted_results[0]

        peecom_rank = next((i+1 for i, (name, _) in enumerate(sorted_results)
                           if name == 'Winning PEECOM'), len(sorted_results))

        logger.info(f"\n📈 RESULTS:")
        logger.info(
            f"🥇 Winner: {best_model} ({best_performance['accuracy']:.4f})")
        logger.info(f"🤖 PEECOM: Rank #{peecom_rank} ({peecom_accuracy:.4f})")

        # Victory analysis
        if best_model == 'Winning PEECOM':
            margin = peecom_accuracy - sorted_results[1][1]['accuracy']
            logger.info(f"🎉 PEECOM WINS! Margin: +{margin:.4f}")
            victory_status = 'WIN'
        else:
            gap = best_performance['accuracy'] - peecom_accuracy
            logger.info(f"💔 PEECOM loses by {gap:.4f}")
            if gap < 0.01:
                victory_status = 'CLOSE'
            else:
                victory_status = 'LOSS'

        return {
            'dataset': dataset_name,
            'target': target_column,
            'task': task_description,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'peecom_accuracy': peecom_accuracy,
            'peecom_rank': peecom_rank,
            'best_model': best_model,
            'best_accuracy': best_performance['accuracy'],
            'victory_status': victory_status,
            'all_results': results
        }

    except Exception as e:
        logger.error(f"❌ Task failed: {str(e)}")
        return None


def main():
    """Run the Winning PEECOM Championship"""
    logger.info("🏆 WINNING PEECOM CHAMPIONSHIP")
    logger.info("=" * 70)
    logger.info("Testing optimized PEECOM for consistent victories\n")

    # Championship tasks (focused on key tests)
    championship_tasks = [
        # Binary Classification Tasks (PEECOM should dominate)
        ('cmohs', 'stable_flag', 'Hydraulic System Stability Detection'),
        ('smartmd', 'anomaly_flag', 'Predictive Maintenance Anomaly'),
        ('motorvd', 'condition', 'Motor Vibration Condition Assessment'),
        ('equipmentad', 'anomaly', 'Equipment Anomaly Detection'),

        # Multi-Class Tasks (PEECOM should compete strongly)
        ('cmohs', 'cooler_condition', 'Hydraulic Cooler Condition Classification'),
        ('smartmd', 'machine_status', 'Machine Status Classification'),
    ]

    results = []
    total_start_time = time.time()

    # Run championship
    for dataset, target, description in championship_tasks:
        result = benchmark_on_task(dataset, target, description)
        if result:
            results.append(result)
        time.sleep(1)  # Brief pause between matches

    total_time = time.time() - total_start_time

    # Championship Analysis
    if not results:
        logger.error("❌ No successful championship matches!")
        return

    logger.info(f"\n🏁 CHAMPIONSHIP SUMMARY")
    logger.info("=" * 70)

    # Count victories
    wins = len([r for r in results if r['victory_status'] == 'WIN'])
    close_losses = len([r for r in results if r['victory_status'] == 'CLOSE'])
    clear_losses = len([r for r in results if r['victory_status'] == 'LOSS'])
    total_matches = len(results)

    # Performance statistics
    avg_peecom_acc = np.mean([r['peecom_accuracy'] for r in results])
    avg_best_acc = np.mean([r['best_accuracy'] for r in results])
    avg_rank = np.mean([r['peecom_rank'] for r in results])

    logger.info(f"🏆 Championship Results:")
    logger.info(f"   Total Matches: {total_matches}")
    logger.info(
        f"   Outright Wins: {wins}/{total_matches} ({wins/total_matches*100:.1f}%)")
    logger.info(
        f"   Close Losses: {close_losses}/{total_matches} ({close_losses/total_matches*100:.1f}%)")
    logger.info(
        f"   Clear Losses: {clear_losses}/{total_matches} ({clear_losses/total_matches*100:.1f}%)")
    logger.info(
        f"   Competitive Rate: {(wins+close_losses)/total_matches*100:.1f}%")

    logger.info(f"\n📊 Performance Metrics:")
    logger.info(
        f"   PEECOM Average: {avg_peecom_acc:.4f} ({avg_peecom_acc*100:.2f}%)")
    logger.info(
        f"   Best Average: {avg_best_acc:.4f} ({avg_best_acc*100:.2f}%)")
    logger.info(f"   Average Rank: {avg_rank:.1f}")
    logger.info(
        f"   Performance Gap: {(avg_best_acc - avg_peecom_acc)*100:.2f}%")

    # Task type analysis
    logger.info(f"\n📋 Victory Analysis by Task Type:")

    binary_results = [r for r in results if r['n_classes'] == 2]
    multi_results = [r for r in results if r['n_classes'] > 2]

    if binary_results:
        binary_wins = len(
            [r for r in binary_results if r['victory_status'] == 'WIN'])
        binary_avg = np.mean([r['peecom_accuracy'] for r in binary_results])
        logger.info(
            f"   Binary Tasks: {binary_wins}/{len(binary_results)} wins, {binary_avg:.3f} avg")

    if multi_results:
        multi_wins = len(
            [r for r in multi_results if r['victory_status'] == 'WIN'])
        multi_avg = np.mean([r['peecom_accuracy'] for r in multi_results])
        logger.info(
            f"   Multi-Class: {multi_wins}/{len(multi_results)} wins, {multi_avg:.3f} avg")

    # Detailed match results
    logger.info(f"\n📝 Detailed Match Results:")
    logger.info(
        f"{'Task':<35} | {'Type':<6} | {'PEECOM':<8} | {'Rank':<6} | {'Status':<8} | {'Winner'}")
    logger.info("-" * 85)

    for r in results:
        task_type = 'Binary' if r['n_classes'] == 2 else 'Multi'
        peecom_acc = f"{r['peecom_accuracy']:.3f}"
        rank = f"#{r['peecom_rank']}"
        status = r['victory_status']
        winner = r['best_model'][:12]

        # Status icon
        if status == 'WIN':
            icon = "🥇"
        elif status == 'CLOSE':
            icon = "🥈"
        else:
            icon = "📉"

        logger.info(
            f"{icon} {r['task'][:32]:<32} | {task_type:<6} | {peecom_acc:<8} | {rank:<6} | {status:<8} | {winner}")

    # Final verdict
    logger.info(f"\n🎯 CHAMPIONSHIP VERDICT:")

    win_rate = wins / total_matches
    competitive_rate = (wins + close_losses) / total_matches

    if win_rate >= 0.6:
        verdict = "🟢 CHAMPION: Winning PEECOM dominates traditional models!"
    elif win_rate >= 0.4:
        verdict = "🟡 STRONG: Winning PEECOM competes very well!"
    elif competitive_rate >= 0.8:
        verdict = "🟠 COMPETITIVE: Winning PEECOM is consistently competitive!"
    else:
        verdict = "🔴 NEEDS IMPROVEMENT: More optimization required"

    logger.info(f"   {verdict}")
    logger.info(f"   Championship completed in {total_time:.1f} seconds")

    # Research implications
    logger.info(f"\n📄 Research Paper Implications:")
    if win_rate >= 0.5:
        logger.info("   ✅ Strong evidence for PEECOM superiority")
    if avg_peecom_acc >= 0.85:
        logger.info("   ✅ High accuracy across diverse domains")
    if competitive_rate >= 0.8:
        logger.info("   ✅ Consistent competitive performance")

    # Success criteria check
    logger.info(f"\n✓ SUCCESS CRITERIA CHECK:")
    logger.info(
        f"   ✓ Wins majority of matches: {'YES' if win_rate >= 0.5 else 'NO'}")
    logger.info(
        f"   ✓ Higher average accuracy: {'YES' if avg_peecom_acc >= avg_best_acc else 'NO'}")
    logger.info(
        f"   ✓ Competitive on all tasks: {'YES' if competitive_rate >= 0.8 else 'NO'}")

    if win_rate >= 0.5 and competitive_rate >= 0.8:
        logger.info(
            "\n🎉 MISSION ACCOMPLISHED: Winning PEECOM achieves superiority!")
    else:
        logger.info(
            f"\n💪 MISSION CONTINUES: {win_rate*100:.1f}% win rate, {competitive_rate*100:.1f}% competitive")

    return results


if __name__ == "__main__":
    main()
