#!/usr/bin/env python3
"""
Fast Winning PEECOM Championship - Testing on key datasets
"""

import warnings
warnings.filterwarnings('ignore')

try:
    from winning_peecom_championship import benchmark_on_task

    print("🏆 FAST WINNING PEECOM CHAMPIONSHIP")
    print("=" * 60)

    # Key test datasets (focusing on the most important ones)
    test_tasks = [
        ('cmohs', 'stable_flag', 'Hydraulic System Stability'),
        ('motorvd', 'condition', 'Motor Vibration Condition'),
        ('equipmentad', 'anomaly', 'Equipment Anomaly Detection'),
        ('cmohs', 'cooler_condition', 'Hydraulic Cooler Condition'),
    ]

    wins = 0
    ties = 0
    total = 0
    peecom_scores = []

    for dataset, target, description in test_tasks:
        print(f"\n🏆 Testing: {description}")
        print("-" * 50)

        try:
            result = benchmark_on_task(dataset, target, description)
            total += 1

            if result['peecom_status'] == 'success':
                peecom_acc = result['peecom_accuracy']
                best_acc = result['best_accuracy']
                rank = result['peecom_rank']
                winner = result['winner']

                peecom_scores.append(peecom_acc)

                print(f"🤖 PEECOM:    {peecom_acc:.4f} ({peecom_acc*100:.1f}%)")
                print(
                    f"🏆 Winner:    {winner} {best_acc:.4f} ({best_acc*100:.1f}%)")
                print(f"📊 Rank:      #{rank}")

                if rank == 1:
                    wins += 1
                    print("🥇 PEECOM WINS!")
                elif peecom_acc >= best_acc - 0.001:  # Close tie
                    ties += 1
                    print("🤝 VERY CLOSE!")
                else:
                    gap = best_acc - peecom_acc
                    print(f"💔 Gap: {gap:.4f}")

            else:
                print(f"❌ PEECOM failed: {result['peecom_error']}")

        except Exception as e:
            print(f"❌ Test failed: {str(e)}")

    # Summary
    print(f"\n🏁 FAST CHAMPIONSHIP RESULTS")
    print("=" * 60)
    print(f"🥇 Outright Wins: {wins}/{total} ({wins/total*100:.1f}%)")
    print(f"🤝 Close/Ties: {ties}/{total} ({ties/total*100:.1f}%)")
    print(
        f"🎯 Competitive: {(wins+ties)}/{total} ({(wins+ties)/total*100:.1f}%)")

    if peecom_scores:
        avg_score = sum(peecom_scores) / len(peecom_scores)
        print(f"📊 PEECOM Average: {avg_score:.4f} ({avg_score*100:.1f}%)")

        if wins > 0:
            print("✅ PEECOM IS WINNING SOME MATCHES!")
        elif wins + ties >= total * 0.8:
            print("✅ PEECOM IS HIGHLY COMPETITIVE!")
        else:
            print("⚠️  PEECOM NEEDS MORE OPTIMIZATION")

except Exception as e:
    print(f"❌ Championship failed: {str(e)}")
    import traceback
    traceback.print_exc()
