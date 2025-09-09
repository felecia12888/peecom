#!/usr/bin/env python3
"""
Quick test of the Winning PEECOM to see if it works
"""

import warnings
warnings.filterwarnings('ignore')

try:
    from winning_peecom_championship import benchmark_on_task

    print("🏆 Quick Winning PEECOM Test")
    print("=" * 50)

    # Test single dataset
    result = benchmark_on_task(
        'cmohs', 'stable_flag', 'Hydraulic System Stability Detection')

    if result['peecom_status'] == 'success':
        print(f"✅ PEECOM worked! Accuracy: {result['peecom_accuracy']:.4f}")
        print(f"🏆 Winner: {result['winner']} ({result['best_accuracy']:.4f})")
        if result['peecom_rank'] == 1:
            print("🥇 PEECOM WON!")
        else:
            print(f"🥈 PEECOM Rank #{result['peecom_rank']}")
    else:
        print(f"❌ PEECOM failed: {result['peecom_error']}")

except Exception as e:
    print(f"❌ Test failed: {str(e)}")
    import traceback
    traceback.print_exc()
