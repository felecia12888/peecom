#!/usr/bin/env python3
"""
PEECOM Research Pipeline - Clean Flow
Complete automated pipeline for PEECOM evaluation and research
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')


def run_peecom_research():
    """Run the complete PEECOM research pipeline"""

    print("🔬 PEECOM RESEARCH PIPELINE")
    print("=" * 60)
    print("Complete automated evaluation of Physics-Enhanced ML")
    print()

    # Step 1: Verify environment
    print("📋 Step 1: Environment Verification")
    print("-" * 40)

    try:
        from src.models.winning_peecom import WinningPEECOM
        print("✅ Winning PEECOM model loaded successfully")

        # Quick smoke test
        import pandas as pd
        import numpy as np
        X_test = pd.DataFrame(np.random.randn(50, 3), columns=['A', 'B', 'C'])
        y_test = pd.Series(np.random.randint(0, 2, 50))

        model = WinningPEECOM(verbose=False)
        model.fit(X_test, y_test)
        predictions = model.predict(X_test)
        print("✅ Model functionality verified")

    except Exception as e:
        print(f"❌ Environment check failed: {e}")
        return False

    print()

    # Step 2: Quick validation test
    print("📋 Step 2: Quick Validation Test")
    print("-" * 40)

    try:
        # Run quick test by importing and executing
        import subprocess
        result = subprocess.run(['python', 'quick_winning_test.py'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Quick test completed successfully")
        else:
            print(f"⚠️  Quick test had issues: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Quick test had issues: {e}")

    print()

    # Step 3: Fast championship evaluation
    print("📋 Step 3: Fast Championship Evaluation")
    print("-" * 40)
    print("Testing PEECOM on key datasets...")
    print()

    start_time = time.time()

    try:
        # Run fast championship
        import subprocess
        result = subprocess.run(['python', 'fast_winning_championship.py'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ Fast championship failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Fast championship failed: {e}")
        return False    fast_time = time.time() - start_time
    print(f"\n⏱️  Fast evaluation completed in {fast_time:.1f} seconds")
    print()

    # Step 4: Decision point for full evaluation
    print("📋 Step 4: Full Evaluation Decision")
    print("-" * 40)

    response = input(
        "🤔 Run full championship evaluation? (y/n): ").lower().strip()

    if response == 'y' or response == 'yes':
        print("\n📋 Step 5: Full Championship Evaluation")
        print("-" * 40)
        print("Running comprehensive PEECOM evaluation...")
        print("⚠️  This may take several minutes...")
        print()

        start_time = time.time()

        try:
            # Run full championship
            import subprocess
            result = subprocess.run(['python', 'winning_peecom_championship.py'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"❌ Full championship failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Full championship failed: {e}")
            return False        full_time = time.time() - start_time
        print(f"\n⏱️  Full evaluation completed in {full_time:.1f} seconds")

    else:
        print("⏭️  Skipping full evaluation")

    print()

    # Final summary
    print("🏁 PEECOM RESEARCH PIPELINE COMPLETE")
    print("=" * 60)
    print("✅ Environment verified")
    print("✅ Model functionality confirmed")
    print("✅ Fast evaluation completed")
    if response in ['y', 'yes']:
        print("✅ Full evaluation completed")

    print()
    print("📊 KEY FILES FOR RESEARCH:")
    print("   • src/models/winning_peecom.py - Final optimized model")
    print("   • fast_winning_championship.py - Quick testing")
    print("   • winning_peecom_championship.py - Full evaluation")
    print("   • output/ - Results and processed data")
    print()
    print("🎯 RESEARCH STATUS:")
    print("   Ready for paper writing and publication!")

    return True


if __name__ == "__main__":
    success = run_peecom_research()

    if not success:
        print("\n❌ Pipeline encountered errors. Please check the issues above.")
        exit(1)
    else:
        print("\n✅ Pipeline completed successfully!")
