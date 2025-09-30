#!/usr/bin/env python3
"""
Create PERFECTLY homogeneous synthetic data for pipeline validation
==================================================================

This generates data where:
1. ALL features have IDENTICAL statistical properties across blocks
2. Classes are randomly assigned (no block-class correlation)
3. Block membership is indistinguishable from feature values alone

This is the true "clean" control for validating our leakage detection pipeline.
"""

import numpy as np
import pandas as pd

def create_perfectly_clean_synthetic():
    """
    Generate synthetic data with NO block-distinguishable signal in features
    """
    
    # Data dimensions (same as real hydraulic data)
    n_samples = 2205
    n_features = 54
    n_blocks = 3
    
    # Samples per block (same as experiments) 
    samples_per_block = [735, 735, 735]  # Equal blocks for simplicity
    
    print(f"Creating {n_samples} samples with {n_features} features across {n_blocks} blocks")
    print(f"Block sizes: {samples_per_block}")
    
    # CRITICAL: Generate ALL features from a SINGLE global distribution
    # This ensures no block can be distinguished from features alone
    print("🧹 PERFECTLY HOMOGENEOUS: Single global feature distribution")
    
    # Generate ALL samples from the same distribution (ignore block structure)
    np.random.seed(42)  # Fixed seed for reproducibility
    
    # Base features - all samples from same distribution
    X = np.random.randn(n_samples, n_features) * 1.2
    
    # Add some mild correlations (same pattern for all samples)
    X[:, 0] += 0.3 * X[:, 1]          # f0 correlated with f1
    X[:, 5] += 0.2 * X[:, 10]         # f5 correlated with f10  
    X[:, 20] += 0.4 * X[:, 25]        # f20 correlated with f25
    
    # Add small amount of structured noise (same for all)
    noise_scale = 0.1
    X += np.random.randn(n_samples, n_features) * noise_scale
    
    print(f"   Features generated from single N(0, 1.2) distribution")
    print(f"   Added mild correlations and {noise_scale} noise")
    
    # Assign block IDs (purely for bookkeeping - not related to features)
    blocks = np.zeros(n_samples, dtype=int)
    blocks[:samples_per_block[0]] = 0
    blocks[samples_per_block[0]:samples_per_block[0]+samples_per_block[1]] = 1  
    blocks[samples_per_block[0]+samples_per_block[1]:] = 2
    
    # RANDOM class assignment (independent of everything)
    np.random.seed(123)  # Different seed for class assignment
    classes = np.random.randint(0, 3, size=n_samples)
    
    print("🎲 Random class assignment (completely independent of blocks and features)")
    
    # Verify no block structure in features
    print("\n🔍 VERIFICATION - Feature statistics by block:")
    for block_idx in range(n_blocks):
        mask = (blocks == block_idx)
        block_mean = X[mask].mean(axis=0)[:5]  # First 5 features
        block_std = X[mask].std(axis=0)[:5]    # First 5 features
        print(f"   Block {block_idx}: mean={block_mean.round(3)}, std={block_std.round(3)}")
    
    print("\n🔍 VERIFICATION - Class distribution by block:")
    for block_idx in range(n_blocks):
        mask = (blocks == block_idx)
        block_classes = classes[mask]
        class_counts = [np.sum(block_classes == c) for c in range(3)]
        class_pcts = [count/len(block_classes)*100 for count in class_counts]
        print(f"   Block {block_idx}: {class_counts} = {[f'{p:.1f}%' for p in class_pcts]}")
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(n_features)])
    df['target'] = classes
    df['original_index'] = np.arange(n_samples)
    
    # Save to file
    output_file = 'synthetic_perfectly_clean.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved perfectly clean synthetic data to: {output_file}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING PERFECTLY CLEAN SYNTHETIC DATA FOR PIPELINE VALIDATION")
    print("=" * 70)
    
    df = create_perfectly_clean_synthetic()
    
    print("\n✅ SUCCESS: Perfectly homogeneous synthetic data created!")
    print("   - No block-distinguishable signal in features")
    print("   - Random class assignment") 
    print("   - Ready for pipeline validation testing")