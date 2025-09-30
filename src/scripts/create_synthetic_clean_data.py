#!/usr/bin/env python3
"""
Create Synthetic CLEAN Data - Same structure as leaky data but no block→class signal
=====================================================================================

Purpose: Generate clean synthetic dataset with:
- Same dimensions and block structure as real hydraulic data  
- Identical statistical properties per block (means, covariances)
- NO correlation between block membership and target classes
- This validates our diagnostic pipeline can detect absence of leakage

Strategy:
- Use same block-specific feature distributions as leaky version
- Randomize class labels independently of block structure
- Should pass all remediation criteria (p >= 0.05, block predictor ≈ chance)
"""

import numpy as np
import pandas as pd
import argparse

def create_clean_synthetic_data(clean=True, random_seed=42):
    """Create synthetic data with optional clean flag"""
    
    print(f"🔧 Creating synthetic {'CLEAN' if clean else 'LEAKY'} hydraulic data...")
    print(f"Random seed: {random_seed}")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Data dimensions (same as real hydraulic data)
    n_samples = 2205
    n_features = 54
    n_blocks = 3
    
    # Samples per block (same as experiments)
    samples_per_block = [733, 731, 741]  # Slightly uneven like real data
    block_starts = [0, 733, 1464]
    
    print(f"Creating {n_samples} samples with {n_features} features across {n_blocks} blocks")
    print(f"Block sizes: {samples_per_block}")
    
    # Initialize feature matrix
    X = np.zeros((n_samples, n_features))
    
    # Create block-specific statistical signatures (same as leaky version)
    # This maintains realistic block-level heterogeneity
    block_means = [
        np.random.randn(n_features) * 0.5,      # Block 0: Centered around different means
        np.random.randn(n_features) * 0.5 + 2,  # Block 1: Shifted up
        np.random.randn(n_features) * 0.5 - 1   # Block 2: Shifted down
    ]
    
    block_stds = [
        np.ones(n_features) * 1.0,     # Block 0: Standard variance
        np.ones(n_features) * 1.5,     # Block 1: Higher variance
        np.ones(n_features) * 0.7      # Block 2: Lower variance
    ]
    
    # Generate features with block-specific signatures
    for block_idx in range(n_blocks):
        start_idx = block_starts[block_idx]
        end_idx = start_idx + samples_per_block[block_idx]
        
        # Features have block-specific means and standard deviations
        # This preserves the statistical heterogeneity that exists in real data
        for feature_idx in range(n_features):
            X[start_idx:end_idx, feature_idx] = np.random.normal(
                loc=block_means[block_idx][feature_idx],
                scale=block_stds[block_idx][feature_idx],
                size=samples_per_block[block_idx]
            )
    
    # Add some realistic correlations between features (within blocks)
    for block_idx in range(n_blocks):
        start_idx = block_starts[block_idx]
        end_idx = start_idx + samples_per_block[block_idx]
        
        # Create some feature interactions (realistic but not class-predictive)
        X[start_idx:end_idx, 0] += 0.3 * X[start_idx:end_idx, 1]  # f0 correlated with f1
        X[start_idx:end_idx, 5] += 0.2 * X[start_idx:end_idx, 10] # f5 correlated with f10
        X[start_idx:end_idx, 20] += 0.4 * X[start_idx:end_idx, 25] # f20 correlated with f25
    
    # Create target labels
    if clean:
        print("🧹 CLEAN MODE: Randomizing class labels (no block→class correlation)")
        # Random assignment of classes, independent of block structure
        # Maintain same class proportions as real data: roughly balanced 3-class
        target = np.random.choice([0, 1, 2], size=n_samples, p=[0.33, 0.33, 0.34])
        
        # Shuffle to ensure no systematic patterns
        np.random.shuffle(target)
        
    else:
        print("🚨 LEAKY MODE: Block-aligned class labels (perfect block→class correlation)")
        # Each block gets a different class (maximum leakage)
        target = np.zeros(n_samples, dtype=int)
        for block_idx in range(n_blocks):
            start_idx = block_starts[block_idx]
            end_idx = start_idx + samples_per_block[block_idx]
            target[start_idx:end_idx] = block_idx
    
    # Create DataFrame with feature names matching real data
    feature_names = [f'f{i}' for i in range(n_features)]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = target
    
    # Add metadata for analysis
    df['original_index'] = range(n_samples)
    
    # Class distribution analysis
    class_counts = pd.Series(target).value_counts().sort_index()
    print(f"\n📊 Class distribution:")
    for class_val, count in class_counts.items():
        print(f"   Class {class_val}: {count} samples ({count/n_samples:.1%})")
    
    # Block-class correlation analysis
    block_labels = np.zeros(n_samples, dtype=int)
    for block_idx in range(n_blocks):
        start_idx = block_starts[block_idx]
        end_idx = start_idx + samples_per_block[block_idx]
        block_labels[start_idx:end_idx] = block_idx
    
    print(f"\n🔗 Block-Class correlation:")
    block_class_crosstab = pd.crosstab(block_labels, target, margins=True)
    print(block_class_crosstab)
    
    # Expected perfect correlation score (for reference)
    perfect_correlation = np.mean(block_labels == target)
    print(f"\nBlock==Class accuracy: {perfect_correlation:.4f}")
    if clean:
        print(f"Expected for clean data: ~0.33 (chance level)")
    else:
        print(f"Expected for leaky data: 1.0 (perfect correlation)")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create synthetic hydraulic data')
    parser.add_argument('--clean', action='store_true', 
                        help='Create clean data (no block-class correlation)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, 
                        default='synthetic_hydraulic_data.csv',
                        help='Output filename')
    
    args = parser.parse_args()
    
    # Generate synthetic data
    df = create_clean_synthetic_data(clean=args.clean, random_seed=args.seed)
    
    # Save to CSV
    output_filename = args.output
    if args.clean and 'clean' not in output_filename:
        output_filename = output_filename.replace('.csv', '_clean.csv')
    elif not args.clean and 'leaky' not in output_filename:
        output_filename = output_filename.replace('.csv', '_leaky.csv')
        
    df.to_csv(output_filename, index=False)
    print(f"\n💾 Saved synthetic data: {output_filename}")
    print(f"Shape: {df.shape}")
    print(f"Mode: {'CLEAN' if args.clean else 'LEAKY'}")
    
    print(f"\n✅ Synthetic data generation complete!")
    print(f"Ready for pipeline validation testing.")