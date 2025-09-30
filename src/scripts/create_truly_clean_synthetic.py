#!/usr/bin/env python3
"""
Create TRULY CLEAN Synthetic Data - Homogeneous feature distributions
=====================================================================

Purpose: Generate synthetic data with NO block-distinguishable features:
- Identical feature means across all blocks
- Identical feature variances across all blocks  
- Identical feature correlations across all blocks
- Only target classes vary (randomly assigned)

This should pass all validation criteria:
- Block predictor ≈ 33% (chance level)
- Target models p >= 0.05 (no leakage)
- Effect sizes near zero
"""

import numpy as np
import pandas as pd
import argparse

def create_truly_clean_data(random_seed=42):
    """Create synthetic data with identical statistical properties per block"""
    
    print(f"🔧 Creating TRULY CLEAN synthetic hydraulic data...")
    print(f"Random seed: {random_seed}")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Data dimensions (same as real hydraulic data)
    n_samples = 2205
    n_features = 54
    n_blocks = 3
    
    # Samples per block (same as experiments)
    samples_per_block = [733, 731, 741]
    block_starts = [0, 733, 1464]
    
    print(f"Creating {n_samples} samples with {n_features} features across {n_blocks} blocks")
    print(f"Block sizes: {samples_per_block}")
    
    # CRITICAL: Use IDENTICAL statistical properties for all blocks
    # This removes any block-distinguishable signal from features
    global_mean = np.random.randn(n_features) * 0.5  # Same mean for all blocks
    global_std = np.ones(n_features) * 1.2            # Same std for all blocks
    
    print("🧹 HOMOGENEOUS MODE: Identical feature distributions across all blocks")
    print(f"   Global feature means: {global_mean[:5].round(3)}... (same for all blocks)")
    print(f"   Global feature stds: {global_std[:5].round(3)}... (same for all blocks)")
    
    # Initialize feature matrix
    X = np.zeros((n_samples, n_features))
    
    # Generate features with IDENTICAL distributions for each block
    for block_idx in range(n_blocks):
        start_idx = block_starts[block_idx]
        end_idx = start_idx + samples_per_block[block_idx]
        
        # All blocks use the same global mean and std
        for feature_idx in range(n_features):
            X[start_idx:end_idx, feature_idx] = np.random.normal(
                loc=global_mean[feature_idx],     # Same mean across blocks
                scale=global_std[feature_idx],    # Same std across blocks
                size=samples_per_block[block_idx]
            )
    
    # Add identical feature correlations across all blocks
    correlation_strength = 0.3
    for block_idx in range(n_blocks):
        start_idx = block_starts[block_idx]
        end_idx = start_idx + samples_per_block[block_idx]
        
        # Same correlation pattern for all blocks
        X[start_idx:end_idx, 0] += correlation_strength * X[start_idx:end_idx, 1]
        X[start_idx:end_idx, 5] += correlation_strength * X[start_idx:end_idx, 10]
        X[start_idx:end_idx, 20] += correlation_strength * X[start_idx:end_idx, 25]
    
    # Random class assignment (independent of blocks)
    print("🎲 Random class assignment (no block-class correlation)")
    target = np.random.choice([0, 1, 2], size=n_samples, p=[0.33, 0.33, 0.34])
    np.random.shuffle(target)  # Extra shuffle for good measure
    
    # Create DataFrame
    feature_names = [f'f{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = target
    df['original_index'] = range(n_samples)
    
    # Validation: Check block-feature discriminability
    print(f"\n🔬 VALIDATION: Testing block discriminability...")
    
    # Generate block labels
    block_labels = np.zeros(n_samples, dtype=int)
    for block_idx in range(n_blocks):
        start_idx = block_starts[block_idx]
        end_idx = start_idx + samples_per_block[block_idx]
        block_labels[start_idx:end_idx] = block_idx
    
    # Test if features can predict blocks (should be ~chance)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    block_scores = cross_val_score(clf, X, block_labels, cv=3, scoring='accuracy')
    block_acc = block_scores.mean()
    
    print(f"Cross-validation block prediction: {block_acc:.4f} ± {block_scores.std():.4f}")
    print(f"Expected for clean data: ~0.33 (chance level)")
    
    if block_acc <= 0.40:
        print("✅ VALIDATION PASSED: Features cannot predict blocks")
    else:
        print("❌ VALIDATION FAILED: Features still encode block information")
    
    # Class distribution analysis  
    class_counts = pd.Series(target).value_counts().sort_index()
    print(f"\n📊 Class distribution:")
    for class_val, count in class_counts.items():
        print(f"   Class {class_val}: {count} samples ({count/n_samples:.1%})")
    
    # Block-class correlation analysis
    print(f"\n🔗 Block-Class correlation:")
    block_class_crosstab = pd.crosstab(block_labels, target, margins=True)
    print(block_class_crosstab)
    
    block_class_acc = np.mean(block_labels == target)
    print(f"\nBlock==Class accuracy: {block_class_acc:.4f} (expected: ~0.33)")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create truly clean synthetic hydraulic data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='synthetic_truly_clean.csv', help='Output filename')
    
    args = parser.parse_args()
    
    # Generate truly clean synthetic data
    df = create_truly_clean_data(random_seed=args.seed)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\n💾 Saved truly clean synthetic data: {args.output}")
    print(f"Shape: {df.shape}")
    
    print(f"\n✅ TRULY CLEAN synthetic data generation complete!")
    print(f"Ready for pipeline validation testing.")