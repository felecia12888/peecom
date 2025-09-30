#!/usr/bin/env python3
"""
Create Synthetic Leaky Data - Mimicking the hydraulic system data structure
This creates data where features encode block identity (like the real problematic data)
"""

import numpy as np
import pandas as pd

print("🔧 Creating synthetic leaky hydraulic data...")

# Set random seed for reproducibility
np.random.seed(42)

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

# Create block-specific statistical signatures
# This simulates the problematic data where each block has different statistical properties
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
    # This creates the "leakage" - features inadvertently encode which block they came from
    X[start_idx:end_idx, :] = (np.random.randn(samples_per_block[block_idx], n_features) * 
                              block_stds[block_idx] + 
                              block_means[block_idx])
    
    print(f"  Block {block_idx}: samples {start_idx:4d}-{end_idx-1:4d}, "
          f"mean features = {X[start_idx:end_idx, :].mean(axis=0)[:3].round(2)}")

# Create target classes (perfectly aligned with blocks - the source of leakage)
y = np.zeros(n_samples, dtype=int)
y[:733] = 0      # Block 0: Class 0
y[733:1464] = 1  # Block 1: Class 1  
y[1464:] = 2     # Block 2: Class 2

# Create block IDs
block_ids = np.zeros(n_samples, dtype=int)
block_ids[:733] = 0      # Block 0
block_ids[733:1464] = 1  # Block 1
block_ids[1464:] = 2     # Block 2

# Convert to DataFrame
feature_names = [f'f{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Add some noise to make it more realistic (but keep the block signatures)
noise_level = 0.1
df[feature_names] += np.random.randn(n_samples, n_features) * noise_level

# Save the synthetic leaky data
df.to_csv('hydraulic_data_processed.csv', index=False)

print(f"\n✅ Synthetic leaky data created: hydraulic_data_processed.csv")
print(f"   Shape: {df.shape}")
print(f"   Features: {feature_names[:5]}... (showing first 5)")
print(f"   Target distribution: {np.bincount(y)}")

# Verify block-class alignment (should be perfect - this IS the leakage)
print(f"\n🔍 Block-Class Alignment Analysis:")
for block_idx in range(n_blocks):
    block_mask = block_ids == block_idx
    classes_in_block = y[block_mask]
    unique_classes, counts = np.unique(classes_in_block, return_counts=True)
    print(f"   Block {block_idx}: Classes {unique_classes} with counts {counts}")

print(f"\n🚨 CRITICAL: This data has PERFECT block-class segregation")
print(f"   - Each block contains exactly one class")
print(f"   - Features have block-specific statistical signatures")
print(f"   - This is the DATA LEAKAGE we're trying to detect!")

print(f"\n🎯 Ready to test block predictor on leaky data...")