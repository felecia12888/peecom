# save as inspect_block_fingerprints.py ; run: python inspect_block_fingerprints.py
import pandas as pd
import numpy as np
import itertools
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("hydraulic_data_processed.csv")  # adjust path
# assume 'target' column and infer blocks from class transitions
if 'block' not in df.columns:
    # infer blocks by contiguous class segments (adapt if different)
    df['block'] = (df['target'].shift(1) != df['target']).cumsum()

feature_cols = [c for c in df.columns if c.startswith('f') and c[1:].isdigit()]
blocks = sorted(df['block'].unique())

print(f"Analyzing {len(feature_cols)} features across {len(blocks)} blocks")
print(f"Block sizes: {[sum(df['block']==b) for b in blocks]}")

# compute per-feature stats
rows = []
for f in feature_cols:
    vals = []
    for b in blocks:
        arr = df.loc[df['block']==b, f].values
        vals.append({'block':b,
                     'mean':np.mean(arr),
                     'std':np.std(arr, ddof=1),
                     'median':np.median(arr)})
    # compute Cohen's d between every pair of blocks
    ds = []
    for (b1,b2) in itertools.combinations(range(len(blocks)),2):
        a = df.loc[df['block']==blocks[b1], f].values
        b = df.loc[df['block']==blocks[b2], f].values
        pooled = np.sqrt(((a.std(ddof=1)**2)+(b.std(ddof=1)**2))/2)
        d = (a.mean()-b.mean())/pooled if pooled>0 else np.nan
        ds.append(d)
    rows.append({
        'feature':f,
        'mean_block0':vals[0]['mean'] if len(vals)>0 else np.nan,
        'mean_block1':vals[1]['mean'] if len(vals)>1 else np.nan,
        'mean_block2':vals[2]['mean'] if len(vals)>2 else np.nan,
        'std_block0':vals[0]['std'] if len(vals)>0 else np.nan,
        'std_block1':vals[1]['std'] if len(vals)>1 else np.nan,
        'std_block2':vals[2]['std'] if len(vals)>2 else np.nan,
        'max_abs_cohens_d': np.nanmax(np.abs(ds))
    })

out = pd.DataFrame(rows).sort_values('max_abs_cohens_d', ascending=False)
out.to_csv("feature_block_stats.csv", index=False)
print("Saved feature_block_stats.csv (sorted by max_abs_cohens_d)")

# Print top 10 most block-separable features
print(f"\n🔍 TOP 10 MOST BLOCK-SEPARABLE FEATURES:")
print("=" * 60)
for i, row in out.head(10).iterrows():
    print(f"{row['feature']:5s}: Cohen's d = {row['max_abs_cohens_d']:6.2f} | "
          f"Means: B0={row['mean_block0']:7.2f}, B1={row['mean_block1']:7.2f}, B2={row['mean_block2']:7.2f}")

# quick covariance difference matrix
print(f"\n🔍 COMPUTING COVARIANCE DIFFERENCES...")
covs = {b: df.loc[df['block']==b, feature_cols].cov() for b in blocks}
diff01 = (covs[blocks[0]] - covs[blocks[1]]).abs().mean().sort_values(ascending=False)
diff12 = (covs[blocks[1]] - covs[blocks[2]]).abs().mean().sort_values(ascending=False)
diff02 = (covs[blocks[0]] - covs[blocks[2]]).abs().mean().sort_values(ascending=False)

cov_summary = pd.DataFrame({
    'diff_block0_block1': diff01, 
    'diff_block1_block2': diff12,
    'diff_block0_block2': diff02
})
cov_summary.to_csv("cov_diff_summary.csv")
print("Saved cov_diff_summary.csv (columns show which features differ most in covariance)")

print(f"\n🔍 TOP 10 FEATURES WITH LARGEST COVARIANCE DIFFERENCES:")
print("=" * 60)
for feat in cov_summary.index[:10]:
    row = cov_summary.loc[feat]
    max_diff = max(row['diff_block0_block1'], row['diff_block1_block2'], row['diff_block0_block2'])
    print(f"{feat:5s}: Max cov diff = {max_diff:6.3f} | "
          f"B0-B1={row['diff_block0_block1']:.3f}, B1-B2={row['diff_block1_block2']:.3f}, B0-B2={row['diff_block0_block2']:.3f}")

# heatmap for top 12 differing features
top_features = out['feature'].head(12).tolist()
plt.figure(figsize=(12,8))
block_means = df.groupby('block')[top_features].mean().T
sns.heatmap(block_means, annot=True, fmt=".2f", cmap='RdYlBu_r', center=0, cbar_kws={'label': 'Mean Value'})
plt.title("Per-block means (top 12 most separable features)", fontsize=14, fontweight='bold')
plt.xlabel("Block ID", fontweight='bold')
plt.ylabel("Feature", fontweight='bold')
plt.tight_layout()
plt.savefig("per_block_means_top12.png", dpi=150, bbox_inches='tight')
print("Saved per_block_means_top12.png")

# Additional diagnostic: feature correlation with block ID
print(f"\n🔍 FEATURE-BLOCK CORRELATIONS:")
print("=" * 40)
block_correlations = []
for f in feature_cols:
    corr = np.corrcoef(df[f], df['block'])[0,1]
    block_correlations.append({'feature': f, 'block_correlation': abs(corr)})

block_corr_df = pd.DataFrame(block_correlations).sort_values('block_correlation', ascending=False)
print("Top 10 features most correlated with block ID:")
for i, row in block_corr_df.head(10).iterrows():
    print(f"{row['feature']:5s}: |r| = {row['block_correlation']:6.3f}")

# Summary statistics
mean_cohens_d = out['max_abs_cohens_d'].mean()
max_cohens_d = out['max_abs_cohens_d'].max()
features_large_effect = sum(out['max_abs_cohens_d'] > 1.0)

print(f"\n📊 SUMMARY STATISTICS:")
print("=" * 30)
print(f"Mean Cohen's d across features: {mean_cohens_d:.2f}")
print(f"Maximum Cohen's d: {max_cohens_d:.2f}")
print(f"Features with large effect (|d| > 1.0): {features_large_effect}/{len(feature_cols)} ({100*features_large_effect/len(feature_cols):.1f}%)")

# Diagnosis
print(f"\n🎯 DIAGNOSTIC ASSESSMENT:")
print("=" * 30)
if max_cohens_d > 5.0:
    print("🚨 EXTREME mean differences detected - likely single feature(s) dominating")
    print("   → Focus on removing/transforming top features")
elif mean_cohens_d > 2.0:
    print("⚠️  LARGE systematic mean shifts across many features")
    print("   → May need global block-normalization approach")
elif max(diff01.max(), diff12.max(), diff02.max()) > 1.0:
    print("🔄 COVARIANCE structure differences dominate")
    print("   → Consider block-covariance alignment approach")
else:
    print("🤔 SUBTLE distributed differences - most challenging case")
    print("   → May need advanced orthogonalization or domain-specific features")

print(f"\n✅ Block fingerprint analysis complete!")
print(f"📁 Files saved: feature_block_stats.csv, cov_diff_summary.csv, per_block_means_top12.png")