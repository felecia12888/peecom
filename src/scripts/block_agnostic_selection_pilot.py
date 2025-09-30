# block_agnostic_selection_pilot.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import time

# === Load data ===
print("Loading data...")
df = pd.read_csv('hydraulic_data_processed.csv')
print(f"Loaded data: {df.shape}")

# Extract features and target
feature_cols = [col for col in df.columns if col.startswith('f') and col[1:].isdigit()]
X = df[feature_cols].values
y = df['target'].values

# Generate block IDs from class transitions (like other experiments)
print("Creating block IDs from class transitions...")
blocks = np.zeros(len(y), dtype=int)
current_block = 0
for i in range(1, len(y)):
    if y[i] != y[i-1]:  # class transition
        current_block += 1
    blocks[i] = current_block

print(f"Features: {len(feature_cols)}, Samples: {len(X)}, Classes: {len(np.unique(y))}, Blocks: {len(np.unique(blocks))}")

def drop_block_predictive_features(X_train, blocks_train, k_block=10):
    """Return column mask keeping features that are least predictive of block.
       We use ANOVA F between blocks to score features; drop highest k_block."""
    # X_train: ndarray (n_samples, n_features)
    # blocks_train: array-like (n_samples,)
    F, p = f_classif(X_train, blocks_train)   # ANOVA across the block labels
    # handle nan/infs
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    # drop top-k highest F (most block-predictive)
    idx_sorted = np.argsort(F)  # ascending
    keep_idx = idx_sorted[:-k_block] if k_block>0 else idx_sorted
    # if k_block >= n_features, keep none -> guard:
    if len(keep_idx) == 0:
        keep_idx = idx_sorted[:max(1, X_train.shape[1]-k_block)]
    return np.sort(keep_idx)

def train_eval_fold(X_train, y_train, blocks_train, X_test, y_test, blocks_test, k_block=10, clf_class=None):
    # compute features to keep using ONLY training data
    keep_idx = drop_block_predictive_features(X_train, blocks_train, k_block=k_block)
    Xtr = X_train[:, keep_idx]
    Xte = X_test[:, keep_idx]
    clf = clf_class() if clf_class else RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=0)
    clf.fit(Xtr, y_train)
    ypred = clf.predict(Xte)
    return accuracy_score(y_test, ypred)

def cross_val_with_block_filter(X, y, blocks, n_splits=3, k_block=10, clf_class=None, random_state=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        btr, bte = blocks[train_idx], blocks[test_idx]
        acc = train_eval_fold(Xtr, ytr, btr, Xte, yte, bte, k_block=k_block, clf_class=clf_class)
        accs.append(acc)
    return np.array(accs)

# --- Pilot parameters ---
n_folds = 3
n_perms = 100          # pilot (fast); later escalate to 1000 if needed
k_block = 10           # drop the top 10 block-predictive features (tweakable)
clf_class = lambda: RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=0)

# Convert to numpy
X = np.asarray(X)
y = np.asarray(y)
blocks = np.asarray(blocks)

print(f"\nRunning block-agnostic validation...")
print(f"Strategy: Drop top {k_block} most block-predictive features from each training fold")

# Baseline (proper per-fold block-filtering)
t0 = time.time()
accs = cross_val_with_block_filter(X, y, blocks, n_splits=n_folds, k_block=k_block, clf_class=clf_class)
baseline = accs.mean()
print("Baseline CV acc (block-filtered):", baseline, "+/-", accs.std(), "time:", time.time()-t0)

# Permutation pilot (parallelized)
print(f"Running {n_perms} permutation tests...")
rng = np.random.RandomState(0)
def one_perm(i):
    y_perm = rng.permutation(y)
    accs_p = cross_val_with_block_filter(X, y_perm, blocks, n_splits=n_folds, k_block=k_block, clf_class=clf_class, random_state=42+i)
    return accs_p.mean()

t0 = time.time()
perms = Parallel(n_jobs=6)(delayed(one_perm)(i) for i in range(n_perms))  # adjust n_jobs
perms = np.array(perms)
pval = (np.sum(perms >= baseline) + 1) / (n_perms + 1)
print("Permutations done. perm-mean:", perms.mean(), "pval:", pval, "time:", time.time()-t0)

# Effect size calculation
effect_size = (baseline - perms.mean()) / perms.std() if perms.std() > 0 else 0
print(f"\nSUMMARY:")
print(f"Baseline accuracy: {baseline:.4f}")
print(f"Null mean accuracy: {perms.mean():.4f}")
print(f"P-value: {pval:.4f}")
print(f"Effect size (Cohen's d): {effect_size:.2f}")

if pval >= 0.05:
    print("✅ REMEDIATION SUCCESS: No significant leakage detected (p >= 0.05)")
else:
    print("❌ LEAKAGE DETECTED: Significant signal remains (p < 0.05)")
    
# Let's also test with different numbers of dropped features
print(f"\n🔬 Testing different k_block values...")
k_values = [5, 10, 20, 30, 40]
for k in k_values:
    if k >= X.shape[1]:
        continue
    accs_k = cross_val_with_block_filter(X, y, blocks, n_splits=n_folds, k_block=k, clf_class=clf_class)
    print(f"k_block={k}: CV accuracy = {accs_k.mean():.4f} ± {accs_k.std():.4f}")