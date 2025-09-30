# save as inspect_synthetic.py and run: python inspect_synthetic.py
import pandas as pd
from collections import Counter
import numpy as np

fn = "synthetic_truly_clean_validation.csv"    # adjust path if needed
df = pd.read_csv(fn)
print("Loaded:", df.shape)
print("Columns:", df.columns.tolist())

# Use the 'target' column specifically
if 'target' in df.columns:
    y = df['target']
    print("Using 'target' column for class analysis")
elif 'class' in df.columns:
    y = df['class']
    print("Using 'class' column for class analysis")
else:
    y = df.iloc[:, -1]   # fallback
    print("Using last column for class analysis")

print("\nClass counts:")
print(y.value_counts().sort_index())

# if you have a 'block' column:
if 'block' in df.columns:
    print("\nBlock counts:")
    print(df['block'].value_counts().sort_index())
else:
    print("\nNo 'block' column found. Attempting to detect block boundaries by runs of identical class...")
    runs = []
    prev = None
    run_len = 0
    for v in y:
        if v==prev:
            run_len += 1
        else:
            if prev is not None:
                runs.append((prev, run_len))
            prev = v
            run_len = 1
    runs.append((prev, run_len))
    print("Detected runs (class, run_length) — show first 20:")
    print(runs[:20])
    lens = [r[1] for r in runs]
    print("run count:", len(runs), "min run len:", min(lens), "mean run len:", np.mean(lens))
# report min class count vs n_splits
n_splits = 3
min_count = y.value_counts().min()
print(f"\nMinimum class count = {min_count}; current n_splits = {n_splits}")
if min_count < n_splits:
    print(">>> WARNING: min class count < n_splits; this WILL make StratifiedKFold unstable/invalid.")
else:
    print("Class counts are sufficient for n_splits.")