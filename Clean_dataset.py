import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# === DO NOT EDIT: Data loading, splitting, and scaling functions ===
def load_all_sensor_data(dataset_dir: str, delimiter='\t'):  # DO NOT EDIT
    files = glob.glob(os.path.join(dataset_dir, "*.txt"))
    sensor_files = [
        f for f in files
        if os.path.basename(f).lower() not in ['profile.txt', 'description.txt', 'documentation.txt']
        and os.path.getsize(f) > 0
    ]
    if not sensor_files:
        raise ValueError(f"No non-empty sensor data files found in directory: {dataset_dir}")

    dfs = []
    common_length = 1000
    for f in sensor_files:
        df = pd.read_csv(f, delimiter=delimiter, header=None, dtype=np.float32)
        if not df.empty:
            resampled = resample(df.values, common_length, axis=0)
            dfs.append(pd.DataFrame(resampled))
    if not dfs:
        raise ValueError("Sensor files found, but no data could be read from any file.")
    try:
        sensor_data = pd.concat(dfs, axis=1)
    except ValueError as e:
        raise ValueError("No objects to concatenate. Check sensor files for valid data.") from e
    return sensor_data.values  # DO NOT EDIT

def load_profile(dataset_dir: str, delimiter='\t'):  # DO NOT EDIT
    profile_path = os.path.join(dataset_dir, "profile.txt")
    if not os.path.isfile(profile_path):
        raise FileNotFoundError(f"Expected profile.txt in {dataset_dir}")
    y = pd.read_csv(profile_path, delimiter=delimiter, header=None,
                    dtype={0: np.int32, 1: np.float32, 2: np.float32, 3: np.float32, 4: np.int32})
    return y.values  # DO NOT EDIT

def preprocess_data(X: np.ndarray):  # DO NOT EDIT
    scaler = RobustScaler()
    return scaler.fit_transform(X)

def create_train_test_split(X: np.ndarray, y: np.ndarray,
                            test_size=0.2, random_state=42):  # DO NOT EDIT
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
# === END DO NOT EDIT ===

if __name__ == '__main__':
    # Example usage with updated dataset path
    dataset_dir = r"C:\Users\28151\Desktop\Updated code files\dataset"
    X = load_all_sensor_data(dataset_dir)
    y = load_profile(dataset_dir)
    X_scaled = preprocess_data(X)
    X_train, X_test, y_train, y_test = create_train_test_split(X_scaled, y)
    print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")
