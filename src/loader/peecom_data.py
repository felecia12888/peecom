import numpy as np
import pandas as pd
import glob
import os
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from scipy import signal

def load_all_sensor_data(dataset_dir, delimiter='\t'):
    files = glob.glob(os.path.join(dataset_dir, "*.txt"))
    sensor_files = [
        f for f in files 
        if os.path.basename(f).lower() not in ['profile.txt', 'description.txt', 'documentation.txt']
        and os.path.getsize(f) > 0  # Filter out empty files
    ]
    if not sensor_files:
        raise ValueError(f"No non-empty sensor data files found in directory: {dataset_dir}")
        
    sensor_dfs = []
    common_length = 1000  # common timeline length
    for f in sensor_files:
        try:
            df = pd.read_csv(f, delimiter=delimiter, header=None, skip_blank_lines=True, dtype=np.float32)
            if not df.empty:
                # Resample each df to common_length rows.
                resampled = signal.resample(df.values, common_length, axis=0)
                sensor_dfs.append(pd.DataFrame(resampled))
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not sensor_dfs:
        raise ValueError("Sensor files found, but no data could be read from any file.")
        
    try:
        sensor_data = pd.concat(sensor_dfs, axis=1)
    except ValueError as e:
        raise ValueError("No objects to concatenate. Check sensor files for valid data.") from e
    return sensor_data.values

def load_profile(dataset_dir, delimiter='\t'):
    profile_path = os.path.join(dataset_dir, "profile.txt")
    y = pd.read_csv(profile_path, delimiter=delimiter, header=None)
    return y.values

def preprocess_data(X):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == '__main__':
    # Use proper absolute path without a leading slash.
    dataset_dir = r"dataset"
    
    X = load_all_sensor_data(dataset_dir)
    y = load_profile(dataset_dir)
    
    X_scaled = preprocess_data(X)
    X_train, X_test, y_train, y_test = create_train_test_split(X_scaled, y)
    
    # Optionally, save the splits
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
