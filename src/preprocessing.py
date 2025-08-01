import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import os

class PEECOMDataProcessor:
    def __init__(self, config_path=None):
        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "model": {
                    "input_timesteps": 10,
                    "batch_size": 8
                }
            }
        
        # Define sensor mapping
        self.sensor_mapping = {
            'PS': {'count': 6, 'cols': list(range(6))},  # PS1-PS6 in first 6 columns
            'FS': {'count': 2, 'cols': list(range(6, 8))},  # FS1-FS2
            'TS': {'count': 4, 'cols': list(range(8, 12))},  # TS1-TS4
            'VS': {'count': 1, 'cols': [12]},  # VS1
            'CE': {'count': 1, 'cols': [13]},  # CE
            'CP': {'count': 1, 'cols': [14]},  # CP
            'SE': {'count': 1, 'cols': [15]}   # SE
        }

    def _assign_sensor_names(self, df):
        """Map numeric columns to sensor names"""
        df.columns = df.columns.astype(str)  # Ensure string columns
        
        # Create mapping of column index to sensor name
        new_columns = []
        for i in range(df.shape[1]):
            assigned = False
            for sensor_type, config in self.sensor_mapping.items():
                if i in config['cols']:
                    idx = config['cols'].index(i)
                    new_columns.append(f"{sensor_type}{idx+1}")
                    assigned = True
                    break
            if not assigned:
                new_columns.append(f"F{i}")  # Fallback name for unmapped columns
        
        df.columns = new_columns
        return df

    def process_all(self):
        # Dummy data for demonstration
        X_df_tr = pd.DataFrame(np.random.randn(100, 60))
        X_df_te = pd.DataFrame(np.random.randn(20, 60))
        
        # Assign proper sensor names
        X_df_tr = self._assign_sensor_names(X_df_tr)
        X_df_te = self._assign_sensor_names(X_df_te)
        
        # Add dummy profile data
        prof_tr = {"stable_flag": np.random.randint(0, 2, 100)}
        prof_te = {"stable_flag": np.random.randint(0, 2, 20)}
        
        return X_df_tr, X_df_te, pd.DataFrame(prof_tr), pd.DataFrame(prof_te)

def create_sequences(data, timesteps):
    seqs = []
    for i in range(len(data) - timesteps + 1):
        seqs.append(data[i:i+timesteps])
    return np.stack(seqs, axis=0)

def analyze_fold_distribution(X: np.ndarray, y: np.ndarray, 
                            fold_indices: list, 
                            sensor_names: list,
                            output_dir: str = "output") -> dict:
    """Analyze data distribution across folds"""
    os.makedirs(output_dir, exist_ok=True)
    
    fold_stats = []
    for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
        # Class distribution
        train_pos = np.mean(y[train_idx])
        val_pos = np.mean(y[val_idx])
        
        # Feature statistics
        train_stats = {
            'mean': np.mean(X[train_idx], axis=0),
            'std': np.std(X[train_idx], axis=0),
            'q1': np.percentile(X[train_idx], 25, axis=0),
            'q3': np.percentile(X[train_idx], 75, axis=0)
        }
        
        val_stats = {
            'mean': np.mean(X[val_idx], axis=0),
            'std': np.std(X[val_idx], axis=0),
            'q1': np.percentile(X[val_idx], 25, axis=0),
            'q3': np.percentile(X[val_idx], 75, axis=0)
        }
        
        # Plot sensor distributions
        plt.figure(figsize=(15, 5))
        for i, sensor in enumerate(sensor_names[:6]):  # Plot first 6 sensors
            plt.subplot(2, 3, i+1)
            sns.kdeplot(X[train_idx][:, i], label='Train')
            sns.kdeplot(X[val_idx][:, i], label='Val')
            plt.title(f'{sensor} Distribution - Fold {fold_idx+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fold_{fold_idx+1}_distributions.png'))
        plt.close()
        
        fold_stats.append({
            'fold': fold_idx + 1,
            'train_pos_rate': train_pos,
            'val_pos_rate': val_pos,
            'train_stats': train_stats,
            'val_stats': val_stats,
            'distribution_shift': np.mean(np.abs(train_stats['mean'] - val_stats['mean']))
        })
    
    return fold_stats
