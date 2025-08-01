import numpy as np
import tensorflow as tf
from src.preprocessing import PEECOMDataProcessor, create_sequences

def build_datasets(config_path="config/config.yaml"):
    """
    1) Loads raw tabular sensor + profile  
    2) Splits temporally  
    3) Turns them into sequences  
    4) Wraps into tf.data.Dataset
    """
    # 1) load & split
    dp = PEECOMDataProcessor(config_path)
    X_tr_df, X_te_df, prof_tr, prof_te = dp.process_all()
    
    # 2) labels are 'stable_flag': 0 or 1
    y_tr = prof_tr['stable_flag'].values
    y_te = prof_te['stable_flag'].values

    # 3) create time-series windows
    timesteps = int(dp.config['model']['input_timesteps'])
    X_tr = create_sequences(X_tr_df.values, timesteps)
    X_te = create_sequences(X_te_df.values, timesteps)
    
    # 4) expand y to match sequence length
    y_tr_seq = np.repeat(y_tr[:,None], timesteps, axis=1)
    y_te_seq = np.repeat(y_te[:,None], timesteps, axis=1)
    
    # 5) build tf.data.Datasets
    train_ds = tf.data.Dataset.from_tensor_slices((
        X_tr.astype(np.float32),
        {"anomaly_prob": y_tr_seq.astype(np.int32), 
         "control_params": np.zeros((len(y_tr_seq), timesteps, 2), dtype=np.float32)}
    ))
    val_ds = tf.data.Dataset.from_tensor_slices((
        X_te.astype(np.float32),
        {"anomaly_prob": y_te_seq.astype(np.int32),
         "control_params": np.zeros((len(y_te_seq), timesteps, 2), dtype=np.float32)}
    ))
    
    # 6) batch & prefetch
    batch_size = int(dp.config['model']['batch_size'])
    train_ds = train_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds
