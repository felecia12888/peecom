#!/usr/bin/env python3
"""
PEECOM Main Application

Physics-Enhanced Equipment Condition Monitoring System
Main entry point supporting multiple execution modes.
"""

from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dropout, MaxPool1D,
    Bidirectional, LSTM, Dense, GlobalAveragePooling1D, Concatenate, Flatten, Activation, RepeatVector, Permute, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Attention, SpatialDropout1D, LayerNormalization
from src.models import *
from src.utils import impute_ps4_from_ps5_ps6
from src.loader.dataset_checker import analyze_dataset
from src.loader import PEECOMDataProcessor, create_sequences, SensorValidator, SensorMonitor
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from src.utils.argument_parser import PEECOMArgumentParser, create_config_from_args, save_config
import os
import sys
import yaml
import random
from typing import Dict, Any
import numpy as np
import tensorflow as tf
import scipy.stats as stats
import seaborn as sns
import pandas as pd
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import argument parser

# Enable experimental features
# sklearn imports

# Local imports - updated for new structure

# TensorFlow imports

K = tf.keras.backend

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def compute_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def custom_focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = tf.exp(-bce)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        loss = alpha_t * (1 - p_t)**gamma * bce
        return tf.reduce_mean(loss)
    return focal_loss


def extract_stats_features(X):
    features = []
    for i in range(X.shape[0]):
        window = X[i]
        flat = window.flatten()
        features.append([
            np.mean(window),
            np.std(window),
            np.median(window),
            np.max(window) - np.min(window),
            np.quantile(flat, 0.25),
            np.quantile(flat, 0.75),
            stats.skew(flat),
            stats.kurtosis(flat),
            np.mean(np.abs(np.diff(window, axis=0)))
        ])
    return np.array(features)


def attention_block(inputs):
    # Use tf.keras.layers.Attention for better compatibility
    # Expand dims to match Attention's expected shape: (batch, time, features)
    attn = Attention()([inputs, inputs])
    return attn


def get_cosine_lr_schedule(initial_lr=1e-3):
    return CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=10,
        t_mul=2.0,
        m_mul=0.8,
        alpha=1e-5
    )


def build_cnn_bilstm_model(timesteps, n_features, stats_dim=0, hp=None):
    seq_input = Input(shape=(timesteps, n_features), name="seq_input")
    x = Conv1D(32, 3, activation='relu', padding='same',
               kernel_regularizer=l2(1e-3))(seq_input)
    x = SpatialDropout1D(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, 3, activation='relu', padding='same',
               kernel_regularizer=l2(1e-3))(x)
    x = SpatialDropout1D(0.2)(x)
    x = MaxPool1D(2)(x)
    x = Bidirectional(LSTM(32, return_sequences=True, dropout=0.3,
                      recurrent_dropout=0.3, kernel_regularizer=l2(1e-3)))(x)
    x = LayerNormalization()(x)
    x = attention_block(x)
    x = GlobalAveragePooling1D()(x)
    if stats_dim > 0:
        stats_input = Input(shape=(stats_dim,), name="stats_input")
        x = Concatenate()([x, stats_input])
        inputs = [seq_input, stats_input]
    else:
        inputs = seq_input
    x = Dense(16, activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, out)
    # Use cosine decay schedule directly in the optimizer (no LR scheduler callback)
    optimizer = Adam(learning_rate=get_cosine_lr_schedule(1e-3))
    model.compile(
        optimizer=optimizer,
        loss=custom_focal_loss(alpha=0.3, gamma=2.5),
        metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(
            name='recall'), AUC(name='pr_auc', curve='PR')]
    )
    return model


def tune_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def rolling_mean(values, window=3):
    return np.convolve(values, np.ones(window)/window, mode='valid')


# Penalize FN more
def optimize_threshold_for_cost(y_true, probs, fp_cost=1.0, fn_cost=4.0):
    best_cost = float('inf')
    best_thr = 0.5
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = fp * fp_cost + fn * fn_cost
        if cost < best_cost:
            best_cost, best_thr = cost, thr
    return best_thr


def augment_sequences(X, y, noise_level=0.02, augment_ratio=0.3):
    X_aug, y_aug, stats_aug_idx = [], [], []
    for i in range(len(X)):
        if y[i] == 1:  # Only augment minority class
            if random.random() < augment_ratio:
                # Time warping (simple interpolation)
                warped = np.apply_along_axis(
                    lambda x: np.interp(
                        np.linspace(0, 1, len(x)),
                        np.linspace(0, 1, len(x)),
                        x
                    ),
                    axis=0, arr=X[i]
                )
                # Gaussian noise
                noisy = warped + np.random.normal(0, noise_level, warped.shape)
                if noisy.ndim == 2:
                    X_aug.append(noisy[np.newaxis, ...])
                else:
                    X_aug.append(noisy)
                y_aug.append([y[i]])
                stats_aug_idx.append(i)
    if X_aug:
        X_aug = np.concatenate(X_aug, axis=0)
        y_aug = np.concatenate(y_aug, axis=0)
        return np.concatenate([X, X_aug], axis=0), np.concatenate([y, y_aug], axis=0), stats_aug_idx
    else:
        return X, y, []


def temporal_augmentation(X, y, timesteps, augment_factor=0.3):
    X_aug, y_aug = [], []
    for i in range(len(X)):
        if y[i] == 1 and random.random() < augment_factor:
            warp_factor = 0.8 + random.random() * 0.4
            warped = np.apply_along_axis(
                lambda x: np.interp(
                    np.linspace(0, len(x)-1, int(len(x)*warp_factor)),
                    np.arange(len(x)), x
                ), axis=0, arr=X[i]
            )
            start = random.randint(0, max(1, warped.shape[0]-timesteps))
            cropped = warped[start:start+timesteps]
            if cropped.shape[0] == timesteps:
                X_aug.append(cropped)
                y_aug.append(y[i])
    if X_aug:
        return np.array(X_aug), np.array(y_aug)
    else:
        return np.empty((0, X.shape[1], X.shape[2])), np.empty((0,))


def plot_confusion_matrix(cm, fold, show=False):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Fold {fold} Confusion Matrix')
    plt.savefig(f"output/cm_fold{fold}.png")
    if show:
        plt.show()
    plt.close()


def augment_and_sync(X, y, stats, timesteps, noise_level=0.02, augment_ratio=0.3):
    # Centralized augmentation: both augment_sequences and temporal_augmentation, keep stats in sync
    X_aug, y_aug, stats_aug_idx = augment_sequences(
        X, y, noise_level=noise_level, augment_ratio=augment_ratio)
    if stats_aug_idx:
        stats_aug = np.concatenate([stats, stats[stats_aug_idx]], axis=0)
    else:
        stats_aug = stats

    X_temp_aug, y_temp_aug = temporal_augmentation(
        X, y, timesteps, augment_factor=augment_ratio)
    if X_temp_aug.shape[0] > 0:
        stats_temp_aug = np.repeat(
            stats, X_temp_aug.shape[0] // X.shape[0], axis=0)
        remainder = X_temp_aug.shape[0] % X.shape[0]
        if remainder > 0:
            stats_temp_aug = np.concatenate(
                [stats_temp_aug, stats[:remainder]], axis=0)
        X_aug = np.concatenate([X_aug, X_temp_aug], axis=0)
        y_aug = np.concatenate([y_aug, y_temp_aug], axis=0)
        stats_aug = np.concatenate([stats_aug, stats_temp_aug], axis=0)
    return X_aug, y_aug, stats_aug


# --- Custom EarlyStopping on smoothed PR-AUC ---
class SmoothedEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=3, window=3, monitor="val_pr_auc", mode="max"):
        super().__init__()
        self.patience = patience
        self.window = window
        self.monitor = monitor
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf
        self.wait = 0
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        val = logs.get(self.monitor)
        self.history.append(val)
        if len(self.history) >= self.window:
            smoothed = np.mean(self.history[-self.window:])
            if (self.mode == "max" and smoothed > self.best) or (self.mode == "min" and smoothed < self.best):
                self.best = smoothed
                self.wait = 0
            else:
                self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


def analyze_fold_distribution(X, y, train_idx, val_idx, logger):
    """Analyze data distribution in train/val splits"""
    # Basic statistics
    train_pos = np.mean(y[train_idx])
    val_pos = np.mean(y[val_idx])

    # Check for temporal patterns
    train_std = np.std(X[train_idx], axis=0).mean()
    val_std = np.std(X[val_idx], axis=0).mean()

    logger.info(
        f"Train positive rate: {train_pos:.3f}, Val positive rate: {val_pos:.3f}")
    logger.info(f"Train/Val std ratio: {train_std/val_std:.3f}")

    return abs(train_pos - val_pos) > 0.1 or train_std/val_std > 1.5


def build_lighter_model(timesteps, n_features, stats_dim=0):
    """Simplified model for harder folds"""
    seq_input = Input(shape=(timesteps, n_features), name="seq_input")

    # Simple CNN backbone
    x = Conv1D(16, 3, activation='relu', padding='same')(seq_input)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)

    if stats_dim > 0:
        stats_input = Input(shape=(stats_dim,), name="stats_input")
        x = Concatenate()([x, stats_input])
        inputs = [seq_input, stats_input]
    else:
        inputs = seq_input

    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)  # Increased dropout
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, out)
    optimizer = Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',  # Simpler loss
        metrics=['accuracy', AUC(name='auc'), AUC(name='pr_auc', curve='PR')]
    )
    return model


# Add tf.function with reduce_retracing for prediction
@tf.function(reduce_retracing=True)
def model_predict(model, inputs):
    """Wrap model prediction in a graph function to avoid retracing"""
    return model(inputs, training=False)


def run_fold(fold, train_idx, val_idx, X_seq, y_seq, stats_features, timesteps, stats_dim, model_builder, compute_class_weights, callbacks_fn, logger):
    # Check fold distribution
    if analyze_fold_distribution(X_seq, y_seq, train_idx, val_idx, logger):
        logger.warning(
            f"Fold {fold+1} shows significant distribution shift - using lighter model")
        model_builder = build_lighter_model

    X_tr, X_val = X_seq[train_idx], X_seq[val_idx]
    y_tr, y_val = y_seq[train_idx], y_seq[val_idx]
    stats_tr, stats_val = stats_features[train_idx], stats_features[val_idx]

    # --- FIX: Use modest, fixed augmentation ---
    fold_augment_ratio = 0.2
    fold_noise_level = 0.02
    X_tr_aug, y_tr_aug, stats_tr_aug = augment_and_sync(
        X_tr, y_tr, stats_tr, timesteps, noise_level=fold_noise_level, augment_ratio=fold_augment_ratio
    )

    class_weights = compute_class_weights(y_tr_aug)
    logger.info(f"Class weights: {class_weights}")

    model = model_builder(timesteps, X_seq.shape[-1], stats_dim=stats_dim)

    # Verify model input shape using model.input_shape
    logger.info(f"Fold {fold+1} model input shape verification:")
    if isinstance(model.input, list):
        logger.info(f"Sequential input shape: {model.input[0].shape}")
        logger.info(f"Stats input shape: {model.input[1].shape}")
    else:
        logger.info(f"Sequential input shape: {model.input.shape}")

    callbacks = callbacks_fn(fold)

    batch_size = 32 if len(X_tr_aug) > 1000 else 16

    history = model.fit(
        [X_tr_aug, stats_tr_aug], y_tr_aug,
        validation_data=([X_val, stats_val], y_val),
        epochs=50,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Convert validation data to tensors once
    val_inputs = [tf.convert_to_tensor(X_val), tf.convert_to_tensor(stats_val)]

    # Threshold tuning (once per fold)
    probs = model_predict(model, val_inputs).numpy().ravel()
    best_thr, best_f1 = tune_threshold(y_val, probs)
    preds = (probs > best_thr).astype(int)

    fold_metrics = {
        "fold": fold+1,
        "best_thr": best_thr,
        "f1": f1_score(y_val, preds),
        "auc": roc_auc_score(y_val, probs),
        "pr_auc": average_precision_score(y_val, probs),
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
        "confusion_matrix": confusion_matrix(y_val, preds)
    }
    logger.info(f"Fold {fold+1} metrics: {fold_metrics}")
    # Show confusion matrix for this fold
    plot_confusion_matrix(fold_metrics["confusion_matrix"], fold+1, show=True)
    return fold_metrics


def ensemble_predictions(models, X_val, stats_val):
    """Optimized ensemble predictions"""
    all_probs = []
    # Create tensor inputs once
    inputs = [tf.convert_to_tensor(X_val), tf.convert_to_tensor(stats_val)]

    for model in models:
        # Use the traced prediction function
        probs = model_predict(model, inputs)
        all_probs.append(probs.numpy().ravel())
    return np.mean(all_probs, axis=0)


def train_cv(X_seq, y_seq, stats_features, timesteps, stats_dim, model_builder, compute_class_weights, callbacks_fn, logger):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    models = []  # Store models for ensembling

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_seq, y_seq)):
        fold_metrics = run_fold(
            fold, train_idx, val_idx, X_seq, y_seq, stats_features, timesteps, stats_dim,
            model_builder, compute_class_weights, callbacks_fn, logger
        )
        metrics.append(fold_metrics)

        # Load best model for this fold
        best_model = tf.keras.models.load_model(
            f"output/best_fold{fold+1}_pr_auc.keras",
            compile=False
        )
        models.append(best_model)

    return metrics, models


def cosine_scheduler(epoch, lr):
    # Ensure output is a Python float, not numpy/TF type
    return float(get_cosine_lr_schedule(1e-3)(epoch))


def callbacks_fn(fold):
    return [
        EarlyStopping(monitor="val_pr_auc", patience=8,
                      mode="max", restore_best_weights=True),
        ModelCheckpoint(
            f"output/best_fold{fold+1}_pr_auc.keras", save_best_only=True, monitor='val_pr_auc')
    ]


# --- Log learning rate each epoch ---
class LRLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        print(f"Epoch {epoch+1}: learning rate is {lr:.6f}")


def verify_sensor_columns(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, List[str]]:
    """Verify and normalize sensor column names"""
    # Get all pressure sensor columns (case-insensitive, handle non-string columns)
    pressure_cols = [
        col for col in df.columns
        if str(col).upper().startswith('PS')
    ]
    logger.info(f"Found pressure sensors: {pressure_cols}")

    # Map to standard names (ensure string handling)
    sensor_map = {}
    for col in pressure_cols:
        std_name = ''.join(filter(str.isdigit, str(col)))
        if std_name:
            sensor_map[f'PS{std_name}'] = col

    missing = [f'PS{i}' for i in range(1, 7) if f'PS{i}' not in sensor_map]
    logger.info(f"Missing sensors: {missing}")

    return {
        'found': pressure_cols,
        'missing': missing,
        'mapping': sensor_map
    }


def impute_pressure_sensors(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict]:
    """Flexible pressure sensor imputation with fallbacks"""
    # Store original PS4 for comparison if it exists
    orig_ps4 = df['PS4'].copy() if 'PS4' in df.columns else None

    # First try PS3/PS5 regression if available
    sensor_info = verify_sensor_columns(df, logger)
    logger.info(f"Sensor verification results: {sensor_info}")

    if all(f'PS{i}' in sensor_info['mapping'] for i in [3, 4, 5]):
        try:
            logger.info("Attempting PS3/PS5 regression for PS4...")
            df = impute_ps4_from_ps5_ps6(
                df, alpha=1.0, zero_threshold=0.1, logger=logger)
        except Exception as e:
            logger.warning(f"PS3/PS5 regression failed: {e}")
    else:
        logger.warning(
            "Cannot use PS3/PS5 regression - missing required sensors")

    # If PS4 still needs imputation, try SimpleImputer first
    if 'PS4' in df.columns and (df['PS4'].isna().any() or (df['PS4'] == 0).any()):
        try:
            logger.info("Using SimpleImputer for remaining missing values...")
            imputer = SimpleImputer(strategy='mean')
            pressure_cols = [
                c for c in df.columns if c.upper().startswith('PS')]
            df_pressure = pd.DataFrame(
                imputer.fit_transform(df[pressure_cols]),
                columns=pressure_cols,
                index=df.index
            )
            df[pressure_cols] = df_pressure
        except Exception as e:
            logger.warning(
                f"SimpleImputer failed: {e}, trying IterativeImputer...")
            try:
                imp = IterativeImputer(
                    max_iter=10,
                    random_state=42,
                    initial_strategy='mean',
                    skip_complete=True
                )
                df_pressure = pd.DataFrame(
                    imp.fit_transform(df[pressure_cols]),
                    columns=pressure_cols,
                    index=df.index
                )
                df[pressure_cols] = df_pressure
            except Exception as e2:
                logger.error(f"All imputation methods failed: {e2}")

    # Log imputation results
    if orig_ps4 is not None and 'PS4' in df.columns:
        zero_before = (orig_ps4 == 0).mean() * 100
        zero_after = (df['PS4'] == 0).mean() * 100
        logger.info(
            f"PS4 zeros before: {zero_before:.2f}%, after: {zero_after:.2f}%")

    return df, sensor_info


def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration"""
    log_level = config.get('output', {}).get('log_level', 'INFO')
    output_dir = config.get('output', {}).get('base_dir', 'output')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'peecom.log')),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def setup_tensorflow(config: Dict[str, Any]):
    """Setup TensorFlow configuration"""
    perf_config = config.get('performance', {})

    # GPU configuration
    if perf_config.get('use_gpu', False):
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                # Enable memory growth
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)

                # Set memory limit if specified
                if perf_config.get('memory_limit'):
                    tf.config.experimental.set_memory_limit(
                        physical_devices[0],
                        int(perf_config['memory_limit'] * 1024)
                    )

                logging.info(
                    f"GPU enabled: {len(physical_devices)} devices found")
            except RuntimeError as e:
                logging.warning(f"GPU setup failed: {e}")
    else:
        # Force CPU usage
        tf.config.set_visible_devices([], 'GPU')
        logging.info("Using CPU only")

    # Mixed precision
    if perf_config.get('mixed_precision', False):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logging.info("Mixed precision enabled")

    # Set random seeds for reproducibility
    random_state = config.get('data', {}).get('random_state', 42)
    tf.random.set_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


def run_preprocessing(config: Dict[str, Any], logger: logging.Logger):
    """Run data preprocessing pipeline"""
    logger.info("Running data preprocessing...")

    # Initialize data processor
    processor = PEECOMDataProcessor(
        config.get('config', 'src/config/config.yaml'))

    # Load and process data
    X_tr_df, X_te_df, prof_tr, prof_te = processor.process_all()

    # Save processed data
    output_dir = config.get('output', {}).get('base_dir', 'output')
    processed_data_dir = os.path.join(output_dir, 'processed_data')
    os.makedirs(processed_data_dir, exist_ok=True)

    np.save(os.path.join(processed_data_dir, 'X_train.npy'), X_tr_df.values)
    np.save(os.path.join(processed_data_dir, 'X_test.npy'), X_te_df.values)
    np.save(os.path.join(processed_data_dir, 'y_train.npy'), prof_tr.values)
    np.save(os.path.join(processed_data_dir, 'y_test.npy'), prof_te.values)

    logger.info(f"Preprocessed data saved to {processed_data_dir}")
    return X_tr_df, X_te_df, prof_tr, prof_te


def run_analysis(config: Dict[str, Any], logger: logging.Logger):
    """Run dataset analysis"""
    logger.info("Running dataset analysis...")

    dataset_path = config.get('data', {}).get(
        'dataset_path', 'dataset/dataset')
    analysis_types = config.get('analysis', {}).get(
        'types', ['basic', 'correlation', 'health'])

    # Run analysis using the dataset checker
    results = analyze_dataset(dataset_path)

    logger.info("Dataset analysis completed")
    return results


def run_training(config: Dict[str, Any], logger: logging.Logger):
    """Run model training"""
    logger.info("Running model training...")

    # Load or preprocess data
    try:
        # Try to load preprocessed data first
        output_dir = config.get('output', {}).get('base_dir', 'output')
        processed_data_dir = os.path.join(output_dir, 'processed_data')
        X_train = np.load(os.path.join(processed_data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(processed_data_dir, 'y_train.npy'))
        logger.info("Loaded preprocessed data")
    except FileNotFoundError:
        # Preprocess data if not found
        logger.info("Preprocessed data not found, running preprocessing...")
        X_tr_df, X_te_df, prof_tr, prof_te = run_preprocessing(config, logger)
        X_train = X_tr_df.values
        y_train = prof_tr['stable_flag'].values

    # Setup cross-validation or single training
    cv_config = config.get('cross_validation', {})
    if cv_config.get('folds', 1) > 1:
        logger.info(f"Running {cv_config['folds']}-fold cross-validation")
        # Use existing train_cv function
        # You would call your train_cv function here
        pass
    else:
        logger.info("Running single model training")
        # Use existing training logic
        # You would call your single training function here
        pass


def run_evaluation(config: Dict[str, Any], logger: logging.Logger):
    """Run model evaluation"""
    logger.info("Running model evaluation...")

    model_path = config.get('model_path')
    if not model_path:
        raise ValueError("Model path required for evaluation mode")

    # Load model and run evaluation
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Loaded model from {model_path}")

    # Add evaluation logic here
    pass


def run_hyperparameter_tuning(config: Dict[str, Any], logger: logging.Logger):
    """Run hyperparameter tuning"""
    logger.info("Running hyperparameter tuning...")

    tune_config = config.get('hyperparameter_tuning', {})
    tuner_type = tune_config.get('tuner', 'random')
    max_trials = tune_config.get('max_trials', 50)

    logger.info(f"Using {tuner_type} tuner with {max_trials} trials")

    # Add hyperparameter tuning logic here
    pass


def run_pipeline(config: Dict[str, Any], logger: logging.Logger):
    """Run complete pipeline (preprocess -> train -> evaluate)"""
    logger.info("Running complete pipeline...")

    # Run preprocessing
    run_preprocessing(config, logger)

    # Run training
    run_training(config, logger)

    # Run evaluation
    try:
        run_evaluation(config, logger)
    except ValueError as e:
        logger.warning(f"Skipping evaluation: {e}")


def main():
    """Main execution function"""
    try:
        # Parse command line arguments
        parser = PEECOMArgumentParser()
        args = parser.parse_args()

        # Create configuration from arguments
        config = create_config_from_args(args)

        # Merge with config file if it exists
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            # Command line args override file config
            for key, value in config.items():
                if isinstance(value, dict) and key in file_config:
                    file_config[key].update(value)
                else:
                    file_config[key] = value
            config = file_config

        # Setup logging
        logger = setup_logging(config)
        logger.info("PEECOM started")
        logger.info(f"Mode: {config['mode']}")

        # Setup TensorFlow
        setup_tensorflow(config)

        # Save configuration
        output_dir = config.get('output', {}).get('base_dir', 'output')
        save_config(config, os.path.join(output_dir, 'config_used.yaml'))

        # Route to appropriate function based on mode
        mode = config['mode']

        if mode == 'preprocess':
            run_preprocessing(config, logger)
        elif mode == 'analyze':
            run_analysis(config, logger)
        elif mode == 'train':
            run_training(config, logger)
        elif mode == 'evaluate':
            run_evaluation(config, logger)
        elif mode == 'tune':
            run_hyperparameter_tuning(config, logger)
        elif mode == 'pipeline':
            run_pipeline(config, logger)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        logger.info("PEECOM completed successfully")

    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
