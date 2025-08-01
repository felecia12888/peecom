# src/utils.py

import os
import gc
import pickle
import datetime
import yaml
import logging  # Add logging import
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)
import tensorflow as tf
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

def visualize_results(
    history: Any,
    output_dir: str = "output",
    predictions: Optional[Dict[str, np.ndarray]] = None,
    y_test: Optional[Dict[str, np.ndarray]] = None,
    attention_weights: Optional[np.ndarray] = None,
    energy_savings: Optional[np.ndarray] = None
) -> None:
    """Memory-optimized visualization function with TF-compatible data handling"""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Convert attention_weights to numpy array safely
    if attention_weights is not None:
        attention_weights = attention_weights.numpy() if hasattr(attention_weights, 'numpy') else np.array(attention_weights)

    # Plot loss trends
    if hasattr(history, 'history'):
        history = history.history
        
    if 'loss' in history:
        fig, ax = plt.subplots()
        ax.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_title('Loss Trends')
        ax.legend()
        fig.savefig(os.path.join(plots_dir, "loss_trends.png"))
        plt.close(fig)
        gc.collect()

    # Energy savings plots
    if energy_savings is not None:
        energy_savings = energy_savings.numpy() if hasattr(energy_savings, 'numpy') else energy_savings
        
        fig, ax = plt.subplots()
        ax.plot(energy_savings)
        ax.set_title('Energy Savings per Cycle')
        fig.savefig(os.path.join(plots_dir, "energy_savings.png"))
        plt.close(fig)
        gc.collect()

    # Prediction metrics
    if predictions and y_test:
        # Convert TF tensors to numpy arrays safely
        predictions = {k: (v.numpy() if hasattr(v, 'numpy') else np.array(v)) for k, v in predictions.items()}
        y_test = {k: (v.numpy() if hasattr(v, 'numpy') else np.array(v)) for k, v in y_test.items()}

        # Anomaly detection metrics
        if 'anomaly_prob' in predictions and 'anomaly_prob' in y_test:
            y_true = y_test['anomaly_prob']
            y_prob = predictions['anomaly_prob']
            
            # Confusion matrix
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_true, y_prob > 0.5)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            fig.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
            plt.close(fig)
            gc.collect()

        # Control parameters visualization
        if 'control_params' in predictions and 'control_params' in y_test:
            fig = plot_control_params(predictions['control_params'], y_test['control_params'])
            fig.savefig(os.path.join(plots_dir, "control_params_prediction.png"))
            plt.close(fig)
            gc.collect()

    # Attention weights visualization
    if attention_weights is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(attention_weights.squeeze(), ax=ax)
        fig.savefig(os.path.join(plots_dir, "attention_weights_heatmap.png"))
        plt.close(fig)
        gc.collect()

def plot_control_params(pred: np.ndarray, true: np.ndarray) -> plt.Figure:
    """Helper function for control parameter visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, name in enumerate(['Valve', 'Pump']):
        axes[i].scatter(true[:, i], pred[:, i], alpha=0.5)
        axes[i].plot([true.min(), true.max()], [true.min(), true.max()], 'r--')
        axes[i].set_title(f'{name} Predictions')
    return fig

def save_artifacts(
    model: tf.keras.Model,
    output_dir: str = "output",
    scaler: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Memory-safe artifact saving with TF 2.x compatibility"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model weights only to conserve memory
    model.save_weights(os.path.join(output_dir, "model_weights.h5"))
    
    # Save scaler if provided
    if scaler is not None:
        with open(os.path.join(output_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
    
    # Save config
    if config is not None:
        with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
            yaml.safe_dump(config, f)
    
    gc.collect()

def generate_performance_summary(
    history: tf.keras.callbacks.History,
    predictions: Dict[str, np.ndarray],
    y_test: Dict[str, np.ndarray],
    energy_savings: float
) -> str:
    """Generate text summary without memory-intensive operations"""
    lines = [
        f"Training Loss: {history.history['loss'][-1]:.4f}",
        f"Validation Loss: {history.history['val_loss'][-1]:.4f}",
        f"Energy Savings: {energy_savings:.2f} W"
    ]
    return "\n".join(lines)

def create_sequences(data: np.ndarray, timesteps: int) -> np.ndarray:
    """Memory-efficient sequence creation using stride tricks"""
    if data.ndim != 2:
        raise ValueError("Input data must be 2D (samples, features)")
    
    n_samples, n_features = data.shape
    new_shape = (n_samples - timesteps + 1, timesteps, n_features)
    strides = (data.strides[0], data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(
        data, shape=new_shape, strides=strides
    ).copy()  # Copy to prevent memory issues

def apply_control_action(
    cycle: np.ndarray, 
    action: float
) -> np.ndarray:
    """Apply control action with type safety"""
    return cycle * (1 - np.clip(action, 0, 1))

def calculate_energy_diff(
    original: np.ndarray, 
    adjusted: np.ndarray
) -> float:
    """Calculate energy difference with validation"""
    if original.shape != adjusted.shape:
        raise ValueError("Input arrays must have the same shape")
    return float(np.sum(original - adjusted))

def bootstrap_ps4_imputation(X_train: np.ndarray, y_train: np.ndarray, 
                           X_pred: np.ndarray, 
                           n_bootstraps: int = 10,
                           alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform bootstrapped PS4 imputation with uncertainty estimates
    Returns: (predictions, uncertainties)
    """
    all_preds = np.zeros((len(X_pred), n_bootstraps))
    
    for i in range(n_bootstraps):
        # Bootstrap sample with replacement
        X_boot, y_boot = resample(X_train, y_train, random_state=i)
        
        # Fit Ridge regression on bootstrap sample
        model = Ridge(alpha=alpha)
        model.fit(X_boot, y_boot)
        
        # Predict and store
        all_preds[:, i] = model.predict(X_pred)
    
    # Calculate mean and uncertainty (std) across bootstraps
    predictions = np.mean(all_preds, axis=1)
    uncertainties = np.std(all_preds, axis=1)
    
    return predictions, uncertainties

def impute_ps4_from_ps5_ps6(df: pd.DataFrame, 
                           alpha: float = 1.0,
                           zero_threshold: float = 0.0,
                           n_bootstraps: int = 10,
                           logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, Dict]:
    """Enhanced PS4 imputation with bootstrapping and uncertainty tracking"""
    df = df.copy()
    
    # Ensure string columns and handle missing sensors
    df.columns = df.columns.astype(str)
    required = ['PS3', 'PS4', 'PS5']
    if not all(col in df.columns for col in required):
        if logger:
            logger.warning(f"Missing required sensors for PS4 imputation")
        return df
    
    # Get valid samples for training
    mask = df['PS4'] > zero_threshold
    X_train = df.loc[mask, ['PS3', 'PS5']].values
    y_train = df.loc[mask, 'PS4'].values
    X_pred = df[['PS3', 'PS5']].values
    
    if len(y_train) == 0:
        if logger:
            logger.warning("No valid PS4 samples for training")
        return df
    
    try:
        # Perform bootstrapped imputation
        predictions, uncertainties = bootstrap_ps4_imputation(
            X_train, y_train, X_pred, 
            n_bootstraps=n_bootstraps,
            alpha=alpha
        )
        
        # Store original PS4 for comparison
        orig_ps4 = df['PS4'].copy()
        
        # Update PS4 values and store uncertainty
        df['PS4'] = predictions
        df['PS4_uncertainty'] = uncertainties
        
        if logger:
            # Log imputation quality metrics
            mse = mean_squared_error(orig_ps4[mask], predictions[mask])
            mean_uncertainty = np.mean(uncertainties)
            logger.info(f"PS4 imputation MSE on valid samples: {mse:.4f}")
            logger.info(f"Mean imputation uncertainty: {mean_uncertainty:.4f}")
            
            # Plot uncertainty distribution
            plt.figure(figsize=(10, 4))
            plt.hist(uncertainties, bins=50)
            plt.title('PS4 Imputation Uncertainty Distribution')
            plt.xlabel('Uncertainty (std)')
            plt.ylabel('Count')
            plt.savefig('output/ps4_uncertainty_dist.png')
            plt.close()
            
    except Exception as e:
        if logger:
            logger.error(f"Bootstrapped imputation failed: {e}")
        # Fallback to simple imputation
        if 'PS4' in df.columns:
            imputer = SimpleImputer(strategy='mean')
            df['PS4'] = imputer.fit_transform(df[['PS4']])
            df['PS4_uncertainty'] = 1.0  # High uncertainty for fallback
    
    return df