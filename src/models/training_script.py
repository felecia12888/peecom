import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
import seaborn as sns

# === 1. Data Preprocessing ===
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

# Add feature scaling
scaler = tf.keras.layers.Normalization(axis=-1)
scaler.adapt(X_train)
X_train = scaler(X_train)
X_val = scaler(X_val)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.GaussianNoise(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])

# === 2. Model Architecture ===
def build_model(timesteps, features):
    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=(timesteps, features)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=True, 
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid'),
        tf.keras.layers.GaussianDropout(0.2)  # Add noise for stability
    ])
    
    # Reduce initial learning rate for stability
    initial_learning_rate = 5e-4
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# === 3. Class weights (from your earlier computation) ===
class_weights = {0: 2.116279069767442, 1: 1.8958333333333333}

# === 4. Threshold‐tuning callback ===
class ThresholdTuner(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_thr = 0.5

    def on_epoch_end(self, epoch, logs=None):
        # Predict probabilities on validation set
        probs = self.model.predict(self.X_val)
        best_f1, best_thr = 0.0, 0.5

        # Search thresholds between 0.1 and 0.9
        for thr in np.linspace(0.1, 0.9, 33):
            preds = (probs >= thr).astype(int).reshape(-1)
            f1 = f1_score(self.y_val, preds)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        # Log & store the best threshold found
        logs = logs or {}
        logs['val_f1']  = best_f1
        logs['val_thr'] = best_thr
        self.best_thr = best_thr

        print(f" — val_f1: {best_f1:.3f} @ thr={best_thr:.2f}")

# Instantiate ThresholdTuner before callbacks
tuner = ThresholdTuner(X_val, y_val)

# === 5. Any other callbacks you need ===
other_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    # …
]

callbacks = [
    tuner,
] + other_callbacks

# === 6. Train ===
timesteps = X_train.shape[1]
features = X_train.shape[2]

# Ensemble training
models = []
histories = []
fold_results = []  # Initialize fold_results list
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f'Training fold {fold + 1}/5')
    
    # Create and train model
    tf.random.set_seed(fold)
    model = build_model(timesteps, features)
    
    # Setup callbacks
    tuner = ThresholdTuner(X_val, y_val)
    callbacks = [
        tuner,
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'best_model_fold_{fold}.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.BackupAndRestore(backup_dir=f'./backup/fold_{fold}'),
        tf.keras.callbacks.TerminateOnNaN()
    ]
    
    # Train with larger batch size for stability
    history = model.fit(
        X_train[train_idx], y_train[train_idx],
        validation_data=(X_train[val_idx], y_train[val_idx]),
        class_weight=class_weights,
        epochs=100,
        batch_size=128,  # Increased batch size
        callbacks=callbacks
    )
    
    # Evaluate fold
    predictions = model.predict(X_train[val_idx])
    precisions, recalls, thresholds = precision_recall_curve(y_train[val_idx], predictions)
    
    # Find optimal threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Apply optimal threshold
    y_pred = (predictions >= optimal_threshold).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_train[val_idx], y_pred)
    report = classification_report(y_train[val_idx], y_pred, output_dict=True)
    
    # Store results for this fold
    fold_results.append({
        'threshold': optimal_threshold,
        'metrics': report,
        'confusion_matrix': cm
    })
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    # Confusion Matrix
    plt.subplot(131)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Fold {fold}')
    
    # PR Curve
    plt.subplot(132)
    plt.plot(recalls, precisions)
    plt.scatter(recalls[optimal_idx], precisions[optimal_idx], color='red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - Fold {fold}')
    
    plt.tight_layout()
    plt.savefig(f'fold_{fold}_evaluation.png')
    plt.close()
    
    models.append(model)
    histories.append(history)

# Ensemble prediction function
def ensemble_predict(X):
    preds = [model.predict(X) for model in models]
    return np.mean(preds, axis=0)

# Analyze results across folds
def analyze_fold_results(fold_results):
    print("\nCross-Validation Summary:")
    metrics = ['precision', 'recall', 'f1-score']
    
    for metric in metrics:
        values = [fold['metrics']['weighted avg'][metric] for fold in fold_results]
        print(f"Mean {metric}: {np.mean(values):.3f} (±{np.std(values):.3f})")
    
    # Identify problematic folds
    for i, result in enumerate(fold_results):
        f1 = result['metrics']['weighted avg']['f1-score']
        if f1 < 0.6:  # Flag poor performing folds
            print(f"\nWarning: Fold {i+1} shows poor performance (F1: {f1:.3f})")
            print("Confusion Matrix:")
            print(result['confusion_matrix'])

# Run analysis
analyze_fold_results(fold_results)