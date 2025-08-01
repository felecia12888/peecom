import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import tensorflow as tf

from src.model import PEECOMHyperModel
from src.pipeline import build_datasets

# Import tqdm callback for progress bar logging
from custom_tqdm_callback import CustomTqdmCallback

# --- 2. Build data & model ---
train_ds, val_ds = build_datasets("config/config.yaml")

timesteps = 10    # match your config
n_features = 60

hypermodel = PEECOMHyperModel(input_shape=(timesteps, n_features), use_baseline=True)
model      = hypermodel.build(hp={})     # or load your best hp

# --- 3. Compute class weights ---
y_tr = np.concatenate([
    y['anomaly_prob'].numpy().reshape(-1) 
    for _, y in train_ds
], axis=0)
cw = class_weight.compute_class_weight("balanced", classes=[0,1], y=y_tr)
class_weights = {0: cw[0], 1: cw[1]}
print("Using class weights:", class_weights)

# --- 4. Recompile for anomaly only ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
      "anomaly_prob": tf.keras.losses.BinaryFocalCrossentropy(gamma=2, label_smoothing=0.1),
      "control_params": tf.keras.losses.Huber()
    },
    loss_weights={"anomaly_prob":1.0, "control_params":0.0},
    metrics={"anomaly_prob": ["accuracy", tf.keras.metrics.AUC(name="auc")]}
)

# --- 5. Fit with class weights and logging ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    class_weight=class_weights,
    callbacks=[
      tf.keras.callbacks.EarlyStopping(
        "val_anomaly_prob_accuracy", patience=5, mode="max", restore_best_weights=True
      ),
      CustomTqdmCallback(verbose=1)  # Add tqdm logging
    ],
    verbose=0  # Suppress default Keras logging so tqdm bar is visible
)

# --- 6. Threshold sweep on val set ---
y_true_val = []
y_score_val = []
for x, y in val_ds:
    preds = model.predict(x, verbose=0)["anomaly_prob"].reshape(-1)
    y_score_val.append(preds)
    y_true_val.append(y["anomaly_prob"].numpy().reshape(-1))

y_score_val = np.concatenate(y_score_val)
y_true_val  = np.concatenate(y_true_val)

best = {"f1":0}
for t in np.linspace(0,1,101):
    p = (y_score_val >= t).astype(int)
    f1 = f1_score(y_true_val, p)
    if f1 > best["f1"]:
        best = {
            "threshold": t,
            "f1": f1,
            "acc": accuracy_score(y_true_val, p),
            "prec": precision_score(y_true_val, p),
            "rec": recall_score(y_true_val, p)
        }

print("Optimal threshold sweep:", best)
