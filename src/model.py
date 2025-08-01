import tensorflow as tf
from keras_tuner import Hyperband, Objective, HyperModel
import numpy as np

# --- Hardware-Aware Configuration for AMD Ryzen 5 3500U, 8GB RAM, Vega 8 GPU ---
# Force CPU if GPU is not suitable for deep learning (Vega 8 is not ideal for heavy models)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Reduce TensorFlow memory usage
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.optimizer.set_experimental_options({'cpu_math_jit': True})
# Enable memory growth and debug mode
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
# TF Data debug
# tf.data.experimental.enable_debug_mode()  # Removed for production use

# Custom Layers
class CalibratedAnomalyHead(tf.keras.layers.Layer):
    def __init__(self, temp=0.5, **kwargs):
        super(CalibratedAnomalyHead, self).__init__(**kwargs)
        self.temp = temp

    def call(self, inputs):
        # Sigmoid activation for probability calibration
        calibrated = tf.sigmoid(inputs / self.temp)
        return calibrated

# NEW: Improved Physics-Attention Layers

class PhysicsConstraintAttention(tf.keras.layers.Layer):
    def __init__(self, pressure_idx=0, flow_idx=1, temp_idx=2, **kwargs):
        super().__init__(**kwargs)
        self.pressure_idx = pressure_idx
        self.flow_idx = flow_idx
        self.temp_idx = temp_idx

    def build(self, input_shape):
        # Learnable physics coefficients
        self.energy_scale = self.add_weight(shape=(1,), initializer='ones', trainable=True)
        self.entropy_weight = self.add_weight(shape=(1,), initializer='zeros', trainable=True)

    def call(self, inputs):
        # Use explicit indices for pressure, flow, temperature
        pressure = inputs[..., self.pressure_idx]
        flow = inputs[..., self.flow_idx]
        temperature = inputs[..., self.temp_idx]
        # Adaptive energy equations
        energy_in = self.energy_scale * pressure * flow
        energy_out = temperature * (0.8 + self.entropy_weight)
        # Stabilized, gradient-friendly energy difference using softplus
        energy_diff = tf.math.softplus(energy_in - energy_out)
        # Compute physics-guided attention weights over time steps
        # Expand dims so that softmax is applied along time axis
        attn_weights = tf.expand_dims(tf.nn.softmax(energy_diff, axis=1), -1)
        # Physics attention: weighted emphasis on inputs
        return inputs * attn_weights

class HybridPhysicsAttention(tf.keras.layers.Layer):
    def __init__(self, units, pressure_idx=0, flow_idx=1, temp_idx=2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # Data-driven attention adopting MultiHeadAttention
        self.data_attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=units)
        # Physics-driven attention module
        self.physics_attention = PhysicsConstraintAttention(
            pressure_idx=pressure_idx, flow_idx=flow_idx, temp_idx=temp_idx
        )
        # Gating mechanism to fuse pathways
        self.gate_dense = tf.keras.layers.Dense(1, activation='sigmoid')
        # Use a custom constraint instead of lambda
        self.learnable_temp = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.5),
            constraint=ClipConstraint(0.1, 2.0),
            trainable=True
        )

    def call(self, inputs):
        # Data-driven attention pathway (self-attention on inputs)
        data_attn = self.data_attention(inputs, inputs)
        # Physics-guided pathway
        physics_attn = self.physics_attention(inputs)
        # Form gating input by concatenating outputs and original inputs along features axis
        gate_input = tf.concat([data_attn, physics_attn, inputs], axis=-1)
        gate = self.gate_dense(gate_input)
        # Adaptive scaling for physics pathway
        physics_scaled = self.learnable_temp * physics_attn
        # Fuse pathways using gating: If gate=1, use data; if gate=0, use physics
        return gate * data_attn + (1 - gate) * physics_scaled

# NEW: Custom constraint to clip values between min_value and max_value
from tensorflow.keras.constraints import Constraint
class ClipConstraint(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)
    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}

# --- Explicit feature ordering for Physics Attention ---
SENSOR_PRIORITY = ['PS', 'FS', 'TS', 'VS', 'CE', 'CP', 'SE', 'EPS']
def get_sorted_sensors(sensor_config):
    return sorted(sensor_config, key=lambda x: SENSOR_PRIORITY.index(x['name'][:2]))

# Hypermodel Definition
class PEECOMHyperModel(HyperModel):
    def __init__(
        self,
        input_shape,
        sensor_config=None,
        y_train=None,
        feature_mapping=None,
        use_baseline=False,
        lstm_dropout=0.0,
        lstm_recurrent_dropout=0.0,
        kernel_regularizer=None,
        dense_dropout=0.0
    ):
        super().__init__()
        self.input_shape = input_shape
        self.sensor_config = sensor_config
        self.y_train = y_train
        self.feature_mapping = feature_mapping or {'pressure': 0, 'flow': 1, 'temperature': 2}
        self.use_baseline = use_baseline
        self.lstm_dropout = lstm_dropout
        self.lstm_recurrent_dropout = lstm_recurrent_dropout
        self.kernel_regularizer = kernel_regularizer
        self.dense_dropout = dense_dropout

    def build(self, hp):
        if self.use_baseline:
            return self._build_baseline(hp)
        return self._build_hybrid(hp)

    def _build_baseline(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.LSTM(
            hp.Int('lstm_units', 64, 128, step=32),
            return_sequences=False,
            dropout=self.lstm_dropout,
            recurrent_dropout=self.lstm_recurrent_dropout,
            kernel_regularizer=self.kernel_regularizer
        )(inputs)
        if self.dense_dropout > 0:
            x = tf.keras.layers.Dropout(self.dense_dropout)(x)
        anomaly = tf.keras.layers.Dense(
            1, activation='sigmoid', name='anomaly_prob',
            kernel_regularizer=self.kernel_regularizer
        )(x)
        control = tf.keras.layers.Dense(
            2, name='control_params',
            kernel_regularizer=self.kernel_regularizer
        )(x)
        model = tf.keras.Model(inputs, {'anomaly_prob': anomaly, 'control_params': control})
        loss_weights = self._get_loss_weights()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Float('learning_rate', 1e-5, 1e-3, sampling='log')
            ),
            loss={
                'anomaly_prob': tf.keras.losses.BinaryFocalCrossentropy(
                    gamma=3, label_smoothing=0.1
                ),
                'control_params': tf.keras.losses.Huber()
            },
            loss_weights=loss_weights,
            metrics={
                'anomaly_prob': ['accuracy', tf.keras.metrics.AUC(name='auc')],
                'control_params': []
            }
        )
        return model

    def _build_hybrid(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv1D(
            hp.Int('conv_filters', 8, 16, step=8),
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=self.kernel_regularizer
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(
            hp.Int('lstm_units', 32, 64, step=32),
            return_sequences=True,
            dropout=self.lstm_dropout,
            recurrent_dropout=self.lstm_recurrent_dropout,
            kernel_regularizer=self.kernel_regularizer
        )(x)
        x = HybridPhysicsAttention(
            hp.Int('attn_units', 16, 32, step=16),
            pressure_idx=self.feature_mapping['pressure'],
            flow_idx=self.feature_mapping['flow'],
            temp_idx=self.feature_mapping['temperature']
        )(x)
        if self.dense_dropout > 0:
            x = tf.keras.layers.Dropout(self.dense_dropout)(x)
        else:
            x = tf.keras.layers.Dropout(hp.Float('dropout_rate', 0.1, 0.15, step=0.05))(x)
        anomaly = tf.keras.layers.TimeDistributed(
            CalibratedAnomalyHead(temp=hp.Float('temp', 0.4, 0.6)),
            name='anomaly_prob'
        )(x)
        control = tf.keras.layers.Dense(
            2, name='control_params',
            kernel_regularizer=self.kernel_regularizer
        )(x)
        model = tf.keras.Model(inputs, {'anomaly_prob': anomaly, 'control_params': control})
        model.add_loss(0.1 * self._physics_constraint_loss(anomaly, control))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Float('learning_rate', 1e-5, 1e-3, sampling='log'),
                clipvalue=hp.Float('grad_clip', 0.5, 1.0)
            ),
            loss={
                'anomaly_prob': tf.keras.losses.BinaryFocalCrossentropy(
                    gamma=3, label_smoothing=0.1
                ),
                'control_params': tf.keras.losses.Huber()
            },
            loss_weights=self._get_loss_weights(),
            metrics={
                'anomaly_prob': ['accuracy', tf.keras.metrics.AUC(name='auc')],
                'control_params': []
            }
        )
        return model

    def _physics_constraint_loss(self, anomaly, control):
        """Domain-specific physical consistency regularization"""
        energy_in = control[..., 0]
        energy_out = control[..., 1] + tf.reduce_mean(anomaly, axis=-1)
        return tf.reduce_mean(tf.square(energy_in - energy_out))

    def _get_loss_weights(self):
        # Increase anomaly loss weight for rare anomalies
        weights = {'anomaly_prob': 0.9, 'control_params': 0.1}
        if self.y_train is not None:
            anomaly_rate = np.mean(self.y_train)
            if anomaly_rate < 0.1:
                weights['anomaly_prob'] = 0.95
                weights['control_params'] = 0.05
        return weights

    def _build_and_compile(self, hp):
        """Build and compile model with caching"""
        if self._cached_model is None:
            self._cached_model = self.build(hp)
        return self._cached_model

    @tf.function(reduce_retracing=True)
    def predict(self, x):
        """Traced prediction function"""
        return self._cached_model(x, training=False)

# --- Validation Data Check Utility ---
def print_anomaly_rates(y_train, y_val):
    print(f"Train anomaly rate: {np.mean(y_train):.2f}")
    print(f"Val anomaly rate: {np.mean(y_val):.2f}")

def check_temporal_leak(X_train, X_val):
    if hasattr(X_train, "index") and hasattr(X_val, "index"):
        assert X_train.index.max() < X_val.index.min(), "Temporal leak detected!"

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, Bidirectional, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model

def attention_block(inputs):
    # Simplified attention block
    x = Dense(inputs.shape[-1], activation='tanh')(inputs)
    x = Dense(inputs.shape[-1], activation='sigmoid')(x)
    return inputs * x

def build_cnn_bilstm_model(timesteps, n_features, stats_dim=0, hp=None):
    seq_input = Input(shape=(timesteps, n_features), name="seq_input")
    
    # Add uncertainty input if available (last feature)
    uncertainty_input = Input(shape=(timesteps, 1), name="uncertainty_input")
    
    # Initial convolution path
    x = Conv1D(32, 3, activation='relu', padding='same')(seq_input)
    x = BatchNormalization()(x)
    
    # Combine with uncertainty
    x = Concatenate()([x, uncertainty_input])
    
    # Rest of the model
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = attention_block(x)
    x = GlobalAveragePooling1D()(x)
    
    if stats_dim > 0:
        stats_input = Input(shape=(stats_dim,), name="stats_input")
        x = Concatenate()([x, stats_input])
        inputs = [seq_input, uncertainty_input, stats_input]
    else:
        inputs = [seq_input, uncertainty_input]
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    anomaly_output = Dense(1, activation='sigmoid', name="anomaly_prob")(x)
    control_output = Dense(2, name="control_params")(x)
    
    model = Model(inputs=inputs, outputs=[anomaly_output, control_output])
    
    return model
