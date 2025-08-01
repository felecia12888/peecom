import tensorflow as tf
from tensorflow.keras import layers, models

class ValveConditionModel(tf.keras.Model):
    def __init__(self, input_shape=(60, 24), dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        # Simple feature extraction specific for valve condition
        self.conv1 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')
        self.lstm = layers.LSTM(128, return_sequences=True)
        self.pool = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.out = layers.Dense(1, name='valve_condition')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.lstm(x)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.out(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "dropout_rate": self.dropout.rate,
        })
        return config
