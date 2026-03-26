"""
WeatherX — model.py
═══════════════════════════════════════════════════════════════════════════════
Fix applied: weighted MSE loss so precipitation (mostly 0) cannot dominate
and cause temperature/humidity predictions to collapse to near-zero.

Target column order MUST match preprocess.py TARGET_COLS:
  index 0 → precipitation
  index 1 → temperature_2m
  index 2 → relative_humidity_2m
═══════════════════════════════════════════════════════════════════════════════
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape


def weighted_mse(y_true, y_pred):
    """
    Custom loss: penalises temperature and humidity errors more heavily
    than precipitation errors.

    Weights are applied per target column (must match TARGET_COLS order [Precip, Temp, Humid]):
      index 0 — precipitation        weight 1.0  (sparse signal, mostly 0)
      index 1 — temperature_2m       weight 5.0  (primary forecast variable)
      index 2 — relative_humidity_2m weight 3.0  (secondary forecast variable)

    Without this, the model learns to predict ~0 for everything because
    near-zero precipitation MSE dominates the combined loss and masks
    large errors on temperature and humidity.
    """
    weights = tf.constant([1.0, 5.0, 3.0], dtype=tf.float32)  # shape (3,)
    squared_errors = tf.square(y_true - y_pred)                # (batch, horizon, 3)
    weighted_errors = squared_errors * weights                  # broadcast over (batch, horizon)
    return tf.reduce_mean(weighted_errors)


def build_weatherx_lstm(seq_len, num_features, horizon, num_targets):
    """
    Builds the WeatherX LSTM for multi-step analog forecasting.

    Args:
        seq_len     : Look-back window in hours (e.g. 24)
        num_features: Input features per timestep
        horizon     : Forecast steps ahead (e.g. 6)
        num_targets : Variables predicted (3: precip, temp, humidity)

    Returns:
        Compiled tf.keras.Model
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, num_features)),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(horizon * num_targets),          # linear activation — no clipping
        Reshape((horizon, num_targets))
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=weighted_mse,                     # ← replaces plain 'mse'
        metrics=['mae']
    )

    return model