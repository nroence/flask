# app/services/lstm_model.py

import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

from .arima_helper import build_daily_series

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _model_path(queue_id):
    suffix = f"queue{queue_id}" if queue_id else "all"
    # use native Keras format
    return os.path.join(os.getcwd(), "models", f"lstm_{suffix}.keras")

def _scaler_path(queue_id):
    suffix = f"queue{queue_id}" if queue_id else "all"
    return os.path.join(os.getcwd(), "models", f"lstm_scaler_{suffix}.pkl")

# ------------------------------------------------------------------
# Dataset creation
# ------------------------------------------------------------------
def create_dataset(series: pd.Series, window_size: int = 14):
    """
    Convert a daily count series into X/y numpy arrays using a sliding window.
    """
    data   = series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size : i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, window_size, 1)  # [samples, timesteps, features]
    y = np.array(y)

    return X, y, scaler

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
def train_lstm_model(date_from, date_to, queue_id=None,
                     window_size: int = 14, epochs: int = 50, batch_size: int = 16):
    """
    Train a fresh LSTM on [date_from .. date_to] for the given queue.
    Saves both model + scaler to disk.
    """
    # Build (and trim) series
    series = build_daily_series(date_from, date_to, queue_id)
    if series.empty:
        raise ValueError(f"No data in date range for queue {queue_id}")

    # Supervised dataset
    X, y, scaler = create_dataset(series, window_size)

    # Model architecture
    model = Sequential([
        LSTM(50, activation="tanh", input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    # Paths & dirs
    mdl_path = _model_path(queue_id)
    scl_path = _scaler_path(queue_id)
    os.makedirs(os.path.dirname(mdl_path), exist_ok=True)

    # Callbacks
    checkpoint = ModelCheckpoint(
        mdl_path,
        save_best_only=True,
        monitor="loss"
    )
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

    # Fit
    model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=0
    )

    # Persist scaler
    with open(scl_path, "wb") as fh:
        pickle.dump(scaler, fh)

    return model, scaler

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
def load_lstm_model(queue_id=None):
    mdl_path = _model_path(queue_id)
    scl_path = _scaler_path(queue_id)
    model  = load_model(mdl_path, compile=False)
    with open(scl_path, "rb") as fh:
        scaler = pickle.load(fh)
    return model, scaler

# ------------------------------------------------------------------
# Forecast
# ------------------------------------------------------------------
def forecast_lstm(date_from, date_to, queue_id=None,
                  steps: int = 14, window_size: int = 14, retrain: bool = False):
    """
    Train (or load) and produce `steps`-day forecast for the specified queue.
    Forecasts are always indexed from date_to + 1 day onward.
    """
    # 1) Build (and trim) the series
    series = build_daily_series(date_from, date_to, queue_id)
    if series.empty:
        raise ValueError(f"No data in date range for queue {queue_id}")

    # 2) Train or load
    mdl_exists = os.path.exists(_model_path(queue_id)) and os.path.exists(_scaler_path(queue_id))
    if retrain or not mdl_exists:
        model, scaler = train_lstm_model(date_from, date_to, queue_id, window_size)
    else:
        model, scaler = load_lstm_model(queue_id)

    # 3) Prepare last window for recursive forecasting
    scaled_full = scaler.transform(series.values.reshape(-1, 1))
    seq = scaled_full[-window_size:].reshape(1, window_size, 1)

    # 4) Recursive forecasting
    preds_scaled = []
    for _ in range(steps):
        next_scaled = float(model.predict(seq, verbose=0)[0, 0])
        preds_scaled.append(next_scaled)
        seq = np.append(seq.flatten()[1:], next_scaled).reshape(1, window_size, 1)

    # 5) Inverse‐scale predictions
    preds = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).flatten()

    # 6) Build a date index starting at date_to + 1 day
    start = pd.to_datetime(date_to) + pd.Timedelta(days=1)
    future_ix = pd.date_range(start=start, periods=steps, freq="D")

    return pd.Series(preds, index=future_ix, name="lstm_forecast")
