# flask-analytics/app/services/arima_model.py

import os
import pickle

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from pandas.tseries.frequencies import to_offset
from typing import Optional
from .arima_helper import build_daily_series

# Default path (for “all queues”) if you ever need a fallback
MODEL_PATH = os.path.join(os.getcwd(), "models", "arima.pkl")


# ───────────────────────────────────────────────────────────────────
# TRAIN
# ───────────────────────────────────────────────────────────────────
def train_arima_model(date_from, date_to, queue_id=None, order=(1, 1, 1)):
    """
    • Fetch daily token counts (optionally filtered by queue_id)
    • Fit an ARIMA(p, d, q) on that daily series
    • Return the fitted statsmodels ARIMAResults object
    """
    # Build a pandas.Series of daily counts, filling missing dates with 0
    series = build_daily_series(date_from, date_to, queue_id)
    if series.empty:
        raise ValueError(f"No data in date range for queue {queue_id}")

    # Fit the ARIMA model
    model = ARIMA(series, order=order)
    fitted = model.fit()

    # Store the training index on the fitted object so forecast_arima can find it
    fitted._train_index = series.index
    return fitted


def save_arima_model(model, path=MODEL_PATH):
    """
    Serialize the fitted ARIMAResults to disk at the given path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)


def load_arima_model(path=MODEL_PATH):
    """
    Load a previously saved ARIMAResults from disk.
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ───────────────────────────────────────────────────────────────────
# FORECAST
# ───────────────────────────────────────────────────────────────────
def forecast_arima(
    fitted: ARIMAResults,
    steps: int = 14,
    date_to: Optional[str] = None
) -> pd.Series:
    """
    Forecast 'steps' periods ahead and return a pandas.Series whose index
    is a proper DatetimeIndex for those future dates. If `date_to` is given,
    the first forecast date will be the day after `date_to`; otherwise the
    first forecast date comes one period after the end of the training index.

    Parameters
    ----------
    fitted : ARIMAResults
        A fitted statsmodels ARIMAResults object, with a `_train_index`
        attribute or accessible `model.data.dates`.
    steps : int, default=14
        Number of days (periods) to forecast.
    date_to : str or None
        If provided, a string "YYYY-MM-DD" indicating the last observed date.
        Forecasts will start on date_to + 1 day. If None, falls back to
        fitted._train_index or fitted.model.data.dates.

    Returns
    -------
    pd.Series
        A Series of length `steps`, indexed by the future dates.
    """
    # 1) Determine the "last date" and frequency
    if date_to is not None:
        last_date = pd.to_datetime(date_to)
        freq = getattr(fitted._train_index, "freq", to_offset("D"))
    elif hasattr(fitted, "_train_index"):
        last_date = fitted._train_index[-1]
        freq = fitted._train_index.freq or to_offset("D")
    elif fitted.model.data.dates is not None:
        last_date = fitted.model.data.dates[-1]
        freq = fitted.model.data.dates.freq or to_offset("D")
    else:
        raise RuntimeError("Cannot infer last date – please refit the model")

    # 2) Build a future date index starting one period after last_date
    offset = to_offset(freq) if not hasattr(freq, "delta") else freq
    future_index = pd.date_range(
        start=last_date + offset,
        periods=steps,
        freq=freq
    )

    # 3) Generate predictions
    preds = fitted.predict(
        start=future_index[0],
        end=future_index[-1],
        dynamic=False
    )

    # 4) Return a Series with that index
    return pd.Series(preds.values, index=future_index, name="arima_forecast")


# ───────────────────────────────────────────────────────────────────
# QUICK HELPER FOR DIRECT CALLS OR ENSEMBLE
# ───────────────────────────────────────────────────────────────────
def fit_and_forecast(date_from, date_to, queue_id=None, steps=14, order=(1, 1, 1), retrain=False):
    """
    Convenience function to either:
      - Train a new ARIMA model on [date_from, date_to], 
        optionally filtered by queue_id, then forecast
      - Or load an existing serialized model for that queue and forecast

    Returns a pandas.Series of length 'steps', indexed by future dates.
    """
    # Choose a file path that’s unique per queue (or "all" if None)
    suffix = f"queue{queue_id}" if queue_id else "all"
    model_path = os.path.join(os.getcwd(), "models", f"arima_{suffix}.pkl")

    # Train or load
    if retrain or not os.path.exists(model_path):
        fitted = train_arima_model(date_from, date_to, queue_id, order)
        save_arima_model(fitted, model_path)
    else:
        fitted = load_arima_model(model_path)

    # Forecast next `steps` days
    return forecast_arima(fitted, steps)
