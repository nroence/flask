# app/services/ensemble.py

import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

import pandas as pd

from .arima_helper import build_daily_series
from .arima_model import (
    train_arima_model,
    save_arima_model,
    load_arima_model,
    forecast_arima,
)
from .lstm_model import forecast_lstm


def _arima_path(queue_id: Optional[int]) -> str:
    """Generate a queue specific ARIMA model path."""
    suffix = f"queue{queue_id}" if queue_id else "all"
    return os.path.join("models", f"arima_{suffix}.pkl")


def _to_date(value: Optional[str], default: datetime.date) -> datetime.date:
    if not value:
        return default
    return pd.to_datetime(value).date()


def _series_to_json(s: pd.Series) -> Dict[str, Any]:
    # Ensure DatetimeIndex with date only
    idx = pd.to_datetime(s.index).date
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "values": [int(round(float(v))) if pd.notna(v) else None for v in s.values],
    }


def run_ensemble(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    *,
    queue_id: Optional[int] = None,
    model: str = "ensemble",             # "arima" | "lstm" | "ensemble"
    steps: int = 14,
    window_size: int = 14,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    retrain: bool = False,
) -> Dict[str, Any]:
    """
    Build ARIMA and or LSTM forecasts, and optionally their point wise average.

    Returns a dict shaped like:
      {
        "historical_mean": <float>,
        "arima":    { "dates": [...], "values": [...] },    if requested
        "lstm":     { "dates": [...], "values": [...] },    if requested
        "ensemble": { "dates": [...], "values": [...] }     if model == "ensemble"
      }
    """

    # 0) Normalize and default dates and basic inputs
    today = datetime.today().date()
    dt_to = _to_date(date_to, today)
    dt_from = _to_date(date_from, dt_to - timedelta(days=60))

    steps = int(steps) if isinstance(steps, int) or str(steps).isdigit() else 14
    steps = max(1, steps)

    window_size = int(window_size) if isinstance(window_size, int) or str(window_size).isdigit() else 14
    window_size = max(1, window_size)

    model = (model or "ensemble").lower()
    want_arima = model in {"arima", "ensemble"}
    want_lstm = model in {"lstm", "ensemble"}

    arima_forecast = pd.Series(dtype="float64")
    lstm_forecast = pd.Series(dtype="float64")

    # 1) ARIMA forecast
    if want_arima:
        try:
            path = _arima_path(queue_id)
            if retrain or not os.path.exists(path):
                fitted = train_arima_model(dt_from, dt_to, queue_id, arima_order)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                save_arima_model(fitted, path)
            else:
                fitted = load_arima_model(path)
            arima_forecast = forecast_arima(fitted, steps=steps)
            # Ensure index is daily dates
            arima_forecast.index = pd.to_datetime(arima_forecast.index).normalize()
        except Exception:
            # Do not break the whole request if ARIMA fails
            arima_forecast = pd.Series(dtype="float64")

    # 2) LSTM forecast
    if want_lstm:
        try:
            lstm_forecast = forecast_lstm(
                dt_from,
                dt_to,
                queue_id=queue_id,
                steps=steps,
                window_size=window_size,
                retrain=retrain,
            )
            lstm_forecast.index = pd.to_datetime(lstm_forecast.index).normalize()
        except Exception:
            lstm_forecast = pd.Series(dtype="float64")

    # 3) Historical mean over the lookback window
    try:
        hist_series = build_daily_series(dt_from, dt_to, queue_id)
        historical_mean = float(hist_series.mean()) if not hist_series.empty else 0.0
    except Exception:
        historical_mean = 0.0

    # 4) Ensemble as point wise mean over aligned dates if requested
    ensemble_series = pd.Series(dtype="float64")
    if model == "ensemble":
        parts = []
        if not arima_forecast.empty:
            parts.append(arima_forecast.rename("arima"))
        if not lstm_forecast.empty:
            parts.append(lstm_forecast.rename("lstm"))
        if parts:
            combined = pd.concat(parts, axis=1, join="outer")
            ensemble_series = combined.mean(axis=1, skipna=True).rename("ensemble")
        # If only one model succeeded, use it as the ensemble so the UI still works
        elif want_arima or want_lstm:
            fallback = arima_forecast if not arima_forecast.empty else lstm_forecast
            ensemble_series = fallback.rename("ensemble")

    # 5) Build the JSON friendly payload
    payload: Dict[str, Any] = {"historical_mean": historical_mean}

    if want_arima and not arima_forecast.empty:
        payload["arima"] = _series_to_json(arima_forecast)

    if want_lstm and not lstm_forecast.empty:
        payload["lstm"] = _series_to_json(lstm_forecast)

    if model == "ensemble" and not ensemble_series.empty:
        payload["ensemble"] = _series_to_json(ensemble_series)

    # If the client asked for ensemble but nothing could be produced, still return a key
    if model == "ensemble" and "ensemble" not in payload:
        payload["ensemble"] = {"dates": [], "values": []}

    return payload
