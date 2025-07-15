# app/services/ensemble.py

import os
from datetime import datetime, timedelta

import pandas as pd

from .arima_helper import build_daily_series
from .arima_model  import (
    train_arima_model,
    save_arima_model,
    load_arima_model,
    forecast_arima,
)
from .lstm_model   import forecast_lstm


def _arima_path(queue_id):
    """Generate queue-specific ARIMA path."""
    suffix = f"queue{queue_id}" if queue_id else "all"
    return os.path.join("models", f"arima_{suffix}.pkl")


def run_ensemble(
    date_from: str | None = None,
    date_to:   str | None = None,
    *,
    queue_id:      int | None = None,
    model:         str = "ensemble",      # "arima" | "lstm" | "ensemble"
    steps:         int = 14,
    window_size:   int = 14,
    arima_order:   tuple[int, int, int] = (1, 1, 1),
    retrain:       bool = False,
):
    """
    Build ARIMA +/- LSTM forecasts and (optionally) their average.

    All key knobs (steps, window_size, arima_order) can be supplied by the UI.
    """

    # ── 0) Determine date range defaults ──────────────────────────────
    today = datetime.today().date()

    date_to   = pd.to_datetime(date_to).date() if date_to   else today
    date_from = pd.to_datetime(date_from).date() if date_from else date_to - timedelta(days=60)

    # ── 1) ARIMA: train or load ───────────────────────────────────────
    want_arima = model in {"arima", "ensemble"}
    if want_arima:
        arima_path = _arima_path(queue_id)
        if retrain or not os.path.exists(arima_path):
            fitted_arima = train_arima_model(date_from, date_to, queue_id, arima_order)
            os.makedirs(os.path.dirname(arima_path), exist_ok=True)
            save_arima_model(fitted_arima, arima_path)
        else:
            fitted_arima = load_arima_model(arima_path)

        arima_forecast = forecast_arima(fitted_arima, steps=steps)

    # ── 2) LSTM: train or load ────────────────────────────────────────
    want_lstm = model in {"lstm", "ensemble"}
    if want_lstm:
        lstm_forecast = forecast_lstm(
            date_from,
            date_to,
            queue_id=queue_id,
            steps=steps,
            window_size=window_size,
            retrain=retrain,
        )

    # ── 3) Historical mean over training window ───────────────────────
    hist_series = build_daily_series(date_from, date_to, queue_id)
    historical_mean = float(hist_series.mean()) if not hist_series.empty else 0.0

    # ── 4) If both forecasts requested, build ensemble (average) ──────
    if model == "ensemble":
        # DEBUG: inspect the two index objects
        print("ARIMA index:", arima_forecast.index)
        print("LSTM index:",  lstm_forecast.index)
    
        combined = pd.concat(
            [arima_forecast.rename("arima"), lstm_forecast.rename("lstm")],
            axis=1,
            join="inner",
        )
        print("Combined empty?", combined.empty)

    # ── 5) Assemble JSON-friendly response ────────────────────────────
    payload = {"historical_mean": historical_mean}

    if want_arima:
        payload["arima"] = {
            "dates":  [d.strftime("%Y-%m-%d") for d in arima_forecast.index],
            "values": list(map(int, arima_forecast.values)),
        }

    if want_lstm:
        payload["lstm"] = {
            "dates":  [d.strftime("%Y-%m-%d") for d in lstm_forecast.index],
            "values": list(map(int, lstm_forecast.values)),
        }

    if model == "ensemble":
        payload["ensemble"] = {
            "dates":  [d.strftime("%Y-%m-%d") for d in ensemble.index],
            "values": list(map(int, ensemble.values)),
        }

    return payload
