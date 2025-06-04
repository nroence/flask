# flask-analytics/app/services/arima_helper.py

import pandas as pd
from .data_fetcher import fetch_daily_visit_counts

def build_daily_series(date_from, date_to, queue_id=None):
    """
    Returns a pandas.Series indexed by datetime.date,
    with one value (token count) for each calendar day between date_from and date_to.
    If queue_id is provided, only tokens for that queue are counted.
    Missing days in the range are filled with 0.
    """
    # 1) Fetch raw rows from tokens (optionally filtered by queue_id)
    raw = fetch_daily_visit_counts(date_from, date_to, queue_id)
    if not raw:
        # If no data in this range (or for this queue), return an empty Series
        return pd.Series(dtype="float64")

    # 2) Convert to DataFrame
    df = pd.DataFrame(raw)
    # Expected columns: ["date" (string), "count" (int)]

    # 3) Parse "date" column to datetime.date
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # 4) Set index to a proper DatetimeIndex
    df.set_index(pd.DatetimeIndex(df["date"]), inplace=True)

    # 5) Build the full date range and reindex with fill_value=0 for missing days
    full_index = pd.date_range(start=date_from, end=date_to, freq="D")
    series = df["count"].reindex(full_index, fill_value=0)

    series.index.name = "date"
    series.name = "visits"

    return series
