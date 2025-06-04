# scripts/train_arima.py

import os
import joblib
import pandas as pd
from datetime import timedelta, datetime
from statsmodels.tsa.arima.model import ARIMA
from flask_analytics.app.services.data_fetcher import fetch_patient_visit_counts

# 1. Fetch the last 180 days
today     = datetime.today().strftime("%Y-%m-%d")
start_180 = (datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d")
records   = fetch_patient_visit_counts(start_180, today)

# 2. Build a daily‐indexed Series (fill gaps with 0)
df = (
    pd.DataFrame(records)
      .set_index("date")["count"]
      .pipe(lambda s: s.asfreq("D", fill_value=0))
)

# 3. Fit ARIMA—tweak (p,d,q) and seasonal (P,D,Q,s) as needed
model = ARIMA(df, order=(2,1,2), seasonal_order=(1,1,1,7))
res   = model.fit()

# 4. Persist to disk
os.makedirs("models", exist_ok=True)
joblib.dump(res, "models/arima.pkl")

print("ARIMA model trained & saved to models/arima.pkl")
