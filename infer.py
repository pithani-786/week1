
import pandas as pd
import joblib
import os

here = os.path.dirname(__file__)
artifacts_dir = os.path.join(os.path.dirname(here), "artifacts")
model = joblib.load(os.path.join(artifacts_dir, "rf_inflow_model.joblib"))
features = joblib.load(os.path.join(artifacts_dir, "features.joblib"))

# Example new data row (replace with your daily inputs)
sample = pd.DataFrame([{
    "rainfall_mm": 12.5,
    "temperature_c": 30.1,
    "humidity_pct": 72.0,
    "evap_mm": 3.2,
    "upstream_flow_cumecs": 65.0,
    "storage_prev_mcm": 180.0,
    "dayofyear": 225,
    "month": 8
}])

y_hat = model.predict(sample[features])[0]
print(f"Predicted inflow (m^3/s): {y_hat:.2f}")
