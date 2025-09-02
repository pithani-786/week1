
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Load data
here = os.path.dirname(__file__)
data_path = os.path.join(os.path.dirname(here), "data", "water_inflow_daily.csv")
df = pd.read_csv(data_path, parse_dates=["date"])

# Feature engineering
df["dayofyear"] = df["date"].dt.dayofyear
df["month"] = df["date"].dt.month

features = [
    "rainfall_mm","temperature_c","humidity_pct","evap_mm",
    "upstream_flow_cumecs","storage_prev_mcm","dayofyear","month"
]
target = "inflow_cumecs"

X = df[features]
y = df[target]

# Split train/test by time to avoid leakage
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.3f}")
print(f"Test MAE : {mae:.3f}")
print(f"Test R^2 : {r2:.3f}")

# Save model and features list
artifacts_dir = os.path.join(os.path.dirname(here), "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)
joblib.dump(model, os.path.join(artifacts_dir, "rf_inflow_model.joblib"))
joblib.dump(features, os.path.join(artifacts_dir, "features.joblib"))

# Plot: Actual vs Predicted (test period)
plt.figure()
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.title("Reservoir Inflow: Actual vs Predicted (Test Set)")
plt.xlabel("Day Index (Test Period)")
plt.ylabel("Inflow (m^3/s)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, "actual_vs_pred.png"), dpi=150)

# Plot: Feature importance
importances = model.feature_importances_
order = np.argsort(importances)[::-1]
plt.figure()
plt.bar(range(len(features)), importances[order])
plt.xticks(range(len(features)), np.array(features)[order], rotation=45, ha="right")
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, "feature_importance.png"), dpi=150)
print("Artifacts saved to:", artifacts_dir)
