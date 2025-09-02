
# Water Resource Management — Reservoir Inflow Forecasting (AI/ML)

End-to-end project to predict next-day reservoir inflow (m³/s) from weather and hydrology features using a Random Forest model.

## Project Structure
```
water_ml_project/
├── data/
│   └── water_inflow_daily.csv
├── src/
│   ├── train_model.py
│   └── infer.py
├── artifacts/                # created after training
├── requirements.txt
└── README.md
```

## Dataset
- **Rows:** ~730 (2 years of daily data)
- **Columns:** date, rainfall_mm, temperature_c, humidity_pct, evap_mm, upstream_flow_cumecs, storage_prev_mcm, inflow_cumecs
- Data is synthetic but realistic: seasonal rainfall, temperature cycles, humidity, evapotranspiration, upstream flow, and storage dynamics.

## How to Run
1. **Create a virtual environment (optional)**  
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**  
   ```bash
   python src/train_model.py
   ```
   - Saves model to `artifacts/rf_inflow_model.joblib`
   - Saves figures: `artifacts/actual_vs_pred.png`, `artifacts/feature_importance.png`
   - Prints RMSE/MAE/R²

4. **Run inference on new data**  
   Edit `src/infer.py` to pass your own daily inputs (or adapt it to read CSV). Then:
   ```bash
   python src/infer.py
   ```

## Customize / Extend
- Swap `RandomForestRegressor` with Gradient Boosting/XGBoost or LSTMs for sequence modeling.
- Add lagged features (e.g., rainfall_last_3d, inflow_last_7d) to capture memory.
- Use real datasets (e.g., IMD rainfall, CWC/USGS streamflows) and re-train.
- Convert this into an API (FastAPI/Flask) or a dashboard (Streamlit).

## License
MIT
