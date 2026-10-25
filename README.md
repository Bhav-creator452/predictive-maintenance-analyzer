# Predictive Maintenance Analyzer

A machine-learning based predictive maintenance system for Remaining Useful Life (RUL) prediction.

## Running the Predictive Maintenance Analyzer

### Project Overview

The Predictive Maintenance Analyzer estimates Remaining Useful Life (RUL) from chronological sensor measurements using machine learning. It combines RUL prediction, experimental uncertainty estimation, risk-based alerts, sensor drift monitoring, a Streamlit dashboard, and a FastAPI prediction service.

This is a research prototype built using NASA C-MAPSS FD001 data. It has not been validated as a safety-critical industrial maintenance system.

### Architecture

- **Data preparation and modeling:** Python, Pandas, NumPy, Scikit-learn, XGBoost, and PyTorch.
- **Sequential model:** LSTM with a 30-cycle input window and six selected sensors.
- **API:** FastAPI, Pydantic, and Uvicorn.
- **Dashboard:** Streamlit and Matplotlib.
- **Drift monitoring:** Evidently.
- **Model artifacts:** Saved LSTM weights, sensor scaler, and model metadata.

The dashboard calls FastAPI for the selected engine's current assessment. Historical prediction charts and fleet summaries are loaded from exported CSV files.

### Requirements

- Python 3.13 (development environment)
- Virtual environment support
- Dependencies listed in `requirements.txt`
- Saved model artifacts and dashboard data files in `artifacts/`

### Local Setup

Create and activate a virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Ensure the required files are available in `artifacts/`:

- `alert_lstm_state.pt`
- `alert_sensor_scaler.joblib`
- `alert_model_metadata.json`
- `sensor_history.csv`
- `rul_predictions.csv`
- `fleet_summary.csv`

The API and dashboard require these artifacts to be available before startup.

### Start the API

From the project root:

```powershell
python -m uvicorn api:app --reload
```

API endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Basic health check |
| POST | `/predict` | Predict RUL from 30 consecutive sensor observations |

Interactive API documentation:

`http://127.0.0.1:8000/docs`

Each prediction request must include an engine ID and exactly 30 chronological sensor observations, each containing the six required sensor values.

### Start the Dashboard

Open a second terminal, activate the virtual environment, and run:

```powershell
python -m streamlit run app.py
```

Open the local URL reported by Streamlit, usually:

`http://localhost:8501`

The dashboard provides an engine selector, sensor trend charts, predicted RUL over time, experimental interval bounds, a risk tier, and a fleet-wide risk table.

### API Testing

Keep the API server running and execute these commands in another terminal:

```powershell
python test_api.py
python test_api_validation.py
```

The validation suite checks successful prediction requests, insufficient readings, nonconsecutive cycles, unknown engine IDs, missing sensor fields, and out-of-order observations.

The current validation suite passes all six test scenarios.

### Model Evaluation

The official FD001 test set contains 100 engines. The primary point-prediction results are:

| Model | RMSE (cycles) | MAE (cycles) | Asymmetric score |
|---|---:|---:|---:|
| Tuned XGBoost | 16.452 | 12.442 | 493.579 |
| LSTM | 16.425 | 12.596 | 423.567 |

For the critical subset with true RUL below 30 cycles:

| Model | RMSE (cycles) | MAE (cycles) |
|---|---:|---:|
| Tuned XGBoost | 6.596 | 5.345 |
| LSTM | 5.405 | 4.438 |

The LSTM has a lower overall RMSE and asymmetric score, while XGBoost has a slightly lower overall MAE. The LSTM also performed better on the evaluated critical low-RUL subset. These results support retaining both models for comparison rather than claiming that either is universally superior.

The training RUL targets were capped at 125 cycles, whereas the official test metrics above were evaluated against the original, uncapped test labels.

### Uncertainty Estimation and Risk Alerts

The system includes experimental prediction intervals and provisional HIGH, MEDIUM, and LOW risk tiers. The interval correction is based on a small calibration sample, and the intervals are not guaranteed bounds on actual physical remaining life.

The illustrative risk thresholds are intended for model development and retrospective analysis. They are not certified maintenance instructions.

### Sensor Drift Monitoring

The project includes Evidently reports comparing training and test sensor distributions and a controlled experiment that artificially shifted three sensor columns. The controlled experiment detected all three deliberately shifted sensors and left all three unchanged sensors unflagged.

- [Baseline drift report](reports/day17_baseline_drift_report.html)
- [Controlled drift report](reports/day17_controlled_drift_report_v2.html)

Distribution drift does not by itself prove that prediction accuracy has deteriorated. A drift alert should prompt investigation rather than automatic retraining.

### Limitations and Future Work

- The current application uses offline FD001 observations, not live industrial telemetry.
- RUL predictions are based on a capped target.
- Prediction intervals and risk thresholds need broader, engine-level validation.
- The API does not yet include authentication or rate limiting and should not be exposed publicly as-is.
- Industrial deployment would require representative equipment data, operational validation, monitoring, access controls, and engineering review.

The project demonstrates an end-to-end predictive-maintenance workflow, but it is not a safety-certified or production-validated industrial system.
