
from pathlib import Path
from typing import Literal
import json
import logging

import joblib
import numpy as np
import pandas as pd
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.lstm_model import RUL_LSTM


# --------------------------------------------------
# Configuration and logging
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "alert_lstm_state.pt"
SCALER_PATH = ARTIFACTS_DIR / "alert_sensor_scaler.joblib"
METADATA_PATH = ARTIFACTS_DIR / "alert_model_metadata.json"

DEVICE = torch.device("cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("rul_api")

app = FastAPI(
    title="Predictive Maintenance Analyzer API",
    description=(
        "Predict capped remaining useful life from the latest "
        "30 consecutive sensor observations."
    ),
    version="0.1.0",
)


# --------------------------------------------------
# Load model artifacts once when the API starts
# --------------------------------------------------

for path in [MODEL_PATH, SCALER_PATH, METADATA_PATH]:
    if not path.exists():
        raise RuntimeError(
            f"Required model artifact is missing: {path}"
        )

with METADATA_PATH.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

SENSORS = metadata["selected_sensors"]
WINDOW_SIZE = int(metadata["window_size"])
RUL_CAP = int(metadata["rul_cap"])

TARGET_MEAN = float(metadata["target_mean"])
TARGET_STD = float(metadata["target_std"])
INTERVAL_CORRECTION = float(
    metadata["interval_correction"]
)

HIGH_THRESHOLD = float(metadata["high_rul_threshold"])
MEDIUM_THRESHOLD = float(metadata["medium_rul_threshold"])

if len(SENSORS) != 6 or WINDOW_SIZE != 30:
    raise RuntimeError(
        "Unexpected model input configuration in metadata."
    )

if TARGET_STD <= 0 or INTERVAL_CORRECTION < 0:
    raise RuntimeError(
        "Invalid target scaling or interval correction."
    )

sensor_scaler = joblib.load(SCALER_PATH)

model = RUL_LSTM(
    input_size=int(metadata["input_size"]),
    hidden_size=int(metadata["hidden_size"]),
    num_layers=int(metadata["num_layers"]),
).to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()

logger.info(
    "Model loaded: sensors=%s, window=%s, device=%s",
    SENSORS,
    WINDOW_SIZE,
    DEVICE,
)


# --------------------------------------------------
# Request and response schemas
# --------------------------------------------------

class SensorReading(BaseModel):
    cycle: int = Field(..., ge=1)

    sensor_2: float = Field(..., allow_inf_nan=False)
    sensor_4: float = Field(..., allow_inf_nan=False)
    sensor_7: float = Field(..., allow_inf_nan=False)
    sensor_9: float = Field(..., allow_inf_nan=False)
    sensor_14: float = Field(..., allow_inf_nan=False)
    sensor_20: float = Field(..., allow_inf_nan=False)


class PredictionRequest(BaseModel):
    engine_id: int | None = Field(default=None, ge=1)
    readings: list[SensorReading]


class PredictionResponse(BaseModel):
    engine_id: int | None
    last_cycle: int
    predicted_rul_cycles: float
    interval_lower_cycles: float
    interval_upper_cycles: float
    risk_tier: Literal["HIGH", "MEDIUM", "LOW"]
    model_name: str
    target_cap_cycles: int
    interval_note: str


# --------------------------------------------------
# Risk classification
# --------------------------------------------------

def assign_risk_tier(
    predicted_rul: float,
    lower_bound: float,
) -> Literal["HIGH", "MEDIUM", "LOW"]:
    if (
        predicted_rul < HIGH_THRESHOLD
        or lower_bound < HIGH_THRESHOLD
    ):
        return "HIGH"

    if (
        predicted_rul < MEDIUM_THRESHOLD
        or lower_bound < MEDIUM_THRESHOLD
    ):
        return "MEDIUM"

    return "LOW"


# --------------------------------------------------
# Basic endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Predictive Maintenance Analyzer API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(DEVICE),
    }


# --------------------------------------------------
# POST /predict
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_rul(
    request: PredictionRequest,
) -> PredictionResponse:

    # Require exactly 30 observations.
    if len(request.readings) != WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Exactly {WINDOW_SIZE} sensor readings are required; "
                f"received {len(request.readings)}."
            ),
        )

    cycles = [
        reading.cycle
        for reading in request.readings
    ]

    # Input must be oldest to newest, without skipped cycles.
    expected_cycles = list(
        range(cycles[0], cycles[0] + WINDOW_SIZE)
    )

    if cycles != expected_cycles:
        raise HTTPException(
            status_code=422,
            detail=(
                "Readings must be ordered oldest to newest "
                "with consecutive cycle numbers."
            ),
        )

    try:
        # Keep the same sensor column order used during training.
        sensor_frame = pd.DataFrame(
            [
                {
                    sensor: getattr(reading, sensor)
                    for sensor in SENSORS
                }
                for reading in request.readings
            ],
            columns=SENSORS,
        )

        raw_values = sensor_frame.to_numpy(dtype=np.float64)

        if not np.isfinite(raw_values).all():
            raise HTTPException(
                status_code=422,
                detail="Sensor readings must be finite numbers.",
            )

        # Apply the fitted training scaler; never refit per request.
        scaled_values = sensor_scaler.transform(
            sensor_frame
        ).astype(np.float32)

        # LSTM input shape: (batch=1, time=30, sensors=6).
        input_tensor = torch.from_numpy(
            scaled_values[np.newaxis, :, :]
        ).to(DEVICE)

        with torch.inference_mode():
            scaled_prediction = float(
                model(input_tensor).cpu().item()
            )

        # Convert standardized target back to RUL cycles.
        raw_prediction = (
            scaled_prediction * TARGET_STD + TARGET_MEAN
        )

        if not np.isfinite(raw_prediction):
            raise RuntimeError(
                "Model produced a non-finite prediction."
            )

        # Match the existing dashboard's capped-RUL display policy.
        predicted_rul = float(
            np.clip(raw_prediction, 0, RUL_CAP)
        )

        # Experimental interval from this model's calibration.
        lower_bound = float(
            np.clip(
                predicted_rul - INTERVAL_CORRECTION,
                0,
                RUL_CAP,
            )
        )

        upper_bound = float(
            np.clip(
                predicted_rul + INTERVAL_CORRECTION,
                0,
                RUL_CAP,
            )
        )

        risk_tier = assign_risk_tier(
            predicted_rul,
            lower_bound,
        )

        logger.info(
            "prediction_completed engine_id=%s last_cycle=%s "
            "predicted_rul=%.2f risk_tier=%s",
            request.engine_id,
            cycles[-1],
            predicted_rul,
            risk_tier,
        )

        return PredictionResponse(
            engine_id=request.engine_id,
            last_cycle=cycles[-1],
            predicted_rul_cycles=round(predicted_rul, 4),
            interval_lower_cycles=round(lower_bound, 4),
            interval_upper_cycles=round(upper_bound, 4),
            risk_tier=risk_tier,
            model_name="alerting_lstm",
            target_cap_cycles=RUL_CAP,
            interval_note=(
                "Experimental interval for capped RUL; "
                "not a guaranteed interval for physical engine life."
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "prediction_failed engine_id=%s",
            request.engine_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. Check the API server logs "
                "for diagnostic information."
            ),
        ) from exc
