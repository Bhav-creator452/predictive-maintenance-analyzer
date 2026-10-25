
from pathlib import Path
import json
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_DIR / "artifacts"

SENSORS = [
    "sensor_2",
    "sensor_4",
    "sensor_7",
    "sensor_9",
    "sensor_14",
    "sensor_20",
]

# Pick an engine that exists in the saved dashboard data.
engine_id = 19

sensor_history = pd.read_csv(
    ARTIFACTS / "sensor_history.csv"
)

engine_data = (
    sensor_history[
        sensor_history["engine_id"] == engine_id
    ]
    .sort_values("cycle")
    .tail(30)
)

if len(engine_data) != 30:
    raise ValueError(
        f"Engine {engine_id} does not have 30 available readings."
    )

payload = {
    "engine_id": engine_id,
    "readings": engine_data[
        ["cycle"] + SENSORS
    ].to_dict(orient="records"),
}

request = Request(
    "http://127.0.0.1:8000/predict",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
        print("HTTP status:", response.status)
        print(json.dumps(result, indent=2))

except Exception as exc:
    print("API request failed:", exc)
    raise
