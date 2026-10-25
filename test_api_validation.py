
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "http://127.0.0.1:8000"
PROJECT_DIR = Path(__file__).resolve().parent
SENSORS = [
    "sensor_2",
    "sensor_4",
    "sensor_7",
    "sensor_9",
    "sensor_14",
    "sensor_20",
]

# Load registered offline sensor data.
sensor_history = pd.read_csv(
    PROJECT_DIR / "artifacts" / "sensor_history.csv"
)

engine_id = 19

engine_data = (
    sensor_history[
        sensor_history["engine_id"] == engine_id
    ]
    .sort_values("cycle")
    .tail(30)
)

assert len(engine_data) == 30, "Need 30 readings for the test."

readings = engine_data[
    ["cycle"] + SENSORS
].to_dict(orient="records")

valid_payload = {
    "engine_id": engine_id,
    "readings": readings,
}


def run_test(name, payload, expected_status):
    """Run one API request and report the result."""
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=15,
        )

        passed = response.status_code == expected_status

        print(
            f"{'PASS' if passed else 'FAIL'} | "
            f"{name} | "
            f"Expected: {expected_status} | "
            f"Got: {response.status_code}"
        )

        if not passed:
            print("Response:", response.text[:500])

        return passed

    except requests.RequestException as exc:
        print(f"ERROR | {name} | {exc}")
        return False


results = []

# 1. Valid request should succeed.
results.append(
    run_test(
        "Valid 30-cycle request",
        valid_payload,
        200,
    )
)

# 2. Only one reading: invalid sequence length.
short_payload = {
    **valid_payload,
    "readings": readings[:1],
}

results.append(
    run_test(
        "Insufficient readings",
        short_payload,
        422,
    )
)

# 3. Break the consecutive cycle sequence.
nonconsecutive = [
    dict(reading) for reading in readings
]
nonconsecutive[10]["cycle"] += 2

results.append(
    run_test(
        "Nonconsecutive cycles",
        {
            **valid_payload,
            "readings": nonconsecutive,
        },
        422,
    )
)

# 4. Engine ID not in the offline fleet registry.
results.append(
    run_test(
        "Unknown engine ID",
        {
            **valid_payload,
            "engine_id": 9999,
        },
        404,
    )
)

# 5. Remove a required sensor field.
missing_sensor = [
    dict(reading) for reading in readings
]
del missing_sensor[0]["sensor_4"]

results.append(
    run_test(
        "Missing required sensor",
        {
            **valid_payload,
            "readings": missing_sensor,
        },
        422,
    )
)

# 6. Swap observations to violate chronological order.
out_of_order = [
    dict(reading) for reading in readings
]
out_of_order[5], out_of_order[6] = (
    out_of_order[6],
    out_of_order[5],
)

results.append(
    run_test(
        "Out-of-order readings",
        {
            **valid_payload,
            "readings": out_of_order,
        },
        422,
    )
)

print("\nValidation test summary")
print(f"Passed: {sum(results)}/{len(results)}")
print(f"Failed: {len(results) - sum(results)}")

if not all(results):
    raise SystemExit("Some API validation tests failed.")
