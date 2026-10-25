from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Analyzer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

SENSOR_FILE = ARTIFACTS_DIR / "sensor_history.csv"
PREDICTION_FILE = ARTIFACTS_DIR / "rul_predictions.csv"
FLEET_FILE = ARTIFACTS_DIR / "fleet_summary.csv"

SENSORS = [
    "sensor_2",
    "sensor_4",
    "sensor_7",
    "sensor_9",
    "sensor_14",
    "sensor_20",
]

RISK_PRIORITY = {
    "HIGH": 0,
    "MEDIUM": 1,
    "LOW": 2,
}


# --------------------------------------------------
# Load exported data
# --------------------------------------------------

@st.cache_data
def load_data():
    required_files = [
        SENSOR_FILE,
        PREDICTION_FILE,
        FLEET_FILE,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required dashboard files are missing:\n"
            + "\n".join(missing_files)
        )

    sensor_history = pd.read_csv(SENSOR_FILE)
    predictions = pd.read_csv(PREDICTION_FILE)
    fleet = pd.read_csv(FLEET_FILE)

    required_sensor_columns = {
        "engine_id", "cycle", *SENSORS
    }

    required_prediction_columns = {
        "engine_id",
        "cycle",
        "predicted_RUL",
        "lower_bound",
        "upper_bound",
        "risk_tier",
    }

    required_fleet_columns = {
        "engine_id",
        "cycle",
        "predicted_RUL",
        "lower_bound",
        "upper_bound",
        "risk_tier",
    }

    if not required_sensor_columns.issubset(
        sensor_history.columns
    ):
        raise ValueError(
            "sensor_history.csv is missing required columns."
        )

    if not required_prediction_columns.issubset(
        predictions.columns
    ):
        raise ValueError(
            "rul_predictions.csv is missing required columns."
        )

    if not required_fleet_columns.issubset(fleet.columns):
        raise ValueError(
            "fleet_summary.csv is missing required columns."
        )

    if sensor_history.empty or predictions.empty or fleet.empty:
        raise ValueError("One or more dashboard files are empty.")

    if predictions[
        ["predicted_RUL", "lower_bound", "upper_bound"]
    ].isna().any().any():
        raise ValueError(
            "Prediction data contains missing RUL values."
        )

    allowed_risks = {"HIGH", "MEDIUM", "LOW"}

    if not set(fleet["risk_tier"]).issubset(allowed_risks):
        raise ValueError("Unexpected risk tier in fleet summary.")

    return sensor_history, predictions, fleet


try:
    sensor_history, predictions, fleet = load_data()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Unable to load dashboard data: {exc}")
    st.info(
        "Check that the three CSV files exist in the "
        "project's artifacts/ folder."
    )
    st.stop()


# --------------------------------------------------
# Dashboard header and limitations
# --------------------------------------------------

st.title("⚙️ Predictive Maintenance Analyzer")

st.markdown(
    "Explore engine sensor trends, estimated remaining useful "
    "life (RUL), prediction intervals, and fleet-level risk."
)

st.warning(
    "Research prototype: these results use NASA C-MAPSS FD001 "
    "test data, a capped-RUL model, and a preliminary uncertainty "
    "correction. Predictions and intervals have not been "
    "validated for real industrial maintenance decisions."
)

st.caption(
    "Offline dataset demonstration — this dashboard does not "
    "currently ingest live equipment telemetry."
)


# --------------------------------------------------
# Normalize and sort table data
# --------------------------------------------------

sensor_history["engine_id"] = sensor_history[
    "engine_id"
].astype(int)

sensor_history["cycle"] = sensor_history[
    "cycle"
].astype(int)

predictions["engine_id"] = predictions[
    "engine_id"
].astype(int)

predictions["cycle"] = predictions[
    "cycle"
].astype(int)

fleet["engine_id"] = fleet["engine_id"].astype(int)

sensor_history = sensor_history.sort_values(
    ["engine_id", "cycle"]
)

predictions = predictions.sort_values(
    ["engine_id", "cycle"]
)

fleet["risk_priority"] = fleet["risk_tier"].map(
    RISK_PRIORITY
)

fleet = fleet.sort_values(
    ["risk_priority", "predicted_RUL"]
).reset_index(drop=True)


# --------------------------------------------------
# Fleet-level metrics
# --------------------------------------------------

risk_counts = (
    fleet["risk_tier"]
    .value_counts()
    .reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
)

st.subheader("Fleet Overview")

metric_columns = st.columns(4)

metric_columns[0].metric(
    "Total engines",
    f"{fleet['engine_id'].nunique():,}",
)

metric_columns[1].metric(
    "High risk",
    int(risk_counts["HIGH"]),
)

metric_columns[2].metric(
    "Medium risk",
    int(risk_counts["MEDIUM"]),
)

metric_columns[3].metric(
    "Low risk",
    int(risk_counts["LOW"]),
)


# --------------------------------------------------
# Sidebar: engine selection and risk filtering
# --------------------------------------------------

with st.sidebar:
    st.header("Dashboard Controls")

    engine_ids = sorted(
        sensor_history["engine_id"].unique().tolist()
    )

    selected_engine = st.selectbox(
        "Select an engine",
        engine_ids,
        index=0,
    )

    st.divider()

    selected_risks = st.multiselect(
        "Fleet risk tiers",
        options=["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
    )

    st.caption(
        "Select an engine to inspect its sensor history, "
        "RUL estimates, and current risk classification."
    )


# --------------------------------------------------
# Selected engine's latest information
# --------------------------------------------------

engine_sensors = sensor_history[
    sensor_history["engine_id"] == selected_engine
].sort_values("cycle")

engine_predictions = predictions[
    predictions["engine_id"] == selected_engine
].sort_values("cycle")

engine_fleet_rows = fleet[
    fleet["engine_id"] == selected_engine
]

if engine_sensors.empty or engine_predictions.empty:
    st.error(
        f"No complete sensor or prediction history found "
        f"for Engine {selected_engine}."
    )
    st.stop()

if engine_fleet_rows.empty:
    st.error("Current fleet summary is unavailable for this engine.")
    st.stop()

latest = engine_fleet_rows.iloc[0]

predicted_rul = float(latest["predicted_RUL"])
lower_bound = float(latest["lower_bound"])
upper_bound = float(latest["upper_bound"])
risk_tier = str(latest["risk_tier"])

st.divider()
st.header(f"Engine {selected_engine} — Current Assessment")

summary_columns = st.columns(4)

summary_columns[0].metric(
    "Latest observed cycle",
    int(engine_sensors["cycle"].max()),
)

summary_columns[1].metric(
    "Predicted RUL",
    f"{predicted_rul:.1f} cycles",
)

summary_columns[2].metric(
    "Lower interval bound",
    f"{lower_bound:.1f} cycles",
)

summary_columns[3].metric(
    "Upper interval bound",
    f"{upper_bound:.1f} cycles",
)

if risk_tier == "HIGH":
    st.error(
        "**HIGH RISK** — Review this engine's condition "
        "and assess maintenance requirements."
    )
elif risk_tier == "MEDIUM":
    st.warning(
        "**MEDIUM RISK** — Review the engine and "
        "consider closer monitoring."
    )
else:
    st.success(
        "**LOW RISK** — Continue routine monitoring."
    )

st.caption(
    "Risk tiers use provisional demonstration thresholds. "
    "They are not safety-certified recommendations."
)


# --------------------------------------------------
# Sensor trends
# --------------------------------------------------

st.divider()
st.header("Sensor Trends")

st.write(
    "Sensor readings are shown in their original measurement "
    "scales across the selected engine's observed cycles."
)

fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(14, 10),
)

for ax, sensor in zip(axes.flat, SENSORS):
    ax.plot(
        engine_sensors["cycle"],
        engine_sensors[sensor],
        linewidth=1.5,
    )

    ax.set_title(sensor)
    ax.set_xlabel("Operating cycle")
    ax.set_ylabel("Sensor value")
    ax.grid(True, alpha=0.25)

fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)


# --------------------------------------------------
# RUL history and uncertainty interval
# --------------------------------------------------

st.divider()
st.header("Predicted RUL Over Time")

st.write(
    "Each historical estimate uses the most recent 30 observed "
    "cycles available at that prediction endpoint."
)

fig_rul, ax_rul = plt.subplots(figsize=(12, 5))

ax_rul.fill_between(
    engine_predictions["cycle"].to_numpy(),
    engine_predictions["lower_bound"].to_numpy(),
    engine_predictions["upper_bound"].to_numpy(),
    alpha=0.20,
    label="Experimental prediction interval",
)

ax_rul.plot(
    engine_predictions["cycle"],
    engine_predictions["predicted_RUL"],
    linewidth=2,
    label="Predicted RUL",
)

ax_rul.set_xlabel("Operating cycle")
ax_rul.set_ylabel("RUL (cycles)")
ax_rul.set_title(
    f"Engine {selected_engine} — RUL estimates"
)

ax_rul.grid(True, alpha=0.25)
ax_rul.legend()
fig_rul.tight_layout()

st.pyplot(fig_rul, use_container_width=True)
plt.close(fig_rul)

st.caption(
    "The shaded band is a preliminary interval for the capped-RUL "
    "target. Its coverage is not guaranteed for uncapped physical "
    "remaining life or real industrial equipment."
)


# --------------------------------------------------
# Fleet-wide urgency table
# --------------------------------------------------

st.divider()
st.header("Fleet Risk Table")

filtered_fleet = fleet[
    fleet["risk_tier"].isin(selected_risks)
].copy()

display_columns = [
    "engine_id",
    "cycle",
    "predicted_RUL",
    "lower_bound",
    "upper_bound",
    "risk_tier",
]

st.dataframe(
    filtered_fleet[display_columns].round(2),
    use_container_width=True,
    hide_index=True,
    height=450,
)

st.caption(
    "Fleet entries are ordered by risk priority, then predicted RUL."
)

st.divider()
st.caption(
    "Predictive Maintenance Analyzer | NASA C-MAPSS FD001 | "
    "Experimental research prototype"
)
