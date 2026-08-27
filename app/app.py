import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.monitoring import log_prediction
from utils import (
    load_models,
    load_threshold,
    create_input_dataframe,
    predict_failure,
    calculate_anomaly
)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Machine Health AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
}

.card {
    padding: 1.2rem;
    border-radius: 14px;
    border: 1px solid rgba(128, 128, 128, 0.2);
    margin-bottom: 1rem;
}

.small-text {
    color: gray;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_ai_models():
    return load_models()


rf_model, autoencoder, ae_scaler = load_ai_models()
anomaly_threshold = load_threshold()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("⚙️ Machine Health AI")

st.markdown(
    "### Predictive Maintenance & Anomaly Detection System"
)

st.caption(
    "Random Forest • PyTorch Autoencoder • Real-time Machine Health Analysis"
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Sensor Configuration")

    product_type = st.selectbox(
        "Product Type",
        ["L", "M", "H"]
    )

    st.divider()

    st.caption(
        "Enter current machine sensor values "
        "and run the AI analysis."
    )


# --------------------------------------------------
# SENSOR INPUT
# --------------------------------------------------

st.subheader("Machine Sensor Input")

col1, col2, col3 = st.columns(3)

with col1:

    air_temp = st.number_input(
        "🌡️ Air Temperature (K)",
        value=298.1,
        format="%.2f"
    )

    process_temp = st.number_input(
        "🔥 Process Temperature (K)",
        value=308.6,
        format="%.2f"
    )


with col2:

    rotational_speed = st.number_input(
        "⚙️ Rotational Speed (RPM)",
        value=1551,
        step=1
    )

    torque = st.number_input(
        "🔩 Torque (Nm)",
        value=42.8,
        format="%.2f"
    )


with col3:

    tool_wear = st.number_input(
        "🛠️ Tool Wear (minutes)",
        value=0,
        step=1
    )

    st.markdown("####")
    analyze_button = st.button(
        "🔍 Analyze Machine",
        use_container_width=True,
        type="primary"
    )


st.divider()


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if analyze_button:

    feature_columns = list(
        rf_model.feature_names_in_
    )

    input_data = create_input_dataframe(
        air_temp,
        process_temp,
        rotational_speed,
        torque,
        tool_wear,
        product_type,
        feature_columns
    )

    # ML prediction
    prediction, probability = predict_failure(
        rf_model,
        input_data
    )

    # Deep learning anomaly detection
    anomaly_error = calculate_anomaly(
        autoencoder,
        ae_scaler,
        input_data
    )

    is_anomaly = (
        anomaly_error > anomaly_threshold
    )

    # --------------------------------------------------
    # MODEL MONITORING
# --------------------------------------------------

    log_prediction(
        prediction=prediction,
        probability=probability,
        anomaly_score=anomaly_error
    )
    # --------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------

    probability_percent = probability * 100

    if probability_percent < 30:
        risk_level = "LOW"
        risk_icon = "🟢"

    elif probability_percent < 70:
        risk_level = "MEDIUM"
        risk_icon = "🟠"

    else:
        risk_level = "HIGH"
        risk_icon = "🔴"


    # --------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------

    st.session_state.history.append({
        "Failure Probability (%)": round(
            probability_percent,
            2
        ),
        "Risk Level": risk_level,
        "Anomaly Score": round(
            anomaly_error,
            6
        ),
        "Anomaly": "YES" if is_anomaly else "NO"
    })


    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.subheader("AI Analysis Results")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Failure Probability",
            f"{probability_percent:.2f}%"
        )

    with metric2:
        st.metric(
            "Risk Level",
            f"{risk_icon} {risk_level}"
        )

    with metric3:
        anomaly_status = (
            "⚠️ ANOMALY"
            if is_anomaly
            else "✅ NORMAL"
        )

        st.metric(
            "Deep Learning Detection",
            anomaly_status
        )


    # --------------------------------------------------
    # GAUGE CHART
    # --------------------------------------------------

    st.subheader("Failure Risk Visualization")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_percent,
            title={
                "text": "Machine Failure Probability (%)"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "thickness": 0.35
                },
                "steps": [
                    {
                        "range": [0, 30]
                    },
                    {
                        "range": [30, 70]
                    },
                    {
                        "range": [70, 100]
                    }
                ],
                "threshold": {
                    "line": {
                        "width": 4
                    },
                    "thickness": 0.75,
                    "value": probability_percent
                }
            }
        )
    )

    gauge.update_layout(
        height=350
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )


    # --------------------------------------------------
    # AI RECOMMENDATION
    # --------------------------------------------------

    st.subheader("AI Recommendation")

    if risk_level == "HIGH" or is_anomaly:

        st.error(
            "⚠️ HIGH RISK DETECTED\n\n"
            "The AI system detected possible abnormal machine behavior. "
            "Immediate inspection and preventive maintenance are recommended."
        )

    elif risk_level == "MEDIUM":

        st.warning(
            "⚠️ MEDIUM RISK DETECTED\n\n"
            "Monitor the machine and schedule an inspection."
        )

    else:

        st.success(
            "✅ LOW RISK\n\n"
            "The machine is operating within the predicted normal range."
        )


    # --------------------------------------------------
    # ANOMALY DETAILS
    # --------------------------------------------------

    st.subheader("Autoencoder Anomaly Analysis")

    anomaly_col1, anomaly_col2 = st.columns(2)

    with anomaly_col1:

        st.metric(
            "Reconstruction Error",
            f"{anomaly_error:.6f}"
        )

    with anomaly_col2:

        st.metric(
            "Anomaly Threshold",
            f"{anomaly_threshold:.6f}"
        )


    # --------------------------------------------------
    # SENSOR DATA
    # --------------------------------------------------

    st.subheader("Processed Sensor Data")

    st.dataframe(
        input_data,
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

if len(st.session_state.history) > 0:

    st.divider()

    st.subheader("Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    st.line_chart(
        history_df["Failure Probability (%)"]
    )

# --------------------------------------------------
# MODEL MONITORING
# --------------------------------------------------

from pathlib import Path
import json


monitoring_file = Path(
    "monitoring_metrics.json"
)


if monitoring_file.exists():

    st.divider()

    st.subheader(
        "📊 Model Monitoring"
    )

    with open(
        monitoring_file,
        "r"
    ) as file:

        monitoring_data = json.load(
            file
        )


    monitoring_df = pd.DataFrame(
        monitoring_data
    )


    if len(monitoring_df) > 0:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Predictions",
                len(monitoring_df)
            )


        with col2:

            average_probability = (
                monitoring_df[
                    "failure_probability"
                ].mean() * 100
            )

            st.metric(
                "Average Failure Risk",
                f"{average_probability:.2f}%"
            )


        with col3:

            anomaly_count = (
                monitoring_df[
                    "anomaly_score"
                ].mean()
            )

            st.metric(
                "Average Anomaly Score",
                f"{anomaly_count:.4f}"
            )


        st.subheader(
            "Failure Probability Over Time"
        )


        monitoring_chart = (
            monitoring_df
            .set_index("timestamp")
            ["failure_probability"]
        )


        st.line_chart(
            monitoring_chart
        )
# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Machine Health AI • "
    "Machine Learning + Deep Learning + MLOps"
)
