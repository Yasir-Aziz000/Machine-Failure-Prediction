from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path
from src.monitoring import log_prediction

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from app.utils import (
    load_models,
    load_threshold,
    create_input_dataframe,
    predict_failure,
    calculate_anomaly
)


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="Machine Health AI API",
    description=(
        "Predict machine failure probability and "
        "detect anomalies using Machine Learning "
        "and Deep Learning."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# INPUT SCHEMA
# --------------------------------------------------

class MachineInput(BaseModel):

    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int
    product_type: str


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

rf_model, autoencoder, ae_scaler = load_models()

anomaly_threshold = load_threshold()


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Machine Health AI API is running",
        "status": "healthy"
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_machine_failure(data: MachineInput):

    # Get model feature names
    feature_columns = list(
        rf_model.feature_names_in_
    )

    # Create model input
    input_data = create_input_dataframe(
        data.air_temperature,
        data.process_temperature,
        data.rotational_speed,
        data.torque,
        data.tool_wear,
        data.product_type,
        feature_columns
    )

    # Machine Learning prediction
    prediction, probability = predict_failure(
        rf_model,
        input_data
    )

    # Deep Learning anomaly detection
    anomaly_error = calculate_anomaly(
        autoencoder,
        ae_scaler,
        input_data
    )

    is_anomaly = (
        anomaly_error > anomaly_threshold
    )
    # Log prediction for monitoring
    log_prediction(
        prediction=prediction,
        probability=probability,
        anomaly_score=anomaly_error
    )
    # Risk level
    probability_percent = probability * 100

    if probability_percent < 30:
        risk_level = "LOW"

    elif probability_percent < 70:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    # Response
    return {

        "prediction": int(prediction),

        "failure_probability": round(
            float(probability_percent),
            2
        ),

        "risk_level": risk_level,

        "anomaly_detected": bool(
            is_anomaly
        ),

        "anomaly_score": round(
            float(anomaly_error),
            6
        ),

        "anomaly_threshold": round(
            float(anomaly_threshold),
            6
        )
    }