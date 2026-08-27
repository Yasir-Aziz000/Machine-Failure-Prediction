import os
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class Autoencoder(nn.Module):

    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Get project paths
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_models():

    rf_model = joblib.load(
        os.path.join(MODEL_DIR, "random_forest.pkl")
    )

    ae_scaler = joblib.load(
        os.path.join(MODEL_DIR, "autoencoder_scaler.pkl")
    )

    input_dim = len(ae_scaler.feature_names_in_) if hasattr(
        ae_scaler,
        "feature_names_in_"
    ) else ae_scaler.n_features_in_

    autoencoder = Autoencoder(input_dim)

    autoencoder.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "autoencoder.pth"),
            map_location="cpu"
        )
    )

    autoencoder.eval()

    return rf_model, autoencoder, ae_scaler


def create_input_dataframe(
    air_temp,
    process_temp,
    rotational_speed,
    torque,
    tool_wear,
    product_type,
    feature_columns
):

    data = pd.DataFrame(
        [[
            air_temp,
            process_temp,
            rotational_speed,
            torque,
            tool_wear
        ]],
        columns=[
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    )

    # Product type encoding
    data["Type_L"] = 0
    data["Type_M"] = 0

    if product_type == "L":
        data["Type_L"] = 1

    elif product_type == "M":
        data["Type_M"] = 1

    # Ensure the same feature order
    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return data


def predict_failure(
    rf_model,
    input_data
):

    probability = rf_model.predict_proba(
        input_data
    )[0][1]

    prediction = int(
        probability >= 0.5
    )

    return prediction, probability


def calculate_anomaly(
    autoencoder,
    ae_scaler,
    input_data
):

    scaled_data = ae_scaler.transform(
        input_data
    )

    input_tensor = torch.tensor(
        scaled_data,
        dtype=torch.float32
    )

    with torch.no_grad():

        reconstructed = autoencoder(
            input_tensor
        )

        reconstruction_error = torch.mean(
            (input_tensor - reconstructed) ** 2,
            dim=1
        ).item()

    return reconstruction_error
def load_threshold():

    threshold = joblib.load(
        os.path.join(
            MODEL_DIR,
            "anomaly_threshold.pkl"
        )
    )

    return threshold