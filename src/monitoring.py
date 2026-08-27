import json
from pathlib import Path
from datetime import datetime


MONITORING_FILE = Path(
    "monitoring_metrics.json"
)


def log_prediction(
    prediction,
    probability,
    anomaly_score
):

    record = {
        "timestamp": datetime.now().isoformat(),

        "prediction": int(prediction),

        "failure_probability": float(
            probability
        ),

        "anomaly_score": float(
            anomaly_score
        )
    }

    if MONITORING_FILE.exists():

        try:

            with open(
                MONITORING_FILE,
                "r"
            ) as file:

                data = json.load(file)

        except json.JSONDecodeError:

            data = []

    else:

        data = []


    data.append(record)


    with open(
        MONITORING_FILE,
        "w"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )