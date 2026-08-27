import pandas as pd


DROP_COLUMNS = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]


def preprocess_data(df):

    print("\nPreprocessing data...")

    df = df.copy()

    # --------------------------------------
    # Remove unnecessary columns
    # --------------------------------------

    df = df.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    # --------------------------------------
    # One-hot encode product type
    # --------------------------------------

    df = pd.get_dummies(
        df,
        columns=["Type"],
        drop_first=False
    )

    print(
        f"Processed dataset shape: {df.shape}"
    )

    return df