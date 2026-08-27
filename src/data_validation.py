import pandas as pd


REQUIRED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure"
]


def validate_data(df):

    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    # --------------------------------------
    # 1. CHECK REQUIRED COLUMNS
    # --------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("✓ Required columns: PASSED")


    # --------------------------------------
    # 2. CHECK EMPTY DATASET
    # --------------------------------------

    if df.empty:

        raise ValueError(
            "Dataset is empty."
        )

    print("✓ Dataset is not empty: PASSED")


    # --------------------------------------
    # 3. CHECK MISSING VALUES
    # --------------------------------------

    missing_values = df.isnull().sum()

    total_missing = missing_values.sum()

    print(
        f"Missing values found: {total_missing}"
    )

    if total_missing > 0:

        print("\nMissing values by column:")

        print(
            missing_values[
                missing_values > 0
            ]
        )


    # --------------------------------------
    # 4. CHECK DUPLICATES
    # --------------------------------------

    duplicates = df.duplicated().sum()

    print(
        f"Duplicate rows: {duplicates}"
    )


    # --------------------------------------
    # 5. CHECK TARGET COLUMN
    # --------------------------------------

    target_values = set(
        df["Machine failure"].unique()
    )

    valid_target_values = {0, 1}

    if not target_values.issubset(
        valid_target_values
    ):

        raise ValueError(
            "Machine failure must contain only 0 or 1."
        )

    print("✓ Target values: PASSED")


    # --------------------------------------
    # 6. DATASET SUMMARY
    # --------------------------------------

    print("\nDATASET SUMMARY")

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n✓ DATA VALIDATION COMPLETED")

    return True


if __name__ == "__main__":

    df = pd.read_csv(
        "data/ai4i2020.csv"
    )

    validate_data(df)