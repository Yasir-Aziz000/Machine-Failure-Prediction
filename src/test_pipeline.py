import pandas as pd

from data_validation import validate_data
from preprocessing import preprocess_data


# Load raw data
df = pd.read_csv(
    "data/ai4i2020.csv"
)


# Validate raw data
validate_data(df)


# Preprocess data
processed_df = preprocess_data(df)


print("\nFINAL DATA")

print(
    processed_df.head()
)

print(
    "\nFinal shape:",
    processed_df.shape
)