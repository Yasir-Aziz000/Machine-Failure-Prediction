# import os
# import pandas as pd
# import joblib
# import mlflow
# import mlflow.sklearn
# from data_validation import validate_data
# from preprocessing import preprocess_data

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score
# )


# # ==================================================
# # 1. CONFIGURATION
# # ==================================================

# EXPERIMENT_NAME = "Machine-Failure-Prediction"
# DATA_PATH = "data/ai4i2020.csv"
# MODEL_PATH = "models/best_random_forest.pkl"

# mlflow.set_experiment(EXPERIMENT_NAME)


# # ==================================================
# # 2. LOAD DATA
# # ==================================================

# print("Loading dataset...")

# # Load raw data
# df = pd.read_csv(DATA_PATH)

# # Validate raw data
# validate_data(df)

# # Preprocess data
# df = preprocess_data(df)
# print(f"Dataset shape: {df.shape}")


# # ==================================================
# # 3. PREPROCESSING
# # ==================================================






# # One-hot encode product type
# df = pd.get_dummies(
#     df,
#     columns=["Type"],
#     drop_first=False
# )


# # Define features and target
# target_column = "Machine failure"

# X = df.drop(columns=[target_column])
# y = df[target_column]


# # ==================================================
# # 4. TRAIN / TEST SPLIT
# # ==================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# print(f"Training samples: {len(X_train)}")
# print(f"Testing samples: {len(X_test)}")


# # ==================================================
# # 5. EXPERIMENT CONFIGURATIONS
# # ==================================================

# experiments = [
#     {
#         "n_estimators": 100,
#         "max_depth": 10
#     },
#     {
#         "n_estimators": 200,
#         "max_depth": 15
#     },
#     {
#         "n_estimators": 300,
#         "max_depth": 20
#     },
#     {
#         "n_estimators": 500,
#         "max_depth": None
#     }
# ]


# # ==================================================
# # 6. BEST MODEL TRACKING
# # ==================================================

# best_model = None
# best_f1_score = 0
# best_config = None


# # ==================================================
# # 7. RUN EXPERIMENTS
# # ==================================================

# for index, config in enumerate(experiments, start=1):

#     print("\n" + "=" * 60)
#     print(f"RUNNING EXPERIMENT {index}")
#     print(config)
#     print("=" * 60)


#     with mlflow.start_run(
#         run_name=f"RandomForest_Experiment_{index}"
#     ):

#         # ------------------------------------------
#         # CREATE MODEL
#         # ------------------------------------------

#         model = RandomForestClassifier(
#             n_estimators=config["n_estimators"],
#             max_depth=config["max_depth"],
#             random_state=42,
#             class_weight="balanced",
#             n_jobs=-1
#         )


#         # ------------------------------------------
#         # TRAIN
#         # ------------------------------------------

#         model.fit(
#             X_train,
#             y_train
#         )


#         # ------------------------------------------
#         # PREDICT
#         # ------------------------------------------

#         predictions = model.predict(
#             X_test
#         )

#         probabilities = model.predict_proba(
#             X_test
#         )[:, 1]


#         # ------------------------------------------
#         # CALCULATE METRICS
#         # ------------------------------------------

#         accuracy = accuracy_score(
#             y_test,
#             predictions
#         )

#         precision = precision_score(
#             y_test,
#             predictions,
#             zero_division=0
#         )

#         recall = recall_score(
#             y_test,
#             predictions,
#             zero_division=0
#         )

#         f1 = f1_score(
#             y_test,
#             predictions,
#             zero_division=0
#         )

#         roc_auc = roc_auc_score(
#             y_test,
#             probabilities
#         )


#         # ------------------------------------------
#         # LOG PARAMETERS
#         # ------------------------------------------

#         mlflow.log_params({
#             "n_estimators": config["n_estimators"],
#             "max_depth": (
#                 "None"
#                 if config["max_depth"] is None
#                 else config["max_depth"]
#             ),
#             "random_state": 42,
#             "class_weight": "balanced"
#         })


#         # ------------------------------------------
#         # LOG METRICS
#         # ------------------------------------------

#         mlflow.log_metrics({
#             "accuracy": accuracy,
#             "precision": precision,
#             "recall": recall,
#             "f1_score": f1,
#             "roc_auc": roc_auc
#         })


#         # ------------------------------------------
#         # LOG MODEL
#         # ------------------------------------------

#         mlflow.sklearn.log_model(
#             sk_model=model,
#             name="random_forest_model"
#         )


#         # ------------------------------------------
#         # PRINT RESULTS
#         # ------------------------------------------

#         print(f"Accuracy:  {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall:    {recall:.4f}")
#         print(f"F1 Score:  {f1:.4f}")
#         print(f"ROC-AUC:   {roc_auc:.4f}")


#         # ------------------------------------------
#         # BEST MODEL SELECTION
#         # ------------------------------------------

#         if f1 > best_f1_score:

#             best_f1_score = f1
#             best_model = model
#             best_config = config.copy()


# # ==================================================
# # 8. SAVE BEST MODEL
# # ==================================================

# os.makedirs(
#     "models",
#     exist_ok=True
# )

# joblib.dump(
#     best_model,
#     MODEL_PATH
# )


# # ==================================================
# # 9. FINAL RESULTS
# # ==================================================

# print("\n" + "=" * 60)
# print("EXPERIMENTS COMPLETED")
# print("=" * 60)

# print("\nBEST MODEL")

# print(f"F1 Score: {best_f1_score:.4f}")
# print(f"Configuration: {best_config}")

# print(f"\nModel saved to: {MODEL_PATH}")




import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from data_validation import validate_data
from preprocessing import preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

EXPERIMENT_NAME = "Machine-Failure-Prediction"
DATA_PATH = "data/ai4i2020.csv"
MODEL_PATH = "models/best_random_forest.pkl"

mlflow.set_experiment(EXPERIMENT_NAME)


# ==================================================
# 2. LOAD DATA
# ==================================================

print("Loading dataset...")

# Load raw data
df = pd.read_csv(DATA_PATH)

# Validate raw data
validate_data(df)

# Preprocess data
df = preprocess_data(df)
print(f"Dataset shape: {df.shape}")

# REMOVE the duplicate one-hot encoding section
# The preprocessing function already handles the "Type" column

# Define features and target
target_column = "Machine failure"

X = df.drop(columns=[target_column])
y = df[target_column]


# ==================================================
# 4. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ==================================================
# 5. EXPERIMENT CONFIGURATIONS
# ==================================================

experiments = [
    {
        "n_estimators": 100,
        "max_depth": 10
    },
    {
        "n_estimators": 200,
        "max_depth": 15
    },
    {
        "n_estimators": 300,
        "max_depth": 20
    },
    {
        "n_estimators": 500,
        "max_depth": None
    }
]


# ==================================================
# 6. BEST MODEL TRACKING
# ==================================================

best_model = None
best_f1_score = 0
best_config = None


# ==================================================
# 7. RUN EXPERIMENTS
# ==================================================

for index, config in enumerate(experiments, start=1):

    print("\n" + "=" * 60)
    print(f"RUNNING EXPERIMENT {index}")
    print(config)
    print("=" * 60)

    with mlflow.start_run(
        run_name=f"RandomForest_Experiment_{index}"
    ):

        # ------------------------------------------
        # CREATE MODEL
        # ------------------------------------------

        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )


        # ------------------------------------------
        # TRAIN
        # ------------------------------------------

        model.fit(
            X_train,
            y_train
        )


        # ------------------------------------------
        # PREDICT
        # ------------------------------------------

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]


        # ------------------------------------------
        # CALCULATE METRICS
        # ------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )


        # ------------------------------------------
        # LOG PARAMETERS
        # ------------------------------------------

        mlflow.log_params({
            "n_estimators": config["n_estimators"],
            "max_depth": (
                "None"
                if config["max_depth"] is None
                else config["max_depth"]
            ),
            "random_state": 42,
            "class_weight": "balanced"
        })


        # ------------------------------------------
        # LOG METRICS
        # ------------------------------------------

        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })


        # ------------------------------------------
        # LOG MODEL
        # ------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model"
        )


        # ------------------------------------------
        # PRINT RESULTS
        # ------------------------------------------

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")


        # ------------------------------------------
        # BEST MODEL SELECTION
        # ------------------------------------------

        if f1 > best_f1_score:

            best_f1_score = f1
            best_model = model
            best_config = config.copy()


# ==================================================
# 8. SAVE BEST MODEL
# ==================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    MODEL_PATH
)


# ==================================================
# 9. FINAL RESULTS
# ==================================================

print("\n" + "=" * 60)
print("EXPERIMENTS COMPLETED")
print("=" * 60)

print("\nBEST MODEL")

print(f"F1 Score: {best_f1_score:.4f}")
print(f"Configuration: {best_config}")

print(f"\nModel saved to: {MODEL_PATH}")