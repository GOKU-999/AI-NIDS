"""
AI-NIDS — Real Test Prediction
PHASE 10.5

Tests the prediction engine against the actual test dataset.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Allow importing predict.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from predict import predict_batch


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "selected_features",
    "X_test_selected.csv"
)

Y_BINARY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "split",
    "y_binary_test.csv"
)

Y_MULTICLASS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "split",
    "y_multiclass_test.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PHASE 10.5 — REAL TEST DATA PREDICTION")
print("=" * 70)


# ============================================================
# LOAD TEST FEATURES
# ============================================================

print("\nLoading test features...")

X_test = pd.read_csv(X_TEST_PATH)

print(f"Test shape: {X_test.shape}")


# ============================================================
# LOAD LABELS
# ============================================================

print("\nLoading binary labels...")

y_binary = pd.read_csv(Y_BINARY_PATH)

print("\nLoading multi-class labels...")

y_multiclass = pd.read_csv(Y_MULTICLASS_PATH)


# Convert to Series
y_binary = y_binary.iloc[:, 0]
y_multiclass = y_multiclass.iloc[:, 0]


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

print(f"X_test samples:       {len(X_test)}")
print(f"Binary labels:        {len(y_binary)}")
print(f"Multi-class labels:   {len(y_multiclass)}")
print(f"Features:             {X_test.shape[1]}")


if len(X_test) != len(y_binary):
    raise ValueError(
        "X_test and binary labels have different row counts."
    )


if len(X_test) != len(y_multiclass):
    raise ValueError(
        "X_test and multiclass labels have different row counts."
    )


print("\n✅ Dataset validation passed.")


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING PREDICTIONS")
print("=" * 70)

print("\nPredicting all test flows...")

start_time = time.time()

predictions = predict_batch(X_test)

elapsed_time = time.time() - start_time


print("\n✅ Predictions completed.")

print(
    f"Prediction time: {elapsed_time:.2f} seconds"
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

print("\nBinary predictions:")

print(
    predictions["binary_prediction"].value_counts()
)


print("\nAttack type predictions:")

print(
    predictions["attack_type"].value_counts()
)


# ============================================================
# COMPARE BINARY PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("BINARY PREDICTION VALIDATION")
print("=" * 70)


binary_prediction_numeric = (
    predictions["is_attack"]
    .astype(int)
    .to_numpy()
)

binary_actual = (
    y_binary
    .astype(int)
    .to_numpy()
)


binary_correct = (
    binary_prediction_numeric == binary_actual
).sum()


binary_accuracy = (
    binary_correct / len(binary_actual)
)


print(
    f"\nBinary accuracy from prediction engine: "
    f"{binary_accuracy:.6f}"
)

print(
    f"Binary correct predictions: "
    f"{binary_correct:,}"
)

print(
    f"Binary incorrect predictions: "
    f"{len(binary_actual) - binary_correct:,}"
)


# ============================================================
# MULTI-CLASS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("MULTI-CLASS PREDICTION VALIDATION")
print("=" * 70)


# Load label mapping
import json

LABEL_MAPPING_PATH = os.path.join(
    BASE_DIR,
    "models",
    "multiclass",
    "label_mapping.json"
)


with open(LABEL_MAPPING_PATH, "r") as f:
    label_mapping = json.load(f)


reverse_mapping = {
    int(k): v
    for k, v in label_mapping.items()
}


# Compare attack types
actual_multiclass = y_multiclass.astype(str)

predicted_multiclass = (
    predictions["attack_type"]
    .astype(str)
)


multiclass_correct = (
    predicted_multiclass.to_numpy()
    == actual_multiclass.to_numpy()
).sum()


multiclass_accuracy = (
    multiclass_correct /
    len(actual_multiclass)
)


print(
    f"\nMulti-class accuracy from prediction engine: "
    f"{multiclass_accuracy:.6f}"
)

print(
    f"Multi-class correct predictions: "
    f"{multiclass_correct:,}"
)

print(
    f"Multi-class incorrect predictions: "
    f"{len(actual_multiclass) - multiclass_correct:,}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "prediction_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)


result_df = pd.DataFrame({

    "Actual_Binary": y_binary,

    "Predicted_Binary": predictions[
        "is_attack"
    ],

    "Actual_Attack_Type": y_multiclass,

    "Predicted_Attack_Type": predictions[
        "attack_type"
    ],

    "Confidence": predictions[
        "confidence"
    ]

})


result_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(
    f"\nPrediction results saved:\n"
    f"{OUTPUT_PATH}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


if len(predictions) == len(X_test):

    print("✅ Prediction count matches test samples.")

else:

    print("❌ Prediction count mismatch.")

    raise ValueError(
        "Prediction count does not match test dataset."
    )


if predictions["confidence"].between(
    0,
    1
).all():

    print("✅ Confidence values are valid.")

else:

    print("❌ Invalid confidence values.")

    raise ValueError(
        "Confidence must be between 0 and 1."
    )


if result_df.isna().sum().sum() == 0:

    print("✅ No missing prediction results.")

else:

    print("❌ Missing prediction results.")


print("\n" + "=" * 70)
print("✅ PHASE 10.5 COMPLETED")
print("=" * 70)

print(
    "\nAI-NIDS prediction engine successfully "
    "tested on real test data."
)