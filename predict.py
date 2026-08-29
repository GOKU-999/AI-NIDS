"""
AI-NIDS — Prediction Engine
PHASE 10.4

Two-stage architecture:
Stage 1: Binary Random Forest
    BENIGN / ATTACK

Stage 2: Multi-class Random Forest
    Attack Type

Models:
ml-service/models/binary/random_forest_binary.joblib
ml-service/models/multiclass/random_forest_multiclass.joblib
"""

import os
import json
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BINARY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "binary",
    "random_forest_binary.joblib"
)

BINARY_FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "binary",
    "feature_names.json"
)

MULTICLASS_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "multiclass",
    "random_forest_multiclass.joblib"
)

MULTICLASS_FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "multiclass",
    "feature_names.json"
)

LABEL_MAPPING_PATH = os.path.join(
    BASE_DIR,
    "models",
    "multiclass",
    "label_mapping.json"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("AI-NIDS — PREDICTION ENGINE")
print("=" * 70)

print("\nLoading Binary Random Forest...")
binary_model = joblib.load(BINARY_MODEL_PATH)
print("✅ Binary model loaded.")

print("\nLoading Multi-class Random Forest...")
multiclass_model = joblib.load(MULTICLASS_MODEL_PATH)
print("✅ Multi-class model loaded.")


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

with open(BINARY_FEATURE_PATH, "r") as f:
    binary_features = json.load(f)

with open(MULTICLASS_FEATURE_PATH, "r") as f:
    multiclass_features = json.load(f)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

with open(LABEL_MAPPING_PATH, "r") as f:
    label_mapping = json.load(f)


# Handle mappings such as:
# {"0": "BENIGN", "1": "DoS", ...}

reverse_label_mapping = {
    int(key): value
    for key, value in label_mapping.items()
}


print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(f"Binary features:      {len(binary_features)}")
print(f"Multi-class features: {len(multiclass_features)}")
print(f"Attack classes:       {len(reverse_label_mapping)}")

print("\nBinary model classes:")
print(binary_model.classes_)

print("\nMulti-class model classes:")
print(multiclass_model.classes_)


# ============================================================
# VALIDATE FEATURE LIST
# ============================================================

if binary_features != multiclass_features:

    print("\n⚠️ WARNING:")
    print("Binary and multi-class feature lists are different.")

    binary_set = set(binary_features)
    multiclass_set = set(multiclass_features)

    print("\nOnly in binary:")
    print(binary_set - multiclass_set)

    print("\nOnly in multiclass:")
    print(multiclass_set - binary_set)

    raise ValueError(
        "Binary and multi-class feature lists must match."
    )


FEATURES = binary_features


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(data):
    """
    Convert input data into a DataFrame containing
    exactly the 50 features expected by the models.
    """

    if isinstance(data, pd.DataFrame):

        df = data.copy()

    elif isinstance(data, dict):

        df = pd.DataFrame([data])

    elif isinstance(data, list):

        df = pd.DataFrame(data)

    else:

        raise TypeError(
            "Input must be pandas DataFrame, dictionary, or list."
        )


    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required features:\n"
            + "\n".join(missing_features)
        )


    # --------------------------------------------------------
    # Select only required features
    # --------------------------------------------------------

    df = df[FEATURES].copy()


    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    for column in FEATURES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    missing_count = df.isna().sum().sum()

    if missing_count > 0:

        raise ValueError(
            f"Input contains {missing_count} missing values."
        )


    # --------------------------------------------------------
    # Check infinity
    # --------------------------------------------------------

    infinity_count = np.isinf(df.to_numpy()).sum()

    if infinity_count > 0:

        raise ValueError(
            f"Input contains {infinity_count} infinite values."
        )


    return df


# ============================================================
# SINGLE FLOW PREDICTION
# ============================================================

def predict_flow(data):
    """
    Predict a single network flow.

    Returns:
        {
            "is_attack": 0/1,
            "binary_prediction": "BENIGN"/"ATTACK",
            "attack_type": "...",
            "confidence": float,
            "binary_probabilities": {...},
            "multiclass_probabilities": {...}
        }
    """

    df = prepare_input(data)


    # ========================================================
    # STAGE 1 — BINARY DETECTION
    # ========================================================

    binary_prediction = binary_model.predict(df)[0]

    binary_probabilities = binary_model.predict_proba(df)[0]

    binary_classes = binary_model.classes_


    binary_probability_dict = {
        int(cls): float(prob)
        for cls, prob in zip(
            binary_classes,
            binary_probabilities
        )
    }


    # --------------------------------------------------------
    # Convert binary result
    # --------------------------------------------------------

    if int(binary_prediction) == 0:

        is_attack = 0
        binary_result = "BENIGN"

    else:

        is_attack = 1
        binary_result = "ATTACK"


    binary_confidence = float(
        np.max(binary_probabilities)
    )


    # ========================================================
    # STAGE 2 — MULTI-CLASS CLASSIFICATION
    # ========================================================

    # Default values
    attack_type = "BENIGN"

    multiclass_confidence = 0.0

    multiclass_probability_dict = {}


    # --------------------------------------------------------
    # Only classify attack type when Stage 1 says ATTACK
    # --------------------------------------------------------

    if is_attack == 1:

        multiclass_prediction = multiclass_model.predict(df)[0]

        multiclass_probabilities = (
            multiclass_model.predict_proba(df)[0]
        )

        multiclass_classes = multiclass_model.classes_


        for cls, prob in zip(
            multiclass_classes,
            multiclass_probabilities
        ):

            class_id = int(cls)

            class_name = reverse_label_mapping.get(
                class_id,
                str(class_id)
            )

            multiclass_probability_dict[
                class_name
            ] = float(prob)


        attack_type = reverse_label_mapping.get(
            int(multiclass_prediction),
            str(multiclass_prediction)
        )


        multiclass_confidence = float(
            np.max(multiclass_probabilities)
        )


    # ========================================================
    # FINAL CONFIDENCE
    # ========================================================

    if is_attack == 0:

        confidence = binary_confidence

    else:

        confidence = multiclass_confidence


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "is_attack": is_attack,

        "binary_prediction": binary_result,

        "attack_type": attack_type,

        "confidence": confidence,

        "binary_confidence": binary_confidence,

        "multiclass_confidence": multiclass_confidence,

        "binary_probabilities": binary_probability_dict,

        "multiclass_probabilities": multiclass_probability_dict
    }


    return result


# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_batch(data):
    """
    Predict multiple network flows.

    Input:
        pandas DataFrame

    Returns:
        pandas DataFrame containing predictions.
    """

    df = prepare_input(data)


    # ========================================================
    # STAGE 1
    # ========================================================

    binary_predictions = binary_model.predict(df)

    binary_probabilities = binary_model.predict_proba(df)


    # ========================================================
    # STAGE 2
    # ========================================================

    multiclass_predictions = multiclass_model.predict(df)

    multiclass_probabilities = (
        multiclass_model.predict_proba(df)
    )


    # ========================================================
    # BUILD RESULTS
    # ========================================================

    results = []


    for i in range(len(df)):

        binary_pred = int(binary_predictions[i])

        if binary_pred == 0:

            is_attack = 0
            binary_result = "BENIGN"
            attack_type = "BENIGN"

            confidence = float(
                np.max(binary_probabilities[i])
            )

        else:

            is_attack = 1
            binary_result = "ATTACK"

            attack_class = int(
                multiclass_predictions[i]
            )

            attack_type = reverse_label_mapping.get(
                attack_class,
                str(attack_class)
            )

            confidence = float(
                np.max(multiclass_probabilities[i])
            )


        results.append({

            "is_attack": is_attack,

            "binary_prediction": binary_result,

            "attack_type": attack_type,

            "confidence": confidence
        })


    return pd.DataFrame(results)


# ============================================================
# TEST FUNCTION
# ============================================================

def run_test():

    print("\n" + "=" * 70)
    print("TESTING PREDICTION ENGINE")
    print("=" * 70)


    # --------------------------------------------------------
    # Create one sample using the training feature names
    # --------------------------------------------------------

    sample = {}

    for feature in FEATURES:

        sample[feature] = 0


    print("\nTesting with sample input...")
    print(f"Features supplied: {len(sample)}")


    try:

        result = predict_flow(sample)


        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)

        print(
            f"\nBinary Result : "
            f"{result['binary_prediction']}"
        )

        print(
            f"Attack Type   : "
            f"{result['attack_type']}"
        )

        print(
            f"Is Attack     : "
            f"{result['is_attack']}"
        )

        print(
            f"Confidence    : "
            f"{result['confidence']:.6f}"
        )


        print("\nBinary probabilities:")

        for key, value in result[
            "binary_probabilities"
        ].items():

            print(
                f"  {key}: {value:.6f}"
            )


        if result["is_attack"] == 1:

            print("\nMulti-class probabilities:")

            for key, value in result[
                "multiclass_probabilities"
            ].items():

                print(
                    f"  {key}: {value:.6f}"
                )


        print("\n" + "=" * 70)
        print("✅ PREDICTION ENGINE TEST PASSED")
        print("=" * 70)


    except Exception as e:

        print("\n❌ PREDICTION TEST FAILED")

        print(
            f"\nError: {e}"
        )

        raise


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_test()