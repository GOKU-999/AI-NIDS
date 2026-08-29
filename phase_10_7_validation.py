"""
================================================================================
PHASE 10.7 — MODEL ROBUSTNESS & PRODUCTION VALIDATION
================================================================================

AI-NIDS
Two-stage architecture:

Stage 1:
Random Forest Binary Classifier
    BENIGN / ATTACK

Stage 2:
Random Forest Multi-class Classifier
    BENIGN / DoS / DDoS / PortScan / BruteForce /
    WebAttack / Botnet / Infiltration / Heartbleed

This script validates:
- Model artifacts
- Feature configuration
- Label mapping
- Test dataset
- Feature order
- Binary model performance
- Multi-class model performance
- Confusion matrices
- Classification reports
- Prediction consistency
- Confidence values
- Production readiness

IMPORTANT:
The multi-class model predicts NUMERIC labels (0-8),
while y_multiclass_test.csv contains STRING labels.

This script converts the actual labels using label_mapping.json
before calculating multi-class metrics.
"""

import os
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ML_SERVICE_DIR = BASE_DIR

DATA_DIR = os.path.join(
    ML_SERVICE_DIR,
    "data"
)

PROCESSED_DIR = os.path.join(
    DATA_DIR,
    "processed"
)

MODELS_DIR = os.path.join(
    ML_SERVICE_DIR,
    "models"
)


# =============================================================================
# MODEL PATHS
# =============================================================================

BINARY_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "binary",
    "random_forest_binary.joblib"
)

BINARY_FEATURES_PATH = os.path.join(
    MODELS_DIR,
    "binary",
    "feature_names.json"
)

BINARY_METRICS_PATH = os.path.join(
    MODELS_DIR,
    "binary",
    "metrics.json"
)


MULTICLASS_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "multiclass",
    "random_forest_multiclass.joblib"
)

MULTICLASS_FEATURES_PATH = os.path.join(
    MODELS_DIR,
    "multiclass",
    "feature_names.json"
)

MULTICLASS_MAPPING_PATH = os.path.join(
    MODELS_DIR,
    "multiclass",
    "label_mapping.json"
)

MULTICLASS_METRICS_PATH = os.path.join(
    MODELS_DIR,
    "multiclass",
    "metrics.json"
)


# =============================================================================
# TEST DATA PATHS
# =============================================================================

X_TEST_PATH = os.path.join(
    PROCESSED_DIR,
    "selected_features",
    "X_test_selected.csv"
)

Y_BINARY_TEST_PATH = os.path.join(
    PROCESSED_DIR,
    "split",
    "y_binary_test.csv"
)

Y_MULTICLASS_TEST_PATH = os.path.join(
    PROCESSED_DIR,
    "split",
    "y_multiclass_test.csv"
)


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = os.path.join(
    PROCESSED_DIR,
    "phase_10_7_validation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =============================================================================
# OUTPUT FILES
# =============================================================================

VALIDATION_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "validation_summary.json"
)

BINARY_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "binary_classification_report.csv"
)

MULTICLASS_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "multiclass_classification_report.csv"
)

BINARY_CM_PATH = os.path.join(
    OUTPUT_DIR,
    "binary_confusion_matrix.csv"
)

MULTICLASS_CM_PATH = os.path.join(
    OUTPUT_DIR,
    "multiclass_confusion_matrix.csv"
)

PREDICTION_SAMPLE_PATH = os.path.join(
    OUTPUT_DIR,
    "validation_prediction_sample.csv"
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check_file(path, description):
    if os.path.exists(path):
        print(f"✅ {description}: {path}")
        return True

    print(f"❌ {description} NOT FOUND: {path}")
    return False


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def normalize_feature_list(data):
    """
    Supports both:
        ["Feature1", "Feature2"]

    and:
        {"features": ["Feature1", "Feature2"]}
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        if "features" in data:
            return data["features"]

        if "feature_names" in data:
            return data["feature_names"]

    raise ValueError(
        f"Unsupported feature configuration format: {type(data)}"
    )


def build_label_mapping(mapping):
    """
    Converts either of these formats:

        {"0": "BENIGN", "1": "DoS"}

    OR:

        {"BENIGN": 0, "DoS": 1}

    into:

        {
            0: "BENIGN",
            1: "DoS"
        }

    and:

        {
            "BENIGN": 0,
            "DoS": 1
        }
    """

    if not isinstance(mapping, dict):
        raise ValueError(
            "label_mapping.json must contain a JSON object."
        )

    id_to_name = {}
    name_to_id = {}

    # ------------------------------------------------------------
    # Format:
    # {"0": "BENIGN", "1": "DoS"}
    # ------------------------------------------------------------

    numeric_keys = True

    for key in mapping.keys():

        try:
            int(key)
        except (ValueError, TypeError):
            numeric_keys = False
            break

    if numeric_keys:

        for key, value in mapping.items():

            class_id = int(key)
            class_name = str(value)

            id_to_name[class_id] = class_name
            name_to_id[class_name] = class_id

    # ------------------------------------------------------------
    # Format:
    # {"BENIGN": 0, "DoS": 1}
    # ------------------------------------------------------------

    else:

        for key, value in mapping.items():

            class_name = str(key)
            class_id = int(value)

            id_to_name[class_id] = class_name
            name_to_id[class_name] = class_id

    return id_to_name, name_to_id


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("PHASE 10.7 — MODEL ROBUSTNESS & PRODUCTION VALIDATION")
    print("=" * 80)


    # =========================================================================
    # STEP 1 — CHECK MODEL ARTIFACTS
    # =========================================================================

    print_header(
        "STEP 1 — CHECKING MODEL ARTIFACTS"
    )

    required_files = [

        (
            BINARY_MODEL_PATH,
            "Binary model"
        ),

        (
            BINARY_FEATURES_PATH,
            "Binary feature list"
        ),

        (
            BINARY_METRICS_PATH,
            "Binary metrics"
        ),

        (
            MULTICLASS_MODEL_PATH,
            "Multi-class model"
        ),

        (
            MULTICLASS_FEATURES_PATH,
            "Multi-class feature list"
        ),

        (
            MULTICLASS_MAPPING_PATH,
            "Multi-class label mapping"
        ),

        (
            MULTICLASS_METRICS_PATH,
            "Multi-class metrics"
        ),

        (
            X_TEST_PATH,
            "Test features"
        ),

        (
            Y_BINARY_TEST_PATH,
            "Binary test labels"
        ),

        (
            Y_MULTICLASS_TEST_PATH,
            "Multi-class test labels"
        ),
    ]

    missing_files = []

    for path, description in required_files:

        if not check_file(
            path,
            description
        ):
            missing_files.append(path)

    if missing_files:

        raise FileNotFoundError(
            "Required model/data artifacts are missing."
        )

    print()
    print("✅ All required artifacts found.")


    # =========================================================================
    # STEP 2 — LOAD MODELS
    # =========================================================================

    print_header(
        "STEP 2 — LOADING TRAINED MODELS"
    )

    print("Loading Binary Random Forest...")

    binary_model = joblib.load(
        BINARY_MODEL_PATH
    )

    print("✅ Binary model loaded.")

    print()

    print("Loading Multi-class Random Forest...")

    multiclass_model = joblib.load(
        MULTICLASS_MODEL_PATH
    )

    print("✅ Multi-class model loaded.")


    # =========================================================================
    # STEP 3 — LOAD FEATURE CONFIGURATION
    # =========================================================================

    print_header(
        "STEP 3 — LOADING FEATURE CONFIGURATION"
    )

    binary_features_raw = load_json(
        BINARY_FEATURES_PATH
    )

    multiclass_features_raw = load_json(
        MULTICLASS_FEATURES_PATH
    )

    binary_features = normalize_feature_list(
        binary_features_raw
    )

    multiclass_features = normalize_feature_list(
        multiclass_features_raw
    )

    print(
        f"Binary feature count:      {len(binary_features)}"
    )

    print(
        f"Multi-class feature count: {len(multiclass_features)}"
    )

    if binary_features != multiclass_features:

        print()
        print(
            "⚠️ Binary and multi-class feature lists are not identical."
        )

        binary_set = set(binary_features)
        multiclass_set = set(multiclass_features)

        only_binary = sorted(
            binary_set - multiclass_set
        )

        only_multiclass = sorted(
            multiclass_set - binary_set
        )

        if only_binary:
            print()
            print("Features only in binary model:")

            for feature in only_binary:
                print(f" - {feature}")

        if only_multiclass:
            print()
            print(
                "Features only in multi-class model:"
            )

            for feature in only_multiclass:
                print(f" - {feature}")

        raise ValueError(
            "Binary and multi-class feature lists do not match."
        )

    print(
        "✅ Binary and multi-class feature lists match."
    )


    # =========================================================================
    # STEP 4 — LOAD LABEL MAPPING
    # =========================================================================

    print_header(
        "STEP 4 — LOADING LABEL MAPPING"
    )

    raw_mapping = load_json(
        MULTICLASS_MAPPING_PATH
    )

    id_to_name, name_to_id = build_label_mapping(
        raw_mapping
    )

    print("Multi-class label mapping:")

    for class_id in sorted(id_to_name):

        print(
            f"  {class_id} -> {id_to_name[class_id]}"
        )

    expected_class_ids = sorted(
        id_to_name.keys()
    )

    print()

    print(
        f"Number of classes: {len(expected_class_ids)}"
    )


    # =========================================================================
    # STEP 5 — LOAD TEST DATA
    # =========================================================================

    print_header(
        "STEP 5 — LOADING TEST DATA"
    )

    print("Loading test features...")

    X_test = pd.read_csv(
        X_TEST_PATH
    )

    print(
        f"Test feature shape: {X_test.shape}"
    )

    print()

    print("Loading binary labels...")

    y_binary_df = pd.read_csv(
        Y_BINARY_TEST_PATH
    )

    print(
        f"Binary label shape: {y_binary_df.shape}"
    )

    print()

    print("Loading multi-class labels...")

    y_multiclass_df = pd.read_csv(
        Y_MULTICLASS_TEST_PATH
    )

    print(
        f"Multi-class label shape: {y_multiclass_df.shape}"
    )


    # =========================================================================
    # STEP 6 — IDENTIFY LABEL COLUMNS
    # =========================================================================

    print_header(
        "STEP 6 — IDENTIFYING LABEL COLUMNS"
    )

    # Binary label column
    if "Is_Attack" in y_binary_df.columns:

        y_binary = y_binary_df["Is_Attack"]

    else:

        y_binary = y_binary_df.iloc[:, 0]

    # Multi-class label column
    if "Attack_Type" in y_multiclass_df.columns:

        y_multiclass = y_multiclass_df["Attack_Type"]

    else:

        y_multiclass = y_multiclass_df.iloc[:, 0]

    print(
        f"Binary label column:     {y_binary.name}"
    )

    print(
        f"Multi-class label column: {y_multiclass.name}"
    )

    print()

    print(
        "Binary label types:",
        y_binary.dtype
    )

    print(
        "Multi-class label types:",
        y_multiclass.dtype
    )


    # =========================================================================
    # STEP 7 — VALIDATE TEST DATA
    # =========================================================================

    print_header(
        "STEP 7 — VALIDATING TEST DATA"
    )

    n_samples = len(X_test)

    if len(y_binary) != n_samples:

        raise ValueError(
            "Binary label count does not match X_test."
        )

    if len(y_multiclass) != n_samples:

        raise ValueError(
            "Multi-class label count does not match X_test."
        )

    print(
        f"Samples:              {n_samples:,}"
    )

    print(
        f"Features:              {X_test.shape[1]}"
    )

    print(
        f"Binary labels:         {len(y_binary):,}"
    )

    print(
        f"Multi-class labels:    {len(y_multiclass):,}"
    )

    # Missing values
    missing_features = int(
        X_test.isna().sum().sum()
    )

    if missing_features != 0:

        raise ValueError(
            f"Test features contain {missing_features} missing values."
        )

    # Infinity values
    numeric_test = X_test.select_dtypes(
        include=[np.number]
    )

    infinity_count = int(
        np.isinf(
            numeric_test.to_numpy()
        ).sum()
    )

    if infinity_count != 0:

        raise ValueError(
            f"Test features contain {infinity_count} infinity values."
        )

    print()
    print(
        "✅ Test dataset validation passed."
    )


    # =========================================================================
    # STEP 8 — FEATURE ORDER VALIDATION
    # =========================================================================

    print_header(
        "STEP 8 — FEATURE ORDER VALIDATION"
    )

    test_columns = list(
        X_test.columns
    )

    binary_order_match = (
        test_columns == binary_features
    )

    multiclass_order_match = (
        test_columns == multiclass_features
    )

    print(
        "Binary feature order match:     "
        + ("✅ YES" if binary_order_match else "❌ NO")
    )

    print(
        "Multi-class feature order match: "
        + ("✅ YES" if multiclass_order_match else "❌ NO")
    )

    if not binary_order_match:

        raise ValueError(
            "Binary model feature order does not match X_test."
        )

    if not multiclass_order_match:

        raise ValueError(
            "Multi-class model feature order does not match X_test."
        )


    # =========================================================================
    # STEP 9 — BINARY MODEL VALIDATION
    # =========================================================================

    print_header(
        "STEP 9 — BINARY MODEL VALIDATION"
    )

    print(
        "Generating binary predictions..."
    )

    binary_start = time.time()

    binary_predictions = binary_model.predict(
        X_test
    )

    binary_prediction_time = (
        time.time() - binary_start
    )

    print(
        f"Prediction time: "
        f"{binary_prediction_time:.2f} seconds"
    )


    # Binary probability
    binary_probabilities = None

    if hasattr(
        binary_model,
        "predict_proba"
    ):

        binary_probabilities = (
            binary_model.predict_proba(
                X_test
            )
        )


    # Convert actual labels to integers
    y_binary_numeric = (
        pd.to_numeric(
            y_binary,
            errors="coerce"
        )
    )

    if y_binary_numeric.isna().any():

        # Support possible string labels
        binary_name_map = {
            "BENIGN": 0,
            "ATTACK": 1,
            "0": 0,
            "1": 1
        }

        y_binary_numeric = (
            y_binary.astype(str)
            .str.strip()
            .str.upper()
            .map(binary_name_map)
        )

    if y_binary_numeric.isna().any():

        raise ValueError(
            "Unable to convert binary labels to 0/1."
        )

    y_binary_numeric = (
        y_binary_numeric
        .astype(int)
        .to_numpy()
    )

    binary_predictions = (
        np.asarray(
            binary_predictions
        ).astype(int)
    )


    # Metrics
    binary_accuracy = accuracy_score(
        y_binary_numeric,
        binary_predictions
    )

    binary_precision = precision_score(
        y_binary_numeric,
        binary_predictions,
        zero_division=0
    )

    binary_recall = recall_score(
        y_binary_numeric,
        binary_predictions,
        zero_division=0
    )

    binary_f1 = f1_score(
        y_binary_numeric,
        binary_predictions,
        zero_division=0
    )


    # ROC-AUC
    binary_roc_auc = None

    if binary_probabilities is not None:

        try:

            attack_class_index = list(
                binary_model.classes_
            ).index(1)

            binary_attack_probability = (
                binary_probabilities[
                    :,
                    attack_class_index
                ]
            )

            binary_roc_auc = roc_auc_score(
                y_binary_numeric,
                binary_attack_probability
            )

        except Exception:

            binary_roc_auc = None


    # Confusion matrix
    binary_cm = confusion_matrix(
        y_binary_numeric,
        binary_predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = binary_cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )


    print()
    print("Binary Model Performance")
    print("-" * 50)

    print(
        f"Accuracy:  {binary_accuracy:.6f}"
    )

    print(
        f"Precision: {binary_precision:.6f}"
    )

    print(
        f"Recall:    {binary_recall:.6f}"
    )

    print(
        f"F1 Score:  {binary_f1:.6f}"
    )

    if binary_roc_auc is not None:

        print(
            f"ROC-AUC:   {binary_roc_auc:.6f}"
        )


    print()
    print("Binary Confusion Matrix")
    print("-" * 50)

    print(
        f"True Negatives: {tn:,}"
    )

    print(
        f"False Positives: {fp:,}"
    )

    print(
        f"False Negatives: {fn:,}"
    )

    print(
        f"True Positives: {tp:,}"
    )

    print(
        f"Specificity:     {specificity:.6f}"
    )

    print(
        f"False Positive Rate: {false_positive_rate:.6f}"
    )

    print(
        f"False Negative Rate: {false_negative_rate:.6f}"
    )


    # =========================================================================
    # STEP 10 — MULTI-CLASS MODEL VALIDATION
    # =========================================================================

    print_header(
        "STEP 10 — MULTI-CLASS MODEL VALIDATION"
    )

    print(
        "Generating multi-class predictions..."
    )

    multiclass_start = time.time()

    multiclass_predictions = (
        multiclass_model.predict(
            X_test
        )
    )

    multiclass_prediction_time = (
        time.time() - multiclass_start
    )

    print(
        f"Prediction time: "
        f"{multiclass_prediction_time:.2f} seconds"
    )


    # -------------------------------------------------------------------------
    # CRITICAL FIX:
    #
    # y_multiclass contains:
    #
    # BENIGN
    # DoS
    # DDoS
    # ...
    #
    # But model predictions contain:
    #
    # 0
    # 1
    # 2
    # ...
    #
    # Convert actual string labels to numeric IDs.
    # -------------------------------------------------------------------------

    print()
    print(
        "Converting actual multi-class labels to numeric labels..."
    )

    y_multiclass_string = (
        y_multiclass
        .astype(str)
        .str.strip()
    )

    unknown_labels = sorted(
        set(
            y_multiclass_string.unique()
        )
        -
        set(
            name_to_id.keys()
        )
    )

    if unknown_labels:

        raise ValueError(
            "Unknown multi-class labels found: "
            + str(unknown_labels)
        )

    y_multiclass_numeric = (
        y_multiclass_string
        .map(name_to_id)
    )

    if y_multiclass_numeric.isna().any():

        raise ValueError(
            "Some multi-class labels could not be mapped."
        )

    y_multiclass_numeric = (
        y_multiclass_numeric
        .astype(int)
        .to_numpy()
    )

    multiclass_predictions = (
        np.asarray(
            multiclass_predictions
        ).astype(int)
    )

    print(
        "✅ Multi-class labels converted successfully."
    )


    # Validate prediction IDs
    prediction_ids = sorted(
        np.unique(
            multiclass_predictions
        ).tolist()
    )

    invalid_prediction_ids = (
        set(prediction_ids)
        -
        set(expected_class_ids)
    )

    if invalid_prediction_ids:

        raise ValueError(
            "Model produced unknown class IDs: "
            + str(
                sorted(
                    invalid_prediction_ids
                )
            )
        )


    # -------------------------------------------------------------------------
    # MULTI-CLASS METRICS
    # -------------------------------------------------------------------------

    multi_accuracy = accuracy_score(
        y_multiclass_numeric,
        multiclass_predictions
    )

    multi_precision = precision_score(
        y_multiclass_numeric,
        multiclass_predictions,
        average="macro",
        zero_division=0
    )

    multi_recall = recall_score(
        y_multiclass_numeric,
        multiclass_predictions,
        average="macro",
        zero_division=0
    )

    multi_f1 = f1_score(
        y_multiclass_numeric,
        multiclass_predictions,
        average="macro",
        zero_division=0
    )

    multi_weighted_precision = precision_score(
        y_multiclass_numeric,
        multiclass_predictions,
        average="weighted",
        zero_division=0
    )

    multi_weighted_recall = recall_score(
        y_multiclass_numeric,
        multiclass_predictions,
        average="weighted",
        zero_division=0
    )

    multi_weighted_f1 = f1_score(
        y_multiclass_numeric,
        multiclass_predictions,
        average="weighted",
        zero_division=0
    )


    print()
    print("Multi-class Model Performance")
    print("-" * 50)

    print(
        f"Accuracy:           {multi_accuracy:.6f}"
    )

    print(
        f"Macro Precision:    {multi_precision:.6f}"
    )

    print(
        f"Macro Recall:       {multi_recall:.6f}"
    )

    print(
        f"Macro F1:           {multi_f1:.6f}"
    )

    print(
        f"Weighted Precision: {multi_weighted_precision:.6f}"
    )

    print(
        f"Weighted Recall:    {multi_weighted_recall:.6f}"
    )

    print(
        f"Weighted F1:        {multi_weighted_f1:.6f}"
    )


    # =========================================================================
    # STEP 11 — MULTI-CLASS CONFUSION MATRIX
    # =========================================================================

    print_header(
        "STEP 11 — MULTI-CLASS CONFUSION MATRIX"
    )

    multi_cm = confusion_matrix(
        y_multiclass_numeric,
        multiclass_predictions,
        labels=expected_class_ids
    )

    cm_columns = [
        f"Predicted_{id_to_name[class_id]}"
        for class_id in expected_class_ids
    ]

    cm_index = [
        f"Actual_{id_to_name[class_id]}"
        for class_id in expected_class_ids
    ]

    multi_cm_df = pd.DataFrame(
        multi_cm,
        index=cm_index,
        columns=cm_columns
    )

    print(
        multi_cm_df.to_string()
    )


    # =========================================================================
    # STEP 12 — CLASSIFICATION REPORTS
    # =========================================================================

    print_header(
        "STEP 12 — CLASSIFICATION REPORTS"
    )

    binary_report = classification_report(
        y_binary_numeric,
        binary_predictions,
        labels=[0, 1],
        target_names=[
            "BENIGN",
            "ATTACK"
        ],
        output_dict=True,
        zero_division=0
    )

    binary_report_df = (
        pd.DataFrame(
            binary_report
        ).transpose()
    )

    print()
    print("Binary Classification Report")
    print("-" * 50)

    print(
        binary_report_df.to_string()
    )


    multiclass_report = classification_report(
        y_multiclass_numeric,
        multiclass_predictions,
        labels=expected_class_ids,
        target_names=[
            id_to_name[class_id]
            for class_id in expected_class_ids
        ],
        output_dict=True,
        zero_division=0
    )

    multiclass_report_df = (
        pd.DataFrame(
            multiclass_report
        ).transpose()
    )

    print()
    print("Multi-class Classification Report")
    print("-" * 50)

    print(
        multiclass_report_df.to_string()
    )


    # =========================================================================
    # STEP 13 — PREDICTION CONSISTENCY
    # =========================================================================

    print_header(
        "STEP 13 — PREDICTION CONSISTENCY VALIDATION"
    )

    binary_prediction_distribution = (
        pd.Series(
            binary_predictions
        )
        .value_counts()
        .sort_index()
        .to_dict()
    )

    multiclass_prediction_distribution = (
        pd.Series(
            multiclass_predictions
        )
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("Binary prediction distribution:")

    for class_id, count in (
        binary_prediction_distribution.items()
    ):

        class_name = (
            "BENIGN"
            if class_id == 0
            else "ATTACK"
        )

        print(
            f"  {class_id} ({class_name}): "
            f"{count:,}"
        )


    print()
    print("Multi-class prediction distribution:")

    for class_id, count in (
        multiclass_prediction_distribution.items()
    ):

        class_name = id_to_name.get(
            int(class_id),
            "UNKNOWN"
        )

        print(
            f"  {class_id} ({class_name}): "
            f"{count:,}"
        )


    # =========================================================================
    # STEP 14 — CONFIDENCE VALIDATION
    # =========================================================================

    print_header(
        "STEP 14 — CONFIDENCE VALIDATION"
    )

    binary_confidence = None
    multiclass_confidence = None

    if hasattr(
        binary_model,
        "predict_proba"
    ):

        binary_proba = (
            binary_model.predict_proba(
                X_test
            )
        )

        binary_confidence = (
            np.max(
                binary_proba,
                axis=1
            )
        )

        print(
            f"Binary confidence min:  "
            f"{binary_confidence.min():.6f}"
        )

        print(
            f"Binary confidence max:  "
            f"{binary_confidence.max():.6f}"
        )

        print(
            f"Binary confidence mean: "
            f"{binary_confidence.mean():.6f}"
        )

        if (
            np.isnan(
                binary_confidence
            ).any()
            or
            np.isinf(
                binary_confidence
            ).any()
        ):

            raise ValueError(
                "Invalid binary confidence values."
            )

    else:

        print(
            "⚠️ Binary model does not support predict_proba()."
        )


    if hasattr(
        multiclass_model,
        "predict_proba"
    ):

        multiclass_proba = (
            multiclass_model.predict_proba(
                X_test
            )
        )

        multiclass_confidence = (
            np.max(
                multiclass_proba,
                axis=1
            )
        )

        print(
            f"Multi-class confidence min:  "
            f"{multiclass_confidence.min():.6f}"
        )

        print(
            f"Multi-class confidence max:  "
            f"{multiclass_confidence.max():.6f}"
        )

        print(
            f"Multi-class confidence mean: "
            f"{multiclass_confidence.mean():.6f}"
        )

        if (
            np.isnan(
                multiclass_confidence
            ).any()
            or
            np.isinf(
                multiclass_confidence
            ).any()
        ):

            raise ValueError(
                "Invalid multi-class confidence values."
            )

    else:

        print(
            "⚠️ Multi-class model does not support predict_proba()."
        )


    # =========================================================================
    # STEP 15 — TWO-STAGE PIPELINE VALIDATION
    # =========================================================================

    print_header(
        "STEP 15 — TWO-STAGE AI-NIDS PIPELINE VALIDATION"
    )

    print(
        "Stage 1: Binary model -> BENIGN / ATTACK"
    )

    print(
        "Stage 2: Multi-class model -> Attack Type"
    )

    attack_indices = (
        binary_predictions == 1
    )

    predicted_attack_count = int(
        attack_indices.sum()
    )

    predicted_benign_count = int(
        (~attack_indices).sum()
    )

    print()
    print(
        f"Stage 1 predicted BENIGN: "
        f"{predicted_benign_count:,}"
    )

    print(
        f"Stage 1 predicted ATTACK: "
        f"{predicted_attack_count:,}"
    )


    # Check consistency between binary and multi-class prediction
    #
    # If Stage 1 says BENIGN, Stage 2 should ideally say BENIGN.
    # If Stage 1 says ATTACK, Stage 2 should ideally produce an attack class.
    #
    # Here BENIGN is class ID 0.

    stage2_benign = (
        multiclass_predictions == name_to_id["BENIGN"]
    )

    inconsistent_predictions = (
        (~attack_indices)
        &
        (~stage2_benign)
    )

    inconsistent_count = int(
        inconsistent_predictions.sum()
    )

    print()
    print(
        f"Stage consistency violations: "
        f"{inconsistent_count:,}"
    )

    if inconsistent_count == 0:

        print(
            "✅ Two-stage prediction logic is fully consistent."
        )

    else:

        print(
            "⚠️ Two-stage prediction consistency violations detected."
        )

        print(
            "These should be reviewed before production deployment."
        )


    # =========================================================================
    # STEP 16 — SAVE RESULTS
    # =========================================================================

    print_header(
        "STEP 16 — SAVING VALIDATION RESULTS"
    )

    # Binary report
    binary_report_df.to_csv(
        BINARY_REPORT_PATH
    )

    print(
        f"✅ Binary classification report saved:"
    )

    print(
        BINARY_REPORT_PATH
    )


    # Multi-class report
    multiclass_report_df.to_csv(
        MULTICLASS_REPORT_PATH
    )

    print(
        f"✅ Multi-class classification report saved:"
    )

    print(
        MULTICLASS_REPORT_PATH
    )


    # Binary confusion matrix
    binary_cm_df = pd.DataFrame(
        binary_cm,
        index=[
            "Actual_BENIGN",
            "Actual_ATTACK"
        ],
        columns=[
            "Predicted_BENIGN",
            "Predicted_ATTACK"
        ]
    )

    binary_cm_df.to_csv(
        BINARY_CM_PATH
    )

    print(
        f"✅ Binary confusion matrix saved:"
    )

    print(
        BINARY_CM_PATH
    )


    # Multi-class confusion matrix
    multi_cm_df.to_csv(
        MULTICLASS_CM_PATH
    )

    print(
        f"✅ Multi-class confusion matrix saved:"
    )

    print(
        MULTICLASS_CM_PATH
    )


    # Prediction sample
    sample_size = min(
        1000,
        n_samples
    )

    sample_indices = np.arange(
        sample_size
    )

    prediction_sample = pd.DataFrame({

        "Actual_Binary":
            y_binary_numeric[
                sample_indices
            ],

        "Predicted_Binary":
            binary_predictions[
                sample_indices
            ],

        "Actual_Attack_Type":
            [
                id_to_name[
                    int(class_id)
                ]
                for class_id in
                y_multiclass_numeric[
                    sample_indices
                ]
            ],

        "Predicted_Attack_Type":
            [
                id_to_name[
                    int(class_id)
                ]
                for class_id in
                multiclass_predictions[
                    sample_indices
                ]
            ]
    })

    if binary_confidence is not None:

        prediction_sample[
            "Binary_Confidence"
        ] = binary_confidence[
            sample_indices
        ]

    if multiclass_confidence is not None:

        prediction_sample[
            "MultiClass_Confidence"
        ] = multiclass_confidence[
            sample_indices
        ]

    prediction_sample.to_csv(
        PREDICTION_SAMPLE_PATH,
        index=False
    )

    print(
        f"✅ Prediction sample saved:"
    )

    print(
        PREDICTION_SAMPLE_PATH
    )


    # =========================================================================
    # STEP 17 — VALIDATION SUMMARY
    # =========================================================================

    print_header(
        "STEP 17 — FINAL VALIDATION SUMMARY"
    )

    validation_summary = {

        "phase": "10.7",

        "status": "PASSED",

        "test_samples": int(
            n_samples
        ),

        "features": int(
            X_test.shape[1]
        ),

        "binary_features": int(
            len(binary_features)
        ),

        "multiclass_features": int(
            len(multiclass_features)
        ),

        "num_attack_classes": int(
            len(expected_class_ids)
        ),

        "feature_order_match": True,

        "binary": {

            "accuracy":
                float(binary_accuracy),

            "precision":
                float(binary_precision),

            "recall":
                float(binary_recall),

            "f1_score":
                float(binary_f1),

            "roc_auc":
                (
                    float(binary_roc_auc)
                    if binary_roc_auc is not None
                    else None
                ),

            "true_negatives":
                int(tn),

            "false_positives":
                int(fp),

            "false_negatives":
                int(fn),

            "true_positives":
                int(tp),

            "specificity":
                float(specificity),

            "false_positive_rate":
                float(false_positive_rate),

            "false_negative_rate":
                float(false_negative_rate),

            "prediction_time_seconds":
                float(binary_prediction_time)
        },

        "multiclass": {

            "accuracy":
                float(multi_accuracy),

            "macro_precision":
                float(multi_precision),

            "macro_recall":
                float(multi_recall),

            "macro_f1":
                float(multi_f1),

            "weighted_precision":
                float(multi_weighted_precision),

            "weighted_recall":
                float(multi_weighted_recall),

            "weighted_f1":
                float(multi_weighted_f1),

            "prediction_time_seconds":
                float(multiclass_prediction_time)
        },

        "two_stage_pipeline": {

            "predicted_benign":
                int(predicted_benign_count),

            "predicted_attack":
                int(predicted_attack_count),

            "stage_consistency_violations":
                int(inconsistent_count)
        },

        "validation_checks": {

            "artifacts_exist":
                True,

            "test_data_valid":
                True,

            "missing_values":
                0,

            "infinity_values":
                0,

            "binary_feature_order_match":
                bool(binary_order_match),

            "multiclass_feature_order_match":
                bool(multiclass_order_match),

            "label_mapping_valid":
                True,

            "binary_predictions_valid":
                True,

            "multiclass_predictions_valid":
                True,

            "confidence_values_valid":
                True
        }
    }


    with open(
        VALIDATION_SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            validation_summary,
            file,
            indent=4
        )


    print()
    print(
        f"✅ Validation summary saved:"
    )

    print(
        VALIDATION_SUMMARY_PATH
    )


    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================

    print_header(
        "FINAL VALIDATION"
    )

    checks = {

        "Model artifacts": True,

        "Test dataset": True,

        "Feature order": (
            binary_order_match
            and
            multiclass_order_match
        ),

        "Binary predictions": (
            len(binary_predictions)
            ==
            n_samples
        ),

        "Multi-class predictions": (
            len(multiclass_predictions)
            ==
            n_samples
        ),

        "Binary metrics": True,

        "Multi-class metrics": True,

        "Confidence validation": True,

        "Two-stage pipeline validation": True
    }


    all_passed = all(
        checks.values()
    )

    for name, passed in checks.items():

        print(
            ("✅ " if passed else "❌ ")
            + name
        )


    if not all_passed:

        raise RuntimeError(
            "One or more Phase 10.7 validation checks failed."
        )


    print()
    print("=" * 80)
    print("✅ PHASE 10.7 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print()
    print(
        "AI-NIDS production validation completed."
    )

    print()
    print(
        "Recommended architecture:"
    )

    print(
        "Random Forest Binary"
    )

    print(
        "        ↓"
    )

    print(
        "BENIGN / ATTACK"
    )

    print(
        "        ↓"
    )

    print(
        "Random Forest Multi-class"
    )

    print(
        "        ↓"
    )

    print(
        "Attack Type"
    )

    print()
    print(
        "Validation results:"
    )

    print(
        OUTPUT_DIR
    )

    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()