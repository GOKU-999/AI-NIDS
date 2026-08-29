"""
======================================================================
AI-NIDS — FastAPI Inference Service
PHASE 11

Architecture:

Network Flow
     |
     v
50 Selected Features
     |
     v
Binary Random Forest
     |
     +---- BENIGN ------> BENIGN
     |
     +---- ATTACK
              |
              v
      Multi-class Random Forest
              |
              v
         Attack Type

Run from project root:

    python -m uvicorn ml-service.src.api:app --reload

Or from ml-service/src:

    uvicorn api:app --reload
======================================================================
"""

from pathlib import Path
import json
import sys
import time
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)


# ======================================================================
# PATH CONFIGURATION
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()

# ml-service/src/api.py
#          ^
#          └── ml-service
BASE_DIR = CURRENT_FILE.parent.parent

MODELS_DIR = BASE_DIR / "models"

BINARY_MODEL_PATH = (
    MODELS_DIR
    / "binary"
    / "random_forest_binary.joblib"
)

BINARY_FEATURES_PATH = (
    MODELS_DIR
    / "binary"
    / "feature_names.json"
)

MULTICLASS_MODEL_PATH = (
    MODELS_DIR
    / "multiclass"
    / "random_forest_multiclass.joblib"
)

MULTICLASS_FEATURES_PATH = (
    MODELS_DIR
    / "multiclass"
    / "feature_names.json"
)

LABEL_MAPPING_PATH = (
    MODELS_DIR
    / "multiclass"
    / "label_mapping.json"
)


# ======================================================================
# FASTAPI APPLICATION
# ======================================================================

app = FastAPI(
    title="AI-NIDS API",
    description=(
        "AI-powered Network Intrusion Detection System using "
        "a two-stage Random Forest architecture."
    ),
    version="1.0.0",
)


# ======================================================================
# CORS
# ======================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================
# GLOBAL MODEL VARIABLES
# ======================================================================

binary_model = None
multiclass_model = None

binary_features: List[str] = []
multiclass_features: List[str] = []

label_mapping: Dict[str, str] = {}


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def load_json(path: Path):
    """
    Load JSON file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_feature_list(data) -> List[str]:
    """
    Normalize different possible feature_names.json formats.

    Supported examples:

        ["feature1", "feature2"]

    or:

        {"features": ["feature1", "feature2"]}

    """

    if isinstance(data, list):
        return [str(x) for x in data]

    if isinstance(data, dict):

        if "features" in data:
            return [str(x) for x in data["features"]]

        if "feature_names" in data:
            return [str(x) for x in data["feature_names"]]

        # Sometimes JSON may contain:
        # {"0": "feature1", "1": "feature2"}

        try:
            ordered_items = sorted(
                data.items(),
                key=lambda item: int(item[0])
            )

            return [str(value) for _, value in ordered_items]

        except Exception:
            pass

    raise ValueError(
        "Unsupported feature_names.json format."
    )


def normalize_label_mapping(data) -> Dict[str, str]:
    """
    Normalize label mapping.

    Expected:

        {
            "0": "BENIGN",
            "1": "DoS",
            ...
        }

    """

    if not isinstance(data, dict):
        raise ValueError(
            "label_mapping.json must contain a JSON object."
        )

    return {
        str(key): str(value)
        for key, value in data.items()
    }


def load_models():
    """
    Load both trained Random Forest models and configurations.
    """

    global binary_model
    global multiclass_model
    global binary_features
    global multiclass_features
    global label_mapping

    print("=" * 70)
    print("AI-NIDS — LOADING MODELS")
    print("=" * 70)

    # --------------------------------------------------------------
    # Binary model
    # --------------------------------------------------------------

    print("\nLoading Binary Random Forest...")

    if not BINARY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Binary model not found:\n{BINARY_MODEL_PATH}"
        )

    binary_model = joblib.load(BINARY_MODEL_PATH)

    print("✅ Binary model loaded.")

    # --------------------------------------------------------------
    # Binary feature list
    # --------------------------------------------------------------

    print("\nLoading binary feature list...")

    binary_feature_data = load_json(
        BINARY_FEATURES_PATH
    )

    binary_features = normalize_feature_list(
        binary_feature_data
    )

    print(
        f"✅ Binary features loaded: "
        f"{len(binary_features)}"
    )

    # --------------------------------------------------------------
    # Multi-class model
    # --------------------------------------------------------------

    print("\nLoading Multi-class Random Forest...")

    if not MULTICLASS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Multi-class model not found:\n"
            f"{MULTICLASS_MODEL_PATH}"
        )

    multiclass_model = joblib.load(
        MULTICLASS_MODEL_PATH
    )

    print("✅ Multi-class model loaded.")

    # --------------------------------------------------------------
    # Multi-class feature list
    # --------------------------------------------------------------

    print("\nLoading multi-class feature list...")

    multiclass_feature_data = load_json(
        MULTICLASS_FEATURES_PATH
    )

    multiclass_features = normalize_feature_list(
        multiclass_feature_data
    )

    print(
        f"✅ Multi-class features loaded: "
        f"{len(multiclass_features)}"
    )

    # --------------------------------------------------------------
    # Label mapping
    # --------------------------------------------------------------

    print("\nLoading label mapping...")

    label_mapping_data = load_json(
        LABEL_MAPPING_PATH
    )

    label_mapping = normalize_label_mapping(
        label_mapping_data
    )

    print("✅ Label mapping loaded.")

    # --------------------------------------------------------------
    # Validate feature lists
    # --------------------------------------------------------------

    if binary_features != multiclass_features:

        raise ValueError(
            "Binary and multi-class feature lists do not match."
        )

    if len(binary_features) != 50:

        print(
            f"⚠️ Warning: expected 50 features, "
            f"found {len(binary_features)}."
        )

    print("\n" + "=" * 70)
    print("MODEL LOADING COMPLETE")
    print("=" * 70)

    print(
        f"Binary features:      {len(binary_features)}"
    )

    print(
        f"Multi-class features: {len(multiclass_features)}"
    )

    print(
        f"Attack classes:       {len(label_mapping)}"
    )

    print("\nLabel mapping:")

    for key, value in label_mapping.items():
        print(f"  {key} -> {value}")

    print("=" * 70)


# ======================================================================
# STARTUP
# ======================================================================

@app.on_event("startup")
def startup_event():
    """
    Load models when FastAPI starts.
    """

    load_models()


# ======================================================================
# FEATURE VALIDATION
# ======================================================================

def prepare_features(
    feature_dict: Dict[str, float]
) -> pd.DataFrame:
    """
    Validate and prepare the 50 input features.

    The model receives columns in exactly the same
    order used during training.
    """

    if not binary_features:
        raise RuntimeError(
            "Models are not loaded."
        )

    expected_features = binary_features

    provided_features = set(feature_dict.keys())
    expected_set = set(expected_features)

    # --------------------------------------------------------------
    # Missing features
    # --------------------------------------------------------------

    missing = sorted(
        expected_set - provided_features
    )

    if missing:

        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing features",
                "missing_features": missing,
                "expected_feature_count": len(
                    expected_features
                ),
                "provided_feature_count": len(
                    provided_features
                ),
            },
        )

    # --------------------------------------------------------------
    # Extra features
    # --------------------------------------------------------------

    extra = sorted(
        provided_features - expected_set
    )

    if extra:

        raise HTTPException(
            status_code=422,
            detail={
                "error": "Unknown features",
                "extra_features": extra,
            },
        )

    # --------------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------------

    values = {}

    for feature in expected_features:

        value = feature_dict[feature]

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):

            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Non-numeric feature value",
                    "feature": feature,
                    "value": str(value),
                },
            )

        if not np.isfinite(numeric_value):

            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Feature value must be finite",
                    "feature": feature,
                    "value": str(value),
                },
            )

        values[feature] = numeric_value

    # --------------------------------------------------------------
    # Create DataFrame in training order
    # --------------------------------------------------------------

    dataframe = pd.DataFrame(
        [values],
        columns=expected_features
    )

    return dataframe


# ======================================================================
# PREDICTION ENGINE
# ======================================================================

def predict_flow(
    feature_dict: Dict[str, float]
):
    """
    Execute the two-stage AI-NIDS prediction pipeline.

    Stage 1:
        Binary Random Forest

    Stage 2:
        Multi-class Random Forest
        ONLY when Stage 1 predicts ATTACK.
    """

    dataframe = prepare_features(
        feature_dict
    )

    # ==============================================================
    # STAGE 1 — BINARY CLASSIFICATION
    # ==============================================================

    binary_prediction = binary_model.predict(
        dataframe
    )[0]

    binary_probabilities = (
        binary_model.predict_proba(dataframe)[0]
    )

    binary_classes = list(
        binary_model.classes_
    )

    # Probability of the predicted class
    binary_class_index = binary_classes.index(
        binary_prediction
    )

    binary_confidence = float(
        binary_probabilities[binary_class_index]
    )

    # ==============================================================
    # BENIGN
    # ==============================================================

    if int(binary_prediction) == 0:

        return {
            "is_attack": False,
            "prediction": "BENIGN",
            "attack_type": "BENIGN",
            "confidence": binary_confidence,
            "binary_confidence": binary_confidence,
            "attack_confidence": None,
        }

    # ==============================================================
    # STAGE 2 — MULTI-CLASS ATTACK CLASSIFICATION
    # ==============================================================

    multiclass_prediction = (
        multiclass_model.predict(dataframe)[0]
    )

    multiclass_probabilities = (
        multiclass_model.predict_proba(dataframe)[0]
    )

    multiclass_classes = list(
        multiclass_model.classes_
    )

    multiclass_class_index = multiclass_classes.index(
        multiclass_prediction
    )

    attack_confidence = float(
        multiclass_probabilities[
            multiclass_class_index
        ]
    )

    # --------------------------------------------------------------
    # Convert numeric class to attack name
    # --------------------------------------------------------------

    attack_type = label_mapping.get(
        str(int(multiclass_prediction)),
        str(multiclass_prediction)
    )

    return {
        "is_attack": True,
        "prediction": "ATTACK",
        "attack_type": attack_type,
        "confidence": attack_confidence,
        "binary_confidence": binary_confidence,
        "attack_confidence": attack_confidence,
    }


# ======================================================================
# ROOT ENDPOINT
# ======================================================================

@app.get("/")
def root():

    return {
        "service": "AI-NIDS",
        "status": "running",
        "version": "1.0.0",
        "architecture": (
            "Binary Random Forest → "
            "Multi-class Random Forest"
        ),
        "features": len(binary_features),
        "attack_classes": len(label_mapping),
    }


# ======================================================================
# HEALTH ENDPOINT
# ======================================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health():

    return {
        "status": "healthy",
        "service": "AI-NIDS",
        "binary_model_loaded": (
            binary_model is not None
        ),
        "multiclass_model_loaded": (
            multiclass_model is not None
        ),
        "feature_count": len(binary_features),
    }


# ======================================================================
# MODEL INFORMATION
# ======================================================================

@app.get(
    "/model-info",
    response_model=ModelInfoResponse
)
def model_info():

    if (
        binary_model is None
        or multiclass_model is None
    ):

        raise HTTPException(
            status_code=503,
            detail="Models are not loaded."
        )

    attack_classes = list(
        label_mapping.values()
    )

    return {
        "binary_model": (
            "RandomForestClassifier"
        ),
        "multiclass_model": (
            "RandomForestClassifier"
        ),
        "feature_count": len(
            binary_features
        ),
        "attack_classes": attack_classes,
        "architecture": (
            "Binary Random Forest → "
            "Multi-class Random Forest"
        ),
    }


# ======================================================================
# PREDICTION ENDPOINT
# ======================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: PredictionRequest
):

    if (
        binary_model is None
        or multiclass_model is None
    ):

        raise HTTPException(
            status_code=503,
            detail="AI models are not loaded."
        )

    start_time = time.perf_counter()

    result = predict_flow(
        request.features
    )

    elapsed = (
        time.perf_counter() - start_time
    )

    # Add internal timing information to console
    print(
        f"Prediction completed in "
        f"{elapsed * 1000:.2f} ms | "
        f"{result['prediction']} | "
        f"{result['attack_type']}"
    )

    return result


# ======================================================================
# FEATURE LIST ENDPOINT
# ======================================================================

@app.get("/features")
def get_features():

    if not binary_features:

        raise HTTPException(
            status_code=503,
            detail="Feature configuration not loaded."
        )

    return {
        "count": len(binary_features),
        "features": binary_features,
    }


# ======================================================================
# ATTACK CLASSES ENDPOINT
# ======================================================================

@app.get("/attack-classes")
def get_attack_classes():

    if not label_mapping:

        raise HTTPException(
            status_code=503,
            detail="Label mapping not loaded."
        )

    return {
        "count": len(label_mapping),
        "classes": label_mapping,
    }


# ======================================================================
# SERVER INFORMATION
# ======================================================================

if __name__ == "__main__":

    import uvicorn

    print()
    print("=" * 70)
    print("AI-NIDS FASTAPI SERVER")
    print("=" * 70)
    print()
    print("Starting server...")
    print()
    print("API:")
    print("  http://127.0.0.1:8000")
    print()
    print("Swagger documentation:")
    print("  http://127.0.0.1:8000/docs")
    print()
    print("ReDoc:")
    print("  http://127.0.0.1:8000/redoc")
    print()
    print("=" * 70)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )