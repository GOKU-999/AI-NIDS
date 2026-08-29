"""
======================================================================
AI-NIDS — FastAPI Inference Service
PHASE 11 — GitHub + Google Drive Deployment Version

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

The large Random Forest .joblib files are downloaded automatically
from Google Drive when they are not available locally.

Run locally:

    cd ml-service
    python src/api.py

Or:

    uvicorn src.api:app --host 0.0.0.0 --port 8000
======================================================================
"""

from pathlib import Path
import json
import time
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

APP_VERSION = "1.0.0"

EXPECTED_FEATURE_COUNT = 50

# ----------------------------------------------------------------------
# Google Drive model IDs
# ----------------------------------------------------------------------
# Binary model:
# 1nT-GtoBTuzwBxnTNYSsm8VliOjPsahj4
#
# Multi-class model:
# 1sNFCe39nc_txzra7p3fScMGvEaMoUVae
#
# IMPORTANT:
# These files must be shared as:
# "Anyone with the link → Viewer"
# ----------------------------------------------------------------------

BINARY_FILE_ID = "1nT-GtoBTuzwBxnTNYSsm8VliOjPsahj4"

MULTICLASS_FILE_ID = "1sNFCe39nc_txzra7p3fScMGvEaMoUVae"


# ======================================================================
# PATH CONFIGURATION
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()

# api.py
#   ↓
# src
#   ↓
# ml-service
BASE_DIR = CURRENT_FILE.parent.parent

MODELS_DIR = BASE_DIR / "models"

BINARY_DIR = MODELS_DIR / "binary"
MULTICLASS_DIR = MODELS_DIR / "multiclass"


BINARY_MODEL_PATH = (
    BINARY_DIR / "random_forest_binary.joblib"
)

BINARY_FEATURES_PATH = (
    BINARY_DIR / "feature_names.json"
)

BINARY_METRICS_PATH = (
    BINARY_DIR / "metrics.json"
)


MULTICLASS_MODEL_PATH = (
    MULTICLASS_DIR / "random_forest_multiclass.joblib"
)

MULTICLASS_FEATURES_PATH = (
    MULTICLASS_DIR / "feature_names.json"
)

MULTICLASS_LABEL_MAPPING_PATH = (
    MULTICLASS_DIR / "label_mapping.json"
)

MULTICLASS_METRICS_PATH = (
    MULTICLASS_DIR / "metrics.json"
)


# ======================================================================
# FASTAPI APPLICATION
# ======================================================================

app = FastAPI(
    title="AI-NIDS API",
    description=(
        "AI-powered Network Intrusion Detection System "
        "using a two-stage Random Forest architecture."
    ),
    version=APP_VERSION,
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
# GOOGLE DRIVE DOWNLOAD
# ======================================================================

def download_from_google_drive(
    file_id: str,
    destination: Path,
) -> None:
    """
    Download a file from Google Drive.

    This is used only when the model file does not already exist.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("DOWNLOADING MODEL FROM GOOGLE DRIVE")
    print("=" * 70)

    print(f"Destination:")
    print(f"  {destination}")

    url = (
        "https://drive.google.com/uc"
        f"?export=download&id={file_id}"
    )

    session = requests.Session()

    try:

        response = session.get(
            url,
            stream=True,
            timeout=60,
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        raise RuntimeError(
            f"Unable to connect to Google Drive: {exc}"
        )


    # --------------------------------------------------------------
    # Google Drive large-file confirmation
    # --------------------------------------------------------------

    confirmation_token = None

    for key, value in response.cookies.items():

        if key.startswith("download_warning"):

            confirmation_token = value
            break


    if confirmation_token:

        url = (
            "https://drive.google.com/uc"
            f"?export=download"
            f"&confirm={confirmation_token}"
            f"&id={file_id}"
        )

        try:

            response = session.get(
                url,
                stream=True,
                timeout=60,
            )

            response.raise_for_status()

        except requests.RequestException as exc:

            raise RuntimeError(
                f"Google Drive download confirmation failed: {exc}"
            )


    # --------------------------------------------------------------
    # Save temporary file
    # --------------------------------------------------------------

    temporary_path = destination.with_suffix(
        destination.suffix + ".download"
    )

    total_size = int(
        response.headers.get(
            "content-length",
            0,
        )
    )

    downloaded = 0

    try:

        with open(
            temporary_path,
            "wb",
        ) as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if not chunk:
                    continue

                file.write(chunk)

                downloaded += len(chunk)

                if total_size:

                    percent = (
                        downloaded / total_size
                    ) * 100

                    print(
                        f"\rProgress: {percent:6.2f}%",
                        end="",
                        flush=True,
                    )

        print()

        # ----------------------------------------------------------
        # Basic validation
        # ----------------------------------------------------------

        if downloaded < 1024:

            raise RuntimeError(
                "Downloaded file is unexpectedly small. "
                "Check Google Drive permissions and file ID."
            )

        # ----------------------------------------------------------
        # Replace old file
        # ----------------------------------------------------------

        temporary_path.replace(destination)

    except Exception:

        if temporary_path.exists():

            temporary_path.unlink()

        raise


    size_mb = downloaded / (
        1024 * 1024
    )

    print(
        f"✅ Model downloaded successfully "
        f"({size_mb:.2f} MB)"
    )


# ======================================================================
# ENSURE MODELS EXIST
# ======================================================================

def ensure_model_files():
    """
    Make sure both large model files exist locally.

    GitHub stores the Python code and configuration.
    Google Drive stores the large model files.
    """

    print("=" * 70)
    print("CHECKING MODEL FILES")
    print("=" * 70)


    # --------------------------------------------------------------
    # Binary model
    # --------------------------------------------------------------

    if BINARY_MODEL_PATH.exists():

        size_mb = (
            BINARY_MODEL_PATH.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"✅ Binary model already exists "
            f"({size_mb:.2f} MB)"
        )

    else:

        print(
            "⚠️ Binary model not found locally."
        )

        print(
            "Downloading from Google Drive..."
        )

        download_from_google_drive(
            BINARY_FILE_ID,
            BINARY_MODEL_PATH,
        )


    # --------------------------------------------------------------
    # Multi-class model
    # --------------------------------------------------------------

    if MULTICLASS_MODEL_PATH.exists():

        size_mb = (
            MULTICLASS_MODEL_PATH.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"✅ Multi-class model already exists "
            f"({size_mb:.2f} MB)"
        )

    else:

        print(
            "⚠️ Multi-class model not found locally."
        )

        print(
            "Downloading from Google Drive..."
        )

        download_from_google_drive(
            MULTICLASS_FILE_ID,
            MULTICLASS_MODEL_PATH,
        )


    print()
    print("✅ All model files are available.")


# ======================================================================
# JSON UTILITIES
# ======================================================================

def load_json(path: Path):
    """
    Load JSON configuration file.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Invalid JSON file: {path}\n{exc}"
        )


# ======================================================================
# FEATURE LIST NORMALIZATION
# ======================================================================

def normalize_feature_list(
    data,
) -> List[str]:
    """
    Normalize feature_names.json.

    Supported formats:

        [
            "feature1",
            "feature2"
        ]

    or:

        {
            "features": [...]
        }

    or:

        {
            "feature_names": [...]
        }
    """

    if isinstance(data, list):

        return [
            str(x)
            for x in data
        ]


    if isinstance(data, dict):

        if "features" in data:

            return [
                str(x)
                for x in data["features"]
            ]


        if "feature_names" in data:

            return [
                str(x)
                for x in data["feature_names"]
            ]


        # Numeric dictionary keys
        try:

            ordered_items = sorted(
                data.items(),
                key=lambda item: int(item[0]),
            )

            return [
                str(value)
                for _, value in ordered_items
            ]

        except Exception:

            pass


    raise ValueError(
        "Unsupported feature_names.json format."
    )


# ======================================================================
# LABEL MAPPING
# ======================================================================

def normalize_label_mapping(
    data,
) -> Dict[str, str]:
    """
    Normalize attack label mapping.
    """

    if not isinstance(data, dict):

        raise ValueError(
            "label_mapping.json must contain "
            "a JSON object."
        )

    return {
        str(key): str(value)
        for key, value in data.items()
    }


# ======================================================================
# LOAD MODELS
# ======================================================================

def load_models():

    global binary_model
    global multiclass_model
    global binary_features
    global multiclass_features
    global label_mapping


    print()
    print("=" * 70)
    print("AI-NIDS — LOADING MODELS")
    print("=" * 70)


    # ==============================================================
    # STEP 1 — ENSURE MODELS
    # ==============================================================

    ensure_model_files()


    # ==============================================================
    # STEP 2 — BINARY MODEL
    # ==============================================================

    print()
    print("Loading Binary Random Forest...")

    try:

        binary_model = joblib.load(
            BINARY_MODEL_PATH
        )

    except Exception as exc:

        raise RuntimeError(
            f"Failed to load binary model:\n{exc}"
        )

    print("✅ Binary model loaded.")


    # ==============================================================
    # STEP 3 — BINARY FEATURES
    # ==============================================================

    print()
    print("Loading binary feature list...")

    binary_feature_data = load_json(
        BINARY_FEATURES_PATH
    )

    binary_features = (
        normalize_feature_list(
            binary_feature_data
        )
    )

    print(
        f"✅ Binary features loaded: "
        f"{len(binary_features)}"
    )


    # ==============================================================
    # STEP 4 — MULTI-CLASS MODEL
    # ==============================================================

    print()
    print("Loading Multi-class Random Forest...")

    try:

        multiclass_model = joblib.load(
            MULTICLASS_MODEL_PATH
        )

    except Exception as exc:

        raise RuntimeError(
            f"Failed to load multi-class model:\n{exc}"
        )

    print("✅ Multi-class model loaded.")


    # ==============================================================
    # STEP 5 — MULTI-CLASS FEATURES
    # ==============================================================

    print()
    print("Loading multi-class feature list...")

    multiclass_feature_data = load_json(
        MULTICLASS_FEATURES_PATH
    )

    multiclass_features = (
        normalize_feature_list(
            multiclass_feature_data
        )
    )

    print(
        f"✅ Multi-class features loaded: "
        f"{len(multiclass_features)}"
    )


    # ==============================================================
    # STEP 6 — LABEL MAPPING
    # ==============================================================

    print()
    print("Loading label mapping...")

    label_mapping_data = load_json(
        MULTICLASS_LABEL_MAPPING_PATH
    )

    label_mapping = (
        normalize_label_mapping(
            label_mapping_data
        )
    )

    print("✅ Label mapping loaded.")


    # ==============================================================
    # STEP 7 — VALIDATION
    # ==============================================================

    if binary_features != multiclass_features:

        raise ValueError(
            "Binary and multi-class feature lists do not match."
        )


    if len(binary_features) != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} "
            f"features but found "
            f"{len(binary_features)}."
        )


    if not label_mapping:

        raise ValueError(
            "Attack label mapping is empty."
        )


    print()
    print("=" * 70)
    print("MODEL LOADING COMPLETE")
    print("=" * 70)

    print(
        f"Binary features:      "
        f"{len(binary_features)}"
    )

    print(
        f"Multi-class features: "
        f"{len(multiclass_features)}"
    )

    print(
        f"Attack classes:       "
        f"{len(label_mapping)}"
    )

    print()
    print("Label mapping:")

    for key, value in label_mapping.items():

        print(
            f"  {key} -> {value}"
        )

    print("=" * 70)


# ======================================================================
# STARTUP
# ======================================================================

@app.on_event("startup")
def startup_event():

    load_models()


# ======================================================================
# FEATURE PREPARATION
# ======================================================================

def prepare_features(
    feature_dict: Dict[str, float]
) -> pd.DataFrame:
    """
    Validate and prepare the 50 model features.
    """

    if not binary_features:

        raise RuntimeError(
            "Models are not loaded."
        )


    expected_features = binary_features

    provided_features = set(
        feature_dict.keys()
    )

    expected_set = set(
        expected_features
    )


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

        except (
            TypeError,
            ValueError,
        ):

            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Non-numeric feature value",
                    "feature": feature,
                    "value": str(value),
                },
            )


        if not np.isfinite(
            numeric_value
        ):

            raise HTTPException(
                status_code=422,
                detail={
                    "error": (
                        "Feature value must be finite"
                    ),
                    "feature": feature,
                    "value": str(value),
                },
            )


        values[feature] = numeric_value


    return pd.DataFrame(
        [values],
        columns=expected_features,
    )


# ======================================================================
# PREDICTION ENGINE
# ======================================================================

def predict_flow(
    feature_dict: Dict[str, float]
):

    dataframe = prepare_features(
        feature_dict
    )


    # ==============================================================
    # STAGE 1 — BINARY
    # ==============================================================

    binary_prediction = (
        binary_model.predict(
            dataframe
        )[0]
    )

    binary_probabilities = (
        binary_model.predict_proba(
            dataframe
        )[0]
    )

    binary_classes = list(
        binary_model.classes_
    )

    binary_class_index = (
        binary_classes.index(
            binary_prediction
        )
    )

    binary_confidence = float(
        binary_probabilities[
            binary_class_index
        ]
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
    # STAGE 2 — MULTI-CLASS
    # ==============================================================

    multiclass_prediction = (
        multiclass_model.predict(
            dataframe
        )[0]
    )

    multiclass_probabilities = (
        multiclass_model.predict_proba(
            dataframe
        )[0]
    )

    multiclass_classes = list(
        multiclass_model.classes_
    )

    multiclass_class_index = (
        multiclass_classes.index(
            multiclass_prediction
        )
    )

    attack_confidence = float(
        multiclass_probabilities[
            multiclass_class_index
        ]
    )


    # --------------------------------------------------------------
    # Numeric prediction → attack name
    # --------------------------------------------------------------

    try:

        attack_key = str(
            int(multiclass_prediction)
        )

    except (
        TypeError,
        ValueError,
    ):

        attack_key = str(
            multiclass_prediction
        )


    attack_type = label_mapping.get(
        attack_key,
        attack_key,
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
# ROOT
# ======================================================================

@app.get("/")
def root():

    return {
        "service": "AI-NIDS",
        "status": "running",
        "version": APP_VERSION,
        "architecture": (
            "Binary Random Forest → "
            "Multi-class Random Forest"
        ),
        "features": len(binary_features),
        "attack_classes": len(label_mapping),
    }


# ======================================================================
# HEALTH
# ======================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    models_ready = (
        binary_model is not None
        and multiclass_model is not None
    )

    return {
        "status": (
            "healthy"
            if models_ready
            else "unhealthy"
        ),
        "service": "AI-NIDS",
        "binary_model_loaded": (
            binary_model is not None
        ),
        "multiclass_model_loaded": (
            multiclass_model is not None
        ),
        "feature_count": len(
            binary_features
        ),
    }


# ======================================================================
# MODEL INFORMATION
# ======================================================================

@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info():

    if (
        binary_model is None
        or multiclass_model is None
    ):

        raise HTTPException(
            status_code=503,
            detail="Models are not loaded.",
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
        "attack_classes": list(
            label_mapping.values()
        ),
        "architecture": (
            "Binary Random Forest → "
            "Multi-class Random Forest"
        ),
    }


# ======================================================================
# PREDICT
# ======================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):

    if (
        binary_model is None
        or multiclass_model is None
    ):

        raise HTTPException(
            status_code=503,
            detail="AI models are not loaded.",
        )


    start_time = time.perf_counter()


    result = predict_flow(
        request.features
    )


    elapsed = (
        time.perf_counter()
        - start_time
    )


    print(
        f"Prediction completed in "
        f"{elapsed * 1000:.2f} ms | "
        f"{result['prediction']} | "
        f"{result['attack_type']}"
    )


    return result


# ======================================================================
# FEATURES
# ======================================================================

@app.get("/features")
def get_features():

    if not binary_features:

        raise HTTPException(
            status_code=503,
            detail=(
                "Feature configuration "
                "not loaded."
            ),
        )


    return {
        "count": len(binary_features),
        "features": binary_features,
    }


# ======================================================================
# ATTACK CLASSES
# ======================================================================

@app.get("/attack-classes")
def get_attack_classes():

    if not label_mapping:

        raise HTTPException(
            status_code=503,
            detail=(
                "Label mapping "
                "not loaded."
            ),
        )


    return {
        "count": len(label_mapping),
        "classes": label_mapping,
    }


# ======================================================================
# SERVER
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
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
