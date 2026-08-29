"""
======================================================================
AI-NIDS — MODEL DOWNLOADER
======================================================================

Downloads trained Random Forest models from Google Drive.

Models:
    1. Binary Random Forest
    2. Multi-class Random Forest

Expected local structure:

ml-service/
│
├── models/
│   ├── binary/
│   │   └── random_forest_binary.joblib
│   │
│   └── multiclass/
│       └── random_forest_multiclass.joblib
│
└── src/
    └── download_models.py

Usage from ml-service:

    python src/download_models.py

The script is also intended to be called automatically
when FastAPI starts.
======================================================================
"""

from pathlib import Path
import os
import sys
import time

import requests


# ======================================================================
# PATH CONFIGURATION
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()

# ml-service/src/download_models.py
#                    ^
#                    |
#              ml-service
BASE_DIR = CURRENT_FILE.parent.parent

MODELS_DIR = BASE_DIR / "models"

BINARY_DIR = MODELS_DIR / "binary"
MULTICLASS_DIR = MODELS_DIR / "multiclass"

BINARY_MODEL_PATH = (
    BINARY_DIR / "random_forest_binary.joblib"
)

MULTICLASS_MODEL_PATH = (
    MULTICLASS_DIR / "random_forest_multiclass.joblib"
)


# ======================================================================
# GOOGLE DRIVE FILE IDs
# ======================================================================

# Environment variables are preferred for deployment.
#
# Local fallback values are provided for your current project.
#
# IMPORTANT:
# If your repository is public, it is better to remove these
# fallback IDs and configure them through environment variables.

BINARY_FILE_ID = os.getenv(
    "BINARY_MODEL_ID",
    "1nT-GtoBTuzwBxnTNYSsm8VliOjPsahj4"
)

MULTICLASS_FILE_ID = os.getenv(
    "MULTICLASS_MODEL_ID",
    "1sNFCe39nc_txzra7p3fScMGvEaMoUVae"
)


# ======================================================================
# CONFIGURATION
# ======================================================================

CHUNK_SIZE = 1024 * 1024  # 1 MB

REQUEST_TIMEOUT = (
    30,
    300,
)


# ======================================================================
# PRINT HELPERS
# ======================================================================

def print_header(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ======================================================================
# GOOGLE DRIVE DOWNLOAD
# ======================================================================

def download_from_google_drive(
    file_id: str,
    destination: Path
) -> bool:
    """
    Download a file from Google Drive.

    Handles the confirmation page that Google Drive can return
    for large files.
    """

    if not file_id:
        raise ValueError(
            "Google Drive file ID is empty."
        )

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    url = (
        "https://drive.google.com/uc"
        f"?export=download&id={file_id}"
    )

    print()
    print(f"📥 Downloading:")
    print(f"   {destination}")

    session = requests.Session()

    try:

        response = session.get(
            url,
            stream=True,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        # --------------------------------------------------------------
        # Google Drive large-file confirmation
        # --------------------------------------------------------------

        confirmation_token = None

        for key, value in response.cookies.items():

            if key.startswith(
                "download_warning"
            ):

                confirmation_token = value
                break

        if confirmation_token:

            print(
                "⚠️ Google Drive confirmation required..."
            )

            confirm_url = (
                "https://drive.google.com/uc"
                "?export=download"
                f"&confirm={confirmation_token}"
                f"&id={file_id}"
            )

            response.close()

            response = session.get(
                confirm_url,
                stream=True,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

        # --------------------------------------------------------------
        # Check response type
        # --------------------------------------------------------------

        content_type = (
            response.headers
            .get("content-type", "")
            .lower()
        )

        if (
            "text/html" in content_type
            and not content_type.startswith(
                "application/octet-stream"
            )
        ):

            # Google Drive may return an HTML page
            # instead of the actual file.

            first_chunk = next(
                response.iter_content(
                    chunk_size=1024
                ),
                b""
            )

            if (
                b"<html" in first_chunk.lower()
                or b"<!doctype" in first_chunk.lower()
            ):

                raise RuntimeError(
                    "Google Drive returned an HTML page "
                    "instead of the model file. "
                    "Check that the file is shared as "
                    "'Anyone with the link - Viewer'."
                )

        # --------------------------------------------------------------
        # File size
        # --------------------------------------------------------------

        total_size = int(
            response.headers.get(
                "content-length",
                0
            )
        )

        if total_size > 0:

            print(
                f"📦 File size: "
                f"{total_size / (1024 * 1024):.2f} MB"
            )

        # --------------------------------------------------------------
        # Temporary file
        # --------------------------------------------------------------

        temporary_path = destination.with_suffix(
            destination.suffix + ".download"
        )

        downloaded = 0
        start_time = time.time()

        with open(
            temporary_path,
            "wb"
        ) as file:

            for chunk in response.iter_content(
                chunk_size=CHUNK_SIZE
            ):

                if not chunk:
                    continue

                file.write(chunk)

                downloaded += len(chunk)

                # ------------------------------------------------------
                # Progress
                # ------------------------------------------------------

                if total_size > 0:

                    percentage = (
                        downloaded /
                        total_size *
                        100
                    )

                    print(
                        f"\r   Progress: "
                        f"{percentage:6.2f}% "
                        f"({downloaded / (1024 * 1024):.1f} MB)",
                        end=""
                    )

                else:

                    print(
                        f"\r   Downloaded: "
                        f"{downloaded / (1024 * 1024):.1f} MB",
                        end=""
                    )

        response.close()

        print()

        # --------------------------------------------------------------
        # Validate download
        # --------------------------------------------------------------

        if downloaded == 0:

            if temporary_path.exists():
                temporary_path.unlink()

            raise RuntimeError(
                "Downloaded file is empty."
            )

        # --------------------------------------------------------------
        # Move temporary file to final destination
        # --------------------------------------------------------------

        temporary_path.replace(
            destination
        )

        elapsed = (
            time.time() - start_time
        )

        print(
            f"✅ Download completed:"
        )

        print(
            f"   File: {destination}"
        )

        print(
            f"   Size: "
            f"{downloaded / (1024 * 1024):.2f} MB"
        )

        print(
            f"   Time: {elapsed:.1f} seconds"
        )

        return True

    except requests.RequestException as error:

        print()
        print(
            "❌ Network error while downloading model:"
        )

        print(
            f"   {error}"
        )

        return False

    except Exception as error:

        print()
        print(
            "❌ Model download failed:"
        )

        print(
            f"   {error}"
        )

        return False


# ======================================================================
# MODEL VALIDATION
# ======================================================================

def validate_model_file(
    path: Path,
    minimum_size_mb: float = 1.0
) -> bool:
    """
    Basic validation that the downloaded file exists
    and has a reasonable size.
    """

    if not path.exists():

        return False

    if not path.is_file():

        return False

    size_mb = (
        path.stat().st_size /
        (1024 * 1024)
    )

    if size_mb < minimum_size_mb:

        print(
            f"⚠️ Model file appears too small: "
            f"{size_mb:.2f} MB"
        )

        return False

    return True


# ======================================================================
# DOWNLOAD ALL MODELS
# ======================================================================

def download_all_models() -> bool:

    print_header(
        "AI-NIDS — MODEL DOWNLOAD MANAGER"
    )

    print(
        f"Models directory:\n"
        f"  {MODELS_DIR}"
    )

    print()
    print(
        "Binary model:"
    )

    print(
        f"  {BINARY_MODEL_PATH}"
    )

    print()
    print(
        "Multi-class model:"
    )

    print(
        f"  {MULTICLASS_MODEL_PATH}"
    )

    # ==================================================================
    # BINARY MODEL
    # ==================================================================

    print_header(
        "1. BINARY RANDOM FOREST"
    )

    if validate_model_file(
        BINARY_MODEL_PATH
    ):

        size_mb = (
            BINARY_MODEL_PATH.stat().st_size /
            (1024 * 1024)
        )

        print(
            f"✅ Binary model already exists."
        )

        print(
            f"   Size: {size_mb:.2f} MB"
        )

    else:

        success = download_from_google_drive(
            BINARY_FILE_ID,
            BINARY_MODEL_PATH
        )

        if not success:

            print(
                "❌ Binary model download failed."
            )

            return False

    # ==================================================================
    # MULTI-CLASS MODEL
    # ==================================================================

    print_header(
        "2. MULTI-CLASS RANDOM FOREST"
    )

    if validate_model_file(
        MULTICLASS_MODEL_PATH
    ):

        size_mb = (
            MULTICLASS_MODEL_PATH.stat().st_size /
            (1024 * 1024)
        )

        print(
            "✅ Multi-class model already exists."
        )

        print(
            f"   Size: {size_mb:.2f} MB"
        )

    else:

        success = download_from_google_drive(
            MULTICLASS_FILE_ID,
            MULTICLASS_MODEL_PATH
        )

        if not success:

            print(
                "❌ Multi-class model download failed."
            )

            return False

    # ==================================================================
    # FINAL VALIDATION
    # ==================================================================

    print_header(
        "FINAL MODEL CHECK"
    )

    binary_ready = validate_model_file(
        BINARY_MODEL_PATH
    )

    multiclass_ready = validate_model_file(
        MULTICLASS_MODEL_PATH
    )

    print(
        f"Binary model:      "
        f"{'✅ READY' if binary_ready else '❌ NOT READY'}"
    )

    print(
        f"Multi-class model: "
        f"{'✅ READY' if multiclass_ready else '❌ NOT READY'}"
    )

    if not binary_ready or not multiclass_ready:

        print()
        print(
            "❌ AI-NIDS model setup failed."
        )

        return False

    print()
    print(
        "🎉 All AI-NIDS models are ready."
    )

    print(
        "FastAPI can now load the trained models."
    )

    return True


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":

    success = download_all_models()

    if success:

        print()
        print("=" * 70)
        print("✅ MODEL DOWNLOAD COMPLETED SUCCESSFULLY")
        print("=" * 70)

        sys.exit(0)

    else:

        print()
        print("=" * 70)
        print("❌ MODEL DOWNLOAD FAILED")
        print("=" * 70)

        sys.exit(1)
