"""
Download trained models from Google Drive
"""

import requests
import os
from pathlib import Path
import sys

# ⚠️ REPLACE THESE WITH YOUR ACTUAL FILE IDs
BINARY_FILE_ID = "1nT-GtoBTuzwBxnTNYSsm8VliOjPsahj4"        # e.g., "1ABC123XYZ789"
MULTICLASS_FILE_ID = "1sNFCe39nc_txzra7p3fScMGvEaMoUVae"  # e.g., "2DEF456UVW012"

# Local paths
BINARY_MODEL_PATH = "models/binary/random_forest_binary.joblib"
MULTICLASS_MODEL_PATH = "models/multiclass/random_forest_multiclass.joblib"

def download_from_google_drive(file_id, destination):
    """Download a file from Google Drive"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print(f"📥 Downloading {destination}...")
    
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Handle large file confirmation
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(url, stream=True)
            break
    
    # Save file
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    total_size = int(response.headers.get('content-length', 0))
    
    downloaded = 0
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"  Progress: {percent:.1f}%", end='\r')
    
    print(f"\n✅ Downloaded: {destination} ({downloaded / (1024*1024):.1f} MB)")
    return True

def download_all_models():
    """Download all model files"""
    print("=" * 60)
    print("📦 AI-NIDS - Downloading Models from Google Drive")
    print("=" * 60)
    print("📁 Source: https://drive.google.com/drive/folders/184lJLXdQmOfhn-X1xE59UaROMK9YGKEg")
    print()
    
    # Download binary model
    if not Path(BINARY_MODEL_PATH).exists():
        download_from_google_drive(BINARY_FILE_ID, BINARY_MODEL_PATH)
    else:
        print(f"✅ {BINARY_MODEL_PATH} already exists")
    
    # Download multiclass model
    if not Path(MULTICLASS_MODEL_PATH).exists():
        download_from_google_drive(MULTICLASS_FILE_ID, MULTICLASS_MODEL_PATH)
    else:
        print(f"✅ {MULTICLASS_MODEL_PATH} already exists")
    
    print("=" * 60)
    print("✅ All models ready!")
    print("=" * 60)

if __name__ == "__main__":
    download_all_models()
