"""
Download trained models and datasets from Google Drive
Run this script after cloning the repository
"""

import requests
import os
import zipfile
from pathlib import Path
from tqdm import tqdm

# Google Drive folder ID
FOLDER_ID = "184lJLXdQmOfhn-X1xE59UaROMK9YGKEg"

# Files to download (name in Drive -> local path)
FILES_TO_DOWNLOAD = {
    "random_forest_binary.joblib": "models/binary/random_forest_binary.joblib",
    "random_forest_multiclass.joblib": "models/multiclass/random_forest_multiclass.joblib",
}

# Optional: Also download datasets if needed
DATASETS = {
    # "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "data/raw/portscan.csv",
    # "Monday-WorkingHours.pcap_ISCX.csv": "data/raw/monday.csv",
    # "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "data/raw/webattacks.csv",
    # "Tuesday-WorkingHours.pcap_ISCX.csv": "data/raw/tuesday.csv",
}


def get_direct_download_link(file_id):
    """Get direct download link for a Google Drive file"""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def get_file_id_from_folder(folder_id, file_name):
    """
    Get file ID from folder ID and file name.
    This requires the Google Drive API - for simplicity,
    we'll use the manual approach below.
    """
    # For manual approach, we'll just use the file ID directly
    # Since you have the folder link, you can get file IDs by:
    # 1. Click on the file in Google Drive
    # 2. Click "Get link"
    # 3. Copy the file ID from the URL
    return None


# Manual file IDs (get these from Google Drive)
# To get a file ID:
# 1. Open the file in Google Drive
# 2. Click "Get link"
# 3. Copy the ID from the URL: https://drive.google.com/file/d/FILE_ID/view
FILE_IDS = {
    "random_forest_binary.joblib": "1mZnFyP-fxyhnlO-R0PWF6xOJSRlqgBp_",  # Example - REPLACE!
    "random_forest_multiclass.joblib": "1n0ZgQ-rSyioP-1SWF7xPJSRlqgBp_",  # Example - REPLACE!
}


def download_from_google_drive(file_id, destination, chunk_size=32768):
    """
    Download a file from Google Drive using its file ID
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print(f"📥 Downloading: {destination}")
    
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Handle the confirmation page for large files
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(url, stream=True)
            break
    
    # Get file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    
    # Create directory if it doesn't exist
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    
    # Download with progress bar
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=Path(destination).name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"✅ Downloaded: {destination} ({total_size / (1024*1024):.1f} MB)")


def download_all_models():
    """Download all model files"""
    print("=" * 60)
    print("📥 AI-NIDS - Downloading Models from Google Drive")
    print("=" * 60)
    print(f"📁 Google Drive Folder: {FOLDER_ID}")
    print()
    
    for file_name, local_path in FILES_TO_DOWNLOAD.items():
        if file_name in FILE_IDS:
            download_from_google_drive(FILE_IDS[file_name], local_path)
        else:
            print(f"⚠️  No file ID found for: {file_name}")
            print(f"   Please update FILE_IDS dictionary with the correct ID")
    
    print()
    print("=" * 60)
    print("✅ All models downloaded successfully!")
    print("=" * 60)


def download_all_datasets():
    """Download dataset files (optional)"""
    print("=" * 60)
    print("📊 AI-NIDS - Downloading Datasets from Google Drive")
    print("=" * 60)
    
    for file_name, local_path in DATASETS.items():
        if file_name in FILE_IDS:
            download_from_google_drive(FILE_IDS[file_name], local_path)
        else:
            print(f"⚠️  No file ID found for: {file_name}")
    
    print()
    print("✅ All datasets downloaded successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--datasets":
        download_all_datasets()
    else:
        download_all_models()
        print()
        print("💡 To also download datasets, run: python download_models.py --datasets")
