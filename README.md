# 🛡️ AI-Powered Network Intrusion Detection System (AI-NIDS)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **🎓 Final Year B.Tech CSE Project** - An enterprise-grade AI-powered security analytics platform for real-time network intrusion detection.

---

## 🌐 Live Demo

| Component | URL | Status |
|-----------|-----|--------|
| **Frontend Dashboard** | http://localhost:8501 | 🟢 Live |
| **API Documentation** | http://127.0.0.1:8000/docs | 🟢 Live |
| **ReDoc API** | http://127.0.0.1:8000/redoc | 🟢 Live |

---

## 📊 Overview

**AI-NIDS** is a complete security analytics platform that uses **Machine Learning** to detect network intrusions in real-time. It combines:

- 🧠 **ML Detection Engine** - Two-stage Random Forest architecture (Binary → Multi-class)
- 🚀 **FastAPI Backend** - High-performance REST API with 50-feature flow analysis
- 🎨 **Streamlit Dashboard** - Professional SOC-style interface with real-time monitoring
- 📊 **Explainable AI** - Feature importance and attack explanations
- 🔗 **Google Drive Integration** - Automatic model downloads for large files

### 🎯 Key Features

#### Core Capabilities
- ✅ **Binary Classification** (BENIGN vs ATTACK) - 98.2% accuracy
- ✅ **Multi-class Attack Detection** - 9 attack types identified
- ✅ **Real-time Flow Analysis** - <100ms per prediction
- ✅ **CSV Batch Analysis** - Upload and analyze entire datasets
- ✅ **Risk Scoring** - 0-100 with severity levels
- ✅ **Explainable Predictions** - Feature importance and confidence scores

#### Attack Types Detected
| ID | Attack Type | Description | Severity |
|----|-------------|-------------|----------|
| 0 | BENIGN | Normal network traffic | 🟢 Safe |
| 1 | DoS | Denial of Service | 🔴 Critical |
| 2 | DDoS | Distributed Denial of Service | 🔴 Critical |
| 3 | PortScan | Port scanning activity | 🟠 High |
| 4 | BruteForce | Password brute force attacks | 🟠 High |
| 5 | WebAttack | Web application attacks | 🟠 High |
| 6 | Botnet | Botnet activity | 🔴 Critical |
| 7 | Infiltration | Network infiltration | 🔴 Critical |
| 8 | Heartbleed | Heartbleed vulnerability exploit | 🔴 Critical |

---

## 🏗️ System Architecture

```
Network Flow
     ↓
50 Features
     ↓
Binary Random Forest
     ↓
BENIGN / ATTACK
     ↓
Multi-class Random Forest
     ↓
Attack Type
```

---

## 🛠️ Technology Stack

### Backend
| Component | Technology |
|-----------|------------|
| **Framework** | FastAPI 0.100+ |
| **ML Library** | Scikit-learn 1.3+ |
| **ML Models** | Random Forest (Binary + Multi-class) |
| **Data Processing** | Pandas, NumPy |
| **Validation** | Pydantic 2.0+ |
| **Server** | Uvicorn |

### Frontend
| Component | Technology |
|-----------|------------|
| **Framework** | Streamlit 1.28+ |
| **UI Library** | Streamlit Components |
| **Charts** | Streamlit, Plotly |
| **HTTP Client** | Requests |

---

## 🚀 Quick Start

### 📥 Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/AI-NIDS.git
cd AI-NIDS
```

### 📦 Step 2: Download Model Files from Google Drive

The trained models are stored on Google Drive due to their size (>100MB each).

**Google Drive Folder:** https://drive.google.com/drive/folders/184lJLXdQmOfhn-X1xE59UaROMK9YGKEg

Download these files:
- `random_forest_binary.joblib` → `models/binary/`
- `random_forest_multiclass.joblib` → `models/multiclass/`

### 🐍 Step 3: Setup Backend

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
python api.py
```

**Server will run at:** `http://127.0.0.1:8000`
- API Docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### 🎨 Step 4: Setup Frontend (Streamlit)

```bash
# Install Streamlit
pip install streamlit

# Run the Streamlit app
streamlit run app.py
```

**Frontend will run at:** `http://localhost:8501`

### 🎯 Step 5: Test the System

```bash
# Test a prediction
python test_prediction.py

# Test with curl
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"flow_features": {"Destination Port": 80, "Flow Duration": 5000, ...}}'
```

---

## 📊 Model Performance

### Binary Classification (BENIGN vs ATTACK)

| Metric | Score |
|--------|-------|
| Accuracy | **98.2%** |
| Precision | **97.4%** |
| Recall | **96.9%** |
| F1-Score | **97.1%** |
| ROC-AUC | **0.99** |

### Multi-class Classification (9 Attack Types)

| Attack Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| BENIGN | 0.98 | 0.99 | 0.98 |
| DoS | 0.96 | 0.95 | 0.95 |
| DDoS | 0.97 | 0.96 | 0.96 |
| PortScan | 0.94 | 0.93 | 0.93 |
| BruteForce | 0.92 | 0.91 | 0.91 |
| WebAttack | 0.89 | 0.88 | 0.88 |
| Botnet | 0.91 | 0.90 | 0.90 |
| Infiltration | 0.87 | 0.85 | 0.86 |
| Heartbleed | 0.90 | 0.88 | 0.89 |

**Average F1-Score:** 0.93

---

## 📁 Project Structure

```
AI-NIDS/
├── api.py                        # FastAPI main application
├── app.py                        # Streamlit frontend
├── predict.py                    # Prediction logic
├── schemas.py                    # Pydantic schemas
├── preprocessing.py              # Data preprocessing
├── train_binary_model.py         # Binary model training
├── train_multiclass_model.py     # Multi-class model training
├── evaluate.py                   # Model evaluation
├── compare_models.py             # Model comparison
├── select_features.py            # Feature selection
├── analyze_confidence.py         # Confidence analysis
├── analyze_features.py           # Feature analysis
├── analyze_imbalance.py          # Imbalance analysis
├── combine_dataset.py            # Dataset combination
├── inspect_all_datasets.py       # Dataset inspection
├── phase_10_7_validation.py      # Validation script
├── split_dataset.py              # Dataset splitting
├── test_dataset.py               # Dataset testing
├── test_prediction.py            # Prediction testing
├── feature_names.json            # Feature list
├── label_mapping.json            # Attack class mapping
├── metrics.json                  # Model performance metrics
├── models/
│   ├── binary/
│   │   └── random_forest_binary.joblib
│   └── multiclass/
│       └── random_forest_multiclass.joblib
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## 📦 Google Drive Integration

Due to GitHub's 100MB file limit, trained models and large datasets are stored on Google Drive.

### 🔗 Google Drive Folder
https://drive.google.com/drive/folders/184lJLXdQmOfhn-X1xE59UaROMK9YGKEg

**Folder Contents:**
| File | Size |
|------|------|
| `random_forest_binary.joblib` | 102.3 MB |
| `random_forest_multiclass.joblib` | 101.1 MB |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 73.3 MB |
| `Monday-WorkingHours.pcap_ISCX.csv` | 168.7 MB |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 49.6 MB |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 128.8 MB |

---

## 🧪 Testing

### API Tests
```bash
# Health check
curl http://127.0.0.1:8000/health

# Get model info
curl http://127.0.0.1:8000/model-info

# Get attack classes
curl http://127.0.0.1:8000/attack-classes

# Get features list
curl http://127.0.0.1:8000/features

# Make a prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"flow_features": {"Destination Port": 443, "Flow Duration": 120500, ...}}'
```

---

## 📈 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API Information |
| GET | `/health` | Health Check |
| GET | `/model-info` | Model Information |
| GET | `/features` | Feature List (50 features) |
| GET | `/attack-classes` | Attack Classes (9 types) |
| POST | `/predict` | Single Flow Prediction |

### Prediction Response Example
```json
{
  "prediction": "ATTACK",
  "attack_type": "DDoS",
  "confidence": 0.94,
  "risk_score": 91,
  "severity": "CRITICAL"
}
```

---

## 🎯 Sample Prediction Request

```json
{
  "flow_features": {
    "Destination Port": 80,
    "Flow Duration": 5000,
    "Total Fwd Packets": 15000,
    "Total Length of Fwd Packets": 6000000,
    "Fwd Packet Length Max": 1500,
    "Fwd Packet Length Min": 64,
    "Fwd Packet Length Mean": 400,
    "Fwd Packet Length Std": 450.5,
    "Bwd Packet Length Max": 1500,
    "Bwd Packet Length Min": 64,
    "Bwd Packet Length Mean": 400,
    "Flow Bytes/s": 1200000.0,
    "Flow Packets/s": 3000.0,
    "Flow IAT Mean": 0.001,
    "Flow IAT Std": 0.001,
    "Flow IAT Max": 0.005,
    "Flow IAT Min": 0.0001,
    "Fwd IAT Mean": 0.001,
    "Fwd IAT Std": 0.001,
    "Fwd IAT Min": 0.0001,
    "Bwd IAT Total": 0.5,
    "Bwd IAT Mean": 0.001,
    "Bwd IAT Std": 0.001,
    "Bwd IAT Max": 0.005,
    "Bwd IAT Min": 0.0001,
    "Fwd PSH Flags": 0,
    "Fwd URG Flags": 0,
    "Fwd Header Length": 120,
    "Bwd Header Length": 120,
    "Bwd Packets/s": 1500,
    "Min Packet Length": 64,
    "Max Packet Length": 1500,
    "Packet Length Mean": 400,
    "Packet Length Variance": 723908.12,
    "FIN Flag Count": 0,
    "RST Flag Count": 0,
    "PSH Flag Count": 0,
    "ACK Flag Count": 15000,
    "URG Flag Count": 0,
    "Down/Up Ratio": 1.0,
    "Init_Win_bytes_forward": 65535,
    "Init_Win_bytes_backward": 65535,
    "act_data_pkt_fwd": 15000,
    "min_seg_size_forward": 0,
    "Active Mean": 0.001,
    "Active Std": 0.001,
    "Active Max": 0.005,
    "Active Min": 0.0001,
    "Idle Mean": 0.001,
    "Idle Std": 0.001
  }
}
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Paths
BINARY_MODEL_PATH=models/binary/random_forest_binary.joblib
MULTICLASS_MODEL_PATH=models/multiclass/random_forest_multiclass.joblib

# Frontend
STREAMLIT_SERVER_PORT=8501
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **CICIDS2017 Dataset** - Canadian Institute for Cybersecurity
- **Scikit-learn Community** - Machine Learning Library
- **FastAPI Team** - High-performance Web Framework
- **Streamlit Team** - Interactive Data Apps

---

## 📧 Contact

**Your Name**
- 📧 Email: your.email@example.com
- 🔗 LinkedIn: linkedin.com/in/yourprofile
- 🐦 GitHub: github.com/yourusername

**Project Link:** https://github.com/yourusername/AI-NIDS

---

## ⭐ Show Your Support

If you found this project helpful, please give it a star on GitHub!

---

## 📊 Final Project Features

### ✅ Completed Features
- [x] ML-based Intrusion Detection (Binary + Multi-class)
- [x] FastAPI Backend with 50 features
- [x] Streamlit Dashboard
- [x] 9 Attack Types
- [x] Risk Scoring (0-100)
- [x] Alert Management
- [x] Explainable AI
- [x] Google Drive Integration
- [x] OpenAPI Documentation

### 🚀 Future Scope
- [ ] Real-time Packet Capture (Scapy)
- [ ] Anomaly Detection (Isolation Forest)
- [ ] SHAP Explanations
- [ ] Docker Containerization
- [ ] CI/CD Pipeline
- [ ] User Authentication
- [ ] Email Alerts

---

## 🖥️ Running the Project

### Terminal 1 - Backend
```bash
cd AI-NIDS
python api.py
```
→ API running at http://127.0.0.1:8000

### Terminal 2 - Frontend
```bash
cd AI-NIDS
streamlit run app.py
```
→ Dashboard running at http://localhost:8501

---

**Made with ❤️ for Final Year B.Tech CSE Project**

*"Defending Networks with AI, One Flow at a Time"*
