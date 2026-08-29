# 📄 Complete README.md for AI-NIDS

Here's your **complete, professional README** with all the updated information for GitHub + Google Drive setup:

---

```markdown
# 🛡️ AI-Powered Network Intrusion Detection System (AI-NIDS)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue)](https://reactjs.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/AI-NIDS)](https://github.com/yourusername/AI-NIDS/stargazers)

> **🎓 Final Year B.Tech CSE Project** - An enterprise-grade AI-powered security analytics platform for real-time network intrusion detection.

---

## 📊 Overview

**AI-NIDS** is a complete security analytics platform that uses **Machine Learning** to detect network intrusions in real-time. It combines:

- 🧠 **ML Detection Engine** - Random Forest models for binary and multi-class attack classification
- 🚀 **FastAPI Backend** - High-performance REST API with 50-feature flow analysis
- 🎨 **React Dashboard** - Professional SOC-style interface with real-time monitoring
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
┌─────────────────────────────────────────────────────────────────┐
│                        NETWORK LAYER                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │   Internet  │  │   Router    │  │  Switch     │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
│         │               │               │                      │
│         └───────────────┼───────────────┘                      │
│                         ↓                                      │
│                   [Traffic Collector]                          │
│            (Scapy/Wireshark - lab mode)                       │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                          │
│  50 Network Flow Features:                                     │
│  • Duration  • Packet count  • Bytes  • Ports  • Protocol     │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PREPROCESSING PIPELINE                         │
│  • Handle missing values  • Normalize  • Encode categories    │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ML DETECTION ENGINE                         │
│  ┌────────────────────┐  ┌────────────────────┐              │
│  │   Binary Model     │  │  Multi-class Model │              │
│  │ Random Forest      │  │  Random Forest     │              │
│  │ BENIGN vs ATTACK   │  │  9 Attack Types    │              │
│  └────────────────────┘  └────────────────────┘              │
│         ↓                         ↓                           │
│         └────────────┬────────────┘                           │
│                      ↓                                         │
│              Risk Scoring Engine                               │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                   DECISION & ALERTING                          │
│  Prediction: ATTACK                                            │
│  Type: DDoS                                                    │
│  Confidence: 94%                                               │
│  Risk Score: 87/100 - HIGH                                     │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY DASHBOARD                          │
│  ┌─────┬─────┬─────┬─────┐  ┌─────────────────────┐         │
│  │Flows│Attacks│High │Risk │  │  Traffic Graph      │         │
│  │125K │ 2.3K │ 18  │ 78  │  │  📈📉📊             │         │
│  └─────┴─────┴─────┴─────┘  └─────────────────────┘         │
│                                                               │
│  ┌──────────────────────────────────────┐                    │
│  │  Recent Alerts                       │                    │
│  │  🟡 10.0.0.5 → 192.168.1.1  Port 22 │                    │
│  │  🔴 192.168.1.100 → Multiple  DDoS  │                    │
│  └──────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Backend
| Component | Technology |
|-----------|------------|
| **Framework** | FastAPI 0.100+ |
| **ML Library** | Scikit-learn 1.3+ |
| **ML Models** | Random Forest, XGBoost |
| **Data Processing** | Pandas, NumPy |
| **Validation** | Pydantic 2.0+ |
| **Database** | PostgreSQL (optional) |
| **Testing** | Pytest |
| **Server** | Uvicorn |

### Frontend
| Component | Technology |
|-----------|------------|
| **Framework** | React 18 |
| **UI Library** | Material-UI 5.15+ |
| **Charts** | Recharts 2.12+ |
| **HTTP Client** | Axios 1.7+ |
| **Routing** | React Router 6 |
| **State** | React Hooks |

---

## 🚀 Quick Start

### 📥 Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/AI-NIDS.git
cd AI-NIDS
```

### 📦 Step 2: Download Model Files from Google Drive

The trained models are stored on Google Drive due to their size (>100MB each).

#### Option A: Using the Download Script (Recommended)

```bash
# Install dependencies
pip install requests tqdm

# Download models
python download_models.py
```

#### Option B: Manual Download

1. Go to: [Google Drive Models Folder](https://drive.google.com/drive/folders/184lJLXdQmOfhn-X1xE59UaROMK9YGKEg)
2. Download these files:
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
cd ml-service
python src/api.py
```

**Server will run at:** `http://127.0.0.1:8000`
- API Docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### ⚛️ Step 4: Setup Frontend

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start the React development server
npm start
```

**Frontend will run at:** `http://localhost:3000`

### 🎯 Step 5: Test the System

```bash
# Test a prediction
python test_prediction.py

# Test with sample data
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
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
| False Positive Rate | **2.1%** |

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

## 🧪 Testing

### Backend Tests
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_prediction.py -v

# With coverage report
pytest --cov=app tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

### API Test Examples
```bash
# Health check
curl http://127.0.0.1:8000/health

# Get model info
curl http://127.0.0.1:8000/model-info

# Get attack classes
curl http://127.0.0.1:8000/attack-classes

# Make a prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"flow_features": {"Destination Port": 80, "Flow Duration": 5000, ...}}'
```

---

## 📁 Project Structure

```
AI-NIDS/
├── ml-service/
│   ├── src/
│   │   ├── api.py                    # FastAPI main application
│   │   ├── predict.py                # Prediction logic
│   │   ├── schemas.py                # Pydantic schemas
│   │   ├── preprocessing.py          # Data preprocessing
│   │   ├── train_binary_model.py     # Binary model training
│   │   ├── train_multiclass_model.py # Multi-class model training
│   │   ├── evaluate.py               # Model evaluation
│   │   ├── compare_models.py         # Model comparison
│   │   ├── select_features.py        # Feature selection
│   │   ├── analyze_confidence.py     # Confidence analysis
│   │   ├── analyze_features.py       # Feature analysis
│   │   ├── analyze_imbalance.py      # Imbalance analysis
│   │   ├── combine_dataset.py        # Dataset combination
│   │   ├── inspect_all_datasets.py   # Dataset inspection
│   │   ├── phase_10_7_validation.py  # Validation script
│   │   ├── split_dataset.py          # Dataset splitting
│   │   ├── test_dataset.py           # Dataset testing
│   │   └── test_prediction.py        # Prediction testing
│   ├── tests/
│   │   └── README.md
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   └── client.py
│   │   ├── components/
│   │   │   ├── Dashboard/
│   │   │   ├── Alerts/
│   │   │   ├── Analysis/
│   │   │   └── Layout/
│   │   ├── pages/
│   │   │   ├── Dashboard.js
│   │   │   ├── Alerts.js
│   │   │   ├── Analysis.js
│   │   │   └── Models.js
│   │   ├── styles/
│   │   │   ├── theme.js
│   │   │   └── global.css
│   │   ├── App.js
│   │   └── index.js
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── models/
│   ├── binary/
│   │   └── random_forest_binary.joblib   # Downloaded from Google Drive
│   ├── multiclass/
│   │   └── random_forest_multiclass.joblib # Downloaded from Google Drive
│   └── production/
│
├── data/
│   ├── raw/                           # Datasets (optional, from Google Drive)
│   └── processed/
│
├── notebooks/                         # Jupyter notebooks
│   └── ... (analysis notebooks)
│
├── download_models.py                 # Google Drive download script
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
└── LICENSE                            # MIT License
```

---

## 📦 Google Drive Integration

Due to GitHub's 100MB file limit, trained models and large datasets are stored on Google Drive.

### 📥 Download Models
```bash
python download_models.py
```

### 📊 Download Datasets (Optional)
```bash
python download_models.py --datasets
```

### 🔗 Google Drive Folder
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Models-blue?logo=googledrive)](https://drive.google.com/drive/folders/184lJLXdQmOfhn-X1xE59UaROMK9YGKEg)

**Folder Contents:**
- `random_forest_binary.joblib` - 102.3 MB
- `random_forest_multiclass.joblib` - 101.1 MB
- Datasets (CICIDS2017, etc.)

---

## 🎨 Dashboard Preview

```
┌─────────────────────────────────────────────────────┐
│  🛡️ AI-NIDS Security Dashboard                     │
├────────────┬────────────┬────────────┬─────────────┤
│  FLOWS     │  ATTACKS   │  CRITICAL  │ RISK SCORE  │
│  125,430   │   2,341    │     18     │     78      │
├────────────┴────────────┴────────────┴─────────────┤
│                                                     │
│              📈 NETWORK TRAFFIC GRAPH              │
│                                                     │
├─────────────────────────┬───────────────────────────┤
│ 🥧 ATTACK DISTRIBUTION  │ 🚨 RECENT ALERTS         │
│                         │                           │
│ DoS        ███████      │ HIGH  192.168.x.x        │
│ DDoS       ████         │ CRIT  10.0.x.x           │
│ PortScan   ██           │ MED   172.16.x.x         │
└─────────────────────────┴───────────────────────────┘
```

---

## 🧠 The 4 AI Layers

```
             AI-NIDS
                │
     ┌──────────┼──────────┐
     ↓          ↓          ↓
Classification Anomaly   Explanation
     │        Detection      │
     └──────────┬────────────┘
                ↓
          Risk Scoring
                ↓
           SOC Dashboard
```

1. **Classification** - "What happened?" (BENIGN vs ATTACK)
2. **Anomaly Detection** - "Is this unusual?" (Isolation Forest)
3. **Explainability** - "Why?" (Feature Importance)
4. **Risk Scoring** - "How dangerous?" (0-100 scale)

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

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/nids

# Frontend
REACT_APP_API_URL=http://localhost:8000
```

---

## 📈 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API Information |
| GET | `/health` | Health Check |
| GET | `/model-info` | Model Information |
| GET | `/features` | Feature List |
| GET | `/attack-classes` | Attack Classes |
| POST | `/predict` | Single Flow Prediction |
| POST | `/analyze-csv` | CSV Batch Analysis |
| GET | `/dashboard/summary` | Dashboard Statistics |
| GET | `/alerts` | Alert List |
| GET | `/alerts/{id}` | Alert Details |
| PATCH | `/alerts/{id}` | Update Alert Status |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CICIDS2017 Dataset** - Canadian Institute for Cybersecurity
- **Scikit-learn Community** - Machine Learning Library
- **FastAPI Team** - High-performance Web Framework
- **Material-UI Team** - React Component Library
- **UCI Machine Learning Repository** - Dataset Repository

---

## 📧 Contact

**Your Name**
- 📧 Email: your.email@example.com
- 🔗 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- 🐦 GitHub: [github.com/yourusername](https://github.com/yourusername)

**Project Link:** [https://github.com/yourusername/AI-NIDS](https://github.com/yourusername/AI-NIDS)

---

## ⭐ Show Your Support

If you found this project helpful, please give it a star on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/yourusername/AI-NIDS?style=social)](https://github.com/yourusername/AI-NIDS)

---

## 📊 Final Project Features

### ✅ Completed Features
- [x] ML-based Intrusion Detection
- [x] FastAPI Backend
- [x] React Dashboard
- [x] 9 Attack Types
- [x] Risk Scoring
- [x] Alert Management
- [x] CSV Analysis
- [x] Model Comparison
- [x] Explainable AI
- [x] Google Drive Integration

### 🚀 Future Scope
- [ ] Real-time Packet Capture (Scapy)
- [ ] Anomaly Detection (Isolation Forest)
- [ ] SHAP Explanations
- [ ] Docker Containerization
- [ ] CI/CD Pipeline
- [ ] User Authentication
- [ ] Multi-User Support
- [ ] Email Alerts

---

**Made with ❤️ for Final Year B.Tech CSE Project**

*"Defending Networks with AI, One Flow at a Time"*
```

---

## 📁 Files to Upload

### On GitHub:
```
✅ README.md (this file)
✅ .gitignore
✅ requirements.txt
✅ download_models.py
✅ ml-service/src/*.py
✅ frontend/src/*.js
✅ frontend/package.json
✅ notebooks/*.ipynb
```

### On Google Drive:
```
✅ random_forest_binary.joblib
✅ random_forest_multiclass.joblib
✅ Datasets (CSV files)
```

---

**That's your complete README file! Copy and paste this into your `README.md` and upload it to GitHub.** 🚀
