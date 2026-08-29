# 🛡️ AI-Powered Network Intrusion Detection System (AI-NIDS)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue)](https://reactjs.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **🎓 Final Year B.Tech CSE Project** - An enterprise-grade AI-powered security analytics platform for real-time network intrusion detection.

---

## 🌐 Live Demo

| Component | URL | Status |
|-----------|-----|--------|
| **Frontend Dashboard** | [http://localhost:8501](http://localhost:8501) | 🟢 Live |
| **API Documentation** | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | 🟢 Live |
| **ReDoc API** | [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) | 🟢 Live |

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
┌─────────────────────────────────────────────────────────────────┐
│ NETWORK LAYER │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ Internet │ │ Router │ │ Switch │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ │
│ │ │ │ │
│ └───────────────┼───────────────┘ │
│ ↓ │
│ [Traffic Collector] │
│ (Scapy/Wireshark - lab mode) │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ FEATURE EXTRACTION │
│ 50 Network Flow Features: │
│ • Duration • Packet count • Bytes • Ports • Protocol │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ PREPROCESSING PIPELINE │
│ • Handle missing values • Normalize • Encode categories │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ ML DETECTION ENGINE │
│ ┌────────────────────┐ ┌────────────────────┐ │
│ │ Binary Model │ │ Multi-class Model │ │
│ │ Random Forest │ │ Random Forest │ │
│ │ BENIGN vs ATTACK │ │ 9 Attack Types │ │
│ └────────────────────┘ └────────────────────┘ │
│ ↓ ↓ │
│ └────────────┬────────────┘ │
│ ↓ │
│ Risk Scoring Engine │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ DECISION & ALERTING │
│ Prediction: ATTACK │
│ Type: DDoS │
│ Confidence: 94% │
│ Risk Score: 87/100 - HIGH │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ SECURITY DASHBOARD │
│ ┌─────┬─────┬─────┬─────┐ ┌─────────────────────┐ │
│ │Flows│Attacks│High │Risk │ │ Traffic Graph │ │
│ │125K │ 2.3K │ 18 │ 78 │ │ 📈📉📊 │ │
│ └─────┴─────┴─────┴─────┘ └─────────────────────┘ │
│ │
│ ┌──────────────────────────────────────┐ │
│ │ Recent Alerts │ │
│ │ 🟡 10.0.0.5 → 192.168.1.1 Port 22 │ │
│ │ 🔴 192.168.1.100 → Multiple DDoS │ │
│ └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘



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

### Architecture
Network Flow → 50 Features → Binary Random Forest → Multi-class Random Forest → Attack Type


---

## 🚀 Quick Start

### 📥 Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/AI-NIDS.git
cd AI-NIDS


📦 Step 2: Download Model Files from Google Drive

The trained models are stored on Google Drive due to their size (>100MB each).

Google Drive Folder: AI-NIDS Models

Download these files:

random_forest_binary.joblib → models/binary/

random_forest_multiclass.joblib → models/multiclass/
