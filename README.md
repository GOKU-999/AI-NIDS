# 🛡️ AI-Powered Network Intrusion Detection System (AI-NIDS)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📊 Overview

AI-NIDS is a complete security analytics platform that uses Machine Learning to detect network intrusions in real-time. It combines:

- **🧠 ML Detection Engine** - Random Forest models for attack classification
- **🚀 FastAPI Backend** - High-performance REST API
- **🎨 React Dashboard** - Professional SOC-style interface
- **📊 Explainable AI** - Feature importance and attack explanations

## 🎯 Features

### Core Capabilities
- ✅ Binary Classification (Normal vs Attack)
- ✅ Multi-class Attack Detection (9 attack types)
- ✅ Real-time Flow Analysis
- ✅ CSV Batch Analysis
- ✅ Risk Scoring (0-100)
- ✅ Explainable Predictions

### Attack Types Detected
| ID | Attack Type | Description |
|----|-------------|-------------|
| 0 | BENIGN | Normal network traffic |
| 1 | DoS | Denial of Service |
| 2 | DDoS | Distributed DoS |
| 3 | PortScan | Port scanning activity |
| 4 | BruteForce | Password brute force |
| 5 | WebAttack | Web application attacks |
| 6 | Botnet | Botnet activity |
| 7 | Infiltration | Network infiltration |
| 8 | Heartbleed | Heartbleed exploit |

## 🏗️ Architecture
