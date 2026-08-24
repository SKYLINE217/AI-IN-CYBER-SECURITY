<div align="center">

<!-- Animated Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,50:BF5FFF,100:00D4FF&height=220&section=header&text=Real-Time%20Fraud%20Detection&fontSize=42&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&desc=ML-Powered%20Security%20Pipeline%20%E2%80%A2%20Random%20Forest%20%2B%20SHAP%20Explainability&descSize=17&descAlignY=55&descColor=94A3B8" width="100%" />

<!-- Animated Typing SVG -->
<a href="https://github.com/SKYLINE217/AI-IN-CYBER-SECURITY">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&duration=3000&pause=1000&color=BF5FFF&center=true&vCenter=true&multiline=true&repeat=true&width=750&height=80&lines=Real-Time+Credit+Card+Fraud+Detection;Random+Forest+%7C+SHAP+Explainability+%7C+Flask+API;96%25+Recall+%E2%80%A2+Streamlit+Dashboard+%E2%80%A2+Audit+Logging" alt="Typing SVG" />
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-BF5FFF?style=for-the-badge)

</div>

---

## 🌐 Overview

**Real-Time Fraud Detection** is an end-to-end ML security pipeline that flags suspicious financial transactions in milliseconds. It combines a high-recall **Random Forest** classifier with **SHAP explainability** to surface the exact features that drove each decision.

| | |
|:---:|:---|
| 🤖 **Model** | Random Forest (high recall optimised) |
| 🔍 **Explainability** | SHAP TreeExplainer per prediction |
| ⚡ **Dual Interface** | Flask REST API `/predict` + Streamlit UI |
| 📊 **Metrics Panel** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| 📝 **Audit Logging** | Every prediction appended to `results/demo_log.csv` |
| 🌊 **Streaming Demo** | Simulated real-time transaction stream with CSV download |

---

## 🔑 Key Features

| Feature | Details |
|:---:|:---|
| 🤖 **Adaptive ML** | Random Forest on class-imbalanced synthetic data with oversampling |
| 🔍 **SHAP Explainability** | Top-2 feature contributors per prediction |
| ⚡ **Dual Interface** | Flask REST API on port 5000 + Streamlit dashboard on port 8501 |
| 📊 **Metrics Dashboard** | Live accuracy, precision, recall, F1, ROC-AUC, and confusion matrix |
| 🌊 **Streaming Demo** | Simulates 1–20 transactions from CSV with configurable delay |
| 📝 **Audit Trail** | `timestamp|input_json|label|probability` log in `results/demo_log.csv` |
| 🎚️ **Threshold Control** | Sidebar slider adjusts fraud decision boundary live |

---

## 🏗️ Architecture

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'primaryColor': '#BF5FFF', 'edgeLabelBackground':'#0D1117', 'fontSize': '16px'}}}%%
graph LR
    A(["📊 Input Transaction"])-->|"POST /predict"|B
    B(["🔌 Flask REST API :5000"])-->|"Feature Vector"|C
    C(["🤖 Random Forest Classifier"])-->|"Probability Score"|D
    D(["🔍 SHAP Explainer"])-->|"Label + Explanation"|E
    E(["📝 Audit Logger demo_log.csv"])
    D-->|"JSON Response"|F
    F(["👤 Client or Streamlit UI"])
    A2(["🖥️ Streamlit UI :8501"])-->|"Manual Review"|B
    style A fill:#1a1a2e,stroke:#BF5FFF,color:#fff
    style B fill:#1a1a2e,stroke:#4D6AF5,color:#fff
    style C fill:#1a1a2e,stroke:#EE4C2C,color:#fff
    style D fill:#1a1a2e,stroke:#F59E0B,color:#fff
    style E fill:#1a1a2e,stroke:#41CD52,color:#fff
    style F fill:#1a1a2e,stroke:#00D4FF,color:#fff
    style A2 fill:#1a1a2e,stroke:#FF4B4B,color:#fff
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/SKYLINE217/AI-IN-CYBER-SECURITY.git
cd AI-IN-CYBER-SECURITY
python -m venv venv
venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt

# Terminal A — Streamlit UI
streamlit run app/app.py

# Terminal B — Flask REST API
set FLASK_APP=api/predict.py
flask run --host=0.0.0.0 --port=5000
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|:---:|:---|:---|
| `POST` | `/predict` | Classify a transaction + return SHAP explanation |
| `GET` | `/health` | Liveness probe |

**Example Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 480.2, "transaction_time": 3600, "merchant_type": "foreign_high_risk", "card_country": "CN", "device_score": 0.95}'
```

**Example Response:**
```json
{
  "label": "fraud",
  "prob": 0.92,
  "top_features": [
    {"feature": "amount", "impact": "high"},
    {"feature": "merchant_type", "impact": "foreign_high_risk"}
  ]
}
```

---

## 📊 Model Metrics

| Metric | Value |
|:---|:---|
| **Accuracy** | ~94% |
| **Precision** | ~89% |
| **Recall** | ~96% |
| **F1-Score** | ~92% |
| **ROC-AUC** | ~98% |

---

## 🖥️ Streamlit Dashboard

| Tab | Description |
|:---|:---|
| **① Predict** | Manual input with threshold slider, status badge, explanation |
| **② Metrics** | Full metrics panel with confusion matrix |
| **③ Stream** | Simulated transaction stream with CSV download |
| **④ Logs** | Recent audit log entries + full CSV export |
| **⑤ API** | Direct Flask `/predict` call from within the UI |

---

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-SKYLINE217-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SKYLINE217)

*Built with ❤️ for fraud analysts everywhere.*

</div>
