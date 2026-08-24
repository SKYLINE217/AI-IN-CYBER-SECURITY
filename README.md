<div align="center">

<!-- Animated Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,50:BF5FFF,100:00D4FF&height=220&section=header&text=Real-Time%20Fraud%20Detection&fontSize=42&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&desc=ML-Powered%20Security%20Pipeline%20%E2%80%A2%20Random%20Forest%20%2B%20SHAP%20Explainability&descSize=17&descAlignY=55&descColor=94A3B8" width="100%" />

<!-- Animated Typing SVG -->
<a href="https://github.com/SKYLINE217/AI-IN-CYBER-SECURITY">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&duration=3000&pause=1000&color=BF5FFF&center=true&vCenter=true&multiline=true&repeat=true&width=750&height=80&lines=Real-Time+Credit+Card+Fraud+Detection;Random+Forest+%7C+SHAP+Explainability+%7C+Flask+API;96%25+Recall+%E2%80%A2+Streamlit+Dashboard+%E2%80%A2+Audit+Logging" alt="Typing SVG" />
</a>

<br/>

<!-- Badges Row 1 – Stack -->
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-BF5FFF?style=for-the-badge)

<br/>

<!-- Badges Row 2 – Quick Links -->
[![Features](https://img.shields.io/badge/🔑_Key-Features-BF5FFF?style=flat-square)](#-key-features)
[![Architecture](https://img.shields.io/badge/🏗️-Architecture-00D4FF?style=flat-square)](#-architecture)
[![API](https://img.shields.io/badge/🔌-API-4D6AF5?style=flat-square)](#-api-reference)
[![Setup](https://img.shields.io/badge/🚀-Quick+Start-41CD52?style=flat-square)](#-quick-start)

</div>

---

> [!CAUTION]
> **Research & Demo Only** — This system is a portfolio demonstration built on synthetic data. It is NOT intended for production financial fraud detection. The authors disclaim all liability for misuse in regulated financial contexts.

---

## 🌐 Overview

**Real-Time Fraud Detection** is an end-to-end ML security pipeline that flags suspicious financial transactions in milliseconds. It combines a high-recall **Random Forest** classifier with **SHAP explainability** to surface the exact features that drove each decision — making it audit-ready by design.

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
| 🤖 **Adaptive ML** | Random Forest trained on class-imbalanced synthetic data with oversampling |
| 🔍 **SHAP Explainability** | Top-2 feature contributors per prediction (falls back to feature importances) |
| ⚡ **Dual Interface** | Flask REST API on port 5000 + Streamlit dashboard on port 8501 |
| 📊 **Metrics Dashboard** | Live accuracy, precision, recall, F1, ROC-AUC, and confusion matrix |
| 🌊 **Streaming Demo** | Simulates 1–20 transactions from CSV with configurable delay |
| 📝 **Audit Trail** | `timestamp|input_json|label|probability` log in `results/demo_log.csv` |
| 🎚️ **Threshold Control** | Sidebar slider adjusts fraud decision boundary live |
| 🔁 **Reproducible** | Artifacts saved to `models/` and `results/` for offline inference |

---

## 🏗️ Architecture

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'primaryColor': '#BF5FFF', 'edgeLabelBackground':'#0D1117', 'fontSize': '16px'}}}%%
graph LR
    A(["📊 Input\nTransaction\n{amount, merchant_type,\ncountry, device_score}"])-->|"POST /predict"|B
    B(["🔌 Flask REST API\n:5000"])-->|"Preprocessed\nFeature Vector"|C
    C(["🤖 Random Forest\nClassifier"])-->|"Probability\nScore"|D
    D(["🔍 SHAP Explainer\nTop-2 Features"])-->|"Label +\nExplanation"|E
    E(["📝 Audit Logger\ndemo_log.csv"])
    D-->|"JSON Response"|F
    F(["👤 Client\nor Streamlit UI"])

    A2(["🖥️ Streamlit UI\n:8501"])-->|"Manual\nReview"|B
    A2-->|"Batch\nStream"|B

    style A fill:#1a1a2e,stroke:#BF5FFF,color:#fff
    style B fill:#1a1a2e,stroke:#4D6AF5,color:#fff
    style C fill:#1a1a2e,stroke:#EE4C2C,color:#fff
    style D fill:#1a1a2e,stroke:#F59E0B,color:#fff
    style E fill:#1a1a2e,stroke:#41CD52,color:#fff
    style F fill:#1a1a2e,stroke:#00D4FF,color:#fff
    style A2 fill:#1a1a2e,stroke:#FF4B4B,color:#fff
```

### 4-Phase ML Pipeline

<table>
<tr>
<td width="20%" align="center">📊<br/><strong>Phase 1</strong><br/>Data</td>
<td width="80%">Synthetic class-imbalanced dataset. Minority class (fraud) oversampled to stabilise training. Features: <code>amount</code>, <code>transaction_time</code>, <code>merchant_type</code>, <code>card_country</code>, <code>device_score</code>.</td>
</tr>
<tr>
<td align="center">⚙️<br/><strong>Phase 2</strong><br/>Preprocessing</td>
<td><code>ColumnTransformer</code>: <code>StandardScaler</code> on numerics + <code>OneHotEncoder</code> on categoricals. Pipeline serialised to <code>models/preprocessing_pipeline.pkl</code>.</td>
</tr>
<tr>
<td align="center">🤖<br/><strong>Phase 3</strong><br/>Model</td>
<td>Logistic Regression baseline → <strong>Random Forest</strong> final model. Optimised for high recall to minimise missed fraud. Serialised to <code>models/fraud_model.pkl</code>.</td>
</tr>
<tr>
<td align="center">🔍<br/><strong>Phase 4</strong><br/>Explain</td>
<td>SHAP TreeExplainer for tree-based models; falls back to <code>feature_importances_</code> or rule-based explanations. Top-2 contributors returned per prediction.</td>
</tr>
</table>

---

## 📁 Repository Structure

```
creditcard-fraud-poc/
├── README.md
├── requirements.txt
├── models/
│   ├── fraud_model.pkl
│   └── preprocessing_pipeline.pkl
├── notebooks/
│   └── baseline_training.ipynb
├── app/
│   ├── app.py
│   └── static/
│       └── style.css
├── api/
│   └── predict.py
├── data/
│   └── sample_transactions.csv
├── demo_script.md
└── results/
    ├── metrics.json
    └── demo_log.csv
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.10+**
- `pip`

### Install

```bash
# 1. Clone the repository
git clone https://github.com/SKYLINE217/AI-IN-CYBER-SECURITY.git
cd AI-IN-CYBER-SECURITY

# 2. Create virtual environment
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# Terminal A — Streamlit UI
streamlit run app/app.py

# Terminal B — Flask REST API
# Windows
set FLASK_APP=api/predict.py
flask run --host=0.0.0.0 --port=5000

# macOS/Linux
export FLASK_APP=api/predict.py
flask run --host=0.0.0.0 --port=5000
```

> [!TIP]
> On first launch, the Streamlit UI auto-generates `models/` artifacts if they don't exist. Run it once before the Flask API.

---

## 🔌 API Reference

All endpoints served at `http://localhost:5000`.

| Method | Endpoint | Description |
|:---:|:---|:---|
| `POST` | `/predict` | Classify a transaction + return explanation |
| `GET` | `/health` | Liveness probe |

### Input Schema

```json
{
  "amount": 250.0,
  "transaction_time": 3600.0,
  "merchant_type": "online",
  "card_country": "US",
  "device_score": 0.7
}
```

| Field | Type | Description |
|:---|:---|:---|
| `amount` | float | Purchase amount (currency units) |
| `transaction_time` | float | Seconds since start of day (0 – 86400) |
| `merchant_type` | string | `grocery`, `online`, `travel`, `foreign_high_risk` |
| `card_country` | string | ISO-like code (e.g., `US`, `UK`, `IN`, `CN`) |
| `device_score` | float | Normalised device risk (0.0 = low → 1.0 = high) |

### Example Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 480.2, "transaction_time": 3600, "merchant_type": "foreign_high_risk", "card_country": "CN", "device_score": 0.95}'
```

### Example Response

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

> [!IMPORTANT]
> Metrics computed on synthetic data. SHAP can be computationally heavy; the system falls back to feature importances automatically on large inputs.

---

## 🖥️ Streamlit Dashboard

The dashboard provides 5 interactive tabs:

| Tab | Description |
|:---|:---|
| **① Predict** | Manual input form with threshold slider, status badge, and explanation |
| **② Metrics** | Full metrics panel with confusion matrix |
| **③ Stream** | Simulated 1–20 transaction stream with CSV download |
| **④ Logs** | Recent `demo_log.csv` entries + full CSV export |
| **⑤ API** | Direct Flask `/predict` call from within the UI |

---

## 🔒 Security & Ethics

- **No PII logging** beyond what's necessary for auditability
- **Threshold tuning** — adjust the sidebar slider to balance precision/recall for your use case
- **Model drift** — retrain periodically on fresh data
- **False positive impact** — review threshold decisions carefully to avoid over-blocking legitimate transactions

---

## ❓ FAQ

| Issue | Solution |
|:---|:---|
| `Model not loaded` | Run Streamlit once — it auto-generates artifacts if missing |
| `Invalid input` / `Missing required field` | Check JSON fields against the input schema |
| `Pipeline mismatch` | Re-run the notebook to regenerate model + pipeline together |

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,50:BF5FFF,100:00D4FF&height=120&section=footer" width="100%"/>

**Built for the Cybersecurity Community** 🛡️

[![GitHub](https://img.shields.io/badge/GitHub-SKYLINE217-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SKYLINE217)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=SKYLINE217.AI-IN-CYBER-SECURITY)

*Built with ❤️ for fraud analysts everywhere.*

</div>

<!-- portfolio upgrade -->
