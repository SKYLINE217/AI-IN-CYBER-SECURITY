# creditcard-fraud-poc

Real-time credit card fraud detector that flags suspicious transactions using a trained ML model, shows explanation scores, and provides a simple interface for human review.

## Summary
- End-to-end real-time fraud detection with training, API, and UI
- Shows probability, label, and top feature explanations
- Metrics panel with accuracy, precision, recall, F1, AUC, confusion matrix
- Simulated streaming of transactions and logging of predictions

## Features
- Training notebook generating model and preprocessing pipeline
- REST API `/predict` returning label, probability, and top features
- Streamlit UI for manual review and streaming demo
- Logging to `results/demo_log.csv`
- Metrics saved to `results/metrics.json`

## Repository Structure
```
creditcard-fraud-poc/
├─ README.md
├─ requirements.txt
├─ models/
│  ├─ fraud_model.pkl
│  └─ preprocessing_pipeline.pkl
├─ notebooks/
│  └─ baseline_training.ipynb
├─ app/
│  ├─ app.py
│  └─ static/
│     └─ style.css
├─ api/
│  └─ predict.py
├─ data/
│  └─ sample_transactions.csv
├─ demo_script.md
└─ results/
   ├─ metrics.json
   └─ demo_log.csv
```

## How to Install
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Windows PowerShell:
```
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Instructions
Streamlit:
```
streamlit run app/app.py
```
Flask API:
```
set FLASK_APP=api/predict.py
flask run --host=0.0.0.0 --port=5000
```
macOS/Linux:
```
export FLASK_APP=api/predict.py
flask run --host=0.0.0.0 --port=5000
```

## API Usage Example
```
curl -X POST http://localhost:5000/predict \
 -H "Content-Type: application/json" \
 -d '{"amount": 250, "merchant_type":"online", "card_country":"US", "transaction_time": 3600, "device_score": 0.7 }'
```

## Model and Metrics Storage
- Model: `models/fraud_model.pkl`
- Preprocessing: `models/preprocessing_pipeline.pkl`
- Metrics: `results/metrics.json`
- Logs: `results/demo_log.csv`

## Demo Script
See `demo_script.md`.

## Slides Outline
- Title and team
- Problem statement
- Dataset and preprocessing
- Model pipeline
- Metrics and confusion matrix
- Live demo
- Limitations and ethics
- Future work
- Q&A

## Troubleshooting
- Model loading errors: ensure `models/` exists; first run generates artifacts automatically
- SHAP performance: disable SHAP in UI or use feature importances fallback
- Pipeline mismatch: recreate artifacts using the training notebook to align encoders
- Threshold tuning: adjust the threshold slider in the UI to balance precision/recall
- Slow latency: pre-load model at app startup and reduce SHAP computations

## Demo Video
Optional: link a screencast of the Streamlit demo

---

## Overview and Architecture
- Data and features: `amount`, `transaction_time`, `merchant_type`, `card_country`, `device_score`
- Preprocessing: `StandardScaler` (numeric), `OneHotEncoder` (categorical) in a `ColumnTransformer`
- Models: Logistic Regression baseline; RandomForest final (used for inference)
- Explainability: SHAP for trees when available; fallback to `feature_importances_` or simple rules
- Artifacts: saved to `models/` and `results/` for reproducible inference and reporting
- Services: Flask REST API (`/predict`) and Streamlit UI (`app/app.py`)
- Streaming: Simulated sequence from `data/sample_transactions.csv` with `time.sleep(0.8)`
- Logging: Every prediction appended to `results/demo_log.csv`

## UI Guide
- Predict tab
  - Inputs with units and help tooltips
  - Shows status (Fraud / Not Fraud / Borderline), probability bar, top factors, and a concise explanation sentence
  - Threshold slider in sidebar controls decision boundary
- Metrics tab
  - Displays accuracy, precision, recall, F1, ROC-AUC
  - Confusion matrix table (rows: True class; columns: Predicted class)
- Stream tab
  - Streams 1–20 sample transactions from `data/sample_transactions.csv`
  - Shows results one by one with a short delay, and offers CSV download of flagged rows
- Logs tab
  - Renders recent `results/demo_log.csv` entries and allows full CSV download
- API tab
  - Lets you call the Flask `/predict` endpoint directly from the UI
  - Requires the Flask server to be running concurrently

## Input Schema and Units
- `amount` (float): purchase amount in currency units
- `transaction_time` (float): seconds since start of day (0–86400)
- `merchant_type` (string): one of `grocery`, `online`, `travel`, `foreign_high_risk`
- `card_country` (string): ISO-like country code, e.g., `US`, `UK`, `IN`, `CN`, `BR`
- `device_score` (float): normalized device risk from 0.0 (low) to 1.0 (high)

Example JSON:
```
{
  "amount": 250.0,
  "transaction_time": 3600.0,
  "merchant_type": "online",
  "card_country": "US",
  "device_score": 0.7
}
```

## API Response Schema
```
{
  "label": "fraud",
  "prob": 0.92,
  "top_features": [
    {"feature": "amount", "impact": "high"},
    {"feature": "merchant_type", "impact": "foreign_high_risk"}
  ]
}
```
- `label`: `fraud` or `not_fraud` according to threshold
- `prob`: probability of fraud from the model
- `top_features`: top 2 contributors from SHAP, importances, or rules

## Logging Format
- File: `results/demo_log.csv`
- Format: `timestamp|input_json|label|probability`
- Example:
```
2025-01-01T12:00:00Z|{"amount":480.2,"merchant_type":"foreign_high_risk",...}|fraud|0.923541
```

## Notebook Walkthrough
- Dataset: synthetic, class-imbalanced by design
- Imbalance handling: simple oversampling of minority class to stabilize training
- Preprocessing: scale numeric features, one-hot encode categoricals
- Models trained: Logistic Regression (baseline) and RandomForest (final)
- Metrics computed: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- Artifacts saved:
  - `models/fraud_model.pkl` (final model)
  - `models/preprocessing_pipeline.pkl` (encoders + scaler)
  - `results/metrics.json` (evaluation metrics)
  - `data/sample_transactions.csv` (for streaming demo)

## Decision Threshold and Explanations
- Threshold rule: predicted `prob >= threshold` → `fraud`, else `not_fraud`
- Default threshold: `0.50` (adjust via Streamlit sidebar)
- Explanations:
  - Fraud: “Unusually high amount” and/or “foreign merchant” and/or “high device risk”
  - Not Fraud: “Likely safe due to small amount, non-foreign merchant, low device risk”

## Running UI and API Together
- Terminal A: Streamlit UI
  - `streamlit run app/app.py`
- Terminal B: Flask API
  - Windows: `set FLASK_APP=api/predict.py` then `flask run --host=0.0.0.0 --port=5000`
  - macOS/Linux: `export FLASK_APP=api/predict.py` then `flask run --host=0.0.0.0 --port=5000`
- The Streamlit “API” tab will auto-test `/health` then call `/predict` with your inputs

## Development and Testing
- Virtual environment recommended
- Install: `pip install -r requirements.txt`
- Optional tests: add simple unit tests for `predict_transaction` (e.g., input validation, probability range, logging side effects)
- Type checking and linting can be added via `ruff` or `flake8` if desired

## Performance Tips
- SHAP can be computationally heavy; the system falls back to feature importances or rules when needed
- Preload artifacts at UI/API startup to minimize per-request latency
- Adjust RandomForest `n_estimators` to trade off speed vs. accuracy

## Security and Ethics
- Do not log PII beyond what is necessary for auditability
- Consider false positives/negatives impact and perform regular threshold reviews
- Monitor for model drift and retrain periodically

## FAQ
- “Model not loaded”: run Streamlit once; it generates artifacts automatically if missing
- “Invalid input” or “Missing required field”: check JSON fields and types against the input schema
- “Pipeline mismatch”: re-run the notebook to regenerate both the model and the pipeline together
