# Demo Script (3–4 minutes)

## Slides
- Title: Real-time Credit Card Fraud Detector
- One-liner: Real-time credit card fraud detector that flags suspicious transactions using a trained ML model, shows explanation scores, and provides a simple interface for human review.
- Problem: Fraud losses, need fast detection and human review
- Data: Synthetic dataset; preprocessing: scaling numeric, encoding categoricals
- Pipeline: Logistic Regression baseline; RandomForest final; imbalance handled
- Metrics: Accuracy, Precision, Recall, F1, AUC, confusion matrix
- Live Demo: UI, API, streaming, logging
- Limitations/Ethics: false positives, bias, privacy
- Future Work: threshold tuning, drift, active learning, model monitoring

## What to Say
- Goal: flag suspicious transactions with probability and explanations
- Explain preprocessing and model choices
- Show metrics panel and confusion matrix
- Highlight explainability with top features
- Emphasize streaming and logging for operations

## Live Steps
1. Open Streamlit UI
2. Enter a safe transaction: small amount, local merchant, low device score
3. Click Predict; show Not Fraud and low probability
4. Enter risky transaction: high amount, foreign_high_risk merchant, high device score
5. Click Predict; show Fraud and high probability; read top features
6. Open Metrics panel; quote Accuracy, Precision, Recall, F1, AUC
7. Click Simulate-Stream for 5–10 rows; watch sequential predictions
8. Download flagged CSV when available

## Metrics to Mention
- Accuracy
- Precision and Recall trade-off
- F1 as balance
- ROC-AUC
- Confusion matrix counts

## Show Risky vs Safe
- Safe: amount < 100, merchant grocery, device_score < 0.3
- Risky: amount > 300, merchant foreign_high_risk, device_score > 0.7

## Streaming Demo
- Use Simulate-Stream; describe per-transaction probability and label
- Explain logging in `results/demo_log.csv`

## Conclusion
- Summarize one-liner
- Mention extensibility: thresholds, retraining, monitoring
- Invite questions

## Troubleshooting Block
- Model loading: first run auto-generates artifacts; ensure `models/` exists
- SHAP issues: disable SHAP and use feature importances; reduce estimators
- Pipeline mismatch: retrain notebook to regenerate encoders and model
- Threshold tuning: adjust slider to meet operational goals
- Latency: preload model on startup; avoid SHAP for batch predictions
