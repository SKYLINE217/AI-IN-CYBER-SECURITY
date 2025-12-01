import json
import time
import sys
import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
import requests
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from api.predict import predict_transaction, ensure_artifacts, load_metrics

BASE_DIR = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")

ensure_artifacts(BASE_DIR)

st.title("Credit Card Fraud Detector")

threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5, 0.01, help="Decision threshold: ≥ threshold = fraud")
st.sidebar.markdown("**Units**")
st.sidebar.caption("transaction_time: clock time (HH:MM:SS), converted to seconds")
st.sidebar.caption("device_score: normalized risk 0 (low) to 1 (high)")

merchant_options = ["grocery", "online", "travel", "foreign_high_risk"]
country_options = ["US", "UK", "IN", "CN", "BR"]

tab_predict, tab_metrics, tab_stream, tab_logs, tab_api, tab_guide = st.tabs(["Predict", "Metrics", "Stream", "Logs", "API", "Guide"])

with tab_predict:
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("amount", min_value=0.0, value=120.0, step=10.0, help="Purchase amount in currency units")
        merchant_type = st.selectbox("merchant_type", merchant_options, help="Type of merchant")
        device_score = st.number_input("device_score (0–1 risk)", min_value=0.0, max_value=1.0, value=0.4, step=0.05, help="0 = low risk device, 1 = high risk device")
    with col2:
        t_clock = st.time_input("transaction_time (clock)", value=datetime.time(1, 0, 0), help="Clock time HH:MM:SS")
        transaction_time = float(t_clock.hour * 3600 + t_clock.minute * 60 + t_clock.second)
        card_country = st.selectbox("card_country", country_options, help="Card issuing country")
    if st.button("Predict"):
        payload = {
            "amount": amount,
            "transaction_time": transaction_time,
            "merchant_type": merchant_type,
            "card_country": card_country,
            "device_score": device_score,
        }
        try:
            result = predict_transaction(payload, threshold=threshold, base_dir=BASE_DIR)
            label = result["label"]
            prob = result["prob"]
            st.subheader("Result")
            if label == "fraud":
                st.error("Fraud")
            elif abs(prob - threshold) < 0.05:
                st.warning("Borderline")
            else:
                st.success("Not Fraud")
            st.progress(int(prob * 100))
            st.write(f"Probability: {prob:.2%} (threshold {threshold:.2f})")
            tops = result.get("top_features", [])
            if tops:
                st.write("Top factors:")
                for t in tops:
                    st.write(f"{t['feature']}: {t['impact']}")
                names = [t["feature"] for t in tops]
                if label == "fraud":
                    if "amount" in names and any("merchant_type" in n for n in names):
                        st.info("Unusually high amount + foreign merchant increased fraud risk.")
                    elif "amount" in names:
                        st.info("Unusually high amount increased fraud risk.")
                    elif any("merchant_type" in n for n in names):
                        st.info("Merchant risk type increased fraud risk.")
                    elif "device_score" in names:
                        st.info("Suspicious device score increased fraud risk.")
                else:
                    safe_reasons = []
                    if amount < 100:
                        safe_reasons.append("small amount")
                    if merchant_type in ["grocery", "online", "travel"]:
                        safe_reasons.append("non-foreign merchant")
                    if device_score < 0.3:
                        safe_reasons.append("low device risk")
                    if card_country in ["US", "UK", "IN", "CN", "BR"]:
                        safe_reasons.append("common card country")
                    if safe_reasons:
                        st.info("Likely safe due to " + ", ".join(safe_reasons) + ".")
            else:
                if label != "fraud":
                    st.info("Likely safe: typical transaction characteristics and low device risk.")
        except ValueError as e:
            st.error(str(e))
        except FileNotFoundError:
            st.error("Model not loaded")
        except Exception:
            st.error("Invalid input")

with tab_metrics:
    metrics = load_metrics(BASE_DIR)
    if metrics:
        colm1, colm2, colm3 = st.columns(3)
        with colm1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        with colm2:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            st.metric("F1", f"{metrics.get('f1', 0):.3f}")
        with colm3:
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        cm = metrics.get("confusion_matrix")
        if cm:
            df_cm = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"])
            st.table(df_cm)

with tab_stream:
    stream_rows = st.number_input("Rows", min_value=1, max_value=20, value=8, step=1)
    if st.button("Simulate-Stream"):
        fp = BASE_DIR / "data" / "sample_transactions.csv"
        try:
            df = pd.read_csv(fp)
            df = df.sample(n=min(stream_rows, len(df)), random_state=42)
            flags = []
            for _, row in df.iterrows():
                payload = row.to_dict()
                res = predict_transaction(payload, threshold=threshold, base_dir=BASE_DIR)
                prob = res["prob"]
                st.write(json.dumps(payload))
                st.write("Fraud" if res["label"] == "fraud" else "Not Fraud", f"({prob:.2%})")
                time.sleep(0.8)
                if res["label"] == "fraud":
                    flags.append({"prob": prob, **payload})
            if flags:
                out = pd.DataFrame(flags)
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download flagged CSV", csv, "flagged.csv", "text/csv")
        except Exception:
            st.error("Missing required field")

with tab_logs:
    fp = BASE_DIR / "results" / "demo_log.csv"
    if fp.exists():
        df_log = pd.read_csv(fp, sep="|")
        st.write(df_log.tail(20))
        st.download_button("Download logs", df_log.to_csv(index=False).encode("utf-8"), "demo_log.csv", "text/csv")
    else:
        st.info("No logs yet")

with tab_api:
    col1, col2 = st.columns(2)
    with col1:
        a_amount = st.number_input("api_amount", min_value=0.0, value=200.0, step=10.0)
        a_merchant_type = st.selectbox("api_merchant_type", merchant_options, index=1)
        a_device_score = st.number_input("api_device_score", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    with col2:
        a_t_clock = st.time_input("api_transaction_time (clock)", value=datetime.time(0, 16, 40))
        a_transaction_time = float(a_t_clock.hour * 3600 + a_t_clock.minute * 60 + a_t_clock.second)
        a_card_country = st.selectbox("api_card_country", country_options)
    if st.button("Call Flask /predict"):
        payload = {
            "amount": a_amount,
            "transaction_time": a_transaction_time,
            "merchant_type": a_merchant_type,
            "card_country": a_card_country,
            "device_score": a_device_score,
        }
        try:
            h = requests.get("http://localhost:5000/health", timeout=2)
            if h.status_code == 200:
                r = requests.post("http://localhost:5000/predict", json=payload, timeout=5)
                if r.status_code == 200:
                    st.success(json.dumps(r.json()))
                else:
                    st.error(r.text)
            else:
                st.error("Model not loaded")
        except Exception:
            st.error("Invalid input")

with tab_guide:
    st.header("How It Works")
    st.write("Enter transaction details and click Predict. The model returns a probability and label. Use the threshold slider to adjust sensitivity.")
    st.subheader("Inputs")
    st.write("amount: purchase amount; merchant_type: type of merchant; card_country: issuing country; device_score: device risk 0–1; transaction_time: clock time HH:MM:SS")
    st.subheader("Outputs")
    st.write("Label (Fraud/Not Fraud), probability, and top factors explaining the decision.")
    st.subheader("Streaming and Logs")
    st.write("Use Stream tab to play sample transactions. Every prediction is logged to results/demo_log.csv.")
    st.subheader("API")
    st.write("Run Flask and use the API tab or curl to post JSON with the same fields.")
