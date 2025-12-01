import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

REQUIRED_FIELDS = ["amount", "transaction_time", "merchant_type", "card_country", "device_score"]
CATEGORICAL = ["merchant_type", "card_country"]
NUMERIC = ["amount", "transaction_time", "device_score"]

app = Flask(__name__)


def ensure_artifacts(base_dir: Path) -> None:
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    data_dir = base_dir / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "fraud_model.pkl"
    pipe_path = models_dir / "preprocessing_pipeline.pkl"
    metrics_path = results_dir / "metrics.json"
    data_path = data_dir / "sample_transactions.csv"
    if not model_path.exists() or not pipe_path.exists():
        generate_and_save_model(model_path, pipe_path, metrics_path, data_path)
    if not metrics_path.exists():
        metrics_path.write_text(json.dumps({}))
    if not data_path.exists():
        df = synthetic_data(500)
        df.sample(10, random_state=42).to_csv(data_path, index=False)


def synthetic_data(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    amount = rng.gamma(2.0, 50.0, n)
    transaction_time = rng.integers(0, 24 * 3600, n)
    merchant_type = rng.choice(["grocery", "online", "travel", "foreign_high_risk"], n, p=[0.4, 0.3, 0.2, 0.1])
    card_country = rng.choice(["US", "UK", "IN", "CN", "BR"], n)
    device_score = rng.random(n)
    risk = (
        (amount > 200).astype(int)
        + (merchant_type == "foreign_high_risk").astype(int)
        + (device_score > 0.7).astype(int)
    )
    y = (risk + rng.integers(0, 2, n)) >= 2
    df = pd.DataFrame({
        "amount": amount,
        "transaction_time": transaction_time.astype(float),
        "merchant_type": merchant_type,
        "card_country": card_country,
        "device_score": device_score,
        "label": y.astype(int),
    })
    return df


def oversample(df: pd.DataFrame) -> pd.DataFrame:
    maj = df[df.label == 0]
    mino = df[df.label == 1]
    if len(mino) == 0:
        return df
    factor = max(1, len(maj) // max(1, len(mino)))
    mino_os = pd.concat([mino] * factor, ignore_index=True).iloc[:len(maj)]
    out = pd.concat([maj, mino_os], ignore_index=True).sample(frac=1.0, random_state=42)
    return out


def build_pipeline() -> ColumnTransformer:
    transformers = [
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ]
    return ColumnTransformer(transformers)


def train_models(df: pd.DataFrame) -> Dict[str, Any]:
    df_os = oversample(df)
    X = df_os[NUMERIC + CATEGORICAL]
    y = df_os["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    ct = build_pipeline()
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    pipe_lr = Pipeline([("ct", ct), ("clf", lr)])
    pipe_rf = Pipeline([("ct", ct), ("clf", rf)])
    pipe_lr.fit(X_train, y_train)
    pipe_rf.fit(X_train, y_train)
    proba_lr = pipe_lr.predict_proba(X_test)[:, 1]
    proba_rf = pipe_rf.predict_proba(X_test)[:, 1]
    preds_rf = (proba_rf >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds_rf)),
        "precision": float(precision_score(y_test, preds_rf)),
        "recall": float(recall_score(y_test, preds_rf)),
        "f1": float(f1_score(y_test, preds_rf)),
        "roc_auc": float(roc_auc_score(y_test, proba_rf)),
        "confusion_matrix": confusion_matrix(y_test, preds_rf).tolist(),
        "baseline_auc": float(roc_auc_score(y_test, proba_lr)),
    }
    return {"pipeline": ct, "final_model": pipe_rf.named_steps["clf"], "metrics": metrics, "feature_names": feature_names(ct, X_train)}


def feature_names(ct: ColumnTransformer, X: pd.DataFrame) -> List[str]:
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            names.extend(list(trans.get_feature_names_out(cols)))
    return names


def generate_and_save_model(model_path: Path, pipe_path: Path, metrics_path: Path, data_path: Path) -> None:
    df = synthetic_data(2000)
    trained = train_models(df)
    joblib.dump(trained["final_model"], model_path)
    joblib.dump(trained["pipeline"], pipe_path)
    metrics_path.write_text(json.dumps(trained["metrics"], indent=2))
    df.sample(12, random_state=42).drop(columns=["label"]).to_csv(data_path, index=False)


def load_metrics(base_dir: Path) -> Dict[str, Any]:
    fp = base_dir / "results" / "metrics.json"
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text())
    except Exception:
        return {}


def predict_transaction(transaction_dict: Dict[str, Any], threshold: float = 0.5, base_dir: Path = None) -> Dict[str, Any]:
    base_dir = base_dir or Path(__file__).resolve().parents[1]
    ensure_artifacts(base_dir)
    for f in REQUIRED_FIELDS:
        if f not in transaction_dict:
            raise ValueError("Missing required field")
    df = pd.DataFrame([transaction_dict])[NUMERIC + CATEGORICAL]
    model_path = base_dir / "models" / "fraud_model.pkl"
    pipe_path = base_dir / "models" / "preprocessing_pipeline.pkl"
    model = joblib.load(model_path)
    pipe = joblib.load(pipe_path)
    Xt = pipe.transform(df)
    prob = float(model.predict_proba(Xt)[0, 1])
    label = "fraud" if prob >= threshold else "not_fraud"
    tops = top_features(model, pipe, df, Xt)
    log_prediction(base_dir, transaction_dict, label, prob)
    return {"label": label, "prob": prob, "top_features": tops}


def top_features(model, pipe, df, Xt) -> List[Dict[str, str]]:
    names = feature_names(pipe, df)
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(Xt)
            if isinstance(vals, list):
                vals = vals[1]
            imp = np.abs(vals[0])
            idx = np.argsort(imp)[-2:][::-1]
            return [{"feature": names[i], "impact": "high"} for i in idx]
        except Exception:
            pass
    try:
        imp = np.array(getattr(model, "feature_importances_"))
        idx = np.argsort(imp)[-2:][::-1]
        feats = []
        for i in idx:
            n = names[i]
            if n.startswith("merchant_type_"):
                feats.append({"feature": "merchant_type", "impact": n.replace("merchant_type_", "")})
            elif n.startswith("card_country_"):
                feats.append({"feature": "card_country", "impact": n.replace("card_country_", "")})
            else:
                feats.append({"feature": n, "impact": "high"})
        return feats
    except Exception:
        amount = float(df.iloc[0]["amount"]) if "amount" in df.columns else 0.0
        mt = str(df.iloc[0]["merchant_type"]) if "merchant_type" in df.columns else ""
        feats = []
        if amount > 200:
            feats.append({"feature": "amount", "impact": "high"})
        if mt == "foreign_high_risk":
            feats.append({"feature": "merchant_type", "impact": "foreign_high_risk"})
        if not feats:
            feats.append({"feature": "device_score", "impact": "medium"})
        return feats[:2]


def log_prediction(base_dir: Path, payload: Dict[str, Any], label: str, prob: float) -> None:
    fp = base_dir / "results" / "demo_log.csv"
    fp.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().isoformat()
    line = f"{ts}|{json.dumps(payload)}|{label}|{prob:.6f}\n"
    with open(fp, "a", encoding="utf-8") as f:
        f.write(line)


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        data = request.get_json(force=True)
        res = predict_transaction(data)
        return jsonify(res)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"error": "Model not loaded"}), 500
    except Exception:
        return jsonify({"error": "Invalid input"}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    ensure_artifacts(base)
    app.run(host="0.0.0.0", port=5000)
