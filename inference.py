"""
inference.py — Phase 5: Deployment

Loads model_bundle.pkl (model, scaler, label_encoder, threshold, feature_columns)
and exposes a single predict() function that replicates Phase 2 preprocessing
exactly, so results match the notebooks bit-for-bit.
"""

import joblib
import numpy as np
import pandas as pd

BUNDLE_PATH = "model_bundle.pkl"

_bundle = None  # lazy-loaded singleton


def load_bundle(path: str = BUNDLE_PATH):
    """Load the model bundle once and cache it."""
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(path)
    return _bundle


def _encode_type(type_value: str, label_encoder) -> int:
    """Encode the 'Type' column exactly as Phase 2 did (H/L/M -> 0/1/2)."""
    type_value = type_value.strip().upper()
    known = set(label_encoder.classes_)
    if type_value not in known:
        raise ValueError(
            f"Unknown Type '{type_value}'. Expected one of {sorted(known)}."
        )
    return int(label_encoder.transform([type_value])[0])


def preprocess(raw: dict, bundle: dict) -> pd.DataFrame:
    """
    Turn a raw sensor reading into the exact scaled feature row the model expects.

    raw expects keys:
      - Type                      (str: 'H', 'L', or 'M')
      - Air temperature [K]       (float)
      - Process temperature [K]   (float)
      - Rotational speed [rpm]    (float/int)
      - Torque [Nm]               (float)
      - Tool wear [min]           (float/int)
    """
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    feature_columns = bundle["feature_columns"]  # order model was trained on

    type_encoded = _encode_type(raw["Type"], label_encoder)

    row = {
        "Air temperature [K]": raw["Air temperature [K]"],
        "Process temperature [K]": raw["Process temperature [K]"],
        "Rotational speed [rpm]": raw["Rotational speed [rpm]"],
        "Torque [Nm]": raw["Torque [Nm]"],
        "Tool wear [min]": raw["Tool wear [min]"],
        "Type_encoded": type_encoded,
    }
    df = pd.DataFrame([row])

    # Scale using the SAME fitted scaler from Phase 2 (no re-fitting -> no leakage)
    scaled_values = scaler.transform(df[scaler.feature_names_in_])
    scaled_df = pd.DataFrame(scaled_values, columns=scaler.feature_names_in_)

    # Ensure exact column order the model was trained on
    return scaled_df[feature_columns]


def risk_level(probability: float, threshold: float) -> str:
    """Simple 3-tier risk bucket for maintenance prioritization."""
    if probability >= threshold:
        return "High"
    elif probability >= threshold * 0.5:
        return "Medium"
    return "Low"


def predict(raw: dict, bundle_path: str = BUNDLE_PATH) -> dict:
    """
    Run one raw sensor reading through the full pipeline and return
    a structured prediction result.
    """
    bundle = load_bundle(bundle_path)
    model = bundle["model"]
    threshold = bundle["threshold"]

    X = preprocess(raw, bundle)
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "failure_probability": round(probability, 4),
        "prediction": prediction,  # 1 = flagged as at-risk, 0 = normal
        "risk_level": risk_level(probability, threshold),
        "threshold_used": round(float(threshold), 4),
        "model_name": bundle.get("model_name", "unknown"),
    }
