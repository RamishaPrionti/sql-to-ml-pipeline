import json
import os
from typing import Any, Dict

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# MUST be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Classification", page_icon="🧬", layout="centered")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Breast Cancer Wisconsin (sklearn) features (30)
FEATURES = [
    "mean radius","mean texture","mean perimeter","mean area","mean smoothness","mean compactness",
    "mean concavity","mean concave points","mean symmetry","mean fractal dimension",
    "radius error","texture error","perimeter error","area error","smoothness error","compactness error",
    "concavity error","concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area","worst smoothness","worst compactness",
    "worst concavity","worst concave points","worst symmetry","worst fractal dimension",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def call_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    return resp.json()

def safe_get_health() -> str:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=10)
        if r.status_code == 200:
            return "healthy"
    except Exception:
        pass
    return "unknown"

def label_from_pred(pred: Any) -> str:
    """
    sklearn breast cancer dataset convention:
      0 = malignant
      1 = benign
    """
    try:
        p = int(pred)
    except Exception:
        return str(pred)

    if p == 0:
        return "Malignant (cancer)"
    if p == 1:
        return "Benign (non-cancer)"
    return str(p)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🧬 Breast Cancer Classification App")
st.write(
    f"This app sends your inputs to the FastAPI backend at **{API_BASE_URL}** for prediction."
)

st.caption(f"API health: **{safe_get_health()}**")

st.header("Input Features")

user_input: Dict[str, Any] = {}

st.subheader("Tumor measurements (30 features)")

# nice defaults so you can click Predict quickly
DEFAULTS = {
    "mean radius": 14.0,
    "mean texture": 19.0,
    "mean perimeter": 92.0,
    "mean area": 650.0,
    "mean smoothness": 0.10,
    "mean compactness": 0.10,
    "mean concavity": 0.10,
    "mean concave points": 0.05,
    "mean symmetry": 0.18,
    "mean fractal dimension": 0.06,
}

# layout: 2 columns of inputs
col1, col2 = st.columns(2)

for i, feat in enumerate(FEATURES):
    default_val = float(DEFAULTS.get(feat, 0.0))
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        user_input[feat] = st.number_input(
            feat,
            value=default_val,
            step=0.01,
            format="%.5f",
            help="Enter a numeric value for this feature.",
            key=feat,
        )

st.markdown("---")

if st.button("🔮 Predict", type="primary"):
    payload = {"instances": [user_input]}

    with st.spinner("Calling API for prediction..."):
        try:
            data = call_api(payload)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
        else:
            preds = data.get("predictions", [])
            if not preds:
                st.warning("⚠️ No predictions returned from API.")
            else:
                pred = preds[0]
                st.success("✅ Prediction successful!")

                st.subheader("Prediction Result")

                st.metric(label="Predicted class", value=label_from_pred(pred))
                st.caption("Label mapping: 0 = Malignant, 1 = Benign (common sklearn convention).")

                with st.expander("📋 View Input Summary"):
                    st.json(user_input)

st.markdown("---")
st.caption(f"🌐 API: `{API_BASE_URL}`  |  Endpoint: `{PREDICT_ENDPOINT}`")
