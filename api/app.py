"""
FastAPI service for Breast Cancer classification prediction.
Loads the trained model and exposes a /predict endpoint.
"""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parents[1]  # api/ -> repo root
MODEL_PATH = REPO_DIR / "models" / "final_model.joblib"

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="FastAPI service for predicting malignancy using a trained ML pipeline",
    version="1.0.0",
)

model = None  # will be loaded on startup
REQUIRED_COLUMNS: List[str] = []  # will be set after model loads


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[int]
    count: int


# -----------------------------------------------------------------------------
# Startup: load model with loud debugging
# -----------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    global model, REQUIRED_COLUMNS
    print("=" * 80)
    print("🚀 Starting API")
    print(f"REPO_DIR   = {REPO_DIR}")
    print(f"MODEL_PATH = {MODEL_PATH}")
    print(f"MODEL_EXISTS? {MODEL_PATH.exists()}")
    print("=" * 80)

    if not MODEL_PATH.exists():
        model = None
        return

    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded OK:", type(model))
        REQUIRED_COLUMNS = list(getattr(model, "feature_names_in_", []))
        print("REQUIRED_COLUMNS:", REQUIRED_COLUMNS)
    except Exception as e:
        model = None
        print("❌ Failed to load model:", repr(e))


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Breast Cancer Prediction API",
        "model_path": str(MODEL_PATH),
        "model_loaded": model is not None,
        "required_columns": REQUIRED_COLUMNS,
        "endpoints": ["/health", "/predict", "/docs"],
    }


@app.get("/health")
def health():
    if model is None:
        return {
            "status": "not_ready",
            "model_loaded": False,
            "model_path": str(MODEL_PATH),
            "hint": "Model did not load. Check terminal logs printed at startup.",
        }

    return {"status": "healthy", "model_loaded": True, "model_path": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health and startup logs.")

    if not request.instances:
        raise HTTPException(status_code=400, detail="No instances provided")

    try:
        X = pd.DataFrame(request.instances)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input format: {e}")

    # enforce required columns
    if REQUIRED_COLUMNS:
        missing = set(REQUIRED_COLUMNS) - set(X.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {sorted(missing)}",
            )
        X = X[REQUIRED_COLUMNS]  # keep correct order

    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictResponse(predictions=[int(p) for p in preds], count=len(preds))
