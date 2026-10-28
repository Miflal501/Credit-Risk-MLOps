"""FastAPI service that serves the trained model."""

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Credit Risk API", version="1.0.0")


class PredictRequest(BaseModel):
    records: list[dict]


@lru_cache(maxsize=1)
def get_model():
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    model = joblib.load(model_dir / "model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text())
    return model, metadata


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError:
        return {"status": "degraded", "model_loaded": False}


@app.get("/metadata")
def metadata():
    _, meta = get_model()
    return meta


@app.post("/predict")
def predict(req: PredictRequest):
    model, meta = get_model()
    cols = meta["feature_columns"]
    df = pd.DataFrame(req.records)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing columns: {missing}")
    proba = model.predict_proba(df[cols])[:, 1]
    return {
        "model_trained_at": meta["trained_at"],
        "predictions": [
            {"default_probability": round(float(p), 4), "label": int(p >= 0.5)} for p in proba
        ],
    }
