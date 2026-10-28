"""SageMaker inference handlers (used by the SKLearn serving container)."""

import json
import os

import joblib
import pandas as pd


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "metadata.json")) as f:
        meta = json.load(f)
    return {"model": model, "columns": meta["feature_columns"]}


def input_fn(body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"unsupported content type: {content_type}")
    return pd.DataFrame(json.loads(body)["records"])


def predict_fn(df, bundle):
    return bundle["model"].predict_proba(df[bundle["columns"]])[:, 1]


def output_fn(proba, accept="application/json"):
    return json.dumps({"default_probability": [round(float(p), 4) for p in proba]})
