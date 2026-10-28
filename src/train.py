"""Train, evaluate, track (MLflow) and package a credit-risk model.

Local:      python src/train.py
CI/tests:   python src/train.py --synthetic --n-iter 3 --cv 3
SageMaker:  the same script runs in script mode; SM_MODEL_DIR is set by SageMaker.
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

SEED = 42


def load_data(synthetic: bool = False):
    """Return (X, y). Default dataset: German Credit (OpenML 'credit-g'), target 1 = bad risk."""
    if synthetic:
        return _synthetic_data()
    from sklearn.datasets import fetch_openml

    ds = fetch_openml("credit-g", version=1, as_frame=True)
    X = ds.data.copy()
    category_cols = X.select_dtypes("category").columns
    X[category_cols] = X[category_cols].astype(object)
    y = (ds.target == "bad").astype(int)
    return X, y


def _synthetic_data(n: int = 800):
    rng = np.random.default_rng(SEED)
    arr, y = make_classification(n_samples=n, n_features=6, n_informative=4, random_state=SEED)
    X = pd.DataFrame(arr, columns=[f"num_{i}" for i in range(6)])
    X["housing"] = rng.choice(["own", "rent", "free"], size=n)
    X["purpose"] = rng.choice(["car", "education", "business", "radio"], size=n)
    X.loc[rng.choice(n, 25, replace=False), "num_0"] = np.nan  # a few missing values
    return X, pd.Series(y, name="target")


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    clf = XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1)
    return Pipeline([("pre", pre), ("clf", clf)])


SEARCH_SPACE = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [3, 4, 5, 6],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.7, 0.85, 1.0],
}


def save_shap_summary(model: Pipeline, X_test: pd.DataFrame, path: Path) -> bool:
    """Save a SHAP feature-importance plot. Returns False if SHAP fails (never breaks training)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        pre, clf = model.named_steps["pre"], model.named_steps["clf"]
        data = pre.transform(X_test.head(200))
        data = data.toarray() if hasattr(data, "toarray") else np.asarray(data)
        names = pre.get_feature_names_out()
        values = shap.TreeExplainer(clf)(data)
        shap.summary_plot(values, data, feature_names=names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] SHAP summary skipped: {exc}")
        return False


def train(X, y, out_dir, n_iter=10, cv=5, min_auc=0.70, register=True, use_mlflow=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    search = RandomizedSearchCV(
        build_pipeline(X),
        SEARCH_SPACE,
        n_iter=n_iter,
        cv=StratifiedKFold(cv, shuffle=True, random_state=SEED),
        scoring="roc_auc",
        random_state=SEED,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "cv_auc": float(search.best_score_),
        "test_auc": float(roc_auc_score(y_test, proba)),
        "test_f1": float(f1_score(y_test, proba >= 0.5)),
    }
    print("metrics:", json.dumps(metrics, indent=2))

    joblib.dump(model, out_dir / "model.joblib")
    X_train.sample(min(len(X_train), 500), random_state=SEED).to_csv(
        out_dir / "reference.csv", index=False
    )
    shap_path = out_dir / "shap_summary.png"
    has_shap = save_shap_summary(model, X_test, shap_path)

    params = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    metadata = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "feature_columns": X.columns.tolist(),
        "params": params,
        "metrics": metrics,
        "passed_quality_gate": metrics["test_auc"] >= min_auc,
    }

    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("credit-risk")
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if has_shap:
                mlflow.log_artifact(str(shap_path))
            gate_ok = metadata["passed_quality_gate"]
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="cloudpickle",
                registered_model_name="credit-risk-model" if (register and gate_ok) else None,
            )
            metadata["mlflow_run_id"] = run.info.run_id
        if register and not metadata["passed_quality_gate"]:
            print(f"[gate] test AUC below {min_auc}: model NOT registered")

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true", help="use generated data (no download)")
    p.add_argument("--n-iter", type=int, default=10, help="random-search iterations")
    p.add_argument("--cv", type=int, default=5, help="cross-validation folds")
    p.add_argument("--min-auc", type=float, default=0.70, help="quality gate for registration")
    p.add_argument("--out-dir", default=os.getenv("SM_MODEL_DIR", "models"))
    args = p.parse_args()

    on_sagemaker = bool(os.getenv("SM_MODEL_DIR"))  # SageMaker has its own registry
    X, y = load_data(args.synthetic)
    train(X, y, args.out_dir, args.n_iter, args.cv, args.min_auc, use_mlflow=not on_sagemaker)


if __name__ == "__main__":
    main()
