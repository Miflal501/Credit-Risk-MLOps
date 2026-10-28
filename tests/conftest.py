import os

import pytest

from train import load_data, train


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    db = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{db}"
    X, y = load_data(synthetic=True)
    meta = train(X, y, out, n_iter=2, cv=3, min_auc=0.6, register=False)
    return out, meta, X
