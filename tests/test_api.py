import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(trained_dir, monkeypatch):
    out, _, _ = trained_dir
    monkeypatch.setenv("MODEL_DIR", str(out))
    import app as app_module

    app_module.get_model.cache_clear()
    return TestClient(app_module.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict(client, trained_dir):
    _, _, X = trained_dir
    head = X.head(3)
    records = head.astype(object).where(head.notna(), None).to_dict(orient="records")
    r = client.post("/predict", json={"records": records})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 3
    assert all(0.0 <= p["default_probability"] <= 1.0 for p in preds)


def test_predict_missing_columns(client):
    r = client.post("/predict", json={"records": [{"num_0": 1.0}]})
    assert r.status_code == 422
