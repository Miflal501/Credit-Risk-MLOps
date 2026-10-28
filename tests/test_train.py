def test_training_creates_artifacts(trained_dir):
    out, meta, _ = trained_dir
    assert (out / "model.joblib").exists()
    assert (out / "metadata.json").exists()
    assert (out / "reference.csv").exists()
    assert meta["metrics"]["test_auc"] > 0.6
    assert meta["passed_quality_gate"] is True
