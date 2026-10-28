import numpy as np
import pandas as pd

from monitor import drift_report


def test_drift_detected_on_shifted_data():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    same = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    shifted = pd.DataFrame({"x": rng.normal(2, 1, 1000)})
    assert not drift_report(ref, same)["drift"].any()
    assert drift_report(ref, shifted)["drift"].all()
