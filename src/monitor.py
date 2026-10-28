"""Simple data-drift check using the Population Stability Index (PSI).

python src/monitor.py --reference models/reference.csv --current new_data.csv
Rule of thumb: PSI < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant drift.
"""

import argparse
import sys

import numpy as np
import pandas as pd


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference.dropna(), edges)[0] / max(reference.notna().sum(), 1)
    cur = np.histogram(current.dropna(), edges)[0] / max(current.notna().sum(), 1)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.2):
    rows = []
    for col in reference.select_dtypes("number").columns:
        if col in current.columns:
            value = psi(reference[col], current[col])
            rows.append({"feature": col, "psi": round(value, 4), "drift": value > threshold})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--threshold", type=float, default=0.2)
    args = p.parse_args()
    report = drift_report(pd.read_csv(args.reference), pd.read_csv(args.current), args.threshold)
    print(report.to_string(index=False))
    sys.exit(1 if report["drift"].any() else 0)


if __name__ == "__main__":
    main()
