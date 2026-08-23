"""
sanity_check.py — confirms the deployed inference path matches the notebook.

Takes a few rows straight from the raw CSV (not the pre-scaled X_test),
runs them through inference.predict(), and compares against the known
Machine failure label. This validates preprocessing end-to-end, not just
the model's raw score on already-scaled data.

Run: python sanity_check.py
"""

import pandas as pd
from inference import predict

RAW_CSV = r"E:\AIS bsc\predictive-maintenance\data\raw\AI4I_2020_Predictive_Maintenance_Dataset_(UCI).csv"

df = pd.read_csv(RAW_CSV)

# Grab a few known failures and a few known non-failures
failures = df[df["Machine failure"] == 1].sample(3, random_state=42)
normals = df[df["Machine failure"] == 0].sample(3, random_state=42)
sample = pd.concat([failures, normals])

print(f"{'Type':<5}{'Actual':<8}{'Pred':<6}{'Prob':<8}{'Risk'}")
print("-" * 40)

for _, row in sample.iterrows():
    raw = {
        "Type": row["Type"],
        "Air temperature [K]": row["Air temperature [K]"],
        "Process temperature [K]": row["Process temperature [K]"],
        "Rotational speed [rpm]": row["Rotational speed [rpm]"],
        "Torque [Nm]": row["Torque [Nm]"],
        "Tool wear [min]": row["Tool wear [min]"],
    }
    result = predict(raw)
    print(
        f"{row['Type']:<5}{row['Machine failure']:<8}"
        f"{result['prediction']:<6}{result['failure_probability']:<8.3f}"
        f"{result['risk_level']}"
    )
