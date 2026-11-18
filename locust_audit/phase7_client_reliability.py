"""
phase7_client_reliability.py — Part 3: per-client reliability statistics,
computed from TRAINING data only (never test), for use by the adaptive
weighting formulas defined in phase7_weighting.py.
"""
import json
import pandas as pd

from phase6_data_prep import load_frozen_dataset

df = load_frozen_dataset()
with open("results/phase6_client_categories.json") as f:
    cats = json.load(f)
ALL_FL_CLIENTS = cats["category_a"] + cats["category_b"]

train = df[df["split"] == "train"]
val = df[df["split"] == "val"]

rows = []
for country in ALL_FL_CLIENTS:
    ctr = train[train["COUNTRYNAME"] == country]
    cva = val[val["COUNTRYNAME"] == country]
    n = len(ctr)
    p = int(ctr["target_t1"].sum())
    rows.append({
        "country": country,
        "train_samples": n,
        "train_positive": p,
        "train_negative": n - p,
        "positive_rate_pct": round(100 * p / n, 4) if n else None,
        "val_samples": len(cva),
        "val_positive": int(cva["target_t1"].sum()),
    })

rel_df = pd.DataFrame(rows).sort_values("train_positive", ascending=False)
rel_df.to_csv("results/phase7_client_reliability.csv", index=False)
print(rel_df.to_string(index=False))
