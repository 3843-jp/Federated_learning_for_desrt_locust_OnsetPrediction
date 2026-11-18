"""
phase6_data_prep.py — Part 1 (freeze dataset) and Part 3 (client eligibility).

FROZEN FEATURE SET (Phase 4's best centralized result, "raw + anomaly",
current month only -- no windows, no anomaly-change, no new variables):

    PPT_T, TMAX_T, SOILMOISTURE_T,
    PPT_anomaly, TMAX_anomaly, SOILMOISTURE_anomaly,
    PPT_anomaly_missing, TMAX_anomaly_missing, SOILMOISTURE_anomaly_missing

9 features total. Documented here once; every other Phase 6 script imports
FROZEN_FEATURES from this module rather than redefining it.
"""
import json
import numpy as np
import pandas as pd

TRAIN_MAX_YEAR = 2013
VAL_MAX_YEAR = 2017

FROZEN_FEATURES = [
    "PPT_T", "TMAX_T", "SOILMOISTURE_T",
    "PPT_anomaly", "TMAX_anomaly", "SOILMOISTURE_anomaly",
    "PPT_anomaly_missing", "TMAX_anomaly_missing", "SOILMOISTURE_anomaly_missing",
]

MIN_TEST_POSITIVES_CORE = 5     # Category A cutoff
MIN_TRAIN_POSITIVES_HETERO = 5  # Category B cutoff


def load_frozen_dataset():
    df = pd.read_csv("data/forecasting/locust_onset_anomaly_dataset.csv")
    df["year_T"] = df["period_T"].str[:4].astype(int)
    for c in ["PPT_anomaly", "TMAX_anomaly", "SOILMOISTURE_anomaly"]:
        df[c] = df[c].fillna(0)  # missingness already flagged in *_anomaly_missing

    def split_of(y):
        if y <= TRAIN_MAX_YEAR:
            return "train"
        elif y <= VAL_MAX_YEAR:
            return "val"
        return "test"
    df["split"] = df["year_T"].apply(split_of)
    return df


if __name__ == "__main__":
    df = load_frozen_dataset()
    print(f"Frozen dataset: {len(df)} rows, features = {FROZEN_FEATURES}")
    print(df.groupby("split").agg(n=("target_t1", "size"), pos=("target_t1", "sum")))

    rows = []
    for country, g in df.groupby("COUNTRYNAME"):
        tr = g[g["split"] == "train"]
        va = g[g["split"] == "val"]
        te = g[g["split"] == "test"]
        row = {
            "country": country,
            "train_samples": len(tr), "val_samples": len(va), "test_samples": len(te),
            "train_positives": int(tr["target_t1"].sum()),
            "val_positives": int(va["target_t1"].sum()),
            "test_positives": int(te["target_t1"].sum()),
            "n_regions": g["REGION"].nunique(),
        }
        if row["test_positives"] >= MIN_TEST_POSITIVES_CORE:
            row["category"] = "A_core_evaluation"
        elif row["train_positives"] >= MIN_TRAIN_POSITIVES_HETERO:
            row["category"] = "B_heterogeneity_client"
        else:
            row["category"] = "excluded_insufficient_data"
        rows.append(row)

    elig = pd.DataFrame(rows).sort_values(["category", "train_positives"], ascending=[True, False])
    elig.to_csv("results/phase6_client_eligibility.csv", index=False)
    print("\n=== Client eligibility ===")
    print(elig.to_string(index=False))

    category_a = elig[elig["category"] == "A_core_evaluation"]["country"].tolist()
    category_b = elig[elig["category"] == "B_heterogeneity_client"]["country"].tolist()
    print(f"\nCategory A (core evaluation clients): {category_a}")
    print(f"Category B (heterogeneity clients, train-only): {category_b}")

    with open("results/phase6_client_categories.json", "w") as f:
        json.dump({"category_a": category_a, "category_b": category_b,
                    "frozen_features": FROZEN_FEATURES}, f, indent=2)
