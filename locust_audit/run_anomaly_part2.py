"""
run_anomaly_part2.py — continuation of run_anomaly_investigation.py.
Reuses the already-computed thresholds (results/anomaly_thresholds.json)
instead of re-running the 5-fold CV search, since that was the expensive
part and doesn't need to be redone. Produces: ablation, country-level,
Kenya comparison, temporal robustness, permutation importance.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score

SEED = 42
TRAIN_MAX_YEAR = 2013
VAL_MAX_YEAR = 2017
CORE_CLIENTS = ["Kenya", "Egypt", "Yemen", "Ethiopia", "Sudan", "Mauritania"]
ANOMALY_COUNTRIES_OF_INTEREST = ["Kenya", "Algeria", "Tunisia", "Yemen", "Sudan", "Uganda"]
MIN_POSITIVES_FOR_METRICS = 5

RAW_FEATURES = ["PPT_T", "TMAX_T", "SOILMOISTURE_T"]
ANOMALY_FEATURES = ["PPT_anomaly", "TMAX_anomaly", "SOILMOISTURE_anomaly"]
ANOMALY_MISSING_FLAGS = ["PPT_anomaly_missing", "TMAX_anomaly_missing", "SOILMOISTURE_anomaly_missing"]
HISTORY_FEATURES = ["LOCUSTPRESENT_T_minus_1", "LOCUSTPRESENT_T_minus_2",
                     "LOCUSTPRESENT_T_minus_1_was_missing", "LOCUSTPRESENT_T_minus_2_was_missing"]
FEATURE_SETS = {
    "A_raw": RAW_FEATURES,
    "B_anomaly": ANOMALY_FEATURES + ANOMALY_MISSING_FLAGS,
    "C_raw_plus_anomaly": RAW_FEATURES + ANOMALY_FEATURES + ANOMALY_MISSING_FLAGS,
    "D_raw_plus_anomaly_plus_history": RAW_FEATURES + ANOMALY_FEATURES + ANOMALY_MISSING_FLAGS + HISTORY_FEATURES,
}

with open("results/anomaly_thresholds.json") as f:
    thresholds_used = json.load(f)

df = pd.read_csv("data/forecasting/locust_onset_anomaly_dataset.csv")
df["year_T"] = df["period_T"].str[:4].astype(int)
for c in ANOMALY_FEATURES:
    df[c] = df[c].fillna(0)
for c in HISTORY_FEATURES:
    df[c] = df[c].fillna(0)

def split_of(y):
    if y <= TRAIN_MAX_YEAR: return "train"
    elif y <= VAL_MAX_YEAR: return "val"
    return "test"
df["split"] = df["year_T"].apply(split_of)
train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)

def evaluate(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int) if y_score is not None else np.zeros(len(y_true))
    out = {"n": len(y_true), "positives": int(sum(y_true)),
           "precision": precision_score(y_true, y_pred, zero_division=0),
           "recall": recall_score(y_true, y_pred, zero_division=0),
           "f1": f1_score(y_true, y_pred, zero_division=0)}
    if y_score is not None and len(set(y_true)) > 1:
        out["pr_auc"] = average_precision_score(y_true, y_score)
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        out["pr_auc"], out["roc_auc"] = None, None
    return out

def fit_rf(tr, feats, n_estimators=150):
    m = RandomForestClassifier(n_estimators=n_estimators, max_depth=6, min_samples_leaf=10,
                                 class_weight="balanced", random_state=SEED, n_jobs=1)
    m.fit(tr[feats], tr["target_t1"])
    return m

print("Step 1/5: fitting the 4 main feature-set models (reusing saved thresholds)...")
fitted = {}
for name, feats in FEATURE_SETS.items():
    fitted[name] = (fit_rf(train, feats), feats, thresholds_used[name]["selected_threshold"])
print("done.")

# ---------------------------------------------------------------------
# 12. Ablation
# ---------------------------------------------------------------------
print("Step 2/5: ablation...")
ablation_sets = {
    "PPT_anomaly_only": ["PPT_anomaly", "PPT_anomaly_missing"],
    "SOILMOISTURE_anomaly_only": ["SOILMOISTURE_anomaly", "SOILMOISTURE_anomaly_missing"],
    "TMAX_anomaly_only": ["TMAX_anomaly", "TMAX_anomaly_missing"],
    "PPT_plus_SOILMOISTURE_anomaly": ["PPT_anomaly", "PPT_anomaly_missing",
                                        "SOILMOISTURE_anomaly", "SOILMOISTURE_anomaly_missing"],
    "all_anomaly": ANOMALY_FEATURES + ANOMALY_MISSING_FLAGS,
}
ablation_rows = []
for name, feats in ablation_sets.items():
    m = fit_rf(train, feats)
    score = m.predict_proba(test[feats])[:, 1]
    # use threshold 0.3 as a fixed, reasonable default consistent with the
    # main models' selected thresholds (0.1-0.3 range) rather than re-running
    # CV per ablation combo -- noted explicitly in the report as a
    # simplification for this specific sub-experiment
    ev = evaluate(test["target_t1"], score, threshold=0.3)
    ablation_rows.append({"ablation": name, **ev, "threshold": 0.3})
ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv("results/anomaly_ablation_results.csv", index=False)
print(ablation_df.to_string(index=False))

# ---------------------------------------------------------------------
# 9. Country-level
# ---------------------------------------------------------------------
print("\nStep 3/5: country-level...")
best_model, best_feats, best_threshold = fitted["D_raw_plus_anomaly_plus_history"]
country_rows = []
for country in sorted(set(CORE_CLIENTS) | set(ANOMALY_COUNTRIES_OF_INTEREST)):
    ctest = test[test["COUNTRYNAME"] == country]
    n_pos = int(ctest["target_t1"].sum())
    row = {"country": country, "n_candidates": len(ctest), "n_positive": n_pos}
    if n_pos < MIN_POSITIVES_FOR_METRICS or len(ctest) == 0:
        row["status"] = f"INSUFFICIENT DATA (n_pos={n_pos})"
    else:
        score = best_model.predict_proba(ctest[best_feats])[:, 1]
        m = evaluate(ctest["target_t1"], score, threshold=best_threshold)
        row.update({"precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                    "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "status": "computed"})
    country_rows.append(row)
country_df = pd.DataFrame(country_rows)
country_df.to_csv("results/anomaly_country_results.csv", index=False)
print(country_df.to_string(index=False))

# ---------------------------------------------------------------------
# 10. Kenya
# ---------------------------------------------------------------------
print("\nStep 4/5: Kenya comparison...")
kenya_test = test[test["COUNTRYNAME"] == "Kenya"]
kenya_rows = []
for name in ["A_raw", "B_anomaly", "C_raw_plus_anomaly"]:
    model, feats, t = fitted[name]
    score = model.predict_proba(kenya_test[feats])[:, 1]
    m = evaluate(kenya_test["target_t1"], score, threshold=t)
    kenya_rows.append({"feature_set": name, **m, "threshold": t})
kenya_df = pd.DataFrame(kenya_rows)
kenya_df.to_csv("results/anomaly_kenya_comparison.csv", index=False)
print(kenya_df.to_string(index=False))

kenya_pos = kenya_test[kenya_test["target_t1"] == 1]
kenya_neg = kenya_test[kenya_test["target_t1"] == 0]
kenya_dist = {
    "onset_positive_PPT_anomaly_mean": float(kenya_pos["PPT_anomaly"].mean()),
    "onset_negative_PPT_anomaly_mean": float(kenya_neg["PPT_anomaly"].mean()),
    "onset_positive_SOILMOISTURE_anomaly_mean": float(kenya_pos["SOILMOISTURE_anomaly"].mean()),
    "onset_negative_SOILMOISTURE_anomaly_mean": float(kenya_neg["SOILMOISTURE_anomaly"].mean()),
    "onset_positive_TMAX_anomaly_mean": float(kenya_pos["TMAX_anomaly"].mean()),
    "onset_negative_TMAX_anomaly_mean": float(kenya_neg["TMAX_anomaly"].mean()),
    "onset_positive_PPT_zscore_mean": float(kenya_pos["PPT_zscore"].mean()),
    "onset_negative_PPT_zscore_mean": float(kenya_neg["PPT_zscore"].mean()),
}
with open("results/anomaly_kenya_distributions.json", "w") as f:
    json.dump(kenya_dist, f, indent=2)
print(json.dumps(kenya_dist, indent=2))

# ---------------------------------------------------------------------
# 13. Temporal robustness (early/late split, full dataset)
# ---------------------------------------------------------------------
print("\nStep 5/5: temporal robustness + permutation importance...")
median_year = int(df["year_T"].median())
early = df[df["year_T"] <= median_year]
late = df[df["year_T"] > median_year]
robustness_rows = []
for name, feats in FEATURE_SETS.items():
    m = fit_rf(early, feats)
    score = m.predict_proba(late[feats])[:, 1]
    ev = evaluate(late["target_t1"], score)
    robustness_rows.append({"feature_set": name, **ev,
                              "train_period": f"<= {median_year}", "test_period": f"> {median_year}"})
robustness_df = pd.DataFrame(robustness_rows)
robustness_df.to_csv("results/anomaly_temporal_robustness.csv", index=False)
print(robustness_df.to_string(index=False))

# ---------------------------------------------------------------------
# 11. Permutation importance for model D
# ---------------------------------------------------------------------
perm = permutation_importance(best_model, test[best_feats], test["target_t1"],
                                n_repeats=10, random_state=SEED, scoring="average_precision")
importance_df = pd.DataFrame({
    "feature": best_feats,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
importance_df.to_csv("results/anomaly_permutation_importance.csv", index=False)
print(importance_df.to_string(index=False))

print("\nAll done.")
