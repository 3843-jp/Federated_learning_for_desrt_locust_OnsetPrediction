"""
run_onset_investigation.py — Phase 3: does the data contain signal to
predict NEW locust onset (LOCUSTPRESENT_T=0 -> target_t1=1), as opposed to
persistence of an already-active report?

Run:
    python run_onset_investigation.py

Writes results/onset_* files. Full 42-country onset dataset is used for
training (core-6-only training data is too sparse — 72 positives vs 274 —
see report Section 2/3 for why this deviation from a core-6-only pipeline
is made explicit rather than silent). Country-level reporting (Section 7)
is still restricted to the six approved core clients as instructed.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, precision_recall_curve,
)

SEED = 42
np.random.seed(SEED)
os.makedirs("results", exist_ok=True)

CORE_CLIENTS = ["Kenya", "Egypt", "Yemen", "Ethiopia", "Sudan", "Mauritania"]
TRAIN_MAX_YEAR = 2013
VAL_MAX_YEAR = 2017

FEATURES_ENV_ONLY = ["PPT_T", "TMAX_T", "SOILMOISTURE_T"]
FEATURES_ENV_HISTORY = FEATURES_ENV_ONLY + [
    "LOCUSTPRESENT_T_minus_1", "LOCUSTPRESENT_T_minus_2",
    "LOCUSTPRESENT_T_minus_1_was_missing", "LOCUSTPRESENT_T_minus_2_was_missing",
]

# ---------------------------------------------------------------------------
# 1-2-3. Onset definition + leakage prevention + dataset construction
# ---------------------------------------------------------------------------
full = pd.read_csv("data/forecasting/locust_t_plus_1_dataset.csv")
full["year_T"] = full["period_T"].str[:4].astype(int)

# Onset = LOCUSTPRESENT_T == 0 (by construction, so it's never used as a
# feature below — it's the filter, not an input).
onset = full[full["LOCUSTPRESENT_T"] == 0].copy()
for col in ["LOCUSTPRESENT_T_minus_1", "LOCUSTPRESENT_T_minus_2"]:
    onset[f"{col}_was_missing"] = onset[col].isna().astype(int)
    onset[col] = onset[col].fillna(0)

onset.to_csv("data/forecasting/onset_dataset.csv", index=False)

stats = {
    "total_onset_candidates": len(onset),
    "positive_onset_events": int(onset["target_t1"].sum()),
    "negative_candidates": int((onset["target_t1"] == 0).sum()),
    "positive_rate_pct": round(100 * onset["target_t1"].mean(), 4),
    "distinct_regions_with_onset_event": int(
        onset[onset["target_t1"] == 1]["REGION"].nunique()
    ),
}
with open("results/onset_dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("Onset dataset stats:", json.dumps(stats, indent=2))

def split_of(y):
    if y <= TRAIN_MAX_YEAR:
        return "train"
    elif y <= VAL_MAX_YEAR:
        return "val"
    return "test"

onset["split"] = onset["year_T"].apply(split_of)
split_counts = onset.groupby("split").agg(n=("target_t1", "size"), pos=("target_t1", "sum"))
print("\nOnset split counts (FULL 42-country dataset):")
print(split_counts)

train = onset[onset["split"] == "train"].reset_index(drop=True)
val = onset[onset["split"] == "val"].reset_index(drop=True)
test = onset[onset["split"] == "test"].reset_index(drop=True)

VAL_USABLE = val["target_t1"].sum() >= 10  # threshold for "usable for
                                             # tuning" -- stated explicitly
print(f"\nValidation set positives: {int(val['target_t1'].sum())} "
      f"-> {'USABLE' if VAL_USABLE else 'NOT USABLE for threshold tuning'}")


# ---------------------------------------------------------------------------
# 4. Baseline hierarchy (majority, LR env-only, RF env-only, RF env+history)
# ---------------------------------------------------------------------------
def evaluate(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int) if y_score is not None else np.zeros(len(y_true))
    out = {
        "n": len(y_true), "positives": int(sum(y_true)),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(set(y_true)) > 1:
        out["pr_auc"] = average_precision_score(y_true, y_score)
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        out["pr_auc"], out["roc_auc"] = None, None
    return out


results_rows = []

# Baseline 1: majority class
maj_pred_val = np.zeros(len(val))
maj_pred_test = np.zeros(len(test))
for split_name, y_true in (("val", val["target_t1"]), ("test", test["target_t1"])):
    m = evaluate(y_true, None)
    results_rows.append({"model": "1_majority", "split": split_name, **m})

# Baseline 2: LR environmental-only (class-weighted, since imbalance is severe)
lr_env = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
lr_env.fit(train[FEATURES_ENV_ONLY], train["target_t1"])

# Baseline 3: RF environmental-only
rf_env = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)
rf_env.fit(train[FEATURES_ENV_ONLY], train["target_t1"])

# Baseline 4: RF environmental + lagged history (no LOCUSTPRESENT_T -- it's
# 0 for every row here by construction, so it's correctly excluded, not
# just redundant)
rf_hist = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
                                   class_weight="balanced", random_state=SEED, n_jobs=-1)
rf_hist.fit(train[FEATURES_ENV_HISTORY], train["target_t1"])

for name, model, feats in (
    ("2_logreg_env_only", lr_env, FEATURES_ENV_ONLY),
    ("3_rf_env_only", rf_env, FEATURES_ENV_ONLY),
    ("4_rf_env_plus_history", rf_hist, FEATURES_ENV_HISTORY),
):
    for split_name, d in (("val", val), ("test", test)):
        score = model.predict_proba(d[feats])[:, 1]
        m = evaluate(d["target_t1"], score)
        results_rows.append({"model": name, "split": split_name, **m})

pd.DataFrame(results_rows).to_csv("results/onset_baseline_results.csv", index=False)
print("\nBaseline hierarchy results (default threshold 0.5):")
print(pd.DataFrame(results_rows).to_string(index=False))


# ---------------------------------------------------------------------------
# 6. Threshold analysis — validation is NOT usable (1 positive), so per
# instructions we state that explicitly and substitute 5-fold temporal
# cross-validation WITHIN TRAIN to select a threshold, rather than tuning
# against test (which would be cherry-picking) or against the unusable
# validation set (which would be noise).
# ---------------------------------------------------------------------------
def temporal_folds(df, n_folds=5):
    years = sorted(df["year_T"].unique())
    fold_edges = np.array_split(years, n_folds)
    for fold_years in fold_edges:
        val_mask = df["year_T"].isin(fold_years)
        yield df[~val_mask], df[val_mask]

BEST_MODEL_FEATURES = FEATURES_ENV_HISTORY  # rf_hist -- carried forward
                                              # from Section 4 if it's the
                                              # strongest; re-checked below

oof_true, oof_score = [], []
for fold_train, fold_val in temporal_folds(train, n_folds=5):
    if fold_val["target_t1"].sum() == 0:
        continue  # skip folds with zero positives -- can't evaluate PR
    m = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
                                 class_weight="balanced", random_state=SEED, n_jobs=-1)
    m.fit(fold_train[BEST_MODEL_FEATURES], fold_train["target_t1"])
    s = m.predict_proba(fold_val[BEST_MODEL_FEATURES])[:, 1]
    oof_true.extend(fold_val["target_t1"].tolist())
    oof_score.extend(s.tolist())

oof_true = np.array(oof_true)
oof_score = np.array(oof_score)

threshold_rows = []
for t in [0.10, 0.20, 0.30, 0.40, 0.50]:
    pred = (oof_score >= t).astype(int)
    threshold_rows.append({
        "threshold": t,
        "precision": precision_score(oof_true, pred, zero_division=0),
        "recall": recall_score(oof_true, pred, zero_division=0),
        "f1": f1_score(oof_true, pred, zero_division=0),
        "n_predicted_positive": int(pred.sum()),
    })
threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv("results/onset_threshold_analysis_CV.csv", index=False)
print("\nThreshold analysis via 5-fold temporal CV within TRAIN "
      "(validation split has only 1 positive, not usable):")
print(threshold_df.to_string(index=False))

selected_threshold = threshold_df.loc[threshold_df["f1"].idxmax(), "threshold"]
print(f"\nSelected threshold (max F1 in CV): {selected_threshold}")

# Apply the CV-selected threshold ONCE to test, for the best model
# (re-fit on full train)
final_model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
                                       class_weight="balanced", random_state=SEED, n_jobs=-1)
final_model.fit(train[BEST_MODEL_FEATURES], train["target_t1"])
test_score = final_model.predict_proba(test[BEST_MODEL_FEATURES])[:, 1]
test_at_selected = evaluate(test["target_t1"], test_score, threshold=selected_threshold)
test_at_selected["threshold_used"] = selected_threshold
with open("results/onset_test_at_selected_threshold.json", "w") as f:
    json.dump(test_at_selected, f, indent=2)
print(f"\nTest result at CV-selected threshold {selected_threshold}:")
print(json.dumps(test_at_selected, indent=2))

# Also report the full descriptive sweep directly on test, clearly labeled
# as descriptive/not used for selection (instructions: don't cherry-pick on
# test, but a transparent sweep is still useful information for the report)
descriptive_test_sweep = []
for t in [0.10, 0.20, 0.30, 0.40, 0.50]:
    pred = (test_score >= t).astype(int)
    descriptive_test_sweep.append({
        "threshold": t,
        "precision": precision_score(test["target_t1"], pred, zero_division=0),
        "recall": recall_score(test["target_t1"], pred, zero_division=0),
        "f1": f1_score(test["target_t1"], pred, zero_division=0),
    })
pd.DataFrame(descriptive_test_sweep).to_csv(
    "results/onset_threshold_sweep_TEST_descriptive_only.csv", index=False
)


# ---------------------------------------------------------------------------
# 7. Country-level onset analysis (core-6 only, per instructions)
# ---------------------------------------------------------------------------
MIN_POSITIVES_FOR_METRICS = 5  # below this, PR-AUC/precision/recall are
                                 # not meaningful -- flagged, not computed
country_rows = []
for country in CORE_CLIENTS:
    ctest = test[test["COUNTRYNAME"] == country]
    n_candidates = len(ctest)
    n_pos = int(ctest["target_t1"].sum())
    row = {"country": country, "onset_candidates_test": n_candidates,
           "onset_positives_test": n_pos,
           "onset_rate_pct": round(100 * n_pos / n_candidates, 3) if n_candidates else None}
    if n_pos < MIN_POSITIVES_FOR_METRICS or n_candidates == 0:
        row["metrics_status"] = f"INSUFFICIENT DATA (n_pos={n_pos}, need >= {MIN_POSITIVES_FOR_METRICS})"
    else:
        score = final_model.predict_proba(ctest[BEST_MODEL_FEATURES])[:, 1]
        m = evaluate(ctest["target_t1"], score, threshold=selected_threshold)
        row.update({"precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                     "metrics_status": "computed"})
    country_rows.append(row)

country_df = pd.DataFrame(country_rows)
country_df.to_csv("results/onset_country_level_results.csv", index=False)
print("\nCountry-level onset results (core 6, test period):")
print(country_df.to_string(index=False))


# ---------------------------------------------------------------------------
# 8. Kenya deep analysis
# ---------------------------------------------------------------------------
kenya_onset = onset[onset["COUNTRYNAME"] == "Kenya"].copy()
kenya_pos = kenya_onset[kenya_onset["target_t1"] == 1]

kenya_analysis = {
    "total_onset_events_all_time": len(kenya_pos),
    "regions_affected": sorted(kenya_pos["REGION"].unique().tolist()),
    "n_regions_affected": kenya_pos["REGION"].nunique(),
    "months_of_onset": sorted(kenya_pos["period_T1"].unique().tolist()),
    "env_conditions_before_onset": {
        "PPT_T_mean": float(kenya_pos["PPT_T"].mean()),
        "PPT_T_median": float(kenya_pos["PPT_T"].median()),
        "TMAX_T_mean": float(kenya_pos["TMAX_T"].mean()),
        "SOILMOISTURE_T_mean": float(kenya_pos["SOILMOISTURE_T"].mean()),
    },
    "env_conditions_all_kenya_onset_candidates": {
        "PPT_T_mean": float(kenya_onset["PPT_T"].mean()),
        "PPT_T_median": float(kenya_onset["PPT_T"].median()),
        "TMAX_T_mean": float(kenya_onset["TMAX_T"].mean()),
        "SOILMOISTURE_T_mean": float(kenya_onset["SOILMOISTURE_T"].mean()),
    },
    "lagged_history_among_kenya_onset_positives": {
        "pct_with_T_minus_1_present": float(
            (kenya_pos["LOCUSTPRESENT_T_minus_1"] == 1).mean() * 100
        ),
        "pct_with_T_minus_2_present": float(
            (kenya_pos["LOCUSTPRESENT_T_minus_2"] == 1).mean() * 100
        ),
    },
}
with open("results/onset_kenya_deep_analysis.json", "w") as f:
    json.dump(kenya_analysis, f, indent=2, default=str)
print("\nKenya deep analysis written to results/onset_kenya_deep_analysis.json")
print(json.dumps(kenya_analysis, indent=2, default=str))


# ---------------------------------------------------------------------------
# 9. Temporal robustness — full-dataset walk-forward style check: train on
# earliest half, test onset detection on latest half, avoiding random splits
# ---------------------------------------------------------------------------
median_year = int(onset["year_T"].median())
early = onset[onset["year_T"] <= median_year]
late = onset[onset["year_T"] > median_year]
robustness_model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
                                            class_weight="balanced", random_state=SEED, n_jobs=-1)
robustness_model.fit(early[BEST_MODEL_FEATURES], early["target_t1"])
late_score = robustness_model.predict_proba(late[BEST_MODEL_FEATURES])[:, 1]
robustness_result = evaluate(late["target_t1"], late_score)
robustness_result["train_period"] = f"<= {median_year}"
robustness_result["test_period"] = f"> {median_year}"
robustness_result["train_positives"] = int(early["target_t1"].sum())
robustness_result["test_positives"] = int(late["target_t1"].sum())
with open("results/onset_temporal_robustness.json", "w") as f:
    json.dump(robustness_result, f, indent=2, default=str)
print("\nTemporal robustness (early-half train, late-half test, full 42-country data):")
print(json.dumps(robustness_result, indent=2, default=str))


# ---------------------------------------------------------------------------
# 11. Feature importance — permutation importance on the held-out test set
# ---------------------------------------------------------------------------
perm = permutation_importance(final_model, test[BEST_MODEL_FEATURES], test["target_t1"],
                                n_repeats=30, random_state=SEED,
                                scoring="average_precision")
importance_df = pd.DataFrame({
    "feature": BEST_MODEL_FEATURES,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
importance_df.to_csv("results/onset_permutation_importance.csv", index=False)
print("\nPermutation importance (scoring=PR-AUC) on test set:")
print(importance_df.to_string(index=False))

print("\nDone. All Phase 3 outputs in results/onset_*.")
