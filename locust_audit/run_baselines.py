"""
run_baselines.py — Phase 2: baseline hierarchy on the approved
locust_t_plus_1_dataset.csv, restricted to the six approved core clients.

Run:
    python run_baselines.py

Writes:
    results/baseline_results.csv       -- one row per experiment
    results/per_country_results.csv    -- per-country breakdown for the
                                           best model
    results/error_analysis.json
    results/config_used.json           -- exact seed/split/features/clients,
                                           for reproducibility
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
)

# ---------------------------------------------------------------------------
# CONFIG — everything experiment-relevant lives here, nothing hardcoded
# further down, per the reproducibility requirement.
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

DATA_PATH = "data/forecasting/locust_t_plus_1_dataset.csv"
CORE_CLIENTS = ["Kenya", "Egypt", "Yemen", "Ethiopia", "Sudan", "Mauritania"]

# Temporal split — chosen by inspecting the actual year_T distribution
# (see report Section 2 for the inspection that produced these numbers,
# not picked arbitrarily): 1985-2013 train (preserves Mauritania's full
# history, which ends 2013), 2014-2017 validation, 2018-2020 test. The test
# window deliberately includes the real 2019-2020 East Africa/Horn upsurge,
# per Phase 0's original recommendation to hold out the upsurge as the hard
# generalization test rather than average it into training.
TRAIN_MAX_YEAR = 2013
VAL_MAX_YEAR = 2017
# test = anything after VAL_MAX_YEAR, i.e. 2018-2020

FEATURE_SETS = {
    "A_environmental_only": ["PPT_T", "TMAX_T", "SOILMOISTURE_T"],
    "B_env_plus_current_presence": ["PPT_T", "TMAX_T", "SOILMOISTURE_T", "LOCUSTPRESENT_T"],
    "C_env_plus_current_plus_history": [
        "PPT_T", "TMAX_T", "SOILMOISTURE_T",
        "LOCUSTPRESENT_T", "LOCUSTPRESENT_T_minus_1", "LOCUSTPRESENT_T_minus_2",
    ],
}
PRIMARY_FEATURE_SET = "C_env_plus_current_plus_history"  # used for baselines
                                                            # 1-4; ablation
                                                            # (Section 8)
                                                            # sweeps all three

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

with open("results/config_used.json", "w") as f:
    json.dump({
        "seed": SEED,
        "core_clients": CORE_CLIENTS,
        "train_max_year": TRAIN_MAX_YEAR,
        "val_max_year": VAL_MAX_YEAR,
        "feature_sets": FEATURE_SETS,
        "primary_feature_set": PRIMARY_FEATURE_SET,
    }, f, indent=2)


# ---------------------------------------------------------------------------
# 1. Data loading / preprocessing
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["year_T"] = df["period_T"].str[:4].astype(int)
    df = df[df["COUNTRYNAME"].isin(CORE_CLIENTS)].copy()

    def split_of(y):
        if y <= TRAIN_MAX_YEAR:
            return "train"
        elif y <= VAL_MAX_YEAR:
            return "val"
        return "test"

    df["split"] = df["year_T"].apply(split_of)

    # Missing-history rows (LOCUSTPRESENT_T-1/T-2 = NaN because no true
    # prior month exists, per Phase 1C Section 3) — for the tree/logistic
    # baselines we need a concrete value. We impute NaN history as 0
    # ("unknown treated as no report") and add an explicit missingness
    # flag rather than silently conflating "confirmed absent" with
    # "no data" (continuing the Phase 1C naming discipline).
    for col in ["LOCUSTPRESENT_T_minus_1", "LOCUSTPRESENT_T_minus_2"]:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)

    return df


def split_summary(df):
    return df.groupby("split").agg(
        n=("target_t1", "size"), positives=("target_t1", "sum")
    ).reindex(["train", "val", "test"])


# ---------------------------------------------------------------------------
# Metrics helper — every experiment reports the same panel
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, y_score=None, label=""):
    result = {
        "n": len(y_true),
        "positive_count": int(sum(y_true)),
        "negative_count": int(len(y_true) - sum(y_true)),
        "predicted_positive_count": int(sum(y_pred)),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(set(y_true)) > 1:
        result["pr_auc"] = average_precision_score(y_true, y_score)
        result["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result["confusion_matrix"] = {"tn": int(tn), "fp": int(fp),
                                    "fn": int(fn), "tp": int(tp)}
    return result


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run_all():
    df = load_data()
    print("Split sizes (core clients only):")
    print(split_summary(df))
    print()

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    y_train, y_val, y_test = train["target_t1"], val["target_t1"], test["target_t1"]

    experiment_rows = []

    def log_experiment(name, features, weighting, split_name, y_true, y_pred, y_score):
        m = evaluate(y_true, y_pred, y_score)
        row = {
            "model": name,
            "features": ",".join(features) if features else "n/a",
            "class_weighting": weighting,
            "train_period": f"<= {TRAIN_MAX_YEAR}",
            "val_period": f"{TRAIN_MAX_YEAR+1}-{VAL_MAX_YEAR}",
            "test_period": f"{VAL_MAX_YEAR+1}-2020",
            "eval_split": split_name,
            **{k: v for k, v in m.items() if k != "confusion_matrix"},
            "confusion_matrix": json.dumps(m["confusion_matrix"]),
        }
        experiment_rows.append(row)
        return m

    # --- Experiment 1: Persistence baseline ---
    print("=== Experiment 1: Persistence baseline ===")
    for split_name, y_true, present in (
        ("val", y_val, val["LOCUSTPRESENT_T"]),
        ("test", y_test, test["LOCUSTPRESENT_T"]),
    ):
        y_pred = present.astype(int).values
        m = log_experiment("1_persistence", ["LOCUSTPRESENT_T"], "n/a",
                            split_name, y_true, y_pred, y_pred)
        print(f"  [{split_name}] precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} pr_auc={m['pr_auc']}")

    # --- Experiment 2: Majority-class baseline ---
    print("\n=== Experiment 2: Majority-class baseline ===")
    majority_class = int(y_train.mode()[0])
    for split_name, y_true in (("val", y_val), ("test", y_test)):
        y_pred = np.full(len(y_true), majority_class)
        m = log_experiment("2_majority_class", [], "n/a", split_name,
                            y_true, y_pred, None)
        print(f"  [{split_name}] majority_class={majority_class} "
              f"precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")

    # --- Experiments 3-4 + class-weighting comparison + feature ablation ---
    # Combined loop: for each feature set x each model type x each
    # weighting choice, fit on train, evaluate on val and test.
    all_history_flag_cols = ["LOCUSTPRESENT_T_minus_1_was_missing",
                              "LOCUSTPRESENT_T_minus_2_was_missing"]

    fitted_models = {}
    for feat_name, feat_cols in FEATURE_SETS.items():
        # include missingness flags only when history features are in play
        cols = feat_cols + (
            all_history_flag_cols if "LOCUSTPRESENT_T_minus_1" in feat_cols else []
        )
        X_train, X_val, X_test = train[cols], val[cols], test[cols]

        for weighting in ("none", "balanced"):
            cw = None if weighting == "none" else "balanced"

            # --- Logistic Regression ---
            lr = LogisticRegression(max_iter=1000, class_weight=cw, random_state=SEED)
            lr.fit(X_train, y_train)
            for split_name, X_eval, y_true in (
                ("val", X_val, y_val), ("test", X_test, y_test)
            ):
                y_score = lr.predict_proba(X_eval)[:, 1]
                y_pred = (y_score >= 0.5).astype(int)
                m = log_experiment(f"3_logreg_{feat_name}", cols, weighting,
                                    split_name, y_true, y_pred, y_score)
            if feat_name == PRIMARY_FEATURE_SET and weighting == "balanced":
                fitted_models["logreg"] = (lr, cols)

            # --- Random Forest ---
            rf = RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=5,
                class_weight=cw, random_state=SEED, n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            for split_name, X_eval, y_true in (
                ("val", X_val, y_val), ("test", X_test, y_test)
            ):
                y_score = rf.predict_proba(X_eval)[:, 1]
                y_pred = (y_score >= 0.5).astype(int)
                m = log_experiment(f"4_randomforest_{feat_name}", cols, weighting,
                                    split_name, y_true, y_pred, y_score)
            if feat_name == PRIMARY_FEATURE_SET and weighting == "balanced":
                fitted_models["random_forest"] = (rf, cols)

    results_df = pd.DataFrame(experiment_rows)
    results_df.to_csv("results/baseline_results.csv", index=False)
    print(f"\nWrote {len(results_df)} experiment rows to results/baseline_results.csv")

    # -----------------------------------------------------------------
    # Per-country evaluation for the best model (Random Forest, primary
    # feature set, balanced weighting, evaluated on TEST)
    # -----------------------------------------------------------------
    best_model, best_cols = fitted_models["random_forest"]
    country_rows = []
    for country in CORE_CLIENTS:
        ctest = test[test["COUNTRYNAME"] == country]
        if len(ctest) == 0:
            country_rows.append({"country": country, "n": 0, "positive_count": 0,
                                  "note": "no test-period rows for this client"})
            continue
        y_true_c = ctest["target_t1"]
        y_score_c = best_model.predict_proba(ctest[best_cols])[:, 1]
        y_pred_c = (y_score_c >= 0.5).astype(int)
        m = evaluate(y_true_c, y_pred_c, y_score_c)
        country_rows.append({
            "country": country,
            "n": m["n"], "positive_count": m["positive_count"],
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
        })
    per_country_df = pd.DataFrame(country_rows)
    per_country_df.to_csv("results/per_country_results.csv", index=False)
    print("\nPer-country test-set performance (Random Forest, primary features):")
    print(per_country_df.to_string(index=False))

    # -----------------------------------------------------------------
    # Error analysis for the best model on TEST
    # -----------------------------------------------------------------
    y_score_test = best_model.predict_proba(test[best_cols])[:, 1]
    y_pred_test = (y_score_test >= 0.5).astype(int)
    test_eval = test.copy()
    test_eval["y_true"] = y_test.values
    test_eval["y_pred"] = y_pred_test
    test_eval["y_score"] = y_score_test

    false_negatives = test_eval[(test_eval["y_true"] == 1) & (test_eval["y_pred"] == 0)]
    false_positives = test_eval[(test_eval["y_true"] == 0) & (test_eval["y_pred"] == 1)]

    error_analysis = {
        "total_test_rows": len(test_eval),
        "false_negatives_count": len(false_negatives),
        "false_positives_count": len(false_positives),
        "false_negatives_by_country": false_negatives["COUNTRYNAME"].value_counts().to_dict(),
        "false_positives_by_country": false_positives["COUNTRYNAME"].value_counts().to_dict(),
        "false_negatives_by_period": false_negatives["period_T"].value_counts().head(10).to_dict(),
        "recall_by_country": per_country_df.set_index("country")["recall"].to_dict()
                              if "recall" in per_country_df.columns else {},
        "precision_by_country": per_country_df.set_index("country")["precision"].to_dict()
                                 if "precision" in per_country_df.columns else {},
    }
    with open("results/error_analysis.json", "w") as f:
        json.dump(error_analysis, f, indent=2, default=str)

    print("\nError analysis written to results/error_analysis.json")
    print(f"False negatives: {len(false_negatives)} / {int(y_test.sum())} true positives missed")
    print(f"False positives: {len(false_positives)}")

    return results_df, per_country_df, error_analysis


if __name__ == "__main__":
    run_all()
