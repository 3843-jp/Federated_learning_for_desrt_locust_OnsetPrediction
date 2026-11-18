"""
run_anomaly_investigation.py — Phase 4: does replacing/augmenting raw
environmental values with region-specific climate anomalies improve onset
detection over Phase 3's raw-feature result?

Run:
    python run_anomaly_investigation.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
)

SEED = 42
np.random.seed(SEED)

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


def load():
    df = pd.read_csv("data/forecasting/locust_onset_anomaly_dataset.csv")
    df["year_T"] = df["period_T"].str[:4].astype(int)
    # Impute anomaly NaNs (insufficient/degenerate climatology history) to 0
    # with the missingness flag already carried from build_anomaly_dataset.py
    # -- same "unknown, not confirmed 0" discipline used throughout.
    for c in ANOMALY_FEATURES:
        df[c] = df[c].fillna(0)
    for c in HISTORY_FEATURES:
        if df[c].isna().any():
            df[c] = df[c].fillna(0)

    def split_of(y):
        if y <= TRAIN_MAX_YEAR:
            return "train"
        elif y <= VAL_MAX_YEAR:
            return "val"
        return "test"
    df["split"] = df["year_T"].apply(split_of)
    return df


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


def fit_rf(train, feats):
    m = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=10,
                                 class_weight="balanced", random_state=SEED, n_jobs=1)
    m.fit(train[feats], train["target_t1"])
    return m


def temporal_cv_threshold(train, feats, n_folds=5):
    years = sorted(train["year_T"].unique())
    fold_edges = np.array_split(years, n_folds)
    oof_true, oof_score = [], []
    for fold_years in fold_edges:
        val_mask = train["year_T"].isin(fold_years)
        f_train, f_val = train[~val_mask], train[val_mask]
        if f_val["target_t1"].sum() == 0:
            continue
        m = fit_rf(f_train, feats)
        s = m.predict_proba(f_val[feats])[:, 1]
        oof_true.extend(f_val["target_t1"].tolist())
        oof_score.extend(s.tolist())
    oof_true, oof_score = np.array(oof_true), np.array(oof_score)
    best_t, best_f1 = 0.5, -1
    sweep = []
    for t in [0.10, 0.20, 0.30, 0.40, 0.50]:
        pred = (oof_score >= t).astype(int)
        f1 = f1_score(oof_true, pred, zero_division=0)
        sweep.append({"threshold": t,
                       "precision": precision_score(oof_true, pred, zero_division=0),
                       "recall": recall_score(oof_true, pred, zero_division=0),
                       "f1": f1})
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, sweep


def main():
    df = load()
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Train n={len(train)} pos={int(train['target_t1'].sum())}, "
          f"Val n={len(val)} pos={int(val['target_t1'].sum())} "
          f"({'NOT usable for tuning' if val['target_t1'].sum() < 10 else 'usable'}), "
          f"Test n={len(test)} pos={int(test['target_t1'].sum())}")

    # -----------------------------------------------------------------
    # 4/6/7. Model A-D, majority baseline, LR comparison, threshold via
    # temporal CV within train, applied once to test.
    # -----------------------------------------------------------------
    model_results = []
    fitted = {}
    thresholds_used = {}

    # Majority baseline
    m = evaluate(test["target_t1"], None)
    model_results.append({"model": "majority", "feature_set": "n/a", **m, "threshold": None})

    for name, feats in FEATURE_SETS.items():
        best_t, cv_sweep = temporal_cv_threshold(train, feats)
        thresholds_used[name] = {"selected_threshold": best_t, "cv_sweep": cv_sweep}

        # Random Forest (primary model type per instructions)
        rf = fit_rf(train, feats)
        fitted[name] = (rf, feats)
        score = rf.predict_proba(test[feats])[:, 1]
        m = evaluate(test["target_t1"], score, threshold=best_t)
        model_results.append({"model": f"RF_{name}", "feature_set": ",".join(feats),
                               **m, "threshold": best_t})

        # Logistic Regression (interpretable comparison, per instructions)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        lr.fit(train[feats], train["target_t1"])
        score_lr = lr.predict_proba(test[feats])[:, 1]
        m_lr = evaluate(test["target_t1"], score_lr, threshold=0.5)
        model_results.append({"model": f"LR_{name}", "feature_set": ",".join(feats),
                               **m_lr, "threshold": 0.5})

    results_df = pd.DataFrame(model_results)
    results_df.to_csv("results/anomaly_model_results.csv", index=False)
    with open("results/anomaly_thresholds.json", "w") as f:
        json.dump(thresholds_used, f, indent=2, default=str)
    print("\n=== Model comparison (test set) ===")
    print(results_df[["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "threshold"]].to_string(index=False))

    # -----------------------------------------------------------------
    # 12. Ablation — individual anomaly features and combinations
    # -----------------------------------------------------------------
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
        best_t, _ = temporal_cv_threshold(train, feats)
        rf = fit_rf(train, feats)
        score = rf.predict_proba(test[feats])[:, 1]
        m = evaluate(test["target_t1"], score, threshold=best_t)
        ablation_rows.append({"ablation": name, **m, "threshold": best_t})
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv("results/anomaly_ablation_results.csv", index=False)
    print("\n=== Anomaly feature ablation (test set) ===")
    print(ablation_df.to_string(index=False))

    # -----------------------------------------------------------------
    # 9. Country-level analysis
    # -----------------------------------------------------------------
    best_name = "D_raw_plus_anomaly_plus_history"
    best_model, best_feats = fitted[best_name]
    best_threshold = thresholds_used[best_name]["selected_threshold"]

    country_rows = []
    all_countries = sorted(set(CORE_CLIENTS) | set(ANOMALY_COUNTRIES_OF_INTEREST))
    for country in all_countries:
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
    print("\n=== Country-level results (best model D) ===")
    print(country_df.to_string(index=False))

    # -----------------------------------------------------------------
    # 10. Kenya-specific test: raw vs anomaly vs raw+anomaly
    # -----------------------------------------------------------------
    kenya_test = test[test["COUNTRYNAME"] == "Kenya"]
    kenya_rows = []
    for name in ["A_raw", "B_anomaly", "C_raw_plus_anomaly"]:
        model, feats = fitted[name]
        t = thresholds_used[name]["selected_threshold"]
        score = model.predict_proba(kenya_test[feats])[:, 1]
        m = evaluate(kenya_test["target_t1"], score, threshold=t)
        kenya_rows.append({"feature_set": name, **m, "threshold": t})
    kenya_df = pd.DataFrame(kenya_rows)
    kenya_df.to_csv("results/anomaly_kenya_comparison.csv", index=False)

    kenya_pos = kenya_test[kenya_test["target_t1"] == 1]
    kenya_neg = kenya_test[kenya_test["target_t1"] == 0]
    kenya_distributions = {
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
        json.dump(kenya_distributions, f, indent=2)

    print("\n=== Kenya: raw vs anomaly vs raw+anomaly (test set) ===")
    print(kenya_df.to_string(index=False))
    print("\nKenya feature distributions (onset positive vs negative):")
    print(json.dumps(kenya_distributions, indent=2))

    # -----------------------------------------------------------------
    # 13. Temporal robustness (full dataset, early/late split, as in Phase 3)
    # -----------------------------------------------------------------
    median_year = int(df["year_T"].median())
    early = df[df["year_T"] <= median_year]
    late = df[df["year_T"] > median_year]

    robustness_rows = []
    for name in ["A_raw", "B_anomaly", "C_raw_plus_anomaly", "D_raw_plus_anomaly_plus_history"]:
        feats = FEATURE_SETS[name]
        m = fit_rf(early, feats)
        score = m.predict_proba(late[feats])[:, 1]
        ev = evaluate(late["target_t1"], score)
        robustness_rows.append({"feature_set": name, **ev,
                                  "train_period": f"<= {median_year}",
                                  "test_period": f"> {median_year}"})
    robustness_df = pd.DataFrame(robustness_rows)
    robustness_df.to_csv("results/anomaly_temporal_robustness.csv", index=False)
    print("\n=== Temporal robustness: raw vs anomaly (full dataset, early/late split) ===")
    print(robustness_df.to_string(index=False))

    # -----------------------------------------------------------------
    # 11. Feature importance for the strongest anomaly model (model D)
    # -----------------------------------------------------------------
    perm = permutation_importance(best_model, test[best_feats], test["target_t1"],
                                    n_repeats=10, random_state=SEED, scoring="average_precision")
    importance_df = pd.DataFrame({
        "feature": best_feats,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv("results/anomaly_permutation_importance.csv", index=False)
    print("\n=== Permutation importance, model D (raw+anomaly+history) ===")
    print(importance_df.to_string(index=False))

    print("\nDone. All Phase 4 outputs in results/anomaly_*.")


if __name__ == "__main__":
    main()
