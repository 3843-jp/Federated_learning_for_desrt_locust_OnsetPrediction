"""
phase6_centralized_local.py — Part 2 (reproduce Phase 4 centralized
baseline with new Phase 6 code), Part 4 (local-only per client), Part 5
(centralized evaluated per client).
"""
import json
import numpy as np
import pandas as pd

from phase6_data_prep import load_frozen_dataset, FROZEN_FEATURES
from phase6_model import (
    LogisticRegressionModel, get_scaler, scale, train_local, predict_proba,
    evaluate, class_balanced_pos_weight, select_threshold_from_train_cv, SEED,
)

df = load_frozen_dataset()
with open("results/phase6_client_categories.json") as f:
    cats = json.load(f)
CATEGORY_A = cats["category_a"]
CATEGORY_B = cats["category_b"]
ALL_FL_CLIENTS = CATEGORY_A + CATEGORY_B

train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)

X_train_all = train[FROZEN_FEATURES].values.astype(np.float32)
y_train_all = train["target_t1"].values.astype(np.float32)
mean, std = get_scaler(X_train_all)

X_train_scaled = scale(X_train_all, mean, std)
X_test_scaled = scale(test[FROZEN_FEATURES].values.astype(np.float32), mean, std)
y_test = test["target_t1"].values.astype(np.float32)

print("=== Part 2: reproducing centralized baseline (PyTorch LR) ===")
threshold, cv_f1 = select_threshold_from_train_cv(
    X_train_scaled, y_train_all, train["year_T"].values
)
print(f"CV-selected threshold: {threshold} (CV F1={cv_f1})")

central_model = LogisticRegressionModel(len(FROZEN_FEATURES))
pw = class_balanced_pos_weight(y_train_all)
train_local(central_model, X_train_scaled, y_train_all, epochs=200, pos_weight=pw)

central_score_test = predict_proba(central_model, X_test_scaled)
central_eval = evaluate(y_test, central_score_test, threshold=threshold)
central_eval["threshold"] = threshold
print("Reproduced centralized (aggregate test) result:", json.dumps(central_eval, indent=2))

phase4_reference = {"precision": 0.048, "recall": 0.536, "f1": 0.088, "pr_auc": 0.059, "roc_auc": 0.570}
comparison = {"phase6_reproduction": central_eval, "phase4_reference_LR_raw_plus_anomaly": phase4_reference}
with open("results/phase6_centralized_reproduction_check.json", "w") as f:
    json.dump(comparison, f, indent=2, default=str)
print("\nComparison to Phase 4 reference (sklearn LR, same feature set):")
print(json.dumps(comparison, indent=2, default=str))

pr_auc_diff = abs(central_eval["pr_auc"] - phase4_reference["pr_auc"])
roc_auc_diff = abs(central_eval["roc_auc"] - phase4_reference["roc_auc"])
material_difference = pr_auc_diff > 0.03 or roc_auc_diff > 0.10
print(f"\nPR-AUC diff={pr_auc_diff:.4f}, ROC-AUC diff={roc_auc_diff:.4f} -> "
      f"{'MATERIAL DIFFERENCE -- STOP AND INVESTIGATE' if material_difference else 'within expected tolerance, proceeding'}")
if material_difference:
    raise SystemExit("Reproduction differs materially from Phase 4 reference. Stopping per instructions.")

centralized_rows = []
for country in ALL_FL_CLIENTS:
    ctest = test[test["COUNTRYNAME"] == country]
    if len(ctest) == 0 or ctest["target_t1"].sum() == 0:
        centralized_rows.append({
            "method": "centralized", "country": country,
            "train_samples": len(train[train["COUNTRYNAME"] == country]),
            "test_samples": len(ctest), "test_positive": int(ctest["target_t1"].sum()),
            "status": "insufficient test data" if country in CATEGORY_B else "n/a",
        })
        continue
    Xc = scale(ctest[FROZEN_FEATURES].values.astype(np.float32), mean, std)
    score = predict_proba(central_model, Xc)
    m = evaluate(ctest["target_t1"].values, score, threshold=threshold)
    centralized_rows.append({
        "method": "centralized", "country": country,
        "train_samples": len(train[train["COUNTRYNAME"] == country]),
        "test_samples": len(ctest), "test_positive": m["positives"],
        "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        "status": "computed",
    })
centralized_df = pd.DataFrame(centralized_rows)
centralized_df.to_csv("results/phase6_centralized_per_client.csv", index=False)
print("\n=== Part 5: centralized, per client ===")
print(centralized_df.to_string(index=False))

print("\n=== Part 4: local-only baselines ===")
local_rows = []
for country in ALL_FL_CLIENTS:
    ctr = train[train["COUNTRYNAME"] == country]
    cte = test[test["COUNTRYNAME"] == country]
    n_train_pos = int(ctr["target_t1"].sum())

    row = {"method": "local_only", "country": country,
           "train_samples": len(ctr), "train_positive": n_train_pos,
           "test_samples": len(cte), "test_positive": int(cte["target_t1"].sum())}

    if n_train_pos == 0:
        row["status"] = "CANNOT TRAIN -- zero positive examples in this client's training data"
        local_rows.append(row)
        continue
    if len(cte) == 0 or cte["target_t1"].sum() == 0:
        row["status"] = "trained, but insufficient test data to evaluate"
        local_rows.append(row)
        continue

    Xc_tr = scale(ctr[FROZEN_FEATURES].values.astype(np.float32), mean, std)
    yc_tr = ctr["target_t1"].values.astype(np.float32)
    local_model = LogisticRegressionModel(len(FROZEN_FEATURES))
    pw_local = class_balanced_pos_weight(yc_tr)
    train_local(local_model, Xc_tr, yc_tr, epochs=200, pos_weight=pw_local)

    Xc_te = scale(cte[FROZEN_FEATURES].values.astype(np.float32), mean, std)
    score = predict_proba(local_model, Xc_te)
    local_t, local_cv_f1 = select_threshold_from_train_cv(Xc_tr, yc_tr, ctr["year_T"].values)
    if local_cv_f1 is None:
        local_t = threshold
        row["threshold_source"] = "fallback to global threshold (insufficient local CV positives)"
    else:
        row["threshold_source"] = "local CV"
    m = evaluate(cte["target_t1"].values, score, threshold=local_t)
    row.update({"status": "computed", "threshold": local_t,
                "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
    local_rows.append(row)

local_df = pd.DataFrame(local_rows)
local_df.to_csv("results/phase6_local_only_per_client.csv", index=False)
print(local_df.to_string(index=False))

print("\nDone with phase6_centralized_local.py")
