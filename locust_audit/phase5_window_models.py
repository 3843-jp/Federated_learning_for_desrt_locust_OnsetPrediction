"""
phase5_window_models.py — Experiment 4 (short window), Experiment 5
(anomaly-window), Experiment 3 (country-specific threshold calibration),
Experiment 6 (leakage documentation), Experiment 7 (consolidated table),
Experiment 8 (global vs per-country).
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score

SEED = 42
TRAIN_MAX_YEAR = 2013
VAL_MAX_YEAR = 2017
CORE_CLIENTS = ["Kenya", "Egypt", "Yemen", "Ethiopia", "Sudan", "Mauritania"]
MIN_POSITIVES_FOR_METRICS = 5
MIN_OOF_POSITIVES_FOR_THRESHOLD = 5

df = pd.read_csv("data/forecasting/locust_onset_window_dataset.csv")
df["year_T"] = df["period_T"].str[:4].astype(int)

RAW = ["PPT_T", "TMAX_T", "SOILMOISTURE_T"]
ANOMALY = ["PPT_anomaly", "TMAX_anomaly", "SOILMOISTURE_anomaly"]
ANOMALY_FLAGS = ["PPT_anomaly_missing", "TMAX_anomaly_missing", "SOILMOISTURE_anomaly_missing"]

WINDOW_B = ["PPT_T", "PPT_Tm1", "PPT_2mo_sum", "TMAX_T", "TMAX_Tm1", "TMAX_2mo_mean",
            "SOILMOISTURE_T", "SOILMOISTURE_Tm1", "SOILMOISTURE_2mo_mean",
            "PPT_change_T_Tm1", "TMAX_change_T_Tm1", "SOILMOISTURE_change_T_Tm1"]
WINDOW_C = WINDOW_B + ["PPT_Tm2", "PPT_3mo_sum", "TMAX_Tm2", "TMAX_3mo_mean",
                        "SOILMOISTURE_Tm2", "SOILMOISTURE_3mo_mean"]
ANOMALY_WINDOW = ANOMALY + ANOMALY_FLAGS + [
    "PPT_anomaly_Tm1", "PPT_anomaly_change",
    "TMAX_anomaly_Tm1", "TMAX_anomaly_change",
    "SOILMOISTURE_anomaly_Tm1", "SOILMOISTURE_anomaly_change",
]

FEATURE_SETS = {
    "raw_only": RAW,
    "window_B_2mo": WINDOW_B,
    "window_C_3mo": WINDOW_C,
    "anomaly_window": ANOMALY_WINDOW,
}

for feats in FEATURE_SETS.values():
    for c in feats:
        if df[c].isna().any():
            flag = f"{c}_was_missing"
            if flag not in df.columns:
                df[flag] = df[c].isna().astype(int)
            df[c] = df[c].fillna(0)

def with_flags(feats):
    out = list(feats)
    for c in feats:
        flag = f"{c}_was_missing"
        if flag in df.columns and flag not in out:
            out.append(flag)
    return out

FEATURE_SETS = {k: with_flags(v) for k, v in FEATURE_SETS.items()}

def split_of(y):
    if y <= TRAIN_MAX_YEAR: return "train"
    elif y <= VAL_MAX_YEAR: return "val"
    return "test"
df["split"] = df["year_T"].apply(split_of)
train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)
print(f"Train n={len(train)} pos={int(train['target_t1'].sum())}, "
      f"Test n={len(test)} pos={int(test['target_t1'].sum())}")

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

def temporal_cv_oof(tr, feats, n_folds=5):
    years = sorted(tr["year_T"].unique())
    fold_edges = np.array_split(years, n_folds)
    oof_idx, oof_score = [], []
    for fold_years in fold_edges:
        val_mask = tr["year_T"].isin(fold_years)
        f_train, f_val = tr[~val_mask], tr[val_mask]
        if f_val["target_t1"].sum() == 0:
            continue
        m = fit_rf(f_train, feats)
        s = m.predict_proba(f_val[feats])[:, 1]
        oof_idx.extend(f_val.index.tolist())
        oof_score.extend(s.tolist())
    return pd.Series(oof_score, index=oof_idx)

def best_threshold_from_oof(oof_true, oof_score):
    best_t, best_f1 = 0.5, -1
    for t in [0.10, 0.20, 0.30, 0.40, 0.50]:
        pred = (oof_score >= t).astype(int)
        f1 = f1_score(oof_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

print("\nFitting window/anomaly-window models (CV threshold selection)...")
model_rows = []
fitted = {}
for name, feats in FEATURE_SETS.items():
    oof_score = temporal_cv_oof(train, feats)
    oof_true = train.loc[oof_score.index, "target_t1"]
    t = best_threshold_from_oof(oof_true, oof_score)

    rf = fit_rf(train, feats)
    fitted[name] = (rf, feats, t)
    score = rf.predict_proba(test[feats])[:, 1]
    ev = evaluate(test["target_t1"], score, threshold=t)
    model_rows.append({"model": f"RF_{name}", **ev, "threshold": t})

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    lr.fit(train[feats], train["target_t1"])
    score_lr = lr.predict_proba(test[feats])[:, 1]
    ev_lr = evaluate(test["target_t1"], score_lr, threshold=0.5)
    model_rows.append({"model": f"LR_{name}", **ev_lr, "threshold": 0.5})

results_df = pd.DataFrame(model_rows)
results_df.to_csv("results/phase5_window_model_results.csv", index=False)
print(results_df.to_string(index=False))

best_name = results_df.loc[results_df["pr_auc"].idxmax(), "model"].replace("RF_", "").replace("LR_", "")
if best_name not in fitted:
    best_name = "raw_only"
print(f"\nUsing feature set '{best_name}' for country-specific threshold calibration.")

best_model, best_feats, global_threshold = fitted[best_name]
oof_score_best = temporal_cv_oof(train, best_feats)
oof_true_best = train.loc[oof_score_best.index, "target_t1"]
oof_country = train.loc[oof_score_best.index, "COUNTRYNAME"]

threshold_compare_rows = []
for country in sorted(set(CORE_CLIENTS) | {"Uganda", "Algeria", "Tunisia", "Somaliland"}):
    ctest = test[test["COUNTRYNAME"] == country]
    n_test_pos = int(ctest["target_t1"].sum())
    c_oof_mask = oof_country == country
    n_oof_pos = int(oof_true_best[c_oof_mask].sum())

    row = {"country": country, "test_candidates": len(ctest), "test_positives": n_test_pos}

    if n_test_pos < MIN_POSITIVES_FOR_METRICS:
        row["status"] = f"INSUFFICIENT TEST DATA (n_pos={n_test_pos})"
        threshold_compare_rows.append(row)
        continue

    score_global = best_model.predict_proba(ctest[best_feats])[:, 1]
    m_global = evaluate(ctest["target_t1"], score_global, threshold=global_threshold)

    if n_oof_pos < MIN_OOF_POSITIVES_FOR_THRESHOLD:
        row["status"] = (f"global threshold only -- country-specific threshold "
                          f"UNAVAILABLE (n_oof_pos={n_oof_pos} < {MIN_OOF_POSITIVES_FOR_THRESHOLD})")
        row["global_threshold"] = global_threshold
        row["global_precision"] = m_global["precision"]
        row["global_recall"] = m_global["recall"]
        row["global_f1"] = m_global["f1"]
        threshold_compare_rows.append(row)
        continue

    country_threshold = best_threshold_from_oof(oof_true_best[c_oof_mask], oof_score_best[c_oof_mask])
    m_country = evaluate(ctest["target_t1"], score_global, threshold=country_threshold)

    row.update({
        "status": "computed",
        "global_threshold": global_threshold,
        "global_precision": m_global["precision"], "global_recall": m_global["recall"], "global_f1": m_global["f1"],
        "country_threshold": country_threshold,
        "country_precision": m_country["precision"], "country_recall": m_country["recall"], "country_f1": m_country["f1"],
    })
    threshold_compare_rows.append(row)

threshold_df = pd.DataFrame(threshold_compare_rows)
threshold_df.to_csv("results/phase5_country_threshold_comparison.csv", index=False)
print("\n=== Experiment 3: global vs country-specific threshold ===")
print(threshold_df.to_string(index=False))

per_country_rows = []
for country in sorted(set(CORE_CLIENTS) | {"Uganda", "Algeria", "Tunisia", "Somaliland", "Sudan"}):
    ctest = test[test["COUNTRYNAME"] == country]
    n_pos = int(ctest["target_t1"].sum())
    if n_pos < MIN_POSITIVES_FOR_METRICS:
        per_country_rows.append({"country": country, "n_pos": n_pos, "status": "insufficient"})
        continue
    score = best_model.predict_proba(ctest[best_feats])[:, 1]
    m = evaluate(ctest["target_t1"], score, threshold=global_threshold)
    per_country_rows.append({"country": country, "n_pos": n_pos, "status": "computed", **m})
per_country_df = pd.DataFrame(per_country_rows)
per_country_df.to_csv("results/phase5_per_country_best_model.csv", index=False)
print("\n=== Experiment 8: per-country performance, best model, global threshold ===")
print(per_country_df.to_string(index=False))

overall_score = best_model.predict_proba(test[best_feats])[:, 1]
overall_m = evaluate(test["target_t1"], overall_score, threshold=global_threshold)
print("\nGlobal (aggregate) performance, same model/threshold:")
print(json.dumps(overall_m, indent=2))
with open("results/phase5_global_vs_percountry_summary.json", "w") as f:
    json.dump({"global": overall_m, "best_feature_set": best_name}, f, indent=2)

print("\nDone with phase5_window_models.py")
