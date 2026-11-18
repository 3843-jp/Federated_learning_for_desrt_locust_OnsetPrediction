"""
assess_calibration.py — Part 2: is the frozen model's sigmoid output a
scientifically demonstrable probability, or only a relative risk score?

Method: reliability analysis (predicted-score deciles vs. observed
positive rate) plus Brier score, computed on the frozen test set as a
DIAGNOSTIC only -- this does not touch, retrain, or reselect anything
about the frozen model or the Phase 8 evaluation; it only asks whether
the existing sigmoid outputs happen to be well-calibrated.
"""
import json
import numpy as np
import pandas as pd
import torch

import sys, os
sys.path.insert(0, os.path.abspath("frozen_inputs"))
os.chdir("frozen_inputs")
from phase6_data_prep import load_frozen_dataset, FROZEN_FEATURES
from phase6_model import LogisticRegressionModel, scale, predict_proba

df = load_frozen_dataset()
test = df[df["split"] == "test"].reset_index(drop=True)
os.chdir("..")

scaler = np.load("config/frozen_scaler.npz")
mean, std = scaler["mean"], scaler["std"]
params_npz = np.load("config/frozen_method_a_model.npz")
params = [params_npz[k] for k in params_npz.files]

model = LogisticRegressionModel(len(FROZEN_FEATURES))
with torch.no_grad():
    for p, new in zip(model.parameters(), params):
        p.copy_(torch.tensor(new, dtype=torch.float32))

X_test = scale(test[FROZEN_FEATURES].values.astype(np.float32), mean, std)
y_test = test["target_t1"].values
scores = predict_proba(model, X_test)

n_bins = 5
bin_edges = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
bin_edges[0], bin_edges[-1] = -0.001, 1.001
bin_ids = np.digitize(scores, bin_edges) - 1

rows = []
for b in range(n_bins):
    mask = bin_ids == b
    n = mask.sum()
    if n == 0:
        continue
    mean_pred = scores[mask].mean()
    observed_rate = y_test[mask].mean()
    rows.append({"bin": b, "n": int(n), "mean_predicted_score": round(float(mean_pred), 4),
                 "observed_positive_rate": round(float(observed_rate), 4),
                 "n_positive_in_bin": int(y_test[mask].sum())})

reliability_df = pd.DataFrame(rows)
brier = float(np.mean((scores - y_test) ** 2))

min_positives_per_bin = reliability_df["n_positive_in_bin"].min() if len(reliability_df) else 0
sample_size_adequate = bool(min_positives_per_bin >= 5)

# Sample-size adequacy is necessary but NOT sufficient -- it only tells us
# whether a bin has enough events to estimate a rate at all. The actual
# calibration question is whether predicted score tracks observed rate.
# Expected Calibration Error (bin-size-weighted mean absolute gap):
reliability_df["abs_gap"] = (reliability_df["mean_predicted_score"] - reliability_df["observed_positive_rate"]).abs()
ece = float((reliability_df["abs_gap"] * reliability_df["n"]).sum() / reliability_df["n"].sum())
# Monotonicity check: a calibrated model's observed rate should increase
# as predicted score increases across bins -- check directly rather than
# assuming it from having "enough" data per bin.
observed_rates = reliability_df["observed_positive_rate"].tolist()
is_monotonic = all(observed_rates[i] <= observed_rates[i + 1] for i in range(len(observed_rates) - 1))

# A model trained with heavy class-weighting (pos_weight in BCE loss, used
# throughout this project to handle severe imbalance) is EXPECTED to be
# poorly calibrated as a probability, even with adequate sample size --
# class weighting deliberately shifts the sigmoid's operating point to
# separate classes better, at the direct cost of the output no longer
# representing a true base rate. This is checked here, not assumed.
CALIBRATION_ERROR_THRESHOLD = 0.10  # set before viewing results: mean
                                      # absolute gap above 10 percentage
                                      # points is not a defensible "this
                                      # is a probability" claim
calibration_demonstrable = bool(sample_size_adequate and is_monotonic and ece <= CALIBRATION_ERROR_THRESHOLD)

result = {
    "brier_score": brier,
    "expected_calibration_error": round(ece, 4),
    "calibration_error_threshold": CALIBRATION_ERROR_THRESHOLD,
    "monotonic_reliability": is_monotonic,
    "n_bins": n_bins,
    "reliability_table": reliability_df.drop(columns=["abs_gap"]).to_dict(orient="records"),
    "min_positives_in_any_bin": int(min_positives_per_bin),
    "sample_size_adequate_for_assessment": sample_size_adequate,
    "calibration_demonstrable_pooled_test_set": calibration_demonstrable,
    "decision": (
        "Display as PROBABILITY (%)" if calibration_demonstrable
        else "Display as SCORE (0-1), not a calibrated probability"
    ),
    "why": (
        "Model was trained with class-weighted BCE loss (pos_weight) throughout this "
        "project to handle severe class imbalance -- this deliberately shifts sigmoid "
        "outputs away from true base rates as a side effect of improving class separation, "
        "so poor calibration here is an expected consequence of a design choice made "
        "correctly for a different goal (discrimination, not calibration), not a bug."
        if not calibration_demonstrable else
        "Reliability curve is monotonic and within the pre-set error tolerance."
    ),
    "per_client_note": (
        "Per-client calibration is even less demonstrable than the pooled result above: "
        "Kenya has 36 test-period positives and Uganda has 17, both far below what would be "
        "needed for a per-client reliability curve with any bin resolution. The pooled result "
        "is the best-supported statement this data can make about calibration, and it applies "
        "(or fails to apply) to the whole system, not to any single client's number specifically."
    ),
}
with open("config/calibration_assessment.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(json.dumps(result, indent=2, default=str))
