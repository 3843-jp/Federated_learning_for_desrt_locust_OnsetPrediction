"""
end_to_end_demo.py — Part 14: complete demo using an actual test-set row
and actual model output. No invented values anywhere in this script.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_support as ds

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_inputs"))
from phase6_data_prep import load_frozen_dataset, FROZEN_FEATURES

_cwd = os.getcwd()
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_inputs"))
df = load_frozen_dataset()
os.chdir(_cwd)

test = df[df["split"] == "test"]
kenya_rows = test[test["COUNTRYNAME"] == "Kenya"]
# pick an actual row with a real region/period from the frozen test set --
# not fabricated
demo_row = kenya_rows.iloc[10]

country = demo_row["COUNTRYNAME"]
region = demo_row["REGION"]
period_T = demo_row["period_T"]
period_T1 = demo_row["period_T1"]
true_label = int(demo_row["target_t1"])

feature_row = {f: float(demo_row[f]) for f in FROZEN_FEATURES}

score = ds.predict_score(feature_row)
tier = ds.risk_tier(score)
warning = ds.early_warning_message(tier)
explanation = ds.explain(feature_row, top_k=3)
rec = ds.recommendation(tier)
evidence = ds.data_evidence(country)

demo_output = {
    "input": {
        "country": country, "region": region,
        "features_observed_at": period_T, "forecast_for": period_T1,
        "raw_feature_values": feature_row,
    },
    "prediction": {"score_display": ds.format_score_display(score), "raw_score": score, "risk_tier": tier},
    "early_warning": warning,
    "why": explanation,
    "recommendation": rec,
    "data_evidence": evidence,
    "limitations": ds.LIMITATIONS,
    "ground_truth_for_reference_only": {
        "actual_target_t1": true_label,
        "note": "Shown here only to demonstrate this is a real historical test-set row; the "
                "decision-support system does not have access to this label at prediction time "
                "and it is not part of the dashboard's user-facing output.",
    },
}

print(json.dumps(demo_output, indent=2, default=str))
with open("end_to_end_demo_output.json", "w") as f:
    json.dump(demo_output, f, indent=2, default=str)
