"""
build_anomaly_dataset.py — Phase 4, Steps 1-3: construct leakage-safe
climate anomalies and produce locust_onset_anomaly_dataset.csv.

Run:
    python build_anomaly_dataset.py

Climatology definition (stated explicitly, not left implicit):
For a given (COUNTRYNAME, REGION, STARTMONTH) — i.e. "this region's usual
January", "this region's usual February", etc. — the climatological mean
and std at year Y are computed from ONLY the years < Y that have an
observation for that exact region/month-of-year in the cleaned same-month
dataset (an EXPANDING window, strictly historical, never including year Y
itself or any later year). This is the standard climatological-anomaly
definition (compare a month's value to that calendar month's own history
at that location) and directly satisfies the "no future leakage" and
"month-of-year, not global mean" requirements.

Minimum history rule: a region/month-of-year needs >= 3 prior yearly
observations before its climatology is considered stable enough to trust.
Below that, PPT_anomaly/TMAX_anomaly/SOILMOISTURE_anomaly (and z-scores)
are left as NaN -- NOT filled with 0 or any other value -- and a
`_climatology_insufficient_history` flag is set. How many rows this
affects is reported directly, not buried.

Z-score formulation: anomaly / historical std, guarded against std=0
(a region/month-of-year with literally constant history) by leaving the
z-score NaN in that case rather than dividing by zero -- flagged
separately from the "insufficient history" case since it's a different
failure mode (enough history, but degenerate variance).
"""

import json

import numpy as np
import pandas as pd

MIN_PRIOR_YEARS = 3

# ---------------------------------------------------------------------------
# Step 1-2: build leakage-safe climatology from the same-month dataset
# ---------------------------------------------------------------------------
cs = pd.read_csv("data/cleaned/cleaned_same_month_dataset.csv")
cs = cs.sort_values(["COUNTRYNAME", "REGION", "STARTMONTH", "STARTYEAR"]).reset_index(drop=True)

clim_rows = []
group_cols = ["COUNTRYNAME", "REGION", "STARTMONTH"]
for key, g in cs.groupby(group_cols, sort=False):
    g = g.sort_values("STARTYEAR")
    for var in ("PPT", "TMAX", "SOILMOISTURE"):
        # expanding stats computed on all PRIOR rows only (shift(1) before
        # expanding would be wrong order; correct pattern: expanding().mean()
        # naturally includes the current row, so shift the SERIES by 1 first,
        # then expanding() -- this guarantees year Y's climatology uses only
        # years < Y)
        shifted = g[var].shift(1)
        exp_mean = shifted.expanding(min_periods=1).mean()
        exp_std = shifted.expanding(min_periods=1).std()
        exp_count = shifted.expanding(min_periods=1).count()
        g[f"{var}_clim_mean"] = exp_mean.values
        g[f"{var}_clim_std"] = exp_std.values
        g[f"{var}_clim_n_prior_years"] = exp_count.values
    clim_rows.append(g)

climatology = pd.concat(clim_rows, ignore_index=True)

# Sanity check on leakage: for the very first year a (region, month) is
# observed, n_prior_years must be 0 (no history at all) -- verify directly
# rather than assuming the shift/expanding logic is correct.
first_obs = climatology.sort_values("STARTYEAR").groupby(group_cols).first()
assert (first_obs["PPT_clim_n_prior_years"] == 0).all(), (
    "Leakage check FAILED: a first-ever observation for some region/month "
    "has non-zero prior-year count -- climatology construction is using "
    "future or same-year data. Stop and fix before proceeding."
)
print("Leakage check passed: every region/month-of-year's first observation "
      "has zero prior-year climatology history, as required.")

climatology_lookup = climatology[
    group_cols + [
        "STARTYEAR",
        "PPT_clim_mean", "PPT_clim_std", "PPT_clim_n_prior_years",
        "TMAX_clim_mean", "TMAX_clim_std", "TMAX_clim_n_prior_years",
        "SOILMOISTURE_clim_mean", "SOILMOISTURE_clim_std", "SOILMOISTURE_clim_n_prior_years",
    ]
]

# ---------------------------------------------------------------------------
# Step 3: build the onset dataset with anomalies attached
# ---------------------------------------------------------------------------
full = pd.read_csv("data/forecasting/locust_t_plus_1_dataset.csv")
full["year_T"] = full["period_T"].str[:4].astype(int)
full["month_T"] = full["period_T"].str[5:7].astype(int)

onset = full[full["LOCUSTPRESENT_T"] == 0].copy()
for col in ["LOCUSTPRESENT_T_minus_1", "LOCUSTPRESENT_T_minus_2"]:
    onset[f"{col}_was_missing"] = onset[col].isna().astype(int)
    onset[col] = onset[col].fillna(0)

onset = onset.merge(
    climatology_lookup,
    left_on=["COUNTRYNAME", "REGION", "month_T", "year_T"],
    right_on=["COUNTRYNAME", "REGION", "STARTMONTH", "STARTYEAR"],
    how="left",
)

report = {"total_rows": len(onset)}

for var in ("PPT", "TMAX", "SOILMOISTURE"):
    val_col = f"{var}_T"
    mean_col = f"{var}_clim_mean"
    std_col = f"{var}_clim_std"
    n_col = f"{var}_clim_n_prior_years"

    has_enough_history = onset[n_col] >= MIN_PRIOR_YEARS
    anomaly = onset[val_col] - onset[mean_col]
    anomaly[~has_enough_history] = np.nan
    onset[f"{var}_anomaly"] = anomaly

    std_nonzero = (onset[std_col] > 0) & has_enough_history
    zscore = anomaly / onset[std_col]
    zscore[~std_nonzero] = np.nan
    onset[f"{var}_zscore"] = zscore

    report[f"{var}_rows_insufficient_history"] = int((~has_enough_history).sum())
    report[f"{var}_rows_zero_variance_history"] = int(
        (has_enough_history & (onset[std_col] == 0)).sum()
    )
    report[f"{var}_pct_rows_with_valid_zscore"] = round(
        100 * std_nonzero.sum() / len(onset), 2
    )

    onset[f"{var}_anomaly_missing"] = (~has_enough_history).astype(int)

with open("results/anomaly_construction_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))

onset.to_csv("data/forecasting/locust_onset_anomaly_dataset.csv", index=False)
print(f"\nWrote data/forecasting/locust_onset_anomaly_dataset.csv "
      f"({len(onset)} rows, {onset['target_t1'].sum()} positive)")
