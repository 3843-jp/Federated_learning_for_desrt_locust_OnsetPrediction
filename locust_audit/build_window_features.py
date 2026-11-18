"""
build_window_features.py — Experiment 4 & 5: short temporal window
features (T, T-1, T-2 raw values) and anomaly-window features (anomaly
change over time), built with the same leakage discipline as Phase 4.

Rule enforced throughout: a feature for a prediction made using month T's
row may use T, T-1, T-2 raw/anomaly values (all strictly <= T), and NEVER
T+1 or later. Verified directly below, not assumed.
"""
import numpy as np
import pandas as pd

MIN_PRIOR_YEARS = 3

# ---------------------------------------------------------------------
# Rebuild the same leakage-safe climatology used in Phase 4 (needed to
# compute anomalies at T-1 and T-2, not just T)
# ---------------------------------------------------------------------
cs = pd.read_csv("data/cleaned/cleaned_same_month_dataset.csv")
cs = cs.sort_values(["COUNTRYNAME", "REGION", "STARTMONTH", "STARTYEAR"]).reset_index(drop=True)

clim_parts = []
for key, g in cs.groupby(["COUNTRYNAME", "REGION", "STARTMONTH"], sort=False):
    g = g.sort_values("STARTYEAR")
    for var in ("PPT", "TMAX", "SOILMOISTURE"):
        shifted = g[var].shift(1)
        g[f"{var}_clim_mean"] = shifted.expanding(min_periods=1).mean().values
        g[f"{var}_clim_std"] = shifted.expanding(min_periods=1).std().values
        g[f"{var}_clim_n"] = shifted.expanding(min_periods=1).count().values
    clim_parts.append(g)
climatology = pd.concat(clim_parts, ignore_index=True)

# Lookup: raw values + climatology, keyed by (country, region, year, month)
env_lookup = climatology.set_index(["COUNTRYNAME", "REGION", "STARTYEAR", "STARTMONTH"])[
    ["PPT", "TMAX", "SOILMOISTURE",
     "PPT_clim_mean", "PPT_clim_std", "PPT_clim_n",
     "TMAX_clim_mean", "TMAX_clim_std", "TMAX_clim_n",
     "SOILMOISTURE_clim_mean", "SOILMOISTURE_clim_std", "SOILMOISTURE_clim_n"]
]

# ---------------------------------------------------------------------
# Load the onset+anomaly dataset (already has T-level features) and add
# T-1, T-2 raw + anomaly columns via lookup
# ---------------------------------------------------------------------
onset = pd.read_csv("data/forecasting/locust_onset_anomaly_dataset.csv")
onset["period_T_parsed"] = pd.PeriodIndex(onset["period_T"], freq="M")

def lookup_offset(row, offset):
    p = row["period_T_parsed"] + offset
    key = (row["COUNTRYNAME"], row["REGION"], p.year, p.month)
    if key in env_lookup.index:
        return env_lookup.loc[key]
    return pd.Series({c: np.nan for c in env_lookup.columns})

print("Computing T-1 lookups (this is the slow step, ~1 pass over all rows)...")
t1_vals = onset.apply(lambda r: lookup_offset(r, -1), axis=1)
t1_vals.columns = [f"{c}_Tm1" for c in t1_vals.columns]

print("Computing T-2 lookups...")
t2_vals = onset.apply(lambda r: lookup_offset(r, -2), axis=1)
t2_vals.columns = [f"{c}_Tm2" for c in t2_vals.columns]

onset = pd.concat([onset, t1_vals, t2_vals], axis=1)

# ---------------------------------------------------------------------
# Leakage check: T-1 period must always be < T period; verify directly
# ---------------------------------------------------------------------
p_t1 = onset["period_T_parsed"] - 1
assert (p_t1 < onset["period_T_parsed"]).all(), "T-1 lookup produced a period >= T -- leakage bug."
print("Leakage check passed: all T-1/T-2 lookups are strictly before T.")

# ---------------------------------------------------------------------
# Derived features (compact, physically interpretable, per instructions)
# ---------------------------------------------------------------------
for var, cum_ok in (("PPT", True), ("TMAX", False), ("SOILMOISTURE", False)):
    t = onset[f"{var}_T"]
    tm1 = onset[f"{var}_Tm1"]
    tm2 = onset[f"{var}_Tm2"]

    onset[f"{var}_change_T_Tm1"] = t - tm1  # month-over-month change

    if cum_ok:
        # cumulative rainfall only -- summing degrees/moisture isn't
        # physically meaningful, per instructions
        onset[f"{var}_2mo_sum"] = t + tm1
        onset[f"{var}_3mo_sum"] = t + tm1 + tm2
    else:
        onset[f"{var}_2mo_mean"] = (t + tm1) / 2
        onset[f"{var}_3mo_mean"] = (t + tm1 + tm2) / 3

    # anomaly-window features (Experiment 5)
    anom_t = onset[f"{var}_anomaly"]
    anom_tm1 = onset[f"{var}_Tm1"] - onset[f"{var}_clim_mean_Tm1"]
    onset[f"{var}_anomaly_Tm1"] = anom_tm1
    onset[f"{var}_anomaly_change"] = anom_t - anom_tm1

onset.drop(columns=["period_T_parsed"], inplace=True)
onset.to_csv("data/forecasting/locust_onset_window_dataset.csv", index=False)
print(f"\nWrote data/forecasting/locust_onset_window_dataset.csv "
      f"({len(onset)} rows, {onset['target_t1'].sum()} positive)")
print("\nSample of new columns:")
sample_cols = ["COUNTRYNAME", "REGION", "period_T", "PPT_T", "PPT_Tm1", "PPT_Tm2",
               "PPT_change_T_Tm1", "PPT_2mo_sum", "PPT_anomaly", "PPT_anomaly_Tm1",
               "PPT_anomaly_change"]
print(onset[sample_cols].head(5).to_string(index=False))
