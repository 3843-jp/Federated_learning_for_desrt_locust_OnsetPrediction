"""
clean_and_transform.py — Phase 1C: cleaning + T->T+1 transformation of the
real LocustLensFinalDataset.csv (verified GLAD42 data, per the Phase-1
forensic audit).

Run:
    python clean_and_transform.py

Reads:  data/raw/LocustLensFinalDataset.csv   (never modified)
Writes: data/cleaned/cleaned_same_month_dataset.csv
        data/forecasting/locust_t_plus_1_dataset.csv
        data/audit/client_temporal_statistics.csv
        data/audit/preprocessing_log.json   (every step's row counts, so
                                              Section 9's reduction table is
                                              generated, not hand-typed)
"""

import json
import os

import pandas as pd

RAW_PATH = "data/raw/LocustLensFinalDataset.csv"
CLEANED_DIR = "data/cleaned"
FORECAST_DIR = "data/forecasting"
AUDIT_DIR = "data/audit"
for d in (CLEANED_DIR, FORECAST_DIR, AUDIT_DIR):
    os.makedirs(d, exist_ok=True)

log = {"steps": []}


def record(step, **kwargs):
    entry = {"step": step, **kwargs}
    log["steps"].append(entry)
    print(f"[{step}] " + ", ".join(f"{k}={v}" for k, v in kwargs.items()))


# ---------------------------------------------------------------------------
# STEP 1 — load + clean country names
# ---------------------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
n_raw = len(df)
countries_before = df["COUNTRYNAME"].nunique()

df["COUNTRYNAME"] = df["COUNTRYNAME"].str.strip()
df["REGION"] = df["REGION"].str.strip()  # same whitespace risk, cheap to fix
countries_after = df["COUNTRYNAME"].nunique()

record("1_load_and_clean_country_names",
       raw_rows=n_raw,
       unique_countries_before_strip=countries_before,
       unique_countries_after_strip=countries_after)
assert countries_before == countries_after, (
    "Unexpected: stripping whitespace changed the country count — "
    "there may be near-duplicate country names differing only in case "
    "or internal spacing that need a closer look before proceeding."
)

# ---------------------------------------------------------------------------
# STEP 2 — TMAX outlier
# ---------------------------------------------------------------------------
outlier_mask = df["TMAX"] > 100  # threshold well above the 46.71 max seen
                                   # everywhere else in the file
outliers = df[outlier_mask]
outlier_records = outliers.to_dict(orient="records")
record("2_tmax_outlier_check",
       rows_flagged=len(outliers),
       flagged_rows=json.dumps(outlier_records, default=str))

df_clean = df[~outlier_mask].copy()
record("2_tmax_outlier_removed", rows_after=len(df_clean))

# ---------------------------------------------------------------------------
# STEP 3 — deduplication to (country, region, year, month) grain
# ---------------------------------------------------------------------------
key_cols = ["COUNTRYNAME", "REGION", "STARTYEAR", "STARTMONTH"]

n_before_dedup = len(df_clean)
key_counts = df_clean.groupby(key_cols).size()
n_unique_keys = len(key_counts)

pos = df_clean[df_clean["LOCUSTPRESENT"] == "yes"]
neg = df_clean[df_clean["LOCUSTPRESENT"] == "no"]
pos_key_counts = pos.groupby(key_cols).size()
keys_with_duplicate_positive = int((pos_key_counts > 1).sum())

record("3_dedup_diagnostics",
       raw_rows_pre_dedup=n_before_dedup,
       unique_keys=n_unique_keys,
       positive_keys=len(pos_key_counts),
       negative_keys=len(neg.groupby(key_cols).size()),
       keys_with_duplicate_positive_observations=keys_with_duplicate_positive)

# What information is lost by collapsing duplicate positive rows?
# Each duplicate group at a given key has IDENTICAL environmental values
# (same region + same year/month -> same TerraClimate extraction), so no
# environmental information differs between duplicates. What IS lost is the
# raw COUNT of distinct swarm reports within that region-month (e.g. 599
# for India/Rajasthan/Sep-2019) -- i.e., report INTENSITY. We preserve that
# as a new column (report_count) rather than silently discarding it, exactly
# because the instructions require documenting what's lost, not just
# dropping it.
dup_group_sizes = pos.groupby(key_cols).size().rename("report_count").reset_index()

# ---------------------------------------------------------------------------
# STEP 4 — resolve label contradictions (keys with both yes and no)
# ---------------------------------------------------------------------------
label_nunique = df_clean.groupby(key_cols)["LOCUSTPRESENT"].nunique()
contradiction_keys = label_nunique[label_nunique > 1].index.tolist()
record("4_label_contradictions_found", n_contradiction_keys=len(contradiction_keys),
       contradiction_keys=str(contradiction_keys))

# IMPORTANT FINDING (discovered while building this step, not assumed going
# in): environmental values are NOT always identical across rows sharing a
# key. Checked directly below. For the 1,820 keys with duplicate positive
# rows, within-key ranges (max-min) go up to 556mm PPT, 23.4C TMAX, 270.8
# SOILMOISTURE (median ranges are much smaller: 2.4mm/1.5C/0.1, but the tail
# is real and large, e.g. Ethiopia/Afar Region/Aug-2020). This means
# "region" is coarser than the environmental extraction grid — different
# point-level swarm reports within the same region/month were pulled from
# different TerraClimate grid cells. Silently taking the first row would
# arbitrarily pick one point's weather to represent the whole region-month
# and bias the result by file row order. Instead we take the MEAN across
# duplicate rows as the representative value, and keep the within-key
# range/std as diagnostic columns so this uncertainty isn't hidden.
grouped = df_clean.groupby(key_cols)

env_consistency_check = grouped[["PPT", "TMAX", "SOILMOISTURE"]].nunique()
inconsistent_keys = env_consistency_check[
    (env_consistency_check["PPT"] > 1)
    | (env_consistency_check["TMAX"] > 1)
    | (env_consistency_check["SOILMOISTURE"] > 1)
]
env_range = grouped[["PPT", "TMAX", "SOILMOISTURE"]].agg(lambda x: x.max() - x.min())
record("4_environmental_consistency_check",
       keys_with_inconsistent_environmental_values=len(inconsistent_keys),
       median_within_key_range_PPT=float(env_range["PPT"].median()),
       median_within_key_range_TMAX=float(env_range["TMAX"].median()),
       max_within_key_range_PPT=float(env_range["PPT"].max()),
       max_within_key_range_TMAX=float(env_range["TMAX"].max()),
       note="Real, not floating-point noise — region is coarser than the "
            "environmental grid. Resolved using per-key MEAN, not "
            "first-row, specifically because of this finding.")

resolved_rows = []
for key, g in grouped:
    country, region, year, month = key
    has_yes = (g["LOCUSTPRESENT"] == "yes").any()
    resolved_rows.append({
        "COUNTRYNAME": country,
        "REGION": region,
        "STARTYEAR": year,
        "STARTMONTH": month,
        "PPT": g["PPT"].mean(),
        "TMAX": g["TMAX"].mean(),
        "SOILMOISTURE": g["SOILMOISTURE"].mean(),
        "PPT_within_key_range": g["PPT"].max() - g["PPT"].min(),
        "TMAX_within_key_range": g["TMAX"].max() - g["TMAX"].min(),
        "SOILMOISTURE_within_key_range": g["SOILMOISTURE"].max() - g["SOILMOISTURE"].min(),
        # explicitly NOT calling this "confirmed absence" — see Step 5
        "LOCUSTPRESENT_RESOLVED": 1 if has_yes else 0,
        "report_count": int((g["LOCUSTPRESENT"] == "yes").sum()),
    })

cleaned_same_month = pd.DataFrame(resolved_rows)

record("4_resolved_same_month_dataset", rows=len(cleaned_same_month),
       positive_rows=int((cleaned_same_month["LOCUSTPRESENT_RESOLVED"] == 1).sum()),
       negative_rows=int((cleaned_same_month["LOCUSTPRESENT_RESOLVED"] == 0).sum()))

# ---------------------------------------------------------------------------
# STEP 5 — rename/document target concept (done via column naming + this
# note, not a separate transformation)
# ---------------------------------------------------------------------------
# LOCUSTPRESENT_RESOLVED = 1 means "a swarm was reported in this
# country/region/month." = 0 means "no swarm was reported in this
# country/region/month" -- i.e. a CANDIDATE NEGATIVE / NON-REPORTED
# region-month, NOT a confirmed ecological absence. This distinction is
# carried into every downstream file's column name and this log entry
# rather than asserted only in prose that could get lost.
record("5_target_semantics_documented",
       column_name="LOCUSTPRESENT_RESOLVED",
       meaning_of_1="swarm reported in this country/region/month",
       meaning_of_0="CANDIDATE NEGATIVE / non-reported region-month "
                    "(NOT confirmed ecological absence — see Phase 1 "
                    "forensic audit Section 3/4 for why)")

cleaned_same_month.to_csv(
    os.path.join(CLEANED_DIR, "cleaned_same_month_dataset.csv"), index=False
)

# ---------------------------------------------------------------------------
# STEP 6 — temporal index
# ---------------------------------------------------------------------------
cleaned_same_month["date"] = pd.to_datetime(
    dict(year=cleaned_same_month["STARTYEAR"],
         month=cleaned_same_month["STARTMONTH"], day=1)
)
cleaned_same_month = cleaned_same_month.sort_values(
    ["COUNTRYNAME", "REGION", "date"]
).reset_index(drop=True)
cleaned_same_month["period"] = cleaned_same_month["date"].dt.to_period("M")

record("6_temporal_index_built", rows=len(cleaned_same_month))

# ---------------------------------------------------------------------------
# STEP 7 + 8 — T -> T+1 construction, only across TRUE consecutive calendar
# months (no assumed transitions across gaps)
# ---------------------------------------------------------------------------
t1_rows = []
missing_month_transitions = 0
total_possible_transitions = 0

for (country, region), g in cleaned_same_month.groupby(["COUNTRYNAME", "REGION"]):
    g = g.sort_values("period").reset_index(drop=True)
    periods = g["period"].tolist()
    period_to_row = {p: i for i, p in enumerate(periods)}

    for i in range(len(g)):
        p_t = g.loc[i, "period"]
        p_t1 = p_t + 1
        total_possible_transitions += 1  # candidate transition FROM this row

        if p_t1 not in period_to_row:
            missing_month_transitions += 1
            continue  # true gap -- do NOT fabricate a transition (Step 8)

        j = period_to_row[p_t1]
        row_t = g.loc[i]
        row_t1 = g.loc[j]

        # historical features: T, T-1, T-2 presence, only if those periods
        # are ALSO truly present (not assumed) -- absence of history is
        # encoded as NaN, not silently as 0, so a model can distinguish
        # "known absent" from "unknown/missing" if desired downstream.
        def presence_at(offset):
            p = p_t + offset
            if p in period_to_row:
                return g.loc[period_to_row[p], "LOCUSTPRESENT_RESOLVED"]
            return None

        t1_rows.append({
            "COUNTRYNAME": country,
            "REGION": region,
            "period_T": str(p_t),
            "period_T1": str(p_t1),
            "PPT_T": row_t["PPT"],
            "TMAX_T": row_t["TMAX"],
            "SOILMOISTURE_T": row_t["SOILMOISTURE"],
            "LOCUSTPRESENT_T": row_t["LOCUSTPRESENT_RESOLVED"],
            "LOCUSTPRESENT_T_minus_1": presence_at(-1),
            "LOCUSTPRESENT_T_minus_2": presence_at(-2),
            "report_count_T": row_t["report_count"],
            "target_t1": row_t1["LOCUSTPRESENT_RESOLVED"],
        })

t_plus_1_df = pd.DataFrame(t1_rows)
t_plus_1_df.to_csv(
    os.path.join(FORECAST_DIR, "locust_t_plus_1_dataset.csv"), index=False
)

record("7_8_t_plus_1_construction",
       total_candidate_transitions=total_possible_transitions,
       missing_month_transitions_skipped=missing_month_transitions,
       pct_transitions_lost_to_gaps=round(
           100 * missing_month_transitions / total_possible_transitions, 2
       ) if total_possible_transitions else None,
       valid_t_plus_1_samples=len(t_plus_1_df))

# ---------------------------------------------------------------------------
# STEP 9 — full reduction table
# ---------------------------------------------------------------------------
reduction_table = [
    {"stage": "raw_rows", "rows": n_raw},
    {"stage": "after_country_name_cleaning", "rows": n_raw},  # cleaning
                                                                # doesn't
                                                                # drop rows
    {"stage": "after_outlier_removal", "rows": len(df_clean)},
    {"stage": "after_dedup_and_contradiction_resolution "
              "(same-month grain)", "rows": len(cleaned_same_month)},
    {"stage": "valid_T_to_T+1_samples", "rows": len(t_plus_1_df)},
]
record("9_reduction_table", table=json.dumps(reduction_table))

# ---------------------------------------------------------------------------
# STEP 10 — client-level T->T+1 statistics
# ---------------------------------------------------------------------------
client_rows = []
for country, g in t_plus_1_df.groupby("COUNTRYNAME"):
    total = len(g)
    pos_t = int((g["target_t1"] == 1).sum())
    neg_t = int((g["target_t1"] == 0).sum())

    # missing-month % specific to this country: recompute from
    # cleaned_same_month, not from t_plus_1_df (which by definition only
    # contains the valid transitions)
    cg = cleaned_same_month[cleaned_same_month["COUNTRYNAME"] == country]
    country_total_transitions = 0
    country_missing_transitions = 0
    for region, rg in cg.groupby("REGION"):
        periods = set(rg["period"])
        for p in periods:
            country_total_transitions += 1
            if (p + 1) not in periods:
                country_missing_transitions += 1
    missing_pct = (
        round(100 * country_missing_transitions / country_total_transitions, 1)
        if country_total_transitions else None
    )

    client_rows.append({
        "country": country,
        "valid_t_plus_1_samples": total,
        "positive_targets": pos_t,
        "negative_targets": neg_t,
        "positive_target_pct": round(100 * pos_t / total, 2) if total else None,
        "n_regions": cg["REGION"].nunique(),
        "years_covered": cg["STARTYEAR"].nunique(),
        "missing_month_transition_pct": missing_pct,
    })

client_stats = pd.DataFrame(client_rows).sort_values(
    "valid_t_plus_1_samples", ascending=False
)
client_stats.to_csv(
    os.path.join(AUDIT_DIR, "client_temporal_statistics.csv"), index=False
)
record("10_client_temporal_statistics_written", n_countries=len(client_stats))

# ---------------------------------------------------------------------------
# Save the preprocessing log
# ---------------------------------------------------------------------------
with open(os.path.join(AUDIT_DIR, "preprocessing_log.json"), "w") as f:
    json.dump(log, f, indent=2, default=str)

print("\nDone. See data/cleaned/, data/forecasting/, data/audit/.")
