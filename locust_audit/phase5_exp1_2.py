"""
phase5_exp1_2.py — Experiment 1 (country environmental signal) and
Experiment 2 (is heterogeneity real, quantified).

Effect size: Cliff's delta (rank-based, robust to the skewed, non-normal
distributions these environmental variables have -- chosen over Cohen's d
specifically because these variables are not close to normal, and Cliff's
delta doesn't assume they are). Statistical test: Mann-Whitney U (matches
the rank-based effect size, appropriate for small, non-normal samples).
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

MIN_POSITIVES = 5

def cliffs_delta(x, y):
    """Cliff's delta: P(X>Y) - P(X<Y). Range -1..1. |d|<0.147 negligible,
    <0.33 small, <0.474 medium, else large (Romano et al. 2006 thresholds,
    a standard, citable convention for this statistic)."""
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    # O(nx*ny) is fine at these small onset-positive sample sizes
    greater = sum((xi > y).sum() for xi in x)
    less = sum((xi < y).sum() for xi in x)
    return (greater - less) / (nx * ny)

def effect_label(d):
    ad = abs(d)
    if ad < 0.147: return "negligible"
    if ad < 0.33: return "small"
    if ad < 0.474: return "medium"
    return "large"

onset = pd.read_csv("data/forecasting/locust_onset_anomaly_dataset.csv")

rows = []
country_effects = {}  # country -> {var: delta} for Experiment 2
for country, g in onset.groupby("COUNTRYNAME"):
    pos = g[g["target_t1"] == 1]
    neg = g[g["target_t1"] == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos < MIN_POSITIVES:
        rows.append({"country": country, "positive_onsets": n_pos,
                      "negative_candidates": n_neg,
                      "PPT_effect": None, "TMAX_effect": None, "SOILMOISTURE_effect": None,
                      "dominant_variable": None,
                      "evidence_strength": f"INSUFFICIENT DATA (n_pos={n_pos} < {MIN_POSITIVES})"})
        continue

    effects = {}
    pvals = {}
    for var in ("PPT_anomaly", "TMAX_anomaly", "SOILMOISTURE_anomaly"):
        p = pos[var].dropna()
        n = neg[var].dropna()
        if len(p) < MIN_POSITIVES or len(n) < 10:
            effects[var] = None
            pvals[var] = None
            continue
        d = cliffs_delta(p.values, n.values)
        try:
            _, pval = mannwhitneyu(p, n, alternative="two-sided")
        except ValueError:
            pval = None
        effects[var] = d
        pvals[var] = pval

    valid = {k: v for k, v in effects.items() if v is not None}
    if valid:
        dominant = max(valid, key=lambda k: abs(valid[k])).replace("_anomaly", "")
        dominant_strength = effect_label(valid[max(valid, key=lambda k: abs(valid[k]))])
        if dominant_strength == "negligible":
            dominant = "no clear signal"
    else:
        dominant, dominant_strength = "no clear signal", "insufficient"

    country_effects[country] = effects

    rows.append({
        "country": country,
        "positive_onsets": n_pos,
        "negative_candidates": n_neg,
        "PPT_effect": round(effects["PPT_anomaly"], 3) if effects["PPT_anomaly"] is not None else None,
        "PPT_pvalue": round(pvals["PPT_anomaly"], 4) if pvals.get("PPT_anomaly") is not None else None,
        "TMAX_effect": round(effects["TMAX_anomaly"], 3) if effects["TMAX_anomaly"] is not None else None,
        "TMAX_pvalue": round(pvals["TMAX_anomaly"], 4) if pvals.get("TMAX_anomaly") is not None else None,
        "SOILMOISTURE_effect": round(effects["SOILMOISTURE_anomaly"], 3) if effects["SOILMOISTURE_anomaly"] is not None else None,
        "SOILMOISTURE_pvalue": round(pvals["SOILMOISTURE_anomaly"], 4) if pvals.get("SOILMOISTURE_anomaly") is not None else None,
        "dominant_variable": dominant,
        "evidence_strength": dominant_strength,
    })

result_df = pd.DataFrame(rows).sort_values("positive_onsets", ascending=False)
result_df.to_csv("results/country_environmental_signal.csv", index=False)
print("=== Experiment 1: country environmental signal ===")
print(result_df.to_string(index=False))

# ---------------------------------------------------------------------
# Experiment 2: is the heterogeneity real / quantified?
# ---------------------------------------------------------------------
countries_with_data = list(country_effects.keys())
print(f"\n{len(countries_with_data)} countries with enough data for effect sizes: {countries_with_data}")

# Build a country x variable effect-size matrix and look at spread per
# variable, plus which variable "wins" per country.
matrix = pd.DataFrame(country_effects).T  # rows=country, cols=PPT/TMAX/SOILMOISTURE anomaly
matrix.columns = [c.replace("_anomaly", "") for c in matrix.columns]
matrix.to_csv("results/country_effect_size_matrix.csv")
print("\nEffect-size matrix (Cliff's delta, onset vs non-onset):")
print(matrix.round(3).to_string())

dominant_counts = result_df["dominant_variable"].value_counts(dropna=False)
heterogeneity_summary = {
    "n_countries_evaluated": len(countries_with_data),
    "dominant_variable_distribution": dominant_counts.to_dict(),
    "PPT_effect_range": [float(matrix["PPT"].min()), float(matrix["PPT"].max())] if "PPT" in matrix else None,
    "TMAX_effect_range": [float(matrix["TMAX"].min()), float(matrix["TMAX"].max())] if "TMAX" in matrix else None,
    "SOILMOISTURE_effect_range": [float(matrix["SOILMOISTURE"].min()), float(matrix["SOILMOISTURE"].max())] if "SOILMOISTURE" in matrix else None,
}
with open("results/heterogeneity_summary.json", "w") as f:
    json.dump(heterogeneity_summary, f, indent=2)
print("\n=== Experiment 2: heterogeneity summary ===")
print(json.dumps(heterogeneity_summary, indent=2))
