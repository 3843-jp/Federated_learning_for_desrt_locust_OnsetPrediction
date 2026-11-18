"""
phase7_weighting.py — Part 5 (positive-aware) and Part 6 (reliability-aware)
adaptive weighting formulas.

Both formulas are fully specified and computed from TRAINING statistics
ONLY, before any method is evaluated on test data (Part 7's absolute rule).
This script is run once, its output (results/phase7_weight_tables.csv) is
frozen and reused as-is by every subsequent experiment -- it is never
re-run or adjusted after seeing test results.

--------------------------------------------------------------------------
METHOD A -- Positive-aware aggregation
--------------------------------------------------------------------------
    raw_A(i) = sqrt(n_i) * (1 + ln(1 + p_i))

  - sqrt(n_i): sublinear in sample count -- 4x the data gives 2x the base
    weight, limiting how much a single large client (e.g. Kenya,
    n=13,076) can dominate purely by volume.
  - (1 + ln(1 + p_i)): log-scaled positive-signal boost, deliberately slow
    growing so a client with 47 positives (Algeria) gets a boost of
    ln(48)=3.87x, not 47x. Zero-positive clients (Uganda, Afghanistan)
    get exactly 1.0 -- no boost, no penalty beyond their sqrt(n) base.

--------------------------------------------------------------------------
METHOD B -- Reliability-aware aggregation
--------------------------------------------------------------------------
    k = median(n_i across all 21 clients)
    m = median(p_i across clients with p_i >= 1)
    raw_B(i) = [ n_i / (n_i + k) ] * [ p_i / (p_i + m) ]

  - Two independent saturating (Michaelis-Menten) terms, each in [0,1).
  - First term caps the marginal benefit of raw volume harder than
    Method A's sqrt term -- directly targets "large negative-only client
    dominates."
  - Second term is exactly zero when p_i=0 (Uganda, Afghanistan) -- a
    deliberately stronger design than Method A, flagged explicitly, not
    hidden -- and saturates for very positive-rich small clients too,
    targeting "tiny positive-rich client dominates."
  - The PRODUCT means a client needs both decent volume AND decent
    positive count to score highly.

Normalization/capping: both methods' raw weights are normalized to sum
to 1, then capped at MAX_CLIENT_SHARE with iterative redistribution of
excess to uncapped clients.
"""
import json
import numpy as np
import pandas as pd

MAX_CLIENT_SHARE = 0.25

rel = pd.read_csv("results/phase7_client_reliability.csv")
n = rel.set_index("country")["train_samples"]
p = rel.set_index("country")["train_positive"]

raw_A = np.sqrt(n) * (1 + np.log1p(p))

k = n.median()
m = p[p >= 1].median()
raw_B = (n / (n + k)) * (p / (p + m))

print(f"Method B smoothing constants (training stats only): k(median n)={k}, m(median p among p>=1)={m}")


def normalize_and_cap(raw, max_share=MAX_CLIENT_SHARE, max_iter=50):
    w = raw / raw.sum()
    for _ in range(max_iter):
        over = w[w > max_share]
        if len(over) == 0:
            break
        excess = (over - max_share).sum()
        w[over.index] = max_share
        under_mask = w < max_share
        under_sum = w[under_mask].sum()
        if under_sum > 0:
            w[under_mask] = w[under_mask] + excess * (w[under_mask] / under_sum)
    return w / w.sum()


weight_A = normalize_and_cap(raw_A.copy())
weight_B = normalize_and_cap(raw_B.copy())
weight_fedavg = n / n.sum()

table = pd.DataFrame({
    "country": n.index,
    "train_samples": n.values,
    "train_positive": p.values,
    "fedavg_weight_sample_count": weight_fedavg.values,
    "adaptive_A_weight": weight_A.values,
    "adaptive_B_weight": weight_B.values,
}).sort_values("train_positive", ascending=False)

table.to_csv("results/phase7_weight_tables.csv", index=False)
print(table.to_string(index=False))


def concentration_stats(w):
    w_sorted = w.sort_values(ascending=False)
    top3_share = w_sorted.iloc[:3].sum()
    entropy = -(w * np.log(w.replace(0, np.nan))).sum(skipna=True)
    max_entropy = np.log(len(w))
    return {
        "max_weight": float(w.max()), "max_weight_country": w.idxmax(),
        "min_weight": float(w.min()), "min_weight_country": w.idxmin(),
        "top3_share": float(top3_share),
        "entropy": float(entropy), "normalized_entropy": float(entropy / max_entropy),
        "n_zero_weight_clients": int((w == 0).sum()),
        "zero_weight_clients": w[w == 0].index.tolist(),
    }


concentration = {
    "fedavg": concentration_stats(weight_fedavg),
    "adaptive_A": concentration_stats(weight_A),
    "adaptive_B": concentration_stats(weight_B),
}
with open("results/phase7_weight_concentration.json", "w") as f:
    json.dump(concentration, f, indent=2, default=str)
print("\n=== Weight concentration (Part 12) ===")
print(json.dumps(concentration, indent=2, default=str))

weights_export = {"adaptive_A": weight_A.to_dict(), "adaptive_B": weight_B.to_dict()}
with open("results/phase7_weights_for_simulation.json", "w") as f:
    json.dump(weights_export, f, indent=2)
