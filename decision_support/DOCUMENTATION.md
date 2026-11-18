# Decision-Support Layer — Documentation

This document explains the early-warning decision-support system built on top of the frozen Phase 8 Method A model. It is a **read-only consumer** of that model — nothing here retrains, fine-tunes, or otherwise modifies it.

## What the score means, and why it's not called a probability

The frozen model outputs a raw sigmoid value between 0 and 1. In a well-calibrated classifier, this value would equal the true likelihood of the event — a well-calibrated 0.72 would mean "events like this happen about 72% of the time." **This model's output does not have that property**, and we checked directly rather than assuming either way.

`assess_calibration.py` bins the frozen model's test-set scores into quintiles and compares each bin's mean predicted score to its actual observed positive rate. The result: **Expected Calibration Error of 0.478**, and a **non-monotonic** reliability curve (predicted scores in the 0.5–0.65 range corresponded to observed positive rates of only 1–7%, and the observed rate did not consistently increase as the predicted score increased). Full numbers in `config/calibration_assessment.json`.

The most likely reason is a deliberate, correct design choice made earlier in this project, not a bug: the model was trained with **class-weighted loss** (`pos_weight` in `BCEWithLogitsLoss`) to handle severe class imbalance in the onset-prediction task. Class weighting improves the model's ability to *separate* positive from negative cases — which is what every earlier phase's PR-AUC/ROC-AUC evaluation actually measured — but it does this by deliberately shifting the sigmoid's output distribution away from true base rates. Better separation and calibrated probability are different goals, and this model was optimized for the first.

**Consequence for this dashboard:** every display path shows `Predicted onset score: 0.XX`, never a percentage or the word "probability." This is enforced in code, not just in the UI text — `format_score_display()` raises an error if the configuration is ever switched to probability mode without a fresh calibration pass being run and its result recorded.

## What LOW / MODERATE / HIGH mean

These are **configurable decision-support thresholds**, stored in `config/thresholds.json`, not hardcoded in any UI file. Current defaults: score < 0.30 → LOW, 0.30–0.60 → MODERATE, > 0.60 → HIGH. These boundaries were chosen as reasonable starting points for a prototype, not derived from any biological or epidemiological threshold — the config file states this explicitly, and every place the dashboard displays a risk tier repeats the same disclaimer.

## How SHAP is interpreted

We use `shap.LinearExplainer`, the exact (not sampled/approximate) explanation method for a linear model — appropriate because the frozen model is a plain logistic regression with no hidden layers. It decomposes a single prediction into each feature's contribution to the model's output, relative to a background distribution (a sample of the frozen training data).

A SHAP value tells you **how much a feature pushed this specific prediction up or down relative to average** — it does not tell you that the feature caused the underlying event. Every SHAP result returned by `decision_support.explain()` carries this distinction in its own data (`"caveat": "contributed to the model prediction, not a claim of scientific/biological causation"`), so the correct wording travels with the number rather than depending on whoever displays it remembering to phrase it carefully.

## Why missingness indicators are shown separately

Three of the frozen model's nine features — `PPT_anomaly_missing`, `TMAX_anomaly_missing`, `SOILMOISTURE_anomaly_missing` — are not environmental measurements. They flag whether a region/month had enough historical climatology (at least 3 prior years) to compute a trustworthy anomaly value at all. These flags were genuinely part of Phase 8 training (verified directly against the frozen model's metadata, not assumed), so they cannot be removed from inference without corrupting every prediction.

But showing "precipitation-anomaly-missing: strong contribution" as the #1 reason in a panel meant to be read in ten seconds would confuse a viewer into thinking it's an environmental signal. So `explain()` splits its output into two groups: **Environmental contributors** (the six real environmental/anomaly features) and **Data-quality indicators** (the three missingness flags, shown whenever active, with an explicit note that they reflect data quality, not conditions on the ground). Nothing is hidden — the split is organizational, not a filter.

## What the recommendations mean

The recommendation engine is a **fixed, three-row lookup table** (`config/thresholds.json`, `recommendation_rules`), keyed only by risk tier:

| Risk tier | Recommendation |
|---|---|
| LOW | Routine monitoring |
| MODERATE | Increase surveillance and monitoring |
| HIGH | Prioritize field surveillance and monitoring in the affected region |

It is never generated by the model, an LLM, or any free-form logic — `recommendation()` is a dictionary lookup. There is structurally no code path by which this system could recommend pesticide application, chemical treatment, crop destruction, evacuation, or emergency intervention; adding such an action would require someone to explicitly add a new row to this table, not a runtime decision by the system.

## Limitations of the system

1. Predictions describe **observed-report onset** (the first monitored report of activity), not confirmed biological outbreak onset.
2. Negative labels are naive/coarse absence, not confirmed ecological absence — survey-effort bias is not corrected for.
3. Model output is a **score**, not a calibrated probability (see above).
4. Some clients (e.g., Uganda, Afghanistan) have zero or very few positive training examples; their predictions rely entirely on pooled/federated signal, not client-specific history.
5. The federated learning system is a **research simulation** (Flower, single machine), not a live multi-institution deployment.
6. Risk-tier thresholds are configurable decision-support settings, not biologically validated limits.
