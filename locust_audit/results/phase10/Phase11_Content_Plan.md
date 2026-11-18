# Phase 11 — Capstone Presentation + Research Paper: Content Plan

**Status: planning only.** No PPT file, no paper draft. Every claim below is traceable to an existing Phase 0–10 result or file; nothing here is a new experiment, a new number, or a new citation invented for this document. Where the project doesn't yet have a real reference, that's listed explicitly in Section F rather than filled in with something unverified.

---

## PART A — PPT Content (12 slides)

**Slide 1 — Title**
"Federated Learning-Based Early Warning and Recommendation System for Desert Locust Onset Prediction." Team/guide/institution placeholders (to be filled by the team, not invented here).

**Slide 2 — Introduction**
Plain-English framing: Desert Locust swarms damage crops across a border-spanning region (West Africa to South-West Asia); effective response depends on early warning; agencies may not be able to share raw surveillance data across borders; this project investigates whether Federated Learning can support cross-border prediction without pooling raw data, and specifically whether a targeted fix to standard federated aggregation helps under real country-level heterogeneity. Evidence: Phase 0 problem framing, Phase 1 dataset audit.

**Slide 3 — Problem Statement**
Four numbered points, each with a one-line evidence anchor:
- Desert Locust onset prediction is hard and imbalanced (Phase 2: persistence dominates aggregate F1; Phase 3: raw-feature onset detection is near-chance, ROC-AUC 0.473).
- Country-level data is genuinely heterogeneous, not just assumed to be (Phase 5: Cliff's delta effect sizes flip sign between countries, e.g. temperature anomaly Uganda delta=-0.596 vs. Mali delta=+0.924).
- Standard FedAvg has a specific, demonstrated weakness under this heterogeneity: a high-volume, low-signal client (Kenya, 13,076 samples, 2 positives) receives 31.4% of aggregation weight, diluting signal from smaller positive-rich clients (Phase 6-7).
- This motivates investigating adaptive, training-statistics-only client reweighting as a targeted countermeasure (Phase 7-8), evaluated honestly rather than assumed to work (Phase 10).

**Slide 4 — Existing Work / Literature Gap**
Only claims already established in this project's own research (Phase 0):
- Centralized ML approaches to locust prediction exist and are established (climate/soil-moisture features + Random Forest/XGBoost/KNN/logistic regression).
- The most directly relevant prior work is LocustLens (Khan et al., PeerJ Computer Science, 2024), which built the GLAD42 dataset this project uses and trained a same-month presence classifier.
- A literature search conducted at the start of this project (Phase 0) found no published work applying Federated Learning to Desert Locust prediction in any form.
- Gap investigated: not "no one has thought of cross-border collaboration" but specifically whether federated methods, and specifically adaptive aggregation, behave well under the real heterogeneity this data exhibits.

**Slide 5 — Dataset**
GLAD42 (via the LocustLensFinalDataset.csv file), forensically verified against the published paper (Phase 1: row count, summary statistics, and one specific worked example all match). 79,528 raw rows, 42 countries, 1985-2020. Columns: region, country, year, month, precipitation, max temperature, soil moisture, locust-present label. State plainly: this is the pre-merged FAO Locust Hub x TerraClimate dataset from the source paper, not something this project merged from scratch (Phase 1 audit finding, repeated in the dataset-clarification review).

**Slide 6 — Feature Engineering**
Exact pipeline, matching the instructed diagram:
`Raw data (8 cols) -> cleaning + T->T+1 restructuring (Phase 1C) -> climate baseline/climatology (leakage-safe, expanding window, Phase 4) -> environmental anomalies -> missingness indicators -> 36-column engineered dataset -> 9 frozen model features`. Explicitly state the 9 features are a *selection* from the 36 engineered columns, all derived from the same original 8-column source file, not additional external data (per the audit's Section 8 finding).

**Slide 7 — Proposed Architecture**
The exact diagram requested, with the explicit label: **"Country-level Federated Learning: research simulation on a single machine, not a live multi-country deployment"** placed directly on or immediately beneath that stage of the diagram, not only in a footnote.

**Slide 8 — Federated Learning**
Clients = 21 countries with sufficient training data (Phase 6 eligibility check); local training = per-client logistic regression fit; server aggregation = Flower's `FedAvg` strategy; why heterogeneity matters = Slide 3's effect-size evidence, restated briefly. State plainly that this is a simulated multi-client setup on one machine using data from a single pre-existing public dataset split by country field, not real institutions exchanging real private data.

**Slide 9 — Adaptive Aggregation**
Exact formulas from the actual code (Phase 7): `raw_A(i) = sqrt(n_i) * (1+ln(1+p_i))`, `raw_B(i) = [n_i/(n_i+k)] * [p_i/(p_i+m)]`, both computed from training-set statistics only, both capped at 25% max client share. One sentence, verbatim in spirit from the critical rules: *"Adaptive aggregation produced client-dependent effects and improved selected clients but did not outperform FedAvg on average."* Show the Kenya weight-share example (31.4% to 10.0%/3.4%) as the concrete mechanism illustration.

**Slide 10 — Results**
One table, five methods, with a visual/textual separator between the fairly-compared block and the asymmetric block:

*Fairly compared (8 clients, 3 seeds):* FedAvg, Adaptive A, Adaptive B, mean PR-AUC 0.1005 / 0.0615 / 0.0776; mean F1 0.0733 / 0.0921 / 0.0867.
*Less symmetric (single seed, fewer clients):* Centralized (3 clients), Uganda ROC-AUC 0.938; Local-only (6 clients, 2 uncomputable for Uganda/Afghanistan, zero local positives).
Client-level Kenya/Uganda table from Phase 10 Section 5, since that's the project's clearest and most defensible result.

**Slide 11 — Novelty, Findings, Limitations**
Novelty: exactly the three bullets given in the instructions, verbatim in substance. Findings: exactly the four bullets given (adaptive aggregation NOT universally better; FedAvg strongest average PR-AUC among fairly-compared FL methods; Adaptive A highest mean F1; benefits client-dependent). Limitations: exactly the four given (8/21 clients evaluable; single-seed centralized/local-only; thin-evidence client effects e.g. Somaliland's 4 test positives; single-machine simulation).

**Slide 12 — Conclusion & Future Work**
Honest conclusion: this project demonstrates a real, mechanistically-explained, multi-seed-stable improvement from adaptive client reweighting for specific heterogeneous, data-sparse clients (Kenya, and Uganda under Method A), while showing plainly that this does not generalize to a uniform federated-learning improvement across the client population. Future work: multi-seed centralized/local-only runs; investigating why Somaliland/Yemen are harmed; an intermediate reweighting formula between A and B (Phase 10's three "if necessary" recommendations), framed as optional, not promised.

---

## PART B — Research Paper Section Outline

**1. Abstract** — What to write: one-paragraph summary mirroring the capstone report's abstract structure (problem, method, honest result). Evidence: Phase 0 abstract draft, Phase 10 verdict (MODERATE contribution). External need: none, this is synthesis of existing material.

**2. Introduction** — Desert Locust impact, cross-border data-sharing constraint, FL as a candidate technical answer, this project's evidence-first approach (test assumptions rather than assume them). Evidence: Phase 0 Section 1. External need: 1-2 citations for Desert Locust economic/food-security impact figures (Section F).

**3. Problem Statement** — The four points from Slide 3, expanded with the specific numbers. Evidence: Phases 2, 3, 5, 6, 7. External need: none.

**4. Related Work / Literature Review** — LocustLens/GLAD42 in detail (methodology, what it did and didn't test); brief survey of other centralized locust-ML approaches found in Phase 0's search; explicit statement of the FL-for-locust gap found (and the caveat that "no published work found" is time-bound to when the search was run, not a permanent claim). Evidence: Phase 0 Section 2 (Related Work), Phase 1 forensic audit (GLAD42 verification). External need: full bibliographic details for the centralized-ML papers only informally described in Phase 0 (Section F).

**5. Dataset and Feature Engineering** — GLAD42 provenance and verification (Phase 1); cleaning and T->T+1 construction with the real reduction numbers (79,528 to 59,536 to 57,987); climatology/anomaly construction with the leakage-safety proof (Phase 4); the 36-vs-9 column distinction (dataset audit). Evidence: Phases 1, 1C, 4, dataset-clarification audit. External need: none, this section is entirely internal evidence.

**6. Proposed Methodology** — Overall system design (Slide 7's diagram, prose version); explicit statement of what's simulated vs. what would be needed for real deployment. Evidence: Phase 6 architecture design. External need: none.

**7. Federated Learning Setup** — Client definition and eligibility criteria (Phase 6, Category A/B split with exact counts); shared model architecture (why plain logistic regression, not the Random Forest used in earlier centralized phases, FedAvg's weight-averaging requires a linear model); Flower implementation details (rounds, local epochs, participation). Evidence: Phase 6 report in full. External need: a citation for FedAvg itself (McMahan et al. 2017) and for Flower (Section E already covers this, already used informally, needs formal citation).

**8. Adaptive Aggregation Methods** — Both formulas, their derivation logic (why sqrt/log for A, why the two-term saturating product for B), the capping/redistribution procedure, and critically the weight-freeze audit proving training-only computation. Evidence: Phase 7 full report, Phase 8 weight-freeze audit. External need: optional citation for prior non-uniform/adaptive FL aggregation work in general ML literature (FedProx, q-FedAvg, etc., if the team wants to position this relative to existing adaptive-FL literature, currently NOT cited anywhere in this project, flagged in Section F).

**9. Experimental Setup** — Temporal split (train <=2013/val 2014-17/test 2018-20) and why (Mauritania's coverage cutoff, upsurge-inclusive test window); the 5-method comparison design; seeds (42/123/2024) and which methods got multi-seed treatment vs. not; metrics (PR-AUC primary, ROC-AUC/F1/precision/recall secondary) and why accuracy was excluded. Evidence: Phase 2 (split rationale), Phase 8 (seeds), Phase 10 (metric justification). External need: none.

**10. Results and Discussion** — Phase 10's full findings, exactly as reported: global means, client-level table, paired comparison (3-of-8 improved/5-of-8 harmed for both methods), the mechanism explanation via the weight table, the two research-question halves (RQ3's yes/no split). This section carries the paper's actual scientific weight. Evidence: Phase 10 report in full, including its "do not force a positive result" framing. External need: none, but this section should NOT be softened relative to the Phase 10 report during drafting.

**11. Early Warning and Recommendation Layer** — Decision-support architecture (Phase 9): score-not-probability display with the calibration finding (ECE 0.478) as the reason; SHAP explanation with the environmental/data-quality split; deterministic recommendation rule table; zero-positive-client handling. Evidence: Phase 9 implementation and its two review rounds. External need: a citation for SHAP/LinearExplainer (Lundberg & Lee, 2017), used but never formally cited (Section F).

**12. Limitations** — Consolidate Phase 10 Section 11 plus Phase 9's observed-report/calibration/simulation caveats into one comprehensive section, this is likely the paper's longest honest-accounting section, and should not be trimmed for length.

**13. Conclusion** — Mirror Slide 12; state the MODERATE-contribution verdict explicitly rather than only implying it through hedged language.

**14. References** — Section E (below) for what exists; Section F for what's needed.

---

## PART C — Figures/Tables to Create

1. **Client-level results table** — all 5 methods x Kenya/Uganda/Afghanistan (+ the other 5 computed clients), from Phase 10 Section 5. (Table)
2. **Aggregation weight comparison chart** — FedAvg vs. Adaptive A vs. Adaptive B weight share per client, highlighting Kenya's 31.4%->10.0%->3.4% collapse. (Bar chart, from `phase7_weight_tables.csv`)
3. **Kenya/Uganda small-multiples plot** — ROC-AUC across all 5 methods with 3-seed error bars where available. (Grouped bar chart with error bars)
4. **Paired comparison chart** — "3 of 8 improved, 5 of 8 harmed" for each adaptive method, visualized as a diverging bar chart (positive/negative delta per client), not just a summary sentence. (Diverging bar chart, from `paired_comparison_vs_fedavg.csv`)
5. **Pipeline diagram** — raw data -> engineered dataset -> 9 frozen features -> model (Slide 6/architecture, reused from the dataset audit).
6. **Effect-size heatmap** — country x environmental variable Cliff's delta matrix from Phase 5, to visually ground the heterogeneity claim (the Uganda/Mali sign flip should be visually obvious).
7. **Dashboard screenshot(s)** — the actual Early Warning & Recommendation Streamlit page, real output (Kenya/Baringo/Nov 2018 demo), not a mockup, since a real screenshot exists.
8. **Data reduction funnel** — 79,528 -> 59,536 -> 57,987 -> 56,787 rows across cleaning stages (simple funnel or step chart).

## PART D — Information Still Missing Before Final Paper

- Team member names, faculty guide name, institution name (Slide 1 placeholders, not something this phase can supply).
- A decision on whether to pursue any of Phase 10's three "if necessary" strengthening experiments before submission, or submit with the current evidence as-is (explicitly the reviewer's call, not assumed here).
- Formal bibliographic entries for every citation in Section F below, currently these are known-but-uncited or informally-described-but-not-verified.
- A decision on how prominently to feature the Somaliland negative result in the paper's abstract/introduction vs. only in Results/Limitations (parallel to the same open question already flagged in Phase 8).
- Confirmation of the venue/format this paper targets (conference length limits, formatting template), affects how much of Part B's outline can actually fit.

## PART E — References Already Present in the Project (usable, need formal citation formatting only)

- Khan, R. et al. — LocustLens (PeerJ Computer Science, 2024) — GLAD42 dataset source, verified directly against this project's data file (Phase 1).
- Abatzoglou, J.T. et al. — TerraClimate (Scientific Data, 2018) — the climate data source underlying GLAD42's environmental variables.
- McMahan, H.B. et al. — FedAvg / "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS, 2017) — the baseline algorithm this project's Adaptive A/B modify.
- Flower (flwr) — federated learning framework used for all FL implementation (Phases 6-8); needs its specific citation (exact citation to be confirmed, Section F).
- FAO Desert Locust Information Service / Locust Hub — the original data-collecting body behind GLAD42, appropriate to cite as the data's ultimate source even though this project used the pre-merged file.

## PART F — Literature That Must Be Researched Externally

- **Formal, verified citation for the Flower framework** (only informally referenced as "flwr" throughout this project; needs the actual paper/citation confirmed, not assumed).
- **SHAP / LinearExplainer** (Lundberg & Lee, 2017, "A Unified Approach to Interpreting Model Predictions"), used extensively in Phase 9 but never formally verified/cited in this project's own research process.
- **Adaptive/non-uniform federated aggregation literature** (e.g. FedProx, q-FFL, FedNova), this project's Adaptive A/B were derived from first-principles reasoning about the observed Kenya dilution problem, not from an existing adaptive-aggregation paper; if the paper wants to position this work relative to that literature, those papers need to be found and read, not assumed similar.
- **Desert Locust economic/food-security impact figures** for the Introduction (the 2019-2022 upsurge's damage estimates were referenced informally in Phase 0's original problem framing but not pinned to a specific verifiable source).
- **Any additional centralized Desert Locust ML papers** beyond LocustLens that Phase 0's search surfaced only informally (geospatial deep learning/Prithvi foundation model work, neuro-fuzzy studies), these were mentioned in Phase 0's search summaries but not carried forward with full bibliographic rigor; needs re-verification before citing formally.
- **Class-imbalance / calibration literature** to support Phase 9's ECE-based decision to display "score" rather than "probability," if the paper wants a citation backing that methodological choice rather than relying only on this project's own diagnostic.

---

Stopping here, as instructed: no PPT file and no paper draft produced. Awaiting approval before generating either.
