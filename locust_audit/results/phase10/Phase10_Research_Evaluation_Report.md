# Phase 10 — Research Evaluation and Novelty Validation

**No new experiments were run for this phase's core findings.** Every number below comes from result files that already existed before this phase started (`phase6_*.csv`, `phase7_*.csv`, `phase8_final_results.csv`, `phase8_summary.csv`). Two new files were created — `phase10/master_all_methods_results.csv` (a consolidation of existing files, no new computation) and `phase10/paired_comparison_vs_fedavg.csv` (arithmetic on existing numbers) — per Part 11's request for machine-readable outputs, not because prior results were insufficient.

**Headline finding, stated up front because Section 9 requires it:** averaged across all 8 clients with usable test-period data, **both adaptive methods have a negative mean ROC-AUC difference versus vanilla FedAvg** (Method A: −0.053; Method B: −0.008), driven by severe degradation at Somaliland. The positive result this project has emphasized — Method A helping Kenya and Uganda simultaneously — is real and multi-seed-validated, but it is **not** a global improvement across the client population. Both things are true at once, and this report holds them together rather than leading with the favorable framing.

---

## 1. Existing Experiments Discovered

| Method | Source file(s) | Seeds | Countries covered |
|---|---|---|---|
| A. Centralized | `phase6_centralized_per_client.csv`, `phase8_final_results.csv` (method="centralized") | 1 (deterministic — no seed dependence in a full-batch closed-form-adjacent LR fit) | Kenya, Uganda, Afghanistan |
| B. Local-only | `phase6_local_only_per_client.csv` | 1 (seed 42, Phase 6 only — **never re-run at seeds 123/2024**) | 21 attempted; 6 have `status=computed` (Kenya, Yemen, Saudi Arabia, Oman, Somaliland, Djibouti); 2 report `CANNOT TRAIN` (Uganda, Afghanistan — zero local positives); 13 have no test-period data |
| C. FedAvg | `phase6_fedavg_per_client.csv` (seed 42 only, superseded), `phase8_final_results.csv` (method="fedavg") | **3 (42, 123, 2024)** | 21 clients, 8 with `status=computed` |
| D. Adaptive Method A | `phase7_adaptive_A_results.csv` (seed 42 only, superseded), `phase8_final_results.csv` (method="adaptive_A") | **3 (42, 123, 2024)** | Same 8 as FedAvg |
| E. Adaptive Method B | `phase7_adaptive_B_results.csv` (seed 42 only, superseded), `phase8_final_results.csv` (method="adaptive_B") | **3 (42, 123, 2024)** | Same 8 as FedAvg |

**No ambiguity found requiring a stop** — `phase8_final_results.csv` is unambiguously the authoritative, most-complete source for FedAvg/Adaptive A/Adaptive B (it supersedes the Phase 6/7 single-seed files, which were intermediate checkpoints later validated across 3 seeds). Centralized and local-only were never re-run multi-seed; this asymmetry is real and is flagged formally in Section 6, not glossed over.

## 2. Method Definitions (from actual code, not re-described)

| Method | Training approach | Aggregation | Purpose |
|---|---|---|---|
| Centralized | Single logistic regression fit on all 21 clients' training data pooled together | N/A (no aggregation step) | Upper-bound reference: what's achievable if raw data pooling were possible |
| Local-only | Separate logistic regression fit per client, using only that client's own training data | N/A | Lower-bound reference: what a client gets with zero collaboration |
| FedAvg | `flwr.server.strategy.FedAvg` (Flower's built-in class, unmodified) | Client updates averaged, weighted by `num_examples` (raw training sample count) — Flower's default behavior | Standard federated baseline |
| Adaptive A | `StaticWeightedFedAvg(client_weights=adaptive_A)` — identical aggregation code to FedAvg except the weight source | Weighted by a **precomputed, training-statistics-only** dict: `raw_A(i) = sqrt(n_i) × (1 + ln(1 + p_i))`, normalized, capped at 25% per client | Test whether down-weighting high-volume/low-positive clients and up-weighting positive-rich clients improves federated performance |
| Adaptive B | Same `StaticWeightedFedAvg` class, different weight dict | `raw_B(i) = [n_i/(n_i+k)] × [p_i/(p_i+m)]`, k/m = training-set medians, normalized, capped at 25% | Test a stronger reweighting that requires both volume *and* positive signal, as a contrast to A |

Nothing here is invented — this table is a direct transcription of `phase6_model.py` (centralized/local training loop), `phase7_strategy.py` (`StaticWeightedFedAvg`), and `phase7_weighting.py` (the two formulas), read from the files themselves for this report.

## 3. Evaluation Metrics

**Available in the source files and used below:** PR-AUC, ROC-AUC, F1, precision, recall — all five are present in every `computed`-status row across `phase6_local_only_per_client.csv` and `phase8_final_results.csv`. **PR-AUC is treated as primary** (per this project's own established convention since Phase 3, given the target's severe imbalance — pooled test-set positive rate is under 4%), ROC-AUC and F1 secondary. **Accuracy was never computed anywhere in this project's result files** and is correctly absent — reporting it now would be inventing a metric outside the original experimental design, which the instructions explicitly prohibit.

## 4. Global Results

Pooling `phase8_summary.csv`'s 3-seed means across the 8 clients with computed metrics:

| Method | Mean PR-AUC | Mean ROC-AUC |
|---|---|---|
| FedAvg | 0.1024 | 0.4076 |
| Adaptive A | 0.0687 | 0.4022 |
| Adaptive B | 0.0898 | 0.4494 |

**Read this table cautiously, not as a headline** — it's a simple unweighted mean across 8 clients of very different sizes and reliability (Djibouti has 1 test positive; Kenya has 36), so it treats a near-noise result the same as a well-powered one. It's reported here because Part 3 asked for global numbers, not because it's the most informative comparison — Section 6 (client-level) is.

## 5. Client-Level Results

Full data: `phase10/master_all_methods_results.csv`. Mean ROC-AUC ± SD (3 seeds) per client, FedAvg vs. both adaptive methods, plus centralized/local-only where available:

| Country | Centralized | Local-only (1 seed) | FedAvg (3 seeds) | Adaptive A (3 seeds) | Adaptive B (3 seeds) |
|---|---|---|---|---|---|
| Kenya | 0.418 | 0.463 | 0.442 ± 0.003 | **0.547 ± 0.007** | **0.621 ± 0.007** |
| Uganda | **0.938** | n/a (0 local positives) | 0.676 ± 0.027 | **0.726 ± 0.016** | 0.604 ± 0.013 |
| Afghanistan | 0.566 | n/a (0 local positives) | 0.509 ± 0.004 | 0.498 ± 0.008 | 0.500 ± 0.007 |
| Saudi Arabia | — | 0.429 | 0.655 ± 0.041 | 0.631 ± 0.021 | **0.679 ± 0.000** |
| Yemen | — | 0.133 | **0.667 ± 0.067** | 0.578 ± 0.038 | 0.467 ± 0.000 |
| Somaliland | — | 0.537 | **0.667 ± 0.022** | 0.158 ± 0.020 | 0.310 ± 0.004 |
| Oman | — | 0.708 | 0.042 ± 0.000 | 0.014 ± 0.024 | 0.042 ± 0.000 |
| Djibouti | — | 0.560 | 0.000 ± 0.000 | 0.080 ± 0.000 | **0.373 ± 0.023** |

**Variation across clients is large and method-dependent, not a minor detail:** FedAvg's ROC-AUC alone ranges from 0.000 (Djibouti) to 0.667 (Yemen/Somaliland) across clients — a 0.667-point spread with only 3 clients (Kenya, Uganda, Yemen) reaching what most of the project's earlier phases treated as "above-chance and worth discussing." **Clients with sparse or zero positive training examples:** Uganda and Afghanistan (0 positives — cannot train locally at all, per Section 1); Kenya (2 positives — technically trainable but extremely thin); several Category B clients (Mali 9, Eritrea 6, Guinea 6) never reach `computed` status because they have zero test-period positives, a structural consequence of the temporal split (documented back in Phase 2), not a new finding.

## 6. Adaptive Aggregation Analysis — Mechanism, from the Actual Code

**What information each adaptive method uses:** only two training-set statistics per client — `n_i` (training sample count) and `p_i` (training positive count) — read from `phase7_client_reliability.csv`, itself derived purely from the train split (verified in Phase 8's weight-freeze audit; test data never touched). No environmental features, no client identity beyond these two numbers, no test-set information of any kind feeds the weight formulas.

**How this differs from FedAvg concretely, using the real numbers:** vanilla FedAvg weights every client by `n_i / Σn_j`. Kenya (`n_i=13,076`) receives **31.4%** of total aggregation weight under this scheme — the single largest share of any client — despite having only 2 positive training examples. Method A's `sqrt(n_i) × (1+ln(1+p_i))` formula reduces Kenya's share to **10.0%**; Method B's stronger formula reduces it further to **3.4%**. Both formulas correspondingly *raise* the relative share of positive-rich, lower-volume clients — Algeria (47 positives, 2,044 samples) goes from 4.9% (FedAvg) to 9.1% (A) / 11.3% (B).

**Why this could plausibly help under heterogeneous data:** if a client with very little positive signal (Kenya) dominates the aggregate purely by data volume, its (mostly negative-class) gradient updates can pull the shared model's decision boundary away from what positive-rich clients are trying to teach it. Down-weighting volume-heavy, signal-poor clients and up-weighting signal-rich ones is a direct, mechanistic countermeasure to that specific failure mode — not a general claim that "personalization helps," but a targeted response to a dilution pattern that is visible in the weight table itself, independent of any downstream performance number.

**Does the evidence support this working as intended?** Partially, and unevenly (Section 5). It works exactly as intended for Kenya (both A and B) and for Uganda (A only) — the three cases most directly matching the "volume-heavy, signal-poor" profile the mechanism targets. It does not help, and in several cases actively hurts, clients whose problem isn't volume-dilution (Somaliland, Yemen) — for those clients, the reweighting removes signal (via redistributing weight away from them or toward less-relevant peers) without the compensating benefit the mechanism was built to provide.

## 7. Robustness Across Seeds

FedAvg, Adaptive A, and Adaptive B all have genuine 3-seed data (42, 123, 2024) in `phase8_final_results.csv`, with mean ± SD reported throughout Sections 4–5. Standard deviations are small relative to between-method gaps for the two primary clients (e.g., Kenya's Method B std of 0.007 vs. its 0.179 improvement over FedAvg) — the Kenya/Uganda-Method-A findings are **stable across the tested seeds**, language chosen deliberately over "statistically significant," which three seeds cannot support and this report does not claim.

**Centralized and local-only have only 1 seed each.** This is stated plainly, not minimized: their numbers in Section 5 should be read as single-run reference points, not as robustness-checked results, and no confidence interval or significance claim is made for either. Local-only's Yemen result (ROC-AUC 0.133, the worst in Section 5's table) is a single seed's worth of evidence and could look meaningfully different at another seed — unverifiable with what exists today.

**Paired comparison** (`phase10/paired_comparison_vs_fedavg.csv`, mean-ROC-AUC delta per client, Method vs. FedAvg, both using the same 3-seed means so the comparison is apples-to-apples): Method A improves 3 of 8 clients, harms 5. Method B improves 3 of 8, harms 5 (a different 3/5 split). Neither method is a majority-of-clients win.

## 8. Fairness / Leakage Checks

All five methods share: same frozen dataset (`locust_onset_anomaly_dataset.csv`), same target (`target_t1`), same temporal split (train ≤2013, val 2014–2017, test 2018–2020), same 9 features, same test set. **One genuine asymmetry, flagged rather than hidden:** local-only and centralized were evaluated at a single seed while FedAvg/A/B were validated across three — this means Section 5's Uganda/Afghanistan "cannot train locally" finding and the centralized reference numbers are less robustly established than the FedAvg-family comparisons sitting next to them in the same table. No evidence of test-set leakage into weight computation was found (already audited directly in Phase 8, hashes/field-source-tracing unchanged since).

## 9. Novelty Assessment

**A. Existing/common components — not this project's contribution:** Federated Learning as a paradigm, FedAvg itself (used unmodified from Flower's library), logistic regression, SHAP, environmental anomaly feature engineering, and the early-warning dashboard are all standard techniques or engineering integration, not research novelty. This project does not claim otherwise anywhere in its result files or code.

**B. This project's actual contribution, as implemented:** the two adaptive weighting formulas (`raw_A`, `raw_B`) — precomputed, training-statistics-only client reweighting schemes designed specifically in response to an observed FedAvg failure mode (Kenya's disproportionate, low-signal aggregate influence). **This is genuinely different from standard FedAvg**, verified at the code level (Section 6, `StaticWeightedFedAvg` substitutes a fixed weight dict for `num_examples` — a real, inspectable algorithmic change, not a relabeling), and the resulting weight redistribution is large and directly traceable (Kenya 31.4%→3.4% under Method B).

**Whether the experiments support a meaningful contribution:** **conditionally, not unconditionally.** The contribution is meaningful and defensible specifically for the failure mode it targets (demonstrated, multi-seed-stable improvement for Kenya under both methods and Uganda under Method A) — that is a real, mechanistically-explained result. It is **not** a general federated-learning improvement; Section 7's paired comparison shows a minority-of-clients win rate, and Section 4's simple mean is negative for both variants. The honest framing is: *a targeted fix for a specific, identified problem, validated on the cases that problem affects, with a documented, non-trivial cost elsewhere* — not *a better aggregation algorithm*.

## 10. Research Question Answers

- **RQ1 — Can FL predict next-month onset using heterogeneous country-level data?** Weakly and unevenly. FedAvg reaches ROC-AUC 0.667 for 2 of 8 evaluable clients (Yemen, Somaliland) and near-chance (≤0.05) for 2 others (Oman, Djibouti) — "can predict" is true for a minority of clients, not the client population generally. PR-AUC values (Section 4) confirm this isn't a strong yes at the aggregate level either.
- **RQ2 — How does FedAvg compare with centralized and local-only?** Mixed, per client, not uniform (Section 5). For Uganda, centralized (0.938) dramatically beats FedAvg (0.676) — pooling raw data helps far more than federating when a client has no local signal at all. For Kenya, local-only (0.463, single seed) slightly beats FedAvg (0.442, 3-seed mean) — collaboration doesn't clearly help Kenya over going alone, though this specific comparison is asymmetric in seed count (Section 8) and should be read with that caveat.
- **RQ3 — Does adaptive aggregation improve performance under heterogeneous data vs. standard FedAvg?** **Yes, for a specific, identifiable subset of clients (Kenya under both methods, Uganda under Method A), and no, on average across the full evaluable client set** (Section 4, Section 7). Both parts of this answer are supported by the same evidence and neither should be dropped in the capstone write-up.
- **RQ4 — How does performance vary with client data amount and positive count?** Substantially. The two clients with zero training positives (Uganda, Afghanistan) cannot be evaluated locally at all; Kenya (2 positives, largest sample) is federated learning's hardest case under vanilla weighting specifically because of that volume/signal mismatch; clients with moderate positive counts and moderate volume (Yemen: 18 positives, Somaliland: 9) get FedAvg's best raw results but are the cases most harmed by both adaptive reweighting schemes — reweighting away from Kenya's dominant volume also moves weight away from these moderately-sized clients' relative position in ways that hurt them specifically.

## 11. Limitations

- Centralized and local-only results are single-seed; no robustness claim is made or possible for them.
- Only 8 of 21 clients have any test-period positive examples to evaluate on at all — a structural limitation of the frozen 2018–2020 temporal split, not something this phase can fix without changing the frozen split (explicitly out of scope).
- The "global mean" metric in Section 4 unweights very different client reliabilities (1 vs. 36 test positives) and should not be over-interpreted as a single headline number — Section 5's per-client table is the more informative source.
- Neither adaptive method was tested at more than 3 seeds; "stable across tested seeds" is the correct claim, not statistical significance.
- Somaliland's degradation under both adaptive methods (largest-magnitude effect in this entire evaluation) is only 4 test-period positives' worth of evidence — real, reported, but not something to generalize from confidently in either direction.

## 12. Recommended Figures/Tables for the Capstone Report

1. **Section 5's client-level table** (all 5 methods × 8 clients) — the single most information-dense table in this evaluation; likely the capstone's primary results table.
2. **A bar chart of the weight table** (`phase7_weight_tables.csv`: FedAvg vs. A vs. B weight per client) — makes Section 6's mechanism visually immediate (Kenya's bar shrinking from 31% to 3%).
3. **Section 7's paired-comparison table**, explicitly framed as "3 of 8 improved, 5 of 8 harmed" rather than a single mean — this is the table that keeps the capstone's claim honest per Section 9's instruction.
4. **A small multiples plot of Kenya and Uganda specifically** across all 5 methods with error bars (3-seed SD) — the clearest visual for the project's actual validated finding.
5. A short callout box or footnote explicitly stating the local-only/centralized single-seed limitation, positioned near wherever those numbers first appear, not buried in a limitations appendix.

---

**Per the strict stop condition:** no modification was made to Phase 9, the frozen Method A model, or the dataset. No results were invented — every number above traces to an existing file, and both new files created (`phase10/master_all_methods_results.csv`, `phase10/paired_comparison_vs_fedavg.csv`) are pure consolidations/arithmetic on those existing numbers, not new experiments. Stopping here for reviewer approval before proceeding to final report writing.
