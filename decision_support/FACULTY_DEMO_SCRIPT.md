# Faculty Demo Script and Q&A — Final Consolidated Version

## 3-5 Minute Demo Script

**1. Problem** — "Desert Locust swarms damage crops across countries in West Africa, the Horn of Africa, and South-West Asia. Countries collecting locust data may not be able to share raw records across borders, so we investigate whether Federated Learning can support prediction without pooling that data."

**2. Dataset** — "We use GLAD42 — FAO locust observations joined with TerraClimate weather data, from a 2024 peer-reviewed paper. 79,528 records, 42 countries, 1985 to 2020. We independently verified this is the genuine dataset before using it."

**3. Feature engineering** — "We build a next-month forecasting task: this month's weather and how unusual it is for that region, predicting next month's locust presence report. That gives 9 model features engineered from the same original 8 columns."

**4. Federated Learning** — "Each of 21 countries is a federated client, training locally; only model updates go to a central server. We compare standard aggregation, FedAvg, against two adaptive reweighting methods we designed."

**5. Adaptive aggregation results** — "For Kenya, FedAvg gets ROC-AUC 0.44; our adaptive Method B raises that to 0.62, stable across three seeds. For Uganda, which has zero local positive examples, Method A raises 0.68 to 0.73. But averaged across all 21 clients, adaptive aggregation does not beat FedAvg overall -- it helps specific clients, not universally."

**6. Live prediction** -- run the dashboard now -- "Kenya, Baringo, January 2018. The model outputs a predicted onset score of 0.56 -- MODERATE risk."

**7. SHAP explanation** — "Soil moisture and rainfall pushed this prediction up; these are shown separately from data-quality flags, so no one confuses a data gap with an environmental cause."

**8. Recommendation** — "MODERATE maps to a fixed rule: increase surveillance and monitoring. This is a lookup table -- the system structurally cannot recommend something like pesticide application."

**9. Target users** — "This dashboard is built for locust-monitoring authorities and field surveillance teams, not farmers directly. Farmers are indirect beneficiaries who'd receive a warning through extension services -- a farmer-facing app is future work, not part of this system."

**10. Novelty** — "Our contribution is the two reweighting formulas -- a real, code-verified change to standard FedAvg, validated across multiple random seeds."

**11. Limitation / future work** — "This is a single-machine research simulation, not real countries exchanging real data. The score isn't a calibrated probability -- we checked that directly. Future work: multi-seed baselines for centralized/local-only, and investigating why some clients are harmed by reweighting."

---

## Faculty Q&A

**What is your input?** Nine engineered features: three raw environmental variables, three anomalies relative to regional climate history, three data-quality flags.

**What is your output?** A 0-1 onset score, LOW/MODERATE/HIGH risk tier, plain-language warning, top contributing features, and a fixed recommendation.

**What is your target?** Whether a country/region reports locust presence next month.

**Why this dataset?** It's the only verified, peer-reviewed dataset combining locust observations with matched climate data at this resolution -- we confirmed it matches the source paper before using it.

**Why Federated Learning?** Countries may not be able to share raw surveillance data across borders; FL lets them collaborate on a model without pooling it.

**Why country-level clients?** It's the real-world unit -- a national agency is who would run a local model in an actual deployment.

**Why not centralized?** We test it as our upper bound (Uganda: 0.938) -- it's not viable when raw data can't be pooled, which is the whole premise we're testing against.

**What is FedAvg?** The standard federated algorithm: local training, then server-side averaging weighted by each client's data volume.

**What is adaptive aggregation?** Our modification: weighting by a formula combining data volume and positive-example count, computed only from training statistics, so high-volume but low-signal clients don't dominate.

**What is your novelty?** The two reweighting formulas -- a genuine, code-level departure from stock FedAvg, validated across seeds, with a documented mechanism.

**Why Logistic Regression?** FedAvg's weight-averaging is well-defined for linear models; it isn't for the tree ensembles used in earlier centralized experiments.

**Why 9 features?** Selected from 36 engineered columns -- the minimal, non-redundant set representing current conditions, historical-normal deviation, and data reliability.

**Why anomaly features?** A region's "unusually wet" matters more for locust breeding conditions than raw rainfall, since normal rainfall varies enormously by location.

**Why missingness flags?** They let the model distinguish "genuinely normal conditions" from "we lack enough history to know" -- without them, missing climatology would silently look identical to a real zero anomaly.

**Why PR-AUC?** Our positive class is under 4% of test data; accuracy would look deceptively high for a useless model, PR-AUC and F1 reflect performance on the rare, important class.

**What does 0.58 mean?** The model's raw output for this input -- higher means the model considers onset more likely, but the number itself isn't a percentage chance.

**Is 0.58 a probability?** No. We tested calibration directly -- Expected Calibration Error of 0.478 -- and it isn't calibrated, so we call it a score, not a probability.

**Who is the target user?** Agricultural and locust-monitoring authorities, extension officers, and field surveillance teams -- not farmers directly.

**Can illiterate farmers use this dashboard?** No -- this is a technical dashboard for monitoring authorities. Farmer accessibility (mobile, local-language, voice) is explicitly future work, not part of the current system.

**Does this predict a confirmed outbreak?** No -- it predicts observed-report onset, whether a report appears in the data. Negative means "not reported," not "confirmed absent," so we never claim biological outbreak prediction.

**Is this real multi-country FL?** No -- a single-machine research simulation using one public dataset split by country field, not real institutions exchanging real data.

**Does this require special hardware?** No -- the full 21-client simulation runs on a single CPU core in about 13 seconds.

**What are the limitations?** Only 8 of 21 clients have enough test data to evaluate reliably; centralized/local-only baselines are single-seed; the score is uncalibrated; some client effects rest on very few positive examples.

**What is future work?** Multi-seed centralized/local-only runs; investigating why specific clients are harmed by reweighting; a farmer-facing communication layer; testing an intermediate reweighting formula.
