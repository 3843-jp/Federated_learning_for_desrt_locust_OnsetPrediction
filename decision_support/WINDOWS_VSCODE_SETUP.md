# Windows + VS Code Setup and Execution Guide

Extract `capstone_full_project.zip` first. You'll get two folders side by side: `locust_audit/` (research pipeline, already run, produces the frozen model) and `decision_support/` (the runnable system, dashboard, tests). **For your demo, you only need `decision_support/`.** `locust_audit/` is included for completeness/reproducibility, not because you need to re-run it.

## 1-8. Environment Setup (PowerShell)

```powershell
# 1. Check Python (3.10+ required; 3.12 is what this project was built/tested on)
python --version

# 2. Open the project folder in VS Code
code capstone_full_project

# 3. Open a terminal inside VS Code (Ctrl+`), then create a virtual environment
cd decision_support
python -m venv venv

# 4. Activate it
.\venv\Scripts\Activate.ps1
# (If PowerShell blocks the script: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass, then retry)

# 5. Upgrade pip
python -m pip install --upgrade pip

# 6. Install dependencies
pip install -r requirements.txt

# 7. Verify key packages installed correctly
python -c "import torch, flwr, shap, streamlit, sklearn; print('all imports OK')"

# 8. Verify the dataset path exists
dir frozen_inputs\data\forecasting\locust_onset_anomaly_dataset.csv
```

If step 8 doesn't find the file, the zip extraction path is wrong -- the CSV must be at `decision_support\frozen_inputs\data\forecasting\locust_onset_anomaly_dataset.csv` relative to wherever you run commands from.

## 9-11. Verify the Frozen System (no retraining needed)

```powershell
# 9. Verify frozen artifact hashes (record these -- compare after any run)
Get-FileHash config\frozen_method_a_model.npz -Algorithm MD5
Get-FileHash config\frozen_scaler.npz -Algorithm MD5
Get-FileHash config\frozen_model_metadata.json -Algorithm MD5
# Expected: 87779c59918d7a9fbab69e7ae689e7c9 / 3471443b2f780bc7748cb3532deaba7b / 67ace2a1988a4361c0be37615e83b0cd

# 10. Run the full test suite
python -m pytest tests/ -v
# Expected: 19 passed, 6 warnings (Ray-internal, harmless), ~2 minutes

# 11. Confirm hashes are unchanged after testing
Get-FileHash config\frozen_method_a_model.npz -Algorithm MD5
# Must match the value from step 9 exactly
```

## 12. Launch the Dashboard

```powershell
streamlit run early_warning_page.py
```
Opens automatically in your browser at `http://localhost:8501`. Select Country, Region, Forecast month, and the prediction, risk level, SHAP explanation, and recommendation appear immediately.

## Optional: Re-running the Research Pipeline (`locust_audit/`)

You do not need this for the demo -- it's what *produced* the frozen model. Only run it if you want to verify the training pipeline from scratch:
```powershell
cd ..\locust_audit
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pandas numpy scipy scikit-learn torch "flwr[simulation]"
python clean_and_transform.py
python build_anomaly_dataset.py
# Phase 6/7/8 federation runs take several minutes each and require the
# same working-directory discipline documented in run_final_experiment.py's
# docstring (cwd must be locust_audit/ for Flower's Ray workers to resolve
# imports correctly).
```

---

# Actual Dataset Details

| | |
|---|---|
| Raw file | `LocustLensFinalDataset.csv` (GLAD42, verified against the published LocustLens paper) |
| Raw rows / columns | 79,528 rows / 8 columns |
| Engineered file | `locust_onset_anomaly_dataset.csv` |
| Engineered rows / columns | 56,787 rows / 36 columns |
| Countries | 42 |
| Target | `target_t1` -- locust presence reported next month |
| 9 frozen model features | `PPT_T, TMAX_T, SOILMOISTURE_T, PPT_anomaly, TMAX_anomaly, SOILMOISTURE_anomaly, PPT_anomaly_missing, TMAX_anomaly_missing, SOILMOISTURE_anomaly_missing` |
| Client definition | one federated client per country |
| Split | train <=2013, validation 2014-2017, test 2018-2020 |

**Why each engineered feature exists, in plain terms for faculty:**
- Raw PPT/TMAX/SOILMOISTURE -- this month's actual weather.
- *_anomaly -- how far this month's weather is from that region's own historical normal (a region's "unusually wet" matters more than raw rainfall, since normal rainfall varies hugely by region).
- *_anomaly_missing -- flags when a region didn't have enough climate history (3+ years) to trust the anomaly calculation, so the model can distinguish "normal conditions" from "we don't actually know."

---

# Actual Model + FL Details

`Input (9 features) -> country clients (21) -> local logistic regression -> federated aggregation (FedAvg / Adaptive A / Adaptive B) -> global model -> onset score`

- **FedAvg:** Flower's built-in strategy, weight proportional to training sample count.
- **Adaptive A:** `raw_A(i) = sqrt(n_i) * (1 + ln(1 + p_i))` -- verified in `phase7_weighting.py`.
- **Adaptive B:** `raw_B(i) = [n_i/(n_i+k)] * [p_i/(p_i+m)]` -- same file, `k`/`m` = training-set medians.
- Both capped at 25% max client weight share.

---

# Actual Evaluation Metrics (real numbers, queried fresh this session)

| Method | Mean PR-AUC (8 clients, 3 seeds) | Kenya ROC-AUC | Uganda ROC-AUC |
|---|---|---|---|
| Centralized (1 seed) | -- | 0.418 | 0.938 |
| FedAvg (3 seeds) | 0.1005 | 0.442 | 0.676 |
| Adaptive A (3 seeds) | 0.0615 | 0.547 | **0.726** |
| Adaptive B (3 seeds) | 0.0776 | **0.621** | 0.604 |

**Adaptive aggregation does not beat FedAvg on average** across the fairly-compared 8-client set -- it helps Kenya (both methods) and Uganda (Method A only), and this is reported as a client-dependent effect, not a universal win.

---

# Actual Live Demo Predictions (real model output, queried fresh this session)

| | Kenya | Uganda (zero-positive) | Afghanistan (zero-positive) |
|---|---|---|---|
| Region | Baringo | Abim | Kabul Province |
| Forecast base to target month | 2018-01 to 2018-02 | 2018-01 to 2018-02 | 2018-01 to 2018-02 |
| Actual historical outcome | 0 (no report) | 0 | 0 |
| Predicted onset score | 0.56 | 0.57 | 0.46 |
| Risk level | MODERATE | MODERATE | MODERATE |
| Training samples / positives | 13,076 / 2 | 5,916 / **0** | 1,740 / **0** |
| Notice shown | "Limited positive-history evidence." | "This client has no positive training examples..." | "This client has no positive training examples..." |
| Recommendation | Increase surveillance and monitoring | Increase surveillance and monitoring | Increase surveillance and monitoring |

No value above was invented -- every number came from a live call to `decision_support.predict_score()` / `explain()` / `data_evidence()` this session.

---

# Test Results (fresh run this session)

```
19 passed, 6 warnings in 126.70s
0 failed, 0 errors
```
Warnings are Ray-internal (`ResourceWarning` on unclosed `/dev/null` handles during actor teardown), not implementation defects.

# Remaining Issues

None blocking. The dashboard UI now includes the header research-simulation badge, "Who is this for?" scope section, and Methodology expander requested for this phase, verified live and hash-checked before/after.
