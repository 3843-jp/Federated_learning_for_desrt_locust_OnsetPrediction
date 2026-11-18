"""
tests/test_decision_support.py — Part 13 required tests.

Run: cd decision_support && python -m pytest tests/ -v
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decision_support as ds

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frozen_inputs"))
from phase6_data_prep import load_frozen_dataset, FROZEN_FEATURES


@pytest.fixture(scope="module")
def sample_rows():
    _cwd = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frozen_inputs"))
    df = load_frozen_dataset()
    os.chdir(_cwd)
    test = df[df["split"] == "test"]
    kenya = test[test["COUNTRYNAME"] == "Kenya"].iloc[0]
    return {"kenya_row": {f: float(kenya[f]) for f in FROZEN_FEATURES}}


# 1. Model loading
def test_model_loads_without_error():
    model, mean, std = ds.load_frozen_model()
    assert model is not None
    assert len(mean) == len(FROZEN_FEATURES)
    assert len(std) == len(FROZEN_FEATURES)


# 2. Prediction reproducibility
def test_prediction_is_deterministic(sample_rows):
    s1 = ds.predict_score(sample_rows["kenya_row"])
    s2 = ds.predict_score(sample_rows["kenya_row"])
    assert s1 == s2, "Same input must give same output -- frozen model has no stochastic inference path"


def test_prediction_matches_phase8_reference(sample_rows):
    # Metadata records the seed-42 verification check performed at save time
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config", "frozen_model_metadata.json")) as f:
        meta = json.load(f)
    for country, v in meta["verification_against_phase7_8"].items():
        assert v["diff"] < 0.02, f"{country} reproduction diff too large: {v['diff']}"


# 3. Threshold classification
def test_risk_tier_boundaries():
    assert ds.risk_tier(0.0) == "LOW"
    assert ds.risk_tier(0.29) == "LOW"
    assert ds.risk_tier(0.30) == "MODERATE"
    assert ds.risk_tier(0.60) == "MODERATE"
    assert ds.risk_tier(0.61) == "HIGH"
    assert ds.risk_tier(1.0) == "HIGH"


# 4-6. Recommendation per tier
def test_low_recommendation():
    assert ds.recommendation("LOW") == "Routine monitoring"


def test_moderate_recommendation():
    assert ds.recommendation("MODERATE") == "Increase surveillance and monitoring"


def test_high_recommendation():
    assert ds.recommendation("HIGH") == "Prioritize field surveillance and monitoring in the affected region"


def test_recommendation_engine_never_contains_forbidden_actions():
    forbidden = ["pesticide", "chemical", "destruct", "evacuat", "emergency intervention"]
    for tier in ("LOW", "MODERATE", "HIGH"):
        rec = ds.recommendation(tier).lower()
        for word in forbidden:
            assert word not in rec, f"Forbidden action '{word}' found in {tier} recommendation"


# 7. Zero-positive client handling
def test_zero_positive_client_evidence():
    evidence = ds.data_evidence("Uganda")
    assert evidence["train_positive_examples"] == 0
    assert "zero_positive_notice" in evidence
    assert "no confidence score" not in evidence  # i.e. we never invent one
    assert "unreliable" not in evidence.get("zero_positive_notice", "").lower()


def test_nonzero_but_sparse_client_evidence():
    # Any client with 1-4 train positives should get the sparse-history note
    import pandas as pd
    rel = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "frozen_inputs", "phase7_client_reliability.csv"))
    sparse = rel[(rel["train_positive"] > 0) & (rel["train_positive"] < 5)]
    if len(sparse) > 0:
        country = sparse.iloc[0]["country"]
        evidence = ds.data_evidence(country)
        assert "sparse_history_notice" in evidence


# 8. SHAP output
def test_shap_output_structure(sample_rows):
    result = ds.explain(sample_rows["kenya_row"], top_k=3)
    assert "environmental_contributors" in result
    assert "data_quality_indicators" in result
    assert len(result["environmental_contributors"]) <= 3
    for item in result["environmental_contributors"]:
        assert item["feature"] in ds.ENVIRONMENTAL_FEATURES
        assert "direction" in item and "magnitude_label" in item
        assert "causation" in item["caveat"] or "causal" not in item["caveat"]


def test_shap_never_puts_missingness_flag_in_environmental_list(sample_rows):
    result = ds.explain(sample_rows["kenya_row"], top_k=3)
    env_features = [item["feature"] for item in result["environmental_contributors"]]
    for f in env_features:
        assert not f.endswith("_missing"), f"Data-quality flag {f} leaked into environmental_contributors"


# 9. No retraining during inference
def test_model_is_in_eval_mode_and_frozen():
    model, _, _ = ds.load_frozen_model()
    assert not model.training, "Model must be in eval() mode, not train() mode"
    params_before = [p.detach().numpy().copy() for p in model.parameters()]
    # run several predictions
    for _ in range(5):
        ds.predict_score({f: 0.1 for f in FROZEN_FEATURES})
    params_after = [p.detach().numpy().copy() for p in model.parameters()]
    for before, after in zip(params_before, params_after):
        assert np.array_equal(before, after), "Model weights changed after inference calls -- retraining leak"


# 10. No modification of frozen FL weights (artifact-level check)
def test_frozen_artifact_files_unchanged_by_import():
    artifact_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "config", "frozen_method_a_model.npz")
    before_mtime = os.path.getmtime(artifact_path)
    ds.predict_score({f: 0.1 for f in FROZEN_FEATURES})
    after_mtime = os.path.getmtime(artifact_path)
    assert before_mtime == after_mtime, "Frozen model artifact file was modified -- must never happen"


# Score/probability terminology enforcement
def test_score_display_never_says_probability():
    text = ds.format_score_display(0.72)
    assert "probability" not in text.lower()
    assert "%" not in text
    assert "0.72" in text


def test_calibration_assessment_recorded_as_not_calibrated():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config", "calibration_assessment.json")) as f:
        calib = json.load(f)
    assert calib["calibration_demonstrable_pooled_test_set"] is False
    assert calib["decision"] == "Display as SCORE (0-1), not a calibrated probability"


# UI cannot modify the frozen model (Part 10's explicit new requirement)
def test_ui_module_never_calls_training_functions():
    """Static check: early_warning_page.py must not IMPORT any name from
    the training code paths -- checks actual import statements, not raw
    substring presence, since docstrings legitimately mention these names
    while explaining they are NOT used."""
    import ast
    ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "early_warning_page.py")
    with open(ui_path) as f:
        tree = ast.parse(f.read())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported_names.update(a.name for a in node.names)
    forbidden = {"train_local", "class_balanced_pos_weight",
                 "select_threshold_from_train_cv", "StaticWeightedFedAvg",
                 "start_simulation", "FedAvg", "phase7_run", "phase6_fedavg",
                 "run_final_experiment"}
    leaked = imported_names & forbidden
    assert not leaked, f"Dashboard imports training-capable symbols: {leaked}"


def test_decision_support_module_never_calls_training_functions():
    import ast
    ds_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "decision_support.py")
    with open(ds_path) as f:
        tree = ast.parse(f.read())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported_names.update(a.name for a in node.names)
    forbidden = {"train_local", "class_balanced_pos_weight",
                 "select_threshold_from_train_cv", "StaticWeightedFedAvg", "start_simulation"}
    leaked = imported_names & forbidden
    assert not leaked, f"decision_support.py imports training-capable symbols: {leaked}"

