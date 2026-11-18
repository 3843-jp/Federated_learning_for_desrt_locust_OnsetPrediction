"""
tests/test_independent_phase8_reproduction.py — the strengthened,
genuinely independent reproducibility test.

WHAT THIS DOES DIFFERENTLY FROM THE EXISTING
test_prediction_matches_phase8_reference():

That test reads config/frozen_model_metadata.json's recorded
"verification_against_phase7_8" numbers -- i.e. it trusts a value that
was computed and written to disk once, back when save_frozen_model.py
first ran. It never re-executes training.

THIS test never opens frozen_model_metadata.json at all. It:
  1. Re-runs the actual frozen Phase 7/8 training procedure from scratch,
     in-process, right now -- same client class, same 21 clients, same
     StaticWeightedFedAvg strategy, same adaptive_A weight dict (loaded
     from phase7_weights_for_simulation.json, which is itself a
     training-statistics-only artifact, unchanged since Phase 7), same
     seed=42, same 15 rounds, same 5 local epochs, same lr=0.05.
  2. Gets a FRESH set of model parameters out of that run -- never touches
     config/frozen_method_a_model.npz during training.
  3. Separately loads the CURRENT frozen artifact
     (config/frozen_method_a_model.npz) exactly as decision_support.py
     does for real inference.
  4. Computes predictions from BOTH parameter sets on the same real test
     rows (all countries with test-period positives).
  5. Compares the two raw prediction arrays element-wise with an explicit,
     documented tolerance.
  6. Fails if any prediction differs by more than that tolerance.

WHAT "PHASE 8 ARTIFACT" ACTUALLY EXISTS (investigated before writing this
test, not assumed): Phase 8 (run_final_experiment.py) saved only
AGGREGATED METRICS per (method, country, seed) --
results/phase8_final_results.csv and phase8_summary.csv. It never saved
raw model weights or raw per-row prediction scores to disk. There is no
"Phase 8 model file" separate from re-running the training procedure
itself. The reference model in this test is therefore necessarily
produced by RE-EXECUTING the same deterministic, seeded, frozen
procedure -- which is the correct and only way to get an independent
Phase 8 reference; using the phase8_final_results.csv metrics directly
would not be a "raw prediction" comparison as required, only another
aggregated-metric comparison like the one being strengthened here.

TOLERANCE: 1e-4 absolute difference per prediction. Chosen before running
this test, not after seeing the result: PyTorch CPU training with a fixed
seed and full-batch (non-shuffled, non-multiprocessed-within-client)
gradient descent is bit-reproducible in principle; Flower's Ray-based
simulation orchestrates clients across process boundaries, which can
introduce floating-point summation-order differences during aggregation
(summing N clients' weighted arrays in a different order can differ in
the last few bits under IEEE 754). 1e-4 is far tighter than anything that
would matter for this model's actual predictions (which span 0-1 and are
bucketed into 3 risk tiers with 0.30/0.30 gaps) while still being loose
enough to tolerate legitimate floating-point summation-order noise, not
a real discrepancy.
"""
import json
import os
import sys

import numpy as np
import pytest
import torch
import flwr as fl
from flwr.simulation import start_simulation

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "frozen_inputs"))

import decision_support as ds  # only for loading the CURRENT frozen artifact + FROZEN_FEATURES
from phase6_data_prep import load_frozen_dataset, FROZEN_FEATURES
from phase6_model import (
    LogisticRegressionModel, get_scaler, scale, train_local, predict_proba,
    class_balanced_pos_weight,
)
from phase7_strategy import StaticWeightedFedAvg

TOLERANCE = 1e-4
SEED = 42
NUM_ROUNDS = 15
LOCAL_EPOCHS = 5
LOCAL_LR = 0.05


def _regenerate_reference_params():
    """Independently re-runs the exact frozen Phase 7/8 training
    procedure and returns a FRESH set of model parameters. Does not read,
    import, or reference frozen_model_metadata.json anywhere.

    IMPORTANT: cwd must remain frozen_inputs/ for the ENTIRE federation
    run, including start_simulation() -- Ray's worker actors resolve
    'phase6_model'/'phase7_strategy' relative to the driver's working
    directory at simulation-start time, not the driver's sys.path alone.
    Restoring cwd before start_simulation() (an earlier version of this
    test did exactly that) causes every Ray actor to fail with
    ModuleNotFoundError, since the actors can't find frozen_inputs/ on
    their own import path. Caught via the test's own failure output,
    not assumed correct in advance.
    """
    _cwd = os.getcwd()
    os.chdir(os.path.join(_ROOT, "frozen_inputs"))
    try:
        df = load_frozen_dataset()
        with open("phase6_client_categories.json") as f:
            cats = json.load(f)
        client_list = cats["category_a"] + cats["category_b"]

        train = df[df["split"] == "train"].reset_index(drop=True)
        X_train_all = train[FROZEN_FEATURES].values.astype(np.float32)
        mean, std = get_scaler(X_train_all)

        with open("phase7_weights_for_simulation.json") as f:
            adaptive_weights = json.load(f)["adaptive_A"]

        client_data = {}
        for country in client_list:
            ctr = train[train["COUNTRYNAME"] == country]
            Xc = scale(ctr[FROZEN_FEATURES].values.astype(np.float32), mean, std)
            yc = ctr["target_t1"].values.astype(np.float32)
            client_data[country] = (Xc, yc)

        def set_model_params(model, params):
            with torch.no_grad():
                for p, new in zip(model.parameters(), params):
                    p.copy_(torch.tensor(new, dtype=torch.float32))

        def get_model_params(model):
            return [p.detach().numpy().copy() for p in model.parameters()]

        class ReferenceClient(fl.client.NumPyClient):
            def __init__(self, country):
                self.country = country
                self.X, self.y = client_data[country]
                torch.manual_seed(SEED)
                self.model = LogisticRegressionModel(len(FROZEN_FEATURES))

            def get_parameters(self, config):
                return get_model_params(self.model)

            def fit(self, parameters, config):
                set_model_params(self.model, parameters)
                pw = class_balanced_pos_weight(self.y) if self.y.sum() > 0 else None
                train_local(self.model, self.X, self.y, epochs=LOCAL_EPOCHS,
                            lr=LOCAL_LR, pos_weight=pw)
                return (get_model_params(self.model), len(self.X),
                        {"country": self.country, "train_loss": 0.0,
                         "train_positives": int(self.y.sum())})

            def evaluate(self, parameters, config):
                set_model_params(self.model, parameters)
                return 0.0, len(self.X), {"country": self.country}

        def client_fn(cid):
            return ReferenceClient(client_list[int(cid)]).to_client()

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        strategy = StaticWeightedFedAvg(
            client_weights=adaptive_weights, fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=len(client_list), min_evaluate_clients=len(client_list),
            min_available_clients=len(client_list),
        )
        # start_simulation() called while cwd is STILL frozen_inputs/ --
        # this is the fix; see docstring above.
        start_simulation(
            client_fn=client_fn, num_clients=len(client_list),
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS), strategy=strategy,
            ray_init_args={"num_cpus": 1, "include_dashboard": False},
        )
        result_params = strategy.captured_final_params["params"]
    finally:
        os.chdir(_cwd)

    return result_params, mean, std


@pytest.mark.slow
def test_independent_phase8_reproduction():
    # --- Step 1-3: independently re-run training, get fresh reference params ---
    reference_params, ref_mean, ref_std = _regenerate_reference_params()

    # --- Step 4: load the CURRENT frozen artifact separately ---
    current_model, current_mean, current_std = ds.load_frozen_model()

    # scaler must match too -- both derived from the same frozen train
    # split, so should be identical; checked explicitly rather than assumed
    assert np.allclose(ref_mean, current_mean, atol=1e-6), "Scaler mean mismatch between reference and frozen artifact"
    assert np.allclose(ref_std, current_std, atol=1e-6), "Scaler std mismatch between reference and frozen artifact"

    # --- Step 2/5: same test inputs for both ---
    _cwd = os.getcwd()
    os.chdir(os.path.join(_ROOT, "frozen_inputs"))
    df = load_frozen_dataset()
    os.chdir(_cwd)
    test = df[df["split"] == "test"]
    test = test[test["target_t1"].notna()]
    X_test = scale(test[FROZEN_FEATURES].values.astype(np.float32), current_mean, current_std)

    # --- reference predictions (freshly retrained model) ---
    reference_model = LogisticRegressionModel(len(FROZEN_FEATURES))
    with torch.no_grad():
        for p, new in zip(reference_model.parameters(), reference_params):
            p.copy_(torch.tensor(new, dtype=torch.float32))
    reference_model.eval()
    reference_scores = predict_proba(reference_model, X_test)

    # --- current frozen artifact predictions ---
    current_scores = predict_proba(current_model, X_test)

    # --- Step 6/7: explicit tolerance comparison ---
    max_abs_diff = float(np.max(np.abs(reference_scores - current_scores)))
    mean_abs_diff = float(np.mean(np.abs(reference_scores - current_scores)))
    n_exceeding = int(np.sum(np.abs(reference_scores - current_scores) > TOLERANCE))

    print(f"\nIndependent Phase 8 reproduction check:")
    print(f"  n_test_predictions_compared = {len(reference_scores)}")
    print(f"  max_abs_diff = {max_abs_diff}")
    print(f"  mean_abs_diff = {mean_abs_diff}")
    print(f"  tolerance = {TOLERANCE}")
    print(f"  n_predictions_exceeding_tolerance = {n_exceeding}")

    assert max_abs_diff <= TOLERANCE, (
        f"Independently reproduced Phase 8 predictions differ from the current frozen "
        f"Method A artifact by up to {max_abs_diff}, exceeding tolerance {TOLERANCE} "
        f"on {n_exceeding} of {len(reference_scores)} test predictions."
    )
