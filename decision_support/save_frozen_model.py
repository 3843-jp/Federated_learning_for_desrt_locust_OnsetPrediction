"""
save_frozen_model.py — materializes the Phase 8 Method A (seed 42) model
as a persistent artifact, since Phase 7/8 captured its parameters only
in-memory during the federation run and never wrote them to disk.

This is NOT a new experiment: it runs the exact same frozen procedure
(same 21 clients, same 15 rounds, same 5 local epochs, same lr, same
weight formula, same seed=42) that Phase 7/8 already validated, and
verifies its output against the Phase 8 seed-42 reference numbers before
anything downstream is allowed to use it. If reproduction fails, this
script stops rather than silently shipping a different model under the
same name.
"""
import json
import numpy as np
import pandas as pd
import torch
import flwr as fl
from flwr.simulation import start_simulation

import sys
sys.path.insert(0, "frozen_inputs")
from phase6_data_prep import load_frozen_dataset, FROZEN_FEATURES
from phase6_model import (
    LogisticRegressionModel, get_scaler, scale, train_local, predict_proba,
    evaluate, class_balanced_pos_weight,
)
from phase7_strategy import StaticWeightedFedAvg

SEED = 42
NUM_ROUNDS = 15
LOCAL_EPOCHS = 5
LOCAL_LR = 0.05

import os
os.chdir("frozen_inputs")
df = load_frozen_dataset()
with open("phase6_client_categories.json") as f:
    cats = json.load(f)
CLIENT_LIST = cats["category_a"] + cats["category_b"]

train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)

X_train_all = train[FROZEN_FEATURES].values.astype(np.float32)
mean, std = get_scaler(X_train_all)

with open("phase6_fedavg_global_threshold.json") as f:
    global_threshold = json.load(f)["global_threshold"]
with open("phase7_weights_for_simulation.json") as f:
    adaptive_weights = json.load(f)["adaptive_A"]

client_data = {}
for country in CLIENT_LIST:
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


class LocustClient(fl.client.NumPyClient):
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
        _, loss = train_local(self.model, self.X, self.y, epochs=LOCAL_EPOCHS,
                                lr=LOCAL_LR, pos_weight=pw)
        return (get_model_params(self.model), len(self.X),
                {"country": self.country, "train_loss": float(loss),
                 "train_positives": int(self.y.sum())})

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        import torch.nn as nn
        X_t = torch.tensor(self.X, dtype=torch.float32)
        y_t = torch.tensor(self.y, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            loss = nn.BCEWithLogitsLoss()(self.model(X_t), y_t).item()
        return loss, len(self.X), {"country": self.country}


def client_fn(cid):
    return LocustClient(CLIENT_LIST[int(cid)]).to_client()


torch.manual_seed(SEED)
np.random.seed(SEED)
strategy = StaticWeightedFedAvg(
    client_weights=adaptive_weights, fraction_fit=1.0, fraction_evaluate=1.0,
    min_fit_clients=len(CLIENT_LIST), min_evaluate_clients=len(CLIENT_LIST),
    min_available_clients=len(CLIENT_LIST),
)

start_simulation(
    client_fn=client_fn, num_clients=len(CLIENT_LIST),
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS), strategy=strategy,
    ray_init_args={"num_cpus": 1, "include_dashboard": False},
)

final_params = strategy.captured_final_params["params"]

# --- Verify against Phase 8 seed-42 reference before saving anything ---
model = LogisticRegressionModel(len(FROZEN_FEATURES))
set_model_params(model, final_params)

ref = {"Kenya": 0.539, "Uganda": 0.740}  # Phase 7 single-seed-42 reference
verification = {}
ok = True
for country, ref_roc in ref.items():
    ctest = test[test["COUNTRYNAME"] == country]
    Xc = scale(ctest[FROZEN_FEATURES].values.astype(np.float32), mean, std)
    score = predict_proba(model, Xc)
    m = evaluate(ctest["target_t1"].values, score, threshold=global_threshold)
    diff = abs(m["roc_auc"] - ref_roc)
    verification[country] = {"reproduced_roc_auc": m["roc_auc"], "reference": ref_roc, "diff": diff}
    if diff > 0.02:
        ok = False

print(json.dumps(verification, indent=2))
if not ok:
    raise SystemExit("Frozen model reproduction check FAILED -- differs from Phase 7/8 seed-42 "
                      "reference by more than tolerance. Not saving. Stop and investigate.")
print("Frozen model reproduction verified within tolerance.")

os.chdir("..")
np.savez("config/frozen_method_a_model.npz", *final_params)
np.savez("config/frozen_scaler.npz", mean=mean, std=std)
with open("config/frozen_model_metadata.json", "w") as f:
    json.dump({
        "method": "Adaptive Aggregation Method A (positive-aware)",
        "seed": SEED, "num_rounds": NUM_ROUNDS, "local_epochs": LOCAL_EPOCHS,
        "features_in_order": FROZEN_FEATURES,
        "global_threshold_F1_optimized": global_threshold,
        "verification_against_phase7_8": verification,
        "source": "Phase 8, validated multi-seed; this artifact reproduces the seed=42 run "
                   "for persistent deployment -- not a new experiment.",
    }, f, indent=2)
print("Saved: config/frozen_method_a_model.npz, config/frozen_scaler.npz, config/frozen_model_metadata.json")
