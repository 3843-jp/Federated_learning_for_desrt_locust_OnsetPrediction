"""
phase6_fedavg.py — Part 6 (vanilla FedAvg via Flower) and Part 7
(evaluate the global FedAvg model per client).

Standard, data-volume-weighted FedAvg: Flower's built-in FedAvg strategy
weights client updates by number of examples by default -- this matches
"use data-volume-weighted FedAvg as the standard baseline" directly,
rather than a custom reimplementation.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
from flwr.common import parameters_to_ndarrays

from phase6_data_prep import load_frozen_dataset, FROZEN_FEATURES
from phase6_model import (
    LogisticRegressionModel, get_scaler, scale, train_local, predict_proba,
    evaluate, class_balanced_pos_weight, SEED,
)

NUM_ROUNDS = 15
LOCAL_EPOCHS = 5
LOCAL_LR = 0.05

df = load_frozen_dataset()
with open("results/phase6_client_categories.json") as f:
    cats = json.load(f)
CATEGORY_A = cats["category_a"]
CATEGORY_B = cats["category_b"]
ALL_FL_CLIENTS = CATEGORY_A + CATEGORY_B
CLIENT_LIST = ALL_FL_CLIENTS

train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)

X_train_all = train[FROZEN_FEATURES].values.astype(np.float32)
mean, std = get_scaler(X_train_all)

with open("results/phase6_centralized_reproduction_check.json") as f:
    global_threshold = json.load(f)["phase6_reproduction"]["threshold"]

client_data = {}
for country in ALL_FL_CLIENTS:
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
        X_t = torch.tensor(self.X, dtype=torch.float32)
        y_t = torch.tensor(self.y, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            loss = nn.BCEWithLogitsLoss()(self.model(X_t), y_t).item()
        return loss, len(self.X), {"country": self.country}


def client_fn(cid):
    return LocustClient(CLIENT_LIST[int(cid)]).to_client()


round_log = []
captured_final_params = {}


class LoggingFedAvg(FedAvg):
    """FedAvg subclass that logs per-round client metrics and captures
    the final aggregated global parameters (Flower's History object
    doesn't retain these, so a subclass hook is the correct way to get
    them, not a workaround)."""

    def aggregate_fit(self, server_round, results, failures):
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)
        entry = []
        for _, fit_res in results:
            m = fit_res.metrics
            entry.append({"country": m["country"], "num_examples": fit_res.num_examples,
                           "train_loss": m["train_loss"], "train_positives": m["train_positives"]})
        round_log.append({"round": server_round, "clients": entry,
                            "n_participating": len(results), "n_failures": len(failures)})
        if agg_params is not None:
            captured_final_params["params"] = parameters_to_ndarrays(agg_params)
            captured_final_params["round"] = server_round
        return agg_params, agg_metrics


strategy = LoggingFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=len(CLIENT_LIST),
    min_evaluate_clients=len(CLIENT_LIST),
    min_available_clients=len(CLIENT_LIST),
)

print(f"Starting FedAvg simulation: {len(CLIENT_LIST)} clients, {NUM_ROUNDS} rounds, "
      f"{LOCAL_EPOCHS} local epochs/round.")
print(f"Clients ({len(CLIENT_LIST)}): {CLIENT_LIST}")

history = start_simulation(
    client_fn=client_fn,
    num_clients=len(CLIENT_LIST),
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    ray_init_args={"num_cpus": 1, "include_dashboard": False},
)

print("\n=== FedAvg training complete ===")
print("Distributed eval loss per round:", history.losses_distributed)

with open("results/phase6_fedavg_round_log.json", "w") as f:
    json.dump({"loss_distributed": history.losses_distributed,
                "per_round_log": round_log,
                "num_rounds": NUM_ROUNDS, "local_epochs": LOCAL_EPOCHS,
                "n_clients": len(CLIENT_LIST), "clients": CLIENT_LIST},
               f, indent=2, default=str)

# ---------------------------------------------------------------------
# Part 7: evaluate the final global FedAvg model per client (TEST)
# ---------------------------------------------------------------------
assert "params" in captured_final_params, "Failed to capture final FedAvg parameters."
final_model = LogisticRegressionModel(len(FROZEN_FEATURES))
set_model_params(final_model, captured_final_params["params"])
print(f"\nCaptured final global parameters from round {captured_final_params['round']}.")

fedavg_rows = []
for country in ALL_FL_CLIENTS:
    ctest = test[test["COUNTRYNAME"] == country]
    if len(ctest) == 0 or ctest["target_t1"].sum() == 0:
        fedavg_rows.append({"method": "fedavg", "country": country,
                              "test_samples": len(ctest),
                              "test_positive": int(ctest["target_t1"].sum()),
                              "status": "insufficient test data"})
        continue
    Xc = scale(ctest[FROZEN_FEATURES].values.astype(np.float32), mean, std)
    score = predict_proba(final_model, Xc)
    m = evaluate(ctest["target_t1"].values, score, threshold=global_threshold)
    fedavg_rows.append({"method": "fedavg", "country": country,
                          "test_samples": len(ctest), "test_positive": m["positives"],
                          "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                          "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                          "threshold": global_threshold, "status": "computed"})

fedavg_df = pd.DataFrame(fedavg_rows)
fedavg_df.to_csv("results/phase6_fedavg_per_client.csv", index=False)
print("\n=== Part 7: FedAvg, per client ===")
print(fedavg_df.to_string(index=False))

# Save final model parameters for the personalization phase (Part 8)
np.savez("results/phase6_fedavg_final_params.npz",
          *captured_final_params["params"])
with open("results/phase6_fedavg_global_threshold.json", "w") as f:
    json.dump({"global_threshold": global_threshold}, f)

print("\nDone with phase6_fedavg.py")
