"""
phase7_run.py — Parts 2, 4, 9-16: reproduce Phase 6 FedAvg, run Adaptive
A and B (single seed first), evaluate all clients, then multi-seed
stability check (Part 17) for FedAvg + the better adaptive method.

Usage:
    python phase7_run.py single      # Parts 2,4,9 -- vanilla+A+B, seed 42
    python phase7_run.py multiseed X # Part 17 -- rerun vanilla + method X
                                       # (A or B) across 3 seeds
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
from flwr.common import parameters_to_ndarrays

from phase6_data_prep import load_frozen_dataset, FROZEN_FEATURES
from phase6_model import (
    LogisticRegressionModel, get_scaler, scale, train_local, predict_proba,
    evaluate, class_balanced_pos_weight,
)
from phase7_strategy import StaticWeightedFedAvg

NUM_ROUNDS = 15
LOCAL_EPOCHS = 5
LOCAL_LR = 0.05

df = load_frozen_dataset()
with open("results/phase6_client_categories.json") as f:
    cats = json.load(f)
CATEGORY_A = cats["category_a"]
CATEGORY_B = cats["category_b"]
ALL_FL_CLIENTS = CATEGORY_A + CATEGORY_B

train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)

X_train_all = train[FROZEN_FEATURES].values.astype(np.float32)
mean, std = get_scaler(X_train_all)

with open("results/phase6_fedavg_global_threshold.json") as f:
    global_threshold = json.load(f)["global_threshold"]

with open("results/phase7_weights_for_simulation.json") as f:
    adaptive_weights = json.load(f)

client_data = {}
for country in ALL_FL_CLIENTS:
    ctr = train[train["COUNTRYNAME"] == country]
    Xc = scale(ctr[FROZEN_FEATURES].values.astype(np.float32), mean, std)
    yc = ctr["target_t1"].values.astype(np.float32)
    client_data[country] = (Xc, yc)

CLIENT_LIST = ALL_FL_CLIENTS


def set_model_params(model, params):
    with torch.no_grad():
        for p, new in zip(model.parameters(), params):
            p.copy_(torch.tensor(new, dtype=torch.float32))


def get_model_params(model):
    return [p.detach().numpy().copy() for p in model.parameters()]


def make_client_class(seed):
    class LocustClient(fl.client.NumPyClient):
        def __init__(self, country):
            self.country = country
            self.X, self.y = client_data[country]
            torch.manual_seed(seed)
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
    return LocustClient


def run_federation(strategy, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ClientCls = make_client_class(seed)

    def client_fn(cid):
        return ClientCls(CLIENT_LIST[int(cid)]).to_client()

    history = start_simulation(
        client_fn=client_fn,
        num_clients=len(CLIENT_LIST),
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        ray_init_args={"num_cpus": 1, "include_dashboard": False},
    )
    return history


def evaluate_final_model(final_params, label):
    model = LogisticRegressionModel(len(FROZEN_FEATURES))
    set_model_params(model, final_params)
    rows = []
    for country in ALL_FL_CLIENTS:
        ctest = test[test["COUNTRYNAME"] == country]
        n_pos = int(ctest["target_t1"].sum())
        if len(ctest) == 0 or n_pos == 0:
            rows.append({"method": label, "country": country,
                          "test_samples": len(ctest), "test_positive": n_pos,
                          "status": "insufficient test data"})
            continue
        Xc = scale(ctest[FROZEN_FEATURES].values.astype(np.float32), mean, std)
        score = predict_proba(model, Xc)
        m = evaluate(ctest["target_t1"].values, score, threshold=global_threshold)
        rows.append({"method": label, "country": country,
                      "test_samples": len(ctest), "test_positive": m["positives"],
                      "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                      "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                      "low_confidence": n_pos < 5, "status": "computed"})
    return pd.DataFrame(rows)


def run_vanilla_fedavg(seed):
    strategy = FedAvg(fraction_fit=1.0, fraction_evaluate=1.0,
                        min_fit_clients=len(CLIENT_LIST),
                        min_evaluate_clients=len(CLIENT_LIST),
                        min_available_clients=len(CLIENT_LIST))
    # need parameter capture -> subclass on the fly
    captured = {}
    orig_agg = strategy.aggregate_fit
    def wrapped(server_round, results, failures):
        params, metrics = orig_agg(server_round, results, failures)
        if params is not None:
            captured["params"] = parameters_to_ndarrays(params)
            captured["round"] = server_round
        return params, metrics
    strategy.aggregate_fit = wrapped
    run_federation(strategy, seed)
    return captured["params"]


def run_adaptive(method_key, seed):
    weights = adaptive_weights[method_key]
    strategy = StaticWeightedFedAvg(
        client_weights=weights,
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=len(CLIENT_LIST),
        min_evaluate_clients=len(CLIENT_LIST),
        min_available_clients=len(CLIENT_LIST),
    )
    run_federation(strategy, seed)
    return strategy.captured_final_params["params"], strategy.round_log


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"

    if mode == "single":
        print("=== Reproducing Phase 6 vanilla FedAvg (Part 2) ===")
        vanilla_params = run_vanilla_fedavg(seed=42)
        vanilla_eval = evaluate_final_model(vanilla_params, "fedavg_vanilla_repro")
        vanilla_eval.to_csv("results/phase7_fedavg_reproduction.csv", index=False)
        print(vanilla_eval.to_string(index=False))

        kenya_row = vanilla_eval[vanilla_eval.country == "Kenya"]
        uganda_row = vanilla_eval[vanilla_eval.country == "Uganda"]
        kenya_roc = float(kenya_row["roc_auc"].iloc[0]) if len(kenya_row) else None
        uganda_roc = float(uganda_row["roc_auc"].iloc[0]) if len(uganda_row) else None
        print(f"\nReproduction check: Kenya ROC-AUC={kenya_roc} (ref ~0.442), "
              f"Uganda ROC-AUC={uganda_roc} (ref ~0.696)")
        kenya_diff = abs(kenya_roc - 0.442) if kenya_roc else None
        uganda_diff = abs(uganda_roc - 0.696) if uganda_roc else None
        material = (kenya_diff and kenya_diff > 0.10) or (uganda_diff and uganda_diff > 0.10)
        print(f"Kenya diff={kenya_diff}, Uganda diff={uganda_diff} -> "
              f"{'MATERIAL -- STOP' if material else 'within tolerance, proceeding'}")
        if material:
            raise SystemExit("FedAvg reproduction differs materially from Phase 6. Stopping.")

        print("\n=== Adaptive Method A (positive-aware), seed 42 ===")
        params_A, log_A = run_adaptive("adaptive_A", seed=42)
        eval_A = evaluate_final_model(params_A, "adaptive_A")
        eval_A.to_csv("results/phase7_adaptive_A_results.csv", index=False)
        with open("results/phase7_adaptive_A_round_log.json", "w") as f:
            json.dump(log_A, f, indent=2, default=str)
        print(eval_A.to_string(index=False))

        print("\n=== Adaptive Method B (reliability-aware), seed 42 ===")
        params_B, log_B = run_adaptive("adaptive_B", seed=42)
        eval_B = evaluate_final_model(params_B, "adaptive_B")
        eval_B.to_csv("results/phase7_adaptive_B_results.csv", index=False)
        with open("results/phase7_adaptive_B_round_log.json", "w") as f:
            json.dump(log_B, f, indent=2, default=str)
        print(eval_B.to_string(index=False))

        print("\nAll single-seed runs complete.")

    elif mode == "multiseed":
        method = sys.argv[2]  # "A" or "B"
        seeds = [42, 123, 2024]
        all_rows = []
        for seed in seeds:
            print(f"\n--- Vanilla FedAvg, seed {seed} ---")
            vp = run_vanilla_fedavg(seed=seed)
            ve = evaluate_final_model(vp, "fedavg_vanilla")
            ve["seed"] = seed
            all_rows.append(ve)

            print(f"--- Adaptive {method}, seed {seed} ---")
            ap, _ = run_adaptive(f"adaptive_{method}", seed=seed)
            ae = evaluate_final_model(ap, f"adaptive_{method}")
            ae["seed"] = seed
            all_rows.append(ae)

        multiseed_df = pd.concat(all_rows, ignore_index=True)
        multiseed_df.to_csv("results/phase7_multiseed_results.csv", index=False)
        print("\n=== Multi-seed results ===")
        print(multiseed_df.to_string(index=False))

        computed = multiseed_df[multiseed_df.status == "computed"]
        summary = computed.groupby(["method", "country"]).agg(
            mean_pr_auc=("pr_auc", "mean"), std_pr_auc=("pr_auc", "std"),
            mean_roc_auc=("roc_auc", "mean"), std_roc_auc=("roc_auc", "std"),
            n_seeds=("seed", "count"),
        ).reset_index()
        summary.to_csv("results/phase7_multiseed_summary.csv", index=False)
        print("\n=== Multi-seed summary ===")
        print(summary.to_string(index=False))
