"""
run_final_experiment.py — Phase 8: the single, clean, reproducible command
that runs every frozen method (centralized, FedAvg, Adaptive A, Adaptive B)
using exactly the frozen Phase 6/7 inputs (dataset, split, features, model,
weight formulas) and produces phase8_final_results.csv + phase8_summary.csv.

Run:
    python run_final_experiment.py

Nothing here is new: no new features, no new dataset, no new aggregation
formula, no new ML method. This script's only job is to run the frozen
Phase 6/7 methods across the same 3 seeds consistently (Phase 7 only did
this for FedAvg + Method A; this run adds Method B's missing multi-seed
pass) and consolidate everything into the two required output files.
"""
import json
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
    evaluate, class_balanced_pos_weight, select_threshold_from_train_cv,
)
from phase7_strategy import StaticWeightedFedAvg

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=None,
                help="Run only this single seed (checkpointed). Omit to run centralized only.")
args = ap.parse_args()

CHECKPOINT_PATH = "results/phase8_final_results.csv"
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

with open("results/phase6_fedavg_global_threshold.json") as f:
    global_threshold = json.load(f)["global_threshold"]

with open("results/phase7_weights_for_simulation.json") as f:
    adaptive_weights = json.load(f)

weight_source_audit = {
    "source_file": "results/phase7_weight_tables.csv",
    "computed_by": "phase7_weighting.py",
    "input_fields_used": ["train_samples", "train_positive"],
    "input_fields_NOT_used": ["val_samples", "val_positive", "test_samples",
                                "test_positive", "any PR_AUC/ROC_AUC/F1 metric"],
    "verified": "phase7_weighting.py reads only results/phase7_client_reliability.csv, "
                "whose own fields (train_samples, train_positive, train_negative, "
                "val_samples, val_positive) are computed from `train = df[df.split==train]` "
                "and `val = df[df.split==val]` -- test rows are never loaded in that script.",
}
with open("results/phase8_weight_freeze_audit.json", "w") as f:
    json.dump(weight_source_audit, f, indent=2)
print("=== Part 12: weight freeze audit ===")
print(json.dumps(weight_source_audit, indent=2))

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

    start_simulation(
        client_fn=client_fn,
        num_clients=len(CLIENT_LIST),
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        ray_init_args={"num_cpus": 1, "include_dashboard": False},
    )


def evaluate_final_model(final_params, label, seed):
    model = LogisticRegressionModel(len(FROZEN_FEATURES))
    set_model_params(model, final_params)
    rows = []
    for country in ALL_FL_CLIENTS:
        ctest = test[test["COUNTRYNAME"] == country]
        n_pos = int(ctest["target_t1"].sum())
        if len(ctest) == 0 or n_pos == 0:
            rows.append({"method": label, "country": country, "seed": seed,
                          "test_positive": n_pos, "status": "insufficient test data"})
            continue
        Xc = scale(ctest[FROZEN_FEATURES].values.astype(np.float32), mean, std)
        score = predict_proba(model, Xc)
        m = evaluate(ctest["target_t1"].values, score, threshold=global_threshold)
        rows.append({"method": label, "country": country, "seed": seed,
                      "test_positive": m["positives"],
                      "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1": m["f1"],
                      "precision": m["precision"], "recall": m["recall"],
                      "low_confidence": n_pos < 5, "status": "computed"})
    return pd.DataFrame(rows)


def run_vanilla_fedavg(seed):
    strategy = FedAvg(fraction_fit=1.0, fraction_evaluate=1.0,
                        min_fit_clients=len(CLIENT_LIST),
                        min_evaluate_clients=len(CLIENT_LIST),
                        min_available_clients=len(CLIENT_LIST))
    captured = {}
    orig_agg = strategy.aggregate_fit
    def wrapped(server_round, results, failures):
        params, metrics = orig_agg(server_round, results, failures)
        if params is not None:
            captured["params"] = parameters_to_ndarrays(params)
        return params, metrics
    strategy.aggregate_fit = wrapped
    run_federation(strategy, seed)
    return captured["params"]


def run_adaptive(method_key, seed):
    weights = adaptive_weights[method_key]
    strategy = StaticWeightedFedAvg(
        client_weights=weights, fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=len(CLIENT_LIST), min_evaluate_clients=len(CLIENT_LIST),
        min_available_clients=len(CLIENT_LIST),
    )
    run_federation(strategy, seed)
    return strategy.captured_final_params["params"]


print("\n=== Part 5: centralized baseline reproduction ===")
threshold, _ = select_threshold_from_train_cv(
    scale(X_train_all, mean, std), train["target_t1"].values.astype(np.float32),
    train["year_T"].values,
)
central_model = LogisticRegressionModel(len(FROZEN_FEATURES))
pw = class_balanced_pos_weight(train["target_t1"].values.astype(np.float32))
train_local(central_model, scale(X_train_all, mean, std),
            train["target_t1"].values.astype(np.float32), epochs=200, pos_weight=pw)

central_rows = []
for country in ["Kenya", "Uganda", "Afghanistan"]:
    ctest = test[test["COUNTRYNAME"] == country]
    Xc = scale(ctest[FROZEN_FEATURES].values.astype(np.float32), mean, std)
    score = predict_proba(central_model, Xc)
    m = evaluate(ctest["target_t1"].values, score, threshold=threshold)
    central_rows.append({"method": "centralized", "country": country, "seed": "n/a",
                          "test_positive": m["positives"], "pr_auc": m["pr_auc"],
                          "roc_auc": m["roc_auc"], "f1": m["f1"],
                          "precision": m["precision"], "recall": m["recall"],
                          "low_confidence": m["positives"] < 5, "status": "computed"})
central_df = pd.DataFrame(central_rows)
print(central_df.to_string(index=False))
ref_kenya, ref_uganda = 0.418, 0.938
k_roc = central_df[central_df.country == "Kenya"]["roc_auc"].iloc[0]
u_roc = central_df[central_df.country == "Uganda"]["roc_auc"].iloc[0]
print(f"vs Phase 6 reference: Kenya {k_roc:.3f} (ref {ref_kenya}), "
      f"Uganda {u_roc:.3f} (ref {ref_uganda})")
if abs(k_roc - ref_kenya) > 0.10 or abs(u_roc - ref_uganda) > 0.10:
    raise SystemExit("Centralized reproduction differs materially. STOP.")
print("Centralized reproduction: within tolerance.")

# ---------------------------------------------------------------------
# Checkpointed execution: write centralized row immediately, then run
# only the requested seed (if any), appending to the same CSV so a
# time-limited tool call can be repeated per seed without losing
# earlier progress or re-running completed work.
# ---------------------------------------------------------------------
if not os.path.exists(CHECKPOINT_PATH):
    central_df.to_csv(CHECKPOINT_PATH, index=False)
    print(f"Initialized {CHECKPOINT_PATH} with centralized row.")
else:
    print(f"{CHECKPOINT_PATH} already exists -- centralized row not re-appended.")

if args.seed is None:
    print("\nNo --seed given; centralized-only run complete. "
          "Call again with --seed 42 / 123 / 2024 to add each seed's FedAvg/Adaptive-A/Adaptive-B rows.")
    raise SystemExit(0)

existing = pd.read_csv(CHECKPOINT_PATH)
already_done = set(
    existing.loc[existing["seed"].astype(str) == str(args.seed), "method"].unique()
)
print(f"\nSeed {args.seed}: methods already checkpointed = {already_done}")

seed_rows = []
if "fedavg" not in already_done:
    print(f"--- seed {args.seed}: vanilla FedAvg ---")
    vp = run_vanilla_fedavg(args.seed)
    seed_rows.append(evaluate_final_model(vp, "fedavg", args.seed))
else:
    print(f"--- seed {args.seed}: vanilla FedAvg already done, skipping ---")

if "adaptive_A" not in already_done:
    print(f"--- seed {args.seed}: Adaptive A ---")
    ap_ = run_adaptive("adaptive_A", args.seed)
    seed_rows.append(evaluate_final_model(ap_, "adaptive_A", args.seed))
else:
    print(f"--- seed {args.seed}: Adaptive A already done, skipping ---")

if "adaptive_B" not in already_done:
    print(f"--- seed {args.seed}: Adaptive B ---")
    bp = run_adaptive("adaptive_B", args.seed)
    seed_rows.append(evaluate_final_model(bp, "adaptive_B", args.seed))
else:
    print(f"--- seed {args.seed}: Adaptive B already done, skipping ---")

if seed_rows:
    updated = pd.concat([existing] + seed_rows, ignore_index=True)
    updated.to_csv(CHECKPOINT_PATH, index=False)
    print(f"Checkpointed seed {args.seed}. Total rows now: {len(updated)}")
else:
    print(f"Seed {args.seed}: nothing new to run.")

final_results = pd.read_csv(CHECKPOINT_PATH)
n_seeds_done = final_results.loc[final_results["method"] == "fedavg", "seed"].nunique()
print(f"\nSeeds fully represented so far: {n_seeds_done} / 3")

if n_seeds_done < 3:
    print("Not all seeds complete yet -- summary not (re)generated this call. "
          "Re-run with the next --seed value.")
    raise SystemExit(0)

computed = final_results[final_results.status == "computed"].copy()
summary = computed.groupby(["method", "country"]).agg(
    mean_pr_auc=("pr_auc", "mean"), std_pr_auc=("pr_auc", "std"),
    mean_roc_auc=("roc_auc", "mean"), std_roc_auc=("roc_auc", "std"),
    mean_f1=("f1", "mean"), std_f1=("f1", "std"),
    n_runs=("country", "count"),
).reset_index()
summary.to_csv("results/phase8_summary.csv", index=False)
print("\n=== Part 19: summary ===")
print(summary.to_string(index=False))

print("\nrun_final_experiment.py complete.")
