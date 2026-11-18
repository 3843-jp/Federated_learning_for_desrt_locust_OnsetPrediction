"""
phase6_personalized.py — Part 8 (personalized FL) and Part 9 (comparison).

Personalization method chosen: FedAvg global model + client-specific local
fine-tuning (Option A from the instructions). Justification, since the
instructions require the method to be justified against the measured
heterogeneity, not defaulted to: our shared architecture is a plain
logistic regression with no BatchNorm layers, so FedBN (which personalizes
BatchNorm statistics) does not apply -- there is nothing for it to
personalize. Fine-tuning is the simplest method that is technically
compatible with this architecture and directly targets what Phase 5 found:
country-specific, sometimes sign-flipped coefficients on the same
features. Fine-tuning every weight locally, starting from the FedAvg
global model, is exactly the right lever for that kind of heterogeneity
(a client whose true coefficient sign differs from the global average
needs to be able to move that specific weight, which fine-tuning permits
and FedBN's normalization-only personalization would not address at all).
"""
import json
import numpy as np
import pandas as pd
import torch

from phase6_data_prep import load_frozen_dataset, FROZEN_FEATURES
from phase6_model import (
    LogisticRegressionModel, get_scaler, scale, train_local, predict_proba,
    evaluate, class_balanced_pos_weight, select_threshold_from_train_cv, SEED,
)

FINE_TUNE_EPOCHS = 100
FINE_TUNE_LR = 0.02  # smaller than local-only training, deliberately, since
                       # this is refinement from a reasonable starting point,
                       # not training from scratch

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

fedavg_params_npz = np.load("results/phase6_fedavg_final_params.npz")
fedavg_params = [fedavg_params_npz[k] for k in fedavg_params_npz.files]

personalized_rows = []
for country in ALL_FL_CLIENTS:
    ctr = train[train["COUNTRYNAME"] == country]
    cte = test[test["COUNTRYNAME"] == country]
    n_train_pos = int(ctr["target_t1"].sum())

    row = {"method": "personalized_fl", "country": country,
           "train_samples": len(ctr), "train_positive": n_train_pos,
           "test_samples": len(cte), "test_positive": int(cte["target_t1"].sum())}

    if len(cte) == 0 or cte["target_t1"].sum() == 0:
        row["status"] = "insufficient test data"
        personalized_rows.append(row)
        continue

    # Start from the FedAvg global model, fine-tune on this client's own
    # training data (even zero-positive clients like Uganda/Afghanistan
    # CAN be fine-tuned here, unlike local-only -- fine-tuning from a
    # reasonable starting point doesn't require local positives to move
    # away from a random initialization, only to adjust an already-useful
    # model, so this is expected to behave very differently from local-only
    # for exactly these two clients).
    personal_model = LogisticRegressionModel(len(FROZEN_FEATURES))
    with torch.no_grad():
        for p, new in zip(personal_model.parameters(), fedavg_params):
            p.copy_(torch.tensor(new, dtype=torch.float32))

    Xc_tr = scale(ctr[FROZEN_FEATURES].values.astype(np.float32), mean, std)
    yc_tr = ctr["target_t1"].values.astype(np.float32)

    if n_train_pos > 0:
        pw = class_balanced_pos_weight(yc_tr)
        train_local(personal_model, Xc_tr, yc_tr, epochs=FINE_TUNE_EPOCHS,
                    lr=FINE_TUNE_LR, pos_weight=pw)
        fine_tuned = True
    else:
        # Zero positives locally: fine-tuning on negatives only would push
        # the model AWAY from predicting positive at all, which would be
        # actively harmful, not neutral. Documented, not silently done --
        # for these clients "personalized" = the FedAvg global model
        # unchanged, since no safe local update direction exists.
        fine_tuned = False
        row["note"] = "zero local training positives -- fine-tuning skipped " \
                       "(would only push toward predicting negative); " \
                       "personalized model = FedAvg global model as-is"

    Xc_te = scale(cte[FROZEN_FEATURES].values.astype(np.float32), mean, std)
    score = predict_proba(personal_model, Xc_te)
    m = evaluate(cte["target_t1"].values, score, threshold=global_threshold)
    row.update({"status": "computed", "fine_tuned": fine_tuned,
                "threshold": global_threshold,
                "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
    personalized_rows.append(row)

personalized_df = pd.DataFrame(personalized_rows)
personalized_df.to_csv("results/phase6_personalized_per_client.csv", index=False)
print("=== Part 8/9: Personalized FL (FedAvg + local fine-tuning), per client ===")
print(personalized_df.to_string(index=False))
print("\nDone with phase6_personalized.py")
