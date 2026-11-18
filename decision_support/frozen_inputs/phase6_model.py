"""
phase6_model.py — shared model architecture and utilities used identically
by centralized, local-only, FedAvg, and personalized-FL experiments, so the
comparison in Part 10 (fairness) is apples-to-apples: same architecture,
same features, same loss, same metric code.

Model: plain logistic regression (single linear layer, no hidden layers,
no BatchNorm/Dropout) -- deliberately the simplest model that matches
Phase 4's best result (sklearn LogisticRegression), reimplemented in
PyTorch specifically so FedAvg's weight-averaging is mathematically
well-defined (averaging tree ensembles, as Random Forest would require,
is not a standard or well-defined FedAvg operation; averaging linear
model weights is exactly what FedAvg was designed for).

Standardization: features are z-scored using TRAIN-set mean/std only
(never val/test), computed once and reused everywhere, to keep every
method's inputs on an identical scale.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)  # raw logits; BCEWithLogitsLoss applied outside


def get_scaler(train_X: np.ndarray):
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def scale(X, mean, std):
    return (X - mean) / std


def get_params(model):
    return [p.detach().numpy().copy() for p in model.parameters()]


def set_params(model, params):
    with torch.no_grad():
        for p, new in zip(model.parameters(), params):
            p.copy_(torch.tensor(new, dtype=torch.float32))


def train_local(model, X, y, epochs=100, lr=0.05, weight_decay=1e-3, pos_weight=None):
    """Full-batch gradient descent (datasets here are small enough that
    full-batch is fine and keeps training deterministic given the seed)."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pw = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()
    return model, float(loss.item())


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy().ravel()


def class_balanced_pos_weight(y):
    """Mirrors sklearn's class_weight='balanced' used in Phases 3-5, so
    the PyTorch reproduction targets the same weighting convention."""
    n_pos = max(int(y.sum()), 1)
    n_neg = len(y) - n_pos
    return n_neg / n_pos


def evaluate(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "n": len(y_true), "positives": int(sum(y_true)),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true)) > 1:
        out["pr_auc"] = average_precision_score(y_true, y_score)
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        out["pr_auc"], out["roc_auc"] = None, None
    return out


def select_threshold_from_train_cv(X_train, y_train, years_train, n_folds=5, epochs=100):
    """Temporal CV within train, exactly the procedure used in Phases 3-5,
    reimplemented for the PyTorch model. Returns the max-F1 threshold from
    out-of-fold predictions -- never touches val or test."""
    years = sorted(set(years_train))
    fold_edges = np.array_split(years, n_folds)
    oof_true, oof_score = [], []
    for fold_years in fold_edges:
        val_mask = np.isin(years_train, fold_years)
        if val_mask.sum() == 0 or y_train[val_mask].sum() == 0:
            continue
        Xf_tr, yf_tr = X_train[~val_mask], y_train[~val_mask]
        Xf_va, yf_va = X_train[val_mask], y_train[val_mask]
        if yf_tr.sum() == 0:
            continue
        m = LogisticRegressionModel(X_train.shape[1])
        pw = class_balanced_pos_weight(yf_tr)
        train_local(m, Xf_tr, yf_tr, epochs=epochs, pos_weight=pw)
        s = predict_proba(m, Xf_va)
        oof_true.extend(yf_va.tolist())
        oof_score.extend(s.tolist())

    if not oof_true or sum(oof_true) == 0:
        return 0.5, None  # no usable OOF positives -- flagged, not guessed

    oof_true = np.array(oof_true)
    oof_score = np.array(oof_score)
    best_t, best_f1 = 0.5, -1
    for t in [0.10, 0.20, 0.30, 0.40, 0.50]:
        pred = (oof_score >= t).astype(int)
        f1 = f1_score(oof_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1
