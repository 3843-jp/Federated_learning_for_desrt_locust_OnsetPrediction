"""
phase7_strategy.py — custom Flower Strategy that aggregates client updates
using a FIXED, precomputed per-client weight dictionary (from
phase7_weighting.py) instead of Flower's default num_examples weighting.
Used for both Adaptive Method A and Adaptive Method B by passing a
different weight dict -- same code path, so any performance difference
between A and B is attributable only to the weight formula, not to
different aggregation code (Part 8's "same everything except the
independent variable" requirement).
"""
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


class StaticWeightedFedAvg(FedAvg):
    def __init__(self, client_weights: dict, **kwargs):
        super().__init__(**kwargs)
        self.client_weights = client_weights
        self.round_log = []
        self.captured_final_params = {}

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weighted_arrays = None
        total_weight_present = 0.0
        entry = []
        for _, fit_res in results:
            country = fit_res.metrics["country"]
            w = self.client_weights[country]
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if weighted_arrays is None:
                weighted_arrays = [np.zeros_like(a) for a in arrays]
            for i, a in enumerate(arrays):
                weighted_arrays[i] = weighted_arrays[i] + w * a
            total_weight_present += w
            entry.append({
                "country": country, "weight": w,
                "num_examples": fit_res.num_examples,
                "train_loss": fit_res.metrics["train_loss"],
                "train_positives": fit_res.metrics["train_positives"],
            })

        if total_weight_present == 0:
            # all participating clients had zero weight this round --
            # should not happen with full participation and our formulas,
            # but guarded explicitly rather than dividing by zero silently
            return None, {}

        aggregated = [a / total_weight_present for a in weighted_arrays]
        self.round_log.append({"round": server_round, "clients": entry,
                                 "total_weight_present": total_weight_present})
        self.captured_final_params["params"] = aggregated
        self.captured_final_params["round"] = server_round
        return ndarrays_to_parameters(aggregated), {}
