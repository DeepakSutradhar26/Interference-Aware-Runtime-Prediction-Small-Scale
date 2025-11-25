"""
Magnitude-based pruning of MLP weights.

We prune by zeroing out a given fraction of the smallest-magnitude weights
across all layers (coefs_ only; biases are kept).
"""

from copy import deepcopy
from typing import Tuple, List

import numpy as np
from sklearn.neural_network import MLPRegressor


def _flatten_weights(coefs: List[np.ndarray]):
    return np.concatenate([c.ravel() for c in coefs])


def prune_mlp(model: MLPRegressor, prune_fraction: float) -> MLPRegressor:
    """
    Returns a deep-copied, pruned version of the given model.
    """
    assert 0.0 <= prune_fraction <= 1.0
    pruned_model = deepcopy(model)

    all_weights = _flatten_weights(pruned_model.coefs_)
    k = int(prune_fraction * all_weights.size)

    if k == 0:
        return pruned_model

    threshold = np.partition(np.abs(all_weights), k)[k]

    # Apply threshold layer by layer
    new_coefs = []
    for W in pruned_model.coefs_:
        W_new = W.copy()
        mask = np.abs(W_new) < threshold
        W_new[mask] = 0.0
        new_coefs.append(W_new)

    pruned_model.coefs_ = new_coefs
    return pruned_model


def predict_pruned(model: MLPRegressor, X):
    """
    Just calls model.predict; kept for symmetry with quantized version.
    """
    return model.predict(X)
