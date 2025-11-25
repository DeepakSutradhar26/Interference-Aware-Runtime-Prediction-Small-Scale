"""
Metrics: Mean Absolute Percentage Error (MAPE) and R^2.
"""

import numpy as np
from sklearn.metrics import r2_score


def mape(y_true, y_pred) -> float:
    """
    Mean Absolute Percentage Error, in [0, inf).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))))


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))
