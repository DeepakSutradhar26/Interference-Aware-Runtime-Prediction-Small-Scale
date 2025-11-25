"""
Simple 8-bit weight quantization for the MLP.

We simulate quantization by:
- Scaling each weight matrix by max_abs -> int8 range [-127, 127].
- Storing scales to de-quantize during forward pass.
- Biases are left in float for simplicity.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.neural_network import MLPRegressor


@dataclass
class QuantizedMLP:
    """Container for quantized weights."""
    q_weights: List[np.ndarray]  # int8 weights
    scales: List[float]          # scale per layer
    biases: List[np.ndarray]     # float biases
    hidden_activation: str       # only 'relu' supported


def quantize_mlp(model: MLPRegressor) -> QuantizedMLP:
    q_weights = []
    scales = []
    biases = [b.copy() for b in model.intercepts_]

    for W in model.coefs_:
        max_abs = float(np.max(np.abs(W))) + 1e-8
        scale = max_abs / 127.0
        W_q = np.round(W / scale).astype(np.int8)
        q_weights.append(W_q)
        scales.append(scale)

    return QuantizedMLP(
        q_weights=q_weights,
        scales=scales,
        biases=biases,
        hidden_activation=model.activation,
    )


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def predict_quantized(qmlp: QuantizedMLP, X: np.ndarray) -> np.ndarray:
    """
    Forward pass using quantized weights (de-quantizing on-the-fly).
    """
    a = X.astype(np.float32)

    for i, (W_q, scale, b) in enumerate(
        zip(qmlp.q_weights, qmlp.scales, qmlp.biases)
    ):
        W = W_q.astype(np.float32) * scale
        a = a @ W + b

        # Apply activation for all but last layer
        if i < len(qmlp.q_weights) - 1:
            if qmlp.hidden_activation == "relu":
                a = _relu(a)
            else:
                raise ValueError("Only ReLU activation is supported in quantized path.")

    return a
