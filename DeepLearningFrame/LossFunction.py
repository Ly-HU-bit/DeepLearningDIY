"""Loss functions for regression and classification."""

from __future__ import annotations

import numpy as np

from .BaseClasses import Loss


def _one_hot_targets(target: np.ndarray, classes: int) -> np.ndarray:
    target = np.asarray(target)
    if target.ndim == 1:
        if not np.issubdtype(target.dtype, np.integer):
            raise ValueError("1-D classification targets must contain integer labels")
        if np.any(target < 0) or np.any(target >= classes):
            raise ValueError("classification target is outside the valid class range")
        return np.eye(classes, dtype=float)[target]
    if target.ndim == 2 and target.shape[1] == classes:
        return target.astype(float, copy=False)
    raise ValueError(
        f"target must have shape (batch,) or (batch, {classes}), got {target.shape}"
    )


class MSE(Loss):
    def __init__(self) -> None:
        self.predictions: np.ndarray | None = None
        self.targets: np.ndarray | None = None

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        t = np.asarray(t, dtype=float)
        if x.shape != t.shape:
            raise ValueError(f"prediction and target shapes differ: {x.shape} vs {t.shape}")
        self.predictions, self.targets = x, t
        return float(np.mean(np.square(x - t)))

    def backward(self) -> np.ndarray:
        if self.predictions is None or self.targets is None:
            raise RuntimeError("forward must be called before backward")
        return 2.0 * (self.predictions - self.targets) / self.predictions.size


class SoftmaxCrossEntropy(Loss):
    """Numerically stable cross entropy that receives unnormalized logits."""

    def __init__(self) -> None:
        self.probabilities: np.ndarray | None = None
        self.targets: np.ndarray | None = None

    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        logits = np.asarray(logits, dtype=float)
        if logits.ndim != 2 or logits.shape[0] == 0:
            raise ValueError("logits must have shape (non_empty_batch, classes)")
        targets = _one_hot_targets(target, logits.shape[1])
        if targets.shape[0] != logits.shape[0]:
            raise ValueError("prediction and target batch sizes differ")

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_normalizer = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probabilities = shifted - log_normalizer
        self.probabilities = np.exp(log_probabilities)
        self.targets = targets
        return float(-np.sum(targets * log_probabilities) / logits.shape[0])

    def backward(self) -> np.ndarray:
        if self.probabilities is None or self.targets is None:
            raise RuntimeError("forward must be called before backward")
        return (self.probabilities - self.targets) / self.targets.shape[0]

