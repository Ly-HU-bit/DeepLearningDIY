


"""Mini-batch training orchestration kept separate from the model."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .BaseClasses import Loss, Optimizer


class Trainer:
    def __init__(
        self,
        model,
        loss: Loss,
        optimizer: Optimizer,
        *,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float | None = None,
        seed: int | None = None,
    ) -> None:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if learning_rate is not None and learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

    def fit(self, x: np.ndarray, y: np.ndarray) -> list[float]:
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 2:
            raise ValueError(f"training input must be two-dimensional, got {x.shape}")
        if len(x) == 0:
            raise ValueError("training data cannot be empty")
        if len(x) != len(y):
            raise ValueError("training inputs and targets have different lengths")

        losses: list[float] = []
        for _ in range(self.epochs):
            indices = self.rng.permutation(len(x))
            total_loss = 0.0
            seen = 0
            for start in range(0, len(x), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                xb, yb = x[batch_indices], y[batch_indices]
                self.model.zero_grad()
                prediction = self.model.forward(xb)
                batch_loss = self.loss.forward(prediction, yb)
                self.model.backward(self.loss.backward())
                self.optimizer.step(self.model.parameters(), self.learning_rate)
                total_loss += batch_loss * len(xb)
                seen += len(xb)
            losses.append(total_loss / seen)
        return losses

    def draw(self,losses:list[float]) :
        plt.plot(losses)
        plt.show()