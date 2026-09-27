"""Activation functions represented as layers."""

from __future__ import annotations

import numpy as np

from .BaseClasses import Layer


class ReLU(Layer):
    def __init__(self) -> None:
        self.input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = np.asarray(x, dtype=float)
        return np.maximum(0.0, self.input)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.input is None:
            raise RuntimeError("forward must be called before backward")
        return np.asarray(grad_output) * (self.input > 0)


class Sigmoid(Layer):
    def __init__(self) -> None:
        self.output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        output = np.empty_like(x)
        positive = x >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        output[~positive] = exp_x / (1.0 + exp_x)
        self.output = output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.output is None:
            raise RuntimeError("forward must be called before backward")
        return np.asarray(grad_output) * self.output * (1.0 - self.output)


class Softmax(Layer):
    """Softmax over the last axis of a 2-D batched input."""

    def __init__(self) -> None:
        self.output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"Softmax expects a 2-D array, got shape {x.shape}")
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.output is None:
            raise RuntimeError("forward must be called before backward")
        grad_output = np.asarray(grad_output)
        if grad_output.shape != self.output.shape:
            raise ValueError("grad_output must have the same shape as Softmax output")
        projection = np.sum(grad_output * self.output, axis=1, keepdims=True)
        return self.output * (grad_output - projection)

class Tanh(Layer):
    def __init__(self) -> None:
        self.output:np.ndarray |None =None

    def forward(self,x:np.ndarray)->np.ndarray:
        x=np.asarray(x, dtype=float)
        self.output=np.tanh(x)
        return self.output

    def backward(self,grad_output:np.ndarray)->np.ndarray:
        if self.output is None:
            raise RuntimeError("forward must be called before backward")
        if grad_output.shape!=self.output.shape:
            raise ValueError("grad_output must have the same shape as Softmax output")
        return grad_output * (1.0 - np.square(self.output))

