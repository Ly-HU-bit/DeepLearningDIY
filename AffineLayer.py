"""Trainable fully-connected layer."""

from __future__ import annotations

import numpy as np

from .BaseClasses import ParameterizedLayer


class AffineLayer(ParameterizedLayer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        initializer: str = "xavier",
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        generator = np.random.default_rng() if rng is None else rng
        scales = {
            "xavier": np.sqrt(2.0 / (input_size + output_size)),
            "he": np.sqrt(2.0 / input_size),
            "small": 0.1,
        }
        if initializer not in scales:
            raise ValueError(f"unknown initializer: {initializer!r}")
        self.weight = self.register_parameter(
            generator.standard_normal((input_size, output_size)) * scales[initializer]
        )
        self.bias = self.register_parameter(np.zeros(output_size))
        self.input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(
                f"expected input shape (batch, {self.input_size}), got {x.shape}"
            )
        self.input = x
        return x @ self.weight.data + self.bias.data

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.input is None:
            raise RuntimeError("forward must be called before backward")
        grad_output = np.asarray(grad_output, dtype=float)
        expected = (self.input.shape[0], self.output_size)
        if grad_output.shape != expected:
            raise ValueError(f"expected grad_output shape {expected}, got {grad_output.shape}")
        self.weight.grad[...] = self.input.T @ grad_output
        self.bias.grad[...] = np.sum(grad_output, axis=0)
        return grad_output @ self.weight.data.T
