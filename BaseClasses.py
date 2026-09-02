"""Core abstractions used by the small neural-network framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np


@dataclass(eq=False)
class Parameter:
    """A trainable array and its gradient."""

    data: np.ndarray
    grad: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        self.grad = np.zeros_like(self.data)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)


class Layer(ABC):
    """Base interface for layers in the MLP."""

    training: bool = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the layer output and cache what backward needs."""

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Return the gradient with respect to this layer's input."""

    def parameters(self) -> Iterator[Parameter]:
        return iter(())

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False


class ParameterizedLayer(Layer):
    """Base class for layers that own trainable parameters."""

    def __init__(self) -> None:
        self._parameters: list[Parameter] = []

    def register_parameter(self, value: np.ndarray) -> Parameter:
        parameter = Parameter(value)
        self._parameters.append(parameter)
        return parameter

    def parameters(self) -> Iterator[Parameter]:
        return iter(self._parameters)


class Loss(ABC):
    """Base interface for scalar loss functions."""

    @abstractmethod
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute and return a scalar batch loss."""

    @abstractmethod
    def backward(self) -> np.ndarray:
        """Return the gradient with respect to the stored predictions."""


class Optimizer(ABC):
    """Base optimizer operating on Parameter objects."""

    def __init__(
        self,
        parameters: Iterable[Parameter] | None = None,
        learning_rate: float = 0.01,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.parameters = list(parameters or [])
        self.learning_rate = float(learning_rate)
        self.state: dict[Parameter, dict[str, np.ndarray]] = {}

    def _resolve_parameters(
        self,
        parameters: Iterable[Parameter] | None,
    ) -> list[Parameter]:
        if parameters is None:
            return self.parameters
        return list(parameters)

    @abstractmethod
    def step(
        self,
        parameters: Iterable[Parameter] | None = None,
        learning_rate: float | None = None,
    ) -> None:
        """Update all supplied parameters exactly once."""

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()
