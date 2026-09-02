"""Gradient-based optimizers."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .BaseClasses import Optimizer, Parameter

ParameterSource = Iterable[Parameter] | None


def _learning_rate(default: float, override: float | None =None) -> float:
    value = default if override is None else float(override)
    if value <= 0:
        raise ValueError("learning_rate must be positive")
    return value


class SGD(Optimizer):
    def step(
        self,
        parameters: ParameterSource = None,
        learning_rate: float | None = None,
    ) -> None:
        lr = _learning_rate(self.learning_rate, learning_rate)
        for parameter in self._resolve_parameters(parameters):
            parameter.data -= lr * parameter.grad


class Momentum(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter] | None = None,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(parameters, learning_rate)
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = float(momentum)

    def step(
        self,
        parameters: ParameterSource = None,
        learning_rate: float | None = None,
    ) -> None:
        lr = _learning_rate(self.learning_rate, learning_rate)
        for parameter in self._resolve_parameters(parameters):
            state = self.state.setdefault(parameter, {})
            velocity = state.setdefault("velocity", np.zeros_like(parameter.data))
            velocity *= self.momentum
            velocity -= lr * parameter.grad
            parameter.data += velocity


class AdaGrad(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter] | None = None,
        learning_rate: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, learning_rate)
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    def step(
        self,
        parameters: ParameterSource = None,
        learning_rate: float | None = None,
    ) -> None:
        lr = _learning_rate(self.learning_rate, learning_rate)
        for parameter in self._resolve_parameters(parameters):
            state = self.state.setdefault(parameter, {})
            cache = state.setdefault("cache", np.zeros_like(parameter.data))
            cache += parameter.grad * parameter.grad
            parameter.data -= lr * parameter.grad / (np.sqrt(cache) + self.eps)


class Adam(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter] | None = None,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, learning_rate)
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.t = 0

    def step(
        self,
        parameters: ParameterSource = None,
        learning_rate: float | None = None,
    ) -> None:
        resolved = self._resolve_parameters(parameters)
        if not resolved:
            return
        lr = _learning_rate(self.learning_rate, learning_rate)
        self.t += 1  # One time step per batch update, not per layer.
        for parameter in resolved:
            state = self.state.setdefault(parameter, {})
            first = state.setdefault("first_moment", np.zeros_like(parameter.data))
            second = state.setdefault("second_moment", np.zeros_like(parameter.data))
            first *= self.beta1
            first += (1.0 - self.beta1) * parameter.grad
            second *= self.beta2
            second += (1.0 - self.beta2) * np.square(parameter.grad)
            first_hat = first / (1.0 - self.beta1**self.t)
            second_hat = second / (1.0 - self.beta2**self.t)
            parameter.data -= lr * first_hat / (np.sqrt(second_hat) + self.eps)
