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
    def __init__(self,data:np.ndarray,grad:np.ndarray | None=None):
        if grad is not None:
            if data.shape != grad.shape:
                raise ValueError(f"Shape of the grad input:{grad.shape} does not match shape of data:{data.shape}")
        self.data=data
        self.grad=grad

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        self.grad = np.zeros_like(self.data)
    def zero_grad(self) -> None:
        self.grad.fill(0.0)


class Layer(ABC):
    """Base interface for layers in the MLP."""

    training: bool = True


    @abstractmethod
    def forward(self, *x:np.ndarray) -> np.ndarray:
        """Return the layer output and cache what backward needs.
           For the sake of further non-sequential layer (EmbeddingDot).do not limit the amount of input"""

    @abstractmethod
    def backward(self, *grad_output: np.ndarray) -> np.ndarray:
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

    def share_parameter(self,parameter: Parameter)->None:
        if not isinstance(parameter, Parameter):
            raise TypeError("the parameter is supposed to be a Parameter instance,which is shared among layers")
        self._parameters.append(parameter)

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

class Model(ABC):
    """Base interface for models that own trainable parameters."""
    def __init__(self) -> None:
        self.layers: list[Layer] = []
        self.loss: Loss |None =None

    def add_layer(self, layer: Layer) -> Layer:
            if not isinstance(layer, Layer):
                raise TypeError("layer must be a Layer instance")
            self.layers.append(layer)
            return layer

    def set_loss(self, loss: Loss) -> None:
        if not isinstance(loss, Loss):
            raise TypeError("loss must be a Loss instance")
        self.loss = loss

    def parameters(self) -> Iterator[Parameter]:
        for layer in self.layers:
            yield from layer.parameters()

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self) -> None:
        "switch the layers to training mode , instead of actually training."
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        "switch the layers to evaluation mode, instead of actually evaluating."
        for layer in self.layers:
            layer.eval()


    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        "the forward propagation of the model"

    @abstractmethod
    def backward(self,grad_output: np.ndarray) -> np.ndarray:
        "the backward propagation of the model"

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run prediction in evaluation mode."""
        previous_modes = [layer.training for layer in self.layers]

        for layer in self.layers:
            layer.eval()

        try:
            return self.forward(x)
        finally:
            for layer, was_training in zip(
                    self.layers, previous_modes
            ):
                if was_training:
                    layer.train()
                else:
                    layer.eval()

class Objective(Model):
    """
    an Objective (Head) can be seen as a small model(component): it (selects) takes in input,calcs the output and the loss
    """
    @abstractmethod
    def forward(self, *inputs) -> float:
        "the forward propagation of the Objective part--mention that it directly calcs the loss"

    @abstractmethod
    def backward(self,dout:int=1) -> np.ndarray:
        "the backward propagation of the Objective part--mention that it is the tail of a model"