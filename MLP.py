"""Sequential multi-layer perceptron model."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .BaseClasses import Layer, Loss, Optimizer, Parameter


class MLP:
    """A small sequential model with manually implemented backpropagation."""

    def __init__(
        self,
        epochs: int = 1,
        batch_size: int = 10,
    ) -> None:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.layers: list[Layer] = []
        self.loss: Loss | None = None
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)

    def add_layer(self, layer: Layer) -> None:
        if not isinstance(layer, Layer):
            raise TypeError("layer must be a Layer instance")
        self.layers.append(layer)

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

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.layers:
            raise ValueError("the network has no layers")
        output = np.asarray(x, dtype=float)
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.layers:
            raise ValueError("the network has no layers")
        gradient = np.asarray(grad_output, dtype=float)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        optimizer: Optimizer,
        *,
        loss: Loss | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> list[float]:
        from .Trainer import Trainer

        selected_loss = loss or self.loss
        if selected_loss is None:
            raise ValueError("no loss function supplied")
        trainer = Trainer(
            model=self,
            loss=selected_loss,
            optimizer=optimizer,
            epochs=self.epochs if epochs is None else epochs,
            batch_size=self.batch_size if batch_size is None else batch_size,
            seed=seed,
        )
        self.loss = selected_loss
        return trainer.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        previous_modes = [layer.training for layer in self.layers]
        for layer in self.layers:
            layer.eval()
        try:
            return self.forward(x)
        finally:
            for layer, was_training in zip(self.layers, previous_modes):
                if was_training:
                    layer.train()
                else:
                    layer.eval()

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        if prediction.ndim != 2:
            raise ValueError("classification predictions must be two-dimensional")
        labels = np.asarray(y)
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=1)
        elif labels.ndim != 1:
            raise ValueError("labels must have shape (batch,) or (batch, classes)")
        if labels.shape[0] != prediction.shape[0]:
            raise ValueError("prediction and target batch sizes differ")
        return float(np.mean(np.argmax(prediction, axis=1) == labels))

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            f"layers.{layer_index}.parameters.{parameter_index}": parameter.data.copy()
            for layer_index, layer in enumerate(self.layers)
            for parameter_index, parameter in enumerate(layer.parameters())
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        expected = self.state_dict()
        if set(state) != set(expected):
            missing = set(expected) - set(state)
            unexpected = set(state) - set(expected)
            raise ValueError(f"state keys differ; missing={missing}, unexpected={unexpected}")
        for layer_index, layer in enumerate(self.layers):
            for parameter_index, parameter in enumerate(layer.parameters()):
                key = f"layers.{layer_index}.parameters.{parameter_index}"
                value = np.asarray(state[key])
                if value.shape != parameter.data.shape:
                    raise ValueError(
                        f"shape mismatch for {key}: {value.shape} vs {parameter.data.shape}"
                    )
                parameter.data[...] = value
