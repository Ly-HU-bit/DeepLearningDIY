from __future__ import annotations

from typing import override

from .BaseClasses import  Layer
import numpy as np


class Dropout(Layer):
    def __init__(self, drop_rate: float=0.5,rng:np.random.Generator |None =None) -> None:
        if not drop_rate in range (0,1):
            raise ValueError("Drop rate must be between 0 and 1")
        super().__init__()
        self.drop_rate = drop_rate
        self.rng = rng if rng is not None else np.random.default_rng()
        self._scaled_mask = None
        self.training=True
        self._last_forward_training: bool | None = None

    @override
    def forward(self,input:np.ndarray)->np.ndarray:
        if self.training:
          keep_prob = 1.0 - self.drop_rate
          self._scaled_mask=(self.rng.random(input.shape)<keep_prob)/keep_prob
          self._last_forward_training=True
          return input * self._scaled_mask
        else:
            self._last_forward_training=True
            return input
    @override
    def backward(self,grad_output:np.ndarray)->np.ndarray:
        grad_output = np.asarray(grad_output, dtype=float)
        if self._last_forward_training is None:
            raise RuntimeError("forward must be called before backward")
        elif self._scaled_mask is None:
            return grad_output
        else:
            result=grad_output * self._scaled_mask
            self._scaled_mask=None
            return result

