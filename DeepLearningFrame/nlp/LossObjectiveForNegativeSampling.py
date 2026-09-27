from ..basics.BaseClasses import Objective
from ..basics.ActivationFunction import Sigmoid
import numpy as np

class SigmoidWithLoss(Objective):
    def __init__(self,eps:float=1e-6)->None:
        super().__init__()
        self.eps=eps
        self.grad_shape=None
        self.sigmoid = Sigmoid()

    def forward(self,answer:np.ndarray,input:np.ndarray)->float:
        input = np.asarray(input, dtype=float)
        answer = np.asarray(answer, dtype=float)
        if answer.shape!=input.shape:
            raise ValueError(f"shape of input{input.shape} does not match shape of answer{answer.shape} ")
        self.grad_shape=input.shape
        self.probabilities=self.sigmoid.forward(input)
        self.answers=answer
        output=np.maximum(input, 0)-answer*input+np.log1p(np.exp(-np.abs(input)))
        output=np.mean(output.sum(axis=1),axis=0)
        return float(output)

    def backward(self,dout:float=1.0)->np.ndarray:
        if self.grad_shape is None:
            raise RuntimeError("forward must be called before backward")
        else:
            batch_size=self.grad_shape[0]
            return dout * (self.probabilities - self.answers) / batch_size







