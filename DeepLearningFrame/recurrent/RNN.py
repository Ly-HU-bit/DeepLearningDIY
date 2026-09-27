import numpy as np
from ..basics.BaseClasses import ParameterizedLayer
from ..basics.LinearLayer import LinearLayer


class RNN(ParameterizedLayer):
    def __init__(self,input_size:int,hidden_size:int,rng_wx:np.random.Generator=None,rng_wh:np.random.Generator=None):
        super().__init__()
        if input_size <=0 or hidden_size<=0:
            raise ValueError(f"input_size:{input_size} and hidden_size:{hidden_size} must be positive in RNN cell")
        self.input_size=input_size
        self.hidden_size=hidden_size
        generatorx=rng_wx if rng_wx is not None else np.random.default_rng()
        generatorh=rng_wh if rng_wh is not None else np.random.default_rng()
        weight_x=generatorx.standard_normal((input_size,hidden_size))
        weight_h=generatorh.standard_normal((hidden_size,hidden_size))
        bias=np.zeros(hidden_size,)
        self.weight_x=self.register_parameter(weight_x)
        self.weight_h=self.register_parameter(weight_h)
        self.bias=self.register_parameter(bias)
        self.h=None




    def forward(self,x:np.ndarray,prev_h:np.ndarray)->np.ndarray:
        if prev_h is None:
            raise RuntimeError("no designated h in base RNN cell!")
        self.x=x
        self.prev_h=prev_h
        self.h=np.tanh(x@self.weight_x.data+prev_h@self.weight_h.data+self.bias.data)
        return self.h

    def backward(self,grad_output:np.ndarray)->dict[str, np.ndarray]:
        if self.h is None:
            raise RuntimeError("forward must be called before backward in base RNN cell!")
        grad_output = np.asarray(grad_output, dtype=float)
        da = grad_output * (1 - self.h ** 2)
        self.bias.grad+=da.sum(axis=0)
        self.weight_h.grad+=self.prev_h.T@da
        self.weight_x.grad+=self.x.T@da
        grad_prev_h=da@self.weight_h.data.T
        grad_x=da@self.weight_x.data.T
        return {"grad_x":grad_x,"grad_prev_h":grad_prev_h}












