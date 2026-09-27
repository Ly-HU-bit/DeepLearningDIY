from ..basics.BaseClasses import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape=None


    def forward(self,input:np.ndarray)->np.ndarray:
        input = np.asarray(input, dtype=float)
        if input.ndim<2:
            raise ValueError('Convolution Input in flatten layer must be 4D')
        self.input_shape = input.shape
        return  input.reshape(input.shape[0],-1)

    def backward(self,grad_output:np.ndarray)->np.ndarray:
        grad_output=np.asarray(grad_output, dtype=float)
        if self.input_shape is None:
            raise RuntimeError('you must forward before backward in flatten layer!')
        if not ( grad_output.ndim == 2 and  grad_output.shape[0] == self.input_shape[0] and grad_output.shape[1]==(np.prod(self.input_shape[1:]))):
            raise ValueError(f'gradient_out of flatten layer error! got {grad_output.shape}')
        return grad_output.reshape(self.input_shape)