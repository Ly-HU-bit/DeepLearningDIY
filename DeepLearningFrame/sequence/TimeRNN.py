import numpy as np
from ..basics.BaseClasses import ParameterizedLayer,Parameter
from ..recurrent.RNN import RNN
from typing import Iterator

class TimeRNN(ParameterizedLayer):
    def __init__(self,cell_numbers:int,input_size:int,hidden_size:int,rng_wx:np.random.Generator=None,rng_wh:np.random.Generator=None,truncate_size:int=10)->None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if truncate_size <=0:
            raise ValueError(f"Truncate_size must >=0, but got {truncate_size}.")
        self.truncate_size = truncate_size
        self.rng_wx=rng_wx if rng_wx is not None else np.random.default_rng()
        self.rng_wh=rng_wh if rng_wh is not None else np.random.default_rng()
        self.wx=Parameter(self.rng_wx.standard_normal((input_size,hidden_size)))
        self.wh=Parameter(self.rng_wh.standard_normal((hidden_size,hidden_size)))
        self.b=Parameter(np.zeros(hidden_size,))
        self.cells=[]
        cell_count=0
        for i in range(cell_numbers):
            cell_count+=1
            cell=RNN(input_size,hidden_size,rng_wx,rng_wh)
            cell.share_parameter(self.wx)
            cell.share_parameter(self.wh)
            cell.share_parameter(self.b)
            cell.share_parameters(self.wx,self.wh,self.b)
            cell.truncated=False
            if cell_count%self.truncate_size==0:
                cell.truncated=True
            self.cells.append(cell)

    def parameters(self)->Iterator[Parameter]:
        return iter((self.wx,self.wh,self.b))

    def forward(self,x:np.ndarray)->np.ndarray:
        self.batch_size=x.shape[0]
        prev_h=np.zeros((self.batch_size,self.hidden_size))
        for index,cell in enumerate(self.cells):
            input=x[:,index,:]
            new_h=cell.forward(input,prev_h)
            prev_h=new_h
        return self.cells[-1].h

    def backward(self,grad_out:np.ndarray)->np.ndarray:
        prev_grad=grad_out
        dx=np.zeros_like((self.batch_size,len(self.cells)-1,self.input_size))
        index=len(self.cells)-1
        for cell in reversed(self.cells):
            if cell.truncated:
                prev_grad=1
            current_grad=cell.backward(prev_grad)
            prev_grad=current_grad["grad_prev_h"]
            dx[:,index,:]=current_grad["grad_x"]
            index-=1
        return dx



