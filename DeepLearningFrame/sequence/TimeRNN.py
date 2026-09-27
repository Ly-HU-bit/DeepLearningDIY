import numpy as np
from ..basics.BaseClasses import ParameterizedLayer,Parameter
from ..recurrent.RNN import RNN

class TimeRNN(ParameterizedLayer):
    def __init__(self,cell_numbers:int,input_size:int,hidden_size:int,rng_wx:np.random.Generator=None,rng_wh:np.random.Generator=None,truncate_size:int=10)->None:
        super().__init__()
        self.input_size = input_size
        self.weight_width = hidden_size
        if truncate_size <=0:
            raise ValueError(f"Truncate_size must >=0, but got {truncate_size}.")
        self.truncate_size = truncate_size
        self.rng_wx=rng_wx if rng_wx is not None else np.random.default_rng()
        self.rng_wh=rng_wh if rng_wh is not None else np.random.default_rng()
        parameter_wx=Parameter(self.rng_wx.standard_normal((input_size,hidden_size)))
        parameter_wh=Parameter(self.rng_wh.standard_normal((hidden_size,hidden_size)))
        parameter_b=Parameter(np.zeros(hidden_size,))
        self.cells=[]
        cell_count=0
        for i in range(cell_numbers):
            cell_count+=1
            cell=RNN(input_size,hidden_size,rng_wx,rng_wh)
            cell.share_parameter(parameter_wx)
            cell.share_parameter(parameter_wh)
            cell.share_parameter(parameter_b)
            cell.truncated=False
            if cell_count%self.truncate_size==0:
                cell.truncated=True
            self.cells.append(cell)

    def forward(self,x:np.ndarray)->np.ndarray:


