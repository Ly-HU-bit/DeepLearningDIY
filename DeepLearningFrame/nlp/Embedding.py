import numpy as np
from ..basics.BaseClasses import ParameterizedLayer,Parameter
from typing import Iterator

class Embedding(ParameterizedLayer):
    weight: Parameter
    index:np.ndarray
    def __init__(self,weight_height:int,weight_width:int,rng:np.random.Generator |None=None) -> None:
        super().__init__()
        if weight_height <=0 or weight_width <=0:
            raise ValueError("vocab_size and embedding_size must be positive")
        generator = rng if rng is not None else np.random.default_rng()
        weight=generator.standard_normal((weight_height,weight_width))
        self.weight=self.register_parameter(weight)

    def forward(self,x:np.ndarray) -> np.ndarray:
        x=np.asarray(x)
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError(
                "Embedding indices must be integers"
            )

        if np.any(x < 0) or np.any(x >= self.weight.data.shape[0]):
            raise ValueError(
                "Embedding index out of vocabulary range"
            )
        self.index=x.copy()
        return self.weight.data[x]


    def backward(self,grad_output:np.ndarray) -> None:
        if self.index is None:
            raise RuntimeError("forward must be called before backward")
        grad_output = np.asarray(grad_output, dtype=float)
        expected_shape = (
                self.index.shape + (self.weight.data.shape[-1],)
        )

        if grad_output.shape != expected_shape:
            raise ValueError(
                f"expected grad_output shape "
                f"{expected_shape}, got {grad_output.shape}"
            )

        np.add.at(
            self.weight.grad,
            self.index,
            grad_output,
        )

        return None


class EmbeddingDot(ParameterizedLayer):
    #use composition to achieve the portraits of a pure Embedding
    def __init__(self,weight_height:int,weight_width:int,rng:np.random.Generator |None=None) -> None:
        """
        mention that when using Embedding to choose, its not necessary to follow the original Embed_size * Voc_size weight size
        for Linear layer,instead, to utilize the existing Embedding mechanism,set your Embedding weight as Voc_size* Embed_size
        """
        super().__init__()
        if weight_height <=0 or weight_width <=0:
            raise ValueError("vocab_size and embedding_size must be positive")
        generator = rng if rng is not None else np.random.default_rng()
        self.embedding=Embedding(weight_height,weight_width,rng=generator)
        self.input:np.ndarray | None =None
        self.selected_weight:np.ndarray | None=None   #selected lines in Embedding

    def parameters(self)->Iterator[Parameter]:
        yield from self.embedding.parameters()


    def forward(self,input:np.ndarray,index:np.ndarray)->np.ndarray:
        """
        supports a batch_size operation
        """
        input=np.asarray(input)
        index=np.asarray(index)
        self.input=input
        target_weight=self.embedding.forward(index)
        self.target_weight=target_weight
        #mention that: the input index should have shape (batch_size,1+k)
        output=(target_weight*self.input[:,None,:]).sum(axis=2)  # after dot product, the output should have shape(batchsize, 1+k,embedding_length)
        self.output_shape=output.shape
        return output

    def backward(self,grad_output:np.ndarray)->np.ndarray:
        if self.output_shape is None:
            raise RuntimeError("forward must be called before backward")
        if grad_output.shape != self.output_shape:
            raise ValueError(f"grad_output is expected to have shape {self.output_shape}, got{grad_output.shape}")
        grad_output=grad_output[:,:,None]
        self.embedding.backward(grad_output*self.input[:,None,:])
        new_grad=(grad_output*self.target_weight).sum(axis=1)
        return new_grad
        


