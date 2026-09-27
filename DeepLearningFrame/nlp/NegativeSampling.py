import numpy as np
from ..basics.BaseClasses import Objective
from .Embedding import Embedding,EmbeddingDot
from .Sampler import *
from .LossObjectiveForNegativeSampling import *


class NegativeSamplingCBOW(Objective):
    """
    for this non-sequential input stream, have to jump out of the basic stream and structure of storing layers
    mention: currently doesn't fit the current trainer!
    """
    #                                                        Sampler
    #the Negative Sampling contains: initial Embedding   --->EmbeddingDot  --->SigmoidWithLoss
    def __init__(self,weight_height:int,weight_width:int,weight_rng:np.random.Generator,sampler:Sampler ,activationAndLoss:Objective |None)->None:
        super().__init__()
        self.embedding=Embedding(weight_height,weight_width,weight_rng)
        self.embeddingDot=EmbeddingDot(weight_height,weight_width,weight_rng)
        self.sampler=sampler
        if activationAndLoss is not None:
            self.activationAndLoss=activationAndLoss
        else:
            self.activationAndLoss=SigmoidWithLoss()

    def forward(self,context:np.ndarray,target:np.ndarray,sample_size:int)->float:
        prediction=self.embedding.forward(context)
        negative_ids=self.sampler.sample(target,sample_size)
        samples = np.concatenate(
            [target[:, None], negative_ids], axis=1
        )  # (B,1+K)
        labels = np.zeros(samples.shape, dtype=float)
        labels[:, 0] = 1.0
        dotProduct=self.embeddingDot.forward(prediction,samples)
        self.output=self.activationAndLoss.forward(labels,dotProduct)
    def backward(self,dout=1)->None:
        dActivationAndLoss=self.activationAndLoss.backward(dout)
        dEmbeddingDot=self.embeddingDot.backward(dActivationAndLoss)
        dEmbedding=self.embedding.backward(dEmbeddingDot)
        return dEmbedding

    def parameters(self):
        yield from self.embedding.parameters()
        yield from self.embeddingDot.parameters()

