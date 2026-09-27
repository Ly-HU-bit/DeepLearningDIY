import numpy as np
from ..basics.BaseClasses import Model,Layer,ParameterizedLayer,Optimizer,Loss
from ..basics.ActivationFunction import Softmax
from ..basics.LinearLayer import LinearLayer
from .Embedding import Embedding
"co-occurance_matrix->PPMI->SVD\cbow\skip-gram"

class CBOW(Model):

    #by default uses the Embedding mechanism, while negative sampling is optional
    #"""by summing the result of embeddings from both former and latter context, the actual order of former and latter is neglected """
  def __init__(self,vocabulary_size:int,embedding_size:int)->None:
      super().__init__()
      self.vocabulary_size=vocabulary_size
      self.embedding_size=embedding_size
      self.context_shape=None
      self.embedding=self.add_layer(Embedding(vocabulary_size,embedding_size))
      self.output_layer=self.add_layer(LinearLayer(embedding_size,vocabulary_size,has_bias=False))

  def forward(self,context:np.ndarray)->np.ndarray:
      #mention that input from former and latter context are combined--multi head with shared params can be combined
      x=context.copy()
      self.context_shape=context.shape[1]
      embedded=self.embedding.forward(x)
      embedded_mean=np.mean(embedded,axis=1)
      return self.output_layer.forward(embedded_mean)

  def backward(self,grad_out:np.ndarray)->np.ndarray:
      if self.context_shape is None:
          raise RuntimeError("forward must be called before backward in CBOW layer!")
      x=grad_out.copy()
      x=self.output_layer.backward(x)
      x = np.repeat(x[:, None, :], self.context_shape, axis=1)
      x/=self.context_shape
      return self.embedding.backward(x)

  def retrieveVectorization(self):
        return self.embedding.weight.data



class SkipGram(Model):
    def __init__(self,vocabulary_size:int,embedding_size:int)->None:
        super().__init__()
        self.vocabulary_size=vocabulary_size
        self.embedding_size=embedding_size
        self.embedding=self.add_layer(Embedding(vocabulary_size,embedding_size))
        self.output_layer=self.add_layer(LinearLayer(embedding_size,vocabulary_size,has_bias=False))

    def forward(self, center):
        #注意：skipgram中windowsize始终是1，意味着embedding后实际展开结果就是二维的，因此不必再像CBOW一样调整维度
        h = self.embedding.forward(center)
        return self.output_layer.forward(h)

    def backward(self, grad_output):
        dh = self.output_layer.backward(grad_output)
        return self.embedding.backward(dh)
