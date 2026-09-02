import numpy as np
import matplotlib.pyplot as plt

class Layer:
    ##this class serves as a base interface for Layers in MLP
    def forward(self,input):
        print("forward undefined for current instance!")
    def backward(self,grad_output):
        print("backward undefined for current instance!")

class ParameterizedLayer(Layer):
    Weights=None
    Bias=None
    grad_Weights=None
    grad_Bias=None

class Loss:
    #serves as interface for Loss Functions in MLP
    def forward(self,input,target):
        print("forward undefined for current instance!")
    def backward(self):
        print("backward undefined for current instance!")

class Optimizer:
    def optimize(self,args,grad,learning_rate):
        print("optimizer undefined for current instance!")

class MLP:
    Layers=[]
    Activations=[]
    LossFunctions=[]
    Optimizers=[]
    HyperParameters={"LearningRate":0.01,"BatchSize":10}
    def __init__(self, epoch, HyperParameters=None):
        if HyperParameters is None:
            HyperParameters = {"LearningRate": 0.01, "BatchSize": 10}
        self.Layers=[]
        self.Activations=[Relu,Sigmoid,Softmax]
        self.LossFunctions=[MSE,CEE]
        self.Optimizers=[]
        self.HyperParameters=HyperParameters
        self.Epoch=epoch

    def AddLayer(self,layer):
        self.Layers.append(layer)
    def Forward(self,X):
        assert(len(self.Layers)>0,"you havn't established the network yet!")
        input=X
        result=None
        for layer in self.Layers:
            result=layer.forward(input)
            input=result
        return result

    def Backward(self,optimizer):
        layers_reversed=self.Layers[::-1]
        assert (len(self.Layers) > 1, "you havn't established the network yet! && The network has to be bigger!")
        assert isinstance(layers_reversed[0],Loss),"The Last part of network must be a Loss Function!"
        grad_output=layers_reversed[0].backward()
        grad_final=None
        for layer in layers_reversed[1:]:
            grad_final=layer.backward(grad_output)
            grad_output=grad_final
        for layer in self.Layers:
            if(isinstance(layer,ParameterizedLayer)):
                layer.Weights=optimizer.optimize(layer.Weights,layer.grad_Weights,self.HyperParameters["LearningRate"])
                layer.Bias=optimizer.optimize(layer.Bias,layer.grad_Bias,self.HyperParameters["LearningRate"])

    def Accuracy(self):
        pass

class AffineLayer(ParameterizedLayer):
    Weights = None
    Bias=None
    Input=None   #for the use of backpropogation caching
    InputSize=0
    NodeSize=0
    grad_Weights=None
    grad_Bias=None
    def __init__(self,InputSize,NodeSize):
        self.InputSize=InputSize
        self.NodeSize=NodeSize
        self.Weights=np.random.randn(InputSize,NodeSize)
        self.Bias=np.random.randn(NodeSize)
    def forward(self,x):
        self.Input=x
        return np.dot(self.Input,self.Weights)+self.Bias
    def backward(self,grad_output):
        self.grad_Weights= np.dot(self.Input.T,grad_output)
        self.grad_Bias = np.sum(grad_output, axis=0)
        grad_Input = np.dot(grad_output,self.Weights.T)
        return grad_Input

#Activation Functions->seen as layers
class Relu(Layer):
    Input=None
    def forward(self,X):
        self.Input=X
        return np.maximum(0,X)
    def backward(self,grad_output):
        mask=self.Input>0
        return grad_output*mask
class Sigmoid(Layer):
    Input=None
    Output=None
    def forward(self,X):
        self.Input=X
        self.Output= 1/(1+np.exp(-X))
        return self.Output
    def backward(self, grad_output):
        return grad_output * self.Output * (1 - self.Output)
class Softmax(Layer):
    Input=None
    Output=None
    def forward(self,x):
        max=np.max(x)
        self.Input=x
        self.Output=np.exp(x-max)/np.sum(np.exp(x-max),axis=0)
        return self.Output
    def backward(self, grad_output):
        S = self.Output

        temp = np.sum(
            grad_output * S,
            axis=1,
            keepdims=True
        )
        return S * (grad_output - temp)

#Loss Functions
class MSE(Loss):
    def forward(self,x,t):
        self.Input=x
        self.Target=t
        return np.mean(np.square(x-t))
    def backward(self):
        size=self.Input.size
        return 2/size*(self.Input-self.Target)
class CEE(Loss):
    delta=1e-7


#Optimizer
class SGD(Optimizer):
    def optimize(self,args,grad,learning_rate):
        return args-learning_rate*grad
class Momentum(Optimizer):
    pass
class AdaGrad(Optimizer):
    pass
class Adam(Optimizer):
    pass





